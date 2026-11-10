from itertools import combinations
from pathlib import Path
import json
import os
import re
import sqlite3
import time

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


ROOT = Path.cwd()
OUTPUT_PATH = ROOT / "medchat_llm_outputs"
DB_PATH = OUTPUT_PATH / "ddinter_llm_demo.db"
MODEL_PATH = OUTPUT_PATH / "champion_severity_model.joblib"
RESULT_PATH = OUTPUT_PATH / "medchat_50_round_eval.json"
EVAL_TURNS = int(os.getenv("MEDCHAT_EVAL_TURNS", "200"))

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = os.getenv("MEDCHAT_EVAL_USE_OPENAI", "1") == "1" and bool(OPENAI_API_KEY and OpenAI)
client = OpenAI(api_key=OPENAI_API_KEY) if USE_OPENAI else None


SYNONYMS = {
    "advil": "Ibuprofen",
    "motrin": "Ibuprofen",
    "tylenol": "Acetaminophen",
    "paracetamol": "Acetaminophen",
    "aspirin": "Acetylsalicylic acid",
    "bayer": "Acetylsalicylic acid",
    "coumadin": "Warfarin",
    "glucophage": "Metformin",
    "prilosec": "Omeprazole",
    "zoloft": "Sertraline",
    "prozac": "Fluoxetine",
    "xanax": "Alprazolam",
    "lipitor": "Atorvastatin",
    "norvasc": "Amlodipine",
    "lasix": "Furosemide",
    "plavix": "Clopidogrel",
    "crestor": "Rosuvastatin",
    "toprol": "Metoprolol",
    "warafin": "Warfarin",
    "warferin": "Warfarin",
    "ibuprofin": "Ibuprofen",
    "metforim": "Metformin",
    "metformine": "Metformin",
    "amoxicilin": "Amoxicillin",
    "potassium": "Potassium chloride",
    "potassium supplement": "Potassium chloride",
}

FALSE_POSITIVES = {
    "take", "taking", "drug", "drugs", "medicine", "medication", "medications",
    "pill", "pills", "tablet", "tablets", "capsule", "dose", "daily", "day",
    "week", "month", "condition", "conditions", "kidney", "liver", "heart",
    "disease", "pressure", "blood", "pregnant", "pregnancy", "asthma", "allergy",
    "allergies", "allergic", "none", "yes", "no", "what", "about", "with",
    "and", "or", "also", "remove", "stopped", "started", "skip", "safe",
    "safety", "watch", "happen", "happens", "risk", "risks", "major",
    "moderate", "minor", "adult", "child", "teen", "senior", "older",
}

CONDITION_KEYWORDS = [
    "kidney disease", "kidney problems", "liver disease", "liver problems",
    "pregnant", "pregnancy", "asthma", "diabetes", "hypertension",
    "high blood pressure", "heart disease", "ulcer", "bleeding disorder",
    "stroke", "blood clot", "depression", "anxiety",
]

EMERGENCY_PHRASES = [
    "chest pain", "can't breathe", "cannot breathe", "shortness of breath",
    "severe bleeding", "black stool", "blood in vomit", "fainting",
    "seizure", "overdose", "suicidal", "throat closing",
]

AGE_GROUPS = ["child", "teen", "adult", "older adult", "senior"]

SYSTEM_INSTRUCTIONS = """
You are MedChat, a final-project drug interaction assistant.
Stay strictly grounded in the provided dataset evidence.
If no exact DDInter record was found, say that this does not prove safety.
For Major interactions, recommend contacting a doctor or pharmacist before combining.
Keep answers concise enough for a live demo.
"""


def load_interactions():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT Drug_A, Drug_B, Level FROM interactions",
            conn,
        )


interactions = load_interactions()
all_drugs = pd.concat([interactions["Drug_A"], interactions["Drug_B"]]).dropna().astype(str).drop_duplicates().tolist()
exact_drugs = {drug.lower(): drug for drug in all_drugs}
champion_bundle = joblib.load(MODEL_PATH)
champion_model = champion_bundle["model"]
champion_model_name = champion_bundle["model_name"]


def build_rag_index(max_rows=None):
    if max_rows is None:
        sample = interactions.reset_index(drop=True)
    else:
        sample = interactions.sample(n=min(max_rows, len(interactions)), random_state=42).reset_index(drop=True)
    documents = (
        sample["Drug_A"].astype(str) + " and " +
        sample["Drug_B"].astype(str) + " have a " +
        sample["Level"].astype(str) + " severity drug interaction."
    ).tolist()
    vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), lowercase=True)
    matrix = vectorizer.fit_transform(documents)
    return sample, documents, vectorizer, matrix


rag_sample, rag_documents, rag_vectorizer, rag_matrix = build_rag_index()


def fuzzy_match_drug(name, threshold=86):
    cleaned = str(name).lower().strip()
    if len(cleaned) <= 3 or cleaned in FALSE_POSITIVES or cleaned.isdigit():
        return None
    if cleaned in SYNONYMS:
        return SYNONYMS[cleaned]
    if cleaned in exact_drugs:
        return exact_drugs[cleaned]
    result = process.extractOne(name, all_drugs, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
    return result[0] if result else None


def extract_drugs(text):
    text_lower = text.lower()
    candidates = []
    for key, value in exact_drugs.items():
        if len(key) > 4 and re.search(rf"\b{re.escape(key)}\b", text_lower):
            candidates.append(value)
    for token in re.findall(r"\b[a-zA-Z][a-zA-Z-]+\b", text_lower):
        if token in SYNONYMS:
            candidates.append(SYNONYMS[token])
    for token in re.findall(r"\b[a-zA-Z][a-zA-Z-]+\b", text_lower):
        if token in FALSE_POSITIVES or token in SYNONYMS:
            continue
        if len(token) >= 6:
            match = fuzzy_match_drug(token)
            if match:
                candidates.append(match)
    resolved = []
    seen = set()
    for drug in candidates:
        if drug.lower() not in seen:
            resolved.append(drug)
            seen.add(drug.lower())
    return resolved


def find_interaction(drug1, drug2):
    with sqlite3.connect(DB_PATH) as conn:
        result = pd.read_sql_query(
            """
            SELECT Drug_A, Drug_B, Level
            FROM interactions
            WHERE (LOWER(Drug_A) = LOWER(?) AND LOWER(Drug_B) = LOWER(?))
               OR (LOWER(Drug_A) = LOWER(?) AND LOWER(Drug_B) = LOWER(?))
            LIMIT 1
            """,
            conn,
            params=[drug1, drug2, drug2, drug1],
        )
    if result.empty:
        return None
    return result.iloc[0].to_dict()


def severity_recommendation(level):
    level = str(level).lower()
    if level == "major":
        return "High concern: do not combine unless a clinician specifically instructed it. Contact a doctor or pharmacist."
    if level == "moderate":
        return "Use caution: ask a clinician whether monitoring, timing, or dose changes are needed."
    if level == "minor":
        return "Lower concern: monitor symptoms and confirm with a pharmacist if unsure."
    return "Unclear severity: do not assume it is safe; ask a pharmacist."


def predict_interaction_severity(drug1, drug2):
    pair_text = f"{drug1} {drug2} {drug2} {drug1}"
    prediction = champion_model.predict([pair_text])[0]
    return {
        "level": prediction,
        "recommendation": severity_recommendation(prediction),
        "source": f"{champion_model_name} estimate",
        "model_name": champion_model_name,
    }


def retrieve_rag_context(query, n_results=5):
    query_vec = rag_vectorizer.transform([query])
    scores = cosine_similarity(query_vec, rag_matrix).ravel()
    top_idx = np.argsort(scores)[::-1][:n_results]
    rows = []
    for idx in top_idx:
        row = rag_sample.iloc[int(idx)]
        rows.append({
            "text": rag_documents[int(idx)],
            "drug_a": row["Drug_A"],
            "drug_b": row["Drug_B"],
            "level": row["Level"],
            "score": float(scores[int(idx)]),
        })
    return rows


def check_interaction_pairs(drugs, include_rag=False):
    evidence = []
    for drug1, drug2 in combinations(drugs, 2):
        row = find_interaction(drug1, drug2)
        if row:
            item = {
                "drug1": drug1,
                "drug2": drug2,
                "found": True,
                "level": row["Level"],
                "recommendation": severity_recommendation(row["Level"]),
                "source": "direct_ddinter_lookup",
            }
        else:
            ml_prediction = predict_interaction_severity(drug1, drug2)
            item = {
                "drug1": drug1,
                "drug2": drug2,
                "found": False,
                "level": "Not found",
                "recommendation": (
                    "No exact DDInter record found. The champion ML model predicts "
                    f"{ml_prediction['level']} severity ({ml_prediction['source']}). "
                    f"{ml_prediction['recommendation']}"
                ),
                "ml_prediction": ml_prediction["level"],
                "ml_source": ml_prediction["source"],
                "source": "champion_ml_fallback",
            }
        if include_rag:
            item["rag_context"] = retrieve_rag_context(f"{drug1} {drug2}", n_results=3)
        evidence.append(item)
    return evidence


def new_session():
    return {
        "stage": "medications",
        "drugs": [],
        "dosing_notes": [],
        "allergies": [],
        "conditions": [],
        "age_group": None,
        "main_concern": None,
        "confirmed": False,
        "last_evidence": [],
        "last_pair": None,
        "turns": 0,
    }


def normalize_session(state):
    base = new_session()
    if isinstance(state, dict):
        base.update(state)
        for key in ["drugs", "dosing_notes", "allergies", "conditions", "last_evidence"]:
            if base.get(key) is None:
                base[key] = []
    return base


def add_unique(target, values):
    added = []
    existing = {str(x).lower() for x in target}
    for value in values:
        if value and str(value).lower() not in existing:
            target.append(value)
            existing.add(str(value).lower())
            added.append(value)
    return added


def detect_conditions(text):
    lower = text.lower()
    return [c for c in CONDITION_KEYWORDS if c in lower]


def detect_age_group(text):
    lower = text.lower()
    for group in AGE_GROUPS:
        if group in lower:
            return "older adult" if group == "senior" else group
    match = re.search(r"\b(\d{1,3})\s*(years old|year old|yo)?\b", lower)
    if match:
        age = int(match.group(1))
        if age < 13:
            return "child"
        if age < 18:
            return "teen"
        if age >= 65:
            return "older adult"
        return "adult"
    return None


def extract_dosing_notes(text):
    notes = []
    patterns = [
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b",
        r"\b(?:once|twice|three times|\d+ times)\s+(?:a|per)\s+day\b",
        r"\b(?:morning|night|evening|daily|weekly|as needed|prn)\b",
    ]
    for pattern in patterns:
        notes.extend(re.findall(pattern, text, flags=re.I))
    return [str(note).strip() for note in notes]


def extract_allergies(text):
    lower = text.lower()
    if not any(word in lower for word in ["allergic", "allergy", "allergies"]):
        return []
    drugs = extract_drugs(text)
    if drugs:
        return drugs
    if "none" in lower or "no" == lower.strip():
        return []
    cleaned = re.sub(r".*allerg(?:ic|y|ies)\s*(to)?", "", text, flags=re.I).strip()
    return [p.strip().title() for p in re.split(r",| and ", cleaned) if p.strip()][:4]


def is_negative_answer(text):
    return text.lower().strip() in {"no", "none", "nope", "not sure", "unknown", "i don't know", "dont know", "skip"}


def is_affirmative_answer(text):
    return text.lower().strip() in {"yes", "yeah", "yep", "sure", "ok", "okay", "correct", "looks good", "confirm", "confirmed"}


def detect_emergency(text):
    return any(phrase in text.lower() for phrase in EMERGENCY_PHRASES)


def next_question(state):
    if state["stage"] == "medications":
        return "Please list the medications, supplements, or over-the-counter drugs you take. Brand names are okay."
    if state["stage"] == "dosing":
        return "Do you know the dose or timing for any of them? For example, `warfarin 5 mg at night`. You can also say `skip`."
    if state["stage"] == "allergies":
        return "Do you have any medication allergies? You can say `none`, or list them."
    if state["stage"] == "conditions":
        return "Do you have important health factors like kidney disease, liver disease, pregnancy, asthma, ulcers, or bleeding problems?"
    if state["stage"] == "age":
        return "What age group should I consider: child, teen, adult, or older adult? You can also type your age."
    if state["stage"] == "concern":
        return "What is your main concern today: safety, side effects, what to watch for, or whether to ask a pharmacist?"
    if state["stage"] == "confirm":
        return "Before I check the dataset, please confirm: is this profile correct? Reply `yes` to continue or add corrections."
    return "You can ask a follow-up like `what could happen?`, `what about aspirin?`, or `remove ibuprofen`."


def profile_text(state):
    last_pair = state.get("last_pair")
    last_pair_text = " + ".join(last_pair) if last_pair else "None yet"
    return (
        f"Medications: {', '.join(state['drugs']) if state['drugs'] else 'None yet'}\n"
        f"Dose/timing notes: {', '.join(state['dosing_notes']) if state['dosing_notes'] else 'None recorded'}\n"
        f"Allergies: {', '.join(state['allergies']) if state['allergies'] else 'None recorded'}\n"
        f"Conditions: {', '.join(state['conditions']) if state['conditions'] else 'None recorded'}\n"
        f"Age group: {state['age_group'] or 'Not recorded'}\n"
        f"Main concern: {state['main_concern'] or 'Not recorded'}\n"
        f"Last checked pair: {last_pair_text}\n"
        f"Conversation stage: {state['stage']}"
    )


def call_openai(task, state, evidence=None, fallback=None):
    if not USE_OPENAI:
        return fallback or task
    payload = {
        "task": task,
        "session_state": state,
        "dataset_evidence": evidence or [],
    }
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=SYSTEM_INSTRUCTIONS,
        input=[{"role": "user", "content": json.dumps(payload)}],
        timeout=20,
    )
    return response.output_text.strip()


def fallback_recommendation(state, evidence):
    if not evidence:
        return next_question(state)
    lines = ["Here is what I found in the DDInter dataset:"]
    for item in evidence:
        lines.append(f"- {item['drug1']} + {item['drug2']}: {item['level']}")
        lines.append(f"  Recommendation: {item['recommendation']}")
        if item.get("rag_context"):
            top = item["rag_context"][0]
            lines.append(f"  RAG top context: {top['drug_a']} + {top['drug_b']} = {top['level']} (score {top['score']:.3f})")
    lines.append("Educational use only; confirm decisions with a healthcare professional.")
    return "\n".join(lines)


def intro_message():
    return "Hi, I am MedChat. What medications are you currently taking?"


def run_recommendation(state):
    state["stage"] = "ready"
    state["confirmed"] = True
    evidence = check_interaction_pairs(state["drugs"], include_rag=True)
    state["last_evidence"] = evidence
    if evidence:
        state["last_pair"] = [evidence[-1]["drug1"], evidence[-1]["drug2"]]
    fallback = fallback_recommendation(state, evidence)
    return call_openai("Explain DDInter interaction results and recommendations naturally.", state, evidence, fallback), state


def handle_user_message(user_text, state):
    state = normalize_session(state)
    state["turns"] += 1
    text = user_text.strip()
    lower = text.lower()

    if not text:
        return next_question(state), state
    if detect_emergency(text):
        return "This may be an emergency symptom. Please seek urgent medical care now or call emergency services.", state
    if lower in {"reset", "start over", "clear", "new session"}:
        return intro_message(), new_session()
    if lower in {"profile", "summary", "what do you know"}:
        return profile_text(state) + "\n\n" + next_question(state), state
    if lower.startswith("remove") or "stop taking" in lower or "no longer take" in lower:
        removed = []
        for drug in extract_drugs(text):
            if drug in state["drugs"]:
                state["drugs"].remove(drug)
                removed.append(drug)
        if len(state["drugs"]) < 2:
            state["stage"] = "medications"
        response = f"Removed {', '.join(removed)}." if removed else "I could not identify which medication to remove."
        return response + "\n\n" + next_question(state), state

    mentioned_drugs = extract_drugs(text)
    mentioned_allergies = extract_allergies(text)
    mentioned_conditions = detect_conditions(text)
    dosing_notes = extract_dosing_notes(text)
    age_group = detect_age_group(text)

    if state["stage"] == "medications":
        add_unique(state["drugs"], mentioned_drugs)
        add_unique(state["dosing_notes"], dosing_notes)
        if len(state["drugs"]) >= 2:
            state["stage"] = "dosing"
            return f"I recorded: {', '.join(state['drugs'])}.\n\n{next_question(state)}", state
        if len(state["drugs"]) == 1:
            return f"I recorded {state['drugs'][0]}. Please tell me at least one more medication so I can check interactions.", state
        return "I could not identify a medication yet. Please type medicine names, for example: `warfarin, ibuprofen`.", state
    if state["stage"] == "dosing":
        add_unique(state["dosing_notes"], dosing_notes)
        state["stage"] = "allergies"
        return "Thanks. " + next_question(state), state
    if state["stage"] == "allergies":
        if not is_negative_answer(text):
            add_unique(state["allergies"], mentioned_allergies)
        state["stage"] = "conditions"
        return "Thanks. " + next_question(state), state
    if state["stage"] == "conditions":
        if not is_negative_answer(text):
            add_unique(state["conditions"], mentioned_conditions)
        state["stage"] = "age"
        return "Got it. " + next_question(state), state
    if state["stage"] == "age":
        if age_group and not is_negative_answer(text):
            state["age_group"] = age_group
        state["stage"] = "concern"
        return "Thanks. " + next_question(state), state
    if state["stage"] == "concern":
        if not is_negative_answer(text):
            state["main_concern"] = text
        state["stage"] = "confirm"
        return profile_text(state) + "\n\n" + next_question(state), state
    if state["stage"] == "confirm":
        if is_affirmative_answer(text):
            return run_recommendation(state)
        add_unique(state["drugs"], mentioned_drugs)
        add_unique(state["dosing_notes"], dosing_notes)
        add_unique(state["allergies"], mentioned_allergies)
        add_unique(state["conditions"], mentioned_conditions)
        if age_group:
            state["age_group"] = age_group
        return profile_text(state) + "\n\nReply `yes` when this is correct, or add another correction.", state

    new_drugs = [drug for drug in mentioned_drugs if drug not in state["drugs"]]
    if new_drugs:
        add_unique(state["drugs"], new_drugs)
        return run_recommendation(state)
    if any(phrase in lower for phrase in ["what could happen", "what happens", "tell me more", "why", "danger", "risk"]):
        evidence = state.get("last_evidence", [])
        return fallback_recommendation(state, evidence), state
    return "I can keep helping. You can ask `what could happen?`, add another medication, remove a medication, or type `profile`.", state


def base_scripted_rounds():
    return [
        ("warfarin_flow", "I take warfarin and ibuprofen", ["Warfarin", "Ibuprofen"], "Major"),
        ("warfarin_flow", "warfarin 5 mg at night and ibuprofen as needed", [], None),
        ("warfarin_flow", "none", [], None),
        ("warfarin_flow", "I have kidney disease", [], None),
        ("warfarin_flow", "adult", [], None),
        ("warfarin_flow", "I want to know if it is safe and what to watch for", [], None),
        ("warfarin_flow", "yes", [], "Major"),
        ("warfarin_flow", "what could happen?", [], "Major"),
        ("warfarin_flow", "what about aspirin?", ["Acetylsalicylic acid"], "Major"),
        ("warfarin_flow", "profile", [], None),
        ("brand_flow", "Can I take Advil with Coumadin?", ["Ibuprofen", "Warfarin"], "Major"),
        ("brand_flow", "skip", [], None),
        ("brand_flow", "no allergies", [], None),
        ("brand_flow", "I have a bleeding disorder", [], None),
        ("brand_flow", "52 years old", [], None),
        ("brand_flow", "safety", [], None),
        ("brand_flow", "yes", [], "Major"),
        ("brand_flow", "remove Advil", ["Ibuprofen"], None),
        ("brand_flow", "I also take Tylenol", ["Acetaminophen"], None),
        ("brand_flow", "profile", [], None),
        ("metformin_flow", "metforim and prilosec", ["Metformin", "Omeprazole"], "Unknown"),
        ("metformin_flow", "metformin 500 mg twice a day", [], None),
        ("metformin_flow", "none", [], None),
        ("metformin_flow", "diabetes", [], None),
        ("metformin_flow", "adult", [], None),
        ("metformin_flow", "side effects and safety", [], None),
        ("metformin_flow", "yes", [], "Unknown"),
        ("metformin_flow", "what could happen?", [], "Unknown"),
        ("metformin_flow", "what about aspirin", ["Acetylsalicylic acid"], None),
        ("metformin_flow", "summary", [], None),
        ("statin_flow", "I use Lipitor and clarithromycin", ["Atorvastatin", "Clarithromycin"], None),
        ("statin_flow", "20 mg daily", [], None),
        ("statin_flow", "allergic to penicillin", [], None),
        ("statin_flow", "liver disease", [], None),
        ("statin_flow", "older adult", [], None),
        ("statin_flow", "interaction risk", [], None),
        ("statin_flow", "yes", [], None),
        ("statin_flow", "tell me more", [], None),
        ("statin_flow", "what about amoxicilin", ["Amoxicillin"], None),
        ("statin_flow", "reset", [], None),
        ("emergency_flow", "I take sertraline and tramadol", ["Sertraline", "Tramadol"], None),
        ("emergency_flow", "skip", [], None),
        ("emergency_flow", "none", [], None),
        ("emergency_flow", "depression", [], None),
        ("emergency_flow", "teen", [], None),
        ("emergency_flow", "danger", [], None),
        ("emergency_flow", "yes", [], None),
        ("emergency_flow", "I have chest pain and cannot breathe", [], "emergency"),
        ("emergency_flow", "what drugs did I tell you", [], None),
        ("emergency_flow", "new session", [], None),
    ]


def scripted_rounds(target_turns=EVAL_TURNS):
    base_rounds = base_scripted_rounds()
    rounds = []
    batch = 1
    while len(rounds) < target_turns:
        for session_id, message, expected_drugs, expected_level in base_rounds:
            rounds.append((f"{session_id}_batch_{batch}", message, expected_drugs, expected_level))
            if len(rounds) >= target_turns:
                break
        batch += 1
    return rounds


def exact_or_reverse_match(context, expected_drugs):
    if len(expected_drugs) < 2:
        return True
    pool = []
    for item in context:
        pool.extend([item["drug_a"], item["drug_b"]])
    pool_lower = {str(x).lower() for x in pool}
    return all(drug.lower() in pool_lower for drug in expected_drugs)


def evaluate():
    states = {}
    turn_results = []
    openai_errors = []
    start = time.time()
    planned_rounds = scripted_rounds()
    for turn_number, (session_id, message, expected_drugs, expected_level) in enumerate(planned_rounds, start=1):
        state = states.setdefault(session_id, new_session())
        error = None
        try:
            reply, state = handle_user_message(message, state)
        except Exception as exc:
            if USE_OPENAI:
                openai_errors.append(str(exc))
                globals()["USE_OPENAI"] = False
                reply, state = handle_user_message(message, state)
            else:
                error = f"{type(exc).__name__}: {exc}"
                reply = ""
        states[session_id] = state

        extracted = extract_drugs(message)
        entity_ok = all(drug in extracted for drug in expected_drugs)

        direct = None
        ml = None
        rag_query = " ".join([message] + extracted + state.get("drugs", []))
        rag = retrieve_rag_context(rag_query, n_results=5)
        if len(expected_drugs) >= 2:
            direct = find_interaction(expected_drugs[0], expected_drugs[1])
            ml = predict_interaction_severity(expected_drugs[0], expected_drugs[1])
        elif len(extracted) >= 2:
            direct = find_interaction(extracted[0], extracted[1])
            ml = predict_interaction_severity(extracted[0], extracted[1])

        expected_level_ok = True
        if expected_level == "emergency":
            expected_level_ok = "urgent" in reply.lower() or "emergency" in reply.lower()
        elif expected_level:
            direct_level = direct["Level"] if direct else None
            ml_level = ml["level"] if ml else None
            evidence_levels = {item.get("level") for item in state.get("last_evidence", [])}
            evidence_levels.update({item.get("ml_prediction") for item in state.get("last_evidence", []) if item.get("ml_prediction")})
            reply_has_level = expected_level.lower() in reply.lower()
            expected_level_ok = expected_level in {direct_level, ml_level, *evidence_levels} or reply_has_level

        rag_ok = exact_or_reverse_match(rag, expected_drugs)
        flow_ok = not error and bool(reply.strip()) and state["stage"] in {
            "medications", "dosing", "allergies", "conditions", "age", "concern", "confirm", "ready"
        }

        turn_results.append({
            "turn": turn_number,
            "session": session_id,
            "user": message,
            "reply_preview": reply[:400],
            "stage": state["stage"],
            "extracted_drugs": extracted,
            "expected_drugs": expected_drugs,
            "entity_ok": entity_ok,
            "expected_level": expected_level,
            "direct_level": direct["Level"] if direct else None,
            "ml_level": ml["level"] if ml else None,
            "rag_top": rag[:3],
            "rag_ok_for_expected_drugs": rag_ok,
            "level_ok": expected_level_ok,
            "flow_ok": flow_ok,
            "error": error,
        })

    total = len(turn_results)
    summary = {
        "total_turns": total,
        "requested_turns": EVAL_TURNS,
        "chatbot_flow_ok": sum(r["flow_ok"] for r in turn_results),
        "entity_turns": sum(bool(r["expected_drugs"]) for r in turn_results),
        "entity_ok": sum(r["entity_ok"] for r in turn_results if r["expected_drugs"]),
        "level_turns": sum(bool(r["expected_level"]) for r in turn_results),
        "level_ok": sum(r["level_ok"] for r in turn_results if r["expected_level"]),
        "rag_pair_turns": sum(len(r["expected_drugs"]) >= 2 for r in turn_results),
        "rag_ok": sum(r["rag_ok_for_expected_drugs"] for r in turn_results if len(r["expected_drugs"]) >= 2),
        "model_name": champion_model_name,
        "openai_requested": bool(OPENAI_API_KEY and OpenAI),
        "openai_used": USE_OPENAI and not openai_errors,
        "openai_errors": openai_errors[:3],
        "rag_index_rows": len(rag_sample),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    output = {"summary": summary, "turns": turn_results}
    RESULT_PATH.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved detailed results to {RESULT_PATH}")


if __name__ == "__main__":
    evaluate()
