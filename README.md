# MedChat: A Hybrid Drug-Interaction Chatbot

MedChat is a final-project conversational assistant that checks drug–drug interactions and explains them in plain language. A user types medications in natural language ("I take warfarin and ibuprofen", "Can I take Advil with Coumadin?"); MedChat extracts the drug names, looks up the pair in the DDInter interaction database, and produces a grounded, safe explanation. On request it also reports a drug's common **side effects and key warnings**, retrieved live from the U.S. FDA's openFDA drug-label API.

MedChat is **not medical advice.** It is an educational demonstration of a hybrid data-science pipeline: information extraction, clustering, multi-model classification, retrieval, an openFDA side-effect/warning layer, and a grounded LLM response layer, all built on a structured interaction database.

*DTI 5125 — Group 2 final project.*

---

## Repository Structure

```
medchat-drug-interaction-chatbot/
├── notebooks/
│   └── medchat_chatbot.ipynb              # full pipeline + Gradio chatbot (Colab-ready)
├── data/
│   └── ddinter/                           # 8 DDInter CSV files (ddinter_downloads_code_*.csv)
├── scripts/
│   ├── run_medchat_50_round_eval.py       # scripted multi-turn chatbot evaluation
│   ├── export_medchat_eval_to_excel.py    # builds the Excel evaluation report
│   └── create_medchat_structure_docx.py   # generates the structure-explanation Word doc
├── medchat_llm_outputs/
│   └── medchat_50_round_eval_report.xlsx  # evaluation results
├── docs/
│   ├── MedChat_Capabilities_Guide.docx
│   └── Live_Demo_Script.docx
├── .env.example                           # copy to .env and add your own key
└── README.md
```

---

## Quick start on Google Colab

1. Open `notebooks/medchat_chatbot.ipynb` in Google Colab (**File → Upload notebook**).
2. **Add your OpenAI key as a Colab Secret** (recommended, so the key never appears in a cell):
   - Click the **key icon** in the left sidebar.
   - Add a secret named `OPENAI_API_KEY`, paste your key as the value.
   - Toggle **Notebook access** on.
3. Run **Cell 0 (Colab Setup)** to install the libraries.
4. Run the **Imports and Configuration** cell — it should print `OpenAI key source: Colab Secrets` and `OpenAI enabled: True`.
5. Run the **Load and Prepare the DDInter Dataset** cell. When prompted, upload the 8 `ddinter_downloads_code_*.csv` files from `data/ddinter/` (or switch to the Google Drive option in that cell).
6. Run the remaining cells top to bottom. The final Gradio cell launches the live chatbot with a public `share=True` link.

If you don't set a Colab Secret, the notebook falls back to an environment variable, then a local `.env` file, then a hidden prompt — so it still runs, it just asks you to paste the key once.

> **Security note:** never commit a real API key to a notebook, a `.env` file, or a shared repo. If a key has ever been shared or uploaded, revoke it at platform.openai.com and generate a new one.

## Running locally (instead of Colab)

1. Copy the 8 CSV files from `data/ddinter/` into the same folder as the notebook.
2. Copy `.env.example` to `.env` next to the notebook and fill in your own `OPENAI_API_KEY`.
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn rapidfuzz openai python-dotenv gradio joblib requests
   ```
4. Open and run `medchat_chatbot.ipynb` (it auto-detects that it is not on Colab and reads CSVs from the current folder).

---

## Data description and source

MedChat uses the **DDInter** drug–drug interaction database, distributed as 8 CSV files split by drug-class code:

`ddinter_downloads_code_A.csv`, `_B`, `_D`, `_H`, `_L`, `_P`, `_R`, `_V`.

The files are combined, cleaned, deduplicated, and written to a SQLite database at `medchat_llm_outputs/ddinter_llm_demo.db`. After deduplication the dataset holds **160,235 unique interaction records** across **1,939 unique drug names**. The columns used are:

- `Drug_A`, `Drug_B` — the two interacting drugs.
- `Level` — DDInter severity: `Major`, `Moderate`, `Minor`, or `Unknown`.

In addition to this static database, MedChat queries the **openFDA drug-label API** (`api.fda.gov/drug/label.json`) live at runtime to fetch a single drug's common side effects and most important warnings. Responses are cached per drug, and a small built-in fallback dictionary keeps the feature working offline.

---

## Methodology

The chatbot uses a hybrid architecture rather than a single LLM:

1. **Drug-name extraction** — exact matching against the DDInter vocabulary, synonym matching for brand names (Advil → Ibuprofen, Coumadin → Warfarin), and RapidFuzz typo matching (metforim → Metformin). Common words (take, daily, blood, yes, no…) are filtered so they aren't mistaken for drugs.
2. **Guided multi-turn intake** — an 8-stage finite-state flow: medications → dosing → allergies → conditions → age → concern → confirm → ready. Session state is remembered across turns.
3. **Direct DDInter lookup** — a SQLite query that checks the pair in **both** drug orderings.
4. **Clustering** — TF-IDF features → TruncatedSVD → KMeans, with silhouette score used to choose *k*, as exploratory validation of structure in the data.
5. **Classification** — four supervised models compared, with a champion selected by weighted F1 (used as a fallback when no exact DDInter pair exists).
6. **Retrieval (RAG-style)** — TF-IDF retrieval over the DDInter interaction text to surface the most relevant records as grounding context.
7. **openFDA side-effect/warning lookup** — when a user asks about a single drug's side effects or warnings ("what are the side effects of metformin?"), or after an interaction result, MedChat pulls the relevant fields from the openFDA drug label, cleans them into short plain-language summaries, and passes them to the response layer as evidence. It never invents side effects.
8. **OpenAI response layer** — the LLM only rewords the retrieved evidence and session state into natural language. It is never the source of medical facts.
9. **Safety rules** — emergency-keyword detection (chest pain, severe bleeding, overdose, seizure…) short-circuits the flow and advises urgent care; every answer carries an educational disclaimer.

---

## Models and algorithms

**Clustering:** KMeans on TF-IDF features of `Drug_A + Drug_B + Level`, reduced with TruncatedSVD; silhouette score across k = 3…7 (best silhouette 0.284 at k = 7).

**Classification (4 algorithms compared):**

- Linear SVM ← **champion**
- Logistic Regression
- Multinomial Naive Bayes
- SGD Classifier

The champion Linear SVM reaches **weighted F1 ≈ 0.819** on a held-out 12,000-row test set.

**Retrieval:** TF-IDF retrieval over the full DDInter interaction table (160,235 rows indexed in the final evaluation).

**LLM:** OpenAI model configured from `OPENAI_MODEL` (default `gpt-4.1-mini`), used only as a response layer after evidence has been retrieved.

**Side effects & warnings (openFDA):** a cached lookup against the openFDA drug-label API scores multiple label matches per drug, extracts the most-common-adverse-reactions sentence and any boxed/important warning, and rewrites them into short readable summaries. A single-drug side-effect question is routed here; the interaction path is also enriched with this context. The layer degrades gracefully to an offline fallback dictionary when the API is unreachable.

---

## Evaluation

Two evaluation tracks are reported:

**Offline ML metrics** (held-out test set, n = 12,000): accuracy, weighted F1, macro F1, and a full per-class precision/recall/F1 report. Per-class F1: Major 0.73, Moderate 0.86, Unknown 0.84, Minor 0.56 (Minor is the smallest and hardest class).

**Online system evaluation** — a scripted chatbot harness. The final run executed **200 turns** (50 unique scripted turns replayed across 4 batches = 20 sessions) and checked four independent dimensions per turn:

- Chatbot flow: 200/200 passed
- Drug extraction: 40/40 passed
- Severity checks: 40/40 passed
- RAG pair retrieval: 20/20 passed
- OpenAI response path: exercised successfully with live calls, 0 API errors
- Champion local ML model: Linear SVM

Results are exported to `medchat_50_round_eval.json` and a 5-sheet Excel report (Methodology / Models Used / Summary / Turn Details / Failed Checks).

> A smaller in-notebook "Simple Evaluation" cell checks 4 hand-picked cases for a fast sanity check; the rigorous evaluation is the separate scripted harness described above.

---

## Error analysis

- 2,218 of 12,000 test examples misclassified (~18.5% error rate) by the champion Linear SVM.
- Dominant confusions are **adjacent** severities: Moderate→Major (638) and Moderate→Unknown (507); opposite-extreme errors (Major↔Minor) are rare.
- Minor is the weakest class (precision ≈ 0.49) due to smallest support even after class weighting.
- **Clustering caveat:** cluster purity is partly inflated because the `Level` word is embedded in the TF-IDF cluster text — a known methodological limitation, not a blind discovery.

**Limitations:**

- Exact DDInter lookup is the most reliable signal; ML predictions are fallback estimates, not confirmed records.
- Rare brand names or unusual misspellings may not always resolve.
- RAG can return similar-but-not-exact context, so direct database evidence is always prioritized.
- LLM responses are grounded in evidence but do not replace a clinician.

---

## How to reproduce the evaluation

Run the notebook first: it builds `medchat_llm_outputs/ddinter_llm_demo.db` and the champion model, which the scripts read. Then run the scripts from the repository root:

```bash
# Full 200-turn evaluation
python3 scripts/run_medchat_50_round_eval.py

# A shorter run
MEDCHAT_EVAL_TURNS=50 python3 scripts/run_medchat_50_round_eval.py

# Regenerate the Excel report
python3 scripts/export_medchat_eval_to_excel.py
```

The SQLite database and trained model are not committed because they are regenerated by the notebook.

---

## Safety note

MedChat is for educational demonstration only. It does not provide medical advice and is not a medical device. Always confirm any medication decision with a doctor or pharmacist.
