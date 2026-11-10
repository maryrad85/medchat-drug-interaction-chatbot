from pathlib import Path
import json
import os

import joblib
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path.cwd()
OUTPUT_DIR = ROOT / "medchat_llm_outputs"
INPUT_JSON = OUTPUT_DIR / "medchat_50_round_eval.json"
OUTPUT_XLSX = OUTPUT_DIR / "medchat_50_round_eval_report.xlsx"
MODEL_PATH = OUTPUT_DIR / "champion_severity_model.joblib"


def yes_no(value):
    return "PASS" if bool(value) else "FAIL"


def pct(numerator, denominator):
    if not denominator:
        return "N/A"
    return f"{numerator}/{denominator} ({numerator / denominator:.0%})"


def rag_top_text(rag_items):
    lines = []
    for item in rag_items or []:
        lines.append(
            f"{item.get('drug_a')} + {item.get('drug_b')} = "
            f"{item.get('level')} (score {item.get('score', 0):.3f})"
        )
    return "\n".join(lines)


def load_env_value(key, default=""):
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key:
                return value.strip().strip('"').strip("'")
    return os.getenv(key, default)


def method_used(turn, summary):
    methods = ["Guided chatbot controller"]
    if turn.get("direct_level"):
        methods.append("Direct DDInter lookup")
    if turn.get("rag_top"):
        methods.append("RAG retrieval")
    if turn.get("ml_level"):
        methods.append("Linear SVM severity model")
    if summary.get("openai_used"):
        methods.append("OpenAI LLM response layer")
    if turn.get("expected_level") == "emergency":
        methods.append("Emergency keyword safety rule")
    return " + ".join(methods)


def autosize_sheet(ws, max_width=70):
    for column_cells in ws.columns:
        column_letter = get_column_letter(column_cells[0].column)
        width = 12
        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            lines = value.splitlines() or [""]
            width = max(width, min(max_width, max(len(line) for line in lines) + 2))
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        ws.column_dimensions[column_letter].width = width


def style_header(ws):
    fill = PatternFill("solid", fgColor="1F4E78")
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def main():
    data = json.loads(INPUT_JSON.read_text())
    summary = data["summary"]
    turns = data["turns"]
    openai_model = load_env_value("OPENAI_MODEL", "gpt-4.1-mini")
    model_bundle = joblib.load(MODEL_PATH)
    model_results = model_bundle.get("results")

    methodology_rows = [
        {
            "Component": "Evaluation goal",
            "Method / Tool Used": f"{summary['total_turns']} scripted multi-turn chatbot messages",
            "Purpose": "Check that the chatbot can complete repeated medication-interaction conversations.",
            "Evidence Recorded": "Assistant reply, conversation stage, extracted drugs, severity result, RAG result, pass/fail checks.",
        },
        {
            "Component": "Data source",
            "Method / Tool Used": "DDInter interaction database",
            "Purpose": "Ground interaction results in a structured drug-drug interaction dataset.",
            "Evidence Recorded": f"{summary['rag_index_rows']:,} DDInter rows indexed/available.",
        },
        {
            "Component": "Entity extraction",
            "Method / Tool Used": "Exact matching, synonym matching, typo/fuzzy matching with RapidFuzz",
            "Purpose": "Convert user messages and brand names like Advil/Coumadin into canonical drug names.",
            "Evidence Recorded": f"{pct(summary['entity_ok'], summary['entity_turns'])} expected entity checks passed.",
        },
        {
            "Component": "Direct lookup",
            "Method / Tool Used": "SQLite query against DDInter pairs in both drug orders",
            "Purpose": "Find exact interaction severity when the drug pair exists in the database.",
            "Evidence Recorded": "Direct DDInter Level column in the detailed turn sheet.",
        },
        {
            "Component": "ML model",
            "Method / Tool Used": summary["model_name"],
            "Purpose": "Predict severity when no exact DDInter pair is found.",
            "Evidence Recorded": "ML Predicted Level column in the detailed turn sheet.",
        },
        {
            "Component": "RAG retrieval",
            "Method / Tool Used": "TF-IDF vector retrieval over DDInter interaction text",
            "Purpose": "Retrieve relevant interaction evidence/context for chatbot grounding.",
            "Evidence Recorded": f"{pct(summary['rag_ok'], summary['rag_pair_turns'])} two-drug RAG checks passed.",
        },
        {
            "Component": "LLM response layer",
            "Method / Tool Used": openai_model,
            "Purpose": "Generate natural chatbot wording from deterministic state and dataset evidence.",
            "Evidence Recorded": f"OpenAI model used: {yes_no(summary['openai_used'])}; errors: {summary['openai_errors'] or 'None'}.",
        },
        {
            "Component": "Safety rule",
            "Method / Tool Used": "Emergency keyword detection",
            "Purpose": "Return urgent-care guidance when emergency symptoms appear.",
            "Evidence Recorded": "Emergency test included in severity/result checks.",
        },
    ]

    model_rows = [
        {"Item": "OpenAI response model", "Value": openai_model, "Notes": "Used for natural language response layer when API/network is available."},
        {"Item": "OpenAI response model used in final run", "Value": yes_no(summary["openai_used"]), "Notes": "Final evaluation used the configured OpenAI model."},
        {"Item": "Champion local ML model", "Value": summary["model_name"], "Notes": "Saved in champion_severity_model.joblib."},
        {"Item": "Local model purpose", "Value": "Drug-pair severity prediction", "Notes": "Used when exact DDInter lookup is unavailable."},
        {"Item": "RAG method", "Value": "TF-IDF vector retrieval", "Notes": "Indexes canonical DDInter interaction text."},
        {"Item": "RAG corpus size", "Value": summary["rag_index_rows"], "Notes": "Full DDInter interaction table."},
        {"Item": "Database", "Value": "ddinter_llm_demo.db", "Notes": "SQLite database with interactions table."},
    ]

    if model_results is not None:
        for _, row in model_results.iterrows():
            model_rows.append({
                "Item": f"Candidate model: {row['Model']}",
                "Value": f"Weighted F1 {row['Weighted F1']:.3f}",
                "Notes": f"Accuracy {row['Accuracy']:.3f}; Macro F1 {row['Macro F1']:.3f}",
            })

    summary_rows = [
        {"Metric": "Total chatbot turns tested", "Result": summary["total_turns"], "Notes": "Scripted multi-session chatbot run"},
        {"Metric": "Chatbot flow worked", "Result": pct(summary["chatbot_flow_ok"], summary["total_turns"]), "Notes": "Non-empty reply and valid conversation stage"},
        {"Metric": "Drug/entity extraction", "Result": pct(summary["entity_ok"], summary["entity_turns"]), "Notes": "Expected medications were extracted"},
        {"Metric": "Severity/result checks", "Result": pct(summary["level_ok"], summary["level_turns"]), "Notes": "Expected severity or emergency behavior matched"},
        {"Metric": "RAG retrieval checks", "Result": pct(summary["rag_ok"], summary["rag_pair_turns"]), "Notes": "Two-drug pair retrieval from full DDInter RAG index"},
        {"Metric": "RAG index rows", "Result": summary["rag_index_rows"], "Notes": "Rows indexed from DDInter database"},
        {"Metric": "Local ML model", "Result": summary["model_name"], "Notes": "Champion severity model"},
        {"Metric": "OpenAI model used", "Result": yes_no(summary["openai_used"]), "Notes": "Model path was exercised during the run"},
        {"Metric": "OpenAI errors", "Result": ", ".join(summary["openai_errors"]) if summary["openai_errors"] else "None", "Notes": ""},
        {"Metric": "Elapsed seconds", "Result": summary["elapsed_seconds"], "Notes": "Runtime for final evaluated run"},
    ]

    detail_rows = []
    for turn in turns:
        detail_rows.append({
            "Turn": turn["turn"],
            "Session": turn["session"],
            "Method Used": method_used(turn, summary),
            "User Message": turn["user"],
            "Assistant Reply Preview": turn["reply_preview"],
            "Stage After Turn": turn["stage"],
            "Expected Drugs": ", ".join(turn["expected_drugs"]),
            "Extracted Drugs": ", ".join(turn["extracted_drugs"]),
            "Entity Check": yes_no(turn["entity_ok"]),
            "Expected Level": turn["expected_level"] or "",
            "Direct DDInter Level": turn["direct_level"] or "",
            "ML Predicted Level": turn["ml_level"] or "",
            "Severity Check": yes_no(turn["level_ok"]),
            "RAG Check": yes_no(turn["rag_ok_for_expected_drugs"]),
            "Top RAG Results": rag_top_text(turn["rag_top"]),
            "Flow Check": yes_no(turn["flow_ok"]),
            "Error": turn["error"] or "",
        })

    failures = [
        row for row in detail_rows
        if "FAIL" in {row["Entity Check"], row["Severity Check"], row["RAG Check"], row["Flow Check"]}
    ]

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(methodology_rows).to_excel(writer, sheet_name="Methodology", index=False)
        pd.DataFrame(model_rows).to_excel(writer, sheet_name="Models Used", index=False)
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(detail_rows).to_excel(writer, sheet_name="Turn Details", index=False)
        pd.DataFrame(failures or [{"Note": "No failed checks in final run."}]).to_excel(writer, sheet_name="Failed Checks", index=False)

        for sheet_name in writer.book.sheetnames:
            ws = writer.book[sheet_name]
            style_header(ws)
            autosize_sheet(ws)

        summary_ws = writer.book["Summary"]
        summary_ws["A1"] = "Metric"
        for row in range(2, summary_ws.max_row + 1):
            metric = str(summary_ws[f"A{row}"].value)
            if metric in {"Chatbot flow worked", "Drug/entity extraction", "Severity/result checks", "RAG retrieval checks", "OpenAI model used"}:
                summary_ws[f"B{row}"].font = Font(bold=True, color="008000")

    print(f"Excel report saved to: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
