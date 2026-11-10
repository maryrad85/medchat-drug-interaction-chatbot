from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


OUTPUT_DIR = Path.cwd() / "medchat_llm_outputs"
DOCX_PATH = OUTPUT_DIR / "medchat_chatbot_structure_explanation.docx"


def add_bullets(document, items):
    for item in items:
        document.add_paragraph(item, style="List Bullet")


def add_numbered(document, items):
    for item in items:
        document.add_paragraph(item, style="List Number")


def add_code_block(document, text):
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.font.name = "Courier New"
    run.font.size = Pt(9)
    paragraph.paragraph_format.left_indent = Inches(0.25)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    doc = Document()

    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(11)

    title = doc.add_heading("MedChat Chatbot Structure Explanation", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph("Drug Interaction Chatbot Using DDInter, ML, RAG, and an OpenAI Response Layer")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("Overview", level=1)
    doc.add_paragraph(
        "MedChat is a drug interaction assistant. The goal is to let a user enter medications "
        "in natural language, then check whether those medications have known interactions using "
        "the DDInter drug interaction database. The chatbot combines rule-based logic, database "
        "lookup, a local machine learning model, RAG retrieval, and an OpenAI language model response layer."
    )

    doc.add_heading("High-Level Architecture", level=1)
    add_code_block(
        doc,
        "User Message\n"
        "-> Drug Extraction\n"
        "-> Conversation State Update\n"
        "-> DDInter Database Lookup\n"
        "-> ML Severity Prediction if Needed\n"
        "-> RAG Evidence Retrieval\n"
        "-> OpenAI Natural Language Response\n"
        "-> Chatbot Reply"
    )

    doc.add_heading("1. User Input Layer", level=1)
    doc.add_paragraph(
        "The user talks to the chatbot in normal language. They do not need to use exact database names."
    )
    add_bullets(doc, [
        "Example: I take warfarin and ibuprofen",
        "Example: Can I take Advil with Coumadin?",
        "Example: What about aspirin?",
    ])

    doc.add_heading("2. Drug Name Extraction", level=1)
    doc.add_paragraph("The chatbot identifies medications using three matching methods:")
    add_numbered(doc, [
        "Exact matching against drug names in the DDInter database.",
        "Synonym matching for brand names such as Advil -> Ibuprofen and Coumadin -> Warfarin.",
        "Fuzzy matching with RapidFuzz for misspellings such as metforim -> Metformin.",
    ])
    doc.add_paragraph(
        "The system also filters common false positives like take, daily, condition, blood, yes, and no, "
        "so normal sentence words are not mistaken for medications."
    )

    doc.add_heading("3. Conversation State", level=1)
    doc.add_paragraph("The chatbot stores conversation memory in a session state. This includes:")
    add_bullets(doc, [
        "medications",
        "dose or timing notes",
        "allergies",
        "health conditions",
        "age group",
        "main concern",
        "last checked drug pair",
        "last evidence found",
        "current conversation stage",
    ])
    doc.add_paragraph(
        "This is what allows multi-turn behavior. For example, if the user first says they take "
        "Warfarin and Ibuprofen, then later asks, What about aspirin?, the chatbot remembers the earlier "
        "medications and checks aspirin against the existing profile."
    )

    doc.add_heading("4. Guided Intake Flow", level=1)
    doc.add_paragraph("The chatbot follows a structured intake flow:")
    add_numbered(doc, [
        "Ask for medications.",
        "Ask for dose or timing.",
        "Ask about allergies.",
        "Ask about health conditions.",
        "Ask age group.",
        "Ask the main concern.",
        "Ask the user to confirm the profile.",
        "Run the interaction check.",
    ])
    doc.add_paragraph(
        "This makes the chatbot more realistic than a single-question system because it collects "
        "context before giving the interaction recommendation."
    )

    doc.add_heading("5. DDInter Database Lookup", level=1)
    doc.add_paragraph(
        "The main medical evidence comes from the DDInter dataset. The CSV files are loaded into a "
        "SQLite database table called interactions. For every pair of medications, the chatbot searches "
        "both drug orders."
    )
    add_code_block(doc, "Warfarin + Ibuprofen\nIbuprofen + Warfarin")
    doc.add_paragraph("If the pair is found, the chatbot returns the DDInter severity level:")
    add_bullets(doc, ["Major", "Moderate", "Minor", "Unknown"])

    doc.add_heading("6. Local Machine Learning Model", level=1)
    doc.add_paragraph(
        "If an exact pair is not found in DDInter, the chatbot uses a local machine learning fallback model. "
        "The champion model is Linear SVM. It was trained on DDInter drug-pair text using TF-IDF features "
        "to predict interaction severity."
    )
    doc.add_paragraph("This creates two levels of evidence:")
    add_numbered(doc, [
        "Exact DDInter lookup when the pair exists in the database.",
        "Linear SVM severity prediction when no exact pair is found.",
    ])

    doc.add_heading("7. RAG Retrieval", level=1)
    doc.add_paragraph(
        "RAG means Retrieval-Augmented Generation. In this project, DDInter interactions are converted "
        "into text chunks such as:"
    )
    add_code_block(doc, "Warfarin and Ibuprofen have a Major severity drug interaction.")
    doc.add_paragraph(
        "The chatbot retrieves the most relevant interaction records for the user query. In the final "
        "evaluation, the RAG index used the full DDInter table with 160,235 interaction rows. This helps "
        "ground the chatbot response in retrieved evidence instead of relying only on the language model."
    )

    doc.add_heading("8. OpenAI LLM Response Layer", level=1)
    doc.add_paragraph(
        "The OpenAI model is used after deterministic evidence is collected. It receives the session state "
        "and the DDInter/RAG/ML evidence, then writes a natural response for the user."
    )
    doc.add_paragraph(
        "The important design point is that the LLM is not the main source of medical facts. The facts come "
        "from the DDInter lookup, RAG retrieved evidence, and the local ML model. The LLM mainly improves "
        "the wording and conversational quality."
    )

    doc.add_heading("9. Safety Rules", level=1)
    doc.add_paragraph(
        "The chatbot includes emergency keyword detection. If the user mentions symptoms such as chest pain, "
        "cannot breathe, severe bleeding, overdose, or seizure, the chatbot immediately advises urgent care "
        "instead of continuing the normal interaction flow."
    )
    doc.add_paragraph(
        "The chatbot also explains that the response is educational and should be confirmed with a doctor "
        "or pharmacist."
    )

    doc.add_heading("10. Evaluation Results", level=1)
    doc.add_paragraph("The chatbot was tested using 50 scripted conversation turns. The evaluation checked:")
    add_bullets(doc, [
        "whether the chatbot replied correctly",
        "whether it stayed in the correct conversation stage",
        "whether drug names were extracted correctly",
        "whether severity results matched expectations",
        "whether RAG retrieved the right drug-pair context",
        "whether the OpenAI model response layer worked",
    ])
    add_code_block(
        doc,
        "Chatbot flow: 50/50 passed\n"
        "Drug extraction: 10/10 passed\n"
        "Severity checks: 10/10 passed\n"
        "RAG pair retrieval: 5/5 passed\n"
        "OpenAI model: used successfully\n"
        "Local ML model: Linear SVM\n"
        "RAG corpus: 160,235 DDInter rows"
    )

    doc.add_heading("Short Presentation Version", level=1)
    doc.add_paragraph(
        "My chatbot uses a hybrid architecture. It is not only an LLM chatbot. First, it extracts drug names "
        "using exact matching, synonyms, and fuzzy matching. Then it keeps a session state for multi-turn "
        "conversation. It checks drug pairs against the DDInter SQLite database. If an exact pair is missing, "
        "it uses a trained Linear SVM model to predict severity. It also uses RAG retrieval over DDInter "
        "interaction records to provide grounded context. Finally, the OpenAI model generates a natural "
        "response using only the retrieved evidence and session information. This makes the chatbot more "
        "reliable because the medical facts come from structured data and retrieval, not just from the "
        "language model."
    )

    doc.save(DOCX_PATH)
    print(f"Word document saved to: {DOCX_PATH}")


if __name__ == "__main__":
    main()
