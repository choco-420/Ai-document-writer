
import os
import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from docx import Document
from docx.shared import Pt, Inches

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(
    model=MODEL,
    base_url=BASE_URL,
    temperature=0.3,
)

OUTPUT_DIR = Path("generated")
OUTPUT_DIR.mkdir(exist_ok=True)


SYSTEM_PROMPT = """
You are a professional AI document writer.

Your job is to understand the user's English instructions
and create complete, accurate, professional documents.

You can write:
- Business plans and proposals
- Standard operating procedures (SOPs)
- Company policies
- HR documents
- Employee handbooks
- Technical documentation
- Project reports
- Meeting minutes
- Business letters
- Financial and administrative reports
- Compliance and security documents

Rules:
1. Understand the user's instructions carefully.
2. Follow the requested document type and purpose.
3. Use professional, clear English.
4. Organize content with meaningful headings.
5. Use numbered lists and bullet points where useful.
6. Include tables when they improve readability.
7. Make documents detailed and practical.
8. Never invent company facts, laws, statistics or financial figures.
9. Mark missing important information as [REQUIRES CONFIRMATION].
10. Return the document itself, not an explanation of how to write it.
"""


def generate_document(prompt: str) -> str:
    response = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", prompt)
    ])

    return response.content


def save_markdown(content: str, title: str) -> Path:
    path = OUTPUT_DIR / f"{title}.md"
    path.write_text(content, encoding="utf-8")
    return path


def save_word(content: str, title: str) -> Path:
    doc = Document()

    section = doc.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)

    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(11)

    in_code_block = False

    for line in content.splitlines():
        line = line.rstrip()

        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue

        if not line.strip():
            continue

        if line.startswith("# "):
            doc.add_heading(line[2:], level=1)

        elif line.startswith("## "):
            doc.add_heading(line[3:], level=2)

        elif line.startswith("### "):
            doc.add_heading(line[4:], level=3)

        elif line.startswith("- ") or line.startswith("* "):
            doc.add_paragraph(line[2:], style="List Bullet")

        elif line[:1].isdigit() and ". " in line[:5]:
            doc.add_paragraph(line.split(". ", 1)[1],
                              style="List Number")

        else:
            paragraph = doc.add_paragraph(line)
            if in_code_block:
                for run in paragraph.runs:
                    run.font.name = "Consolas"

    path = OUTPUT_DIR / f"{title}.docx"
    doc.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Local AI Document Writer"
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Describe the document you want"
    )

    parser.add_argument(
        "--title",
        default="AI_Generated_Document",
        help="Output document filename"
    )

    args = parser.parse_args()

    print("\nAI is understanding your instructions...")
    print("Generating your document...\n")

    try:
        content = generate_document(args.prompt)

        safe_title = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in args.title.replace(" ", "_")
        )

        md_path = save_markdown(content, safe_title)
        docx_path = save_word(content, safe_title)

        print("Document generated successfully!")
        print(f"Markdown: {md_path}")
        print(f"Word: {docx_path}")

    except Exception as error:
        print(f"Generation failed: {error}")


if __name__ == "__main__":
    main()
