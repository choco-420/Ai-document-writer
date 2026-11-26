"""Local Documentation Writer Agent.

Generates README documentation and Google-style docstrings for Python modules
using a locally running Ollama model.

Examples:
    python docagent.py
    python docagent.py --file my_module.py
    python docagent.py --file my_module.py --format readme
    python docagent.py --file my_module.py --format docstrings
    python docagent.py --file my_module.py --format both --output-dir generated
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


# Load settings from a .env file in the current project directory, if present.
load_dotenv()

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "generated")
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))


README_PROMPT = """You are an experienced Python technical writer.
Create a complete, accurate README.md for the supplied Python module.

Use this structure where relevant:
1. Project/module name and concise description
2. Overview and key features
3. Requirements and installation
4. Quick start with a realistic example based on the supplied code
5. API reference for public classes and functions, including parameters,
   return values, and exceptions when they can be determined
6. Configuration and environment variables, if present
7. Error handling and troubleshooting
8. Example usage

Rules:
- Base every technical claim on the supplied code.
- Do not invent dependencies, commands, URLs, environment variables, or behavior.
- Clearly mark information that cannot be determined as [NEEDS CONFIRMATION].
- Use valid GitHub-flavored Markdown.
- Do not wrap the entire README in a Markdown code fence.
"""


DOCSTRING_PROMPT = """You are an expert Python maintainer.
Add clear Google-style docstrings to every public function, class, and method
that does not already have a useful docstring.

Important rules:
- Return the COMPLETE updated Python source file, not a patch or explanation.
- Preserve all imports, names, signatures, control flow, and program behavior.
- Do not delete, simplify, or rewrite working code.
- Do not invent behavior, return values, or exceptions.
- Include Args, Returns, Raises, and Examples only when relevant and supported
  by the code.
- Keep existing useful docstrings.
- Do not add comments about the task or the AI model.
- Do not wrap the source in Markdown fences.
"""


SAMPLE_CODE = r'''import hashlib
import os
import re
from datetime import datetime


def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    if salt is None:
        salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations=600_000
    )
    return salt, digest


def parse_date(date_string: str) -> datetime:
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%Y%m%d"]
    for date_format in formats:
        try:
            return datetime.strptime(date_string, date_format)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {date_string}")


class UserValidator:
    MIN_PASSWORD_LENGTH = 8

    def validate_username(self, username: str) -> tuple[bool, str]:
        if len(username) < 3:
            return False, "Username too short"
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, "Only letters, numbers, and underscores are allowed"
        return True, ""

    def validate_password(self, password: str) -> tuple[bool, str]:
        if len(password) < self.MIN_PASSWORD_LENGTH:
            return False, (
                f"Password must be at least {self.MIN_PASSWORD_LENGTH} characters"
            )
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"
        return True, ""
'''


def create_llm(model: str, base_url: str, num_ctx: int) -> ChatOllama:
    """Create the Ollama chat model client."""
    return ChatOllama(
        model=model,
        temperature=0,
        base_url=base_url,
        num_ctx=num_ctx,
    )


def _response_text(response: Any) -> str:
    """Convert a LangChain response content value into plain text."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _clean_markdown_fence(text: str) -> str:
    """Remove an outer Markdown code fence from an LLM source response."""
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _format_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Format a function signature from its AST node."""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    try:
        arguments = ast.unparse(node.args)
    except Exception:
        arguments = "..."

    signature = f"{prefix} {node.name}({arguments})"
    if node.returns is not None:
        try:
            signature += f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass
    return signature


def extract_structure(code: str) -> str:
    """Extract top-level functions, classes, and class methods from Python code.

    Args:
        code: Python source code to inspect.

    Returns:
        A readable outline of the module's public structure. If parsing fails,
        a message describing the parsing problem is returned.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Could not parse Python structure: {exc}"

    structure: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            structure.append(_format_function_signature(node))
        elif isinstance(node, ast.ClassDef):
            structure.append(f"class {node.name}:")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    structure.append(f"    {_format_function_signature(item)}")

    return "\n".join(structure) if structure else "No top-level functions or classes found."


def generate_readme(
    llm: ChatOllama,
    code: str,
    filename: str,
    max_code_chars: int = 20000,
) -> str:
    """Generate a README for a Python module using the local model.

    Args:
        llm: Configured Ollama chat model.
        code: Complete Python source text.
        filename: Source filename used in the prompt.
        max_code_chars: Maximum source characters sent to the model. Use 0 to
            send the entire file.

    Returns:
        Generated Markdown README text.
    """
    structure = extract_structure(code)
    source_for_prompt = code
    truncation_note = ""

    if max_code_chars > 0 and len(code) > max_code_chars:
        source_for_prompt = code[:max_code_chars]
        truncation_note = (
            f"\nNOTE: The source included below is truncated to {max_code_chars} "
            f"characters from a file containing {len(code)} characters. "
            "Do not claim to have reviewed code that is not shown.\n"
        )

    messages = [
        SystemMessage(content=README_PROMPT),
        HumanMessage(
            content=(
                f"Source filename: {filename}\n"
                f"{truncation_note}\n"
                f"Python structure:\n{structure}\n\n"
                f"Source code:\n```python\n{source_for_prompt}\n```"
            )
        ),
    ]
    return _response_text(llm.invoke(messages))


def add_docstrings(llm: ChatOllama, code: str, filename: str) -> str:
    """Ask the local model to add docstrings and validate Python syntax.

    The function never modifies the original source file. The caller saves the
    generated result to a separate output file.

    Args:
        llm: Configured Ollama chat model.
        code: Original Python source code.
        filename: Source filename used in the prompt.

    Returns:
        Updated Python source code with docstrings.

    Raises:
        ValueError: If the model does not return syntactically valid Python
            after one correction attempt.
    """
    messages = [
        SystemMessage(content=DOCSTRING_PROMPT),
        HumanMessage(
            content=(
                f"Add docstrings to this Python file: {filename}\n\n"
                f"```python\n{code}\n```"
            )
        ),
    ]

    result = _clean_markdown_fence(_response_text(llm.invoke(messages)))

    try:
        ast.parse(result)
        return result + ("\n" if not result.endswith("\n") else "")
    except SyntaxError as first_error:
        # Give the model one chance to repair syntax without changing behavior.
        repair_messages = [
            SystemMessage(
                content=(
                    "You are a Python syntax repair assistant. Return only the "
                    "complete corrected Python source code. Preserve behavior "
                    "and make the smallest possible syntax-only correction."
                )
            ),
            HumanMessage(
                content=(
                    f"The following generated Python file has a syntax error: "
                    f"{first_error}\n\n```python\n{result}\n```\n\n"
                    "Correct the syntax and return the entire file."
                )
            ),
        ]
        repaired = _clean_markdown_fence(
            _response_text(llm.invoke(repair_messages))
        )
        try:
            ast.parse(repaired)
        except SyntaxError as second_error:
            raise ValueError(
                "The model returned invalid Python even after a repair attempt. "
                f"First error: {first_error}. Second error: {second_error}. "
                "The generated code was not saved."
            ) from second_error
        return repaired + ("\n" if not repaired.endswith("\n") else "")


def read_source_file(path: Path) -> str:
    """Read a UTF-8 Python source file and report useful file errors."""
    if not path.exists():
        raise FileNotFoundError(f"Python file not found: {path}")
    if not path.is_file():
        raise ValueError(f"The supplied path is not a file: {path}")
    if path.suffix.lower() != ".py":
        raise ValueError(f"Expected a .py file, received: {path.name}")
    return path.read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate Python README documentation and docstrings locally with Ollama."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--file", help="Path to a Python file to document")
    source_group.add_argument("--code", help="Python source code supplied directly")

    parser.add_argument(
        "--format",
        choices=["readme", "docstrings", "both"],
        default="both",
        help="Which documentation outputs to generate (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Ollama server URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_NUM_CTX,
        help=f"Ollama context window (default: {DEFAULT_NUM_CTX})",
    )
    parser.add_argument(
        "--max-code-chars",
        type=int,
        default=20000,
        help="Max source characters sent for README generation; 0 sends all code",
    )
    return parser


def main() -> int:
    """Run the command-line documentation generator."""
    parser = build_parser()
    args = parser.parse_args()

    if args.num_ctx < 1024:
        parser.error("--num-ctx must be at least 1024")
    if args.max_code_chars < 0:
        parser.error("--max-code-chars cannot be negative")

    try:
        if args.file:
            source_path = Path(args.file).expanduser().resolve()
            code = read_source_file(source_path)
            filename = source_path.name
        elif args.code is not None:
            code = args.code
            filename = "module.py"
        else:
            code = SAMPLE_CODE
            filename = "validators.py"
            print("\nUsing the built-in sample validators module.")

        # Catch invalid source before spending time or resources on the model.
        try:
            ast.parse(code)
        except SyntaxError as exc:
            print(f"Error: input is not valid Python: {exc}", file=sys.stderr)
            return 1

        output_dir = Path(args.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(filename).stem

        print(f"\nDocumentation target: {filename}")
        print(f"Ollama model: {args.model}")
        print(f"Ollama server: {args.base_url}")
        print(f"Output directory: {output_dir.resolve()}\n")

        llm = create_llm(args.model, args.base_url, args.num_ctx)

        if args.format in ("readme", "both"):
            print("Generating README...")
            readme = generate_readme(
                llm=llm,
                code=code,
                filename=filename,
                max_code_chars=args.max_code_chars,
            )
            readme_path = output_dir / f"README_{stem}.md"
            readme_path.write_text(readme.rstrip() + "\n", encoding="utf-8")
            print(f"Saved README: {readme_path.resolve()}")

        if args.format in ("docstrings", "both"):
            print("Generating documented Python source...")
            documented_code = add_docstrings(llm, code, filename)
            documented_path = output_dir / f"documented_{filename}"
            documented_path.write_text(documented_code, encoding="utf-8")
            print(f"Saved documented code: {documented_path.resolve()}")

        print("\nDone. The original source file was not changed.")
        return 0

    except FileNotFoundError as exc:
        print(f"File error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Generation error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print("\nCould not complete the request.", file=sys.stderr)
        print(f"Details: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\nCheck that Ollama is installed and running, that the model is "
            "downloaded, and that the server URL is correct. Try:\n"
            f"  ollama pull {DEFAULT_MODEL}\n"
            "  ollama list\n"
            "  ollama serve",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
