# AI Document Writer

A locally running AI-powered document generation tool built with Python and Ollama.

## Features

- Generate business plans and proposals
- Create HR policies and employee documents
- Generate SOPs and technical documentation
- Save documents as Word (.docx) and Markdown (.md)
- Run AI locally using Ollama

## Requirements

- Python 3.10+
- Ollama
- Qwen 2.5 model

## Installation

1. Clone this repository.
2. Install dependencies:

   pip install -r requirements.txt

3. Install Ollama from https://ollama.com
4. Download the model:

   ollama pull qwen2.5-coder:3b

5. Run the application:

   python writer.py --title "My_Document" --prompt "Write a professional business proposal."

## Configuration

Create a .env file:

OLLAMA_MODEL=qwen2.5-coder:3b
OLLAMA_BASE_URL=http://localhost:11434

## License

MIT
