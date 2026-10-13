# PaperLens

**An AI-powered research paper assistant for understanding, summarizing, and exploring academic papers.**

PaperLens helps students and researchers navigate complex research papers through structured explanations, concise summaries, contextual question answering, technical word definitions, and deeper topic exploration.

## Features

- **Understand Paper:** Explore the paper's main ideas and technical content in an accessible, structured format.
- **Paper Summary:** Review the key ideas, methodology, findings, and conclusions.
- **Ask the Paper:** Ask questions about the uploaded document and receive context-based responses.
- **Word Explorer:** Understand unfamiliar technical terms.
- **Deep Dive:** Explore a particular topic or section in greater detail.

## Tech Stack

- **Python** — application logic
- **Streamlit** — interactive web interface
- **pypdf** — PDF text extraction
- **Groq API** — hosted language-model access
- **`openai/gpt-oss-20b`** — configured language model
- **python-dotenv** — local environment configuration
- **langchain-text-splitters** — text chunking

## How It Works

1. Upload a research paper in PDF format.
2. Extract and prepare the document text.
3. Select one of the five available features.
4. Prepare the required context and retrieve relevant text where applicable.
5. Generate an AI response through Groq.
6. Read the result and explore the paper further.

The current question-answering retrieval approach uses basic keyword overlap rather than semantic vector search.

## Getting Started

### Requirements

- Python
- A Groq API key
- Internet access for model requests

### Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/paperlens.git
cd paperlens
```

Create and activate a virtual environment if desired.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root and add the Groq API key using the environment-variable name expected by `app.py`.

Start the application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Project Structure

```text
paperlens/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── PROJECT_DOCUMENTATION.md
├── test_chunking.py
└── test_groq.py
```

The structure above describes the intended repository files. Local-only files, including `.env` and the virtual environment, should be excluded from Git.

## Current Limitations

- Scanned PDFs may require OCR.
- Retrieval currently relies on keyword overlap rather than semantic embeddings.
- AI-generated outputs should be checked against the original paper.
- Model access depends on Groq availability, account limits, and network connectivity.

## Future Scope

- Semantic search and embedding-based retrieval.
- Section-aware document processing.
- Paper-grounded answers with source references.
- OCR for scanned PDFs.
- More systematic evaluation of answer quality.

## Documentation

For the detailed technical architecture, technology-selection rationale, processing pipeline, workflows, limitations, and future improvements, see [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md).

## Security

Never commit API keys, `.env` files, or other secrets to GitHub.

## License

A license can be added once the intended reuse and distribution terms for the project have been decided.
