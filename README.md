# Health Article Simplifier & Medical Q&A

A Python + **Streamlit** + **LangChain** + **Groq** + **FAISS** application that fetches any health article from a **URL** (or pasted text / PDF), simplifies it into **Simple English** or **Indian regional languages** without altering clinical facts, builds a searchable **Medical Terms Glossary**, and lets users ask follow-up questions grounded in the article using **FAISS RAG**.

## Features

- **Article Link (URL) Reader**: Paste any health article link (WHO, CDC, Mayo Clinic, MedlinePlus, health news) — automatically extracts clean article text using `trafilatura` + `BeautifulSoup`.
- **Paste Text & PDF/TXT Upload**: Also supports pasting raw text (with 2 built-in samples) or uploading `.pdf` / `.txt` medical articles.
- **In-Website Groq API Key Setup**: Enter your free [Groq API Key](https://console.groq.com/keys) directly in the Streamlit sidebar (no `.env` editing required).
- **Fact-Preserving Multilingual Rewrite**: Rewrites complex terms into Simple English, Marathi, Hindi, Gujarati, Tamil, Telugu, Bengali, Kannada, Malayalam, Punjabi, Urdu, or Spanish while listing preserved clinical numbers, dosages, and warnings.
- **Searchable Medical Terms Glossary**: Extracts medical jargon with plain-language meanings and article context, exportable as `.csv`.
- **FAISS RAG Document Q&A**: Splits the full article into chunks, embeds them locally with `sentence-transformers/all-MiniLM-L6-v2` (100% free, no extra API key), indexes them in **FAISS**, and answers questions with retrieved source excerpts.

## Quick Start

### Option 1: One-Click Script
- **Windows:** Double-click `run.bat`
- **macOS / Linux:**
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

### Option 2: Manual Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`), paste your **Groq API Key** in the left sidebar, paste a health article link, choose your language, and click **Simplify Article & Build Glossary**.

## Project Structure

```text
health-article-simplifier-pro/
├── app.py              # Streamlit UI (URL/text/PDF tabs, Groq sidebar, glossary, FAISS Q&A)
├── extractor.py        # Web URL (trafilatura + BeautifulSoup) and PDF text extraction
├── simplifier.py       # LangChain + ChatGroq fact-preserving simplification & glossary chain
├── rag_qa.py           # LangChain + FAISS + HuggingFaceEmbeddings RAG Q&A engine
├── requirements.txt    # Python dependencies
├── run.bat             # Windows launcher
├── run.sh              # macOS / Linux launcher
└── README.md           # Documentation
```

> **Note:** On the first time you use the **Ask About This Article (FAISS RAG)** tab, `sentence-transformers` downloads the lightweight `all-MiniLM-L6-v2` embedding model (~90 MB) once and caches it locally.
