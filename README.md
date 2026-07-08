# BAE — talk to your data like it's your bae

A local RAG system that knows when to calculate and when to search. Routes math questions to pandas (exact computation) and text questions to vector search (semantic retrieval). No data leaves your machine. Free to use.

**This project is for local/portfolio use only. Not licensed for commercial deployment.**

## How it works

```
Browser (frontend/index.html)
   │  upload CSV, Excel, PDF, or DOCX
   ▼
FastAPI backend (backend/main.py)
   │
   ├──► Structured engine (data_engine.py)
   │     CSV/XLSX → pandas → schema → LLM picks safe operation
   │     "average fare?" → groupby_agg → exact numbers
   │
   └──► Semantic engine (rag_engine.py)
           PDF/DOCX/TXT → chunk → sentence-transformers → vector search
           "why did the Titanic sink?" → retrieve chunks → LLM summarizes
```

**Key design principle:** The LLM never touches your data or executes code. It only picks from a fixed whitelist of 8 safe operations. All computation happens in real pandas. Numbers cannot be hallucinated.

## What you need

- Python **3.10 – 3.12** (3.14 may work but requires manual torch/sentence-transformers setup)
- A **free Groq API key** from https://console.groq.com
- ~1 GB free disk space (for the sentence-transformers model)
- Internet connection (first run only — downloads the embedding model + Groq calls)

## Setup instructions

### 1. Clone the repository

```bash
git clone https://github.com/shreyassindhur/BAE.git
cd BAE
```

### 2. Set up the backend

```bash
cd backend
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure your API key

Create a `.env` file inside the `backend` folder:

```
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Get a free key at https://console.groq.com — no credit card required.

### 4. Start the backend

```bash
python -m uvicorn main:app --reload --port 8000
```

Wait for the server to fully start. The first startup will download the `all-MiniLM-L6-v2` embedding model (~90 MB). This happens once.

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 5. Open the frontend

Open `frontend/index.html` in any modern browser (Chrome, Edge, Firefox).

The frontend is already configured to connect to `http://localhost:8000`.

### 6. Upload data and ask questions

**Structured data (CSV/XLSX):**
- "What is the average fare by class?"
- "Show top 5 oldest passengers"
- "Which columns have missing data?"
- "What are the common traits of survivors?"

**Documents (PDF/DOCX/TXT):**
- "Why did the Titanic sink?"
- "Explain the historical context"
- Any question answered by your document's content

## Files

| File | Purpose |
| :--- | :--- |
| `backend/main.py` | FastAPI server — upload, query, clear endpoints |
| `backend/llm_agent.py` | LLM interaction — routing, operation planning, summarization |
| `backend/data_engine.py` | Pandas operations — 8 safe operations on structured data |
| `backend/rag_engine.py` | Document RAG — text extraction, chunking, embeddings, search |
| `backend/requirements.txt` | Python dependencies |
| `frontend/index.html` | Single-page UI — upload, chat, results |

## Limitations (read before using)

- **No multi-source merge** — one question uses either structured OR semantic, not both
- **No derived columns** — can only query columns that exist in the file
- **No predictive analytics** — no forecasting, regression, or ML
- **No multi-file joins** — works on one structured dataset per session
- **Not for production deployment** — single-user, in-memory storage

## License

This project is for **personal and portfolio use only**. Commercial deployment, hosting as a service, or redistribution for profit is not permitted without explicit permission.

## Built with

- [FastAPI](https://fastapi.tiangolo.com/)
- [sentence-transformers](https://www.sbert.net/)
- [Groq](https://groq.com/)
- [LangChain Text Splitters](https://python.langchain.com/)
- [pandas](https://pandas.pydata.org/)
