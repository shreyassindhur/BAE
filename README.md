# BAE — talk to your data like it's your bae!

A tool that lets a small business owner upload a CSV/Excel file and ask
questions about it in plain English — "what are my top 5 products?", "show
me sales trend by month" — and get back a real, computed answer (not an AI
guess), with a table and plain-English explanation.

## How it works (architecture)

```
Browser (index.html)
   │  upload file
   ▼
FastAPI backend (main.py)
   │  parses file -> pandas DataFrame, infers schema
   ▼
llm_agent.py  ──►  Groq API (free, Llama model)
   │  "Given this schema + question, output ONE structured
   │   operation spec (JSON) from a fixed menu of operations."
   ▼
data_engine.py
   │  executes ONLY that operation, using real pandas code
   │  (groupby, sort, resample, etc.) — never arbitrary code
   ▼
Result table (guaranteed-correct numbers)
   │
llm_agent.py (2nd call) -> turns table into a friendly sentence
   ▼
Returned to browser, rendered as chat bubble + table
```

**Why this design matters (and is worth explaining in interviews):**
The LLM is never allowed to execute code or do arithmetic itself. It only
ever picks from a small, fixed set of operations (`groupby_agg`,
`trend_over_time`, `value_counts`, etc.), each with validated column names.
The actual computation always happens in real pandas. This means:
- Numbers can never be hallucinated
- A malicious/confused LLM response can, at worst, be rejected — never executed as code
- It's a genuinely defensible security/reliability story, not just "we called an LLM"

## Running it locally (free)

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # then edit .env and add your GROQ_API_KEY
```

Get a **free** Groq API key at https://console.groq.com (no card required
at time of writing — always check their current pricing/limits page).

```bash
export $(cat .env | xargs)   # loads env vars on Mac/Linux
uvicorn main:app --reload --port 8000
```

Backend is now running at `http://localhost:8000`.

### 2. Frontend

Just open `frontend/index.html` in a browser. If your backend is running
locally on port 8000, it already points there by default (`API_BASE` at the
top of the `<script>` tag).

Upload a CSV, ask a question, done.
See the go-to-market plan discussed alongside this build: target one small
vertical first (retail/e-commerce shop owners are the easiest first
customers), price around ₹1,500–3,000/month or a flat setup fee, and lead
with the pitch: "Upload your sales sheet, ask questions in plain English,
no Excel formulas needed."
