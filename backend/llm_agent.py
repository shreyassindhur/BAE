"""
llm_agent.py

Talks to Groq's OpenAI-compatible free API to turn a natural-language question
+ dataset schema into a STRUCTURED operation spec (JSON) that data_engine.py
can safely execute. The LLM never touches the data directly and never writes
executable code - it only ever picks from a fixed menu of operations.

Get a free API key at https://console.groq.com (free tier, no card required
at time of writing - always double check current limits on their site).
"""

import os
import json
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are a data-analysis planning assistant. You do NOT answer questions directly.
Instead, given a dataset schema and a user's question (plus recent conversation history),
you output ONE JSON object describing which operation to run against the data.

You must pick "operation" from exactly this list, with this field structure:

1. groupby_agg: {"operation":"groupby_agg","group_by":"<col>","agg_column":"<col>","agg_func":"sum|mean|count|min|max|median|nunique","sort_desc":true,"limit":10}
2. filter_sort_top_n: {"operation":"filter_sort_top_n","sort_column":"<col>","sort_desc":true,"limit":10,"filters":[{"column":"<col>","operator":"==|>|<|>=|<=|contains","value":<val>}]}
3. value_counts: {"operation":"value_counts","column":"<col>","limit":20}
4. describe: {"operation":"describe","column":"<col or omit for whole dataset>"}
5. trend_over_time: {"operation":"trend_over_time","date_column":"<col>","value_column":"<col>","agg_func":"sum|mean|count","freq":"D|W|M|Y"}
   NOTE: ONLY use this if the dataset contains a column with type "date". If not, do NOT use this.
6. correlation: {"operation":"correlation","columns":["<col>", "..."] }
7. raw_preview: {"operation":"raw_preview","limit":10}
8. missing_values: {"operation":"missing_values","limit":50}

Use missing_values whenever the user asks about blanks, nulls, missing data,
missing rows, or data quality.

If the user asks for "traits", "characteristics", "profiles", or "differences" 
between groups, prioritize "groupby_agg" to compare averages/stats across those groups, 
or "correlation" to see what relates to the target. Do NOT just use "value_counts" 
unless specifically asked for frequencies.

CRITICAL: When the user asks about a specific subgroup (e.g. "who died", 
"survivors", "first class", "men", "women") — ALWAYS use "groupby_agg" with the 
relevant column as group_by to show BOTH groups side by side, OR "filter_sort_top_n"
with the correct filter. Do NOT accidentally compute the wrong group.

Rules:
- ONLY use column names that literally exist in the provided schema.
- Respond with ONLY ONE valid JSON object. No markdown, no explanations, no backticks.
- If the question requires multiple steps, pick the most important ONE.
- If the question is ambiguous or unanswerable with these operations, respond with:
  {"operation":"clarify","message":"<a short clarifying question to ask the user>"}
"""


def _build_user_prompt(schema: dict, question: str, history: list[dict]) -> str:
    history_text = ""
    if history:
        recent = history[-6:]  # keep prompt small
        history_text = "\n".join(
            f"{h['role']}: {h['content']}" for h in recent
        )
    return f"""DATASET SCHEMA:
{json.dumps(schema, indent=2)}

RECENT CONVERSATION:
{history_text}

USER QUESTION:
{question}

Respond with the JSON operation spec now."""


def get_operation_spec(schema: dict, question: str, history: list[dict]) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and set it as an environment variable."
        )

    # Retry logic: if parsing fails, ask the model to fix it up to 2 times.
    for attempt in range(2):
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(schema, question, history)},
            ],
            "temperature": 0,
            "max_tokens": 500,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        raw_text = resp.json()["choices"][0]["message"]["content"].strip()

        # Defensive cleanup
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        # In case it returned a list, just take the first one
        if cleaned.startswith("[") and cleaned.endswith("]"):
            import ast
            try:
                cleaned = json.dumps(ast.literal_eval(cleaned)[0])
            except:
                pass

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            if attempt == 0:
                # Add a nudge in the system prompt for the next turn
                payload["messages"].insert(1, {"role": "system", "content": "The previous output was invalid JSON. Output ONLY valid JSON."})
                continue
            else:
                raise RuntimeError(f"Model failed to return valid JSON after retry: {raw_text}")


def expand_followup(question: str, history: list[dict]) -> str:
    """
    If the current question is short/ambiguous, expand it using the last
    Q&A pair in conversation history. This ensures follow-up questions like
    "In which country?" become "In which country was Netflix founded?".
    Returns the (possibly expanded) question.
    """
    if len(question.split()) >= 4:
        return question

    recent = [h for h in history[-4:] if h["role"] in ("user", "assistant")]
    if not recent:
        return question

    history_text = "\n".join(
        f"{h['role']}: {h['content'][:200]}" for h in recent
    )

    prompt = f"""Given this conversation history and a new follow-up question,
rewrite the follow-up into a complete, standalone question that captures all
necessary context. Respond with ONLY the rewritten question, nothing else.

CONVERSATION:
{history_text}

FOLLOW-UP: {question}

REWRITTEN QUESTION:"""

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return question

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You rewrite follow-up questions to be standalone. Output only the rewritten question."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 100,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        expanded = resp.json()["choices"][0]["message"]["content"].strip().strip('"')
        return expanded if len(expanded.split()) >= len(question.split()) else question
    except Exception:
        return question


def classify_question(question: str, has_structured: bool, has_documents: bool, history: list[dict]) -> str:
    """
    Decides which engine should answer this question:
    - "structured": needs computation over tabular data (grouping, trends, stats)
    - "semantic":   needs retrieval over document text (contracts, reports, policies)
    - "both":       ambiguous enough to try structured first, fall back to semantic
    This is the core "standout feature" - being explicit about routing instead
    of just always doing retrieval, which is where naive RAG systems fail on
    numeric/aggregation questions.
    """
    if has_structured and not has_documents:
        return "structured"
    if has_documents and not has_structured:
        return "semantic"

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    prompt = f"""A user has uploaded BOTH a structured dataset (spreadsheet/CSV) AND
one or more documents (PDFs/Word docs) in this session.

Question: "{question}"

Decide which source should answer it:
- "structured" if it needs computation over rows/columns (totals, averages, trends, top-N, correlations, missing values)
- "semantic" if it needs information from document text (policies, clauses, definitions, explanations, contract terms)

Respond with ONLY one word: structured or semantic"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a routing classifier. Respond with exactly one word."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 10,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    choice = resp.json()["choices"][0]["message"]["content"].strip().lower()
    return "structured" if "structured" in choice else "semantic"


def answer_from_documents(question: str, retrieved_chunks: list[dict], history: list[dict]) -> dict:
    """
    Answers a question using ONLY the retrieved chunks, with citations back
    to source file + chunk index. If nothing relevant was retrieved, says so
    explicitly instead of guessing - this is the "I don't know is a valid
    answer" feature.
    """
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find anything in your uploaded documents relevant to that question. "
                      "Try rephrasing, or check the document actually covers this topic.",
            "citations": [],
        }

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    context_blocks = "\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = f"""Answer the user's question using ONLY the context below.
Start directly with the answer. Be confident if the context supports it.
If the context lacks the information entirely, say so — but don't lead with
what's missing when you already have enough to answer.
Do not invent facts not present in the context.

CONTEXT:
{context_blocks}

QUESTION: {question}

Give a clear, plain-English answer (3-6 sentences)."""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You answer strictly from provided context and never invent facts."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"].strip()

    citations = [
        {"source": c["source"], "chunk_index": c["chunk_index"], "relevance": c["score"]}
        for c in retrieved_chunks
    ]
    return {"answer": answer, "citations": citations}


def summarize_result(question: str, operation_result: dict, history: list[dict]) -> str:
    """
    Optional second LLM call: turn the structured table result into a short,
    plain-English answer. Keeps the heavy lifting (actual math) in pandas,
    and only uses the LLM for phrasing - so numbers can't be hallucinated.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        # Fallback: just return the machine summary if no key configured
        return operation_result.get("summary", "Here are the results.")

    table = operation_result.get("table", [])
    if not isinstance(table, list):
        table = [table]
    table_preview = json.dumps(table[:15], indent=2)

    prompt = f"""The user asked: "{question}"

We computed this result using real pandas operations (numbers are exact, already correct):
{table_preview}

Operation summary: {operation_result.get('summary')}

Write a short (2-4 sentence), plain-English answer for a non-technical small
business owner. 
CRITICAL: Do NOT just describe the table. Provide an ANALYTICAL insight or 
conclusion based on the data. For example, explain *why* this result matters, 
or highlight the most important pattern/difference.
Do not invent any numbers not present in the table above."""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You explain data results in plain, friendly English."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 300,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
