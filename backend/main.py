"""
main.py - FastAPI backend for the Chat-With-Your-Data tool.

Endpoints:
- POST /upload         -> upload a CSV/XLSX, get back a session_id + schema
- POST /query          -> ask a natural-language question about an uploaded dataset
- GET  /health         -> simple healthcheck

Run locally:
    uvicorn main:app --reload --port 8000

Free deployment: Render.com free web service, Railway free tier, or
Fly.io free tier all work well for a small FastAPI app like this.
"""

import os
import uuid
import shutil
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # reads .env in this folder and sets the environment variables

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import data_engine
import llm_agent
import rag_engine

app = FastAPI(title="Chat With Your Data API")

# Allow the frontend (any origin during dev; tighten this before going live)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory session store: session_id -> {
#   "df": DataFrame | None, "schema": dict | None,
#   "doc_store": DocumentStore | None, "history": [...]
# }
# NOTE: fine for an MVP / single instance. For real production with multiple
# users at once, swap this for Redis or a proper DB-backed session store.
SESSIONS: dict[str, dict] = {}

STRUCTURED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"}


class QueryRequest(BaseModel):
    session_id: str
    question: str


class ClearRequest(BaseModel):
    session_id: str


class RemoveSourceRequest(BaseModel):
    session_id: str
    source: str


def _new_session() -> dict:
    return {"df": None, "schema": None, "structured_source": None, "doc_store": None, "history": []}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    """
    Routes the uploaded file by extension:
    - .csv/.xlsx/.xls  -> structured engine (pandas)
    - .pdf/.docx/.txt/.md -> document engine (RAG)
    Pass an existing session_id to add more files (of either kind) to the
    same session, so a user can mix a spreadsheet and documents together.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot read \"{file.filename}\" (this system does not support image input)."
        )
    if ext not in STRUCTURED_EXTENSIONS and ext not in DOCUMENT_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    if session_id and session_id in SESSIONS:
        session = SESSIONS[session_id]
    else:
        session_id = str(uuid.uuid4())
        session = _new_session()
        SESSIONS[session_id] = session

    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if ext in STRUCTURED_EXTENSIONS:
        try:
            df = data_engine.load_dataframe(file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")
        session["df"] = df
        session["schema"] = data_engine.infer_schema(df)
        session["structured_source"] = file.filename
        message = f"Loaded {len(df)} rows and {len(df.columns)} columns from {file.filename}."
    else:
        if session["doc_store"] is None:
            session["doc_store"] = rag_engine.DocumentStore()
        try:
            n_chunks = session["doc_store"].add_document(file_path, file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not process document: {e}")
        message = f"Indexed {file.filename} into {n_chunks} searchable chunks."

    return {
        "session_id": session_id,
        "schema": session["schema"],
        "documents": session["doc_store"].list_sources() if session["doc_store"] else [],
        "message": message,
    }


@app.post("/clear")
async def clear_sources(req: ClearRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    session["df"] = None
    session["schema"] = None
    session["structured_source"] = None
    session["doc_store"] = None
    session["history"] = []
    return {"message": "All sources cleared."}


@app.post("/remove-source")
async def remove_source(req: RemoveSourceRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if session.get("structured_source") == req.source:
        session["df"] = None
        session["schema"] = None
        session["structured_source"] = None
    elif session["doc_store"]:
        session["doc_store"].remove_source(req.source)
    else:
        raise HTTPException(status_code=404, detail="Source not found.")

    remaining = []
    if session["structured_source"]:
        remaining.append({"name": session["structured_source"], "type": "structured"})
    if session["doc_store"]:
        for s in session["doc_store"].list_sources():
            remaining.append({"name": s, "type": "document"})

    return {"message": f"Removed {req.source}.", "sources": remaining, "schema": session["schema"]}


@app.post("/query")
async def query(req: QueryRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please re-upload your file.")

    history = session["history"]
    has_structured = session["df"] is not None
    has_documents = session["doc_store"] is not None and session["doc_store"].has_documents()

    if not has_structured and not has_documents:
        raise HTTPException(status_code=400, detail="Upload a spreadsheet or document first.")

    # Expand follow-up questions using conversation history
    question = llm_agent.expand_followup(req.question, history)

    try:
        route = llm_agent.classify_question(question, has_structured, has_documents, history)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if route == "semantic":
        chunks = session["doc_store"].search(question)
        result = llm_agent.answer_from_documents(question, chunks, history)
        history.append({"role": "user", "content": req.question})
        history.append({"role": "assistant", "content": result["answer"]})
        return {
            "answer": result["answer"],
            "table": None,
            "chart_hint": "none",
            "citations": result["citations"],
            "route": "semantic",
        }

    # structured path
    df = session["df"]
    schema = session["schema"]

    try:
        spec = llm_agent.get_operation_spec(schema, question, history)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if spec.get("operation") == "clarify":
        answer = spec.get("message", "Could you clarify your question?")
        history.append({"role": "user", "content": req.question})
        history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "table": None, "chart_hint": "none", "route": "structured"}

    try:
        result = data_engine.execute_operation(df, spec)
    except data_engine.OperationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    answer = llm_agent.summarize_result(question, result, history)

    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "table": result["table"],
        "chart_hint": result["chart_hint"],
        "operation": spec.get("operation"),
        "route": "structured",
    }
