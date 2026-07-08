"""
rag_engine.py

RAG pipeline: extract text -> split with LangChain's RecursiveCharacterTextSplitter
-> embed each chunk via sentence-transformers -> retrieve via
cosine similarity (plain numpy).
"""

import os
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import docx  # python-docx

from sentence_transformers import SentenceTransformer

# Load model once at startup
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Text extraction ----------

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        d = docx.Document(file_path)
        return "\n".join(p.text for p in d.paragraphs)
    if ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError(f"Unsupported document type: {ext}")


# ---------- Chunking ----------

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return _splitter.split_text(text)


# ---------- Embeddings (sentence-transformers) ----------

def embed_texts(texts: list[str]) -> np.ndarray:
    return _embedding_model.encode(texts, normalize_embeddings=True)


def _cosine_sim(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norms @ query_norm


# ---------- Store + retrieval (per session, in-memory) ----------

class DocumentStore:
    def __init__(self):
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.chunk_indices: list[int] = []
        self.embeddings: np.ndarray | None = None

    def add_document(self, file_path: str, source_name: str) -> int:
        text = extract_text(file_path)
        new_chunks = chunk_text(text)
        if not new_chunks:
            return 0

        new_embeddings = embed_texts(new_chunks)

        self.chunks.extend(new_chunks)
        self.sources.extend([source_name] * len(new_chunks))
        self.chunk_indices.extend(range(len(new_chunks)))

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        return len(new_chunks)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.3) -> list[dict]:
        if self.embeddings is None or not self.chunks:
            return []

        query_vec = embed_texts([query])[0]
        scores = _cosine_sim(query_vec, self.embeddings)

        ranked_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in ranked_idx:
            score = float(scores[i])
            if score < min_score:
                continue
            results.append({
                "text": self.chunks[i],
                "source": self.sources[i],
                "chunk_index": self.chunk_indices[i],
                "score": round(score, 3),
            })
        return results

    def has_documents(self) -> bool:
        return len(self.chunks) > 0

    def list_sources(self) -> list[str]:
        return sorted(set(self.sources))

    def remove_source(self, source_name: str) -> bool:
        """Remove all chunks belonging to a source. Returns True if found."""
        indices = [i for i, s in enumerate(self.sources) if s == source_name]
        if not indices:
            return False

        keep = [i for i in range(len(self.chunks)) if i not in indices]

        self.chunks = [self.chunks[i] for i in keep]
        self.sources = [self.sources[i] for i in keep]
        self.chunk_indices = [self.chunk_indices[i] for i in keep]
        if self.embeddings is not None and len(keep) > 0:
            self.embeddings = self.embeddings[keep]
        elif not keep:
            self.embeddings = None

        return True
