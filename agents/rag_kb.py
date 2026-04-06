"""
RAG Knowledge Base — Chunking, Embedding, and Retrieval
========================================================

This module implements lightweight RAG over the domain knowledge base that is
built per-session by the Knowledge Agent.

Architecture
------------
- Chunking  : each KB entry is already structured as 4 sections × 5-6 sub-
              sections (e.g. "1.1 Process Flow...", "2.3 Phase Behaviour...").
              We split on subsection headings → each chunk ≈ 200-600 words
              (well within the 800-token target).
- Embedding : AzureOpenAIEmbeddings (text-embedding-ada-002 or the configured
              model) — reuses the same Azure endpoint + API key.  Falls back to
              TF-IDF keyword overlap when the embedding call fails so the system
              never hard-crashes.
- Store     : lightweight numpy cosine-similarity index held in memory and
              persisted as JSON in the SQLite `kb_embeddings` table so it
              survives session resumes.  No extra dependencies required.
- Retrieval : top-k cosine similarity, returns ranked chunk texts joined for
              injection into the analyst prompt.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# EMBEDDING MODEL FACTORY
# ---------------------------------------------------------------------------

def _get_embedding_model():
    """
    Returns a callable embed(texts: list[str]) -> list[list[float]].

    Tries AzureOpenAIEmbeddings first; falls back to a simple TF-IDF
    bag-of-words approximation so the system stays functional even when
    the LLM endpoint is unreachable.
    """
    import httpx

    use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
    embed_deploy = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
                             os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"))
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_ver  = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

    _is_private = any(
        endpoint.replace("https://", "").replace("http://", "").startswith(p)
        for p in ("10.", "172.", "192.168.")
    )
    host_header = os.getenv("AZURE_OPENAI_HOST_HEADER", "")

    if use_azure:
        from langchain_openai import AzureOpenAIEmbeddings

        default_headers = {}
        if _is_private and host_header:
            default_headers["Host"] = host_header

        http_client = httpx.Client(
            verify=False,
            trust_env=not _is_private,
            timeout=httpx.Timeout(60.0, connect=15.0),
            headers=default_headers,
        )

        try:
            model = AzureOpenAIEmbeddings(
                azure_endpoint=endpoint,
                api_key=api_key,
                azure_deployment=embed_deploy,
                api_version=api_ver,
                http_client=http_client,
            )
            # Return a simple wrapper that always returns list[list[float]]
            def azure_embed(texts: List[str]) -> List[List[float]]:
                return model.embed_documents(texts)
            return azure_embed
        except Exception:
            pass  # Fall through to TF-IDF fallback

    # ── TF-IDF fallback (no external calls, pure numpy) ───────────────────
    def tfidf_embed(texts: List[str]) -> List[List[float]]:
        """
        Very lightweight bag-of-words TF-IDF embedding so RAG still works
        when the Azure embedding endpoint is unavailable.
        """
        # Build vocabulary
        tokenize = lambda t: re.findall(r"[a-z0-9]+", t.lower())
        tokenized = [tokenize(t) for t in texts]
        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)

        dim = max(len(vocab), 1)
        # Term frequency matrix
        tf = np.zeros((len(texts), dim), dtype=np.float32)
        for i, tokens in enumerate(tokenized):
            for tok in tokens:
                tf[i, vocab[tok]] += 1
            row_sum = tf[i].sum()
            if row_sum > 0:
                tf[i] /= row_sum

        # IDF
        df = (tf > 0).sum(axis=0).astype(np.float32)
        idf = np.log((len(texts) + 1) / (df + 1)) + 1.0
        tfidf = tf * idf

        # L2-normalise
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tfidf = tfidf / norms

        return tfidf.tolist()

    return tfidf_embed


# ---------------------------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------------------------

# Subsection heading patterns we enforce in the KB prompt:
#   "1.1 Process Flow...", "2.3 Phase Behaviour...", "4.5 Analytical..."
_SUBSECTION_RE = re.compile(r"(?m)^(\d+\.\d+\s+.+)$")

def chunk_knowledge_entry(category: str, topic: str, knowledge_text: str) -> List[dict]:
    """
    Splits one KB entry (a single category/topic block) into sub-section
    chunks.  Each chunk carries:
      - chunk_id   : "<category>_<section_index>"
      - section_heading : the sub-section title (e.g. "1.1 Process Flow …")
      - text       : the full text of that subsection (heading + body)
      - category   : parent category ("Process", "Physics/Chemistry", etc.)
    """
    chunks = []
    # Split text on subsection headings
    parts = _SUBSECTION_RE.split(knowledge_text)
    # parts alternates: [pre-heading-text, heading, body, heading, body, ...]

    # Emit any leading text before first heading as its own chunk
    if parts[0].strip():
        chunks.append({
            "chunk_id": f"{category}_intro",
            "section_heading": f"{category} — Introduction",
            "text": f"[{category}] {topic}\n\n{parts[0].strip()}",
            "category": category,
        })

    # Walk heading/body pairs
    i = 1
    while i < len(parts) - 1:
        heading = parts[i].strip()
        body    = parts[i + 1].strip() if (i + 1) < len(parts) else ""
        chunk_text = f"[{category}] {topic}\n\n{heading}\n{body}"
        chunks.append({
            "chunk_id": f"{category}_{heading[:20].replace(' ', '_')}",
            "section_heading": heading,
            "text": chunk_text,
            "category": category,
        })
        i += 2

    # Fallback: if no subsections were detected, treat entire entry as one chunk
    if not chunks:
        chunks.append({
            "chunk_id": f"{category}_full",
            "section_heading": f"{category} — {topic}",
            "text": f"[{category}] {topic}\n\n{knowledge_text}",
            "category": category,
        })

    return chunks


def chunk_knowledge_base(kb_entries: List[dict]) -> List[dict]:
    """
    Chunks all 4 KB entries into subsection-level chunks.

    Args:
        kb_entries: list of dicts with keys "category", "topic", "knowledge_text"
    Returns:
        Flat list of chunk dicts (with chunk_id, section_heading, text, category)
    """
    all_chunks = []
    for entry in kb_entries:
        cat   = entry.get("category", "Unknown")
        topic = entry.get("topic", "")
        text  = entry.get("knowledge_text", "")
        all_chunks.extend(chunk_knowledge_entry(cat, topic, text))
    return all_chunks


# ---------------------------------------------------------------------------
# SQLITE VECTOR STORE
# ---------------------------------------------------------------------------

DB_PATH = "database/app.db"

def _ensure_embeddings_table():
    """Creates kb_embeddings table if it doesn't exist."""
    with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kb_embeddings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id   TEXT NOT NULL,
                chunk_id    TEXT NOT NULL,
                section_heading TEXT,
                category    TEXT,
                chunk_text  TEXT NOT NULL,
                embedding   TEXT NOT NULL,   -- JSON float array
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(thread_id, chunk_id)
            )
        """)
        conn.commit()


def save_embeddings(thread_id: str, chunks: List[dict], embeddings: List[List[float]]):
    """Persists chunk texts + their embedding vectors to SQLite."""
    _ensure_embeddings_table()
    import datetime
    now = datetime.datetime.now().isoformat()
    with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
        # Remove stale embeddings for this thread first
        conn.execute("DELETE FROM kb_embeddings WHERE thread_id = ?", (thread_id,))
        for chunk, emb in zip(chunks, embeddings):
            conn.execute(
                """INSERT OR REPLACE INTO kb_embeddings
                   (thread_id, chunk_id, section_heading, category, chunk_text, embedding, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    thread_id,
                    chunk["chunk_id"],
                    chunk.get("section_heading", ""),
                    chunk.get("category", ""),
                    chunk["text"],
                    json.dumps(emb),
                    now,
                )
            )
        conn.commit()


def load_embeddings(thread_id: str) -> Tuple[List[dict], np.ndarray | None]:
    """
    Loads stored chunks + embedding matrix for a thread.

    Returns:
        (chunks, matrix) where matrix is shape (N, dim) float32 array,
        or ([], None) if nothing is stored yet.
    """
    _ensure_embeddings_table()
    with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
        rows = conn.execute(
            "SELECT chunk_id, section_heading, category, chunk_text, embedding "
            "FROM kb_embeddings WHERE thread_id = ? ORDER BY id",
            (thread_id,)
        ).fetchall()

    if not rows:
        return [], None

    chunks = [
        {"chunk_id": r[0], "section_heading": r[1], "category": r[2], "text": r[3]}
        for r in rows
    ]
    matrix = np.array([json.loads(r[4]) for r in rows], dtype=np.float32)
    return chunks, matrix


def embeddings_exist(thread_id: str) -> bool:
    """Returns True if this thread already has embeddings stored."""
    _ensure_embeddings_table()
    with sqlite3.connect(DB_PATH, timeout=5.0) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM kb_embeddings WHERE thread_id = ?", (thread_id,)
        ).fetchone()[0]
    return count > 0


# ---------------------------------------------------------------------------
# COSINE SIMILARITY SEARCH
# ---------------------------------------------------------------------------

def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Vectorised cosine similarity between query (1-D) and matrix (N×D)."""
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return np.zeros(len(matrix))
    m_norms = np.linalg.norm(matrix, axis=1)
    m_norms[m_norms == 0] = 1.0
    return (matrix @ query_vec) / (m_norms * q_norm)


def retrieve_relevant_chunks(
    query: str,
    thread_id: str,
    top_k: int = 5,
) -> str:
    """
    Main retrieval function.  Given a natural-language query and a thread_id:
      1. Loads stored chunk embeddings for the thread
      2. Embeds the query using the same model
      3. Returns the top-k chunks ranked by cosine similarity, as a
         formatted string ready to inject into a prompt.

    Returns "" if no embeddings exist (caller should fall back to full KB).
    """
    chunks, matrix = load_embeddings(thread_id)
    if not chunks or matrix is None:
        return ""

    embed_fn = _get_embedding_model()
    try:
        query_emb = np.array(embed_fn([query])[0], dtype=np.float32)
    except Exception:
        # Fallback: return first top_k chunks without ranking
        return _format_chunks(chunks[:top_k])

    scores = _cosine_similarity(query_emb, matrix)
    top_indices = np.argsort(scores)[::-1][:top_k]

    ranked_chunks = [chunks[i] for i in top_indices]

    # Print retrieval scores to terminal for visibility
    separator = "-" * 60
    print(f"\n{separator}")
    print(f"🔍 RAG RETRIEVAL — top {top_k} chunks for query: '{query[:80]}'")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  #{rank}  score={scores[idx]:.4f}  [{chunks[idx]['category']}] {chunks[idx]['section_heading'][:60]}")
    print(separator)

    return _format_chunks(ranked_chunks)


def _format_chunks(chunks: List[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"--- Chunk {i}: [{c.get('category','')}] {c.get('section_heading','')} ---\n"
            f"{c['text']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# BUILD / REBUILD EMBEDDINGS FOR A THREAD
# ---------------------------------------------------------------------------

def build_kb_embeddings(thread_id: str, kb_entries: List[dict]) -> int:
    """
    Chunks and embeds the KB for a thread, storing results in SQLite.

    Args:
        thread_id:  LangGraph thread id
        kb_entries: list of {"category":..., "topic":..., "knowledge_text":...}
    Returns:
        Number of chunks embedded.
    """
    if not kb_entries:
        return 0

    chunks = chunk_knowledge_base(kb_entries)
    texts  = [c["text"] for c in chunks]

    embed_fn = _get_embedding_model()
    try:
        embeddings = embed_fn(texts)
    except Exception as e:
        print(f"⚠️  RAG embedding failed: {e}. Chunks stored without embeddings — falling back to TF-IDF.")
        from agents.rag_kb import _get_embedding_model as _get  # re-import to force TF-IDF
        tfidf = _get()   # Will return TF-IDF since Azure call already failed
        embeddings = tfidf(texts)

    save_embeddings(thread_id, chunks, embeddings)

    print(f"✅ RAG: embedded {len(chunks)} chunks for thread {thread_id[:8]}…")
    return len(chunks)


def build_kb_embeddings_from_db(thread_id: str) -> int:
    """
    Convenience: loads domain_knowledge from SQLite for a thread and builds
    embeddings.  Used when resuming a saved session.
    """
    with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
        rows = conn.execute(
            "SELECT category, topic, knowledge_text FROM domain_knowledge WHERE thread_id = ?",
            (thread_id,)
        ).fetchall()

    if not rows:
        return 0

    kb_entries = [{"category": r[0], "topic": r[1], "knowledge_text": r[2]} for r in rows]
    return build_kb_embeddings(thread_id, kb_entries)
