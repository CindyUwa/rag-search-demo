"""
main.py : FastAPI application for RAG Enterprise Search.

Endpoints:
    POST /ingest     → upload a document (PDF or TXT)
    POST /search     → ask a question, get an answer
    GET  /documents  → list ingested documents
    GET  /health     → health check

This demo simulates the core of Sinequa's enterprise
search platform: ingest documents, search semantically,
generate grounded answers using AI.

Usage:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your_key
    uvicorn main:app --reload
    Docs: http://localhost:8000/docs
"""

import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import tempfile
import shutil

from ingestion import ingest_document
from retrieval import retrieve
from generator import generate_answer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
log = logging.getLogger("main")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Enterprise Search Demo",
    version="1.0.0",
    description="""
    Enterprise search demo powered by RAG (Retrieval Augmented Generation).
    Upload documents, ask questions in natural language,
    get grounded answers with source attribution.

    Mirrors the core architecture of enterprise search platforms
    like Sinequa: data ingestion, semantic retrieval, AI generation.
    """
)

# ── In-memory document store ──────────────────────────────────────────────────
# In production Sinequa: distributed search index (Elasticsearch-like)
_document_store: dict = {}


# ── Request models ────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    doc_ids: list[str] = []  # empty = search all documents


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check — verify API and Anthropic connection."""
    api_key_set = bool(os.getenv("ANTHROPIC_API_KEY"))
    return {
        "status": "healthy",
        "documents": len(_document_store),
        "anthropic_key": "configured" if api_key_set else "missing"
    }


@app.post("/ingest")
async def ingest_document_endpoint(file: UploadFile = File(...)):
    """
    Upload and ingest a document.
    Supported formats: PDF, TXT, MD

    The document is chunked into passages and stored
    in memory for semantic search.
    """
    allowed = [".pdf", ".txt", ".md"]
    suffix = "." + file.filename.split(".")[-1].lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {allowed}"
        )

    # Save to temp file for processing
    with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = ingest_document(tmp_path)
        doc_id = file.filename

        _document_store[doc_id] = {
            "filename": file.filename,
            "chunks": chunks,
            "chunk_count": len(chunks)
        }

        log.info("Ingested %s: %d chunks", file.filename, len(chunks))

        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "chunk_count": len(chunks),
            "status": "ingested"
        }

    finally:
        os.unlink(tmp_path)


@app.post("/search")
async def search(request: SearchRequest):
    """
    Search across ingested documents and generate an answer.

    Flow:
    1. Retrieve relevant passages from documents
    2. Pass passages + query to Claude
    3. Return grounded answer with source attribution

    This is the core RAG pipeline:
    Retrieval + Augmented Generation
    """
    if not _document_store:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Use POST /ingest first."
        )

    # Determine which documents to search
    if request.doc_ids:
        docs_to_search = {
            k: v for k, v in _document_store.items()
            if k in request.doc_ids
        }
    else:
        docs_to_search = _document_store

    # Collect all chunks from selected documents
    all_chunks = []
    for doc_id, doc in docs_to_search.items():
        for chunk in doc["chunks"]:
            all_chunks.append({**chunk, "doc_id": doc_id})

    # Retrieve most relevant passages
    passages = retrieve(request.query, all_chunks, top_k=request.top_k)

    log.info(
        "Query: '%s' | Documents: %d | Passages found: %d",
        request.query, len(docs_to_search), len(passages)
    )

    # Generate grounded answer
    result = generate_answer(request.query, passages)

    return {
        "query": request.query,
        "answer": result["answer"],
        "sources": result["sources"],
        "model": result["model"],
        "docs_searched": len(docs_to_search),
        "passages_used": len(passages)
    }


@app.get("/documents")
async def list_documents():
    """List all ingested documents."""
    return {
        "count": len(_document_store),
        "documents": [
            {
                "doc_id": doc_id,
                "filename": doc["filename"],
                "chunk_count": doc["chunk_count"]
            }
            for doc_id, doc in _document_store.items()
        ]
    }