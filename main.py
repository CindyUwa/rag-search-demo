"""
main.py : FastAPI application for RAG Enterprise Search.

Endpoints:
    POST /ingest     → upload a document (PDF or TXT)
    POST /search     → ask a question, get an answer
    GET  /documents  → list ingested documents
    GET  /health     → health check

Usage:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your_key
    uvicorn main:app --reload

Docs:
    http://localhost:8000/docs
"""

import os
import logging
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingestion import ingest_document
from retrieval import retrieve
from generator import generate_answer


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

log = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Enterprise Search Demo",
    version="1.0.0",
    description="""
    Enterprise AI Search platform demo using:
    - Retrieval Augmented Generation (RAG)
    - FastAPI
    - Semantic retrieval
    - Grounded AI answers
    """
)

# CORS FIX FOR ANGULAR FRONTEND
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# In-memory document store
# ─────────────────────────────────────────────────────────────

_document_store = {}


# ─────────────────────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    doc_ids: list[str] = []


# ─────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():

    api_key_set = bool(os.getenv("ANTHROPIC_API_KEY"))

    return {
        "status": "healthy",
        "documents": len(_document_store),
        "anthropic_key": "configured" if api_key_set else "missing"
    }


# ─────────────────────────────────────────────────────────────
# Ingest Document
# ─────────────────────────────────────────────────────────────

@app.post("/ingest")
async def ingest_document_endpoint(
    file: UploadFile = File(...)
):

    allowed = [".pdf", ".txt", ".md"]

    suffix = "." + file.filename.split(".")[-1].lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {allowed}"
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
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

        log.info(
            "Ingested %s with %d chunks",
            file.filename,
            len(chunks)
        )

        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": file.filename,
            "chunk_count": len(chunks)
        }

    finally:

        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────
# Search Endpoint
# ─────────────────────────────────────────────────────────────

@app.post("/search")
async def search(request: SearchRequest):

    if not _document_store:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet"
        )

    # Determine target documents
    if request.doc_ids:

        docs_to_search = {
            k: v
            for k, v in _document_store.items()
            if k in request.doc_ids
        }

    else:

        docs_to_search = _document_store

    # Flatten chunks
    all_chunks = []

    for doc_id, doc in docs_to_search.items():

        for chunk in doc["chunks"]:

            all_chunks.append({
                **chunk,
                "doc_id": doc_id
            })

    # Retrieve relevant passages
    passages = retrieve(
        request.query,
        all_chunks,
        top_k=request.top_k
    )

    log.info(
        "Query='%s' | docs=%d | passages=%d",
        request.query,
        len(docs_to_search),
        len(passages)
    )

    # Generate answer
    result = generate_answer(
        request.query,
        passages
    )

    return {
        "query": request.query,
        "answer": result["answer"],
        "sources": result["sources"],
        "model": result["model"],
        "documents_searched": len(docs_to_search),
        "passages_used": len(passages)
    }


# ─────────────────────────────────────────────────────────────
# List Documents
# ─────────────────────────────────────────────────────────────

@app.get("/documents")
async def list_documents():

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