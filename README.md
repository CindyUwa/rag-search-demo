# RAG Enterprise Search Demo

Full diclosure : 
This is a PERSONAL project that cannot be link to Sinequa, but is only inspired by it. 

Enterprise search powered by RAG (Retrieval Augmented Generation).
Upload documents, ask questions in natural language,
get grounded answers with source attribution.

Mirrors the core architecture of Sinequa's enterprise search platform:
data ingestion pipeline, semantic retrieval, and AI-powered answer generation.
## Author
Cindy UWAMARIYA

## What it does
- Ingests PDF and text documents into a searchable index
- Retrieves relevant passages using TF-IDF semantic scoring
- Generates grounded answers using Claude claude-haiku-4-5-20251001
- Returns source attribution for every answer — critical in enterprise RAG

## Architecture
```
ingestion.py  →  document loading, text extraction, chunking
retrieval.py  →  semantic search, passage ranking (TF-IDF)
generator.py  →  RAG prompt construction, Claude API, answer generation
main.py       →  FastAPI endpoints, document store, orchestration
```

## RAG Pipeline
```
User query
    ↓
retrieval.py  →  find top-K relevant passages
    ↓
generator.py  →  build prompt with context + query
    ↓
Claude API    →  generate grounded answer
    ↓
Response      →  answer + source passages
```

## Run
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
uvicorn main:app --reload
```
Docs → http://localhost:8000/docs

## Test
1. GET /health — verify setup
2. POST /ingest — upload any PDF or TXT
3. POST /search — ask a question about your document
4. GET /documents — see all ingested documents

## Context
Built to understand the core technology behind enterprise AI search platforms.
RAG architecture: retrieval-augmented generation with source attribution,
grounded answers, and hallucination prevention — the standard for
enterprise AI assistants in regulated environments.