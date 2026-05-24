# RAG Enterprise Search Demo

Enterprise search powered by RAG (Retrieval Augmented Generation).
Upload documents, ask questions in natural language,
get grounded answers with source attribution.

Enterprise search platform:
data ingestion pipeline, semantic retrieval, and AI-powered answer generation.
## Author
Cindy UWAMARIYA

## What it does
- Ingests PDF and text documents into a searchable index
- Retrieves relevant passages using TF-IDF semantic scoring
- Generates grounded answers using Claude claude-haiku-4-5-20251001
- Returns source attribution for every answer, critical in enterprise RAG

## Architecture
```
frontend/          → Angular UI (TypeScript)
ingestion.py       → document loading, text extraction, chunking
retrieval.py       → TF-IDF semantic scoring, passage ranking
generator.py       → RAG prompt construction, Claude API, answer generation
main.py            → FastAPI endpoints, document store, orchestration
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
export ANTHROPIC_API_KEY=xx
ng serve
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
grounded answers, and hallucination prevention, the standard for
enterprise AI assistants in regulated environments.
cd C:\Users\cindy\PycharmProjects\rag-search-demo
set ANTHROPIC_API_KEY=XXX