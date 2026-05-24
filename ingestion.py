"""
ingestion.py : Document loading and chunking.

Takes a PDF or text file and splits it into chunks.
Each chunk is a passage that can be searched independently.

In entreprise search's world: this is the data ingestion pipeline
that indexes enterprise content (SharePoint, Confluence,
email, databases) into the search platform.
"""

import re
from pathlib import Path


def load_pdf(filepath: str) -> str:
    """
    Extract text from a PDF file.
    In production : handles 100+ file formats
    including Office, PDF, HTML, email.
    """
    try:
        import PyPDF2
        text = ""
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract PDF: {e}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Split text into overlapping chunks.

    Why overlap? So that answers spanning two chunks
    are not missed. Standard pattern in RAG pipelines.

    chunk_size : number of words per chunk
    overlap    : number of words shared between consecutive chunks

    In Sinequa's world: chunks are indexed as documents
    in the search engine with metadata (source, page, date).
    """
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "id":       chunk_id,
            "text":     chunk_text,
            "start":    start,
            "end":      end,
            "word_count": len(chunk_words)
        })

        chunk_id += 1
        # Move forward by chunk_size minus overlap
        start += chunk_size - overlap

    return chunks


def ingest_document(filepath: str) -> list[dict]:
    """
    Full ingestion pipeline for one document.
    Load → extract text → chunk → return passages.
    """
    path = Path(filepath)

    if path.suffix.lower() == ".pdf":
        text = load_pdf(filepath)
    elif path.suffix.lower() in [".txt", ".md"]:
        text = load_text_file(filepath)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    chunks = chunk_text(text)
    print(f"Ingested {path.name}: {len(chunks)} chunks")
    return chunks