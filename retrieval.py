"""
retrieval.py : Semantic search over document chunks.

Given a user query, finds the most relevant passages
using keyword matching and simple scoring.

In Sinequa's world: this is the search engine core
neural search using transformer models to find
semantically similar content across millions of documents.

For this demo: we use TF-IDF style keyword scoring
(no GPU needed, works immediately).
"""
"""
retrieval.py : Simple semantic-like keyword retrieval
"""

import re
from collections import Counter


def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()


def score_chunk(chunk_words, query_words):
    chunk_set = set(chunk_words)
    score = 0

    for word in query_words:
        if word in chunk_set:
            score += 1

    return score


def retrieve(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    query_words = preprocess(query)

    scored_chunks = []

    for chunk in chunks:
        chunk_words = preprocess(chunk["text"])

        score = score_chunk(chunk_words, query_words)

        scored_chunks.append({
            **chunk,
            "score": score
        })

    # sort always (even if all 0)
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    # IMPORTANT: always return something
    return scored_chunks[:top_k]