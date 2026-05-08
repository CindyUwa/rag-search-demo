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

import re
import math
from collections import Counter


def preprocess(text: str) -> list[str]:
    """Lowercase and tokenize text."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()


def compute_tf(words: list[str]) -> dict:
    """Term frequency: how often each word appears."""
    count = Counter(words)
    total = len(words)
    return {word: freq / total for word, freq in count.items()}


def compute_idf(chunks: list[dict]) -> dict:
    """
    Inverse document frequency: rare words are more important.
    Words appearing in many chunks are less discriminative.
    """
    num_chunks = len(chunks)
    word_doc_count = Counter()

    for chunk in chunks:
        words = set(preprocess(chunk["text"]))
        for word in words:
            word_doc_count[word] += 1

    idf = {}
    for word, count in word_doc_count.items():
        idf[word] = math.log(num_chunks / (1 + count))

    return idf


def score_chunk(chunk_words: list[str], query_words: list[str],
                tf: dict, idf: dict) -> float:
    """
    TF-IDF score for a chunk given a query.
    Higher score = more relevant to the query.
    """
    score = 0.0
    for word in query_words:
        if word in tf and word in idf:
            score += tf[word] * idf[word]
    return score


def retrieve(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    Find the top_k most relevant chunks for a query.

    This is the core search operation:
    1. Preprocess query
    2. Score each chunk
    3. Return top matches with scores

    In Sinequa: neural embeddings replace TF-IDF,
    enabling semantic search across 100M+ documents.
    """
    query_words = preprocess(query)
    idf = compute_idf(chunks)

    scored_chunks = []
    for chunk in chunks:
        chunk_words = preprocess(chunk["text"])
        tf = compute_tf(chunk_words)
        score = score_chunk(chunk_words, query_words, tf, idf)
        scored_chunks.append({**chunk, "score": round(score, 4)})

    # Sort by score descending
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    # Return top_k with score > 0
    results = scored_chunks[:top_k]
    return results