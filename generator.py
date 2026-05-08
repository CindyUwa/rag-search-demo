"""
generator.py : Answer generation using Anthropic Claude API.

Takes retrieved passages + user query and generates
a grounded, accurate answer.

This is the "Generative AI" layer in a RAG pipeline:
Retrieval Augmented Generation.

RAG = Retrieval + Augmented + Generation
  Retrieval   : find relevant passages (retrieval.py)
  Augmented   : add those passages as context
  Generation  : LLM generates answer grounded in context

In Sinequa's world: this is the AI Assistant layer
that sits on top of the search engine and answers
questions using retrieved enterprise content.
"""

import anthropic


def build_prompt(query: str, passages: list[dict]) -> str:
    """
    Build the prompt for Claude.

    We inject the retrieved passages as context.
    Claude answers based ONLY on this context.
    This prevents hallucination — a key requirement
    in enterprise environments (finance, legal, pharma).
    """
    context = "\n\n".join([
        f"[Passage {i+1}]\n{p['text']}"
        for i, p in enumerate(passages)
    ])

    return f"""You are an enterprise search assistant.
Answer the question using ONLY the passages provided below.
If the answer is not in the passages, say "I could not find 
this information in the provided documents."
Be concise and precise.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


def generate_answer(query: str, passages: list[dict]) -> dict:
    """
    Generate an answer using Claude claude-haiku-4-5-20251001.

    Returns the answer text and the passages used as sources.
    Source attribution is critical in enterprise RAG —
    users need to know WHERE the answer came from.
    """
    if not passages:
        return {
            "answer": "No relevant passages found for this query.",
            "sources": [],
            "model": "claude-haiku-4-5-20251001"
        }

    client = anthropic.Anthropic()
    prompt = build_prompt(query, passages)

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = message.content[0].text.strip()

    return {
        "answer":  answer,
        "sources": [
            {
                "chunk_id": p["id"],
                "score":    p["score"],
                "preview":  p["text"][:150] + "..."
            }
            for p in passages
        ],
        "model": "claude-haiku-4-5-20251001"
    }