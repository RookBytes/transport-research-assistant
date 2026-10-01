from __future__ import annotations

import httpx


SYSTEM_PROMPT = """You are a transportation research assistant.

Rules:
1. Answer only from the supplied SOURCES.
2. Every substantive factual claim must cite one or more source IDs such as [S1].
3. Never invent a source, file, line number, experiment result, parameter, or implementation detail.
4. If the sources do not contain enough evidence, say exactly:
   "I don't have enough evidence in the indexed sources to answer that reliably."
5. Distinguish measured results from interpretation.
6. Prefer concise technical explanations.
"""


def generate_with_ollama(
    ollama_url: str,
    model: str,
    question: str,
    context: str,
    timeout: float = 120.0,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"SOURCES:\n{context}\n\nQUESTION:\n{question}",
            },
        ],
        "options": {"temperature": 0.1},
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{ollama_url.rstrip('/')}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
    return data["message"]["content"].strip()
