from __future__ import annotations

import json

import httpx

from transport_rag.models import SearchHit


RERANK_SYSTEM_PROMPT = """You are a strict retrieval reranker.
Given a QUESTION and numbered candidate source chunks, rank candidates by how directly and completely they can answer the question.
Prefer primary implementation/configuration evidence over broad summaries when both are relevant.
Do not answer the question.
Return JSON only in this exact shape: {\"ranking\":[\"C3\",\"C1\",\"C2\"]}.
Include every candidate ID exactly once, ordered best to worst.
"""


def rerank_with_ollama(
    question: str,
    hits: list[SearchHit],
    ollama_url: str,
    model: str,
    top_k: int,
    max_chars_per_chunk: int = 1400,
    timeout: float = 180.0,
) -> list[SearchHit]:
    """Listwise rerank hybrid candidates with the already-local Ollama LLM.

    The original retrieval scores are preserved for diagnostics and abstention.
    If the reranker returns malformed output, candidate order falls back safely
    to the original hybrid ranking.
    """
    if not hits:
        return []

    top_k = max(1, min(top_k, len(hits)))
    candidate_ids = [f"C{i}" for i in range(1, len(hits) + 1)]
    by_id = dict(zip(candidate_ids, hits))

    blocks = []
    for candidate_id, hit in zip(candidate_ids, hits):
        chunk = hit.chunk
        text = chunk.text[:max_chars_per_chunk]
        blocks.append(
            f"[{candidate_id}] {chunk.source_path}:{chunk.start_line}-{chunk.end_line}\n{text}"
        )

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"QUESTION:\n{question}\n\nCANDIDATES:\n\n" + "\n\n".join(blocks),
            },
        ],
        "options": {"temperature": 0.0},
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{ollama_url.rstrip('/')}/api/chat", json=payload)
            response.raise_for_status()
            raw = response.json()["message"]["content"]
        parsed = json.loads(raw)
        ranking = parsed.get("ranking", [])
    except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError):
        return hits[:top_k]

    if not isinstance(ranking, list):
        return hits[:top_k]

    ordered: list[SearchHit] = []
    seen: set[str] = set()
    for candidate_id in ranking:
        if candidate_id in by_id and candidate_id not in seen:
            ordered.append(by_id[candidate_id])
            seen.add(candidate_id)

    # Robustly append omitted candidates in original hybrid order.
    for candidate_id in candidate_ids:
        if candidate_id not in seen:
            ordered.append(by_id[candidate_id])

    return ordered[:top_k]
