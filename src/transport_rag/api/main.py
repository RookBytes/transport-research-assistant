from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from transport_rag.config import settings
from transport_rag.rag import RAGService


app = FastAPI(
    title="Transportation Research Assistant",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    return RAGService(settings)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return get_service().answer(request.question)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="RAG index not found. Run the ingestion command first.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
