from __future__ import annotations

from transport_rag.config import Settings
from transport_rag.generation.ollama import generate_with_ollama
from transport_rag.retrieval.index import DenseIndex
from transport_rag.retrieval.rerank import rerank_with_ollama


ABSTAIN = "I don't have enough evidence in the indexed sources to answer that reliably."


class RAGService:
    def __init__(self, settings: Settings, retrieval_mode: str | None = None):
        self.settings = settings
        self.retrieval_mode = (retrieval_mode or settings.retrieval_mode).strip().lower()
        if self.retrieval_mode not in {"dense", "hybrid", "hybrid_rerank"}:
            raise ValueError("retrieval_mode must be 'dense', 'hybrid', or 'hybrid_rerank'")
        self.index = DenseIndex.load(settings.index_dir, ollama_url=settings.ollama_url)

    def _hybrid_rerank_once(self, question: str, k: int):
        candidate_k = max(self.settings.rerank_candidate_k, k)
        hybrid_hits = self.index.hybrid_search(
            question,
            top_k=candidate_k,
            candidate_k=max(self.settings.hybrid_candidate_k, candidate_k),
            rrf_k=self.settings.rrf_k,
        )
        reranked_hits = rerank_with_ollama(
            question=question,
            hits=hybrid_hits,
            ollama_url=self.settings.ollama_url,
            model=self.settings.rerank_model,
            top_k=k,
            max_chars_per_chunk=self.settings.rerank_max_chars_per_chunk,
        )
        return reranked_hits, hybrid_hits

    def retrieve(self, question: str, top_k: int | None = None):
        k = top_k or self.settings.top_k
        if self.retrieval_mode == "hybrid_rerank":
            reranked_hits, _ = self._hybrid_rerank_once(question, k)
            return reranked_hits
        if self.retrieval_mode == "hybrid":
            return self.index.hybrid_search(
                question,
                top_k=k,
                candidate_k=self.settings.hybrid_candidate_k,
                rrf_k=self.settings.rrf_k,
            )
        return self.index.search(question, top_k=k)

    def retrieve_with_diagnostics(self, question: str, top_k: int | None = None) -> dict:
        """Retrieve once and expose pre/post-rerank ordering for evaluation.

        For V3 this avoids calling the reranker twice just to inspect its effect.
        For other retrieval modes, pre and post lists are identical.
        """
        k = top_k or self.settings.top_k
        if self.retrieval_mode == "hybrid_rerank":
            reranked_hits, hybrid_candidates = self._hybrid_rerank_once(question, k)
            return {
                "hits": reranked_hits,
                "pre_rerank_hits": hybrid_candidates[:k],
                "rerank_candidate_hits": hybrid_candidates,
            }

        hits = self.retrieve(question, top_k=k)
        return {
            "hits": hits,
            "pre_rerank_hits": hits,
            "rerank_candidate_hits": hits,
        }

    def _expand_context_hits(self, question: str, primary_hits, max_context_hits: int = 10):
        """Add supporting chunks from source files already selected by retrieval.

        The primary top-K stays source-diversified for retrieval evaluation. For
        answer generation, however, one chunk per file can omit nearby or related
        implementation details. Re-run the same hybrid ranking without the
        per-source cap and append only extra chunks from files that were already
        present in the primary retrieval set.
        """
        if self.retrieval_mode != "hybrid" or not primary_hits:
            return primary_hits

        selected_sources = {hit.chunk.source_path for hit in primary_hits}
        selected_ids = {hit.chunk.chunk_id for hit in primary_hits}
        expanded = list(primary_hits)

        raw_top_k = max(max_context_hits + 2, self.settings.top_k * 2)
        raw_hits = self.index.hybrid_search(
            question,
            top_k=raw_top_k,
            candidate_k=max(self.settings.hybrid_candidate_k, raw_top_k),
            rrf_k=self.settings.rrf_k,
            max_chunks_per_source=0,
        )

        for hit in raw_hits:
            if len(expanded) >= max_context_hits:
                break
            if hit.chunk.chunk_id in selected_ids:
                continue
            if hit.chunk.source_path not in selected_sources:
                continue
            expanded.append(hit)
            selected_ids.add(hit.chunk.chunk_id)

        return expanded

    @staticmethod
    def _format_context(hits) -> str:
        blocks = []
        for i, hit in enumerate(hits, start=1):
            c = hit.chunk
            blocks.append(f"[S{i}] {c.source_path}:{c.start_line}-{c.end_line}\n{c.text}")
        return "\n\n".join(blocks)

    def answer(self, question: str) -> dict:
        primary_hits = self.retrieve(question)
        # Reranking changes order, but the original retrieval scores are kept.
        # Use the strongest candidate score so the confidence threshold means
        # the same thing for dense, hybrid, and reranked retrieval.
        retrieval_confidence = max((hit.score for hit in primary_hits), default=0.0)
        if not primary_hits or retrieval_confidence < self.settings.min_score:
            return {"answer": ABSTAIN, "abstained": True, "sources": []}

        hits = self._expand_context_hits(question, primary_hits)
        context = self._format_context(hits)
        answer = generate_with_ollama(
            ollama_url=self.settings.ollama_url,
            model=self.settings.ollama_model,
            question=question,
            context=context,
        )
        sources = []
        for i, hit in enumerate(hits, start=1):
            c = hit.chunk
            sources.append(
                {
                    "id": f"S{i}",
                    "score": hit.score,
                    "source_path": c.source_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "text": c.text,
                }
            )
        return {
            "answer": answer,
            "abstained": answer.strip() == ABSTAIN,
            "sources": sources,
        }
