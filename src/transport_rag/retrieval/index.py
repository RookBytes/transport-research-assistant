from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import numpy as np

from transport_rag.models import Chunk, SearchHit


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def lexical_tokens(text: str) -> list[str]:
    """Tokenize prose and code identifiers without external NLP dependencies."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        lower = raw.lower()
        tokens.append(lower)

        # Add identifier pieces so maxReroutesGrid can also match
        # "max reroutes grid" while preserving the exact whole identifier.
        pieces = []
        for underscore_piece in raw.split("_"):
            pieces.extend(_CAMEL_RE.split(underscore_piece))
        for piece in pieces:
            piece = piece.lower()
            if piece and piece != lower:
                tokens.append(piece)
    return tokens


class OllamaEmbedder:
    def __init__(
        self,
        model_name: str,
        ollama_url: str = "http://localhost:11434",
        timeout: float = 300.0,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.model_name, "input": texts},
            )
            response.raise_for_status()
            data = response.json()

        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError(f"Ollama returned no embeddings for {self.model_name!r}.")
        return np.asarray(embeddings, dtype=np.float32)


class DenseIndex:
    def __init__(
        self,
        model_name: str,
        ollama_url: str = "http://localhost:11434",
        embedder=None,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.embedder = embedder or OllamaEmbedder(model_name, self.ollama_url)
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

        # Lexical/BM25 structures are built from persisted chunk text, so no
        # second index file or additional dependency is required.
        self._doc_tokens: list[list[str]] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_freq: dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._path_tokens: list[list[str]] = []
        self._path_term_freqs: list[Counter[str]] = []
        self._path_doc_freq: dict[str, int] = {}
        self._avg_path_len: float = 0.0

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norms, 1e-12, None)

    def _build_lexical_index(self) -> None:
        # Build two BM25 fields: chunk content and source path. Keeping the path
        # separate lets exact identifiers such as SweepConfig or SimulationEngine
        # receive a strong metadata match without distorting chunk-length norms.
        self._doc_tokens = [lexical_tokens(c.text) for c in self.chunks]
        self._path_tokens = [lexical_tokens(c.source_path) for c in self.chunks]
        self._term_freqs = [Counter(tokens) for tokens in self._doc_tokens]
        self._path_term_freqs = [Counter(tokens) for tokens in self._path_tokens]

        df: defaultdict[str, int] = defaultdict(int)
        for tf in self._term_freqs:
            for term in tf:
                df[term] += 1
        self._doc_freq = dict(df)

        path_df: defaultdict[str, int] = defaultdict(int)
        for tf in self._path_term_freqs:
            for term in tf:
                path_df[term] += 1
        self._path_doc_freq = dict(path_df)

        self._avg_doc_len = (
            sum(len(x) for x in self._doc_tokens) / len(self._doc_tokens)
            if self._doc_tokens else 0.0
        )
        self._avg_path_len = (
            sum(len(x) for x in self._path_tokens) / len(self._path_tokens)
            if self._path_tokens else 0.0
        )

    def build(self, chunks: list[Chunk], batch_size: int = 32) -> None:
        if not chunks:
            raise ValueError("Cannot build an index with zero chunks.")

        batches = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            print(
                f"Embedding {start + 1}-{start + len(batch)} / {len(chunks)} "
                f"with Ollama/{self.model_name}"
            )
            batches.append(self.embedder.encode([c.text for c in batch]))

        self.chunks = chunks
        self.embeddings = self._normalize(np.vstack(batches).astype(np.float32))
        self._build_lexical_index()

    def search(self, query: str, top_k: int = 6) -> list[SearchHit]:
        """Dense semantic retrieval (V1 baseline)."""
        if self.embeddings is None or not self.chunks:
            raise RuntimeError("Index is not loaded.")

        q = self._normalize(self.embedder.encode([query]).astype(np.float32))[0]
        scores = self.embeddings @ q
        top_k = max(1, min(top_k, len(self.chunks)))
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]

        return [
            SearchHit(chunk=self.chunks[int(i)], score=float(scores[int(i)]))
            for i in idx
        ]

    def bm25_search(
        self,
        query: str,
        top_k: int = 15,
        k1: float = 1.5,
        b: float = 0.75,
        path_weight: float = 1.0,
    ) -> list[SearchHit]:
        """Dependency-free fielded BM25 over chunk content and source paths."""
        if not self.chunks:
            raise RuntimeError("Index is not loaded.")
        if not self._term_freqs:
            self._build_lexical_index()

        query_terms = lexical_tokens(query)
        if not query_terms:
            return []

        n_docs = len(self.chunks)
        avg_len = max(self._avg_doc_len, 1e-9)
        avg_path_len = max(self._avg_path_len, 1e-9)
        scores = np.zeros(n_docs, dtype=np.float64)

        def add_field_scores(
            term: str,
            term_freqs: list[Counter[str]],
            doc_freq: dict[str, int],
            token_lists: list[list[str]],
            average_len: float,
            weight: float,
        ) -> None:
            df = doc_freq.get(term, 0)
            if df == 0 or weight <= 0.0:
                return
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(term_freqs):
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                field_len = len(token_lists[i])
                denom = freq + k1 * (1.0 - b + b * field_len / average_len)
                scores[i] += weight * idf * (freq * (k1 + 1.0)) / denom

        # Repeated query terms do not need to multiply the same evidence. Source
        # paths are a separate weighted field so filename/class-name matches matter.
        for term in set(query_terms):
            add_field_scores(
                term, self._term_freqs, self._doc_freq, self._doc_tokens, avg_len, 1.0
            )
            add_field_scores(
                term,
                self._path_term_freqs,
                self._path_doc_freq,
                self._path_tokens,
                avg_path_len,
                path_weight,
            )

        positive = np.flatnonzero(scores > 0.0)
        if positive.size == 0:
            return []
        positive = positive[np.argsort(-scores[positive])]
        positive = positive[: max(1, min(top_k, len(positive)))]

        # Normalize lexical scores to 0..1 for readable diagnostics. RRF uses
        # only the rank positions, not these magnitudes.
        max_score = float(scores[positive[0]])
        return [
            SearchHit(
                chunk=self.chunks[int(i)],
                score=float(scores[int(i)] / max_score) if max_score > 0 else 0.0,
            )
            for i in positive
        ]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 6,
        candidate_k: int = 10,
        rrf_k: int = 20,
        max_chunks_per_source: int = 1,
    ) -> list[SearchHit]:
        """Fuse dense and BM25 rankings with Reciprocal Rank Fusion (V2)."""
        candidate_k = max(candidate_k, top_k)
        dense_hits = self.search(query, top_k=min(candidate_k, len(self.chunks)))
        lexical_hits = self.bm25_search(query, top_k=candidate_k)

        by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        fused: defaultdict[str, float] = defaultdict(float)

        for rank, hit in enumerate(dense_hits, start=1):
            fused[hit.chunk.chunk_id] += 1.0 / (rrf_k + rank)
        for rank, hit in enumerate(lexical_hits, start=1):
            fused[hit.chunk.chunk_id] += 1.0 / (rrf_k + rank)

        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))

        # Source-aware diversification. First take the best chunk from each source
        # in fused-rank order, then the second-best from each source, and so on up
        # to max_chunks_per_source. This prevents duplicate chunks from one large
        # file from consuming the entire final top-K while still permitting a second
        # chunk from the same file when useful. A value <= 0 disables this step.
        if max_chunks_per_source > 0:
            diversified: list[tuple[str, float]] = []
            for source_slot in range(max_chunks_per_source):
                seen_this_round: set[str] = set()
                prior_counts: defaultdict[str, int] = defaultdict(int)
                for chosen_id, _ in diversified:
                    prior_counts[by_id[chosen_id].source_path] += 1

                for chunk_id, score in ranked:
                    source = by_id[chunk_id].source_path
                    if source in seen_this_round:
                        continue
                    if prior_counts[source] != source_slot:
                        continue
                    diversified.append((chunk_id, score))
                    seen_this_round.add(source)
                    if len(diversified) >= top_k:
                        break
                if len(diversified) >= top_k:
                    break
            ranked = diversified
        else:
            ranked = ranked[: max(1, min(top_k, len(ranked)))]

        # Maximum possible RRF score occurs when a chunk is rank 1 in both
        # lists. Normalize against that so scores remain roughly 0..1 and the
        # existing abstention plumbing can continue to operate.
        max_rrf = 2.0 / (rrf_k + 1.0)
        return [
            SearchHit(chunk=by_id[chunk_id], score=float(score / max_rrf))
            for chunk_id, score in ranked
        ]

    def save(self, directory: Path) -> None:
        if self.embeddings is None:
            raise RuntimeError("No embeddings to save.")
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        metadata = {
            "model_name": self.model_name,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "source_path": c.source_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "text": c.text,
                }
                for c in self.chunks
            ],
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        directory: Path,
        ollama_url: str = "http://localhost:11434",
        embedder=None,
    ) -> "DenseIndex":
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        obj = cls(metadata["model_name"], ollama_url=ollama_url, embedder=embedder)
        obj.chunks = [Chunk(**item) for item in metadata["chunks"]]
        obj.embeddings = np.load(directory / "embeddings.npy").astype(np.float32)
        obj._build_lexical_index()
        return obj
