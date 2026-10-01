import numpy as np

from transport_rag.models import Chunk, SearchHit
from transport_rag.retrieval.index import DenseIndex


class FakeEmbedder:
    def encode(self, texts):
        rows = []
        for text in texts:
            text = text.lower()
            rows.append([
                1.0 if "rerouting" in text else 0.0,
                1.0 if "congestion" in text else 0.0,
                1.0 if "objective" in text else 0.0,
            ])
        return np.asarray(rows, dtype=np.float32)


def test_dense_search_orders_by_similarity():
    embedder = FakeEmbedder()
    index = DenseIndex("fake", embedder=embedder)
    index.chunks = [
        Chunk("a", "a.md", 1, 1, "rerouting threshold"),
        Chunk("b", "b.md", 1, 1, "objective metric"),
    ]
    index.embeddings = DenseIndex._normalize(embedder.encode([c.text for c in index.chunks]))

    hits = index.search("rerouting", top_k=2)
    assert hits[0].chunk.chunk_id == "a"
    assert hits[0].score > hits[1].score


def test_bm25_finds_exact_identifier():
    embedder = FakeEmbedder()
    index = DenseIndex("fake", embedder=embedder)
    index.chunks = [
        Chunk("a", "config.java", 1, 1, "maxReroutesGrid controls rerouting"),
        Chunk("b", "readme.md", 1, 1, "general route simulation overview"),
    ]
    index.embeddings = DenseIndex._normalize(
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    )
    index._build_lexical_index()

    hits = index.bm25_search("maxReroutesGrid", top_k=2)
    assert hits[0].chunk.chunk_id == "a"


def test_hybrid_rrf_can_promote_lexical_match():
    class HybridFakeEmbedder:
        def encode(self, texts):
            rows = []
            for text in texts:
                # Semantic side deliberately prefers README-like wording.
                rows.append([1.0, 0.0] if "overview" in text.lower() or "rerouting" in text.lower() else [0.2, 1.0])
            return np.asarray(rows, dtype=np.float32)

    embedder = HybridFakeEmbedder()
    index = DenseIndex("fake", embedder=embedder)
    index.chunks = [
        Chunk("a", "readme.md", 1, 1, "rerouting overview and behavioral description"),
        Chunk("b", "config.java", 1, 1, "maxReroutesGrid property"),
    ]
    index.embeddings = DenseIndex._normalize(embedder.encode([c.text for c in index.chunks]))
    index._build_lexical_index()

    hits = index.hybrid_search("maxReroutesGrid", top_k=2, candidate_k=2)
    assert {h.chunk.chunk_id for h in hits} == {"a", "b"}
    assert hits[0].score > 0.0


def test_reranker_reorders_valid_candidate_ids(monkeypatch):
    from transport_rag.retrieval import rerank as rerank_module

    hits = [
        SearchHit(Chunk("a", "a.md", 1, 2, "general overview"), 0.9),
        SearchHit(Chunk("b", "b.java", 10, 20, "exact implementation"), 0.8),
        SearchHit(Chunk("c", "c.md", 1, 3, "other detail"), 0.7),
    ]

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": '{"ranking":["C2","C1","C3"]}'}}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(rerank_module.httpx, "Client", FakeClient)
    ranked = rerank_module.rerank_with_ollama(
        "Where is the implementation?", hits, "http://localhost:11434", "lfm2.5", top_k=3
    )
    assert [hit.chunk.chunk_id for hit in ranked] == ["b", "a", "c"]
    assert ranked[0].score == 0.8


def test_reranker_appends_omitted_candidates(monkeypatch):
    from transport_rag.retrieval import rerank as rerank_module

    hits = [
        SearchHit(Chunk("a", "a.md", 1, 1, "a"), 0.9),
        SearchHit(Chunk("b", "b.md", 1, 1, "b"), 0.8),
    ]

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": '{"ranking":["C2"]}'}}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(rerank_module.httpx, "Client", FakeClient)
    ranked = rerank_module.rerank_with_ollama("q", hits, "http://localhost:11434", "lfm2.5", top_k=2)
    assert [hit.chunk.chunk_id for hit in ranked] == ["b", "a"]
