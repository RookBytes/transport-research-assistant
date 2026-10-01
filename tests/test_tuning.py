from transport_rag.evaluation.tune_retrieval import (
    diversify_ranked_chunks,
    leaderboard_sort_key,
)


def test_diversification_round_robins_sources_before_second_chunk():
    ranked = [
        ("a1", 1.0),
        ("a2", 0.9),
        ("a3", 0.8),
        ("b1", 0.7),
        ("c1", 0.6),
        ("b2", 0.5),
    ]
    sources = {
        "a1": "a.md",
        "a2": "a.md",
        "a3": "a.md",
        "b1": "b.md",
        "b2": "b.md",
        "c1": "c.md",
    }
    result = diversify_ranked_chunks(ranked, sources, top_k=5, max_chunks_per_source=2)
    assert [chunk_id for chunk_id, _ in result] == ["a1", "b1", "c1", "a2", "b2"]


def test_no_diversification_preserves_fused_order():
    ranked = [("a1", 1.0), ("a2", 0.9), ("b1", 0.8)]
    sources = {"a1": "a.md", "a2": "a.md", "b1": "b.md"}
    result = diversify_ranked_chunks(ranked, sources, top_k=2, max_chunks_per_source=0)
    assert result == ranked[:2]


def test_leaderboard_prefers_eligible_before_higher_mrr_ineligible():
    eligible = {
        "recall_at_5": 0.95,
        "mrr": 0.75,
        "recall_at_1": 0.55,
        "recall_at_3": 0.90,
        "candidate_k": 15,
        "max_chunks_per_source": 2,
    }
    ineligible = {
        "recall_at_5": 0.90,
        "mrr": 0.90,
        "recall_at_1": 0.80,
        "recall_at_3": 0.90,
        "candidate_k": 15,
        "max_chunks_per_source": 2,
    }
    assert leaderboard_sort_key(eligible, 0.95) > leaderboard_sort_key(ineligible, 0.95)
