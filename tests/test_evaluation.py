from transport_rag.evaluation.run_eval import (
    citation_validity,
    compare_rank,
    recall_at,
    reciprocal_rank,
    source_rank,
)


def test_source_rank_and_retrieval_metrics():
    paths = ["a.md", "correct.java", "c.py"]
    rank = source_rank(paths, ["correct.java"])
    assert rank == 2
    assert recall_at(rank, 1) == 0.0
    assert recall_at(rank, 3) == 1.0
    assert reciprocal_rank(rank) == 0.5


def test_missing_gold_source_scores_zero():
    rank = source_rank(["a.md", "b.md"], ["missing.md"])
    assert rank is None
    assert recall_at(rank, 5) == 0.0
    assert reciprocal_rank(rank) == 0.0


def test_citation_validity():
    valid, total, score = citation_validity("Answer [S1] and [S3] plus [S9].", 3)
    assert valid == 2
    assert total == 3
    assert score == 2 / 3


def test_compare_rank():
    assert compare_rank(3, 1) == "improved"
    assert compare_rank(1, 3) == "worsened"
    assert compare_rank(2, 2) == "unchanged"
    assert compare_rank(None, 4) == "improved"
    assert compare_rank(4, None) == "worsened"
    assert compare_rank(None, None) == "unchanged"
