from transport_rag.evaluation.run_eval import all_gold_sources_at, source_coverage_at


def test_source_coverage_handles_multihop_gold_sources():
    retrieved = ["a.java", "x.md", "b.java", "c.py"]
    gold = ["a.java", "b.java"]
    assert source_coverage_at(retrieved, gold, 1) == 0.5
    assert source_coverage_at(retrieved, gold, 3) == 1.0
    assert all_gold_sources_at(retrieved, gold, 2) == 0.0
    assert all_gold_sources_at(retrieved, gold, 3) == 1.0
