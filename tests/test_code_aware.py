from transport_rag.models import Chunk, SearchHit
from transport_rag.retrieval.code_aware import CodeAwareRetriever, extract_chunk_symbols


class FakeBaseIndex:
    def __init__(self, chunks, rankings=None):
        self.chunks = chunks
        self.rankings = rankings or chunks

    def hybrid_search(self, query, top_k=6, candidate_k=10, rrf_k=20, max_chunks_per_source=1):
        chunks = list(self.rankings)
        if max_chunks_per_source > 0:
            out = []
            counts = {}
            for chunk in chunks:
                count = counts.get(chunk.source_path, 0)
                if count >= max_chunks_per_source:
                    continue
                counts[chunk.source_path] = count + 1
                out.append(chunk)
                if len(out) >= top_k:
                    break
            chunks = out
        else:
            chunks = chunks[:top_k]
        return [SearchHit(chunk, 1.0 - i * 0.05) for i, chunk in enumerate(chunks[:top_k])]


def test_extracts_java_type_method_and_field_symbols():
    chunk = Chunk(
        "c1",
        "simulation/src/main/java/porto/sweep/sim/SimulationEngine.java",
        100,
        130,
        """
public class SimulationEngine {
    private double memoryLearningRate;

    private void learnObservedRouteAsCandidate(TaxiState taxi) {
        return;
    }
}
""".strip(),
    )
    names = {symbol.name for symbol in extract_chunk_symbols(chunk)}
    assert "SimulationEngine" in names
    assert "memoryLearningRate" in names
    assert "learnObservedRouteAsCandidate" in names


def test_symbol_search_promotes_definition_chunk_for_exact_identifier():
    generic = Chunk("a", "simulation/README.md", 1, 20, "Rerouting behavior and route choice overview.")
    definition = Chunk(
        "b",
        "simulation/src/main/java/porto/sweep/config/SweepConfig.java",
        20,
        40,
        "public double slowdownTriggerRatio;\npublic double slowdownTriggerSec;",
    )
    retriever = CodeAwareRetriever(FakeBaseIndex([generic, definition], rankings=[generic, definition]))
    hits = retriever.symbol_search("Where is slowdownTriggerRatio configured?", top_k=2)
    assert hits
    assert hits[0].chunk.chunk_id == "b"
    assert "slowdownTriggerRatio" in retriever.matched_query_symbols("Where is slowdownTriggerRatio configured?")


def test_code_hybrid_uses_symbol_signal_to_promote_definition_file():
    generic = Chunk("a", "simulation/README.md", 1, 20, "slowdown and rerouting overview")
    definition = Chunk(
        "b",
        "simulation/src/main/java/porto/sweep/config/SweepConfig.java",
        20,
        40,
        "public double slowdownTriggerRatio;",
    )
    other = Chunk("c", "other.md", 1, 10, "unrelated")
    base = FakeBaseIndex([generic, definition, other], rankings=[generic, other, definition])
    retriever = CodeAwareRetriever(base, symbol_weight=1.0)
    result = retriever.search_with_diagnostics(
        "What does slowdownTriggerRatio control?", top_k=2, candidate_k=3, rrf_k=20
    )
    assert result["matched_symbols"]
    assert result["hits"][0].chunk.source_path == definition.source_path


def test_dependency_expansion_can_cross_files_to_definition_chunk():
    caller = Chunk(
        "caller",
        "simulation/src/main/java/porto/sweep/app/RunSweepMain.java",
        100,
        120,
        "SweepConfig config = SweepConfig.load(path);\nrun(config);",
    )
    definition = Chunk(
        "definition",
        "simulation/src/main/java/porto/sweep/config/SweepConfig.java",
        1,
        30,
        "public class SweepConfig { }",
    )
    unrelated = Chunk("u", "README.md", 1, 5, "overview")
    base = FakeBaseIndex([caller, definition, unrelated], rankings=[caller, unrelated, definition])
    retriever = CodeAwareRetriever(base)
    deps = retriever.dependency_hits(
        "How does the runner use configuration?",
        [SearchHit(caller, 1.0)],
        max_hits=3,
    )
    assert any(hit.chunk.chunk_id == "definition" for hit in deps)


def test_plain_language_query_without_symbol_match_preserves_base_ranking():
    first = Chunk("a", "README.md", 1, 10, "overview of the experiment")
    second = Chunk("b", "notes.md", 1, 10, "other notes")
    base = FakeBaseIndex([first, second], rankings=[first, second])
    retriever = CodeAwareRetriever(base)
    result = retriever.search_with_diagnostics(
        "What is the overall purpose of the project?", top_k=2, candidate_k=2, rrf_k=20
    )
    assert result["matched_symbols"] == []
    assert [hit.chunk.chunk_id for hit in result["hits"]] == ["a", "b"]
