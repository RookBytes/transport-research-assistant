from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath

from transport_rag.models import Chunk, SearchHit
from transport_rag.retrieval.index import lexical_tokens


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_JAVA_TYPE_RE = re.compile(r"\b(class|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)")
_JAVA_METHOD_RE = re.compile(
    r"(?ms)^\s*(?:(?:public|protected|private|static|final|abstract|synchronized|native|default|strictfp)\s+)*"
    r"(?:<[^>{}]+>\s*)?(?:[A-Za-z_$][\w$<>\[\],.? ]*\s+)?"
    r"([A-Za-z_$][\w$]*)\s*\([^;{}]*?\)\s*(?:throws\s+[^{}]+)?\{"
)
_JAVA_FIELD_RE = re.compile(
    r"(?m)^\s*(?:(?:public|protected|private|static|final|volatile|transient)\s+)+"
    r"[A-Za-z_$][\w$<>\[\],.? ]*\s+([A-Za-z_$][\w$]*)\s*(?:=[^;]*)?;"
)
_PY_DEF_RE = re.compile(r"(?m)^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_PY_CLASS_RE = re.compile(r"(?m)^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b")
_MD_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*$")
_PROPERTY_RE = re.compile(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*=")

_MIN_SYMBOL_SCORE = 2.5

_CONTROL_WORDS = {
    "if", "for", "while", "switch", "catch", "return", "new", "throw", "throws",
    "try", "else", "case", "default", "do", "this", "super", "true", "false", "null",
    "get", "set", "run", "main", "size", "add", "put", "list", "map", "string", "double",
    "int", "long", "boolean", "void", "object", "value", "values", "result", "results",
}


@dataclass(frozen=True)
class CodeSymbol:
    name: str
    kind: str
    chunk_id: str
    source_path: str
    start_line: int
    end_line: int


def _is_informative_symbol(name: str) -> bool:
    lowered = name.lower()
    if lowered in _CONTROL_WORDS or len(name) < 4:
        return False
    return (
        "_" in name
        or "." in name
        or any(ch.isupper() for ch in name[1:])
        or len(name) >= 8
    )


def _dedupe_symbols(symbols: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for kind, name in symbols:
        key = (kind, name)
        if key in seen or not _is_informative_symbol(name):
            continue
        seen.add(key)
        out.append(key)
    return out


def extract_chunk_symbols(chunk: Chunk) -> list[CodeSymbol]:
    """Extract lightweight code/document symbols from one persisted text chunk.

    This deliberately avoids tree-sitter so V4 remains fully local and does not
    add another runtime/model dependency. The parser is heuristic; it is used as
    an additional retrieval signal rather than as a compiler-grade AST.
    """
    suffix = PurePosixPath(chunk.source_path).suffix.lower()
    text = chunk.text
    found: list[tuple[str, str]] = []

    if suffix == ".java":
        found.extend((match.group(1), match.group(2)) for match in _JAVA_TYPE_RE.finditer(text))
        found.extend(("method", match.group(1)) for match in _JAVA_METHOD_RE.finditer(text))
        found.extend(("field", match.group(1)) for match in _JAVA_FIELD_RE.finditer(text))
    elif suffix == ".py":
        found.extend(("class", match.group(1)) for match in _PY_CLASS_RE.finditer(text))
        found.extend(("function", match.group(1)) for match in _PY_DEF_RE.finditer(text))
    elif suffix in {".properties", ".toml"}:
        found.extend(("property", match.group(1)) for match in _PROPERTY_RE.finditer(text))
    elif suffix == ".md":
        for match in _MD_HEADING_RE.finditer(text):
            heading = re.sub(r"[`*_#]", "", match.group(1)).strip()
            # Headings are useful structural anchors but only keep compact ones.
            if 1 <= len(heading.split()) <= 8:
                found.append(("heading", heading))

    return [
        CodeSymbol(
            name=name,
            kind=kind,
            chunk_id=chunk.chunk_id,
            source_path=chunk.source_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        )
        for kind, name in _dedupe_symbols(found)
    ]


def _query_identifiers(query: str) -> list[str]:
    return list(dict.fromkeys(_IDENTIFIER_RE.findall(query)))


def _symbol_match_score(symbol: CodeSymbol, query: str, query_tokens: set[str], identifiers: set[str]) -> float:
    name = symbol.name
    lower = name.lower()
    score = 0.0

    if lower in identifiers:
        score += 8.0
    if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", query, flags=re.IGNORECASE):
        score += 4.0

    pieces = set(lexical_tokens(name))
    overlap = pieces & query_tokens
    if pieces and overlap:
        score += 2.0 * len(overlap) / len(pieces)
        if pieces <= query_tokens:
            score += 1.0

    # Code definitions are stronger anchors than prose headings.
    if score > 0 and symbol.kind != "heading":
        score += 0.5
    return score


class CodeAwareRetriever:
    """Structural retrieval layered on top of the existing dense+BM25 index."""

    def __init__(self, base_index, symbol_weight: float = 0.75):
        self.base_index = base_index
        self.symbol_weight = symbol_weight
        self.by_id: dict[str, Chunk] = {chunk.chunk_id: chunk for chunk in base_index.chunks}
        self.symbols: list[CodeSymbol] = []
        self.symbols_by_name: defaultdict[str, list[CodeSymbol]] = defaultdict(list)
        self.symbols_by_chunk: defaultdict[str, list[CodeSymbol]] = defaultdict(list)

        for chunk in base_index.chunks:
            for symbol in extract_chunk_symbols(chunk):
                self.symbols.append(symbol)
                self.symbols_by_name[symbol.name.lower()].append(symbol)
                self.symbols_by_chunk[chunk.chunk_id].append(symbol)

    def matched_query_symbols(self, query: str) -> list[str]:
        query_tokens = set(lexical_tokens(query))
        identifiers = {value.lower() for value in _query_identifiers(query)}
        scored: dict[str, float] = {}
        for symbol in self.symbols:
            score = _symbol_match_score(symbol, query, query_tokens, identifiers)
            if score >= _MIN_SYMBOL_SCORE:
                scored[symbol.name] = max(scored.get(symbol.name, 0.0), score)
        return [name for name, _ in sorted(scored.items(), key=lambda item: (-item[1], item[0].lower()))]

    def symbol_search(self, query: str, top_k: int = 12) -> list[SearchHit]:
        query_tokens = set(lexical_tokens(query))
        identifiers = {value.lower() for value in _query_identifiers(query)}
        chunk_scores: defaultdict[str, float] = defaultdict(float)

        for symbol in self.symbols:
            score = _symbol_match_score(symbol, query, query_tokens, identifiers)
            if score < _MIN_SYMBOL_SCORE:
                continue
            # Multiple relevant symbols in one chunk should reinforce it, but the
            # max component dominates so giant declaration chunks do not win only
            # because they contain many names.
            previous = chunk_scores[symbol.chunk_id]
            chunk_scores[symbol.chunk_id] = max(previous, score) + min(previous, score) * 0.15

        if not chunk_scores:
            return []

        ranked = sorted(chunk_scores.items(), key=lambda item: (-item[1], item[0]))[: max(1, top_k)]
        max_score = ranked[0][1]
        return [
            SearchHit(self.by_id[chunk_id], score / max_score if max_score > 0 else 0.0)
            for chunk_id, score in ranked
        ]

    @staticmethod
    def _diversify(ranked: list[tuple[str, float]], by_id: dict[str, Chunk], top_k: int) -> list[tuple[str, float]]:
        selected: list[tuple[str, float]] = []
        seen_sources: set[str] = set()
        for chunk_id, score in ranked:
            source = by_id[chunk_id].source_path
            if source in seen_sources:
                continue
            selected.append((chunk_id, score))
            seen_sources.add(source)
            if len(selected) >= top_k:
                break
        return selected

    def search_with_diagnostics(
        self,
        query: str,
        *,
        top_k: int = 6,
        candidate_k: int = 10,
        rrf_k: int = 20,
    ) -> dict:
        matched_symbols = self.matched_query_symbols(query)
        symbol_hits = self.symbol_search(query, top_k=max(candidate_k, top_k))

        # If no code/document symbol is implicated, preserve the frozen V2.3
        # retriever exactly rather than perturbing ordinary natural-language QA.
        if not symbol_hits:
            base_hits = self.base_index.hybrid_search(
                query,
                top_k=top_k,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
            )
            return {
                "hits": base_hits,
                "base_hits": base_hits,
                "symbol_hits": [],
                "matched_symbols": [],
            }

        raw_k = max(candidate_k, top_k)
        base_hits = self.base_index.hybrid_search(
            query,
            top_k=raw_k,
            candidate_k=raw_k,
            rrf_k=rrf_k,
            max_chunks_per_source=0,
        )

        fused: defaultdict[str, float] = defaultdict(float)
        for rank, hit in enumerate(base_hits, start=1):
            fused[hit.chunk.chunk_id] += 1.0 / (rrf_k + rank)
        for rank, hit in enumerate(symbol_hits, start=1):
            fused[hit.chunk.chunk_id] += self.symbol_weight / (rrf_k + rank)

        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        ranked = self._diversify(ranked, self.by_id, top_k)
        max_possible = (1.0 + self.symbol_weight) / (rrf_k + 1.0)
        hits = [
            SearchHit(self.by_id[chunk_id], min(1.0, score / max_possible))
            for chunk_id, score in ranked
        ]
        return {
            "hits": hits,
            "base_hits": base_hits,
            "symbol_hits": symbol_hits,
            "matched_symbols": matched_symbols,
        }

    def search(self, query: str, **kwargs) -> list[SearchHit]:
        return self.search_with_diagnostics(query, **kwargs)["hits"]

    def dependency_hits(self, query: str, seed_hits: list[SearchHit], max_hits: int = 4) -> list[SearchHit]:
        """Find definition chunks connected to query/seed identifiers.

        This is a lightweight code graph: nodes are persisted chunks containing
        definitions and edges arise when one chunk mentions a symbol defined in
        another chunk. It is intentionally conservative and only expands to known
        definitions, never arbitrary lexical neighbors.
        """
        selected_ids = {hit.chunk.chunk_id for hit in seed_hits}
        selected_sources = {hit.chunk.source_path for hit in seed_hits}
        query_identifiers = {value.lower() for value in _query_identifiers(query)}
        scores: defaultdict[str, float] = defaultdict(float)

        # Query-named definitions are the strongest structural dependencies.
        for name in query_identifiers:
            for symbol in self.symbols_by_name.get(name, []):
                if symbol.chunk_id not in selected_ids:
                    scores[symbol.chunk_id] += 5.0

        # Then follow identifier references from selected chunks to definitions.
        for seed_rank, hit in enumerate(seed_hits, start=1):
            mentions = {value.lower() for value in _query_identifiers(hit.chunk.text)}
            for name in mentions:
                if name not in self.symbols_by_name:
                    continue
                if not _is_informative_symbol(name):
                    continue
                for symbol in self.symbols_by_name[name]:
                    if symbol.chunk_id in selected_ids:
                        continue
                    weight = 1.0 / seed_rank
                    if symbol.source_path not in selected_sources:
                        weight *= 1.5
                    scores[symbol.chunk_id] += weight

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:max_hits]
        max_score = ranked[0][1]
        return [
            SearchHit(self.by_id[chunk_id], score / max_score if max_score > 0 else 0.0)
            for chunk_id, score in ranked
        ]

    def expand_context(
        self,
        query: str,
        primary_hits: list[SearchHit],
        *,
        candidate_k: int = 10,
        rrf_k: int = 20,
        max_context_hits: int = 12,
    ) -> list[SearchHit]:
        if not primary_hits:
            return primary_hits

        expanded = list(primary_hits)
        selected_ids = {hit.chunk.chunk_id for hit in expanded}
        selected_sources = {hit.chunk.source_path for hit in expanded}

        # Preserve the successful V2.3 behavior first: add more context from files
        # that primary retrieval already selected.
        raw_top_k = max(max_context_hits + 4, candidate_k * 2)
        raw_hits = self.base_index.hybrid_search(
            query,
            top_k=raw_top_k,
            candidate_k=max(candidate_k, raw_top_k),
            rrf_k=rrf_k,
            max_chunks_per_source=0,
        )
        same_source_budget = min(max_context_hits, len(primary_hits) + 3)
        for hit in raw_hits:
            if len(expanded) >= same_source_budget:
                break
            if hit.chunk.chunk_id in selected_ids or hit.chunk.source_path not in selected_sources:
                continue
            expanded.append(hit)
            selected_ids.add(hit.chunk.chunk_id)

        # V4 addition: follow symbol-definition edges across files for multi-hop QA.
        for hit in self.dependency_hits(query, expanded, max_hits=max_context_hits):
            if len(expanded) >= max_context_hits:
                break
            if hit.chunk.chunk_id in selected_ids:
                continue
            expanded.append(hit)
            selected_ids.add(hit.chunk.chunk_id)

        return expanded
