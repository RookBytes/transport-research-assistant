from __future__ import annotations

import inspect
import json

from transport_rag.config import settings
from transport_rag.rag import RAGService


TARGET = {
    "path_weight": 1.0,
    "candidate_k": 10,
    "rrf_k": 20,
    "max_chunks_per_source": 1,
}


def show_attr(obj, name):
    if hasattr(obj, name):
        print(f"{name} = {getattr(obj, name)!r}")
    else:
        print(f"{name} = <missing>")


def main():
    print("=== SETTINGS OBJECT ===")
    for name in TARGET:
        show_attr(settings, name)

    print("\n=== RAGService.retrieve SOURCE ===")
    try:
        print(inspect.getsource(RAGService.retrieve))
    except Exception as exc:
        print(f"Could not inspect RAGService.retrieve: {exc}")

    service = RAGService(settings)
    index = service.index

    print("\n=== INDEX METHODS ===")
    for method_name in ("hybrid_search", "search_hybrid", "search"):
        if hasattr(index, method_name):
            method = getattr(index, method_name)
            print(f"\n{method_name}{inspect.signature(method)}")
            try:
                print(inspect.getsource(method))
            except Exception:
                pass

    print("\n=== TARGET TUPLE ===")
    print(json.dumps(TARGET, indent=2))

    question = "What was the headline result of the final 20-seed replication?"
    print("\n=== PROBE QUESTION ===")
    print(question)

    # Production path
    try:
        prod_hits = service.retrieve(question)
        print("\nPRODUCTION:")
        for i, hit in enumerate(prod_hits[:6], 1):
            print(i, hit.chunk.source_path, round(hit.score, 6))
    except Exception as exc:
        print(f"Production retrieve failed: {exc}")
        prod_hits = None

    # Explicit tuned call, bypassing RAGService defaults
    hybrid = None
    if hasattr(index, "hybrid_search"):
        hybrid = index.hybrid_search
    elif hasattr(index, "search_hybrid"):
        hybrid = index.search_hybrid

    if hybrid is None:
        print("\nNo hybrid search method found on index.")
        return

    sig = inspect.signature(hybrid)
    kwargs = {}

    if "top_k" in sig.parameters:
        kwargs["top_k"] = getattr(settings, "top_k", 6)
    for name, value in TARGET.items():
        if name in sig.parameters:
            kwargs[name] = value

    print("\nEXPLICIT TUNED CALL:")
    print(kwargs)

    try:
        tuned_hits = hybrid(question, **kwargs)
        for i, hit in enumerate(tuned_hits[:6], 1):
            print(i, hit.chunk.source_path, round(hit.score, 6))
    except Exception as exc:
        print(f"Explicit tuned call failed: {exc}")
        return

    if prod_hits is not None:
        prod_paths = [h.chunk.source_path for h in prod_hits[:6]]
        tuned_paths = [h.chunk.source_path for h in tuned_hits[:6]]
        print("\n=== COMPARISON ===")
        print("same_order =", prod_paths == tuned_paths)
        print("production =", prod_paths)
        print("tuned      =", tuned_paths)


if __name__ == "__main__":
    main()
