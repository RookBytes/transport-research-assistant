from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()

INDEX = ROOT / "src" / "transport_rag" / "retrieval" / "index.py"
RAG = ROOT / "src" / "transport_rag" / "rag.py"

TARGET = {
    "path_weight": "1.0",
    "candidate_k": "10",
    "rrf_k": "20",
    "max_chunks_per_source": "1",
}

def replace_once(text: str, patterns: list[tuple[str, str]], label: str) -> tuple[str, bool]:
    for pattern, replacement in patterns:
        new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if n:
            print(f"[OK] {label}")
            return new_text, True
    print(f"[SKIP] {label}: no matching target found")
    return text, False

def patch_index() -> int:
    if not INDEX.exists():
        raise SystemExit(f"Missing file: {INDEX}")

    text = INDEX.read_text(encoding="utf-8")
    original = text
    changes = 0

    # 1) Freeze path_weight default wherever hybrid/BM25 exposes it.
    text, changed = replace_once(
        text,
        [
            (r"(path_weight\s*:\s*float\s*=\s*)[0-9.]+", rf"\g<1>{TARGET['path_weight']}"),
            (r"(path_weight\s*=\s*)[0-9.]+", rf"\g<1>{TARGET['path_weight']}"),
        ],
        "index.py path_weight -> 1.0",
    )
    changes += int(changed)

    # 2) Freeze candidate_k default.
    text, changed = replace_once(
        text,
        [
            (r"(candidate_k\s*:\s*int\s*=\s*)\d+", rf"\g<1>{TARGET['candidate_k']}"),
            (r"(candidate_k\s*=\s*)\d+", rf"\g<1>{TARGET['candidate_k']}"),
        ],
        "index.py candidate_k -> 10",
    )
    changes += int(changed)

    # 3) Freeze rrf_k default.
    text, changed = replace_once(
        text,
        [
            (r"(rrf_k\s*:\s*int\s*=\s*)\d+", rf"\g<1>{TARGET['rrf_k']}"),
            (r"(rrf_k\s*=\s*)\d+", rf"\g<1>{TARGET['rrf_k']}"),
        ],
        "index.py rrf_k -> 20",
    )
    changes += int(changed)

    # 4) Freeze source diversification default.
    text, changed = replace_once(
        text,
        [
            (r"(max_chunks_per_source\s*:\s*int\s*=\s*)\d+", rf"\g<1>{TARGET['max_chunks_per_source']}"),
            (r"(max_chunks_per_source\s*=\s*)\d+", rf"\g<1>{TARGET['max_chunks_per_source']}"),
        ],
        "index.py max_chunks_per_source -> 1",
    )
    changes += int(changed)

    if text != original:
        backup = INDEX.with_suffix(".py.v22.bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
            print(f"[BACKUP] {backup}")
        INDEX.write_text(text, encoding="utf-8")
    else:
        print("[INFO] index.py unchanged; values may already be frozen elsewhere.")

    return changes


def patch_rag() -> int:
    if not RAG.exists():
        raise SystemExit(f"Missing file: {RAG}")

    text = RAG.read_text(encoding="utf-8")
    original = text
    changes = 0

    # Prefer making the production hybrid search call explicit. This handles
    # different earlier patch variants where defaults in index.py may be bypassed.
    #
    # Match calls like:
    # self.index.hybrid_search(question, top_k=...)
    # or:
    # self.index.search_hybrid(question, top_k=...)
    call_patterns = [
        r"(self\.index\.(?:hybrid_search|search_hybrid)\(\s*question\s*,)(.*?)(\))",
        r"(self\.index\.(?:hybrid_search|search_hybrid)\(\s*query\s*,)(.*?)(\))",
    ]

    explicit_args = (
        "\n                top_k=self.settings.top_k,"
        "\n                candidate_k=10,"
        "\n                rrf_k=20,"
        "\n                path_weight=1.0,"
        "\n                max_chunks_per_source=1,"
        "\n            "
    )

    for pat in call_patterns:
        m = re.search(pat, text, flags=re.DOTALL)
        if not m:
            continue

        body = m.group(2)

        # Only rewrite if this looks like one function call rather than spanning
        # unrelated code.
        if len(body) > 800:
            continue

        # Keep any existing top_k expression if present.
        topk = re.search(r"top_k\s*=\s*([^,\n\)]+)", body)
        topk_expr = topk.group(1).strip() if topk else "self.settings.top_k"

        replacement_args = (
            f"\n                top_k={topk_expr},"
            "\n                candidate_k=10,"
            "\n                rrf_k=20,"
            "\n                path_weight=1.0,"
            "\n                max_chunks_per_source=1,"
            "\n            "
        )

        replacement = m.group(1) + replacement_args + m.group(3)
        text = text[:m.start()] + replacement + text[m.end():]
        print("[OK] rag.py production hybrid call frozen to tuned V2.2 parameters")
        changes += 1
        break

    # If the call shape differs, patch named constants/defaults in rag.py.
    if changes == 0:
        for name, value in TARGET.items():
            patterns = [
                (rf"({re.escape(name)}\s*:\s*(?:float|int)\s*=\s*)[0-9.]+", rf"\g<1>{value}"),
                (rf"({re.escape(name)}\s*=\s*)[0-9.]+", rf"\g<1>{value}"),
            ]
            text, changed = replace_once(text, patterns, f"rag.py {name} -> {value}")
            changes += int(changed)

    # Add a diagnostic print if none exists already.
    marker = "Hybrid retrieval parameters:"
    if marker not in text:
        # Put it after service/index initialization when possible.
        insertion_patterns = [
            r"(self\.index\s*=\s*DenseIndex\.load\([^\n]+\)\n)",
            r"(self\.index\s*=\s*[^\n]+\n)",
        ]
        inserted = False
        for pat in insertion_patterns:
            m = re.search(pat, text)
            if m:
                diagnostic = (
                    '        print("Hybrid retrieval parameters: '
                    'path_weight=1.0, candidate_k=10, rrf_k=20, '
                    'max_chunks_per_source=1")\n'
                )
                text = text[:m.end()] + diagnostic + text[m.end():]
                print("[OK] rag.py added effective-parameter diagnostic")
                changes += 1
                inserted = True
                break
        if not inserted:
            print("[SKIP] Could not safely insert diagnostic print.")
    else:
        print("[INFO] Diagnostic print already present.")

    if text != original:
        backup = RAG.with_suffix(".py.v22.bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
            print(f"[BACKUP] {backup}")
        RAG.write_text(text, encoding="utf-8")
    else:
        print("[INFO] rag.py unchanged.")

    return changes


def main():
    print(f"Repo: {ROOT}")
    index_changes = patch_index()
    rag_changes = patch_rag()

    print()
    print(f"Changes applied: index.py={index_changes}, rag.py={rag_changes}")
    print("Target configuration:")
    print("  path_weight=1.0")
    print("  candidate_k=10")
    print("  rrf_k=20")
    print("  max_chunks_per_source=1")
    print()
    print("Next:")
    print("  python -m pytest -q")
    print("  python -m transport_rag.evaluation.run_eval --mode retrieval --retriever hybrid")


if __name__ == "__main__":
    main()
