from __future__ import annotations

import py_compile
import re
import shutil
from pathlib import Path

ROOT = Path.cwd()
RAG = ROOT / "src" / "transport_rag" / "rag.py"
RAG_BAK = ROOT / "src" / "transport_rag" / "rag.py.v22.bak"
CONFIG = ROOT / "src" / "transport_rag" / "config.py"
INDEX = ROOT / "src" / "transport_rag" / "retrieval" / "index.py"

TARGETS = {
    "path_weight": "1.0",
    "candidate_k": "10",
    "rrf_k": "20",
    "max_chunks_per_source": "1",
}

ENV_DEFAULTS = {
    "path_weight": ("TRA_PATH_WEIGHT", "1.0"),
    "candidate_k": ("TRA_CANDIDATE_K", "10"),
    "rrf_k": ("TRA_RRF_K", "20"),
    "max_chunks_per_source": ("TRA_MAX_CHUNKS_PER_SOURCE", "1"),
}


def restore_rag():
    if not RAG_BAK.exists():
        raise SystemExit(
            f"Backup not found: {RAG_BAK}\n"
            "Do not continue automatically. Restore rag.py manually from git or paste it for repair."
        )
    shutil.copy2(RAG_BAK, RAG)
    print(f"[OK] Restored {RAG} from {RAG_BAK}")


def patch_config():
    if not CONFIG.exists():
        print("[WARN] config.py not found; skipping.")
        return

    text = CONFIG.read_text(encoding="utf-8")
    original = text
    changed = 0

    for name, (env_name, default) in ENV_DEFAULTS.items():
        # Common form:
        # rrf_k: int = int(os.getenv("TRA_RRF_K", "60"))
        patterns = [
            (
                rf'({re.escape(name)}\s*:\s*int\s*=\s*int\(os\.getenv\(\s*"{re.escape(env_name)}"\s*,\s*")[^"]+("\s*\)\))',
                rf'\g<1>{default}\g<2>',
            ),
            (
                rf'({re.escape(name)}\s*:\s*float\s*=\s*float\(os\.getenv\(\s*"{re.escape(env_name)}"\s*,\s*")[^"]+("\s*\)\))',
                rf'\g<1>{default}\g<2>',
            ),
            (
                rf'({re.escape(name)}\s*=\s*int\(os\.getenv\(\s*"{re.escape(env_name)}"\s*,\s*")[^"]+("\s*\)\))',
                rf'\g<1>{default}\g<2>',
            ),
            (
                rf'({re.escape(name)}\s*=\s*float\(os\.getenv\(\s*"{re.escape(env_name)}"\s*,\s*")[^"]+("\s*\)\))',
                rf'\g<1>{default}\g<2>',
            ),
            (
                rf'({re.escape(name)}\s*:\s*(?:int|float)\s*=\s*)[0-9.]+',
                rf'\g<1>{default}',
            ),
        ]

        local_changed = False
        for pat, repl in patterns:
            text2, n = re.subn(pat, repl, text, count=1, flags=re.MULTILINE)
            if n:
                text = text2
                print(f"[OK] config.py {name} default -> {default}")
                changed += 1
                local_changed = True
                break

        if not local_changed:
            print(f"[INFO] config.py: no {name} setting found")

    if text != original:
        backup = CONFIG.with_suffix(".py.v22repair.bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        CONFIG.write_text(text, encoding="utf-8")


def patch_index_defaults():
    if not INDEX.exists():
        print("[WARN] index.py not found; skipping.")
        return

    text = INDEX.read_text(encoding="utf-8")
    original = text

    for name, value in TARGETS.items():
        patterns = [
            (rf'({re.escape(name)}\s*:\s*int\s*=\s*)\d+', rf'\g<1>{value}'),
            (rf'({re.escape(name)}\s*:\s*float\s*=\s*)[0-9.]+', rf'\g<1>{value}'),
        ]
        for pat, repl in patterns:
            text2, n = re.subn(pat, repl, text, count=1, flags=re.MULTILINE)
            if n:
                text = text2
                print(f"[OK] index.py fallback {name} -> {value}")
                break

    if text != original:
        INDEX.write_text(text, encoding="utf-8")


def compile_check():
    failures = []
    for path in (RAG, CONFIG, INDEX):
        if not path.exists():
            continue
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"[PASS] syntax: {path.relative_to(ROOT)}")
        except Exception as exc:
            failures.append((path, exc))
            print(f"[FAIL] syntax: {path.relative_to(ROOT)}: {exc}")

    if failures:
        raise SystemExit("Syntax check failed. No evaluation should be run yet.")


def main():
    print(f"Repo: {ROOT}")
    restore_rag()
    patch_config()
    patch_index_defaults()
    compile_check()
    print()
    print("Repair complete.")
    print("Run:")
    print("  python -m pytest -q")
    print("  python -m transport_rag.evaluation.run_eval --mode retrieval --retriever hybrid")


if __name__ == "__main__":
    main()
