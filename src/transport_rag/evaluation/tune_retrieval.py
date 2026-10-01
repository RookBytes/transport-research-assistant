from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import mean

from transport_rag.config import settings
from transport_rag.evaluation.run_eval import (
    EvalItem,
    load_eval_set,
    recall_at,
    reciprocal_rank,
    source_rank,
)
from transport_rag.models import SearchHit
from transport_rag.retrieval.index import DenseIndex


def parse_float_grid(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one number")
    return values


def parse_int_grid(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one integer")
    return values


def diversify_ranked_chunks(
    ranked: list[tuple[str, float]],
    source_by_chunk: dict[str, str],
    top_k: int,
    max_chunks_per_source: int,
) -> list[tuple[str, float]]:
    """Round-robin source diversification used by the V2.1 experiment.

    A non-positive max disables diversification and simply truncates the fused
    ranking. Positive values take the best chunk from each source first, then
    the second-best from each source, up to the configured per-source cap.
    """
    if top_k <= 0:
        return []
    if max_chunks_per_source <= 0:
        return ranked[:top_k]

    selected: list[tuple[str, float]] = []
    selected_ids: set[str] = set()
    source_counts: defaultdict[str, int] = defaultdict(int)

    for source_slot in range(max_chunks_per_source):
        seen_this_round: set[str] = set()
        for chunk_id, score in ranked:
            if chunk_id in selected_ids:
                continue
            source = source_by_chunk[chunk_id]
            if source in seen_this_round:
                continue
            if source_counts[source] != source_slot:
                continue
            selected.append((chunk_id, score))
            selected_ids.add(chunk_id)
            source_counts[source] += 1
            seen_this_round.add(source)
            if len(selected) >= top_k:
                return selected
    return selected


def fuse_hits(
    dense_hits: list[SearchHit],
    lexical_hits: list[SearchHit],
    top_k: int,
    rrf_k: int,
    max_chunks_per_source: int,
) -> list[SearchHit]:
    """Fuse cached dense and lexical candidates with RRF, then diversify."""
    by_id = {hit.chunk.chunk_id: hit.chunk for hit in dense_hits + lexical_hits}
    fused: defaultdict[str, float] = defaultdict(float)

    for rank, hit in enumerate(dense_hits, start=1):
        fused[hit.chunk.chunk_id] += 1.0 / (rrf_k + rank)
    for rank, hit in enumerate(lexical_hits, start=1):
        fused[hit.chunk.chunk_id] += 1.0 / (rrf_k + rank)

    ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    source_by_chunk = {chunk_id: by_id[chunk_id].source_path for chunk_id, _ in ranked}
    ranked = diversify_ranked_chunks(
        ranked,
        source_by_chunk=source_by_chunk,
        top_k=top_k,
        max_chunks_per_source=max_chunks_per_source,
    )

    max_rrf = 2.0 / (rrf_k + 1.0)
    return [
        SearchHit(chunk=by_id[chunk_id], score=float(score / max_rrf))
        for chunk_id, score in ranked
    ]


def score_configuration(
    items: list[EvalItem],
    dense_cache: dict[str, list[SearchHit]],
    lexical_cache: dict[float, dict[str, list[SearchHit]]],
    *,
    path_weight: float,
    candidate_k: int,
    rrf_k: int,
    max_chunks_per_source: int,
    top_k: int,
) -> dict:
    answerable = [item for item in items if item.answerable]
    rows: list[dict] = []

    for item in answerable:
        dense_hits = dense_cache[item.id][:candidate_k]
        lexical_hits = lexical_cache[path_weight][item.id][:candidate_k]
        hits = fuse_hits(
            dense_hits,
            lexical_hits,
            top_k=top_k,
            rrf_k=rrf_k,
            max_chunks_per_source=max_chunks_per_source,
        )
        paths = [hit.chunk.source_path for hit in hits]
        rank = source_rank(paths, item.gold_sources)
        rows.append(
            {
                "id": item.id,
                "rank": rank,
                "r1": recall_at(rank, 1),
                "r3": recall_at(rank, 3),
                "r5": recall_at(rank, 5),
                "rr": reciprocal_rank(rank),
            }
        )

    return {
        "path_weight": path_weight,
        "candidate_k": candidate_k,
        "rrf_k": rrf_k,
        "max_chunks_per_source": max_chunks_per_source,
        "recall_at_1": mean(row["r1"] for row in rows),
        "recall_at_3": mean(row["r3"] for row in rows),
        "recall_at_5": mean(row["r5"] for row in rows),
        "mrr": mean(row["rr"] for row in rows),
        "top5_miss_ids": [row["id"] for row in rows if not row["r5"]],
        "rank1_ids": [row["id"] for row in rows if row["r1"]],
    }


def leaderboard_sort_key(row: dict, min_recall5: float) -> tuple:
    """Prefer eligible configurations, then ranking quality, then coverage."""
    eligible = row["recall_at_5"] >= min_recall5
    return (
        1 if eligible else 0,
        row["mrr"],
        row["recall_at_1"],
        row["recall_at_3"],
        row["recall_at_5"],
        -row["candidate_k"],
        -row["max_chunks_per_source"],
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = [
        "rank",
        "eligible",
        "path_weight",
        "candidate_k",
        "rrf_k",
        "max_chunks_per_source",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "mrr",
        "top5_miss_ids",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "eligible": row["eligible"],
                    "path_weight": row["path_weight"],
                    "candidate_k": row["candidate_k"],
                    "rrf_k": row["rrf_k"],
                    "max_chunks_per_source": row["max_chunks_per_source"],
                    "recall_at_1": row["recall_at_1"],
                    "recall_at_3": row["recall_at_3"],
                    "recall_at_5": row["recall_at_5"],
                    "mrr": row["mrr"],
                    "top5_miss_ids": json.dumps(row["top5_miss_ids"]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep hybrid retrieval parameters without generating answers. "
            "Use this on a development set, not a final held-out test set."
        )
    )
    parser.add_argument("--questions", type=Path, default=Path("evals/questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evals/tuning"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--path-weights",
        type=parse_float_grid,
        default=parse_float_grid("0,0.1,0.25,0.5,1,2"),
        help="Comma-separated BM25 source-path field weights.",
    )
    parser.add_argument(
        "--candidate-k",
        type=parse_int_grid,
        default=parse_int_grid("10,15,20"),
        help="Comma-separated dense/BM25 candidate-pool sizes.",
    )
    parser.add_argument(
        "--rrf-k",
        type=parse_int_grid,
        default=parse_int_grid("20,60,100"),
        help="Comma-separated RRF constants.",
    )
    parser.add_argument(
        "--max-chunks-per-source",
        type=parse_int_grid,
        default=parse_int_grid("0,1,2,3"),
        help="0 disables diversification; otherwise cap chunks per source.",
    )
    parser.add_argument(
        "--min-recall5",
        type=float,
        default=0.95,
        help="Configurations below this Recall@5 are ranked behind eligible ones.",
    )
    parser.add_argument("--show", type=int, default=15, help="Rows to print from leaderboard.")
    args = parser.parse_args()

    if args.top_k < 5:
        raise SystemExit("--top-k must be at least 5 so Recall@5 is meaningful.")
    if not 0.0 <= args.min_recall5 <= 1.0:
        raise SystemExit("--min-recall5 must be in [0,1].")
    if any(value <= 0 for value in args.candidate_k):
        raise SystemExit("--candidate-k values must be positive.")
    if any(value < 0 for value in args.rrf_k):
        raise SystemExit("--rrf-k values must be non-negative.")

    items = load_eval_set(args.questions)
    answerable = [item for item in items if item.answerable]
    if not answerable:
        raise SystemExit("No answerable questions found.")

    print(
        "WARNING: This is a tuning/development benchmark. Because the current 50-question "
        "set has already been inspected repeatedly, do not report the tuned score as final "
        "held-out performance."
    )
    print(f"Questions: {len(items)} total, {len(answerable)} answerable")

    index = DenseIndex.load(settings.index_dir, ollama_url=settings.ollama_url)
    max_candidate_k = max(max(args.candidate_k), args.top_k)

    print(f"\nCaching dense candidates (top {max_candidate_k})...")
    dense_cache: dict[str, list[SearchHit]] = {}
    for pos, item in enumerate(answerable, start=1):
        print(f"  dense [{pos:02d}/{len(answerable):02d}] {item.id}")
        dense_cache[item.id] = index.search(item.question, top_k=max_candidate_k)

    lexical_cache: dict[float, dict[str, list[SearchHit]]] = {}
    for weight in args.path_weights:
        print(f"\nCaching BM25 candidates for path_weight={weight:g}...")
        lexical_cache[weight] = {}
        for pos, item in enumerate(answerable, start=1):
            if pos == 1 or pos == len(answerable) or pos % 10 == 0:
                print(f"  bm25 [{pos:02d}/{len(answerable):02d}] {item.id}")
            lexical_cache[weight][item.id] = index.bm25_search(
                item.question,
                top_k=max_candidate_k,
                path_weight=weight,
            )

    configs = list(
        product(
            args.path_weights,
            args.candidate_k,
            args.rrf_k,
            args.max_chunks_per_source,
        )
    )
    print(f"\nScoring {len(configs)} configurations...")
    rows: list[dict] = []
    for path_weight, candidate_k, rrf_k, max_chunks in configs:
        row = score_configuration(
            items,
            dense_cache,
            lexical_cache,
            path_weight=path_weight,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            max_chunks_per_source=max_chunks,
            top_k=args.top_k,
        )
        row["eligible"] = row["recall_at_5"] >= args.min_recall5
        rows.append(row)

    rows.sort(key=lambda row: leaderboard_sort_key(row, args.min_recall5), reverse=True)

    baseline = next(
        (
            row
            for row in rows
            if row["path_weight"] == 0.0
            and row["candidate_k"] == 15
            and row["rrf_k"] == 60
            and row["max_chunks_per_source"] == 0
        ),
        None,
    )
    best = rows[0]

    print("\n=== LEADERBOARD ===")
    header = "#   path_w  cand  rrf  max/src   R@1    R@3    R@5    MRR    eligible"
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(rows[: max(1, args.show)], start=1):
        print(
            f"{rank:<3} {row['path_weight']:<7g} {row['candidate_k']:<5d} "
            f"{row['rrf_k']:<4d} {row['max_chunks_per_source']:<9d} "
            f"{row['recall_at_1']:.3f}  {row['recall_at_3']:.3f}  "
            f"{row['recall_at_5']:.3f}  {row['mrr']:.3f}  {row['eligible']}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output / f"retrieval_tuning_{timestamp}.csv"
    json_path = args.output / f"retrieval_tuning_{timestamp}.json"
    summary_path = args.output / f"retrieval_tuning_{timestamp}_summary.json"

    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "questions_file": str(args.questions),
        "answerable_questions": len(answerable),
        "configurations_tested": len(rows),
        "selection_rule": (
            f"Recall@5 >= {args.min_recall5:.3f}, then maximize MRR, Recall@1, Recall@3, Recall@5"
        ),
        "best": best,
        "baseline_v2_if_present": baseline,
        "warning": (
            "Development/tuning result only. Use a fresh held-out question set for final reporting."
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== BEST CONFIG ===")
    print(json.dumps(best, indent=2))
    if baseline is not None:
        print("\n=== V2 BASELINE (path_weight=0, candidate_k=15, rrf_k=60, no diversification) ===")
        print(json.dumps(baseline, indent=2))
    print(f"\nLeaderboard CSV: {csv_path}")
    print(f"All results JSON: {json_path}")
    print(f"Summary JSON:     {summary_path}")


if __name__ == "__main__":
    main()
