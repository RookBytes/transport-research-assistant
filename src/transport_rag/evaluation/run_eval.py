from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

import httpx

from transport_rag.config import settings
from transport_rag.rag import RAGService


@dataclass
class EvalItem:
    id: str
    category: str
    difficulty: str
    answerable: bool
    question: str
    gold_sources: list[str]
    expected_facts: list[str]


def load_eval_set(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            items.append(EvalItem(**data))
    if not items:
        raise ValueError(f"No evaluation items found in {path}")
    return items


def source_rank(retrieved_paths: list[str], gold_sources: list[str]) -> int | None:
    gold = set(gold_sources)
    for rank, path in enumerate(retrieved_paths, start=1):
        if path in gold:
            return rank
    return None



def compare_rank(before: int | None, after: int | None) -> str:
    """Classify whether reranking improved the first gold-source position."""
    if before == after:
        return "unchanged"
    if before is None:
        return "improved" if after is not None else "unchanged"
    if after is None:
        return "worsened"
    return "improved" if after < before else "worsened"


def hit_label(hit) -> str:
    c = hit.chunk
    return f"{c.source_path}:{c.start_line}-{c.end_line}"


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def recall_at(rank: int | None, k: int) -> float:
    return float(rank is not None and rank <= k)


def cited_source_ids(answer: str) -> list[str]:
    return sorted(set(re.findall(r"\[S(\d+)\]", answer)))


def citation_validity(answer: str, n_sources: int) -> tuple[int, int, float | None]:
    ids = cited_source_ids(answer)
    if not ids:
        return 0, 0, None
    valid = sum(1 for value in ids if 1 <= int(value) <= n_sources)
    return valid, len(ids), valid / len(ids)


def _coerce_judge_bool(value) -> bool | None:
    """Convert common JSON boolean variants without treating arbitrary values as truthy."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _normalize_judge_flags(parsed: object, expected_count: int) -> list[bool]:
    """Normalize known local-LLM judge response shapes into one boolean per expected fact."""
    if not isinstance(parsed, dict):
        raise ValueError("Judge response must be a JSON object.")

    # Some local models emit {"covered": true, "expected_fact_1": true, ...}.
    # Prefer the per-fact fields because they preserve fact-level information.
    indexed_values = []
    for i in range(1, expected_count + 1):
        key = f"expected_fact_{i}"
        if key not in parsed:
            indexed_values = []
            break
        value = _coerce_judge_bool(parsed[key])
        if value is None:
            raise ValueError(f"Judge field {key!r} is not boolean-like.")
        indexed_values.append(value)
    if len(indexed_values) == expected_count:
        return indexed_values

    covered = parsed.get("covered")
    if isinstance(covered, list):
        if len(covered) != expected_count:
            raise ValueError(
                f"Judge returned {len(covered)} coverage flags for {expected_count} expected facts."
            )
        normalized = [_coerce_judge_bool(value) for value in covered]
        if any(value is None for value in normalized):
            raise ValueError("Judge coverage array contains a non-boolean-like value.")
        return [bool(value) for value in normalized]

    # A scalar is unambiguous only when there is exactly one expected fact.
    scalar = _coerce_judge_bool(covered)
    if expected_count == 1 and scalar is not None:
        return [scalar]

    raise ValueError("Judge response does not contain one unambiguous flag per expected fact.")


def _call_fact_judge(
    *,
    answer: str,
    numbered_facts: str,
    expected_count: int,
    ollama_url: str,
    model: str,
    timeout: float,
    retry_note: str | None = None,
) -> str:
    schema_example = '{"covered": [' + ", ".join(["true"] * expected_count) + "]}"
    system = (
        "You are a strict evaluation grader. Determine whether each expected fact is substantively "
        "present in the candidate answer. Paraphrases count. Do not give credit for facts absent from "
        "the answer. Return JSON only. The JSON must contain exactly one key named 'covered'. Its value "
        f"must be an array containing exactly {expected_count} booleans, one for each expected fact in "
        f"the same order. Example shape: {schema_example}"
    )
    user = f"EXPECTED FACTS:\n{numbered_facts}\n\nCANDIDATE ANSWER:\n{answer}"
    if retry_note:
        user += (
            "\n\nYour previous response did not match the required schema. "
            f"Problem: {retry_note}\nReturn the corrected JSON object only."
        )

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0.0},
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{ollama_url.rstrip('/')}/api/chat", json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]


def judge_expected_facts(
    answer: str,
    expected_facts: list[str],
    ollama_url: str,
    model: str,
    timeout: float = 180.0,
) -> dict:
    """Approximate local grading. This is useful, but not an independent human gold score."""
    if not expected_facts:
        return {"covered": 0, "total": 0, "coverage": None, "details": [], "attempts": 0}

    numbered = "\n".join(f"{i + 1}. {fact}" for i, fact in enumerate(expected_facts))

    raw = _call_fact_judge(
        answer=answer,
        numbered_facts=numbered,
        expected_count=len(expected_facts),
        ollama_url=ollama_url,
        model=model,
        timeout=timeout,
    )
    fallback_used = False
    try:
        flags = _normalize_judge_flags(json.loads(raw), len(expected_facts))
        attempts = 1
    except (json.JSONDecodeError, ValueError) as first_exc:
        retry_raw = _call_fact_judge(
            answer=answer,
            numbered_facts=numbered,
            expected_count=len(expected_facts),
            ollama_url=ollama_url,
            model=model,
            timeout=timeout,
            retry_note=str(first_exc),
        )
        attempts = 2
        try:
            flags = _normalize_judge_flags(json.loads(retry_raw), len(expected_facts))
        except (json.JSONDecodeError, ValueError):
            # Some small local models repeatedly collapse a multi-fact verdict into
            # {"covered": true}. At that point, grade each fact independently.
            # A scalar boolean is unambiguous for a single expected fact, so this
            # fallback is robust without guessing how a multi-fact scalar should map.
            fallback_used = True
            flags = []
            for fact in expected_facts:
                single_numbered = f"1. {fact}"
                single_raw = _call_fact_judge(
                    answer=answer,
                    numbered_facts=single_numbered,
                    expected_count=1,
                    ollama_url=ollama_url,
                    model=model,
                    timeout=timeout,
                )
                attempts += 1
                try:
                    single_flag = _normalize_judge_flags(json.loads(single_raw), 1)[0]
                except (json.JSONDecodeError, ValueError) as single_first_exc:
                    single_retry_raw = _call_fact_judge(
                        answer=answer,
                        numbered_facts=single_numbered,
                        expected_count=1,
                        ollama_url=ollama_url,
                        model=model,
                        timeout=timeout,
                        retry_note=str(single_first_exc),
                    )
                    attempts += 1
                    try:
                        single_flag = _normalize_judge_flags(json.loads(single_retry_raw), 1)[0]
                    except (json.JSONDecodeError, ValueError) as single_second_exc:
                        raise ValueError(
                            "Judge failed batch grading and per-fact fallback. "
                            f"Fact: {fact!r} | First: {single_raw} | "
                            f"Second: {single_retry_raw} | Error: {single_second_exc}"
                        ) from single_second_exc
                flags.append(single_flag)

    covered = sum(flags)
    return {
        "covered": covered,
        "total": len(flags),
        "coverage": covered / len(flags),
        "details": [
            {"fact": fact, "covered": flag}
            for fact, flag in zip(expected_facts, flags)
        ],
        "attempts": attempts,
        "fallback_used": fallback_used,
    }


def build_summary(rows: list[dict], mode: str, judge: bool) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    summary = {
        "mode": mode,
        "questions": len(rows),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(rows) - len(answerable),
        "retrieval": {
            "recall_at_1": mean(r["recall_at_1"] for r in answerable) if answerable else None,
            "recall_at_3": mean(r["recall_at_3"] for r in answerable) if answerable else None,
            "recall_at_5": mean(r["recall_at_5"] for r in answerable) if answerable else None,
            "mrr": mean(r["reciprocal_rank"] for r in answerable) if answerable else None,
        },
    }

    rerank_rows = [r for r in rows if "gold_source_rank_before" in r]
    if rerank_rows:
        answerable_rerank = [r for r in rerank_rows if r["answerable"]]
        summary["reranking"] = {
            "order_changed_questions": sum(bool(r["rerank_order_changed"]) for r in rerank_rows),
            "order_changed_rate": mean(float(r["rerank_order_changed"]) for r in rerank_rows),
            "answerable_rank_improved": sum(r["rerank_rank_change"] == "improved" for r in answerable_rerank),
            "answerable_rank_worsened": sum(r["rerank_rank_change"] == "worsened" for r in answerable_rerank),
            "answerable_rank_unchanged": sum(r["rerank_rank_change"] == "unchanged" for r in answerable_rerank),
            "gold_entered_top_k": sum(
                r["gold_source_rank_before"] is None and r["gold_source_rank_after"] is not None
                for r in answerable_rerank
            ),
            "gold_left_top_k": sum(
                r["gold_source_rank_before"] is not None and r["gold_source_rank_after"] is None
                for r in answerable_rerank
            ),
        }

    if mode == "full":
        summary["generation"] = {
            "abstention_accuracy": mean(float(r["abstention_correct"]) for r in rows),
            "mean_total_latency_sec": mean(r["total_latency_sec"] for r in rows),
            "mean_answer_latency_sec": mean(r["answer_latency_sec"] for r in rows),
            "citation_validity": mean(
                r["citation_validity"] for r in rows if r["citation_validity"] is not None
            ) if any(r["citation_validity"] is not None for r in rows) else None,
        }
        if judge:
            judged = [r for r in answerable if r.get("fact_coverage") is not None]
            successful_calls = [r for r in answerable if r.get("judge_status") == "ok"]
            failed_calls = [r for r in answerable if r.get("judge_status") == "error"]
            abstained_answerable = [r for r in answerable if r.get("judge_status") == "answerable_abstained"]
            summary["generation"]["mean_fact_coverage_local_judge"] = (
                mean(r["fact_coverage"] for r in judged) if judged else None
            )
            summary["generation"]["judge_successful_calls"] = len(successful_calls)
            summary["generation"]["judge_failed_calls"] = len(failed_calls)
            summary["generation"]["answerable_abstentions_scored_zero"] = len(abstained_answerable)
            summary["generation"]["judge_retries"] = sum(
                max(0, int(r.get("judge_attempts", 0)) - 1) for r in successful_calls
            )
            summary["generation"]["judge_per_fact_fallbacks"] = sum(
                bool(r.get("judge_fallback_used", False)) for r in successful_calls
            )
            summary["generation"]["judge_note"] = (
                "Fact coverage is graded by the configured local Ollama model and is not an independent human score. "
                "Answerable abstentions are scored as zero coverage; judge parse failures are excluded and reported. "
                "If batch grading fails twice, expected facts are graded individually as a conservative fallback."
            )

    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    flat_rows = []
    for row in rows:
        flat = dict(row)
        for key in (
            "gold_sources",
            "retrieved_sources",
            "pre_rerank_sources",
            "post_rerank_sources",
            "judge_details",
        ):
            if key in flat:
                flat[key] = json.dumps(flat[key], ensure_ascii=False)
        flat_rows.append(flat)

    fields: list[str] = []
    for row in flat_rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the transportation RAG assistant.")
    parser.add_argument("--questions", type=Path, default=Path("evals/questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evals/results"))
    parser.add_argument(
        "--retriever",
        choices=("dense", "hybrid", "hybrid_rerank"),
        default="dense",
        help=("dense = V1 semantic baseline; hybrid = V2 dense + BM25 + RRF; "
              "hybrid_rerank = V3 hybrid candidates reranked by local LFM2.5."),
    )
    parser.add_argument(
        "--mode",
        choices=("retrieval", "full"),
        default="retrieval",
        help="retrieval = no answer generation; full = run LFM2.5 answers too.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="In full mode, use the local Ollama model to grade expected-fact coverage.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=6,
        help="Number of chunks to retrieve. Must be at least 5 for Recall@5.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional small smoke-test limit.")
    args = parser.parse_args()

    if args.max_k < 5:
        raise SystemExit("--max-k must be at least 5 so Recall@5 is meaningful.")
    if args.judge and args.mode != "full":
        raise SystemExit("--judge requires --mode full.")

    items = load_eval_set(args.questions)
    if args.limit > 0:
        items = items[: args.limit]

    service = RAGService(settings, retrieval_mode=args.retriever)
    rows: list[dict] = []

    for i, item in enumerate(items, start=1):
        print(f"[{i:02d}/{len(items):02d}] {item.id}: {item.question}")

        retrieval_start = time.perf_counter()
        retrieval = service.retrieve_with_diagnostics(item.question, top_k=args.max_k)
        retrieval_latency = time.perf_counter() - retrieval_start
        hits = retrieval["hits"]
        pre_hits = retrieval["pre_rerank_hits"]
        retrieved_paths = [hit.chunk.source_path for hit in hits]
        pre_paths = [hit.chunk.source_path for hit in pre_hits]
        rank = source_rank(retrieved_paths, item.gold_sources) if item.answerable else None
        rank_before = source_rank(pre_paths, item.gold_sources) if item.answerable else None
        pre_labels = [hit_label(hit) for hit in pre_hits]
        post_labels = [hit_label(hit) for hit in hits]

        row = {
            "id": item.id,
            "category": item.category,
            "difficulty": item.difficulty,
            "answerable": item.answerable,
            "question": item.question,
            "gold_sources": item.gold_sources,
            "retrieved_sources": retrieved_paths,
            "gold_source_rank": rank,
            "recall_at_1": recall_at(rank, 1) if item.answerable else None,
            "recall_at_3": recall_at(rank, 3) if item.answerable else None,
            "recall_at_5": recall_at(rank, 5) if item.answerable else None,
            "reciprocal_rank": reciprocal_rank(rank) if item.answerable else None,
            "retrieval_latency_sec": round(retrieval_latency, 4),
        }

        if args.retriever == "hybrid_rerank":
            row.update(
                {
                    "pre_rerank_sources": pre_labels,
                    "post_rerank_sources": post_labels,
                    "gold_source_rank_before": rank_before,
                    "gold_source_rank_after": rank,
                    "rerank_rank_change": compare_rank(rank_before, rank) if item.answerable else None,
                    "rerank_order_changed": pre_labels != post_labels,
                }
            )

        if args.mode == "full":
            total_start = time.perf_counter()
            answer_start = time.perf_counter()
            result = service.answer(item.question)
            answer_latency = time.perf_counter() - answer_start
            total_latency = time.perf_counter() - total_start

            expected_abstain = not item.answerable
            abstention_correct = result["abstained"] == expected_abstain
            valid, cited, validity = citation_validity(result["answer"], len(result["sources"]))

            row.update(
                {
                    "answer": result["answer"],
                    "abstained": result["abstained"],
                    "expected_abstain": expected_abstain,
                    "abstention_correct": abstention_correct,
                    "valid_citation_ids": valid,
                    "cited_source_ids": cited,
                    "citation_validity": validity,
                    "answer_latency_sec": round(answer_latency, 4),
                    "total_latency_sec": round(total_latency, 4),
                }
            )

            if args.judge and item.answerable and not result["abstained"]:
                try:
                    judged = judge_expected_facts(
                        result["answer"],
                        item.expected_facts,
                        settings.ollama_url,
                        settings.ollama_model,
                    )
                    row["fact_coverage"] = judged["coverage"]
                    row["judge_details"] = judged["details"]
                    row["judge_status"] = "ok"
                    row["judge_attempts"] = judged["attempts"]
                    row["judge_fallback_used"] = judged.get("fallback_used", False)
                    row["judge_error"] = None
                except Exception as exc:
                    print(f"  Judge warning: {exc}")
                    row["fact_coverage"] = None
                    row["judge_details"] = []
                    row["judge_status"] = "error"
                    row["judge_attempts"] = 2
                    row["judge_fallback_used"] = False
                    row["judge_error"] = str(exc)
            elif args.judge:
                row["fact_coverage"] = 0.0 if item.answerable else None
                row["judge_details"] = []
                row["judge_status"] = "answerable_abstained" if item.answerable else "not_applicable"
                row["judge_attempts"] = 0
                row["judge_fallback_used"] = False
                row["judge_error"] = None

        rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"eval_{args.retriever}_{args.mode}_{timestamp}"
    csv_path = args.output / f"{stem}.csv"
    json_path = args.output / f"{stem}.json"
    summary_path = args.output / f"{stem}_summary.json"

    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = build_summary(rows, args.mode, args.judge)
    summary["retriever"] = args.retriever
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed CSV: {csv_path}")
    print(f"Detailed JSON: {json_path}")
    print(f"Summary JSON:  {summary_path}")


if __name__ == "__main__":
    main()
