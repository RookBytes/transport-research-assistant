# Evaluation

`questions.jsonl` is the fixed 50-question gold evaluation set. Do not tune individual answers against this file; use it as a held-out benchmark while comparing retrieval/generation versions.

## 1. Fast retrieval evaluation

This tests whether the correct source file appears near the top of retrieval. It does **not** ask LFM2.5 to generate answers.

```powershell
python -m transport_rag.evaluation.run_eval --mode retrieval
```

Reported retrieval metrics over answerable questions:

- Recall@1
- Recall@3
- Recall@5
- MRR (mean reciprocal rank)
- retrieval latency

A gold source counts as retrieved when the `source_path` of a retrieved chunk exactly matches one of the question's `gold_sources`.

## 2. Full RAG evaluation

This additionally sends every question through the complete RAG pipeline and LFM2.5.

```powershell
python -m transport_rag.evaluation.run_eval --mode full
```

It adds:

- generated answer
- expected vs actual abstention
- abstention accuracy
- citation-ID validity
- answer latency

## 3. Optional local answer-quality judge

To get an approximate expected-fact coverage score, use the configured local Ollama model as a grader:

```powershell
python -m transport_rag.evaluation.run_eval --mode full --judge
```

This is useful for iteration, but it is **not an independent evaluation** because the same local model may be generating and grading answers. For portfolio reporting, keep retrieval metrics deterministic and manually review a sample of answer-quality judgments.

## Smoke test

Run only the first five questions:

```powershell
python -m transport_rag.evaluation.run_eval --mode retrieval --limit 5
```

Or test five full RAG answers:

```powershell
python -m transport_rag.evaluation.run_eval --mode full --limit 5
```

## Outputs

Every run writes timestamped files under `evals/results/`:

```text
evals/results/
├── eval_retrieval_YYYYMMDD_HHMMSS.csv
├── eval_retrieval_YYYYMMDD_HHMMSS.json
└── eval_retrieval_YYYYMMDD_HHMMSS_summary.json
```

Full runs use `eval_full_...` filenames.
