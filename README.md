# Transportation Research Assistant

A source-grounded RAG assistant for answering questions about transportation simulation research, experiment configurations, implementation details, and results.

The first target corpus is the public `porto-route-simulation` project. The assistant preserves source file paths and line ranges so every answer can cite the evidence it used.

## Why this project exists

A useful research assistant should do more than send retrieved text to an LLM. This repository is designed to support three experimentally comparable systems:

| Version | Method |
| --- | --- |
| Baseline | LLM answers without retrieved context |
| Basic RAG | Dense retrieval over fixed-size source chunks |
| Improved RAG | Structure-aware chunking, metadata filters, reranking, and calibrated abstention |

The MVP implements the Basic RAG path with source citations and abstention.

## Features

- Ingests Markdown, Python, Java, properties, JSON, YAML, TOML, and text files.
- Preserves `source_path`, `start_line`, and `end_line`.
- Uses Ollama-hosted local embeddings; no Hugging Face runtime dependency.
- Uses a simple persisted NumPy cosine-search index.
- Generates answers locally with Ollama LFM 2.5.
- Requires citations in the generated response.
- Refuses to answer when retrieval evidence is below a configurable threshold.
- FastAPI backend.
- Streamlit chat UI.
- Evaluation dataset format already included.
- Unit tests for chunking and retrieval behavior.

## Repository structure

```text
transport-research-assistant/
├── README.md
├── pyproject.toml
├── .env.example
├── src/
│   └── transport_rag/
│       ├── config.py
│       ├── models.py
│       ├── rag.py
│       ├── ingestion/
│       │   ├── chunking.py
│       │   ├── loaders.py
│       │   └── cli.py
│       ├── retrieval/
│       │   └── index.py
│       ├── generation/
│       │   └── ollama.py
│       └── api/
│           └── main.py
├── app/
│   └── streamlit_app.py
├── evals/
│   └── questions.sample.jsonl
├── tests/
│   ├── test_chunking.py
│   └── test_retrieval.py
└── data/
    └── README.md
```

## Quick start

Create and activate an environment, then install the project:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install -e ".[dev]"
```

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Make sure Ollama is running. Pull the two local models once:

```powershell
ollama pull lfm2.5; ollama pull nomic-embed-text
```

`lfm2.5` generates answers and `nomic-embed-text` creates retrieval embeddings. After those model files are present, the application does not need Hugging Face or a remote LLM API.

Clone the Porto source repository next to this project if you do not already have it:

```powershell
git clone https://github.com/billysuryono1412-code/porto-route-simulation.git
```

Build the retrieval index:

```powershell
python -m transport_rag.ingestion.cli --source ..\porto-route-simulation --index .rag_index
```


Ask a question immediately from the terminal:

```powershell
python -m transport_rag.ask_cli "How is background congestion inferred?"
```

Start the API:

```powershell
uvicorn transport_rag.api.main:app --reload
```

In another terminal, start the UI:

```powershell
streamlit run app\streamlit_app.py
```

## Example questions

- Which behavioral configuration performed best?
- Did rerouting improve the final objective?
- How is background congestion inferred?
- Which properties control rerouting?
- Which file defines the accepted route-choice modes?
- Why should `sim_edge_lookup_freeflow.csv` be used instead of an observed congested lookup?

## Citation format

Retrieved sources are supplied to the generator with IDs such as:

```text
[S1] simulation/README.md:120-148
```

The model is instructed to cite claims with those IDs, for example:

```text
The objective combines four normalized error components [S1].
```

The API also returns the raw source metadata so the UI can show the actual file and line range.

## Abstention behavior

Before generation, the retriever checks the similarity score of the best result. If it is below `TRA_MIN_SCORE`, the assistant returns:

> I don't have enough evidence in the indexed sources to answer that reliably.

This threshold is intentionally configurable because one of the later evaluation tasks is to calibrate answer coverage against groundedness.

## Evaluation plan

`evals/questions.sample.jsonl` contains the starter schema. The full portfolio version should grow to roughly 50–100 manually verified questions across:

- experiment results;
- preprocessing;
- configuration;
- implementation;
- metrics;
- limitations;
- deliberately unanswerable questions.

For each version we will measure:

- Retrieval Recall@K
- reciprocal rank / MRR
- answer correctness
- citation correctness
- groundedness
- abstention precision/recall
- latency

## Next upgrades

1. Build the first 50-question gold evaluation set.
2. Add lexical BM25 retrieval and compare it with dense retrieval.
3. Add reciprocal-rank fusion.
4. Add a cross-encoder reranker.
5. Add metadata-aware filters for source type and repository path.
6. Compare fixed-size chunks against Markdown/code-aware chunking.
7. Add an API-model generator as an optional backend.
8. Add Docker and GitHub Actions after the evaluation pipeline is stable.


## Local-only model path

```text
Question
  -> Ollama / nomic-embed-text
  -> local NumPy cosine retrieval
  -> source chunks with file + line metadata
  -> Ollama / lfm2.5
  -> cited answer
```

LFM 2.5 is a generative model, so retrieval still uses a dedicated embedding model. Both are served by the same local Ollama installation.

## Run the 50-question evaluation

Start with deterministic retrieval metrics:

```powershell
python -m transport_rag.evaluation.run_eval --mode retrieval
```

Then test the complete LFM2.5 RAG pipeline:

```powershell
python -m transport_rag.evaluation.run_eval --mode full
```

For an optional approximate local-model grading of the `expected_facts` fields:

```powershell
python -m transport_rag.evaluation.run_eval --mode full --judge
```

Results are written to `evals/results/` as detailed CSV/JSON files plus a compact summary JSON. See `evals/README.md` for metric definitions and caveats.

## V2 experiment: hybrid retrieval

V1 remains available as `dense`. V2 adds a dependency-free BM25 lexical retriever and fuses it with dense retrieval using Reciprocal Rank Fusion (RRF). It uses the same `.rag_index`; **you do not need to rebuild embeddings**.

Run the original V1 baseline again:

```powershell
python -m transport_rag.evaluation.run_eval --mode retrieval --retriever dense
```

Run V2 hybrid retrieval:

```powershell
python -m transport_rag.evaluation.run_eval --mode retrieval --retriever hybrid
```

The comparison is intentionally controlled: same chunks, embedding model, gold questions, and evaluation metrics. Only the retrieval method changes.

If V2 improves retrieval, test end-to-end generation with:

```powershell
python -m transport_rag.evaluation.run_eval --mode full --retriever hybrid
```

To make the normal CLI/API/UI use V2, set this in `.env`:

```text
TRA_RETRIEVAL_MODE=hybrid
```

Hybrid defaults:

```text
TRA_HYBRID_CANDIDATE_K=15
TRA_RRF_K=60
```

## V3: local LLM reranking

V3 keeps the V2 hybrid candidate generator and uses the already-installed LFM2.5 model as a listwise reranker. No additional model download is required.

Run retrieval evaluation:

```powershell
python -m transport_rag.evaluation.run_eval --mode retrieval --retriever hybrid_rerank
```

Then, if ranking improves, run the full system:

```powershell
python -m transport_rag.evaluation.run_eval --mode full --retriever hybrid_rerank
```

The default reranker considers the top 10 hybrid candidates and returns the requested top K. Adjust with `TRA_RERANK_CANDIDATE_K` if needed.
