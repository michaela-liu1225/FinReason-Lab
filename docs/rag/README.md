# FinReason-Lab v2: evidence retrieval and bounded tool reasoning

This directory documents Yuxin Liu's independent post-course extension of the
six-person UCL project. The original LoRA/QLoRA experiments and team attribution
remain unchanged. The extension asks a separate question: how much of the gap
between full-report and oracle-evidence reasoning can a deployable retrieval and
tool pipeline close?

## Architecture

```text
FinQA report
  └─ canonical loader
       └─ paragraph + table-row chunks with stable evidence IDs
            ├─ BM25 ───────────────┐
            └─ dense embeddings ───┴─ reciprocal-rank fusion
                                         └─ optional cross-encoder reranker
                                                ↓
                              structured {program, citations}
                                                ↓
                            citation + numeric grounding checks
                                                ↓
                         allow-listed Decimal FinQA executor
                                                ↓
                              verifier → answer or abstention
                                  ↖ at most N repairs
```

The default benchmark retrieves only within the report associated with each
question. That matches FinQA's full-context task: the report is known, but its
supporting facts are not. Retrieval APIs accept chunks rather than examples, so
they cannot read gold evidence labels while ranking.

The workflow is deliberately bounded. A model may repair an invalid program,
unsupported citation, ungrounded number, tool failure, or failed verification,
but cannot enter an unbounded loop. It returns an explicit answer or an explicit
abstention reason, plus a trace of every stage.

## Implemented components

- Canonical loaders for official FinQA JSON and the project's flattened JSONL.
  Context-only flattened records are supported for inference, but labelled
  retrieval evaluation requires source-preserving text/table provenance.
- Paragraph, table-row and optional table-cell chunks with stable provenance.
- Dependency-free Okapi BM25.
- Pluggable dense retrieval, RRF fusion, and optional cross-encoder reranking.
- Recall@k, MRR@k, AP@k, evidence-type slices and latency reporting.
- A Decimal-only, allow-listed FinQA DSL executor; no `eval` or arbitrary SQL.
- Numeric and symbolic table operations grounded in cited evidence.
- Structured OpenAI-compatible/vLLM model adapter.
- Bounded `retrieve → generate → validate → execute → verify → repair/abstain`
  workflow.
- FastAPI endpoints, Docker packaging, pytest coverage gate and GitHub Actions.
- Deterministic experiment IDs plus unique, immutable run artifacts recording
  dataset/config/code hashes, Git revision and dirty-worktree status.
- A leakage-safe fixed-model end-to-end evaluator with strict answer accuracy,
  coverage, citation, execution, repair, token-usage and latency accounting.

## Reproduce the current benchmarks

Run these commands from the repository root with Python 3.10 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[api,semantic,dev]"
python scripts/fetch_finqa.py dev

.venv/bin/finreason-eval-retrieval \
  --dataset raw_data/finqa/dev.json \
  --method bm25 \
  --top-k 10 \
  --ks 1,3,5,10

.venv/bin/finreason-eval-retrieval \
  --dataset raw_data/finqa/dev.json \
  --method hybrid \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --top-k 10 \
  --ks 1,3,5,10 \
  --candidate-k 40

.venv/bin/finreason-eval-retrieval \
  --dataset raw_data/finqa/dev.json \
  --method hybrid \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --reranker-model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a \
  --top-k 10 \
  --ks 1,3,5,10 \
  --candidate-k 40

.venv/bin/finreason-audit-executor \
  --dataset raw_data/finqa/dev.json
```

Generated data, model caches, and run artifacts are intentionally ignored by
Git. Each report has a neighbouring manifest containing its deterministic
`experiment_id`, unique `run_id`, dataset and source-code SHA-256 values, Git
revision/dirty status, exact configuration, and headline metrics. Reports and
manifests use exclusive creation, so reruns cannot silently overwrite history.

## Measured development-set results

Dataset: official FinQA `dev.json`, 883 questions and 1,513 labelled supporting
facts. Source revision and checksum are pinned by `scripts/fetch_finqa.py`.
Exact machine-readable metrics are retained in
[`results_snapshot.json`](results_snapshot.json).

| Retriever | R@1 | R@3 | R@5 | R@10 | MRR@10 | mAP@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 35.62% | 56.16% | 68.24% | 84.80% | 61.96% | 54.35% |
| BM25 + MiniLM dense + RRF | 41.64% | 67.14% | 78.01% | 89.18% | 70.90% | 62.88% |
| Hybrid + MS MARCO MiniLM reranker | **43.94%** | **69.89%** | **81.08%** | **92.34%** | **73.89%** | **66.43%** |

The reranked hybrid run improves Recall@5 by **12.84 percentage points** and
MRR@10 by **11.93 points** over this BM25 implementation. Mean retrieval latency
in the sequential local run was 66.63 ms per query (p95 106.37 ms), including
per-question embedding and reranking. These are evidence-retrieval results, not
end-to-end answer accuracy.

The largest intended gain is on table-only questions (542 examples): Recall@5
rose from **63.65%** with BM25 to **80.26%** with the reranked hybrid pipeline.
Text-only Recall@5 was 92.89%, so table retrieval remains the more important
residual bottleneck.

Final sequential local run IDs were
`20260916T160112334135Z-f62d0e074f07-e069fc05` (BM25),
`20260916T160219579244Z-6ee2d90d1fe3-c1b983ca` (hybrid), and
`20260916T160335135470Z-e4d2977bee08-36d9577b` (hybrid + reranker). The executor
audit run was `20260916T160341278292Z-865bd9143226-bea463f8`. All four record
source-code SHA-256
`83612aee92fc6f84c8f243d73f6df1f2146566f06595be6a03dc256291d0eb47` and
honestly mark the worktree as dirty because this extension was benchmarked for
review before a local commit. Run artifacts are local-only; the snapshot,
commands, hashes, and manifests make the result auditable and regenerable.

The safe executor audit supplied the official gold program only to test DSL
coverage: **883/883 programs executed and matched `qa.exe_ans`** within the
documented tolerance. This is an oracle diagnostic, not a model result.

## Run a fixed-model end-to-end evaluation

Follow [`e2e_protocol.md`](e2e_protocol.md) before opening the test split. The
first development benchmark is designed for the pinned Qwen3-4B base revision
shown below because the historical LoRA weights are not available in this
repository and their output protocol is not compatible with the structured
workflow.

Start an OpenAI-compatible model server separately, then configure the client:

```bash
export FINREASON_MODEL_BASE_URL=http://localhost:8001/v1
export FINREASON_MODEL_NAME=Qwen/Qwen3-4B
export FINREASON_MODEL_REVISION=1cfa9a7208912126459214e8b04321603b3df60c
export FINREASON_TOKENIZER_REVISION="$FINREASON_MODEL_REVISION"
export FINREASON_MODEL_TEMPERATURE=0
export FINREASON_MODEL_TOP_P=1
export FINREASON_MODEL_MAX_TOKENS=512
export FINREASON_MODEL_SEED=42
export FINREASON_MODEL_ENABLE_THINKING=false
```

Run the development split with the measured retrieval configuration:

```bash
.venv/bin/finreason-eval-reasoning \
  --dataset raw_data/finqa/dev.json \
  --method hybrid \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --reranker-model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a \
  --top-k 5 \
  --candidate-k 40 \
  --max-repairs 1
```

The primary `accuracy` uses the strict full-split denominator: abstentions,
malformed output, invalid citations and failed execution all count as wrong.
Percentage auto-scaling is disabled unless `--percent-auto-scale` is supplied,
so the default reuses the earlier `accuracy_base` numeric tolerance without its
ambiguous gold-field fallback.
The primary execution-accuracy scorer uses `qa.exe_ans`, the output of FinQA's
gold program, rather than the inconsistently rounded/scaled display field
`qa.answer`; it falls back to the display answer only when no execution answer
exists. Each per-query record states the selected `gold_source`.
The report also retains coverage, answered-only accuracy, a 95% Wilson interval,
program/execution success, citation precision/recall/F1, repair outcomes,
model-reported token use and mean/p50/p95 latency. Per-query latency starts
before chunk and retrieval-index construction; one-time semantic-model loading
and external model-server startup are excluded from that distribution.

## Run the service

Retrieval works without a model endpoint:

```bash
FINREASON_DATASET_PATH=raw_data/finqa/dev.json \
  .venv/bin/uvicorn finreason.api:create_app --factory --port 8000

curl -s http://localhost:8000/v1/retrieve \
  -H 'content-type: application/json' \
  -d '{"example_id":"V/2008/page_17.pdf-1","top_k":5}'
```

For `/v1/reason`, point the service at a vLLM or other OpenAI-compatible chat
endpoint:

```bash
export FINREASON_MODEL_BASE_URL=http://localhost:8001/v1
export FINREASON_MODEL_NAME=Qwen/Qwen3-4B
export FINREASON_MODEL_API_KEY=local-token
```

The API returns the program, evidence citations, answer, latency, trace ID,
repair count, and every workflow stage. If no model is configured, the reason
endpoint returns HTTP 503 instead of silently substituting gold data or a fake
answer.

The packaged API is a portfolio/reference service: it runs as a non-root
container and enforces request, timeout and model-response limits, but it does
not implement authentication or rate limiting. Do not expose it directly to the
public internet without an authenticated gateway. Compose binds it to
`127.0.0.1` by default.

## Evaluation boundaries

- Retrieval labels are used only after ranking.
- Retrieval evaluation rejects empty or unmappable evidence labels (apart from
  FinQA's documented `text_-1` anomaly) instead of silently reporting a false
  low score for context-only flattened data.
- Gold programs are used only by the separately named executor audit.
- Current measured results cover retrieval and deterministic execution. The
  end-to-end generator benchmark command is implemented, but the score still
  requires a live pinned model endpoint and is not claimed here.
- The development split is for engineering iteration; final model selection
  should be frozen before a single test-set run.
- The generic MiniLM reranker is retained because its full 883-question
  ablation improved every reported retrieval metric; a framework or model name
  without this comparison would not be treated as a result.
