# FinReason-Lab

Thinking-aware baselines and parameter-efficient post-training for financial numerical reasoning.

FinReason-Lab is a six-person UCL COMP0087 Natural Language Processing team project. It studies whether low-data supervised fine-tuning can improve Qwen3 models on [FinQA](https://github.com/czyssrs/FinQA) under a controlled evaluation protocol.

> **Portfolio mirror:** this repository presents Yuxin Liu's contribution to the team project and preserves the original reproducibility code. It is not presented as solo work. The canonical team repository is [Quarkgluonmixture/FinQA](https://github.com/Quarkgluonmixture/FinQA).

## Independent extension: retrieval and bounded tool reasoning

The `feature/rag-agent-upgrade` work is Yuxin Liu's independent post-course
extension. It adds table-aware evidence retrieval, hybrid BM25/dense ranking,
an allow-listed Decimal program executor, bounded repair/abstention, a FastAPI
service, Docker packaging, tests and CI. It is kept distinct from the original
six-person submission and does not relabel the team result as solo work.

On the official 883-question FinQA development split, the BM25 + MiniLM dense +
RRF + cross-encoder reranking run reached **81.08% evidence Recall@5**, versus **68.24%**
for the new BM25 baseline (+12.84 percentage points). These are retrieval
results, not end-to-end answer accuracy. The safe executor separately reproduced
**883/883** gold-program execution answers; that is an explicitly oracle
diagnostic.

See [`docs/rag/README.md`](docs/rag/README.md) for the architecture, commands,
measured results, provenance safeguards and remaining evaluation boundary.

## Headline result

Under the primary **oracle-evidence, no-thinking, final-answer-tag** evaluation setup on the full FinQA test set (**n = 1,147**), the best reported Qwen3-4B LoRA SFT run improved the legacy strict numeric metric (`accuracy_base`) from **24.93%** zero-shot to **32.43%**: an absolute gain of **7.50 percentage points**. The pipeline also records a separate `accuracy_mathverify` field; the headline values should not be relabelled as that metric.

| Model | Adaptation | `accuracy_base` | Absolute change |
| --- | --- | ---: | ---: |
| Qwen3-4B | Zero-shot | 24.93% | — |
| Qwen3-4B | LoRA SFT (best reported run) | **32.43%** | **+7.50 pp** |

“Oracle evidence” means that the relevant evidence from the financial report is supplied to the model. It does not mean that the answer or reasoning program is supplied.

## Research scope

- Compared Qwen3-4B and Qwen3-8B on FinQA using a fixed answer format and verifier.
- Applied parameter-efficient SFT: LoRA for Qwen3-4B and QLoRA for Qwen3-8B.
- Evaluated data-size ablations at 250, 1,000, and the strict-clean full split.
- Examined training-length effects and changes in residual error categories.
- Used a strict-clean FinQA training split of **3,277** examples and a development split of **475** examples.

The `25,185` count in the cleaning metadata is the size of the merged source pool before source filtering and validation; it is **not** the number of examples used to train the reported FinQA models.

## Yuxin Liu's contribution

As described in Yuxin Liu's CV:

- Fine-tuned Qwen3-4B and Qwen3-8B with LoRA/QLoRA supervised fine-tuning.
- Conducted data-size and training-length ablations and classified residual errors.

## Team

This work was completed by:

- Yike Zhang
- Jiaming Wei
- Yuhao Wang
- Yuxin Liu
- Jiaqi Wang
- Victor Lang

## Repository map

```text
.
├── finreason/        # Independent RAG, retrieval, tools, workflow, and API extension
├── tests/            # Unit and API tests for the extension
├── finqa_baseline/   # Zero-shot and adapter evaluation with math-verify
├── stage1/           # SFT, LoRA/QLoRA, data preparation, and orchestration
└── docs/
    ├── paper/        # Report-availability note
    ├── rag/          # Extension architecture and measured retrieval results
    └── repro/        # Protocol, experiment matrix, and reproduction notes
```

## Quick start

Run each command block from the repository root.

For the independent retrieval/tool extension, use:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[api,semantic,dev]"
python scripts/fetch_finqa.py dev
.venv/bin/finreason-eval-retrieval \
  --dataset raw_data/finqa/dev.json \
  --method bm25 --top-k 10 --ks 1,3,5,10
```

The original team experiment environments remain documented below.

### 1. Create the environments

Python 3.10 or later is recommended.

```bash
cd stage1
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

cd ../finqa_baseline
bash setup.sh
```

### 2. Run the packaged sanity check

The repository includes a small debug sample for testing the training-to-checkpoint path without the full experiment data.

```bash
cd stage1
bash scripts/run_debug.sh
```

### 3. Run the primary no-thinking evaluation protocol

The command below evaluates the Qwen3-4B zero-shot baseline with oracle evidence. It may download the model and FinQA data and requires suitable accelerator memory.

```bash
cd finqa_baseline
.venv/bin/python eval_finqa.py \
  --model_name Qwen/Qwen3-4B \
  --setting oracle \
  --split test \
  --no-enable_thinking \
  --evaluator math_verify \
  --prompt_protocol stage1_train_text \
  --answer_format final_answer_tag \
  --final_answer_tag FINAL_ANSWER \
  --cache_dir "${HF_HOME:-$HOME/.cache/huggingface}"
```

To evaluate an SFT adapter, add `--adapter_path /path/to/checkpoint-last` while keeping the same protocol flags.

## Main experiment entrypoints

- `finqa_baseline/run_verification_matrix.sh` — zero-shot evaluation matrix
- `stage1/scripts/run_train_eval_matrix_clean_strict.sh` — strict-clean LoRA/QLoRA SFT matrix
- `stage1/scripts/run_eval_matrix_prompt_aligned.sh` — prompt-aligned adapter evaluation
- `stage1/scripts/run_8b_fullsteps_train_eval.sh` — extended Qwen3-8B training and evaluation
- `stage1/scripts/analyze_error_shift.py` — residual-error shift analysis

See [`docs/repro/README.md`](docs/repro/README.md) for the reproduction contract and limitations.

## Artifact availability and limitations

This mirror contains source code, experiment orchestration, subset IDs, cleaning summaries, and a small debug sample. It does **not** package:

- full reconstructed training and development JSONL files;
- model checkpoints or adapters;
- runtime logs and per-run `summary.json` result files; or
- the submitted report PDF.

The report is not redistributed here because the submitted copy contains team members' institutional email addresses. The result above is therefore a report-backed project result, not a claim that it can be independently verified from bundled checkpoints and outputs alone. Reproducing it requires reconstructing the data, downloading the base models, training the adapters, and rerunning evaluation. When comparing against the reported headline, read `accuracy_base` from the generated summary rather than the primary `accuracy` field.

## Attribution and license

This personal mirror retains the team attribution and links to the [canonical upstream repository](https://github.com/Quarkgluonmixture/FinQA). Repository code is distributed under the included Apache License 2.0; upstream datasets and models remain subject to their own licenses and terms. In particular, consult the original [FinQA repository](https://github.com/czyssrs/FinQA) and the Qwen model cards before reuse.
