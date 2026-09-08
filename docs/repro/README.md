# Reproducibility guide

This document describes the reproducibility contract for the portfolio mirror of the UCL COMP0087 FinReason-Lab team project.

## Primary evaluation protocol

The reported headline result uses the following locked protocol:

- benchmark and split: FinQA test, **n = 1,147**;
- evidence setting: `oracle`;
- inference mode: **no-thinking**;
- output format: `FINAL_ANSWER` tag;
- evaluator invocation: `math_verify` (the run also records legacy numeric metrics);
- reported headline field: `accuracy_base`, not `accuracy_mathverify`.

Keep these settings fixed when comparing the zero-shot and SFT results. In particular, `eval_finqa.py` enables thinking by default, so the primary protocol must explicitly pass `--no-enable_thinking`.

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

For an adapter run, add `--adapter_path /path/to/checkpoint-last` to the same command.

## Environment

Python 3.10 or later is recommended.

### Training environment

```bash
cd stage1
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```

### Evaluation environment

```bash
cd finqa_baseline
bash setup.sh
```

The scripts default to `${HOME}/.cache/huggingface`. The cache can be redirected with `HF_CACHE_ROOT`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE`.

## Packaged sanity check

```bash
cd stage1
bash scripts/run_debug.sh
```

The included debug sample checks configuration loading, preprocessing, training, checkpoint writing, and dry-run inference. It is a pipeline sanity test, not a reproduction of the reported accuracy.

## Experiment entrypoints

```bash
# Zero-shot protocol matrix
cd finqa_baseline
bash run_verification_matrix.sh

# Strict-clean LoRA/QLoRA SFT matrix
cd ../stage1
bash scripts/run_train_eval_matrix_clean_strict.sh

# Prompt-aligned evaluation
bash scripts/run_eval_matrix_prompt_aligned.sh

# Extended Qwen3-8B run
bash scripts/run_8b_fullsteps_train_eval.sh
```

Residual-error analysis is implemented in `stage1/scripts/analyze_error_shift.py`.

## Data contract

The reported experiments use a strict-clean FinQA split with **3,277 training examples** and **475 development examples**. The tracked metadata also records `25,185` merged source records before source filtering and validation; that number is not the training-set size.

Only subset IDs, cleaning summaries, and a small debug sample are included. The full reconstructed JSONL files are intentionally omitted and must be regenerated before running the complete matrix.

## Results and artifact limits

The report records a Qwen3-4B improvement in the legacy strict numeric field `accuracy_base` from **24.93%** zero-shot to **32.43%** for the best reported LoRA SFT run (**+7.50 percentage points**) under the primary setup. `eval_finqa.py` also records `accuracy_mathverify` and `accuracy_adjusted`; these are distinct fields and must not be substituted for the headline metric.

This mirror does not include the model checkpoints, adapters, runtime logs, or per-run `summary.json` outputs used for that result. Evaluation code expects metrics in the latest record of `summary.json` (`runs[-1]` when a `runs` list exists, otherwise the top-level fields); `stage1/scripts/summary_utils.py` is the canonical reader.

Accordingly, the packaged files document the implementation and experimental design, but the headline number cannot be independently verified from bundled artifacts alone without retraining and reevaluation.

## Report and provenance

The submitted report PDF is not redistributed in this personal mirror because it contains institutional email addresses for the full team. The canonical project history remains at [Quarkgluonmixture/FinQA](https://github.com/Quarkgluonmixture/FinQA).
