# Result Snapshot (Report-Oriented)

This file records compact metrics that map to the main report tables.

## Baseline Reference

- Task: FinQA test
- Evaluator: math-verify
- Primary protocol: oracle + no-thinking + final-answer tag

## Key outcome summary

- Reported Qwen3-4B legacy strict numeric accuracy (`accuracy_base`):
  **24.93%** zero-shot and **32.43%** for the best reported LoRA SFT run
  (**+7.50 percentage points**).
- These values are not the separate `accuracy_mathverify` field. Keep metric
  names explicit when comparing or citing runs.

- Best zero-shot baseline in the verified matrix is below full task saturation.
- Stage-1 SFT outcomes are sensitive to prompt protocol alignment.
- The repository includes scripts for:
  - baseline matrix reporting
  - clean strict SFT matrix
  - prompt-aligned re-evaluation
  - 8B full-steps retrain and dual-protocol evaluation

The historical per-run `summary.json` files are not bundled in this portfolio
mirror. Newly generated runs can be read through
`stage1/scripts/summary_utils.py`.
