# Fixed-model end-to-end evaluation protocol

This protocol defines the first end-to-end FinReason-Lab benchmark. Its primary
question is whether retrieval plus bounded tool reasoning improves final-answer
accuracy when the generator is held fixed. Retrieval quality and executor
coverage remain diagnostics; neither is a substitute for answer accuracy.

## Phase-one model

The original Qwen3-4B LoRA weights are not present in this repository. They
therefore cannot be loaded, hashed, or independently evaluated. In addition,
that adapter was trained for the earlier final-answer-tag protocol, whereas the
new workflow requires structured `{program, citations}` output. It must not be
presented as compatible without a separate conversion or retraining study.

Phase one consequently uses the unadapted `Qwen/Qwen3-4B` base model at revision
`1cfa9a7208912126459214e8b04321603b3df60c`, recovered from the historical
experiment configuration. Before the first development run, record immutable
model and tokenizer revisions in the freeze manifest; a moving branch or
`main` revision is not acceptable. The adapter field must be explicitly
`null`. Every compared retrieval/workflow variant uses this same model and
tokenizer. Set `FINREASON_MODEL_ENABLE_THINKING=false`; the OpenAI-compatible
request records this as `chat_template_kwargs.enable_thinking=false` so Qwen3
returns only the frozen structured-output protocol rather than an additional
thinking trace.

## Data isolation

- Pin the official FinQA source revision, file SHA-256 values, and example IDs
  for `train`, `dev`, and `test`. Do not move examples between splits.
- `train` is the only split allowed for fitting or selecting demonstrations. If
  the fixed-model phase uses no demonstrations, record that fact.
- `dev` may be used to choose prompts, chunking, retrieval parameters,
  validation rules, repair policy, and the final system variant. Development
  scores must always be labelled as development results.
- `test` is sealed until all choices and hashes below are frozen. The runtime
  receives only the question and its report-derived chunks. Gold answers,
  programs, execution answers, and evidence labels are exposed only to the
  offline scorer after predictions have been written and sealed.
- Indexing the report associated with a test question is allowed because that
  report is the task input. Chunks may contain only report text and tables with
  provenance. QA fields and gold annotations must never enter the index,
  retriever, prompt, generator, validator, executor, or repair loop.

## Freeze gate

Create and commit a freeze manifest before opening `test`. It must contain:

- dataset source revision, split hashes, example count, and ordered ID hash;
- clean Git commit and source-code hash;
- model, tokenizer, adapter, embedding-model, and reranker revisions or hashes;
- full prompt text and response schema hashes;
- chunking options, BM25 settings, dense/RRF settings, `candidate_k`, `top_k`,
  reranker depth, context ordering, and token budget;
- decoding settings, including chat template, thinking mode, temperature,
  `top_p`, maximum new tokens, stop strings, and random seed;
- citation and grounding rules, executor/scorer versions, timeout policy,
  abstention rules, maximum repairs, and repair prompt;
- dependency lock or environment export, inference backend, hardware, and
  deterministic-runtime settings.

Floating model aliases, omitted seeds, or configuration supplied only through
unrecorded environment variables fail the gate. Changing any frozen item after
viewing test output creates a new exploratory experiment, not a replacement
official result.

## Comparisons and ablations

Run every system on the same examples, in the same order, with the same frozen
generator, response schema, context budget, decoding settings, and scorer.

| Variant | Purpose |
| --- | --- |
| No retrieval | Fixed-model lower baseline with no report evidence |
| BM25 | Sparse RAG baseline |
| BM25 + dense + RRF | Hybrid-retrieval contribution |
| Hybrid + reranker | Reranking contribution |
| Full bounded workflow | Retrieval, structured program, validation, execution, verification, and repair/abstention |
| Oracle evidence | Gold-evidence upper bound, always labelled `oracle` |

At minimum, ablate table-aware chunks, dense retrieval, reranking, tool
execution, citation/grounding validation, verifier, and repair. Also compare
the full workflow with repair disabled and with abstention disabled. Change one
factor at a time; otherwise label the comparison as a system comparison rather
than a component ablation. Oracle evidence and gold programs are prohibited
from every non-oracle variant.

## Scoring

The primary metric is **full-split end-to-end answer accuracy**:

```text
accuracy = correct final answers / all examples in the split
```

The final-answer parser and numeric comparator must be frozen on `dev`. Reuse
the earlier project's strict numeric tolerance (`atol=1e-3`, `rtol=1e-3`) with
no automatic percent rescaling, but use `qa.exe_ans` consistently as the
execution-accuracy target. The human-facing `qa.answer` field may be reported
separately but must not replace that target because its percentage scale and
rounding are inconsistent. Score only the workflow's declared final answer;
never recover a more favourable number from reasoning text. Missing answers,
abstentions, invalid structured output, invalid citations, unsupported
programs, execution or verification failures, timeouts, and exhausted repairs
all count as incorrect in the primary denominator.

Report the following secondary metrics without promoting them to the primary
score:

- coverage (`answered / all`) and answered-only accuracy, always together;
- citation precision, recall, and F1 against gold evidence after generation;
- valid-structure rate, executable-program rate, program exact/equivalent
  accuracy, repair rate, and abstention rate;
- retrieval Recall@k and MRR@k for diagnosis only;
- end-to-end and per-stage latency, with mean, median, p95, timeout policy, and
  warm-up treatment stated.

Report a 95% Wilson interval for each system's accuracy. For planned system
comparisons, also report a paired 95% bootstrap interval for the accuracy
difference and use a paired test such as McNemar's test. Disclose any correction
for multiple comparisons. Include slices by text/table evidence, operation
type, program length, and whether gold evidence was retrieved.

## Artifacts and failure accounting

The implemented development evaluator completes the entire split's inference
pass before it reads any gold targets. Before an official test release, add an
external prediction-only gate that writes and seals one immutable JSONL record
per example before invoking the scorer. Each record must include the example
ID, variant, retrieved evidence IDs/ranks and scores, rendered-context hash and
token count, raw model output, parsed program and citations,
validation/execution/verification trace, repair count, final answer or
abstention reason, stage latencies, and failure category. The test scorer may
append the gold value and correctness only after the prediction file is sealed.

Use mutually exclusive primary failure categories: retrieval miss, reranking
loss, context truncation, malformed structure, invalid citation, ungrounded
number, wrong program with sufficient evidence, executor failure, verifier
rejection, repair regression, timeout, abstention, wrong executed answer, or
scoring/normalization failure.

Each run also receives an exclusively created manifest containing a unique run
ID, frozen configuration, hashes and revisions, command, environment, start/end
time, aggregate metrics, and the SHA-256 of its per-example artifact. Preserve
raw artifacts for all baselines and ablations, including failed runs.

## Test release rule

Select one primary system on `dev`, record the selection rule, and freeze the
manifest before test labels are accessed. Run each preregistered test variant
once and publish all of them. A rerun is permissible only for a documented
infrastructure failure identified without inspecting scored outputs; otherwise
it is exploratory and must not replace the first result.

Recall@k measures whether evidence was retrieved. The 883/883 oracle executor
audit measures DSL coverage when the gold program is supplied. Neither is
model-generated, end-to-end answer accuracy, and neither may be described as
such in the README, CV, report, or project page.
