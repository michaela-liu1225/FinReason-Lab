"""Leakage-safe end-to-end evaluation for retrieval-augmented FinQA reasoning."""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from decimal import Decimal
from math import ceil, isfinite, sqrt
from pathlib import Path
from statistics import mean
from typing import Any

from .chunking import chunk_example
from .data import load_finqa
from .executor import ProgramError, parse_decimal, parse_program
from .retrieval import (
    BM25Retriever,
    HybridRetriever,
    SentenceTransformerCrossEncoder,
    SentenceTransformerEncoder,
)
from .schema import FinQAExample, RetrievalHit
from .workflow import FinReasonWorkflow, StructuredGenerator


def _non_negative_decimal(value: float, *, name: str) -> Decimal:
    if not isinstance(value, int | float) or isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = Decimal(str(value))
    if result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _answer_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    try:
        return parse_decimal(str(value))
    except ProgramError:
        return None


def _is_close(left: Decimal, right: Decimal, *, atol: Decimal, rtol: Decimal) -> bool:
    tolerance = max(atol, rtol * max(abs(left), abs(right)))
    return abs(left - right) <= tolerance


def score_answer(
    prediction: Any,
    gold: Any,
    *,
    question: str = "",
    atol: float = 0.001,
    rtol: float = 0.001,
    percent_auto_scale: bool = False,
) -> bool:
    """Score one answer with exact categorical and tolerant Decimal semantics.

    The optional percent recovery matches the earlier FinQA evaluator: only a
    percentage-like question or gold answer may recover a prediction expressed
    on the 0--100 scale. Empty predictions never score as correct.
    """

    absolute_tolerance = _non_negative_decimal(atol, name="atol")
    relative_tolerance = _non_negative_decimal(rtol, name="rtol")
    if prediction is None or gold is None:
        return False
    predicted_text = str(prediction).strip()
    gold_text = str(gold).strip()
    if not predicted_text or not gold_text:
        return False
    if _NON_FINITE_NUMBER_RE.fullmatch(predicted_text) or _NON_FINITE_NUMBER_RE.fullmatch(
        gold_text
    ):
        return False

    predicted_number = _answer_decimal(prediction)
    gold_number = _answer_decimal(gold)
    if predicted_number is None or gold_number is None:
        return predicted_text.casefold() == gold_text.casefold()

    if _is_close(
        predicted_number,
        gold_number,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    ):
        return True
    percentage_like = "%" in gold_text or any(
        marker in question.casefold() for marker in ("percent", "percentage", "%")
    )
    return bool(
        percent_auto_scale
        and percentage_like
        and abs(gold_number) < 1
        and abs(predicted_number) > 1
        and _is_close(
            predicted_number / Decimal(100),
            gold_number,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    )


def _gold_target(example: FinQAExample) -> tuple[Any, str]:
    """Select FinQA's program-execution target for the primary metric.

    ``qa.answer`` is a human-facing display field with inconsistent percent
    scales and rounding in the official data.  ``qa.exe_ans`` is the result of
    the gold program and is therefore the correct target for execution accuracy.
    """

    execution_answer = example.metadata.get("execution_answer")
    if execution_answer is not None and str(execution_answer).strip():
        return execution_answer, "execution_answer"
    return example.answer, "answer_fallback"


class _CandidateKRetriever:
    """Bind HybridRetriever's candidate depth to the workflow's simple protocol."""

    def __init__(self, retriever: HybridRetriever, candidate_k: int) -> None:
        self.retriever = retriever
        self.candidate_k = candidate_k

    def retrieve(self, query: str, *, k: int = 5) -> tuple[RetrievalHit, ...]:
        return self.retriever.retrieve(query, k=k, candidate_k=self.candidate_k)


class _RecordingRetriever:
    """Record exactly the hits observed by the generator for per-query artifacts."""

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever
        self.last_hits: tuple[RetrievalHit, ...] = ()

    def retrieve(self, query: str, *, k: int = 5) -> tuple[RetrievalHit, ...]:
        self.last_hits = tuple(self.retriever.retrieve(query, k=k))
        return self.last_hits


class _SetupErrorRetriever:
    """Turn per-example index/setup failures into a scored abstention."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def retrieve(self, query: str, *, k: int = 5) -> tuple[RetrievalHit, ...]:
        raise RuntimeError(
            f"retriever setup failed ({type(self.error).__name__}): {self.error}"
        ) from self.error


def _inference_view(example: FinQAExample) -> FinQAExample:
    """Remove every supervision field before retrieval or generation."""

    return FinQAExample(
        example_id=example.example_id,
        report_id=example.report_id,
        question=example.question,
        pre_text=example.pre_text,
        table=example.table,
        post_text=example.post_text,
        answer="",
        program="",
        gold_evidence_ids=(),
        gold_evidence_text={},
        metadata={},
    )


def _gold_kind(gold_evidence_ids: Sequence[str]) -> str:
    prefixes = {evidence_id.split("_", 1)[0] for evidence_id in gold_evidence_ids}
    if prefixes == {"text"}:
        return "text"
    if prefixes == {"table"}:
        return "table"
    return "mixed" if prefixes else "unlabelled"


_CELL_ID_RE = re.compile(r"^(table_[0-9]+)_cell_[0-9]+$")
_NON_FINITE_NUMBER_RE = re.compile(
    r"^[+-]?(?:nan|snan|inf(?:inity)?)$",
    flags=re.IGNORECASE,
)
_SENSITIVE_CONFIG_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "api_token",
        "access_token",
        "auth_token",
        "bearer_token",
        "authorization",
        "password",
        "secret",
        "token",
    }
)


def _redact_model_config(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact common credential fields before artifact writing."""

    normalized_key = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_") if key else ""
    if normalized_key in _SENSITIVE_CONFIG_KEYS or normalized_key.endswith(
        ("_api_key", "_password", "_secret", "_access_token")
    ):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            str(item_key): _redact_model_config(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_redact_model_config(item) for item in value]
    return value


def _citation_label(evidence_id: str) -> str:
    match = _CELL_ID_RE.fullmatch(evidence_id)
    return match.group(1) if match else evidence_id


def _citation_scores(
    predicted: Sequence[str],
    gold: Sequence[str],
) -> tuple[int, int, int, float, float, float]:
    predicted_set = {_citation_label(item) for item in predicted}
    # ``text_-1`` is the official FinQA unaddressable-evidence anomaly. It
    # cannot be produced by the chunker and must not depress citation recall.
    gold_set = {_citation_label(item) for item in gold if item != "text_-1"}
    true_positives = len(predicted_set & gold_set)
    precision = true_positives / len(predicted_set) if predicted_set else 0.0
    recall = true_positives / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return true_positives, len(predicted_set), len(gold_set), precision, recall, f1


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _hit_dict(hit: RetrievalHit) -> dict[str, Any]:
    return {
        "evidence_id": hit.chunk.evidence_id,
        "kind": hit.chunk.kind,
        "rank": hit.rank,
        "score": hit.score,
        "component_scores": dict(hit.scores),
        "text": hit.chunk.text,
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return a deterministic 95% Wilson interval for a binomial rate."""

    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = z * sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _telemetry_cursor(generator: StructuredGenerator) -> int | None:
    try:
        value = generator.telemetry_count  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _telemetry_since(generator: StructuredGenerator, cursor: int | None) -> list[dict[str, Any]]:
    if cursor is None:
        return []
    try:
        values = generator.telemetry_since(cursor)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        return []
    return [asdict(value) for value in values if hasattr(value, "__dataclass_fields__")]


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    answered = sum(row["status"] == "answered" for row in rows)
    answered_correct = sum(bool(row["correct"]) for row in rows if row["status"] == "answered")
    program_successes = sum(bool(row["program_valid"]) for row in rows)
    execution_successes = sum(bool(row["execution_success"]) for row in rows)
    repair_triggered = sum(bool(row["repair_triggered"]) for row in rows)
    repair_answered = sum(bool(row["repair_answered"]) for row in rows)
    repair_successes = sum(bool(row["repair_success"]) for row in rows)
    citation_true_positives = sum(int(row["citation_true_positives"]) for row in rows)
    citation_predictions = sum(int(row["citation_predictions"]) for row in rows)
    citation_gold = sum(int(row["citation_gold"]) for row in rows)
    citation_precision = (
        citation_true_positives / citation_predictions if citation_predictions else 0.0
    )
    citation_recall = citation_true_positives / citation_gold if citation_gold else 0.0
    citation_f1 = (
        2 * citation_precision * citation_recall / (citation_precision + citation_recall)
        if citation_precision + citation_recall
        else 0.0
    )
    latencies = [float(row["latency_ms"]) for row in rows]
    stop_counts = Counter(str(row["stop_reason"]) for row in rows)
    error_counts = Counter(
        str(row["last_error_code"]) for row in rows if row.get("last_error_code") is not None
    )
    accuracy_ci_low, accuracy_ci_high = _wilson_interval(correct, total)
    model_calls = sum(int(row["attempts"]) for row in rows)
    model_completions_reported = sum(len(row.get("model_telemetry", ())) for row in rows)
    prompt_tokens = sum(
        int(call["prompt_tokens"])
        for row in rows
        for call in row.get("model_telemetry", ())
        if call.get("prompt_tokens") is not None
    )
    completion_tokens = sum(
        int(call["completion_tokens"])
        for row in rows
        for call in row.get("model_telemetry", ())
        if call.get("completion_tokens") is not None
    )
    total_tokens = sum(
        int(call["total_tokens"])
        for row in rows
        for call in row.get("model_telemetry", ())
        if call.get("total_tokens") is not None
    )
    model_latencies = [
        float(call["latency_ms"])
        for row in rows
        for call in row.get("model_telemetry", ())
        if call.get("latency_ms") is not None
    ]
    reported_models = Counter(
        str(call["model"])
        for row in rows
        for call in row.get("model_telemetry", ())
        if call.get("model")
    )
    finish_reasons = Counter(
        str(call["finish_reason"])
        for row in rows
        for call in row.get("model_telemetry", ())
        if call.get("finish_reason")
    )
    return {
        "examples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "accuracy_ci95_low": accuracy_ci_low,
        "accuracy_ci95_high": accuracy_ci_high,
        "accuracy_ci95_method": "wilson",
        "answered": answered,
        "coverage": answered / total if total else 0.0,
        "answered_correct": answered_correct,
        "answered_accuracy": answered_correct / answered if answered else 0.0,
        "program_successes": program_successes,
        "program_success_rate": program_successes / total if total else 0.0,
        "execution_successes": execution_successes,
        "execution_success_rate": execution_successes / total if total else 0.0,
        "citation_true_positives": citation_true_positives,
        "citation_predictions": citation_predictions,
        "citation_gold": citation_gold,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_f1": citation_f1,
        "repair_triggered": repair_triggered,
        "repair_rate": repair_triggered / total if total else 0.0,
        "repair_answered": repair_answered,
        "repair_answer_rate": repair_answered / repair_triggered if repair_triggered else 0.0,
        "repair_successes": repair_successes,
        "repair_success_rate": repair_successes / repair_triggered if repair_triggered else 0.0,
        "stop_counts": dict(sorted(stop_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "mean_latency_ms": mean(latencies) if latencies else 0.0,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "model_calls": model_calls,
        "model_completions_reported": model_completions_reported,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "mean_model_latency_ms": mean(model_latencies) if model_latencies else 0.0,
        "reported_models": dict(sorted(reported_models.items())),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
    }


def run_reasoning_evaluation(
    dataset_path: str | Path,
    *,
    generator: StructuredGenerator,
    method: str = "bm25",
    top_k: int = 5,
    limit: int | None = None,
    embedding_model: str | None = None,
    embedding_revision: str | None = None,
    reranker_model: str | None = None,
    reranker_revision: str | None = None,
    candidate_k: int = 40,
    include_cells: bool = False,
    max_repairs: int = 1,
    atol: float = 0.001,
    rtol: float = 0.001,
    percent_auto_scale: bool = False,
    model_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one fixed generator over a FinQA dataset without label leakage."""

    if method not in {"bm25", "hybrid"}:
        raise ValueError("method must be 'bm25' or 'hybrid'")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if max_repairs < 0:
        raise ValueError("max_repairs cannot be negative")
    _non_negative_decimal(atol, name="atol")
    _non_negative_decimal(rtol, name="rtol")
    if method == "hybrid" and not embedding_model:
        raise ValueError("hybrid reasoning evaluation requires embedding_model")
    if method == "bm25" and any(
        (embedding_model, embedding_revision, reranker_model, reranker_revision)
    ):
        raise ValueError("semantic model options require method='hybrid'")
    if embedding_revision and not embedding_model:
        raise ValueError("embedding_revision requires embedding_model")
    if reranker_revision and not reranker_model:
        raise ValueError("reranker_revision requires reranker_model")
    if model_config is not None and not isinstance(model_config, Mapping):
        raise TypeError("model_config must be a mapping or None")

    # Heavy semantic models are constructed once, outside the example loop.
    dense_encoder = (
        SentenceTransformerEncoder(embedding_model, revision=embedding_revision)
        if embedding_model is not None
        else None
    )
    reranker = (
        SentenceTransformerCrossEncoder(reranker_model, revision=reranker_revision)
        if reranker_model is not None
        else None
    )

    examples = load_finqa(dataset_path)
    if limit is not None:
        examples = examples[:limit]
    example_ids = [example.example_id for example in examples]
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("dataset contains duplicate example IDs")

    prediction_rows: dict[str, dict[str, Any]] = {}

    for example in examples:
        telemetry_cursor = _telemetry_cursor(generator)
        started = time.perf_counter()
        inference_example = _inference_view(example)
        try:
            chunks = chunk_example(inference_example, include_cells=include_cells)
            if method == "bm25":
                base_retriever: Any = BM25Retriever(chunks)
            else:
                hybrid = HybridRetriever(
                    chunks,
                    dense_encoder=dense_encoder,
                    reranker=reranker,
                )
                base_retriever = _CandidateKRetriever(hybrid, candidate_k)
        except Exception as exc:
            # A bad document/vector must count as one failed example rather
            # than silently shortening or aborting the strict denominator.
            base_retriever = _SetupErrorRetriever(exc)
        recording_retriever = _RecordingRetriever(base_retriever)
        workflow = FinReasonWorkflow(
            recording_retriever,
            generator,
            top_k=top_k,
            max_repairs=max_repairs,
        )

        result = workflow.run(inference_example.question)
        latency_ms = (time.perf_counter() - started) * 1000
        model_telemetry = _telemetry_since(generator, telemetry_cursor)

        program_valid = False
        if result.program:
            try:
                parse_program(result.program)
            except ProgramError:
                pass
            else:
                program_valid = True
        repair_triggered = result.repairs_used > 0

        row: dict[str, Any] = {
            "example_id": example.example_id,
            "retrieved_hits": [_hit_dict(hit) for hit in recording_retriever.last_hits],
            "prediction": result.answer,
            "status": result.status,
            "stop_reason": result.stop_reason,
            "program": result.program,
            "program_valid": program_valid,
            "execution_success": result.execution is not None,
            "citations": list(result.citations),
            "attempts": result.attempts,
            "repairs_used": result.repairs_used,
            "repair_triggered": repair_triggered,
            "last_error_code": result.last_error_code,
            "last_error_message": result.last_error_message,
            "trace": [asdict(event) for event in result.trace],
            "latency_ms": latency_ms,
            "model_telemetry": model_telemetry,
        }
        prediction_rows[example.example_id] = row

    # No supervision is read until every prediction has completed.  This makes
    # the inference/scoring boundary explicit even for development runs.
    per_query: dict[str, dict[str, Any]] = {}
    grouped_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        row = prediction_rows[example.example_id]
        gold_answer, gold_source = _gold_target(example)
        gold_evidence_ids = tuple(example.gold_evidence_ids)
        gold_kind = _gold_kind(gold_evidence_ids)
        correct = row["status"] == "answered" and score_answer(
            row["prediction"],
            gold_answer,
            question=example.question,
            atol=atol,
            rtol=rtol,
            percent_auto_scale=percent_auto_scale,
        )
        (
            citation_true_positives,
            citation_predictions,
            citation_gold,
            citation_precision,
            citation_recall,
            citation_f1,
        ) = _citation_scores(row["citations"], gold_evidence_ids)
        repair_triggered = bool(row["repair_triggered"])
        row.update(
            {
                "gold_kind": gold_kind,
                "gold": gold_answer,
                "gold_source": gold_source,
                "correct": bool(correct),
                "gold_evidence_ids": list(gold_evidence_ids),
                "citation_true_positives": citation_true_positives,
                "citation_predictions": citation_predictions,
                "citation_gold": citation_gold,
                "citation_precision": citation_precision,
                "citation_recall": citation_recall,
                "citation_f1": citation_f1,
                "repair_answered": repair_triggered and row["status"] == "answered",
                "repair_success": repair_triggered and bool(correct),
            }
        )
        per_query[example.example_id] = row
        grouped_rows[gold_kind].append(row)

    rows = list(per_query.values())
    return {
        "config": {
            "mode": "end_to_end_reasoning",
            "method": method,
            "top_k": top_k,
            "limit": limit,
            "embedding_model": embedding_model,
            "embedding_revision": embedding_revision,
            "reranker_model": reranker_model,
            "reranker_revision": reranker_revision,
            "candidate_k": candidate_k,
            "include_cells": include_cells,
            "max_repairs": max_repairs,
            "atol": atol,
            "rtol": rtol,
            "percent_auto_scale": percent_auto_scale,
            "gold_policy": "execution_answer_then_display_answer_fallback",
            "model_config": _redact_model_config(dict(model_config or {})),
        },
        "summary": _summarize(rows),
        "by_gold_kind": {
            kind: _summarize(kind_rows) for kind, kind_rows in sorted(grouped_rows.items())
        },
        "per_query": per_query,
    }


__all__ = ["run_reasoning_evaluation", "score_answer"]
