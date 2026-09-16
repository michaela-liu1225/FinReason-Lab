"""Leakage-safe retrieval metrics and dataset-level aggregation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

from .schema import EvidenceChunk, RetrievalHit, RetrievalMetrics

RetrievedItem: TypeAlias = str | EvidenceChunk | RetrievalHit


@dataclass(frozen=True)
class AggregateRetrievalMetrics:
    """Macro-averaged retrieval metrics for a dataset."""

    recall_at_k: Mapping[int, float]
    mean_reciprocal_rank: float
    mean_average_precision: float
    query_count: int
    total_retrieved: int
    total_relevant: int


@dataclass(frozen=True)
class RetrievalEvaluation:
    """Per-query measurements together with their dataset aggregate."""

    per_query: Mapping[str, RetrievalMetrics]
    aggregate: AggregateRetrievalMetrics


@dataclass(frozen=True)
class OracleGap:
    """Difference between an oracle-evidence score and a deployable score."""

    oracle_score: float
    system_score: float
    absolute_gap: float
    relative_gap: float | None
    oracle_retention: float | None


def _item_id(item: RetrievedItem) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, RetrievalHit):
        return item.chunk.evidence_id
    if isinstance(item, EvidenceChunk):
        return item.evidence_id
    raise TypeError("Retrieved items must be evidence-id strings, EvidenceChunk, or RetrievalHit")


def _unique_ids(items: Iterable[RetrievedItem]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        evidence_id = _item_id(item)
        if evidence_id not in seen:
            result.append(evidence_id)
            seen.add(evidence_id)
    return tuple(result)


def _normalize_ks(ks: Iterable[int]) -> tuple[int, ...]:
    values = tuple(ks)
    if any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in values):
        raise ValueError("Every retrieval cutoff must be a positive integer")
    return tuple(sorted(set(values)))


def evaluate_retrieval(
    retrieved: Iterable[RetrievedItem],
    relevant_ids: Iterable[str],
    *,
    ks: Iterable[int] = (1, 3, 5),
) -> RetrievalMetrics:
    """Calculate Recall@k, reciprocal rank, and average precision.

    Reciprocal rank and average precision are calculated over the supplied
    ranked list.  If callers supply only the first ``N`` results, they must be
    reported as MRR@N and AP@N rather than as unbounded metrics.

    Duplicate retrieved ids count only at their first rank, preventing malformed
    result lists from inflating metrics.  An example with no relevance labels is
    defined to have zero for every metric.
    """

    cutoffs = _normalize_ks(ks)
    ranked_ids = _unique_ids(retrieved)
    relevant = frozenset(str(item) for item in relevant_ids)
    relevant_count = len(relevant)

    if not relevant_count:
        return RetrievalMetrics(
            recall_at_k={k: 0.0 for k in cutoffs},
            reciprocal_rank=0.0,
            average_precision=0.0,
            retrieved=len(ranked_ids),
            relevant=0,
        )

    recall_at_k = {k: len(relevant.intersection(ranked_ids[:k])) / relevant_count for k in cutoffs}

    first_relevant_rank = next(
        (rank for rank, item in enumerate(ranked_ids, start=1) if item in relevant),
        None,
    )
    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(ranked_ids, start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank
    average_precision = precision_sum / relevant_count

    return RetrievalMetrics(
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
        average_precision=average_precision,
        retrieved=len(ranked_ids),
        relevant=relevant_count,
    )


def aggregate_retrieval_metrics(
    metrics: Iterable[RetrievalMetrics],
    *,
    ks: Iterable[int] | None = None,
) -> AggregateRetrievalMetrics:
    """Macro-average per-query retrieval metrics.

    When ``ks`` is omitted, the union of cutoffs present in the input is used.
    Passing ``ks`` is useful for producing a stable zero-query report.
    """

    measurements = tuple(metrics)
    cutoffs = (
        _normalize_ks(ks)
        if ks is not None
        else tuple(sorted({k for measurement in measurements for k in measurement.recall_at_k}))
    )
    query_count = len(measurements)
    if query_count == 0:
        return AggregateRetrievalMetrics(
            recall_at_k={k: 0.0 for k in cutoffs},
            mean_reciprocal_rank=0.0,
            mean_average_precision=0.0,
            query_count=0,
            total_retrieved=0,
            total_relevant=0,
        )

    missing = [
        k for k in cutoffs if any(k not in measurement.recall_at_k for measurement in measurements)
    ]
    if missing:
        missing_text = ", ".join(str(k) for k in missing)
        raise ValueError(f"Per-query metrics are missing requested cutoffs: {missing_text}")

    return AggregateRetrievalMetrics(
        recall_at_k={
            k: sum(measurement.recall_at_k[k] for measurement in measurements) / query_count
            for k in cutoffs
        },
        mean_reciprocal_rank=sum(measurement.reciprocal_rank for measurement in measurements)
        / query_count,
        mean_average_precision=sum(measurement.average_precision for measurement in measurements)
        / query_count,
        query_count=query_count,
        total_retrieved=sum(measurement.retrieved for measurement in measurements),
        total_relevant=sum(measurement.relevant for measurement in measurements),
    )


def evaluate_retrieval_dataset(
    retrieved_by_query: Mapping[str, Sequence[RetrievedItem]],
    relevant_by_query: Mapping[str, Sequence[str]],
    *,
    ks: Iterable[int] = (1, 3, 5),
) -> RetrievalEvaluation:
    """Evaluate every labelled query, treating missing result lists as empty.

    Results for unknown query ids are rejected because silently dropping a
    train/test id mismatch can make an experiment look better than it is.
    """

    cutoffs = _normalize_ks(ks)
    unknown_queries = set(retrieved_by_query) - set(relevant_by_query)
    if unknown_queries:
        names = ", ".join(sorted(unknown_queries))
        raise ValueError(f"Retrieved results contain unknown query ids: {names}")

    per_query = {
        query_id: evaluate_retrieval(retrieved_by_query.get(query_id, ()), relevant_ids, ks=cutoffs)
        for query_id, relevant_ids in sorted(relevant_by_query.items())
    }
    return RetrievalEvaluation(
        per_query=per_query,
        aggregate=aggregate_retrieval_metrics(per_query.values(), ks=cutoffs),
    )


def compute_oracle_gap(*, oracle_score: float, system_score: float) -> OracleGap:
    """Describe how much performance is lost without oracle evidence.

    Scores can be proportions, percentages, or another common scale.  Values are
    intentionally not clamped: a negative gap truthfully records the uncommon
    case where the deployable system outperforms the oracle-context run.
    """

    oracle = float(oracle_score)
    system = float(system_score)
    if not isfinite(oracle) or not isfinite(system):
        raise ValueError("Oracle and system scores must be finite")
    gap = oracle - system
    if oracle == 0.0:
        relative_gap = None
        retention = None
    else:
        relative_gap = gap / oracle
        retention = system / oracle
    return OracleGap(
        oracle_score=oracle,
        system_score=system,
        absolute_gap=gap,
        relative_gap=relative_gap,
        oracle_retention=retention,
    )


# Short alias for reporting scripts.
oracle_gap = compute_oracle_gap


__all__ = [
    "AggregateRetrievalMetrics",
    "OracleGap",
    "RetrievalEvaluation",
    "aggregate_retrieval_metrics",
    "compute_oracle_gap",
    "evaluate_retrieval",
    "evaluate_retrieval_dataset",
    "oracle_gap",
]
