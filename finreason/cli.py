"""Command-line evaluation and inspection entry points."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean
from typing import Any

from .chunking import chunk_example
from .data import load_finqa
from .evaluation import aggregate_retrieval_metrics, evaluate_retrieval
from .executor import ProgramError, execute_program, parse_decimal
from .manifest import build_run_manifest
from .retrieval import (
    BM25Retriever,
    HybridRetriever,
    SentenceTransformerCrossEncoder,
    SentenceTransformerEncoder,
)
from .schema import EvidenceChunk, FinQAExample, RetrievalHit, RetrievalMetrics

_ALLOWED_UNADDRESSABLE_GOLD_IDS = frozenset({"text_-1"})


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between zero and one")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _gold_kind(example: FinQAExample) -> str:
    prefixes = {evidence_id.split("_", 1)[0] for evidence_id in example.gold_evidence_ids}
    if prefixes == {"text"}:
        return "text"
    if prefixes == {"table"}:
        return "table"
    return "mixed" if prefixes else "unlabelled"


def _metrics_dict(metrics: RetrievalMetrics, ranking_depth: int) -> dict[str, Any]:
    return {
        "recall_at_k": {str(k): value for k, value in metrics.recall_at_k.items()},
        "reciprocal_rank_at_k": metrics.reciprocal_rank,
        "average_precision_at_k": metrics.average_precision,
        "ranking_depth": ranking_depth,
        "retrieved": metrics.retrieved,
        "relevant": metrics.relevant,
    }


def _aggregate_dict(
    measurements: Sequence[RetrievalMetrics],
    ks: Sequence[int],
    ranking_depth: int,
) -> dict[str, Any]:
    aggregate = aggregate_retrieval_metrics(measurements, ks=ks)
    return {
        "recall_at_k": {str(k): value for k, value in aggregate.recall_at_k.items()},
        "mean_reciprocal_rank_at_k": aggregate.mean_reciprocal_rank,
        "mean_average_precision_at_k": aggregate.mean_average_precision,
        "ranking_depth": ranking_depth,
        "query_count": aggregate.query_count,
        "total_retrieved": aggregate.total_retrieved,
        "total_relevant": aggregate.total_relevant,
    }


def _validate_retrieval_gold(
    example: FinQAExample,
    chunks: Sequence[EvidenceChunk],
) -> None:
    """Reject evidence labels that cannot be evaluated against source chunks."""

    if not example.gold_evidence_ids:
        raise ValueError(
            f"example {example.example_id!r} has no gold_evidence_ids; "
            "retrieval evaluation requires non-empty gold labels (missing IDs: [])"
        )

    chunk_ids = {chunk.evidence_id for chunk in chunks}
    missing = sorted(
        set(example.gold_evidence_ids) - chunk_ids - _ALLOWED_UNADDRESSABLE_GOLD_IDS
    )
    if missing:
        raise ValueError(
            f"example {example.example_id!r} has gold evidence IDs with no source chunk; "
            f"missing IDs: {missing}"
        )


def run_retrieval_evaluation(
    dataset_path: str | Path,
    *,
    method: str = "bm25",
    top_k: int = 10,
    ks: Sequence[int] = (1, 3, 5, 10),
    limit: int | None = None,
    embedding_model: str | None = None,
    embedding_revision: str | None = None,
    reranker_model: str | None = None,
    reranker_revision: str | None = None,
    candidate_k: int = 40,
    include_cells: bool = False,
) -> dict[str, Any]:
    """Evaluate retrieval against FinQA supporting-fact labels.

    Retrieval is intentionally performed within the report attached to each
    FinQA question, matching the task's full-context setting.  Labels are used
    only after ranking to calculate metrics.
    """

    if method not in {"bm25", "hybrid"}:
        raise ValueError("method must be 'bm25' or 'hybrid'")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    cutoffs = tuple(sorted(set(int(k) for k in ks)))
    if not cutoffs or any(k <= 0 for k in cutoffs):
        raise ValueError("ks must contain positive integers")
    if top_k < max(cutoffs):
        raise ValueError("top_k must be at least the largest evaluation cutoff")
    if method == "hybrid" and not embedding_model:
        raise ValueError("hybrid retrieval requires embedding_model")
    if method == "bm25" and any(
        (embedding_model, embedding_revision, reranker_model, reranker_revision)
    ):
        raise ValueError("semantic model options require method='hybrid'")
    if embedding_revision and not embedding_model:
        raise ValueError("embedding_revision requires embedding_model")
    if reranker_revision and not reranker_model:
        raise ValueError("reranker_revision requires reranker_model")

    examples = load_finqa(dataset_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        examples = examples[:limit]
    example_ids = [example.example_id for example in examples]
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("dataset contains duplicate example IDs")

    prepared_examples: list[tuple[FinQAExample, list[EvidenceChunk]]] = []
    for example in examples:
        chunks = chunk_example(example, include_cells=include_cells)
        _validate_retrieval_gold(example, chunks)
        prepared_examples.append((example, chunks))

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

    per_query: dict[str, Any] = {}
    metrics: list[RetrievalMetrics] = []
    stratified: defaultdict[str, list[RetrievalMetrics]] = defaultdict(list)
    latencies_ms: list[float] = []
    context_ratios: list[float] = []

    for example, chunks in prepared_examples:
        started = time.perf_counter()
        if method == "bm25":
            hits = BM25Retriever(chunks).retrieve(example.question, k=top_k)
        else:
            hits = HybridRetriever(
                chunks,
                dense_encoder=dense_encoder,
                reranker=reranker,
            ).retrieve(example.question, k=top_k, candidate_k=candidate_k)
        elapsed_ms = (time.perf_counter() - started) * 1000

        measurement = evaluate_retrieval(hits, example.gold_evidence_ids, ks=cutoffs)
        metrics.append(measurement)
        stratified[_gold_kind(example)].append(measurement)
        latencies_ms.append(elapsed_ms)

        full_chars = sum(len(chunk.text) for chunk in chunks)
        retrieved_chars = sum(len(hit.chunk.text) for hit in hits)
        context_ratio = retrieved_chars / full_chars if full_chars else 0.0
        context_ratios.append(context_ratio)

        per_query[example.example_id] = {
            "gold_kind": _gold_kind(example),
            "gold_evidence_ids": list(example.gold_evidence_ids),
            "metrics": _metrics_dict(measurement, top_k),
            "latency_ms": elapsed_ms,
            "context_retained_ratio": context_ratio,
            "hits": [_hit_dict(hit) for hit in hits],
        }

    return {
        "config": {
            "method": method,
            "top_k": top_k,
            "ks": list(cutoffs),
            "limit": limit,
            "embedding_model": embedding_model,
            "embedding_revision": embedding_revision,
            "reranker_model": reranker_model,
            "reranker_revision": reranker_revision,
            "candidate_k": candidate_k,
            "include_cells": include_cells,
        },
        "aggregate": _aggregate_dict(metrics, cutoffs, top_k),
        "by_gold_kind": {
            name: _aggregate_dict(group, cutoffs, top_k)
            for name, group in sorted(stratified.items())
        },
        "efficiency": {
            "mean_latency_ms": mean(latencies_ms) if latencies_ms else 0.0,
            "p50_latency_ms": _percentile(latencies_ms, 0.50),
            "p95_latency_ms": _percentile(latencies_ms, 0.95),
            "mean_context_retained_ratio": mean(context_ratios) if context_ratios else 0.0,
        },
        "per_query": per_query,
    }


def _example_table_values(example: FinQAExample) -> dict[str, tuple[str, ...]]:
    table_values: dict[str, tuple[str, ...]] = {}
    for row in example.table[1:]:
        if not row or not row[0].strip():
            continue
        numeric_cells: list[str] = []
        for cell in row[1:]:
            try:
                parse_decimal(cell)
            except ProgramError:
                continue
            numeric_cells.append(cell)
        if numeric_cells:
            table_values[row[0]] = tuple(numeric_cells)
    return table_values


def _execution_answer_matches(prediction: str, gold: Any) -> bool:
    if gold is None:
        return False
    if prediction in {"yes", "no"}:
        return str(gold).strip().casefold() == prediction
    try:
        predicted_value = Decimal(prediction)
        gold_value = Decimal(str(gold))
    except (InvalidOperation, ValueError):
        return False
    tolerance = max(Decimal("0.00002"), abs(gold_value) * Decimal("0.000001"))
    return abs(predicted_value - gold_value) <= tolerance


def run_executor_audit(
    dataset_path: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Audit the safe executor against gold programs, never as inference.

    This diagnostic establishes DSL coverage and arithmetic fidelity.  It is
    explicitly not an end-to-end model score because the program is supplied by
    the dataset.
    """

    examples = load_finqa(dataset_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        examples = examples[:limit]

    rows: dict[str, Any] = {}
    execution_successes = 0
    answer_matches = 0
    scored = 0
    error_counts: defaultdict[str, int] = defaultdict(int)
    for example in examples:
        gold = example.metadata.get("execution_answer")
        if gold is not None:
            scored += 1
        try:
            execution = execute_program(
                example.program,
                table_values=_example_table_values(example),
            )
        except ProgramError as exc:
            error_counts[exc.code] += 1
            rows[example.example_id] = {
                "executed": False,
                "error_code": exc.code,
                "error": str(exc),
            }
            continue

        execution_successes += 1
        matches = _execution_answer_matches(execution.answer, gold)
        answer_matches += int(matches)
        rows[example.example_id] = {
            "executed": True,
            "answer": execution.answer,
            "execution_answer": gold,
            "matches_execution_answer": matches,
            "steps": len(execution.steps),
        }

    total = len(examples)
    return {
        "config": {"mode": "gold_program_executor_audit", "limit": limit},
        "summary": {
            "examples": total,
            "execution_successes": execution_successes,
            "execution_success_rate": execution_successes / total if total else 0.0,
            "scored_examples": scored,
            "answer_matches": answer_matches,
            "answer_match_rate": answer_matches / scored if scored else 0.0,
            "error_counts": dict(sorted(error_counts.items())),
        },
        "per_query": rows,
    }


def _hit_dict(hit: RetrievalHit) -> dict[str, Any]:
    return {
        "evidence_id": hit.chunk.evidence_id,
        "kind": hit.chunk.kind,
        "rank": hit.rank,
        "score": hit.score,
        "component_scores": dict(hit.scores),
        "text": hit.chunk.text,
    }


def _parse_ks(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cutoffs must be comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("cutoffs must be positive")
    return values


def evaluate_retrieval_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FinQA evidence retrieval")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--method", choices=("bm25", "hybrid"), default="bm25")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ks", type=_parse_ks, default=(1, 3, 5, 10))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--reranker-model")
    parser.add_argument("--reranker-revision")
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--include-cells", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/retrieval"))
    args = parser.parse_args()

    report = run_retrieval_evaluation(
        args.dataset,
        method=args.method,
        top_k=args.top_k,
        ks=args.ks,
        limit=args.limit,
        embedding_model=args.embedding_model,
        embedding_revision=args.embedding_revision,
        reranker_model=args.reranker_model,
        reranker_revision=args.reranker_revision,
        candidate_k=args.candidate_k,
        include_cells=args.include_cells,
    )
    manifest = build_run_manifest(
        config=report["config"],
        dataset_path=args.dataset,
        metrics={
            "aggregate": report["aggregate"],
            "by_gold_kind": report["by_gold_kind"],
            "efficiency": report["efficiency"],
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{manifest.run_id}.json"
    manifest_path = args.output_dir / f"{manifest.run_id}.manifest.json"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "report": str(report_path),
                "manifest": str(manifest_path),
                **report["aggregate"],
                **report["efficiency"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def audit_executor_main() -> None:
    parser = argparse.ArgumentParser(description="Audit the safe executor on gold FinQA programs")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/executor"))
    args = parser.parse_args()

    report = run_executor_audit(args.dataset, limit=args.limit)
    manifest = build_run_manifest(
        config=report["config"],
        dataset_path=args.dataset,
        metrics=report["summary"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{manifest.run_id}.json"
    manifest_path = args.output_dir / f"{manifest.run_id}.manifest.json"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "report": str(report_path),
                "manifest": str(manifest_path),
                **report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def demo_main() -> None:
    parser = argparse.ArgumentParser(description="Inspect BM25 evidence for one FinQA example")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    examples = {example.example_id: example for example in load_finqa(args.dataset)}
    if args.example_id not in examples:
        parser.error(f"example id not found: {args.example_id}")
    example = examples[args.example_id]
    hits = BM25Retriever(chunk_example(example)).retrieve(example.question, k=args.top_k)
    print(
        json.dumps(
            {
                "example_id": example.example_id,
                "question": example.question,
                "gold_evidence_ids": list(example.gold_evidence_ids),
                "hits": [_hit_dict(hit) for hit in hits],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    evaluate_retrieval_main()
