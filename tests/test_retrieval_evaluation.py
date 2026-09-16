from __future__ import annotations

import math

import pytest

from finreason.evaluation import (
    aggregate_retrieval_metrics,
    compute_oracle_gap,
    evaluate_retrieval,
    evaluate_retrieval_dataset,
)
from finreason.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from finreason.schema import EvidenceChunk, RetrievalHit


def chunk(evidence_id: str, text: str) -> EvidenceChunk:
    return EvidenceChunk(
        evidence_id=evidence_id,
        report_id="report-1",
        kind="text",
        text=text,
    )


def hit(item: EvidenceChunk, rank: int, score: float = 1.0) -> RetrievalHit:
    return RetrievalHit(chunk=item, rank=rank, score=score)


def test_bm25_ranks_matching_financial_evidence() -> None:
    chunks = [
        chunk("cash", "Cash and equivalents increased to 20 million"),
        chunk("revenue", "Revenue increased from 8 to 12 million"),
        chunk("noise", "The company appointed a new director"),
    ]

    results = BM25Retriever(chunks).retrieve("revenue increased", k=3)

    assert [result.chunk.evidence_id for result in results] == ["revenue", "cash"]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score > results[1].score > 0


def test_bm25_ties_are_deterministic_by_evidence_id() -> None:
    chunks = [chunk("z", "revenue"), chunk("a", "revenue")]
    retriever = BM25Retriever(chunks)

    first = retriever.retrieve("revenue", k=2)
    second = retriever.retrieve("revenue", k=2)

    assert [item.chunk.evidence_id for item in first] == ["a", "z"]
    assert first == second


def test_bm25_and_dense_handle_empty_inputs() -> None:
    class FailingEncoder:
        def encode(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("an empty corpus must not call the encoder")

    assert BM25Retriever([]).retrieve("revenue") == ()
    assert BM25Retriever([chunk("a", "revenue")]).retrieve("", k=5) == ()
    assert DenseRetriever([], FailingEncoder()).retrieve("revenue") == ()


def test_dense_retrieval_and_ties_are_deterministic() -> None:
    class Encoder:
        def encode(self, texts: list[str]) -> list[list[float]]:
            vectors = {
                "alpha": [1.0, 0.0],
                "beta": [0.0, 1.0],
                "query": [1.0, 0.0],
                "zero": [0.0, 0.0],
            }
            return [vectors[text] for text in texts]

    retriever = DenseRetriever(
        [chunk("z", "beta"), chunk("a", "alpha"), chunk("m", "zero")],
        Encoder(),
    )
    results = retriever.retrieve("query", k=3)

    assert [item.chunk.evidence_id for item in results] == ["a", "m", "z"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == results[2].score == 0.0


def test_reciprocal_rank_fusion_combines_sources_and_breaks_ties() -> None:
    a, b, c = chunk("a", "A"), chunk("b", "B"), chunk("c", "C")
    rankings = {
        "sparse": [hit(a, 1), hit(b, 2)],
        "dense": [hit(c, 1), hit(b, 2)],
    }

    fused = reciprocal_rank_fusion(rankings, rank_constant=60)

    assert [item.chunk.evidence_id for item in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(2 / 62)
    # a and c have identical fused scores; evidence_id is the stable tiebreaker.
    assert fused[1].score == fused[2].score
    assert reciprocal_rank_fusion({}, limit=5) == ()
    assert reciprocal_rank_fusion(rankings, limit=0) == ()


def test_hybrid_retriever_accepts_pluggable_encoder_and_reranker() -> None:
    class Encoder:
        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] if "dense" in text else [0.0, 1.0] for text in texts]

    class Reranker:
        def score(self, query: str, documents: list[str]) -> list[float]:
            assert query == "dense query"
            return [1.0 if "preferred" in document else 0.0 for document in documents]

    retriever = HybridRetriever(
        [chunk("a", "dense candidate"), chunk("b", "preferred lexical query")],
        dense_encoder=Encoder(),
        reranker=Reranker(),
        rank_constant=0,
    )

    results = retriever.retrieve("dense query", k=2)

    assert [item.chunk.evidence_id for item in results] == ["b", "a"]
    assert results[0].scores["reranker"] == 1.0


def test_retrieval_metrics_recall_mrr_and_average_precision() -> None:
    metrics = evaluate_retrieval(
        ["noise", "g1", "noise", "g2"],
        ["g1", "g2", "missing"],
        ks=(1, 2, 3, 5),
    )

    assert metrics.recall_at_k == {
        1: 0.0,
        2: pytest.approx(1 / 3),
        3: pytest.approx(2 / 3),
        5: pytest.approx(2 / 3),
    }
    assert metrics.reciprocal_rank == pytest.approx(0.5)
    # De-duplication yields ranks: noise, g1, g2.
    assert metrics.average_precision == pytest.approx((1 / 2 + 2 / 3) / 3)
    assert metrics.retrieved == 3
    assert metrics.relevant == 3


def test_metrics_and_aggregation_handle_empty_inputs() -> None:
    empty = evaluate_retrieval([], [], ks=(1, 5))
    assert empty.recall_at_k == {1: 0.0, 5: 0.0}
    assert empty.reciprocal_rank == empty.average_precision == 0.0

    aggregate = aggregate_retrieval_metrics([], ks=(1, 5))
    assert aggregate.query_count == 0
    assert aggregate.recall_at_k == {1: 0.0, 5: 0.0}


def test_dataset_evaluation_includes_queries_with_no_results() -> None:
    evaluation = evaluate_retrieval_dataset(
        {"q1": ["g1"]},
        {"q1": ["g1"], "q2": ["g2"]},
        ks=(1,),
    )

    assert evaluation.aggregate.query_count == 2
    assert evaluation.aggregate.recall_at_k[1] == pytest.approx(0.5)
    assert evaluation.aggregate.mean_reciprocal_rank == pytest.approx(0.5)
    assert evaluation.per_query["q2"].retrieved == 0

    with pytest.raises(ValueError, match="unknown query ids"):
        evaluate_retrieval_dataset({"wrong": []}, {"q1": []})


def test_oracle_gap_reports_absolute_relative_and_zero_oracle_cases() -> None:
    gap = compute_oracle_gap(oracle_score=0.8, system_score=0.6)
    assert gap.absolute_gap == pytest.approx(0.2)
    assert gap.relative_gap == pytest.approx(0.25)
    assert gap.oracle_retention == pytest.approx(0.75)

    zero = compute_oracle_gap(oracle_score=0.0, system_score=0.0)
    assert zero.relative_gap is None
    assert zero.oracle_retention is None

    with pytest.raises(ValueError, match="finite"):
        compute_oracle_gap(oracle_score=math.nan, system_score=0.0)
