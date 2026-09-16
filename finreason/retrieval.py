"""Deterministic retrieval primitives for FinReason-Lab.

The module intentionally keeps the baseline dependency-free.  Dense retrieval
and cross-encoder reranking are expressed as small protocols, so experiments can
inject local/test encoders without importing a model framework.  Convenience
adapters for ``sentence-transformers`` are lazy and fail with an actionable
message when that optional dependency is not installed.

Only :class:`~finreason.schema.EvidenceChunk` objects are indexed.  In
particular, the APIs do not accept ``FinQAExample`` and therefore cannot inspect
gold evidence while ranking.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from math import isfinite, log, sqrt
from typing import Protocol, TypeAlias, runtime_checkable

from .schema import EvidenceChunk, RetrievalHit

Tokenize: TypeAlias = Callable[[str], Sequence[str]]

_TOKEN_RE = re.compile(r"[^\W_]+(?:[.,][^\W_]+)*", flags=re.UNICODE)


def default_tokenize(text: str) -> tuple[str, ...]:
    """Return simple Unicode-aware, case-folded lexical tokens.

    Keeping punctuation out makes values such as ``$1,250`` searchable via the
    token ``1,250`` while avoiding a language/model dependency in the baseline.
    """

    if not text:
        return ()
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(text))


def _validate_chunks(chunks: Sequence[EvidenceChunk]) -> tuple[EvidenceChunk, ...]:
    materialized = tuple(chunks)
    seen: set[str] = set()
    for chunk in materialized:
        if not chunk.evidence_id:
            raise ValueError("Every chunk must have a non-empty evidence_id")
        if chunk.evidence_id in seen:
            raise ValueError(f"Duplicate evidence_id: {chunk.evidence_id!r}")
        seen.add(chunk.evidence_id)
    return materialized


def _limit(k: int, available: int) -> int:
    if k <= 0 or available <= 0:
        return 0
    return min(k, available)


class BM25Retriever:
    """A small, deterministic Okapi BM25 implementation.

    Ranking ties are resolved by ``evidence_id`` rather than corpus insertion
    order, which makes evaluations reproducible when ingestion is parallelised.
    Documents with no matching query term are not returned.
    """

    def __init__(
        self,
        chunks: Sequence[EvidenceChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Tokenize = default_tokenize,
    ) -> None:
        if not isfinite(k1) or k1 < 0:
            raise ValueError("k1 must be a finite non-negative number")
        if not isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be a finite number in [0, 1]")

        self.chunks = _validate_chunks(chunks)
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = tokenizer

        self._term_frequencies: tuple[Counter[str], ...] = tuple(
            Counter(tokenizer(chunk.text)) for chunk in self.chunks
        )
        self._document_lengths = tuple(
            sum(term_counts.values()) for term_counts in self._term_frequencies
        )
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for term_counts in self._term_frequencies:
            document_frequency.update(term_counts.keys())
        self._document_frequency = document_frequency

    def _idf(self, term: str) -> float:
        """Robertson/Sparck Jones IDF with the common positive smoothing."""

        document_count = len(self.chunks)
        frequency = self._document_frequency.get(term, 0)
        return log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))

    def retrieve(self, query: str, *, k: int = 5) -> tuple[RetrievalHit, ...]:
        limit = _limit(k, len(self.chunks))
        if not limit:
            return ()

        query_terms = Counter(self.tokenizer(query))
        if not query_terms:
            return ()

        scored: list[tuple[float, EvidenceChunk]] = []
        average_length = self._average_document_length or 1.0
        for chunk, term_counts, document_length in zip(
            self.chunks,
            self._term_frequencies,
            self._document_lengths,
            strict=True,
        ):
            score = 0.0
            length_normalizer = self.k1 * (1.0 - self.b + self.b * document_length / average_length)
            for term, query_frequency in query_terms.items():
                term_frequency = term_counts.get(term, 0)
                if not term_frequency:
                    continue
                denominator = term_frequency + length_normalizer
                # k1 == 0 is valid and makes every matching term binary.
                saturation = (
                    1.0 if denominator == 0 else term_frequency * (self.k1 + 1.0) / denominator
                )
                score += query_frequency * self._idf(term) * saturation
            if score > 0.0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].evidence_id))
        return tuple(
            RetrievalHit(
                chunk=chunk,
                score=score,
                rank=rank,
                scores={"bm25": score},
            )
            for rank, (score, chunk) in enumerate(scored[:limit], start=1)
        )

    # ``search`` is a convenient spelling for interactive use while the shared
    # pipeline consistently calls ``retrieve``.
    search = retrieve


@runtime_checkable
class DenseEncoder(Protocol):
    """Minimal interface implemented by dense text encoders."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode texts into equal-width numeric vectors."""


@runtime_checkable
class Reranker(Protocol):
    """Minimal interface implemented by query/document rerankers."""

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        """Return one relevance score per document."""


def _materialize_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
    context: str,
) -> tuple[tuple[float, ...], ...]:
    materialized = tuple(tuple(float(value) for value in vector) for vector in vectors)
    if len(materialized) != expected_count:
        raise ValueError(
            f"{context} returned {len(materialized)} vectors for {expected_count} texts"
        )
    if not materialized:
        return ()
    dimensions = len(materialized[0])
    if dimensions == 0:
        raise ValueError(f"{context} returned an empty vector")
    for vector in materialized:
        if len(vector) != dimensions:
            raise ValueError(f"{context} returned vectors with inconsistent dimensions")
        if not all(isfinite(value) for value in vector):
            raise ValueError(f"{context} returned a non-finite vector value")
    return materialized


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Dense query and document vectors have different dimensions")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


class DenseRetriever:
    """In-memory cosine-similarity retrieval using any pluggable encoder."""

    def __init__(
        self,
        chunks: Sequence[EvidenceChunk],
        encoder: DenseEncoder,
    ) -> None:
        self.chunks = _validate_chunks(chunks)
        self.encoder = encoder
        self._vectors = (
            _materialize_vectors(
                encoder.encode([chunk.text for chunk in self.chunks]),
                expected_count=len(self.chunks),
                context="Dense encoder",
            )
            if self.chunks
            else ()
        )

    def retrieve(self, query: str, *, k: int = 5) -> tuple[RetrievalHit, ...]:
        limit = _limit(k, len(self.chunks))
        if not limit or not query.strip():
            return ()
        query_vectors = _materialize_vectors(
            self.encoder.encode([query]), expected_count=1, context="Dense encoder"
        )
        query_vector = query_vectors[0]
        scored = [
            (_cosine_similarity(query_vector, vector), chunk)
            for chunk, vector in zip(self.chunks, self._vectors, strict=True)
        ]
        scored.sort(key=lambda item: (-item[0], item[1].evidence_id))
        return tuple(
            RetrievalHit(
                chunk=chunk,
                score=score,
                rank=rank,
                scores={"dense": score},
            )
            for rank, (score, chunk) in enumerate(scored[:limit], start=1)
        )

    search = retrieve


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RetrievalHit]],
    *,
    rank_constant: float = 60.0,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> tuple[RetrievalHit, ...]:
    """Fuse named rankings with weighted reciprocal-rank fusion (RRF).

    Each source contributes ``weight / (rank_constant + rank)``.  Only the
    first occurrence of an evidence id in a source contributes.  Source names
    are processed in sorted order so even floating-point addition order is
    deterministic.
    """

    if not isfinite(rank_constant) or rank_constant < 0:
        raise ValueError("rank_constant must be a finite non-negative number")
    if limit is not None and limit <= 0:
        return ()

    unknown_weights = set(weights or ()) - set(rankings)
    if unknown_weights:
        names = ", ".join(sorted(unknown_weights))
        raise ValueError(f"Weights supplied for unknown rankings: {names}")

    total_scores: defaultdict[str, float] = defaultdict(float)
    chunks: dict[str, EvidenceChunk] = {}
    component_scores: defaultdict[str, dict[str, float]] = defaultdict(dict)

    for name in sorted(rankings):
        weight = float((weights or {}).get(name, 1.0))
        if not isfinite(weight) or weight < 0:
            raise ValueError(f"Weight for {name!r} must be finite and non-negative")
        seen_in_source: set[str] = set()
        for rank, hit in enumerate(rankings[name], start=1):
            evidence_id = hit.chunk.evidence_id
            if evidence_id in seen_in_source:
                continue
            seen_in_source.add(evidence_id)
            prior = chunks.get(evidence_id)
            if prior is not None and prior != hit.chunk:
                raise ValueError(
                    f"Rankings disagree on chunk content for evidence_id {evidence_id!r}"
                )
            chunks[evidence_id] = hit.chunk
            contribution = weight / (rank_constant + rank)
            total_scores[evidence_id] += contribution
            component_scores[evidence_id][f"rrf.{name}"] = contribution
            component_scores[evidence_id][f"raw.{name}"] = float(hit.score)

    ordered = sorted(total_scores, key=lambda item: (-total_scores[item], item))
    if limit is not None:
        ordered = ordered[:limit]
    return tuple(
        RetrievalHit(
            chunk=chunks[evidence_id],
            score=total_scores[evidence_id],
            rank=rank,
            scores=dict(component_scores[evidence_id]),
        )
        for rank, evidence_id in enumerate(ordered, start=1)
    )


class HybridRetriever:
    """Combine BM25 and an optional dense retriever, then optionally rerank.

    ``dense_encoder=None`` deliberately leaves a useful, dependency-free BM25
    baseline.  A reranker sees only the candidates returned by retrieval; it
    never sees labels or gold evidence.
    """

    def __init__(
        self,
        chunks: Sequence[EvidenceChunk],
        *,
        dense_encoder: DenseEncoder | None = None,
        reranker: Reranker | None = None,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        rank_constant: float = 60.0,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        tokenizer: Tokenize = default_tokenize,
    ) -> None:
        self.chunks = _validate_chunks(chunks)
        self.bm25 = BM25Retriever(self.chunks, k1=bm25_k1, b=bm25_b, tokenizer=tokenizer)
        self.dense = (
            DenseRetriever(self.chunks, dense_encoder) if dense_encoder is not None else None
        )
        self.reranker = reranker
        self.weights = {"bm25": float(sparse_weight), "dense": float(dense_weight)}
        self.rank_constant = rank_constant

        # Validate weights/constant even for an empty corpus, where fusion may
        # never otherwise be invoked.
        for name, weight in self.weights.items():
            if not isfinite(weight) or weight < 0:
                raise ValueError(f"{name} weight must be finite and non-negative")
        if not isfinite(rank_constant) or rank_constant < 0:
            raise ValueError("rank_constant must be a finite non-negative number")

    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        candidate_k: int | None = None,
    ) -> tuple[RetrievalHit, ...]:
        limit = _limit(k, len(self.chunks))
        if not limit or not query.strip():
            return ()

        if candidate_k is None:
            candidate_limit = min(len(self.chunks), max(limit, limit * 4))
        else:
            if candidate_k <= 0:
                return ()
            candidate_limit = min(len(self.chunks), max(limit, candidate_k))

        rankings: dict[str, Sequence[RetrievalHit]] = {
            "bm25": self.bm25.retrieve(query, k=candidate_limit)
        }
        weights = {"bm25": self.weights["bm25"]}
        if self.dense is not None:
            rankings["dense"] = self.dense.retrieve(query, k=candidate_limit)
            weights["dense"] = self.weights["dense"]

        fused = reciprocal_rank_fusion(
            rankings,
            rank_constant=self.rank_constant,
            weights=weights,
            limit=candidate_limit,
        )
        if self.reranker is None or not fused:
            return tuple(
                RetrievalHit(
                    chunk=hit.chunk,
                    score=hit.score,
                    rank=rank,
                    scores=hit.scores,
                )
                for rank, hit in enumerate(fused[:limit], start=1)
            )

        reranker_scores = tuple(
            float(value) for value in self.reranker.score(query, [hit.chunk.text for hit in fused])
        )
        if len(reranker_scores) != len(fused):
            raise ValueError(
                f"Reranker returned {len(reranker_scores)} scores for {len(fused)} candidates"
            )
        if not all(isfinite(value) for value in reranker_scores):
            raise ValueError("Reranker returned a non-finite score")

        reranked = list(zip(fused, reranker_scores, strict=True))
        reranked.sort(key=lambda item: (-item[1], -item[0].score, item[0].chunk.evidence_id))
        return tuple(
            RetrievalHit(
                chunk=hit.chunk,
                score=reranker_score,
                rank=rank,
                scores={**hit.scores, "rrf": hit.score, "reranker": reranker_score},
            )
            for rank, (hit, reranker_score) in enumerate(reranked[:limit], start=1)
        )

    search = retrieve


class SentenceTransformerEncoder:
    """Optional ``sentence-transformers`` adapter for dense retrieval."""

    def __init__(self, model_name: str, **model_kwargs: object) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "SentenceTransformerEncoder requires the optional "
                "'sentence-transformers' package. Install it with "
                "`pip install sentence-transformers`."
            ) from exc
        self.model = SentenceTransformer(model_name, **model_kwargs)

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return ()
        return self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class SentenceTransformerCrossEncoder:
    """Optional ``sentence-transformers`` cross-encoder reranker adapter."""

    def __init__(self, model_name: str, **model_kwargs: object) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "SentenceTransformerCrossEncoder requires the optional "
                "'sentence-transformers' package. Install it with "
                "`pip install sentence-transformers`."
            ) from exc
        self.model = CrossEncoder(model_name, **model_kwargs)

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        if not documents:
            return ()
        return self.model.predict([[query, document] for document in documents])


__all__ = [
    "BM25Retriever",
    "DenseEncoder",
    "DenseRetriever",
    "HybridRetriever",
    "Reranker",
    "SentenceTransformerCrossEncoder",
    "SentenceTransformerEncoder",
    "default_tokenize",
    "reciprocal_rank_fusion",
]
