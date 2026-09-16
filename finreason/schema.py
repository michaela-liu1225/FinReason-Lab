"""Typed, model-independent contracts shared by ingestion, retrieval, and tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceChunk:
    """One retrievable unit with stable provenance.

    Gold labels deliberately live on :class:`FinQAExample`, not on the chunk, so
    retrieval code cannot accidentally index supervision metadata.
    """

    evidence_id: str
    report_id: str
    kind: str
    text: str
    source_index: int | None = None
    table_id: str | None = None
    row_index: int | None = None
    column_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinQAExample:
    """Canonical FinQA record used by every new pipeline stage."""

    example_id: str
    report_id: str
    question: str
    pre_text: tuple[str, ...]
    table: tuple[tuple[str, ...], ...]
    post_text: tuple[str, ...]
    answer: str
    program: str
    gold_evidence_ids: tuple[str, ...]
    gold_evidence_text: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def all_text(self) -> tuple[str, ...]:
        return self.pre_text + self.post_text


@dataclass(frozen=True)
class RetrievalHit:
    """A chunk plus the final and component retrieval scores."""

    chunk: EvidenceChunk
    score: float
    rank: int
    scores: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalMetrics:
    """Per-query retrieval measurements before dataset aggregation."""

    recall_at_k: Mapping[int, float]
    reciprocal_rank: float
    average_precision: float
    retrieved: int
    relevant: int


def as_tuple_rows(rows: Sequence[Sequence[Any]] | None) -> tuple[tuple[str, ...], ...]:
    """Normalize arbitrary JSON table cells into an immutable string matrix."""

    if not rows:
        return ()
    return tuple(tuple(str(cell) for cell in row) for row in rows)
