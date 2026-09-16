"""FastAPI service for evidence retrieval and bounded financial reasoning."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from .chunking import chunk_example
from .data import load_finqa
from .generator import OpenAICompatibleProgramGenerator
from .retrieval import (
    HybridRetriever,
    SentenceTransformerCrossEncoder,
    SentenceTransformerEncoder,
)
from .schema import FinQAExample, RetrievalHit, as_tuple_rows
from .workflow import FinReasonWorkflow, StructuredGenerator, WorkflowResult

try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from pydantic import BaseModel, ConfigDict, Field, model_validator
except ImportError as exc:  # pragma: no cover - import guard depends on environment
    raise ImportError("The API requires `pip install 'finreason-lab[api]'`") from exc


ContextText = Annotated[str, Field(max_length=20_000)]
ContextList = Annotated[list[ContextText], Field(max_length=500)]
TableRow = Annotated[list[ContextText], Field(max_length=200)]
TablePayload = Annotated[list[TableRow], Field(max_length=2_000)]


class QueryRequest(BaseModel):
    """Select a packaged FinQA example or provide an inline report."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    example_id: str | None = Field(default=None, max_length=512)
    question: str | None = Field(default=None, max_length=2_000)
    report_id: str = Field(default="inline-report", max_length=512)
    pre_text: ContextList = Field(default_factory=list)
    table: TablePayload = Field(default_factory=list)
    post_text: ContextList = Field(default_factory=list)
    top_k: Annotated[int, Field(ge=1, le=50)] = 5

    @model_validator(mode="after")
    def validate_source(self) -> QueryRequest:
        if self.example_id is None:
            if not self.question:
                raise ValueError("question is required for inline context")
            if not (self.pre_text or self.table or self.post_text):
                raise ValueError("inline context must contain text or a table")
        return self


class ReasonRequest(QueryRequest):
    max_repairs: Annotated[int, Field(ge=0, le=3)] = 1


class HitResponse(BaseModel):
    evidence_id: str
    kind: str
    rank: int
    score: float
    component_scores: dict[str, float]
    text: str


class RetrievalResponse(BaseModel):
    trace_id: str
    example_id: str
    question: str
    hits: list[HitResponse]
    latency_ms: float


class TraceResponse(BaseModel):
    stage: str
    outcome: str
    attempt: int | None = None
    code: str | None = None
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class ReasonResponse(BaseModel):
    trace_id: str
    status: str
    stop_reason: str
    answer: str | None
    program: str | None
    citations: list[str]
    attempts: int
    repairs_used: int
    latency_ms: float
    trace: list[TraceResponse]


class _ApplicationState:
    def __init__(
        self,
        *,
        dataset_path: str | Path | None,
        generator: StructuredGenerator | None,
        embedding_model: str | None,
        embedding_revision: str | None,
        reranker_model: str | None,
        reranker_revision: str | None,
    ) -> None:
        if embedding_revision and not embedding_model:
            raise ValueError("embedding_revision requires embedding_model")
        if reranker_revision and not reranker_model:
            raise ValueError("reranker_revision requires reranker_model")
        examples = load_finqa(dataset_path) if dataset_path else []
        self.examples = {example.example_id: example for example in examples}
        if len(self.examples) != len(examples):
            raise ValueError("dataset contains duplicate example IDs")
        self.generator = generator
        self.dense_encoder = (
            SentenceTransformerEncoder(embedding_model, revision=embedding_revision)
            if embedding_model
            else None
        )
        self.reranker = (
            SentenceTransformerCrossEncoder(reranker_model, revision=reranker_revision)
            if reranker_model
            else None
        )
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model

    def resolve(self, query: QueryRequest) -> FinQAExample:
        if query.example_id is not None:
            try:
                source = self.examples[query.example_id]
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="example_id not found") from exc
            if query.question and query.question != source.question:
                return FinQAExample(
                    example_id=source.example_id,
                    report_id=source.report_id,
                    question=query.question,
                    pre_text=source.pre_text,
                    table=source.table,
                    post_text=source.post_text,
                    answer="",
                    program="",
                    gold_evidence_ids=(),
                )
            return source

        return FinQAExample(
            example_id="inline",
            report_id=query.report_id,
            question=query.question or "",
            pre_text=tuple(query.pre_text),
            table=as_tuple_rows(query.table),
            post_text=tuple(query.post_text),
            answer="",
            program="",
            gold_evidence_ids=(),
        )

    def retriever(self, example: FinQAExample) -> HybridRetriever:
        return HybridRetriever(
            chunk_example(example),
            dense_encoder=self.dense_encoder,
            reranker=self.reranker,
        )


def _default_generator() -> StructuredGenerator | None:
    base_url = os.getenv("FINREASON_MODEL_BASE_URL", "").strip()
    model = os.getenv("FINREASON_MODEL_NAME", "").strip()
    if not base_url and not model:
        return None
    return OpenAICompatibleProgramGenerator.from_environment()


def create_app(
    *,
    dataset_path: str | Path | None = None,
    generator: StructuredGenerator | None = None,
    embedding_model: str | None = None,
    embedding_revision: str | None = None,
    reranker_model: str | None = None,
    reranker_revision: str | None = None,
) -> FastAPI:
    """Create an app; keyword injection keeps startup and tests deterministic."""

    resolved_dataset = dataset_path or os.getenv("FINREASON_DATASET_PATH") or None
    resolved_embedding = embedding_model or os.getenv("FINREASON_EMBEDDING_MODEL") or None
    resolved_embedding_revision = (
        embedding_revision or os.getenv("FINREASON_EMBEDDING_REVISION") or None
    )
    resolved_reranker = reranker_model or os.getenv("FINREASON_RERANKER_MODEL") or None
    resolved_reranker_revision = (
        reranker_revision or os.getenv("FINREASON_RERANKER_REVISION") or None
    )
    resolved_generator = generator or _default_generator()
    state = _ApplicationState(
        dataset_path=resolved_dataset,
        generator=resolved_generator,
        embedding_model=resolved_embedding,
        embedding_revision=resolved_embedding_revision,
        reranker_model=resolved_reranker,
        reranker_revision=resolved_reranker_revision,
    )

    app = FastAPI(
        title="FinReason-Lab API",
        version="0.2.0",
        description="Evidence retrieval and bounded tool-augmented FinQA reasoning",
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next) -> Response:
        trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex
        request.state.trace_id = trace_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-trace-id"] = trace_id
        response.headers["x-process-time-ms"] = f"{(time.perf_counter() - started) * 1000:.3f}"
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "dataset_examples": len(state.examples),
            "generator_configured": state.generator is not None,
            "retrieval": "hybrid" if state.dense_encoder is not None else "bm25",
            "reranker_configured": state.reranker is not None,
        }

    @app.post("/v1/retrieve", response_model=RetrievalResponse)
    def retrieve(payload: QueryRequest, request: Request) -> RetrievalResponse:
        started = time.perf_counter()
        example = state.resolve(payload)
        hits = state.retriever(example).retrieve(example.question, k=payload.top_k)
        return RetrievalResponse(
            trace_id=request.state.trace_id,
            example_id=example.example_id,
            question=example.question,
            hits=[_hit_response(hit) for hit in hits],
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    @app.post("/v1/reason", response_model=ReasonResponse)
    def reason(payload: ReasonRequest, request: Request) -> ReasonResponse:
        if state.generator is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No model generator configured; set FINREASON_MODEL_BASE_URL "
                    "and FINREASON_MODEL_NAME"
                ),
            )
        started = time.perf_counter()
        example = state.resolve(payload)
        result = FinReasonWorkflow(
            state.retriever(example),
            state.generator,
            top_k=payload.top_k,
            max_repairs=payload.max_repairs,
        ).run(example.question)
        return _reason_response(
            result,
            trace_id=request.state.trace_id,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    return app


def _hit_response(hit: RetrievalHit) -> HitResponse:
    return HitResponse(
        evidence_id=hit.chunk.evidence_id,
        kind=hit.chunk.kind,
        rank=hit.rank,
        score=hit.score,
        component_scores=dict(hit.scores),
        text=hit.chunk.text,
    )


def _reason_response(
    result: WorkflowResult,
    *,
    trace_id: str,
    latency_ms: float,
) -> ReasonResponse:
    return ReasonResponse(
        trace_id=trace_id,
        status=result.status,
        stop_reason=result.stop_reason,
        answer=result.answer,
        program=result.program,
        citations=list(result.citations),
        attempts=result.attempts,
        repairs_used=result.repairs_used,
        latency_ms=latency_ms,
        trace=[TraceResponse(**asdict(event)) for event in result.trace],
    )


__all__ = ["create_app"]
