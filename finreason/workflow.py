"""A bounded retrieval-and-tool workflow for grounded FinQA inference.

The workflow consumes only a question at inference time.  It deliberately has
no ``answer`` or gold-evidence input, which makes accidental use of FinQA labels
outside evaluation structurally difficult.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from .executor import (
    ExecutionResult,
    FinQAExecutor,
    ProgramError,
    literal_arguments,
    parse_decimal,
    parse_program,
)
from .schema import EvidenceChunk, RetrievalHit


@dataclass(frozen=True)
class GeneratedProgram:
    """The only model-generated payload accepted by the tool workflow."""

    program: str
    citations: tuple[str, ...]

    def __post_init__(self) -> None:
        # Fakes and JSON adapters often naturally provide lists.  Normalize at
        # the boundary while retaining an immutable contract downstream.
        object.__setattr__(self, "program", str(self.program).strip())
        object.__setattr__(
            self,
            "citations",
            tuple(str(citation).strip() for citation in self.citations),
        )


@dataclass(frozen=True)
class RepairFeedback:
    """Machine-readable feedback for one bounded regeneration attempt."""

    attempt: int
    error_code: str
    message: str
    previous_program: str | None
    previous_citations: tuple[str, ...]


@dataclass(frozen=True)
class GenerationRequest:
    """Input to a structured generator; it contains no gold answer."""

    question: str
    evidence: tuple[RetrievalHit, ...]
    feedback: RepairFeedback | None = None


@runtime_checkable
class StructuredGenerator(Protocol):
    """Adapter boundary for an LLM, deterministic baseline, or test fake."""

    def generate(self, request: GenerationRequest) -> GeneratedProgram:
        """Return exactly a FinQA program and retrieved evidence identifiers."""


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, *, k: int = 5) -> Sequence[RetrievalHit]:
        """Retrieve evidence without observing labels or a gold answer."""


@dataclass(frozen=True)
class VerificationRequest:
    question: str
    generation: GeneratedProgram
    evidence: tuple[EvidenceChunk, ...]
    execution: ExecutionResult


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reason: str = ""


@runtime_checkable
class ResultVerifier(Protocol):
    def verify(self, request: VerificationRequest) -> VerificationResult:
        """Check an executed answer without comparing it with a gold answer."""


class DeterministicVerifier:
    """Minimum verifier used when no semantic verifier is configured.

    Parsing, citation validation and evidence grounding happen before this
    point.  The default verifier therefore checks the invariants of the
    deterministic tool result; a model-based semantic verifier can be injected
    behind the same no-gold contract.
    """

    def verify(self, request: VerificationRequest) -> VerificationResult:
        if not request.evidence:
            return VerificationResult(False, "no cited evidence")
        if not request.execution.steps:
            return VerificationResult(False, "execution has no steps")
        if request.execution.answer == "":
            return VerificationResult(False, "execution produced an empty answer")
        return VerificationResult(True, "deterministic checks passed")


class WorkflowValidationError(ValueError):
    code = "validation_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class GenerationProtocolError(WorkflowValidationError):
    code = "invalid_generation"


class InvalidCitationError(WorkflowValidationError):
    code = "invalid_citation"


class UngroundedArgumentError(WorkflowValidationError):
    code = "ungrounded_numeric_argument"


@dataclass(frozen=True)
class TraceEvent:
    """One compact, serialisable workflow observation."""

    stage: str
    outcome: str
    attempt: int | None = None
    code: str | None = None
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowResult:
    """Terminal workflow output with an explicit reason for stopping."""

    status: str
    stop_reason: str
    answer: str | None
    program: str | None
    citations: tuple[str, ...]
    attempts: int
    repairs_used: int
    trace: tuple[TraceEvent, ...]
    execution: ExecutionResult | None = None
    last_error_code: str | None = None
    last_error_message: str | None = None

    @property
    def answered(self) -> bool:
        return self.status == "answered"


# Two alternatives keep a closing parenthesis from being mistaken as an
# accounting-negative marker unless an opening parenthesis was also captured.
_NUMBER_IN_TEXT_RE = re.compile(
    r"(?<![\w#])(?:"
    r"\(\s*[$£€]?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)"
    r"(?:\.\d+)?(?:[eE][+-]?\d+)?\s*%?\s*\)"
    r"|"
    r"[$£€]?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)"
    r"(?:\.\d+)?(?:[eE][+-]?\d+)?\s*%?"
    r")(?![\w])"
)


def extract_evidence_numbers(text: str) -> frozenset[Decimal]:
    """Extract exact numeric values from one evidence chunk."""

    numbers: set[Decimal] = set()
    for match in _NUMBER_IN_TEXT_RE.finditer(text):
        candidate = match.group(0).strip()
        try:
            value = parse_decimal(candidate)
            numbers.add(value)
            # FinQA's flattened programs commonly preserve the displayed
            # percentage magnitude (``27.5``) while the evidence says
            # ``27.5%``; accept both lexical magnitude and mathematical value.
            if "%" in candidate:
                numbers.add(value * Decimal(100))
        except ProgramError:
            # The regex is deliberately permissive around financial formatting;
            # an unparseable candidate simply is not grounding evidence.
            continue
    return frozenset(numbers)


def _normalise_table_label(label: str) -> str:
    return " ".join(label.casefold().split())


def _values_from_table_chunk(chunk: EvidenceChunk) -> tuple[Decimal, ...]:
    """Read values from the stable table-row/cell text emitted by chunking."""

    values: list[Decimal] = []
    for raw_part in chunk.text.split("|"):
        part = raw_part.strip()
        if ":" not in part:
            continue
        field, raw_value = part.split(":", 1)
        field = field.strip().casefold()
        if chunk.kind == "table_cell" and field != "value":
            continue
        if chunk.kind != "table_cell" and field in {
            "row",
            "unit",
            "column",
            "value",
        }:
            continue
        try:
            values.append(parse_decimal(raw_value.strip()))
        except ProgramError:
            continue
    return tuple(values)


def extract_table_values(
    evidence: Sequence[EvidenceChunk],
) -> dict[str, tuple[Decimal, ...]]:
    """Build row-label values from cited table chunks, never from gold labels."""

    row_chunks: dict[str, tuple[Decimal, ...]] = {}
    cell_chunks: dict[str, list[Decimal]] = {}
    for chunk in evidence:
        raw_label = chunk.metadata.get("row_label")
        if not isinstance(raw_label, str) or not raw_label.strip():
            continue
        label = _normalise_table_label(raw_label)
        values = _values_from_table_chunk(chunk)
        if not values:
            continue
        if chunk.kind == "table_row":
            row_chunks[label] = values
        elif chunk.kind == "table_cell":
            cell_chunks.setdefault(label, []).extend(values)

    result = dict(row_chunks)
    for label, values in cell_chunks.items():
        # Prefer a cited row chunk so citing both a row and one of its cells does
        # not double-count the cell in table_sum/table_average.
        result.setdefault(label, tuple(values))
    return result


def _symbolic_table_label(step: Any) -> str | None:
    if (
        step.operation.startswith("table_")
        and len(step.arguments) == 2
        and step.arguments[1].strip().casefold() == "none"
    ):
        return _normalise_table_label(step.arguments[0])
    return None


def validate_generation(
    generation: GeneratedProgram,
    retrieved: Sequence[RetrievalHit],
) -> tuple[EvidenceChunk, ...]:
    """Validate structure, citations and grounding for direct numeric leaves."""

    if not isinstance(generation, GeneratedProgram):
        raise GenerationProtocolError(
            "generator must return a GeneratedProgram", code="invalid_generation_type"
        )
    if not generation.program:
        raise GenerationProtocolError("generator returned an empty program")
    if not generation.citations or any(not item for item in generation.citations):
        raise InvalidCitationError("at least one non-empty citation is required")
    if len(set(generation.citations)) != len(generation.citations):
        raise InvalidCitationError("duplicate citations are not allowed", code="duplicate_citation")

    by_id = {hit.chunk.evidence_id: hit.chunk for hit in retrieved}
    unknown = tuple(citation for citation in generation.citations if citation not in by_id)
    if unknown:
        raise InvalidCitationError(
            "citation was not returned by the retriever: " + ", ".join(unknown),
            code="citation_not_retrieved",
        )
    cited_chunks = tuple(by_id[citation] for citation in generation.citations)

    steps = parse_program(generation.program)
    table_values = extract_table_values(cited_chunks)
    for step in steps:
        table_label = _symbolic_table_label(step)
        if table_label is not None and table_label not in table_values:
            raise UngroundedArgumentError(
                f"table row is absent from cited evidence: {step.arguments[0]}",
                code="ungrounded_table_row",
            )
    grounded_values: set[Decimal] = set()
    for chunk in cited_chunks:
        grounded_values.update(extract_evidence_numbers(chunk.text))

    ungrounded: list[str] = []
    for literal in literal_arguments(steps):
        value = parse_decimal(literal)
        if value not in grounded_values:
            ungrounded.append(literal)
    if ungrounded:
        raise UngroundedArgumentError(
            "numeric argument is absent from cited evidence: " + ", ".join(ungrounded)
        )
    return cited_chunks


class FinReasonWorkflow:
    """Retrieve, generate, validate, execute and verify with bounded repairs."""

    def __init__(
        self,
        retriever: Retriever,
        generator: StructuredGenerator,
        *,
        executor: FinQAExecutor | None = None,
        verifier: ResultVerifier | None = None,
        top_k: int = 5,
        max_repairs: int = 1,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if max_repairs < 0:
            raise ValueError("max_repairs cannot be negative")
        self.retriever = retriever
        self.generator = generator
        self.executor = executor or FinQAExecutor()
        self.verifier = verifier or DeterministicVerifier()
        self.top_k = top_k
        self.max_repairs = max_repairs

    @staticmethod
    def _trace_failure(
        trace: list[TraceEvent],
        *,
        stage: str,
        attempt: int,
        code: str,
        message: str,
    ) -> None:
        trace.append(
            TraceEvent(
                stage=stage,
                outcome="failed",
                attempt=attempt,
                code=code,
                message=message,
            )
        )

    @staticmethod
    def _abstained(
        *,
        reason: str,
        trace: list[TraceEvent],
        attempts: int,
        repairs_used: int,
        generation: GeneratedProgram | None = None,
        execution: ExecutionResult | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> WorkflowResult:
        trace.append(
            TraceEvent(
                stage="stop",
                outcome="abstained",
                code=reason,
                message=error_message or "",
            )
        )
        return WorkflowResult(
            status="abstained",
            stop_reason=reason,
            answer=None,
            program=generation.program if generation else None,
            citations=generation.citations if generation else (),
            attempts=attempts,
            repairs_used=repairs_used,
            trace=tuple(trace),
            execution=execution,
            last_error_code=error_code,
            last_error_message=error_message,
        )

    def run(self, question: str) -> WorkflowResult:
        """Run inference from a question only; no gold answer is accepted."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        question = question.strip()
        trace: list[TraceEvent] = [TraceEvent("retrieve", "started")]
        try:
            retrieved = tuple(self.retriever.retrieve(question, k=self.top_k))
            if any(not isinstance(hit, RetrievalHit) for hit in retrieved):
                raise TypeError("retriever must return RetrievalHit objects")
        except Exception as exc:
            self._trace_failure(
                trace,
                stage="retrieve",
                attempt=0,
                code="retrieval_error",
                message=str(exc),
            )
            return self._abstained(
                reason="retrieval_error",
                trace=trace,
                attempts=0,
                repairs_used=0,
                error_code="retrieval_error",
                error_message=str(exc),
            )

        trace.append(
            TraceEvent(
                "retrieve",
                "succeeded",
                details={"hit_count": len(retrieved), "top_k": self.top_k},
            )
        )
        if not retrieved:
            return self._abstained(
                reason="no_evidence",
                trace=trace,
                attempts=0,
                repairs_used=0,
                error_code="no_evidence",
                error_message="retriever returned no evidence",
            )

        feedback: RepairFeedback | None = None
        last_generation: GeneratedProgram | None = None
        last_execution: ExecutionResult | None = None
        last_code: str | None = None
        last_message: str | None = None

        # attempt=0 is the initial generation.  Each later iteration consumes
        # exactly one repair, so calls can never exceed max_repairs + 1.
        for attempt in range(self.max_repairs + 1):
            display_attempt = attempt + 1
            trace.append(TraceEvent("generate", "started", attempt=display_attempt))
            try:
                generated = self.generator.generate(
                    GenerationRequest(
                        question=question,
                        evidence=retrieved,
                        feedback=feedback,
                    )
                )
                if not isinstance(generated, GeneratedProgram):
                    raise GenerationProtocolError(
                        "generator must return a GeneratedProgram",
                        code="invalid_generation_type",
                    )
                last_generation = generated
                trace.append(
                    TraceEvent(
                        "generate",
                        "succeeded",
                        attempt=display_attempt,
                        details={"citation_count": len(generated.citations)},
                    )
                )
            except Exception as exc:
                code = getattr(exc, "code", "generation_error")
                message = str(exc)
                last_code, last_message = code, message
                self._trace_failure(
                    trace,
                    stage="generate",
                    attempt=display_attempt,
                    code=code,
                    message=message,
                )
                generated = last_generation
                failure_stage = "generate"
            else:
                trace.append(TraceEvent("validate", "started", attempt=display_attempt))
                try:
                    cited_evidence = validate_generation(generated, retrieved)
                except (WorkflowValidationError, ProgramError) as exc:
                    code = getattr(exc, "code", "validation_error")
                    message = str(exc)
                    last_code, last_message = code, message
                    self._trace_failure(
                        trace,
                        stage="validate",
                        attempt=display_attempt,
                        code=code,
                        message=message,
                    )
                    failure_stage = "validate"
                else:
                    trace.append(
                        TraceEvent(
                            "validate",
                            "succeeded",
                            attempt=display_attempt,
                            details={"cited_evidence_count": len(cited_evidence)},
                        )
                    )
                    trace.append(TraceEvent("execute", "started", attempt=display_attempt))
                    try:
                        last_execution = self.executor.execute(
                            generated.program,
                            table_values=extract_table_values(cited_evidence),
                        )
                    except ProgramError as exc:
                        code = exc.code
                        message = str(exc)
                        last_code, last_message = code, message
                        self._trace_failure(
                            trace,
                            stage="execute",
                            attempt=display_attempt,
                            code=code,
                            message=message,
                        )
                        failure_stage = "execute"
                    else:
                        trace.append(
                            TraceEvent(
                                "execute",
                                "succeeded",
                                attempt=display_attempt,
                                details={
                                    "answer": last_execution.answer,
                                    "step_count": len(last_execution.steps),
                                },
                            )
                        )
                        trace.append(TraceEvent("verify", "started", attempt=display_attempt))
                        try:
                            verification = self.verifier.verify(
                                VerificationRequest(
                                    question=question,
                                    generation=generated,
                                    evidence=cited_evidence,
                                    execution=last_execution,
                                )
                            )
                            if not isinstance(verification, VerificationResult):
                                raise TypeError("verifier must return a VerificationResult")
                            if not verification.passed:
                                raise WorkflowValidationError(
                                    verification.reason or "verification failed",
                                    code="verification_failed",
                                )
                        except Exception as exc:
                            code = getattr(exc, "code", "verification_error")
                            message = str(exc)
                            last_code, last_message = code, message
                            self._trace_failure(
                                trace,
                                stage="verify",
                                attempt=display_attempt,
                                code=code,
                                message=message,
                            )
                            failure_stage = "verify"
                        else:
                            trace.append(
                                TraceEvent(
                                    "verify",
                                    "succeeded",
                                    attempt=display_attempt,
                                    message=verification.reason,
                                )
                            )
                            trace.append(
                                TraceEvent(
                                    "stop",
                                    "answered",
                                    attempt=display_attempt,
                                    code="completed",
                                )
                            )
                            return WorkflowResult(
                                status="answered",
                                stop_reason="completed",
                                answer=last_execution.answer,
                                program=generated.program,
                                citations=generated.citations,
                                attempts=display_attempt,
                                repairs_used=attempt,
                                trace=tuple(trace),
                                execution=last_execution,
                            )

            if attempt >= self.max_repairs:
                return self._abstained(
                    reason="repair_budget_exhausted",
                    trace=trace,
                    attempts=display_attempt,
                    repairs_used=attempt,
                    generation=last_generation,
                    execution=last_execution,
                    error_code=last_code,
                    error_message=last_message,
                )

            trace.append(
                TraceEvent(
                    "repair",
                    "scheduled",
                    attempt=display_attempt + 1,
                    code=last_code,
                    message=last_message or "",
                    details={"failed_stage": failure_stage},
                )
            )
            feedback = RepairFeedback(
                attempt=display_attempt + 1,
                error_code=last_code or "unknown_error",
                message=last_message or "",
                previous_program=(last_generation.program if last_generation else None),
                previous_citations=(last_generation.citations if last_generation else ()),
            )

        raise AssertionError("bounded workflow loop terminated unexpectedly")


# A descriptive alias for downstream callers and documentation.
BoundedToolWorkflow = FinReasonWorkflow


__all__ = [
    "BoundedToolWorkflow",
    "DeterministicVerifier",
    "FinReasonWorkflow",
    "GeneratedProgram",
    "GenerationProtocolError",
    "GenerationRequest",
    "InvalidCitationError",
    "RepairFeedback",
    "ResultVerifier",
    "Retriever",
    "StructuredGenerator",
    "TraceEvent",
    "UngroundedArgumentError",
    "VerificationRequest",
    "VerificationResult",
    "WorkflowResult",
    "WorkflowValidationError",
    "extract_evidence_numbers",
    "extract_table_values",
    "validate_generation",
]
