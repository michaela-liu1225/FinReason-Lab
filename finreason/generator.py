"""Structured-program generation adapters for OpenAI-compatible model servers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .workflow import GeneratedProgram, GenerationRequest


class ModelClientError(RuntimeError):
    """A model endpoint failed or returned an invalid protocol response."""

    code = "model_client_error"


class StructuredOutputError(ModelClientError):
    code = "invalid_structured_output"


class ChatClient(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return the assistant message content for a chat request."""


@dataclass(frozen=True)
class CompletionMetadata:
    """Immutable telemetry for one successful model completion."""

    request_id: str | None
    model: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float


@dataclass
class OpenAICompatibleChatClient:
    """Small dependency-free client for vLLM or another compatible endpoint."""

    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    seed: int | None = None
    enable_thinking: bool | None = None
    max_response_bytes: int = 1_000_000
    _completion_history: list[CompletionMetadata] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url:
            raise ValueError("base_url cannot be empty")
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_tokens <= 0 or self.max_response_bytes <= 0:
            raise ValueError("response limits must be positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not math.isfinite(self.top_p) or not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be a finite number in [0, 1]")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
        ):
            raise ValueError("seed must be an integer or None")
        if self.enable_thinking is not None and not isinstance(self.enable_thinking, bool):
            raise ValueError("enable_thinking must be a boolean or None")

    @property
    def history(self) -> tuple[CompletionMetadata, ...]:
        """Return a read-only snapshot of successful completion telemetry."""

        return tuple(self._completion_history)

    @property
    def completions(self) -> tuple[CompletionMetadata, ...]:
        """Alias for :attr:`history` for telemetry-oriented callers."""

        return self.history

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "response_format": {"type": "json_object"},
        }
        if self.seed is not None:
            request_payload["seed"] = self.seed
        if self.enable_thinking is not None:
            request_payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        body = json.dumps(request_payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read(self.max_response_bytes + 1)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4_096).decode("utf-8", errors="replace")
            raise ModelClientError(f"model endpoint returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ModelClientError(f"model endpoint request failed: {exc}") from exc

        if len(raw) > self.max_response_bytes:
            raise ModelClientError("model endpoint response exceeded the byte limit")
        try:
            payload = json.loads(raw)
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ModelClientError("model endpoint returned an invalid chat response") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelClientError("model endpoint returned empty assistant content")

        usage = payload.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        self._completion_history.append(
            CompletionMetadata(
                request_id=_optional_text(payload.get("id")),
                model=_optional_text(payload.get("model")),
                finish_reason=_optional_text(choice.get("finish_reason")),
                prompt_tokens=_optional_token_count(usage.get("prompt_tokens")),
                completion_tokens=_optional_token_count(usage.get("completion_tokens")),
                total_tokens=_optional_token_count(usage.get("total_tokens")),
                latency_ms=(time.perf_counter() - started) * 1_000,
            )
        )
        return content


class OpenAICompatibleProgramGenerator:
    """Prompt an OpenAI-compatible model for a cited, executable FinQA program."""

    def __init__(self, client: ChatClient) -> None:
        self.client = client

    @classmethod
    def from_environment(cls) -> OpenAICompatibleProgramGenerator:
        base_url = os.getenv("FINREASON_MODEL_BASE_URL", "").strip()
        model = os.getenv("FINREASON_MODEL_NAME", "").strip()
        if not base_url or not model:
            raise ValueError("FINREASON_MODEL_BASE_URL and FINREASON_MODEL_NAME are required")
        client = OpenAICompatibleChatClient(
            base_url=base_url,
            model=model,
            api_key=os.getenv("FINREASON_MODEL_API_KEY") or None,
            timeout_seconds=_environment_float("FINREASON_MODEL_TIMEOUT", 60.0),
            temperature=_environment_float("FINREASON_MODEL_TEMPERATURE", 0.0),
            max_tokens=_environment_int("FINREASON_MODEL_MAX_TOKENS", 512),
            top_p=_environment_float("FINREASON_MODEL_TOP_P", 1.0),
            seed=_optional_environment_int("FINREASON_MODEL_SEED"),
            enable_thinking=_optional_environment_bool("FINREASON_MODEL_ENABLE_THINKING"),
        )
        return cls(client)

    @property
    def telemetry_count(self) -> int:
        """Number of completions exposed by telemetry-capable clients."""

        return len(self._telemetry())

    def telemetry_since(self, index: int) -> tuple[CompletionMetadata, ...]:
        """Return completion metadata recorded at or after ``index``.

        Custom clients used by tests or applications are not required to expose
        telemetry; those clients safely produce an empty tuple.
        """

        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("telemetry index must be an integer")
        if index < 0:
            raise ValueError("telemetry index cannot be negative")
        return self._telemetry()[index:]

    @property
    def prompt_sha256(self) -> str:
        """Hash the exact structured-generation system prompt."""

        return hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest()

    def _telemetry(self) -> tuple[CompletionMetadata, ...]:
        for attribute in ("completions", "history"):
            try:
                values = getattr(self.client, attribute)
            except (AttributeError, TypeError):
                continue
            if not isinstance(values, Sequence) or isinstance(values, str | bytes):
                continue
            return tuple(item for item in values if isinstance(item, CompletionMetadata))
        return ()

    def generate(self, request: GenerationRequest) -> GeneratedProgram:
        evidence = "\n".join(
            f"[{hit.chunk.evidence_id}] {hit.chunk.text}" for hit in request.evidence
        )
        user_parts = [
            f"Question:\n{request.question}",
            f"Retrieved evidence:\n{evidence}",
        ]
        if request.feedback is not None:
            user_parts.append(
                "Repair feedback:\n"
                f"error_code={request.feedback.error_code}\n"
                f"message={request.feedback.message}\n"
                f"previous_program={request.feedback.previous_program}\n"
                f"previous_citations={list(request.feedback.previous_citations)}"
            )
        raw = self.client.complete(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(user_parts)},
            ]
        )
        payload = _parse_json_object(raw)
        unexpected = set(payload) - {"program", "citations"}
        if unexpected:
            raise StructuredOutputError(
                "model output contains unexpected fields: " + ", ".join(sorted(unexpected))
            )
        program = payload.get("program")
        citations = payload.get("citations")
        if not isinstance(program, str):
            raise StructuredOutputError("field 'program' must be a string")
        if not isinstance(citations, list) or not all(
            isinstance(citation, str) for citation in citations
        ):
            raise StructuredOutputError("field 'citations' must be a list of strings")
        return GeneratedProgram(program=program, citations=tuple(citations))


def _parse_json_object(raw: str) -> Mapping[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError("model output is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise StructuredOutputError("model output must be a JSON object")
    return payload


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_token_count(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _optional_environment_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    return int(value) if value else None


def _environment_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def _environment_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


def _optional_environment_bool(name: str) -> bool | None:
    value = os.getenv(name, "").strip().casefold()
    if not value:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false, yes/no, on/off, or 1/0")


_SYSTEM_PROMPT = """You solve numerical questions using only retrieved evidence.
Return one JSON object with exactly this shape:
{"program":"<FinQA DSL>","citations":["<evidence_id>"]}

Allowed operations are add, subtract, multiply, divide, exp, greater,
table_sum, table_average, table_max, and table_min. Separate multiple steps by
commas and reference earlier zero-based steps with #0, #1, and so on. Use
const_N constants where appropriate. `greater(a,b)` returns yes when a is
strictly greater than b, otherwise no. Constants are restricted to -1, 0,
1-10, 100, 1,000, 10,000, 100,000, 1,000,000, 10,000,000, and 1,000,000,000.
Every direct numeric argument must
appear in one of the cited evidence chunks. Cite only IDs shown in the prompt.
Do not include prose, Markdown, an answer, or keys other than program and
citations. If repair feedback is present, correct the specified failure.
"""


__all__ = [
    "ChatClient",
    "CompletionMetadata",
    "ModelClientError",
    "OpenAICompatibleChatClient",
    "OpenAICompatibleProgramGenerator",
    "StructuredOutputError",
]
