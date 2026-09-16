from __future__ import annotations

import json
import urllib.error

import pytest

from finreason.generator import (
    CompletionMetadata,
    ModelClientError,
    OpenAICompatibleChatClient,
    OpenAICompatibleProgramGenerator,
    StructuredOutputError,
)
from finreason.schema import EvidenceChunk, RetrievalHit
from finreason.workflow import GenerationRequest, RepairFeedback


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return self.response


class FakeHTTPResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit: int) -> bytes:
        return self.payload[:limit]


def _request(*, repair: bool = False) -> GenerationRequest:
    hit = RetrievalHit(
        chunk=EvidenceChunk(
            evidence_id="table_1",
            report_id="report.pdf",
            kind="table_row",
            text="Row: revenue | 2021: 100 | 2022: 125",
        ),
        score=1.0,
        rank=1,
    )
    feedback = (
        RepairFeedback(2, "invalid_reference", "#1 is invalid", "add(#1, 1)", ("table_1",))
        if repair
        else None
    )
    return GenerationRequest("What was the increase?", (hit,), feedback)


def test_generator_parses_json_and_prompts_only_retrieved_evidence() -> None:
    client = FakeClient(json.dumps({"program": "subtract(125, 100)", "citations": ["table_1"]}))

    result = OpenAICompatibleProgramGenerator(client).generate(_request())

    assert result.program == "subtract(125, 100)"
    assert result.citations == ("table_1",)
    prompt = client.messages[1]["content"]
    assert "[table_1]" in prompt
    assert "gold" not in prompt.casefold()

    generator = OpenAICompatibleProgramGenerator(client)
    assert generator.telemetry_count == 0
    assert generator.telemetry_since(0) == ()


def test_generator_includes_machine_readable_repair_feedback() -> None:
    client = FakeClient('```json\n{"program":"subtract(125,100)","citations":["table_1"]}\n```')

    OpenAICompatibleProgramGenerator(client).generate(_request(repair=True))

    prompt = client.messages[1]["content"]
    assert "error_code=invalid_reference" in prompt
    assert "previous_program=add(#1, 1)" in prompt


@pytest.mark.parametrize(
    "response,message",
    [
        ("not-json", "not valid JSON"),
        ('{"program": 1, "citations": []}', "program"),
        ('{"program": "add(1,2)", "citations": "table_1"}', "citations"),
        (
            '{"program": "add(1,2)", "citations": ["table_1"], "answer": 3}',
            "unexpected fields",
        ),
    ],
)
def test_generator_rejects_malformed_structured_output(response, message) -> None:
    with pytest.raises(StructuredOutputError, match=message):
        OpenAICompatibleProgramGenerator(FakeClient(response)).generate(_request())


def test_dependency_free_chat_client_sends_bounded_authenticated_request(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        payload = {
            "id": "completion-123",
            "model": "served-model-revision",
            "choices": [
                {
                    "message": {"content": '{"program":"add(1,2)"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        }
        return FakeHTTPResponse(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    timer = iter((10.0, 10.125))
    monkeypatch.setattr("finreason.generator.time.perf_counter", lambda: next(timer))
    client = OpenAICompatibleChatClient(
        "http://localhost:8001/v1/",
        "local-model",
        api_key="test-key",
        timeout_seconds=3,
        temperature=0.2,
        max_tokens=256,
        top_p=0.9,
        seed=42,
        enable_thinking=False,
    )

    content = client.complete([{"role": "user", "content": "question"}])

    assert content == '{"program":"add(1,2)"}'
    assert captured["request"].full_url.endswith("/v1/chat/completions")
    assert captured["request"].get_header("Authorization") == "Bearer test-key"
    assert captured["timeout"] == 3
    request_body = json.loads(captured["request"].data)
    assert request_body["temperature"] == 0.2
    assert request_body["max_tokens"] == 256
    assert request_body["top_p"] == 0.9
    assert request_body["seed"] == 42
    assert request_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert client.history == (
        CompletionMetadata(
            request_id="completion-123",
            model="served-model-revision",
            finish_reason="stop",
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            latency_ms=125.0,
        ),
    )
    assert client.completions == client.history
    assert "test-key" not in repr(client)


def test_chat_client_omits_unset_seed_and_allows_missing_telemetry(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        payload = {"choices": [{"message": {"content": "{}"}}]}
        return FakeHTTPResponse(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleChatClient("http://localhost/v1", "model")

    assert client.complete([]) == "{}"
    assert "seed" not in captured["body"]
    assert "chat_template_kwargs" not in captured["body"]
    assert captured["body"]["top_p"] == 1.0
    assert client.history[0].request_id is None
    assert client.history[0].model is None
    assert client.history[0].finish_reason is None
    assert client.history[0].prompt_tokens is None
    assert client.history[0].completion_tokens is None
    assert client.history[0].total_tokens is None


def test_chat_client_reports_transport_and_response_errors(monkeypatch) -> None:
    client = OpenAICompatibleChatClient("http://localhost/v1", "model", max_response_bytes=10)

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )
    with pytest.raises(ModelClientError, match="request failed"):
        client.complete([])
    assert client.history == ()

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: FakeHTTPResponse(b"x" * 11),
    )
    with pytest.raises(ModelClientError, match="byte limit"):
        client.complete([])


def test_generator_environment_factory_validates_configuration(monkeypatch) -> None:
    monkeypatch.delenv("FINREASON_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("FINREASON_MODEL_NAME", raising=False)
    with pytest.raises(ValueError, match="required"):
        OpenAICompatibleProgramGenerator.from_environment()


def test_generator_environment_factory_loads_reproducible_settings(monkeypatch) -> None:
    monkeypatch.setenv("FINREASON_MODEL_BASE_URL", "http://localhost:8001/v1/")
    monkeypatch.setenv("FINREASON_MODEL_NAME", "fixed-model")
    monkeypatch.setenv("FINREASON_MODEL_API_KEY", "secret-key")
    monkeypatch.setenv("FINREASON_MODEL_TIMEOUT", "12.5")
    monkeypatch.setenv("FINREASON_MODEL_TEMPERATURE", "0.15")
    monkeypatch.setenv("FINREASON_MODEL_MAX_TOKENS", "321")
    monkeypatch.setenv("FINREASON_MODEL_TOP_P", "0.85")
    monkeypatch.setenv("FINREASON_MODEL_SEED", "0")
    monkeypatch.setenv("FINREASON_MODEL_ENABLE_THINKING", "false")

    generator = OpenAICompatibleProgramGenerator.from_environment()

    client = generator.client
    assert isinstance(client, OpenAICompatibleChatClient)
    assert client.base_url == "http://localhost:8001/v1"
    assert client.model == "fixed-model"
    assert client.timeout_seconds == 12.5
    assert client.temperature == 0.15
    assert client.max_tokens == 321
    assert client.top_p == 0.85
    assert client.seed == 0
    assert client.enable_thinking is False
    assert "secret-key" not in repr(client)


def test_generator_environment_factory_rejects_invalid_thinking_mode(monkeypatch) -> None:
    monkeypatch.setenv("FINREASON_MODEL_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("FINREASON_MODEL_NAME", "fixed-model")
    monkeypatch.setenv("FINREASON_MODEL_ENABLE_THINKING", "sometimes")

    with pytest.raises(ValueError, match="FINREASON_MODEL_ENABLE_THINKING"):
        OpenAICompatibleProgramGenerator.from_environment()


def test_program_generator_exposes_client_telemetry(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        payload = {
            "id": "request-1",
            "model": "model-revision",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "program": "subtract(125, 100)",
                                "citations": ["table_1"],
                            }
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        return FakeHTTPResponse(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleChatClient("http://localhost/v1", "model", seed=42)
    generator = OpenAICompatibleProgramGenerator(client)
    start = generator.telemetry_count

    generator.generate(_request())

    assert start == 0
    assert generator.telemetry_count == 1
    assert generator.telemetry_since(start) == client.history
    with pytest.raises(ValueError, match="negative"):
        generator.telemetry_since(-1)
