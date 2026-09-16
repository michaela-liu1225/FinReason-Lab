from __future__ import annotations

import json
import urllib.error

import pytest

from finreason.generator import (
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
        payload = {"choices": [{"message": {"content": '{"program":"add(1,2)"}'}}]}
        return FakeHTTPResponse(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleChatClient(
        "http://localhost:8001/v1/",
        "local-model",
        api_key="test-key",
        timeout_seconds=3,
    )

    content = client.complete([{"role": "user", "content": "question"}])

    assert content == '{"program":"add(1,2)"}'
    assert captured["request"].full_url.endswith("/v1/chat/completions")
    assert captured["request"].get_header("Authorization") == "Bearer test-key"
    assert captured["timeout"] == 3


def test_chat_client_reports_transport_and_response_errors(monkeypatch) -> None:
    client = OpenAICompatibleChatClient("http://localhost/v1", "model", max_response_bytes=10)

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )
    with pytest.raises(ModelClientError, match="request failed"):
        client.complete([])

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
