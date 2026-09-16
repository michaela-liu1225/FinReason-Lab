from __future__ import annotations

import json

from fastapi.testclient import TestClient

from finreason.api import create_app
from finreason.workflow import GeneratedProgram


def _dataset(path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "sample-1",
                    "filename": "report.pdf",
                    "pre_text": ["Revenue rose during the year."],
                    "post_text": [],
                    "table": [
                        ["metric", "2021", "2022"],
                        ["revenue", "100", "125"],
                    ],
                    "qa": {
                        "question": "By how much did revenue increase?",
                        "answer": "25",
                        "exe_ans": 25,
                        "program": "subtract(125, 100)",
                        "gold_inds": {"table_1": "gold text must not enter retrieval"},
                    },
                }
            ]
        ),
        encoding="utf-8",
    )


class FixedGenerator:
    def generate(self, request):
        assert all("gold text" not in hit.chunk.text for hit in request.evidence)
        return GeneratedProgram("subtract(125, 100)", ("table_1",))


def test_health_and_retrieval_endpoints(tmp_path) -> None:
    dataset = tmp_path / "dev.json"
    _dataset(dataset)
    client = TestClient(create_app(dataset_path=dataset))

    health = client.get("/health")
    response = client.post(
        "/v1/retrieve",
        json={"example_id": "sample-1", "top_k": 3},
        headers={"x-trace-id": "test-trace"},
    )

    assert health.json()["dataset_examples"] == 1
    assert response.status_code == 200
    assert response.headers["x-trace-id"] == "test-trace"
    assert "table_1" in {hit["evidence_id"] for hit in response.json()["hits"]}
    assert "gold" not in json.dumps(response.json()).casefold()


def test_reason_endpoint_runs_bounded_tool_workflow(tmp_path) -> None:
    dataset = tmp_path / "dev.json"
    _dataset(dataset)
    client = TestClient(create_app(dataset_path=dataset, generator=FixedGenerator()))

    response = client.post(
        "/v1/reason",
        json={"example_id": "sample-1", "top_k": 3, "max_repairs": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["answer"] == "25"
    assert body["program"] == "subtract(125, 100)"
    assert body["citations"] == ["table_1"]
    assert body["trace"][-1]["outcome"] == "answered"


def test_reason_requires_model_and_inline_context_is_validated(tmp_path) -> None:
    dataset = tmp_path / "dev.json"
    _dataset(dataset)
    client = TestClient(create_app(dataset_path=dataset))

    assert client.post("/v1/reason", json={"example_id": "sample-1"}).status_code == 503
    assert client.post("/v1/retrieve", json={"question": "missing context"}).status_code == 422
    inline = client.post(
        "/v1/retrieve",
        json={
            "question": "What was revenue in 2022?",
            "table": [["metric", "2022"], ["revenue", "125"]],
        },
    )
    assert inline.status_code == 200
    assert inline.json()["example_id"] == "inline"
