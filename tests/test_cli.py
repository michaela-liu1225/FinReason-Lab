from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from finreason.cli import (
    _parse_ks,
    _percentile,
    reasoning_evaluation_main,
    run_executor_audit,
    run_retrieval_evaluation,
)


def _write_dataset(path) -> None:
    payload = [
        {
            "id": "report-1-q1",
            "filename": "report-1.pdf",
            "pre_text": ["Revenue increased in 2022.", "Nothing relevant here."],
            "post_text": [],
            "table": [
                ["metric", "2021", "2022"],
                ["revenue", "100", "125"],
            ],
            "qa": {
                "question": "What was revenue in 2022?",
                "answer": "125",
                "exe_ans": 125,
                "program": "add(125, const_0)",
                "gold_inds": {"table_1": "revenue row"},
            },
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_retrieval_evaluation_returns_auditable_report(tmp_path) -> None:
    dataset = tmp_path / "dev.json"
    _write_dataset(dataset)

    report = run_retrieval_evaluation(dataset, top_k=2, ks=(1, 2))

    assert report["aggregate"]["query_count"] == 1
    assert report["aggregate"]["recall_at_k"]["2"] == 1.0
    assert report["aggregate"]["ranking_depth"] == 2
    assert "mean_reciprocal_rank_at_k" in report["aggregate"]
    assert "mean_reciprocal_rank" not in report["aggregate"]
    row = report["per_query"]["report-1-q1"]
    assert row["gold_evidence_ids"] == ["table_1"]
    assert all("answer" not in hit for hit in row["hits"])


def test_retrieval_evaluation_rejects_unmappable_flattened_gold(tmp_path) -> None:
    dataset = tmp_path / "flattened.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "flattened-missing-table",
                "question": "What was revenue?",
                "context": "Revenue was 125.",
                "answer": "125",
                "program": "add(125, const_0)",
                "evidence_indices": {"table_1": "revenue row"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        run_retrieval_evaluation(dataset, top_k=1, ks=(1,))

    assert "flattened-missing-table" in str(error.value)
    assert "missing IDs: ['table_1']" in str(error.value)


def test_retrieval_evaluation_requires_nonempty_gold(tmp_path) -> None:
    dataset = tmp_path / "no-gold.json"
    payload = [
        {
            "id": "no-gold-example",
            "filename": "report.pdf",
            "pre_text": ["Revenue was 125."],
            "post_text": [],
            "table": [],
            "qa": {
                "question": "What was revenue?",
                "answer": "125",
                "program": "add(125, const_0)",
                "gold_inds": {},
            },
        }
    ]
    dataset.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as error:
        run_retrieval_evaluation(dataset, top_k=1, ks=(1,))

    assert "no-gold-example" in str(error.value)
    assert "non-empty gold labels" in str(error.value)
    assert "missing IDs: []" in str(error.value)


def test_retrieval_evaluation_allows_official_text_minus_one_anomaly(tmp_path) -> None:
    dataset = tmp_path / "dev.json"
    _write_dataset(dataset)
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    payload[0]["qa"]["gold_inds"]["text_-1"] = "unaddressable official annotation"
    dataset.write_text(json.dumps(payload), encoding="utf-8")

    report = run_retrieval_evaluation(dataset, top_k=2, ks=(1, 2))

    assert report["aggregate"]["query_count"] == 1
    assert report["per_query"]["report-1-q1"]["gold_evidence_ids"] == [
        "table_1",
        "text_-1",
    ]


def test_cli_validation_and_percentile() -> None:
    assert _percentile([4.0, 1.0, 2.0, 3.0], 0.5) == 2.0
    assert _parse_ks("1, 3,5") == (1, 3, 5)
    with pytest.raises(ValueError, match="embedding_model"):
        run_retrieval_evaluation("unused", method="hybrid", embedding_model=None)
    with pytest.raises(ValueError, match="method='hybrid'"):
        run_retrieval_evaluation("unused", method="bm25", embedding_model="unused")


def test_executor_audit_is_explicitly_gold_program_only(tmp_path) -> None:
    dataset = tmp_path / "dev.json"
    _write_dataset(dataset)

    report = run_executor_audit(dataset)

    assert report["config"]["mode"] == "gold_program_executor_audit"
    assert report["summary"]["execution_success_rate"] == 1.0
    assert report["summary"]["answer_match_rate"] == 1.0


def test_reasoning_cli_requires_an_immutable_model_revision(tmp_path, monkeypatch, capsys) -> None:
    dataset = tmp_path / "dev.json"
    _write_dataset(dataset)
    monkeypatch.delenv("FINREASON_MODEL_REVISION", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["finreason-eval-reasoning", "--dataset", str(dataset)],
    )

    with pytest.raises(SystemExit, match="2"):
        reasoning_evaluation_main()

    assert "model-revision" in capsys.readouterr().err


def test_reasoning_cli_requires_an_explicit_thinking_mode(tmp_path, monkeypatch, capsys) -> None:
    dataset = tmp_path / "dev.json"
    _write_dataset(dataset)
    monkeypatch.setenv("FINREASON_MODEL_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("FINREASON_MODEL_NAME", "fixed-model")
    monkeypatch.setenv("FINREASON_MODEL_REVISION", "model-commit-123")
    monkeypatch.setenv("FINREASON_MODEL_SEED", "42")
    monkeypatch.delenv("FINREASON_MODEL_ENABLE_THINKING", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["finreason-eval-reasoning", "--dataset", str(dataset)],
    )

    with pytest.raises(SystemExit, match="2"):
        reasoning_evaluation_main()

    assert "FINREASON_MODEL_ENABLE_THINKING" in capsys.readouterr().err


def test_reasoning_cli_writes_fixed_model_report_and_manifest(
    tmp_path, monkeypatch, capsys
) -> None:
    dataset = tmp_path / "dev.json"
    output_dir = tmp_path / "artifacts"
    _write_dataset(dataset)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            response = {
                "id": "request-1",
                "model": "served-fixed-model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "program": "add(125, const_0)",
                                    "citations": ["table_1"],
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28,
                },
            }
            return json.dumps(response).encode("utf-8")

    captured_requests = []

    def fake_urlopen(request, timeout):
        captured_requests.append(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("FINREASON_MODEL_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("FINREASON_MODEL_NAME", "fixed-model")
    monkeypatch.setenv("FINREASON_MODEL_API_KEY", "must-not-be-persisted")
    monkeypatch.setenv("FINREASON_MODEL_SEED", "42")
    monkeypatch.setenv("FINREASON_MODEL_ENABLE_THINKING", "false")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finreason-eval-reasoning",
            "--dataset",
            str(dataset),
            "--model-revision",
            "model-commit-123",
            "--max-repairs",
            "0",
            "--output-dir",
            str(output_dir),
        ],
    )

    reasoning_evaluation_main()

    command_output = json.loads(capsys.readouterr().out)
    report_path = Path(command_output["report"])
    manifest_path = Path(command_output["manifest"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report["summary"]["accuracy"] == 1.0
    assert report["summary"]["model_calls"] == 1
    assert report["config"]["model_config"]["model_revision"] == "model-commit-123"
    assert report["config"]["model_config"]["seed"] == 42
    assert report["config"]["model_config"]["enable_thinking"] is False
    assert captured_requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert manifest["metrics"]["summary"]["accuracy"] == 1.0
    assert "must-not-be-persisted" not in report_path.read_text(encoding="utf-8")
    assert "must-not-be-persisted" not in manifest_path.read_text(encoding="utf-8")
