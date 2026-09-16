from __future__ import annotations

import json

import pytest

from finreason.cli import _parse_ks, _percentile, run_executor_audit, run_retrieval_evaluation


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
