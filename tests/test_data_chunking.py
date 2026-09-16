from __future__ import annotations

import json

import pytest

from finreason.chunking import chunk_example, chunk_examples
from finreason.data import FinQADataError, load_finqa, load_finqa_json, load_finqa_jsonl


def _raw_record() -> dict:
    # This mirrors the official FinQA train/dev/test record structure.
    return {
        "pre_text": [
            "The company reports results in the table below.",
            "Revenue rose during the year.",
        ],
        "post_text": ["Management expects growth to continue."],
        "filename": "ACME/2023/page_12.pdf",
        "table": [
            ["(in millions)", "2023", "2022"],
            ["Revenue", "$120", "$100"],
            ["Operating margin", "25%", "20%"],
        ],
        "qa": {
            "question": "What was the increase in revenue?",
            "answer": "20%",
            "exe_ans": 0.2,
            "explanation": "This is intentionally not retrieval context.",
            "program": "subtract(120, 100), divide(#0, 100)",
            "gold_inds": {
                "text_1": "SUPERVISION-TEXT-ONLY",
                "table_1": "SUPERVISION-TABLE-ONLY",
            },
        },
        "id": "acme-2023-q1",
    }


def test_load_official_json_and_keep_labels_separate(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(json.dumps([_raw_record()]), encoding="utf-8")

    examples = load_finqa(path)

    assert len(examples) == 1
    example = examples[0]
    assert example.example_id == "acme-2023-q1"
    assert example.report_id == "ACME/2023/page_12.pdf"
    assert example.pre_text == (
        "The company reports results in the table below.",
        "Revenue rose during the year.",
    )
    assert example.post_text == ("Management expects growth to continue.",)
    assert example.answer == "20%"
    assert example.metadata["execution_answer"] == "0.2"
    assert example.program == "subtract(120, 100), divide(#0, 100)"
    assert example.gold_evidence_ids == ("text_1", "table_1")
    assert example.gold_evidence_text == {
        "text_1": "SUPERVISION-TEXT-ONLY",
        "table_1": "SUPERVISION-TABLE-ONLY",
    }

    chunks = chunk_example(example)
    assert [chunk.evidence_id for chunk in chunks] == [
        "text_0",
        "text_1",
        "text_2",
        "table_0",
        "table_1",
        "table_2",
    ]
    indexed_text = "\n".join(chunk.text for chunk in chunks)
    assert "20%" in indexed_text  # legitimate table source content remains
    assert "subtract(120, 100)" not in indexed_text
    assert "SUPERVISION-TEXT-ONLY" not in indexed_text
    assert "SUPERVISION-TABLE-ONLY" not in indexed_text
    assert "intentionally not retrieval context" not in indexed_text


def test_table_rows_are_header_and_unit_aware(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(json.dumps([_raw_record()]), encoding="utf-8")
    example = load_finqa_json(path, source_format="raw")[0]

    rows = {chunk.evidence_id: chunk for chunk in chunk_example(example)}
    revenue = rows["table_1"]

    assert revenue.kind == "table_row"
    assert revenue.table_id == "table_0"
    assert revenue.row_index == 1
    assert revenue.source_index == 1
    assert revenue.metadata["headers"] == ("(in millions)", "2023", "2022")
    assert revenue.metadata["row_label"] == "Revenue"
    assert revenue.metadata["unit"] == "in millions"
    assert revenue.text == "Row: Revenue | 2023: $120 | 2022: $100 | Unit: in millions"


def test_optional_cell_chunks_have_stable_provenance(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(json.dumps([_raw_record()]), encoding="utf-8")
    example = load_finqa(path)[0]

    chunks = chunk_example(example, include_cells=True)
    cells = [chunk for chunk in chunks if chunk.kind == "table_cell"]

    assert len(cells) == 9
    value = next(chunk for chunk in cells if chunk.evidence_id == "table_1_cell_1")
    assert value.row_index == 1
    assert value.column_index == 1
    assert value.text == (
        "Row: Revenue | Column: 2023 | Value: $120 | Unit: in millions"
    )
    assert value.metadata["header"] == "2023"


def test_flattened_jsonl_compatibility_does_not_index_targets(tmp_path):
    flattened = {
        "id": "flat-1",
        "question": "What is operating income?",
        "context": "Operating income was $42 million.",
        "answer": "42",
        "source_dataset": "finqa",
        "program": ["const(42)"],
        "evidence_indices": {"text_0": "Operating income was $42 million."},
    }
    path = tmp_path / "train.jsonl"
    path.write_text("\n" + json.dumps(flattened) + "\n\n", encoding="utf-8")

    example = load_finqa_jsonl(path, source_format="flattened")[0]
    chunks = chunk_example(example)

    assert example.pre_text == ("Operating income was $42 million.",)
    assert example.table == ()
    assert example.program == "const(42)"
    assert example.gold_evidence_ids == ("text_0",)
    assert example.metadata == {
        "source_format": "flattened",
        "source_dataset": "finqa",
    }
    assert [(chunk.evidence_id, chunk.text) for chunk in chunks] == [
        ("text_0", "Operating income was $42 million.")
    ]
    assert "const(42)" not in chunks[0].text


def test_flattened_structured_context_preserves_raw_style_ids(tmp_path):
    record = {
        "id": "structured-flat",
        "question": "Which year was larger?",
        "pre_text": ["Results follow."],
        "table": [["", "2023", "2022"], ["Sales", "9", "7"]],
        "post_text": ["End of report."],
        "answer": "2023",
        "program": "greater(9, 7)",
        "evidence_indices": ["text_1", "table_1", "table_1"],
    }
    path = tmp_path / "flat.jsonl"
    path.write_text(json.dumps(record), encoding="utf-8")

    example = load_finqa(path)[0]

    assert example.gold_evidence_ids == ("text_1", "table_1")
    assert [chunk.evidence_id for chunk in chunk_example(example)] == [
        "text_0",
        "text_1",
        "table_0",
        "table_1",
    ]


def test_chunking_is_deterministic_and_preserves_example_order(tmp_path):
    first = _raw_record()
    second = _raw_record()
    second["id"] = "acme-2023-q2"
    second["qa"] = dict(second["qa"], question="What was the margin?")
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"data": [first, second]}), encoding="utf-8")
    examples = load_finqa(path)

    once = chunk_examples(examples, include_cells=True)
    twice = chunk_examples(examples, include_cells=True)

    assert once == twice
    assert once[:15] == chunk_example(examples[0], include_cells=True)
    assert once[15:] == chunk_example(examples[1], include_cells=True)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"pre_text": [], "table": [], "qa": {}}, "question"),
        (
            {"pre_text": [], "table": ["not-a-row"], "qa": {"question": "Q"}},
            "expected a list of cells",
        ),
        (
            {
                "question": "Q",
                "context": "C",
                "evidence_indices": ["not_an_evidence_id"],
            },
            "expected an evidence ID",
        ),
    ],
)
def test_validation_errors_include_record_location(tmp_path, record, message):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([record]), encoding="utf-8")

    with pytest.raises(FinQADataError) as error:
        load_finqa(path)

    assert "record 1" in str(error.value)
    assert message in str(error.value)


def test_malformed_jsonl_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('\n{"question": "ok", "context": "c"}\n{bad}\n', encoding="utf-8")

    with pytest.raises(FinQADataError, match=r"line 3"):
        load_finqa(path)


def test_missing_path_and_invalid_schema_selection(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_finqa(tmp_path / "missing.json")
    path = tmp_path / "flat.jsonl"
    path.write_text('{"question": "Q", "context": "C"}', encoding="utf-8")
    with pytest.raises(FinQADataError, match="expected raw record"):
        load_finqa(path, source_format="raw")
    with pytest.raises(ValueError, match="source_format"):
        load_finqa(path, source_format="unknown")  # type: ignore[arg-type]


def test_empty_and_ragged_sources_keep_original_indices(tmp_path):
    record = _raw_record()
    record["pre_text"] = ["first", "   "]
    record["post_text"] = ["third"]
    record["table"] = [["", "2023", "2022"], ["Revenue", "10"]]
    path = tmp_path / "ragged.json"
    path.write_text(json.dumps([record]), encoding="utf-8")

    chunks = chunk_example(load_finqa(path)[0], include_cells=True)

    assert [chunk.evidence_id for chunk in chunks if chunk.kind == "text"] == [
        "text_0",
        "text_2",
    ]
    row = next(chunk for chunk in chunks if chunk.evidence_id == "table_1")
    assert row.text == "Row: Revenue | 2023: 10"
    assert any(chunk.evidence_id == "table_1_cell_1" for chunk in chunks)


def test_official_unaddressable_negative_evidence_id_is_preserved_not_indexed(tmp_path):
    record = _raw_record()
    record["qa"]["gold_inds"] = {
        "table_1": "Revenue ; $120 ; $100",
        "text_-1": "annotation without a source passage",
    }
    path = tmp_path / "negative-id.json"
    path.write_text(json.dumps([record]), encoding="utf-8")

    example = load_finqa(path)[0]
    chunks = chunk_example(example)

    assert example.gold_evidence_ids == ("table_1", "text_-1")
    assert "text_-1" not in {chunk.evidence_id for chunk in chunks}
    assert "annotation without a source passage" not in "\n".join(
        chunk.text for chunk in chunks
    )
