"""Canonical, dependency-free loaders for FinQA data.

The official FinQA release stores examples in a JSON array and nests labels in
``qa``.  The original project also produced a flattened JSONL representation.
This module accepts both forms and normalizes them to :class:`FinQAExample`
without mixing supervision fields into the source context.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from .schema import FinQAExample, as_tuple_rows

SourceFormat = Literal["auto", "raw", "flattened"]
PathLike = str | os.PathLike[str]


class FinQADataError(ValueError):
    """Raised when an input file is valid JSON but not valid FinQA data."""


def load_finqa(
    path: PathLike,
    *,
    source_format: SourceFormat = "auto",
) -> list[FinQAExample]:
    """Load official FinQA JSON or the project's flattened JSONL format.

    The transport format is inferred from the filename (``.jsonl`` means one
    record per line); the record schema is inferred independently.  Set
    ``source_format`` to make schema validation strict.  Errors include the
    path and record/line number so bad generated datasets are easy to audit.
    """

    if source_format not in {"auto", "raw", "flattened"}:
        raise ValueError(
            "source_format must be one of 'auto', 'raw', or 'flattened'; "
            f"got {source_format!r}"
        )

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"FinQA data file not found: {data_path}")
    if not data_path.is_file():
        raise FinQADataError(f"FinQA data path is not a file: {data_path}")

    if data_path.suffix.lower() in {".jsonl", ".ndjson"}:
        located_records = _read_jsonl(data_path)
    else:
        located_records = _read_json(data_path)

    examples: list[FinQAExample] = []
    for record_number, record in located_records:
        location = f"{data_path} (record {record_number})"
        record_mapping = _require_mapping(record, location)
        inferred = _infer_source_format(record_mapping)
        if source_format != "auto" and inferred != source_format:
            raise FinQADataError(
                f"{location}: expected {source_format} record, found {inferred} record"
            )
        if inferred == "raw":
            example = _parse_raw(record_mapping, record_number, location)
        else:
            example = _parse_flattened(record_mapping, record_number, location)
        examples.append(example)
    return examples


def load_finqa_json(
    path: PathLike,
    *,
    source_format: SourceFormat = "auto",
) -> list[FinQAExample]:
    """Load FinQA records from a JSON document.

    This explicit entry point rejects JSONL regardless of the file extension.
    It is useful for callers that want transport-format validation.
    """

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"FinQA data file not found: {data_path}")
    records = _read_json(data_path)
    return _parse_records(records, data_path, source_format)


def load_finqa_jsonl(
    path: PathLike,
    *,
    source_format: SourceFormat = "auto",
) -> list[FinQAExample]:
    """Load FinQA records from a line-delimited JSON document."""

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"FinQA data file not found: {data_path}")
    records = _read_jsonl(data_path)
    return _parse_records(records, data_path, source_format)


def _parse_records(
    located_records: Iterable[tuple[int, Any]],
    data_path: Path,
    source_format: SourceFormat,
) -> list[FinQAExample]:
    if source_format not in {"auto", "raw", "flattened"}:
        raise ValueError(
            "source_format must be one of 'auto', 'raw', or 'flattened'; "
            f"got {source_format!r}"
        )
    examples: list[FinQAExample] = []
    for record_number, record in located_records:
        location = f"{data_path} (record {record_number})"
        record_mapping = _require_mapping(record, location)
        inferred = _infer_source_format(record_mapping)
        if source_format != "auto" and inferred != source_format:
            raise FinQADataError(
                f"{location}: expected {source_format} record, found {inferred} record"
            )
        parser = _parse_raw if inferred == "raw" else _parse_flattened
        examples.append(parser(record_mapping, record_number, location))
    return examples


def _read_json(path: Path) -> list[tuple[int, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except UnicodeDecodeError as exc:
        raise FinQADataError(f"{path}: input is not valid UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FinQADataError(
            f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if isinstance(payload, Mapping) and "data" in payload:
        payload = payload["data"]
    elif isinstance(payload, Mapping):
        payload = [payload]

    if not isinstance(payload, list):
        raise FinQADataError(
            f"{path}: top-level JSON must be a record, a list of records, "
            "or an object with a 'data' list"
        )
    return list(enumerate(payload, start=1))


def _read_jsonl(path: Path) -> list[tuple[int, Any]]:
    records: list[tuple[int, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise FinQADataError(
                        f"{path}: invalid JSON on line {line_number}, "
                        f"column {exc.colno}: {exc.msg}"
                    ) from exc
                records.append((line_number, record))
    except UnicodeDecodeError as exc:
        raise FinQADataError(f"{path}: input is not valid UTF-8: {exc}") from exc
    return records


def _infer_source_format(record: Mapping[str, Any]) -> Literal["raw", "flattened"]:
    # Official records always nest the question and labels inside ``qa``.
    if "qa" in record:
        if not isinstance(record["qa"], Mapping):
            raise FinQADataError("field 'qa' must be an object")
        return "raw"
    if "question" in record or "context" in record:
        return "flattened"
    raise FinQADataError(
        "record is neither official FinQA (missing 'qa') nor flattened FinQA "
        "(missing 'question'/'context')"
    )


def _parse_raw(
    record: Mapping[str, Any],
    record_number: int,
    location: str,
) -> FinQAExample:
    qa = _require_mapping(record.get("qa"), f"{location}.qa")
    example_id = _identifier(record.get("id"), f"finqa_{record_number}", location)
    report_id = _identifier(
        record.get("report_id") or record.get("filename"), example_id, location
    )

    question = _required_text(qa.get("question"), f"{location}.qa.question")
    pre_text = _text_sequence(record.get("pre_text"), f"{location}.pre_text")
    post_text = _text_sequence(record.get("post_text"), f"{location}.post_text")
    table = _table(record.get("table"), f"{location}.table")
    answer = _optional_scalar_text(
        qa.get("answer", qa.get("exe_ans", "")), f"{location}.qa.answer"
    )
    program = _program_text(qa.get("program", ""), f"{location}.qa.program")
    gold_ids, gold_text = _gold_evidence(
        qa.get("gold_inds", qa.get("evidence_indices")),
        f"{location}.qa.gold_inds",
    )

    metadata: dict[str, Any] = {"source_format": "raw"}
    if record.get("filename") is not None:
        metadata["filename"] = str(record["filename"])
    if qa.get("exe_ans") is not None:
        metadata["execution_answer"] = _optional_scalar_text(
            qa["exe_ans"], f"{location}.qa.exe_ans"
        )

    return FinQAExample(
        example_id=example_id,
        report_id=report_id,
        question=question,
        pre_text=pre_text,
        table=table,
        post_text=post_text,
        answer=answer,
        program=program,
        gold_evidence_ids=gold_ids,
        gold_evidence_text=gold_text,
        metadata=metadata,
    )


def _parse_flattened(
    record: Mapping[str, Any],
    record_number: int,
    location: str,
) -> FinQAExample:
    example_id = _identifier(record.get("id"), f"finqa_{record_number}", location)
    report_id = _identifier(
        record.get("report_id") or record.get("filename"), example_id, location
    )
    question = _required_text(record.get("question"), f"{location}.question")

    has_structured_context = any(
        field in record for field in ("pre_text", "post_text", "table")
    )
    if has_structured_context:
        pre_text = _text_sequence(record.get("pre_text"), f"{location}.pre_text")
        post_text = _text_sequence(record.get("post_text"), f"{location}.post_text")
        table = _table(record.get("table"), f"{location}.table")
    else:
        context = _optional_scalar_text(record.get("context", ""), f"{location}.context")
        pre_text = (context,) if context else ()
        post_text = ()
        table = ()

    answer = _optional_scalar_text(record.get("answer", ""), f"{location}.answer")
    program = _program_text(record.get("program", ""), f"{location}.program")
    gold_ids, gold_text = _gold_evidence(
        record.get("evidence_indices", record.get("gold_inds")),
        f"{location}.evidence_indices",
    )

    metadata: dict[str, Any] = {"source_format": "flattened"}
    for field in ("source_dataset", "filename", "split"):
        value = record.get(field)
        if value is not None:
            metadata[field] = str(value)

    return FinQAExample(
        example_id=example_id,
        report_id=report_id,
        question=question,
        pre_text=pre_text,
        table=table,
        post_text=post_text,
        answer=answer,
        program=program,
        gold_evidence_ids=gold_ids,
        gold_evidence_text=gold_text,
        metadata=metadata,
    )


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinQADataError(f"{location}: expected a JSON object")
    return value


def _identifier(value: Any, fallback: str, location: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, Mapping | list | tuple):
        raise FinQADataError(f"{location}: identifier must be a scalar")
    result = str(value).strip()
    return result or fallback


def _required_text(value: Any, location: str) -> str:
    result = _optional_scalar_text(value, location).strip()
    if not result:
        raise FinQADataError(f"{location}: field is required and cannot be empty")
    return result


def _optional_scalar_text(value: Any, location: str) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping | list | tuple):
        raise FinQADataError(f"{location}: expected a scalar value")
    return str(value)


def _text_sequence(value: Any, location: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, bytes | bytearray):
        raise FinQADataError(f"{location}: expected a string or list of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise FinQADataError(f"{location}[{index}]: expected a string")
        result.append(item)
    return tuple(result)


def _table(value: Any, location: str) -> tuple[tuple[str, ...], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise FinQADataError(f"{location}: expected a list of table rows")
    validated: list[list[Any]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, str | bytes | bytearray):
            raise FinQADataError(f"{location}[{row_index}]: expected a list of cells")
        cells: list[Any] = []
        for column_index, cell in enumerate(row):
            if isinstance(cell, Mapping | list | tuple):
                raise FinQADataError(
                    f"{location}[{row_index}][{column_index}]: cell must be scalar"
                )
            cells.append("" if cell is None else cell)
        validated.append(cells)
    return as_tuple_rows(validated)


def _program_text(value: Any, location: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        if any(isinstance(part, Mapping | list | tuple) for part in value):
            raise FinQADataError(f"{location}: program steps must be scalar")
        return ", ".join(str(part) for part in value)
    if isinstance(value, Mapping):
        raise FinQADataError(f"{location}: expected a string or list of steps")
    return str(value)


# One official training example contains ``text_-1``.  It denotes an annotation
# with no addressable source passage, so we preserve the label for auditing but
# never synthesize a chunk from its gold text.
_EVIDENCE_ID = re.compile(r"^(?:text|table)_-?\d+$")


def _gold_evidence(
    value: Any,
    location: str,
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    """Normalize labels while keeping them entirely separate from context."""

    if value is None:
        return (), {}

    ids: list[str] = []
    evidence_text: dict[str, str] = {}

    if isinstance(value, Mapping):
        # Official FinQA: {"text_2": "...", "table_1": "..."}.
        if all(_EVIDENCE_ID.fullmatch(str(key)) for key in value):
            for key, text in value.items():
                evidence_id = str(key)
                ids.append(evidence_id)
                if text is not None and not isinstance(text, Mapping | list | tuple):
                    evidence_text[evidence_id] = str(text)
            return tuple(_deduplicate(ids)), evidence_text

        # Some flattened datasets group row indices by source kind.
        if set(value).issubset({"text", "table"}):
            for kind in ("text", "table"):
                entries = value.get(kind, ())
                if entries is None:
                    continue
                if not isinstance(entries, Sequence) or isinstance(
                    entries, str | bytes | bytearray
                ):
                    raise FinQADataError(f"{location}.{kind}: expected a list")
                for entry in entries:
                    token = str(entry)
                    ids.append(token if _EVIDENCE_ID.fullmatch(token) else f"{kind}_{token}")
            return tuple(_deduplicate(ids)), evidence_text

        raise FinQADataError(
            f"{location}: evidence object keys must be text_N/table_N IDs or "
            "the grouped keys 'text'/'table'"
        )

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, entry in enumerate(value):
            if not isinstance(entry, str) or not _EVIDENCE_ID.fullmatch(entry):
                raise FinQADataError(
                    f"{location}[{index}]: expected an evidence ID like text_2 or table_1"
                )
            ids.append(entry)
        return tuple(_deduplicate(ids)), evidence_text

    raise FinQADataError(
        f"{location}: expected an evidence mapping, a list of IDs, or null"
    )


def _deduplicate(values: Iterable[str]) -> list[str]:
    # dict preserves insertion order on every supported Python version.
    return list(dict.fromkeys(values))
