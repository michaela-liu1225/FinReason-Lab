"""Deterministic, table-aware chunking for canonical FinQA examples."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from .schema import EvidenceChunk, FinQAExample


def chunk_example(
    example: FinQAExample,
    *,
    include_cells: bool = False,
) -> list[EvidenceChunk]:
    """Create retrievable text and table chunks for one FinQA example.

    Text indices span ``pre_text`` followed by ``post_text``, mirroring the
    official ``text_N`` gold identifiers.  Table rows similarly retain their
    original zero-based ``table_N`` identifiers.  Answers, programs, questions,
    and gold labels are deliberately never read by this function.
    """

    chunks: list[EvidenceChunk] = []
    chunks.extend(_text_chunks(example))
    chunks.extend(_table_row_chunks(example))
    if include_cells:
        chunks.extend(_table_cell_chunks(example))
    return chunks


def chunk_examples(
    examples: Iterable[FinQAExample],
    *,
    include_cells: bool = False,
) -> list[EvidenceChunk]:
    """Chunk examples in input order without global state or re-numbering."""

    chunks: list[EvidenceChunk] = []
    for example in examples:
        if not isinstance(example, FinQAExample):
            raise TypeError(
                "chunk_examples expects FinQAExample instances; "
                f"got {type(example).__name__}"
            )
        chunks.extend(chunk_example(example, include_cells=include_cells))
    return chunks


def _text_chunks(example: FinQAExample) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    passages = (
        *(("pre", index, text) for index, text in enumerate(example.pre_text)),
        *(("post", index, text) for index, text in enumerate(example.post_text)),
    )
    for source_index, (section, section_index, text) in enumerate(passages):
        clean_text = _clean(text)
        # Keep original numbering even when an empty source passage is omitted.
        if not clean_text:
            continue
        chunks.append(
            EvidenceChunk(
                evidence_id=f"text_{source_index}",
                report_id=example.report_id,
                kind="text",
                text=clean_text,
                source_index=source_index,
                metadata={"section": section, "section_index": section_index},
            )
        )
    return chunks


def _table_row_chunks(example: FinQAExample) -> list[EvidenceChunk]:
    if not example.table:
        return []
    headers = example.table[0]
    table_unit = _infer_unit((*headers,))
    chunks: list[EvidenceChunk] = []
    for row_index, row in enumerate(example.table):
        unit = _infer_unit((row[0],)) if row else None
        unit = unit or table_unit or _infer_unit(row[1:])
        text = _format_row(row_index, row, headers, unit)
        chunks.append(
            EvidenceChunk(
                evidence_id=f"table_{row_index}",
                report_id=example.report_id,
                kind="table_row",
                text=text,
                source_index=row_index,
                table_id="table_0",
                row_index=row_index,
                metadata={
                    "headers": tuple(headers),
                    "row_label": row[0] if row else "",
                    **({"unit": unit} if unit else {}),
                },
            )
        )
    return chunks


def _table_cell_chunks(example: FinQAExample) -> list[EvidenceChunk]:
    if not example.table:
        return []
    headers = example.table[0]
    table_unit = _infer_unit((*headers,))
    chunks: list[EvidenceChunk] = []
    for row_index, row in enumerate(example.table):
        row_label = _clean(row[0]) if row else f"Row {row_index}"
        row_unit = (_infer_unit((row[0],)) if row else None) or table_unit
        for column_index, value in enumerate(row):
            header = _header_at(headers, column_index)
            unit = row_unit or _infer_unit((value,))
            chunks.append(
                EvidenceChunk(
                    evidence_id=f"table_{row_index}_cell_{column_index}",
                    report_id=example.report_id,
                    kind="table_cell",
                    text=_format_cell(
                        row_index=row_index,
                        column_index=column_index,
                        row_label=row_label,
                        header=header,
                        value=value,
                        unit=unit,
                    ),
                    source_index=row_index,
                    table_id="table_0",
                    row_index=row_index,
                    column_index=column_index,
                    metadata={
                        "header": header,
                        "row_label": row_label,
                        **({"unit": unit} if unit else {}),
                    },
                )
            )
    return chunks


def _format_row(
    row_index: int,
    row: Sequence[str],
    headers: Sequence[str],
    unit: str | None,
) -> str:
    if row_index == 0:
        columns = [
            f"Column {index}: {_clean(value) or '[empty]'}"
            for index, value in enumerate(row)
        ]
        parts = ["Table header", *columns]
    else:
        label = _clean(row[0]) if row else ""
        parts = [f"Row: {label or f'Row {row_index}'}"]
        for column_index in range(1, len(row)):
            parts.append(
                f"{_header_at(headers, column_index)}: "
                f"{_clean(row[column_index]) or '[empty]'}"
            )
        # A one-column table still needs to retain its only value.
        if len(row) == 1 and label:
            parts.append(f"Value: {label}")
    if unit:
        parts.append(f"Unit: {unit}")
    return " | ".join(parts)


def _format_cell(
    *,
    row_index: int,
    column_index: int,
    row_label: str,
    header: str,
    value: str,
    unit: str | None,
) -> str:
    parts = [
        f"Row: {row_label or f'Row {row_index}'}",
        f"Column: {header or f'Column {column_index}'}",
        f"Value: {_clean(value) or '[empty]'}",
    ]
    if unit:
        parts.append(f"Unit: {unit}")
    return " | ".join(parts)


def _header_at(headers: Sequence[str], column_index: int) -> str:
    if column_index < len(headers):
        header = _clean(headers[column_index])
        if header:
            return header
    return f"Column {column_index}"


def _clean(value: Any) -> str:
    return " ".join(str(value).split())


_PARENTHETICAL_UNIT = re.compile(
    r"\(([^)]*(?:thousand|million|billion|percent|percentage|per share|"
    r"usd|eur|gbp|dollars?|euros?|pounds?)[^)]*)\)",
    re.IGNORECASE,
)
_PLAIN_UNIT = re.compile(
    r"\b((?:usd|eur|gbp|dollars?|euros?|pounds?)?\s*(?:in\s+)?"
    r"(?:thousands?|millions?|billions?|percent(?:age)?|per share))\b",
    re.IGNORECASE,
)


def _infer_unit(values: Sequence[str]) -> str | None:
    for raw_value in values:
        value = _clean(raw_value)
        match = _PARENTHETICAL_UNIT.search(value) or _PLAIN_UNIT.search(value)
        if match:
            return _clean(match.group(1))

    nonempty = [_clean(value) for value in values if _clean(value)]
    if nonempty and all("%" in value for value in nonempty):
        return "%"
    currency_symbols = (("$", "USD ($)"), ("€", "EUR (€)"), ("£", "GBP (£)"))
    for symbol, label in currency_symbols:
        if nonempty and all(symbol in value for value in nonempty):
            return label
    return None
