from __future__ import annotations

import json

import pytest

from finreason.manifest import (
    build_run_manifest,
    canonical_json,
    code_sha256,
    config_fingerprint,
    file_sha256,
)


def test_canonical_json_and_fingerprint_ignore_mapping_order() -> None:
    left = {"top_k": 5, "retriever": "bm25"}
    right = {"retriever": "bm25", "top_k": 5}

    assert canonical_json(left) == canonical_json(right)
    assert config_fingerprint(left) == config_fingerprint(right)


def test_fingerprint_rejects_unhelpful_lengths() -> None:
    with pytest.raises(ValueError, match="between 8 and 64"):
        config_fingerprint({}, length=4)


def test_build_and_write_manifest(tmp_path) -> None:
    dataset = tmp_path / "dev.json"
    dataset.write_text('[{"id":"sample"}]\n', encoding="utf-8")

    manifest = build_run_manifest(
        config={"retriever": "bm25", "top_k": 3},
        dataset_path=dataset,
        repository=tmp_path,
        metrics={"mrr": 0.5},
    )
    output = manifest.write(tmp_path / "artifacts" / "manifest.json")
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert f"-{manifest.experiment_id}-" in manifest.run_id
    assert len(manifest.experiment_id) == 12
    assert saved["dataset_sha256"] == file_sha256(dataset)
    assert saved["code_sha256"] == code_sha256(tmp_path)
    assert saved["metrics"] == {"mrr": 0.5}
    assert saved["git_revision"] is None
    assert saved["git_dirty"] is None

    with pytest.raises(FileExistsError):
        manifest.write(output)


def test_experiment_identity_tracks_code_but_run_ids_are_unique(tmp_path) -> None:
    dataset = tmp_path / "dev.json"
    dataset.write_text("[]\n", encoding="utf-8")
    package = tmp_path / "finreason"
    package.mkdir()
    source = package / "example.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    first = build_run_manifest(config={"top_k": 3}, dataset_path=dataset, repository=tmp_path)
    second = build_run_manifest(config={"top_k": 3}, dataset_path=dataset, repository=tmp_path)

    assert first.experiment_id == second.experiment_id
    assert first.run_id != second.run_id

    source.write_text("VALUE = 2\n", encoding="utf-8")
    changed = build_run_manifest(config={"top_k": 3}, dataset_path=dataset, repository=tmp_path)
    assert changed.experiment_id != first.experiment_id
    assert changed.code_sha256 != first.code_sha256
