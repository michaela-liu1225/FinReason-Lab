"""Reproducible experiment identities and immutable run manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

_CODE_PATTERNS = ("finreason/**/*.py", "scripts/**/*.py", "pyproject.toml")


def canonical_json(value: Any) -> str:
    """Serialize config data deterministically for hashing and persistence."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_fingerprint(config: Mapping[str, Any], *, length: int = 12) -> str:
    if length < 8 or length > 64:
        raise ValueError("fingerprint length must be between 8 and 64")
    return hashlib.sha256(canonical_json(dict(config)).encode("utf-8")).hexdigest()[:length]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_sha256(repository: str | Path = ".") -> str:
    """Hash the experiment implementation independently of Git state.

    Paths as well as contents are included so renames and deletions change the
    digest.  This also covers uncommitted source files, unlike a commit hash.
    """

    root = Path(repository).resolve()
    files = sorted(
        {
            path
            for pattern in _CODE_PATTERNS
            for path in root.glob(pattern)
            if path.is_file()
        },
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _git_output(repository: str | Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout


def current_git_revision(repository: str | Path = ".") -> str | None:
    output = _git_output(repository, "rev-parse", "HEAD")
    if output is None:
        return None
    revision = output.strip()
    return revision or None


def git_is_dirty(repository: str | Path = ".") -> bool | None:
    """Return whether tracked or untracked files differ from ``HEAD``."""

    output = _git_output(repository, "status", "--porcelain", "--untracked-files=all")
    return None if output is None else bool(output.strip())


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    experiment_id: str
    created_at: str
    config: Mapping[str, Any]
    dataset_path: str
    dataset_sha256: str
    code_sha256: str
    git_revision: str | None = None
    git_dirty: bool | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
            handle.write("\n")
        return destination


def build_run_manifest(
    *,
    config: Mapping[str, Any],
    dataset_path: str | Path,
    repository: str | Path = ".",
    metrics: Mapping[str, Any] | None = None,
) -> RunManifest:
    source = Path(dataset_path).resolve()
    root = Path(repository).resolve()
    resolved_config = dict(config)
    dataset_digest = file_sha256(source)
    code_digest = code_sha256(root)
    experiment_id = config_fingerprint(
        {
            "config": resolved_config,
            "dataset_sha256": dataset_digest,
            "code_sha256": code_digest,
        }
    )
    created_at = datetime.now(timezone.utc)
    run_id = (
        f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-{experiment_id}-{uuid4().hex[:8]}"
    )
    return RunManifest(
        run_id=run_id,
        experiment_id=experiment_id,
        created_at=created_at.isoformat(),
        config=resolved_config,
        dataset_path=str(source),
        dataset_sha256=dataset_digest,
        code_sha256=code_digest,
        git_revision=current_git_revision(root),
        git_dirty=git_is_dirty(root),
        metrics=dict(metrics or {}),
    )
