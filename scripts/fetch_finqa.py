#!/usr/bin/env python3
"""Fetch an official FinQA split without committing the dataset to this mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

UPSTREAM_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
BASE_URL = f"https://raw.githubusercontent.com/czyssrs/FinQA/{UPSTREAM_COMMIT}/dataset"
KNOWN_SPLITS = {"train", "dev", "test", "private_test"}
EXPECTED_SHA256 = {
    "dev": "a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(split: str, output_dir: Path, *, overwrite: bool = False) -> Path:
    if split not in KNOWN_SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {sorted(KNOWN_SPLITS)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{split}.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass --overwrite to replace it")

    url = f"{BASE_URL}/{split}.json"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{split}.", suffix=".json", dir=output_dir)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        urllib.request.urlretrieve(url, temporary)
        payload = json.loads(temporary.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"downloaded {url} is not a non-empty FinQA JSON array")
        actual_hash = sha256(temporary)
        expected_hash = EXPECTED_SHA256.get(split)
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(
                f"checksum mismatch for {split}: expected {expected_hash}, got {actual_hash}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "path": str(destination),
                "split": split,
                "examples": len(payload),
                "sha256": actual_hash,
                "source": url,
            },
            indent=2,
        )
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=sorted(KNOWN_SPLITS))
    parser.add_argument("--output-dir", type=Path, default=Path("raw_data/finqa"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    fetch(args.split, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
