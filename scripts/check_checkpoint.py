#!/usr/bin/env python3
"""Verify the released StreamingWAM checkpoint before evaluation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


EXPECTED_SIZE = 12_042_077_420
EXPECTED_SHA256 = "35499c8b2ac7bc879c988c9af4f9e9ff9052caccd22d582b7fadd90de185d496"
EXPECTED_GROUP_SIZES = {"mot": 1649, "proprio_encoder": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file(path: Path, *, check_hash: bool = True) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    size = path.stat().st_size
    if size != EXPECTED_SIZE:
        raise RuntimeError(f"Unexpected file size: {size} bytes (expected {EXPECTED_SIZE})")
    print(f"size: {size} bytes [ok]")

    if check_hash:
        actual_hash = sha256(path)
        if actual_hash != EXPECTED_SHA256:
            raise RuntimeError(f"SHA-256 mismatch: {actual_hash}")
        print(f"sha256: {actual_hash} [ok]")


def check_payload(path: Path) -> None:
    try:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except TypeError:
        # PyTorch versions before mmap/weights_only support are not part of the
        # released environment, but this gives them a useful fallback.
        payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a dict checkpoint, got {type(payload).__name__}")
    for group, expected_count in EXPECTED_GROUP_SIZES.items():
        state_dict = payload.get(group)
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Missing state-dict group: {group}")
        actual_count = len(state_dict)
        if actual_count != expected_count:
            raise RuntimeError(
                f"Unexpected tensor count for {group}: {actual_count} (expected {expected_count})"
            )
        print(f"{group}: {actual_count} tensors [ok]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip the full-file SHA-256 pass (size and payload are still checked).",
    )
    args = parser.parse_args()

    path = args.checkpoint.expanduser().resolve()
    check_file(path, check_hash=not args.skip_sha256)
    check_payload(path)
    print(f"checkpoint: {path} [ready]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
