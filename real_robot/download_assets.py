#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import urllib.request


REPO_ID = "rsj2003/StreamingWAM-RealWorld"
REVISION = "bdee7eeb4873bd94dcfaae62b4e20e5c468696af"
RELEASE_BASE = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}"
CHECKPOINT_FILE = "streamingwam_realworld.pt"
CHECKPOINT_SHA256 = "ed68b39a9d7025188e6216a9336a3e265893d360b0c02b91f31629c317d1f66a"
RUNTIME_ARCHIVE = "streamingwam-realworld-runtime-assets.tar.gz"
RUNTIME_ARCHIVE_SHA256 = "251466fc7be4883f2e4c3196f0c9522e52d5f12123a9cc3eed9cc6d5f78007ed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_checked(name: str, expected: str, directory: Path, *, local_name: str | None = None) -> Path:
    target = directory / (local_name or name)
    if target.is_file() and sha256(target) == expected:
        return target
    temporary = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(f"{RELEASE_BASE}/{name}") as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, 16 * 1024 * 1024)
    actual = sha256(temporary)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {name}: {actual}")
    os.replace(temporary, target)
    return target


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            resolved = (destination / member.name).resolve()
            if root != resolved and root not in resolved.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"archive links are not allowed: {member.name}")
        stream.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    downloads = output / ".downloads"
    downloads.mkdir(exist_ok=True)

    checkpoint = output / "step_030000.pt"
    if checkpoint.exists() and sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError(f"refusing to overwrite corrupt existing file: {checkpoint}")
    if not checkpoint.exists():
        download_checked(CHECKPOINT_FILE, CHECKPOINT_SHA256, output, local_name="step_030000.pt")

    archive = download_checked(RUNTIME_ARCHIVE, RUNTIME_ARCHIVE_SHA256, downloads)
    safe_extract(archive, output)
    print(output)


if __name__ == "__main__":
    main()
