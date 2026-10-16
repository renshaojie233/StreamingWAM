#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import urllib.request


RELEASE_BASE = "https://github.com/renshaojie233/StreamingWAM/releases/download/realworld-v1"
CHECKPOINT_SHA256 = "ed68b39a9d7025188e6216a9336a3e265893d360b0c02b91f31629c317d1f66a"
RUNTIME_ARCHIVE = "streamingwam-realworld-runtime-assets.tar.gz"
RUNTIME_ARCHIVE_SHA256 = "251466fc7be4883f2e4c3196f0c9522e52d5f12123a9cc3eed9cc6d5f78007ed"
PART_SHA256 = {
    "step_030000.pt.part-000": "c33b79bf7928a42392d221bf2d5590d604c55433fe63bd3bfae08b5459738abc",
    "step_030000.pt.part-001": "dd357e17eb2dfd8bb0a81f90bbf1baeafb94426193da154cb1a5f155426c0034",
    "step_030000.pt.part-002": "b94dd7c9305ea3c153767db9ea4fb54e32f833986907ecc87f4c9ad08e89410a",
    "step_030000.pt.part-003": "b67ef5d55c49deb383dea225387a6b4bafcaf1c99eb898c2ee324377cc1c3cd6",
    "step_030000.pt.part-004": "e61e0672f263fb2f199ecd11d8022944b5a3b5cd3395bc61abdb6566ba589382",
    "step_030000.pt.part-005": "bd98ec56aca2090715f74631191046969fe493745925e5c1659664af923e921c",
    "step_030000.pt.part-006": "243ba8d2263b9a1540923c073bc8b58cf71a6b4e20ded18d6f7bf28791d5316e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_checked(name: str, expected: str, directory: Path) -> Path:
    target = directory / name
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
        parts = [download_checked(name, digest, downloads) for name, digest in PART_SHA256.items()]
        assembling = downloads / "step_030000.pt.assembling"
        with assembling.open("wb") as destination:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, destination, 16 * 1024 * 1024)
        actual = sha256(assembling)
        if actual != CHECKPOINT_SHA256:
            raise RuntimeError(f"assembled checkpoint SHA-256 mismatch: {actual}")
        os.replace(assembling, checkpoint)

    archive = download_checked(RUNTIME_ARCHIVE, RUNTIME_ARCHIVE_SHA256, downloads)
    safe_extract(archive, output)
    print(output)


if __name__ == "__main__":
    main()
