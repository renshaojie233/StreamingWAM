#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
    "assets/libero_plus_full_10030.txt": "32594adaeb21d20b604621f46e75d7e37da65a2fd8e1968f246beed163ef27ca",
    "assets/libero_dataset_stats.json": "a98bee79c78308a6a1ff63165a57c48c31d21d1f52c8b48f0a2b824a30ab8718",
}
REQUIRED_RUNTIME_FILES = {
    "scripts/ds_configs/ds_zero1_config.json",
    "scripts/ds_configs/ds_zero2_config.json",
    "scripts/ds_configs/ds_zero2_offload_config.json",
}
FORBIDDEN_NAMES = {"source.tar.gz", "PROVENANCE_FASTWAM_SDP.txt"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".zip", ".tar", ".gz"}
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".toml", ".json", ".sh"}
PATTERNS = {
    "private absolute path": re.compile(r"/(?:home|root|Users)/[^\s'\"`]+"),
    "credential assignment": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*['\"][^'\"]+"
    ),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    errors: list[str] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if path.name in FORBIDDEN_NAMES or ".bak" in path.name:
            errors.append(f"forbidden backup/archive: {rel}")
        if path.suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden generated/archive suffix: {rel}")
        if path.suffix in TEXT_SUFFIXES or path.name in {"LICENSE", "NOTICE"}:
            content = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in PATTERNS.items():
                if pattern.search(content):
                    errors.append(f"{label}: {rel}")

    manifest = ROOT / "assets/libero_plus_full_10030.txt"
    if manifest.exists():
        count = sum(1 for line in manifest.read_text().splitlines() if line.strip())
        if count != 10030:
            errors.append(f"task manifest has {count} non-empty lines; expected 10030")
    else:
        errors.append("missing assets/libero_plus_full_10030.txt")

    for rel, expected_hash in EXPECTED_HASHES.items():
        path = ROOT / rel
        if not path.exists():
            errors.append(f"missing required asset: {rel}")
        else:
            actual_hash = sha256(path)
            print(f"{actual_hash}  {rel}")
            if actual_hash != expected_hash:
                errors.append(f"hash mismatch: {rel}")

    for rel in sorted(REQUIRED_RUNTIME_FILES):
        if not (ROOT / rel).is_file():
            errors.append(f"missing required runtime file: {rel}")

    if errors:
        print("\nRelease check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("\nRelease check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
