"""Versioned LIBERO-Plus instruction handling, independent of simulator imports."""

import json
import os
import re
import tempfile
from pathlib import Path


_LIBERO_PLUS_METADATA_SUFFIXES = (
    re.compile(r"\s+(?:table|tb|light|add)\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"\s+level\d+\s+sample\d+\s*$", re.IGNORECASE),
    re.compile(
        r"\s+view(?:\s+-?\d+(?:\.\d+)?){5}"
        r"\s+initstate\s+\d+\s+noise\s+\d+\s*$",
        re.IGNORECASE,
    ),
)


def validate_prompt_protocol(protocol: str) -> str:
    if protocol not in ("clean", "legacy"):
        raise ValueError(f"Unknown prompt protocol {protocol!r}; use 'clean' or 'legacy'.")
    return protocol


def canonicalize_libero_plus_instruction(task_description: str) -> str:
    """Remove benchmark metadata IDs while preserving language perturbations."""
    canonical = " ".join(str(task_description).split())
    for pattern in _LIBERO_PLUS_METADATA_SUFFIXES:
        canonical = pattern.sub("", canonical)
    return canonical.strip()


def instruction_for_model(task_description: str, protocol: str = "clean") -> str:
    validate_prompt_protocol(protocol)
    if protocol == "legacy":
        return task_description
    instruction = canonicalize_libero_plus_instruction(task_description)
    if not instruction:
        raise ValueError(f"Prompt canonicalization produced an empty instruction: {task_description!r}")
    return instruction


def _check_result_protocol(payload: dict, protocol: str, path: Path) -> None:
    # Results predating protocol labels used the raw instruction.
    recorded = payload.get("prompt_protocol", "legacy")
    if recorded != protocol:
        raise ValueError(
            f"Prompt protocol mismatch in {path}: found {recorded!r}, requested {protocol!r}. "
            "Use a separate EVALUATION.output_dir for each protocol."
        )


def existing_result_ok(path: Path, protocol: str) -> bool:
    validate_prompt_protocol(protocol)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    _check_result_protocol(payload, protocol, path)
    return "error" not in payload


def prepare_prompt_output(output_dir: Path, protocol: str) -> str:
    """Keep protocols separate even when worker IDs or skip_existing change."""
    validate_prompt_protocol(protocol)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "prompt_protocol.json"
    if not marker.exists():
        for path in output_dir.rglob("*_results.json"):
            existing_result_ok(path, protocol)
        # Publish a complete marker atomically, without replacing another worker's.
        with tempfile.NamedTemporaryFile(mode="w", dir=output_dir, delete=False) as f:
            temp_path = Path(f.name)
            json.dump({"prompt_protocol": protocol}, f)
        try:
            try:
                os.link(temp_path, marker)
            except FileExistsError:
                pass
        finally:
            temp_path.unlink()
    _check_result_protocol(json.loads(marker.read_text(encoding="utf-8")), protocol, marker)
    return protocol
