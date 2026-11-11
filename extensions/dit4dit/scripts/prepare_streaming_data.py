"""Build an independent official-loader view of repaired LIBERO v3 data.

Use --source-root, --output-root,
and --checkpoint-stats. Raw source and checkpoint files are read-only.
The output directory must not exist. --dry-run validates without writing.
Generated stats_gr00t.json is a CHECKPOINT NORMALIZATION REFERENCE, not a
claim of measured per-suite statistics. Every suite uses the same reference
so mixture metadata and evaluation agree exactly on action min/max.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SUITES = ("spatial", "object", "goal", "10")
STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper")
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
STAT_KEYS = ("min", "max", "mean", "std", "q01", "q99")


def checkpoint_reference(path: Path) -> dict:
    reference = json.loads(path.read_text())["franka"]
    for kind, size in (("state", 8), ("action", 7)):
        for stat in STAT_KEYS:
            value = np.asarray(reference[kind][stat], dtype=np.float64)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint {kind}.{stat}")
    if reference["action"].get("mask") != [True] * 6 + [False]:
        raise ValueError("Expected six normalized action dimensions and unnormalized gripper")
    return {
        "observation.state": {key: reference["state"][key] for key in STAT_KEYS},
        "action": {key: reference["action"][key] for key in STAT_KEYS},
    }


def inspect_suite(source: Path) -> tuple[dict, dict]:
    """Validate all low-dimensional rows, not just stats.json or one sample."""
    info = json.loads((source / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0" or info.get("fps") != 20:
        raise ValueError(f"Expected LIBERO v3, 20 FPS: {source}")
    features = info["features"]
    if features["observation.state"]["shape"] != [8] or features["action"]["shape"] != [7]:
        raise ValueError(f"Unexpected state/action dimensions: {source}")
    state_names = features["observation.state"]["names"]["motors"]
    if state_names != ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper", "gripper"]:
        raise ValueError(f"Unexpected raw state field order: {state_names}")
    action_names = features["action"]["names"]["motors"]
    if action_names != ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]:
        raise ValueError(f"Unexpected raw action field order: {action_names}")
    for key in ("observation.images.image", "observation.images.wrist_image"):
        feature = features[key]
        if feature["shape"][-1] != 3 or feature["info"]["video.channels"] != 3:
            raise ValueError(f"Expected RGB video: {source}/{key}")
        if feature["info"]["video.fps"] != 20:
            raise ValueError(f"Expected 20 FPS video: {source}/{key}")
        feature["names"] = ["height", "width", "channel"]
        feature["video_info"] = {"video.fps": 20, "video.channels": 3}
    for relative in ("data", "videos", "meta/episodes", "meta/tasks.parquet"):
        if not (source / relative).exists():
            raise FileNotFoundError(source / relative)
    files = sorted((source / "data").glob("*/*.parquet"))
    if not files:
        raise ValueError(f"No data parquet files: {source}")
    count = 0
    gripper_values = set()
    for path in files:
        table = pd.read_parquet(path, columns=["observation.state", "action"])
        states = np.stack(table["observation.state"])
        actions = np.stack(table["action"])
        if states.shape != (len(table), 8) or actions.shape != (len(table), 7):
            raise ValueError(f"Invalid row shapes: {path}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"Nonfinite state/action: {path}")
        gripper_values.update(np.unique(actions[:, 6]).tolist())
        count += len(table)
    if not gripper_values or not gripper_values <= {0.0, 1.0}:
        raise ValueError(f"Gripper is not already binary [0,1]: {source}: {gripper_values}; no automatic conversion")
    if count != info["total_frames"]:
        raise ValueError(f"Frame count mismatch: {source}: {count} != {info['total_frames']}")
    return info, {"frames": count, "gripper_values": sorted(gripper_values)}


def modality_metadata() -> dict:
    # 'pad' is the first gripper qpos (raw index 6), NOT an inserted zero.
    # absolute=True matches the official metadata defaults; streaming validity
    # masks plus post-normalization zeroing define boundary padding separately.
    def fields(keys, original_key):
        return {key: {"start": i, "end": i + 1, "original_key": original_key,
                      "dtype": "float64", "absolute": True}
                for i, key in enumerate(keys)}
    return {
        "state": fields(STATE_KEYS, "observation.state"),
        "action": fields(ACTION_KEYS, "action"),
        "video": {
            "primary_image": {"original_key": "observation.images.image"},
            "wrist_image": {"original_key": "observation.images.wrist_image"},
        },
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-stats", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Independent output root must not overlap source root")
    if output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(f"Refusing to reuse or overwrite output root: {output_root}")
    stats = checkpoint_reference(args.checkpoint_stats)
    stats_hash = hashlib.sha256(args.checkpoint_stats.read_bytes()).hexdigest()
    prepared = []
    for suite in SUITES:
        name = f"libero_{suite}_no_noops_1.0.0_lerobot"
        source = (source_root / f"libero_{suite}_no_noops_lerobot").resolve(strict=True)
        info, summary = inspect_suite(source)
        info["streaming_adapter"] = {
            "source": str(source), "checkpoint_stats_sha256": stats_hash,
            "normalization": "official_checkpoint_franka_not_per_suite",
            "state_pad_semantics": "observation.state[6] = first gripper qpos",
            "action_gripper_conversion": "none: raw binary 0/1 validated across all rows",
        }
        prepared.append((name, source, info))
        print(json.dumps({"dataset": name, "source": str(source), **summary}))
    if args.dry_run:
        print("Dry run passed; no files written.")
        return
    output_root.mkdir(parents=True, exist_ok=False)
    for name, source, info in prepared:
        target = output_root / name
        meta = target / "meta"
        meta.mkdir(parents=True)
        for relative in ("data", "videos", "meta/episodes", "meta/tasks.parquet"):
            original = source / relative
            (target / relative).symlink_to(original, target_is_directory=original.is_dir())
        for filename, content in (("info.json", info), ("modality.json", modality_metadata()),
                                  ("stats_gr00t.json", stats), ("stats.json", stats)):
            (meta / filename).write_text(json.dumps(content, indent=2) + "\n")
    print(f"Prepared independent streaming view: {output_root}")


if __name__ == "__main__":
    main()
