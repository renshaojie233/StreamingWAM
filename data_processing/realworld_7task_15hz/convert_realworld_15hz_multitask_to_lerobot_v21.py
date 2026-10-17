#!/usr/bin/env python3
"""Convert synchronized 15 Hz FR3 tasks to one FastWAM LeRobot v2.1 dataset.

The source tree is read-only. Videos are copied byte-for-byte into the LeRobot
layout; no frame is decoded, inserted, removed, or temporally resampled here.
State/action semantics follow the validated apple-task conversion contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


VERSION = "fastwam_realworld_15hz_multitask_lerobot_v21_v1.0"
FPS = 15
DROID_JOINT_DELTA_RAD = 0.2
CAMERA_ORDER = ("camera_1", "camera_3", "camera_2")
CAMERA_CONTRACT = {
    "camera_1": "external view 1",
    "camera_3": "external view 2",
    "camera_2": "wrist view",
}
TASKS = (
    {
        "directory": "先称苹果，然后把它放进碗里（统一15Hz版）",
        "prompt": "weigh the apple, then put it into the bowl",
        "records": 100,
        "frames": 19_076,
    },
    {
        "directory": "关上抽屉（统一15Hz版）",
        "prompt": "close the drawer",
        "records": 80,
        "frames": 18_435,
    },
    {
        "directory": "打地鼠（统一15Hz版）",
        "prompt": "hit the mole",
        "records": 30,
        "frames": 60_742,
    },
    {
        "directory": "打开抽屉（统一15Hz版）",
        "prompt": "open the drawer",
        "records": 80,
        "frames": 25_705,
    },
    {
        "directory": "把苹果放进碗里_小车（统一15Hz版）",
        "prompt": "put the apple into the bowl using the mobile setup",
        "records": 200,
        "frames": 26_665,
    },
    {
        "directory": "把苹果放进碗里（静态+动态15Hz抽帧版）",
        "prompt": "put the apple into the bowl",
        "records": 414,
        "frames": 62_992,
    },
    {
        "directory": "搭积木",
        "prompt": "stack the blocks",
        "records": 179,
        "frames": 83_393,
    },
)
EXPECTED_RECORDS = sum(int(task["records"]) for task in TASKS)
EXPECTED_FRAMES = sum(int(task["frames"]) for task in TASKS)
CHUNK_SIZE = 1000


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def one_match(parent: Path, pattern: str) -> Path:
    matches = sorted(parent.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern!r} under {parent}, got {matches}")
    return matches[0]


def run_json(cmd: list[str]) -> dict[str, Any]:
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def probe_video(path: Path) -> dict[str, Any]:
    payload = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}: {streams}")
    s = streams[0]
    num, den = (int(x) for x in str(s["avg_frame_rate"]).split("/"))
    fps = num / den
    return {
        "video.height": int(s["height"]),
        "video.width": int(s["width"]),
        "video.codec": str(s["codec_name"]),
        "video.pix_fmt": str(s["pix_fmt"]),
        "video.is_depth_map": False,
        "video.fps": int(round(fps)),
        "video.channels": 3,
        "has_audio": False,
        "nb_frames": int(s["nb_frames"]),
    }


def copy_with_sha256(source: Path, destination: Path) -> str:
    """Copy once while hashing the exact bytes written to the destination."""
    h = hashlib.sha256()
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as src, tmp.open("xb") as dst:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
            dst.write(block)
            h.update(block)
    shutil.copystat(source, tmp)
    os.replace(tmp, destination)
    return h.hexdigest()


def array_stats(x: np.ndarray) -> dict[str, list[Any]]:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return {
        "min": np.min(x, axis=0).tolist(),
        "max": np.max(x, axis=0).tolist(),
        "mean": np.mean(x, axis=0, dtype=np.float64).tolist(),
        "std": np.std(x, axis=0, dtype=np.float64).tolist(),
        "count": [int(len(x))],
    }


def global_fastwam_stats(states: np.ndarray, actions: np.ndarray, episodes: int) -> dict[str, Any]:
    def field(x: np.ndarray) -> dict[str, Any]:
        q01 = np.quantile(x, 0.01, axis=0)
        q99 = np.quantile(x, 0.99, axis=0)
        base = {
            "min": np.min(x, axis=0),
            "max": np.max(x, axis=0),
            "mean": np.mean(x, axis=0, dtype=np.float64),
            "std": np.std(x, axis=0, dtype=np.float64),
            "q01": q01,
            "q99": q99,
        }
        out: dict[str, Any] = {}
        for prefix in ("global_", "stepwise_"):
            for key, value in base.items():
                out[prefix + key] = np.asarray(value).tolist()
        return out

    return {
        "action": {"default": field(actions)},
        "state": {"default": field(states)},
        "num_episodes": int(episodes),
        "num_transition": int(len(states)),
    }


def load_source_record(record: Path) -> dict[str, Any]:
    action_json = one_match(record / "action", "action_data_*.json")
    action_h5 = one_match(record / "action", "action_data_*.h5")
    payload = json.loads(action_json.read_text(encoding="utf-8"))
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Missing non-empty data list: {action_json}")
    n = len(rows)

    ts_path = record / "video" / "frame_timestamps.json"
    ts_payload = json.loads(ts_path.read_text(encoding="utf-8"))
    source_ts = np.asarray(ts_payload["timestamps_unix"], dtype=np.float64)
    row_ts = np.asarray([row["frame_timestamp"] for row in rows], dtype=np.float64)
    if n != int(ts_payload["frame_count"]) or n != len(source_ts):
        raise RuntimeError(f"JSON/timestamp length mismatch: {record}")
    if float(ts_payload["target_fps"]) != FPS:
        raise RuntimeError(f"Unexpected target_fps in {ts_path}: {ts_payload['target_fps']}")
    if not np.array_equal(source_ts, row_ts):
        raise RuntimeError(f"frame_timestamps differs from action JSON: {record}")

    q = np.asarray([row["franka_joints"]["position"] for row in rows], dtype=np.float64)
    gello_q = np.asarray([row["gello_joints"]["position"] for row in rows], dtype=np.float64)
    gripper = np.asarray([row["gripper_command"] for row in rows], dtype=np.float64)
    expected_names = tuple(f"fr3_joint{i}" for i in range(1, 8))
    for row in rows:
        if tuple(row["franka_joints"].get("names") or ()) != expected_names:
            raise RuntimeError(f"Unexpected Franka joint order in {action_json}")
    if q.shape != (n, 7) or gello_q.shape != (n, 7) or gripper.shape != (n,):
        raise RuntimeError(f"Unexpected source array shape: {record}")
    if not np.isfinite(q).all() or not np.isfinite(gello_q).all() or not np.isfinite(gripper).all():
        raise RuntimeError(f"Non-finite source values: {record}")
    if float(gripper.min()) < -1e-6 or float(gripper.max()) > 1.0 + 1e-6:
        raise RuntimeError(f"gripper_command outside [0,1]: {record}")

    with h5py.File(action_h5, "r") as h5:
        for name in (
            "franka/positions",
            "franka/velocities",
            "franka/efforts",
            "gello/positions",
            "gripper/positions",
            "gripper/velocities",
            "gripper/efforts",
            "end_effector/positions",
            "end_effector/orientations",
            "timestamps",
        ):
            if h5[name].shape[0] != n:
                raise RuntimeError(f"H5 length mismatch {name}: {record}")
        if not np.array_equal(h5["franka/positions"][:], q):
            raise RuntimeError(f"H5/JSON measured Franka q mismatch: {record}")
        if not np.array_equal(h5["gello/positions"][:], gello_q):
            raise RuntimeError(f"H5/JSON GELLO q mismatch: {record}")

    videos: dict[str, Path] = {}
    probes: dict[str, dict[str, Any]] = {}
    for camera in CAMERA_ORDER:
        path = one_match(record / "video", f"{camera}_*.mp4")
        probe = probe_video(path)
        if probe["video.fps"] != FPS or probe["nb_frames"] != n:
            raise RuntimeError(f"Video header mismatch for {path}: {probe}, expected {n}@{FPS}")
        videos[camera] = path
        probes[camera] = probe

    state = np.concatenate([q, gripper[:, None]], axis=1).astype(np.float32)
    action = np.zeros((n, 8), dtype=np.float64)
    action[:-1, :7] = np.diff(q, axis=0) / DROID_JOINT_DELTA_RAD
    action[:, 7] = gripper
    if not np.isfinite(action).all():
        raise RuntimeError(f"Non-finite derived action: {record}")
    max_native = float(np.max(np.abs(action[:, :7])))
    if max_native > 1.0 + 1e-6:
        raise RuntimeError(f"Native DROID action outside [-1,1] ({max_native}): {record}")

    return {
        "record": record,
        "action_json": action_json,
        "action_h5": action_h5,
        "timestamps_json": ts_path,
        "frames": n,
        "source_timestamp": source_ts,
        "state": state,
        "action": action.astype(np.float32),
        "videos": videos,
        "video_probes": probes,
        "max_native_action": max_native,
    }


def fixed_list_array(x: np.ndarray, width: int) -> pa.FixedSizeListArray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape != (len(x), width):
        raise ValueError((x.shape, width))
    return pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1), type=pa.float32()), width)


def write_episode_parquet(
    path: Path,
    state: np.ndarray,
    action: np.ndarray,
    source_timestamp: np.ndarray,
    episode_index: int,
    global_start: int,
    task_index: int,
) -> None:
    n = len(state)
    frame_index = np.arange(n, dtype=np.int64)
    table = pa.table(
        {
            "observation.state": fixed_list_array(state, 8),
            "action": fixed_list_array(action, 8),
            "observation.source_timestamp": pa.array(source_timestamp, type=pa.float64()),
            "timestamp": pa.array(frame_index.astype(np.float32) / np.float32(FPS), type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(np.full(n, episode_index, np.int64), type=pa.int64()),
            "index": pa.array(np.arange(global_start, global_start + n, dtype=np.int64), type=pa.int64()),
            "task_index": pa.array(np.full(n, task_index, np.int64), type=pa.int64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def feature_info(video_headers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for camera in CAMERA_ORDER:
        probe = dict(video_headers[camera])
        probe.pop("nb_frames", None)
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": [probe["video.height"], probe["video.width"], 3],
            "names": ["height", "width", "channels"],
            "info": probe,
        }
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": [
                    "fr3_joint1_rad",
                    "fr3_joint2_rad",
                    "fr3_joint3_rad",
                    "fr3_joint4_rad",
                    "fr3_joint5_rad",
                    "fr3_joint6_rad",
                    "fr3_joint7_rad",
                    "gello_gripper_0_closed_1_open",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": [8],
                "names": [
                    "droid_dq1_over_0.2",
                    "droid_dq2_over_0.2",
                    "droid_dq3_over_0.2",
                    "droid_dq4_over_0.2",
                    "droid_dq5_over_0.2",
                    "droid_dq6_over_0.2",
                    "droid_dq7_over_0.2",
                    "gello_gripper_absolute_0_closed_1_open",
                ],
            },
            "observation.source_timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    audit_root = args.audit_root.resolve()
    staging = output_root.with_name(f".{output_root.name}.conversion-v1")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_root}")
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite staging: {staging}")
    source_tasks: list[dict[str, Any]] = []
    for task_index, task in enumerate(TASKS):
        task_root = source_root / str(task["directory"])
        if not (task_root / "README.md").is_file():
            raise FileNotFoundError(f"Required source README missing: {task_root / 'README.md'}")
        records = sorted(p for p in task_root.glob("record_*") if p.is_dir())
        if len(records) != int(task["records"]):
            raise RuntimeError(
                f"Expected {task['records']} records for {task['directory']}, got {len(records)}"
            )
        source_tasks.append({**task, "task_index": task_index, "root": task_root, "records_list": records})
    records_with_tasks = [
        (task, record)
        for task in source_tasks
        for record in task["records_list"]
    ]
    if len(records_with_tasks) != EXPECTED_RECORDS:
        raise RuntimeError(f"Expected {EXPECTED_RECORDS} total records, got {len(records_with_tasks)}")
    audit_root.mkdir(parents=True, exist_ok=False)
    staging.mkdir(parents=True, exist_ok=False)

    episode_rows: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    source_map: list[dict[str, Any]] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    video_headers: dict[str, dict[str, Any]] | None = None
    global_index = 0
    max_native = 0.0

    task_totals: dict[str, dict[str, int]] = {
        str(task["directory"]): {"episodes": 0, "frames": 0} for task in source_tasks
    }

    for episode_index, (task, record) in enumerate(records_with_tasks):
        task_index = int(task["task_index"])
        prompt = str(task["prompt"])
        src = load_source_record(record)
        n = int(src["frames"])
        if video_headers is None:
            video_headers = src["video_probes"]
        else:
            for camera in CAMERA_ORDER:
                a = dict(video_headers[camera]); b = dict(src["video_probes"][camera])
                a.pop("nb_frames", None); b.pop("nb_frames", None)
                if a != b:
                    raise RuntimeError(f"Inconsistent {camera} video format at {record}: {b} != {a}")

        chunk = episode_index // CHUNK_SIZE
        parquet_path = staging / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        write_episode_parquet(
            parquet_path,
            src["state"],
            src["action"],
            src["source_timestamp"],
            episode_index,
            global_index,
            task_index,
        )

        video_sha: dict[str, str] = {}
        for camera in CAMERA_ORDER:
            video_key = f"observation.images.{camera}"
            dst = staging / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            video_sha[camera] = copy_with_sha256(src["videos"][camera], dst)
            if dst.stat().st_size != src["videos"][camera].stat().st_size:
                raise RuntimeError(f"Copied video size mismatch: {src['videos'][camera]} -> {dst}")

        frame_idx = np.arange(n, dtype=np.int64)
        ep_stats = {
            "observation.state": array_stats(src["state"]),
            "action": array_stats(src["action"]),
            "observation.source_timestamp": array_stats(src["source_timestamp"]),
            "timestamp": array_stats(frame_idx.astype(np.float64) / FPS),
            "frame_index": array_stats(frame_idx),
            "episode_index": array_stats(np.full(n, episode_index, np.int64)),
            "index": array_stats(np.arange(global_index, global_index + n, dtype=np.int64)),
            "task_index": array_stats(np.full(n, task_index, dtype=np.int64)),
        }
        episode_rows.append({"episode_index": episode_index, "tasks": [prompt], "length": n})
        episode_stats.append({"episode_index": episode_index, "stats": ep_stats})
        source_map.append(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "task": prompt,
                "source_task": str(task["directory"]),
                "source_record": record.name,
                "frames": n,
                "action_json": str(src["action_json"]),
                "action_h5": str(src["action_h5"]),
                "timestamps_json": str(src["timestamps_json"]),
                "source_videos": {k: str(v) for k, v in src["videos"].items()},
                "copied_video_sha256": video_sha,
            }
        )
        all_states.append(src["state"])
        all_actions.append(src["action"])
        global_index += n
        max_native = max(max_native, float(src["max_native_action"]))
        task_totals[str(task["directory"])]["episodes"] += 1
        task_totals[str(task["directory"])]["frames"] += n

        json_write(
            audit_root / "conversion_progress.json",
            {
                "converter_version": VERSION,
                "episodes_complete": episode_index + 1,
                "episodes_total": len(records_with_tasks),
                "frames_complete": global_index,
                "current_task": str(task["directory"]),
                "last_source_record": record.name,
            },
        )
        print(
            f"converted {episode_index + 1:04d}/{len(records_with_tasks)} "
            f"task={task['directory']} {record.name} frames={n} total={global_index}",
            flush=True,
        )

    if global_index != EXPECTED_FRAMES:
        raise RuntimeError(f"Expected {EXPECTED_FRAMES} frames, got {global_index}")
    for task in source_tasks:
        actual = task_totals[str(task["directory"])]
        expected = {"episodes": int(task["records"]), "frames": int(task["frames"])}
        if actual != expected:
            raise RuntimeError(f"Task totals mismatch for {task['directory']}: {actual} != {expected}")
    assert video_headers is not None
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "fr3",
        "total_episodes": len(records_with_tasks),
        "total_frames": global_index,
        "total_tasks": len(source_tasks),
        "total_videos": len(records_with_tasks) * len(CAMERA_ORDER),
        "total_chunks": (len(records_with_tasks) + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{len(records_with_tasks)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feature_info(video_headers),
    }
    json_write(staging / "meta/info.json", info)
    jsonl_write(
        staging / "meta/tasks.jsonl",
        [{"task_index": int(task["task_index"]), "task": str(task["prompt"])} for task in source_tasks],
    )
    jsonl_write(staging / "meta/episodes.jsonl", episode_rows)
    jsonl_write(staging / "meta/episodes_stats.jsonl", episode_stats)
    jsonl_write(staging / "meta/source_records.jsonl", source_map)
    stats = global_fastwam_stats(states, actions, len(records_with_tasks))
    json_write(staging / "fastwam_dataset_stats.json", stats)

    docs = staging / "documentation"
    docs.mkdir(parents=True, exist_ok=True)
    source_readmes = docs / "source_readmes"
    source_readmes.mkdir(parents=True, exist_ok=True)
    for task in source_tasks:
        shutil.copy2(task["root"] / "README.md", source_readmes / f"task_{task['task_index']:02d}.md")
    dataset_readme = "# FastWAM Real-World Multi-Task 15Hz LeRobot v2.1\n\n"
    dataset_readme += f"Converter: `{VERSION}`\n\n"
    dataset_readme += f"- Source root: `{source_root}` (read-only)\n"
    dataset_readme += f"- Episodes: {len(records_with_tasks)} across {len(source_tasks)} tasks (all train)\n"
    dataset_readme += f"- Frames: {global_index} at {FPS}Hz\n"
    dataset_readme += "- Camera order for FastWAM: camera_1, camera_3, camera_2\n"
    dataset_readme += "- Each camera remains a separate video; FastWAM resizes each to 224x128 and uses `concat_multi_camera: vertical`.\n"
    dataset_readme += "- State: 7 measured Franka joint positions in radians + continuous GELLO gripper command (0 closed, 1 open).\n"
    dataset_readme += "- Action: `(q[t+1]-q[t])/0.2` for 7 joints + absolute GELLO gripper command; final joint action is zero.\n"
    dataset_readme += "- Videos were copied byte-for-byte; there was no interpolation, downsampling, frame insertion, or re-encoding.\n"
    dataset_readme += "- FastWAM Wan2.2 VAE spatial factor 16 and DiT patch 2x2 give 384x224 -> 24x14 -> 12x7 = 84 spatial tokens/frame.\n"
    dataset_readme += "\n## Tasks\n\n"
    for task in source_tasks:
        totals = task_totals[str(task["directory"])]
        dataset_readme += (
            f"- `{task['task_index']}`: {task['prompt']} "
            f"({totals['episodes']} episodes, {totals['frames']} frames; source `{task['directory']}`)\n"
        )
    (staging / "README.md").write_text(dataset_readme, encoding="utf-8")

    report = {
        "status": "complete",
        "converter_version": VERSION,
        "source_root": str(source_root),
        "dataset_root": str(output_root),
        "episodes": len(records_with_tasks),
        "frames": global_index,
        "tasks": [
            {
                "task_index": int(task["task_index"]),
                "task": str(task["prompt"]),
                "source_task": str(task["directory"]),
                **task_totals[str(task["directory"])],
            }
            for task in source_tasks
        ],
        "fps": FPS,
        "camera_order": list(CAMERA_ORDER),
        "camera_contract": CAMERA_CONTRACT,
        "videos": len(records_with_tasks) * len(CAMERA_ORDER),
        "state_contract": "7 measured Franka q radians + continuous GELLO gripper command 0 closed 1 open",
        "action_contract": "7 normalized DROID joint increments (q[t+1]-q[t])/0.2 + current absolute GELLO gripper; final joint increment is zero",
        "max_abs_native_joint_action": max_native,
        "source_timing": "original synchronized 15Hz frames; source absolute timestamps retained in observation.source_timestamp",
        "lerobot_timestamp": "frame_index/15 seconds, aligned to constant-frame-rate MP4 PTS",
        "videos_copied_byte_for_byte": True,
        "validation_split": None,
    }
    json_write(staging / "meta/conversion_report.json", report)
    os.replace(staging, output_root)
    json_write(audit_root / "CONVERSION_COMPLETE.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
