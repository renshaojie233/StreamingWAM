#!/usr/bin/env python3
"""Full validator and FastWAM loader smoke test for the 7-task 15 Hz dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import av
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


FPS = 15
DROID_JOINT_DELTA_RAD = 0.2
CAMERA_ORDER = ("camera_1", "camera_3", "camera_2")
TASKS = (
    ("先称苹果，然后把它放进碗里（统一15Hz版）", "weigh the apple, then put it into the bowl", 100, 19_076),
    ("关上抽屉（统一15Hz版）", "close the drawer", 80, 18_435),
    ("打地鼠（统一15Hz版）", "hit the mole", 30, 60_742),
    ("打开抽屉（统一15Hz版）", "open the drawer", 80, 25_705),
    ("把苹果放进碗里_小车（统一15Hz版）", "put the apple into the bowl using the mobile setup", 200, 26_665),
    ("把苹果放进碗里（静态+动态15Hz抽帧版）", "put the apple into the bowl", 414, 62_992),
    ("搭积木", "stack the blocks", 179, 83_393),
)
EXPECTED_RECORDS = sum(task[2] for task in TASKS)
EXPECTED_FRAMES = sum(task[3] for task in TASKS)
EXPECTED_VIDEOS = EXPECTED_RECORDS * len(CAMERA_ORDER)
CHUNK_SIZE = 1000


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def one(parent: Path, pattern: str) -> Path:
    values = sorted(parent.glob(pattern))
    if len(values) != 1:
        raise RuntimeError(f"Expected one {pattern!r} under {parent}, got {values}")
    return values[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decode_video_job(job: tuple[str, str, int, int, int]) -> dict[str, Any]:
    output_path_s, source_path_s, expected_frames, expected_width, expected_height = job
    output_path = Path(output_path_s)
    source_path = Path(source_path_s)
    if output_path.stat().st_size != source_path.stat().st_size:
        raise RuntimeError(f"Video byte-size mismatch: {output_path}")
    source_sha = sha256(source_path)
    output_sha = sha256(output_path)
    if source_sha != output_sha:
        raise RuntimeError(f"Video SHA-256 mismatch: {output_path}")
    count = 0
    pts: list[float] = []
    observed: set[tuple[int, int]] = set()
    with av.open(str(output_path)) as container:
        streams = container.streams.video
        if len(streams) != 1:
            raise RuntimeError(f"Expected one stream: {output_path}")
        stream = streams[0]
        stream.thread_type = "AUTO"
        rate = float(stream.average_rate)
        if abs(rate - FPS) > 1e-9:
            raise RuntimeError(f"FPS mismatch {rate}: {output_path}")
        for frame in container.decode(stream):
            count += 1
            observed.add((int(frame.width), int(frame.height)))
            if frame.time is None:
                raise RuntimeError(f"Missing decoded PTS: {output_path}")
            pts.append(float(frame.time))
    if count != expected_frames:
        raise RuntimeError(f"Decoded frames {count} != {expected_frames}: {output_path}")
    if observed != {(expected_width, expected_height)}:
        raise RuntimeError(f"Resolution mismatch {observed}: {output_path}")
    expected_pts = np.arange(expected_frames, dtype=np.float64) / FPS
    max_pts_error = float(np.max(np.abs(np.asarray(pts) - expected_pts)))
    if max_pts_error > 1e-6:
        raise RuntimeError(f"PTS mismatch {max_pts_error}: {output_path}")
    return {
        "path": str(output_path),
        "frames": count,
        "sha256": output_sha,
        "max_pts_error_s": max_pts_error,
    }


def list_column(table, key: str) -> np.ndarray:
    values = table[key].to_pylist()
    return np.asarray(values)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def frame_at(path: Path, index: int) -> np.ndarray:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i == index:
                rgb = frame.to_ndarray(format="rgb24")
                return np.asarray(Image.fromarray(rgb).resize((224, 128), Image.Resampling.BILINEAR))
    raise RuntimeError(f"Frame {index} missing: {path}")


def make_visualization(
    out_path: Path,
    episode_index: int,
    n: int,
    videos: dict[str, Path],
    action: np.ndarray,
) -> None:
    indices = [0, n // 2, n - 1]
    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    grid = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.25])
    for row, frame_index in enumerate(indices):
        for col, camera in enumerate(CAMERA_ORDER):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(frame_at(videos[camera], frame_index))
            ax.set_title(f"{camera} | frame {frame_index}")
            ax.axis("off")
    ax = fig.add_subplot(grid[3, :])
    time_s = np.arange(n) / FPS
    for joint in range(7):
        ax.plot(time_s, action[:, joint], linewidth=0.8, label=f"joint{joint + 1}")
    ax.plot(time_s, action[:, 7], linewidth=1.5, color="black", label="gripper")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("action")
    ax.grid(alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.suptitle(f"episode {episode_index:06d}: camera_1, camera_3, camera_2 + action")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_fastwam_smoke(dataset_root: Path, fastwam_root: Path, smoke_work_dir: Path) -> dict[str, Any]:
    import sys
    sys.path.insert(0, str(fastwam_root / "src"))
    import torch
    import torchvision
    from omegaconf import OmegaConf
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
    from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
    from fastwam.datasets.lerobot.transforms.image import ToTensor
    from fastwam.utils import misc

    misc.register_work_dir(smoke_work_dir)

    shape_meta = {
        "images": [
            {"key": "camera_1", "raw_shape": [3, 1080, 1920], "shape": [3, 128, 224]},
            {"key": "camera_3", "raw_shape": [3, 1080, 1920], "shape": [3, 128, 224]},
            {"key": "camera_2", "raw_shape": [3, 720, 1280], "shape": [3, 128, 224]},
        ],
        "action": [{"key": "default", "raw_shape": 8, "shape": 8}],
        "state": [{"key": "default", "raw_shape": 8, "shape": 8}],
    }
    # Preserve Hydra's ListConfig shape objects: FastWAM's processor compares
    # torch.Size directly against these sequences during real training.
    shape_config = OmegaConf.create(shape_meta)
    transforms = [ToTensor(), torchvision.transforms.Resize(size=[128, 224], antialias=True)]

    def build_processor(num_obs_steps: int) -> FastWAMProcessor:
        return FastWAMProcessor(
            shape_meta=shape_config,
            num_obs_steps=num_obs_steps,
            num_output_cameras=3,
            action_output_dim=8,
            proprio_output_dim=8,
            action_state_transforms=None,
            use_stepwise_action_norm=False,
            norm_default_mode="min/max",
            norm_exception_mode=None,
            action_state_merger=ConcatLeftAlign(),
            train_transforms=transforms,
            val_transforms=transforms,
            delta_action_dim_mask={"default": [True, True, True, True, True, True, True, False]},
        )

    def build_dataset(processor: FastWAMProcessor, **temporal_kwargs: Any) -> RobotVideoDataset:
        return RobotVideoDataset(
            dataset_dirs=[str(dataset_root)],
            shape_meta=shape_config,
            video_size=[384, 224],
            camera_key=None,
            processor=processor,
            text_embedding_cache_dir=str(dataset_root / "text_embeds_cache"),
            context_len=128,
            pretrained_norm_stats=str(dataset_root / "fastwam_dataset_stats.json"),
            val_set_proportion=0.0,
            is_training_set=True,
            global_sample_stride=1,
            skip_padding_as_possible=False,
            concat_multi_camera="vertical",
            **temporal_kwargs,
        )

    original_context = RobotVideoDataset._get_cached_text_context
    RobotVideoDataset._get_cached_text_context = lambda self, prompt: (
        torch.zeros((self.context_len, 4096), dtype=torch.float32),
        torch.ones((self.context_len,), dtype=torch.bool),
    )
    try:
        joint_dataset = build_dataset(
            build_processor(33),
            num_frames=33,
            action_video_freq_ratio=4,
        )
        sdp_dataset = build_dataset(
            build_processor(21),
            num_frames=17,
            action_video_freq_ratio=4,
            history_frame_stride=1,
            num_history_frames=5,
            future_action_horizon=16,
            action_clean_prefix_steps=16,
        )
        joint_sample = joint_dataset[0]
        sdp_sample = sdp_dataset[0]
        joint_batch = next(iter(torch.utils.data.DataLoader(joint_dataset, batch_size=2, num_workers=0)))
        sdp_batch = next(iter(torch.utils.data.DataLoader(sdp_dataset, batch_size=2, num_workers=0)))
    finally:
        RobotVideoDataset._get_cached_text_context = original_context

    def check_sample(
        name: str,
        sample: dict[str, Any],
        batch: dict[str, Any],
        expected_proprio_shape: list[int],
    ) -> dict[str, Any]:
        public_shape = list(sample["video"].shape)
        pre_permute_shape = list(sample["video"].permute(1, 0, 2, 3).shape)
        batch_shape = list(batch["video"].shape)
        if public_shape != [3, 9, 384, 224]:
            raise RuntimeError(f"{name} video shape mismatch: {public_shape}")
        if pre_permute_shape != [9, 3, 384, 224]:
            raise RuntimeError(f"{name} pre-permute shape mismatch: {pre_permute_shape}")
        if list(sample["action"].shape) != [32, 8] or list(sample["proprio"].shape) != expected_proprio_shape:
            raise RuntimeError(
                f"{name} action/proprio mismatch: {sample['action'].shape}, {sample['proprio'].shape}"
            )
        if not torch.isfinite(sample["video"]).all() or not torch.isfinite(sample["action"]).all():
            raise RuntimeError(f"Non-finite {name} FastWAM smoke sample")
        return {
            "sample_C_T_H_W": public_shape,
            "pre_permute_T_C_H_W": pre_permute_shape,
            "batch_B_C_T_H_W": batch_shape,
            "action_shape": list(sample["action"].shape),
            "proprio_shape": list(sample["proprio"].shape),
        }

    joint_shapes = check_sample("joint", joint_sample, joint_batch, [32, 8])
    sdp_shapes = check_sample("sdp", sdp_sample, sdp_batch, [1, 8])

    vae_factor = 16
    patch_h = patch_w = 2
    latent_h, latent_w = 384 // vae_factor, 224 // vae_factor
    token_h, token_w = latent_h // patch_h, latent_w // patch_w
    if (token_h, token_w) != (12, 7):
        raise RuntimeError((token_h, token_w))
    return {
        "status": "passed",
        "dataset_class": "fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset",
        "camera_order": list(CAMERA_ORDER),
        "per_camera_resize_h_w": [128, 224],
        "concat_multi_camera": "vertical",
        "source_and_loader_frequency_hz": FPS,
        "action_video_freq_ratio": 4,
        "joint_temporal_contract": {
            "relative_video_frames": [0, 4, 8, 12, 16, 20, 24, 28, 32],
            "relative_actions": [0, 31],
            **joint_shapes,
        },
        "sdp_temporal_contract": {
            "relative_video_frames": [-4, -3, -2, -1, 0, 4, 8, 12, 16],
            "relative_actions": [-16, 15],
            "clean_action_prefix_steps": 16,
            "future_action_horizon": 16,
            **sdp_shapes,
        },
        "vae_class": "WanVideoVAE38",
        "vae_spatial_factor": vae_factor,
        "vae_latent_grid_h_w": [latent_h, latent_w],
        "dit_patch_h_w": [patch_h, patch_w],
        "dit_spatial_token_grid_h_w": [token_h, token_w],
        "tokens_per_frame": token_h * token_w,
        "text_context_note": "Synthetic zero context used only to isolate the dataloader/camera smoke test; run the official precompute_text_embeds.py before training.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--fastwam-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    dataset_root = args.dataset_root.resolve()
    audit_root = args.audit_root.resolve()
    fastwam_root = args.fastwam_root.resolve()
    if audit_root.exists():
        raise FileExistsError(f"Refusing to overwrite validation audit: {audit_root}")
    audit_root.mkdir(parents=True, exist_ok=False)

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    episodes = read_jsonl(dataset_root / "meta/episodes.jsonl")
    tasks = read_jsonl(dataset_root / "meta/tasks.jsonl")
    source_map = read_jsonl(dataset_root / "meta/source_records.jsonl")
    if info["codebase_version"] != "v2.1" or info["fps"] != FPS:
        raise RuntimeError(f"Bad info version/fps: {info}")
    if info["total_episodes"] != EXPECTED_RECORDS or info["total_frames"] != EXPECTED_FRAMES:
        raise RuntimeError("Info episode/frame count mismatch")
    if info["total_videos"] != EXPECTED_VIDEOS:
        raise RuntimeError("Info video count mismatch")
    if info["splits"] != {"train": f"0:{EXPECTED_RECORDS}"}:
        raise RuntimeError(f"Unexpected splits: {info['splits']}")
    expected_tasks = [
        {"task_index": task_index, "task": prompt}
        for task_index, (_directory, prompt, _records, _frames) in enumerate(TASKS)
    ]
    if tasks != expected_tasks:
        raise RuntimeError(f"Unexpected tasks: {tasks}")
    if len(episodes) != EXPECTED_RECORDS or len(source_map) != EXPECTED_RECORDS:
        raise RuntimeError("Metadata row count mismatch")

    total_frames = 0
    global_index = 0
    max_state_error = 0.0
    max_action_error = 0.0
    max_source_ts_error = 0.0
    max_lerobot_ts_error = 0.0
    max_native_action = 0.0
    video_jobs: list[tuple[str, str, int, int, int]] = []
    vis_payloads: dict[int, tuple[int, dict[str, Path], np.ndarray]] = {}
    records_with_tasks: list[tuple[int, str, str, Path]] = []
    for task_index, (directory, prompt, expected_records, _expected_frames) in enumerate(TASKS):
        task_root = source_root / directory
        records = sorted(path for path in task_root.glob("record_*") if path.is_dir())
        if len(records) != expected_records:
            raise RuntimeError(f"Source record count mismatch for {directory}: {len(records)}")
        records_with_tasks.extend((task_index, directory, prompt, record) for record in records)
    visualized = []
    offset = 0
    for _task_index, (_directory, _prompt, records, _frames) in enumerate(TASKS):
        visualized.append(offset + records // 2)
        offset += records

    task_frame_totals = {directory: 0 for directory, _prompt, _records, _frames in TASKS}
    for episode_index, (task_index, source_task, prompt, rec) in enumerate(records_with_tasks):
        action_json = one(rec / "action", "action_data_*.json")
        action_h5 = one(rec / "action", "action_data_*.h5")
        rows = json.loads(action_json.read_text(encoding="utf-8"))["data"]
        n = len(rows)
        q = np.asarray([r["franka_joints"]["position"] for r in rows], dtype=np.float64)
        gripper = np.asarray([r["gripper_command"] for r in rows], dtype=np.float64)
        source_ts = np.asarray([r["frame_timestamp"] for r in rows], dtype=np.float64)
        timestamp_json = json.loads((rec / "video/frame_timestamps.json").read_text(encoding="utf-8"))
        if not np.array_equal(source_ts, np.asarray(timestamp_json["timestamps_unix"], np.float64)):
            raise RuntimeError(f"Source timestamp mismatch: {rec}")
        with h5py.File(action_h5, "r") as h5:
            if not np.array_equal(h5["franka/positions"][:], q):
                raise RuntimeError(f"Source H5/JSON q mismatch: {rec}")

        expected_state = np.concatenate([q, gripper[:, None]], axis=1).astype(np.float32)
        expected_action = np.zeros((n, 8), np.float64)
        expected_action[:-1, :7] = np.diff(q, axis=0) / DROID_JOINT_DELTA_RAD
        expected_action[:, 7] = gripper
        expected_action = expected_action.astype(np.float32)
        chunk = episode_index // CHUNK_SIZE
        parquet = dataset_root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        table = pq.read_table(parquet)
        expected_columns = {
            "observation.state", "action", "observation.source_timestamp", "timestamp",
            "frame_index", "episode_index", "index", "task_index",
        }
        if set(table.column_names) != expected_columns or table.num_rows != n:
            raise RuntimeError(f"Parquet schema/length mismatch: {parquet}")
        state = list_column(table, "observation.state").astype(np.float32)
        action = list_column(table, "action").astype(np.float32)
        out_source_ts = np.asarray(table["observation.source_timestamp"].to_numpy(), np.float64)
        lerobot_ts = np.asarray(table["timestamp"].to_numpy(), np.float64)
        frame_index = np.asarray(table["frame_index"].to_numpy(), np.int64)
        episode_col = np.asarray(table["episode_index"].to_numpy(), np.int64)
        index_col = np.asarray(table["index"].to_numpy(), np.int64)
        task_col = np.asarray(table["task_index"].to_numpy(), np.int64)
        if not all(np.isfinite(x).all() for x in (state, action, out_source_ts, lerobot_ts)):
            raise RuntimeError(f"Non-finite parquet values: {parquet}")
        max_state_error = max(max_state_error, float(np.max(np.abs(state - expected_state))))
        max_action_error = max(max_action_error, float(np.max(np.abs(action - expected_action))))
        max_source_ts_error = max(max_source_ts_error, float(np.max(np.abs(out_source_ts - source_ts))))
        max_lerobot_ts_error = max(
            max_lerobot_ts_error,
            float(np.max(np.abs(lerobot_ts - np.arange(n, dtype=np.float64) / FPS))),
        )
        max_native_action = max(max_native_action, float(np.max(np.abs(action[:, :7]))))
        if not np.array_equal(frame_index, np.arange(n, dtype=np.int64)):
            raise RuntimeError(f"Frame index mismatch: {parquet}")
        if not np.array_equal(episode_col, np.full(n, episode_index, np.int64)):
            raise RuntimeError(f"Episode index mismatch: {parquet}")
        if not np.array_equal(index_col, np.arange(global_index, global_index + n, dtype=np.int64)):
            raise RuntimeError(f"Global index mismatch: {parquet}")
        if np.any(task_col != task_index):
            raise RuntimeError(f"Task index mismatch: {parquet}")
        if max_state_error != 0.0 or max_action_error != 0.0 or max_source_ts_error != 0.0:
            raise RuntimeError(f"Exact state/action/source timestamp mismatch by episode {episode_index}")

        out_videos: dict[str, Path] = {}
        for camera in CAMERA_ORDER:
            src_video = one(rec / "video", f"{camera}_*.mp4")
            key = f"observation.images.{camera}"
            out_video = dataset_root / f"videos/chunk-{chunk:03d}/{key}/episode_{episode_index:06d}.mp4"
            ft = info["features"][key]
            video_jobs.append(
                (str(out_video), str(src_video), n, int(ft["shape"][1]), int(ft["shape"][0]))
            )
            out_videos[camera] = out_video
        if episode_index in visualized:
            vis_payloads[episode_index] = (n, out_videos, action)
        if episodes[episode_index] != {"episode_index": episode_index, "tasks": [prompt], "length": n}:
            raise RuntimeError(f"episodes.jsonl mismatch: episode {episode_index}")
        expected_source_fields = {
            "source_record": rec.name,
            "source_task": source_task,
            "task_index": task_index,
            "task": prompt,
        }
        if any(source_map[episode_index].get(key) != value for key, value in expected_source_fields.items()):
            raise RuntimeError(f"source_records.jsonl mismatch: episode {episode_index}")
        total_frames += n
        global_index += n
        task_frame_totals[source_task] += n
        if (episode_index + 1) % 50 == 0:
            print(f"numeric validation {episode_index + 1}/{EXPECTED_RECORDS}", flush=True)

    if total_frames != EXPECTED_FRAMES or len(video_jobs) != EXPECTED_VIDEOS:
        raise RuntimeError((total_frames, len(video_jobs)))
    for directory, _prompt, _records, expected_frames in TASKS:
        if task_frame_totals[directory] != expected_frames:
            raise RuntimeError(
                f"Source frame count mismatch for {directory}: "
                f"{task_frame_totals[directory]} != {expected_frames}"
            )

    decoded = 0
    max_pts_error = 0.0
    # PyAV's internal decoder threads can deadlock after a Linux fork. A thread
    # pool still parallelizes I/O/FFmpeg work and avoids that unsafe state.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(decode_video_job, job) for job in video_jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            decoded += 1
            max_pts_error = max(max_pts_error, float(result["max_pts_error_s"]))
            if decoded % 50 == 0:
                print(f"full video decode {decoded}/{EXPECTED_VIDEOS}", flush=True)

    vis_dir = audit_root / "visualizations"
    for episode_index in visualized:
        n, videos, action = vis_payloads[episode_index]
        make_visualization(vis_dir / f"episode_{episode_index:06d}.png", episode_index, n, videos, action)
        print(f"visualized episode {episode_index:06d}", flush=True)

    smoke = run_fastwam_smoke(dataset_root, fastwam_root, audit_root / "fastwam_smoke_work")
    json_write(audit_root / "FASTWAM_DATALOADER_SMOKE_PASSED.json", smoke)
    report = {
        "status": "passed",
        "dataset_root": str(dataset_root),
        "source_root": str(source_root),
        "lerobot_version": "v2.1",
        "episodes": EXPECTED_RECORDS,
        "frames": total_frames,
        "fps": FPS,
        "validation_split": None,
        "tasks": expected_tasks,
        "camera_order": list(CAMERA_ORDER),
        "videos_full_decoded": decoded,
        "max_video_pts_error_s": max_pts_error,
        "video_sha256_matches_source": True,
        "max_state_abs_error_vs_source_definition": max_state_error,
        "max_action_abs_error_vs_source_definition": max_action_error,
        "max_source_timestamp_abs_error_s": max_source_ts_error,
        "max_lerobot_cfr_timestamp_abs_error_s": max_lerobot_ts_error,
        "max_abs_native_joint_action": max_native_action,
        "nan_or_inf_count": 0,
        "missing_or_bad_frame_count": 0,
        "out_of_bounds_index_count": 0,
        "visualized_episode_indices": visualized,
        "visualization_files": [str(vis_dir / f"episode_{x:06d}.png") for x in visualized],
        "fastwam_smoke": smoke,
    }
    json_write(audit_root / "VALIDATION_REPORT.json", report)
    json_write(audit_root / "FASTWAM_REALWORLD_15HZ_MULTITASK_V21_PASSED.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
