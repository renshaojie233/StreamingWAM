import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import hydra
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict


def _setup_paths() -> Path:
    project_root = Path.cwd().resolve()
    for path in (project_root, project_root / "experiments" / "libero"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return project_root


_setup_paths()

from eval_libero_single import (  # noqa: E402
    FastWAMProcessor,
    NumpyEncoder,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _validate_visualize_future_video_cfg,
    benchmark,
    load_dataset_stats_from_json,
    run_single_task,
    set_global_seed,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)], replace=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _read_task_list(path: str | os.PathLike[str]) -> list[tuple[str, int]]:
    tasks: list[tuple[str, int]] = []
    task_path = Path(os.path.expanduser(os.path.expandvars(str(path))))
    with task_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 2:
                raise ValueError(f"Bad task-list line {line_no} in {task_path}: {raw_line!r}")
            tasks.append((parts[0], int(parts[1])))
    if not tasks:
        raise ValueError(f"Task list is empty: {task_path}")
    return tasks


def _shard_tasks(tasks: list[tuple[str, int]], worker_index: int, num_workers: int) -> list[tuple[str, int]]:
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError(f"worker_index must be in [0, {num_workers}), got {worker_index}")
    return [task for i, task in enumerate(tasks) if i % num_workers == worker_index]


def _existing_result_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return "error" not in payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, cls=NumpyEncoder)


def _parse_k_values(raw: Any) -> list[int]:
    if raw is None:
        return [1, 2, 4, 8, 16]
    if isinstance(raw, str):
        values = [int(x) for x in raw.replace(",", " ").split()]
    else:
        values = [int(x) for x in raw]
    if not values:
        raise ValueError("EVALUATION.sdp_k_values is empty.")
    return values


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig):
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    PartialState().config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    task_list_file = cfg.EVALUATION.get("task_list_file")
    if task_list_file is None:
        raise ValueError("Pass +EVALUATION.task_list_file=/path/to/tasks.txt")
    worker_index = int(cfg.EVALUATION.get("worker_index", 0))
    num_workers = int(cfg.EVALUATION.get("num_workers", 1))
    k_values = _parse_k_values(cfg.EVALUATION.get("sdp_k_values"))
    base_output_root = Path(str(cfg.EVALUATION.output_dir))

    all_tasks = _read_task_list(task_list_file)
    tasks = _shard_tasks(all_tasks, worker_index, num_workers)
    if not tasks:
        raise ValueError(f"Worker {worker_index}/{num_workers} received no tasks from {task_list_file}")

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    logging.info(
        "Worker %s/%s loading model once on %s for %s tasks x K=%s.",
        worker_index,
        num_workers,
        model_device,
        len(tasks),
        k_values,
    )
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = int(cfg.data.train.num_frames) - 1 if action_horizon_cfg is None else int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])

    benchmark_dict = benchmark.get_benchmark_dict()
    suite_cache: dict[str, Any] = {}
    overall_summary: dict[str, Any] = {
        "worker_index": worker_index,
        "num_workers": num_workers,
        "gpu_id": int(cfg.gpu_id),
        "task_list_file": str(task_list_file),
        "assigned_tasks": len(tasks),
        "k_values": k_values,
        "per_k": {},
        "elapsed_seconds": 0,
    }

    requested_save_rollout_video = bool(cfg.EVALUATION.get("save_rollout_video", True))

    for k in k_values:
        with open_dict(cfg.EVALUATION):
            cfg.EVALUATION.use_sdp_rolling_inference = True
            cfg.EVALUATION.sdp_action_steps = int(k)
            cfg.EVALUATION.sdp_execute_steps = int(k)
            cfg.EVALUATION.replan_steps = int(k)
            cfg.EVALUATION.use_action_ensembler = False
            cfg.EVALUATION.visualize_future_video = False
            cfg.EVALUATION.save_rollout_video = requested_save_rollout_video
            cfg.EVALUATION.output_dir = str(base_output_root / f"K{k}")

        output_root = Path(str(cfg.EVALUATION.output_dir))
        output_root.mkdir(parents=True, exist_ok=True)
        skip_existing = bool(cfg.EVALUATION.get("skip_existing", True))
        completed: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        logging.info("Worker %s/%s starts K=%s with %s tasks.", worker_index, num_workers, k, len(tasks))
        for local_idx, (suite_name, task_id) in enumerate(tasks, start=1):
            with open_dict(cfg.EVALUATION):
                cfg.EVALUATION.task_suite_name = suite_name
                cfg.EVALUATION.task_id = int(task_id)

            suite_output_dir = output_root / suite_name
            output_file = suite_output_dir / f"gpu{cfg.gpu_id}_task{task_id}_results.json"
            if skip_existing and _existing_result_ok(output_file):
                skipped.append({"task_suite": suite_name, "task_id": int(task_id)})
                continue

            task_start = time.time()
            try:
                if suite_name not in suite_cache:
                    suite_cache[suite_name] = benchmark_dict[suite_name]()
                task_suite = suite_cache[suite_name]
                task = task_suite.get_task(task_id)
                initial_states = task_suite.get_task_init_states(task_id)
                while len(initial_states) < int(cfg.EVALUATION.num_trials):
                    initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

                results = {
                    "task_suite": suite_name,
                    "task_id": int(task_id),
                    "task_description": None,
                    "successes": 0,
                    "total_episodes": int(cfg.EVALUATION.num_trials),
                    "gpu_id": int(cfg.gpu_id),
                    "worker_index": worker_index,
                    "num_workers": num_workers,
                    "success_episodes": [],
                    "failure_episodes": [],
                    "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": 0,
                    "use_sdp_rolling_inference": True,
                    "sdp_action_steps": int(k),
                    "sdp_execute_steps": int(k),
                    "sdp_num_noise_levels": int(cfg.EVALUATION.get("sdp_num_noise_levels", 32)),
                    "sdp_warm_start": bool(cfg.EVALUATION.get("sdp_warm_start", True)),
                    "save_rollout_video": requested_save_rollout_video,
                }
                task_results = run_single_task(
                    task=task,
                    initial_states=initial_states,
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    video_dir=output_root / suite_name / "videos",
                    predicted_video_dir=output_root / suite_name / "predicted_videos",
                    action_horizon=action_horizon,
                    input_w=input_w,
                    input_h=input_h,
                    model_device=model_device,
                )
                results.update(task_results)
                results["duration"] = time.time() - task_start
                _write_json(output_file, results)
                completed.append(
                    {
                        "task_suite": suite_name,
                        "task_id": int(task_id),
                        "successes": int(results["successes"]),
                        "total_episodes": int(results["total_episodes"]),
                        "duration": float(results["duration"]),
                    }
                )
                logging.info(
                    "Worker %s/%s K=%s finished %s task_id=%s: %s/%s in %.2fs (%s/%s).",
                    worker_index,
                    num_workers,
                    k,
                    suite_name,
                    task_id,
                    results["successes"],
                    results["total_episodes"],
                    results["duration"],
                    local_idx,
                    len(tasks),
                )
            except torch.cuda.OutOfMemoryError:
                raise
            except BaseException as exc:
                error_payload = {
                    "task_suite": suite_name,
                    "task_id": int(task_id),
                    "gpu_id": int(cfg.gpu_id),
                    "worker_index": worker_index,
                    "num_workers": num_workers,
                    "successes": 0,
                    "total_episodes": int(cfg.EVALUATION.num_trials),
                    "duration": time.time() - task_start,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "sdp_action_steps": int(k),
                }
                _write_json(output_file, error_payload)
                failed.append(error_payload)
                logging.exception("Worker %s/%s K=%s failed %s task_id=%s.", worker_index, num_workers, k, suite_name, task_id)

            summary = {
                "worker_index": worker_index,
                "num_workers": num_workers,
                "gpu_id": int(cfg.gpu_id),
                "task_list_file": str(task_list_file),
                "assigned_tasks": len(tasks),
                "k": int(k),
                "completed_tasks": len(completed),
                "failed_tasks": len(failed),
                "skipped_tasks": len(skipped),
                "elapsed_seconds": time.time() - start_time,
                "completed": completed,
                "failed": [
                    {"task_suite": x["task_suite"], "task_id": x["task_id"], "error": x["error"]} for x in failed
                ],
                "skipped": skipped,
            }
            _write_json(output_root / f"worker_{worker_index:02d}_summary.json", summary)

        overall_summary["per_k"][str(k)] = {
            "completed_tasks": len(completed),
            "failed_tasks": len(failed),
            "skipped_tasks": len(skipped),
        }
        overall_summary["elapsed_seconds"] = time.time() - start_time
        _write_json(base_output_root / f"worker_{worker_index:02d}_multi_k_summary.json", overall_summary)

    logging.info("Worker %s/%s done all K=%s in %.2fs.", worker_index, num_workers, k_values, time.time() - start_time)


if __name__ == "__main__":
    main()
