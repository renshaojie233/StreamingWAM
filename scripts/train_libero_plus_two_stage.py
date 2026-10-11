#!/usr/bin/env python3
"""Run the published Joint-to-StreamingWAM training protocol with one command.

The stage boundary is saved for fault recovery, but users do not need to locate
or pass the intermediate checkpoint. Stage two loads model weights only and
therefore starts with a fresh optimizer and learning-rate scheduler, matching
the protocol described in the paper.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
from pathlib import Path
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
ACCELERATE_CONFIG = REPO_ROOT / "scripts/accelerate_configs/accelerate_zero1_ds.yaml"
TRAIN_SCRIPT = REPO_ROOT / "scripts/train.py"
JOINT_TASK = "fastwam_joint_libero_plus_stage1"
STREAMING_TASK = "streamingwam_libero_plus"
JOINT_STEPS = 34_720
STREAMING_STEPS = 40_000
STREAMING_SCHEDULER_STEPS = 60_000


def _launch_command(
    *,
    nproc_per_node: int,
    accelerate_config: Path,
    task: str,
    output_dir: Path,
    overrides: Sequence[str] = (),
) -> list[str]:
    return [
        "accelerate",
        "launch",
        "--config_file",
        str(accelerate_config),
        "--num_processes",
        str(nproc_per_node),
        str(TRAIN_SCRIPT),
        f"task={task}",
        f"output_dir={output_dir}",
        f"wandb.name={task}",
        *overrides,
    ]


def build_stage_commands(
    *,
    nproc_per_node: int,
    run_dir: Path,
    accelerate_config: Path = ACCELERATE_CONFIG,
    joint_steps: int = JOINT_STEPS,
    streaming_steps: int = STREAMING_STEPS,
    streaming_scheduler_steps: int = STREAMING_SCHEDULER_STEPS,
    extra_overrides: Sequence[str] = (),
) -> tuple[list[str], list[str], Path]:
    if nproc_per_node <= 0:
        raise ValueError("`nproc_per_node` must be positive.")
    if joint_steps <= 0 or streaming_steps <= 0:
        raise ValueError("Both training stages must contain at least one step.")
    if streaming_scheduler_steps < streaming_steps:
        raise ValueError(
            "`streaming_scheduler_steps` must be greater than or equal to "
            "`streaming_steps`."
        )

    joint_dir = run_dir / "stage1_joint"
    streaming_dir = run_dir / "stage2_streaming"
    joint_checkpoint = (
        joint_dir
        / "checkpoints"
        / "weights"
        / f"step_{joint_steps:06d}.pt"
    )

    stage1 = _launch_command(
        nproc_per_node=nproc_per_node,
        accelerate_config=accelerate_config,
        task=JOINT_TASK,
        output_dir=joint_dir,
        overrides=[
            f"max_steps={joint_steps}",
            f"lr_scheduler_total_steps={joint_steps}",
            *extra_overrides,
        ],
    )
    stage2 = _launch_command(
        nproc_per_node=nproc_per_node,
        accelerate_config=accelerate_config,
        task=STREAMING_TASK,
        output_dir=streaming_dir,
        overrides=[
            f"resume={joint_checkpoint}",
            f"max_steps={streaming_steps}",
            f"lr_scheduler_total_steps={streaming_scheduler_steps}",
            "lr_warmup_ratio=0.0",
            *extra_overrides,
        ],
    )
    return stage1, stage2, joint_checkpoint


def run_two_stage(
    *,
    nproc_per_node: int,
    run_dir: Path,
    accelerate_config: Path = ACCELERATE_CONFIG,
    joint_steps: int = JOINT_STEPS,
    streaming_steps: int = STREAMING_STEPS,
    streaming_scheduler_steps: int = STREAMING_SCHEDULER_STEPS,
    extra_overrides: Sequence[str] = (),
    dry_run: bool = False,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[list[str], list[str]]:
    stage1, stage2, joint_checkpoint = build_stage_commands(
        nproc_per_node=nproc_per_node,
        run_dir=run_dir,
        accelerate_config=accelerate_config,
        joint_steps=joint_steps,
        streaming_steps=streaming_steps,
        streaming_scheduler_steps=streaming_scheduler_steps,
        extra_overrides=extra_overrides,
    )
    print("[stage 1/2] FastWAM-Joint")
    print(shlex.join(stage1))
    print("[stage 2/2] StreamingWAM")
    print(shlex.join(stage2))
    if dry_run:
        return stage1, stage2

    run_dir.mkdir(parents=True, exist_ok=True)
    runner(stage1, cwd=REPO_ROOT, check=True)
    if not joint_checkpoint.is_file():
        raise FileNotFoundError(
            "Joint training completed but the expected stage-boundary checkpoint "
            f"was not found: {joint_checkpoint}"
        )
    runner(stage2, cwd=REPO_ROOT, check=True)
    return stage1, stage2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument(
        "--accelerate-config",
        type=Path,
        default=ACCELERATE_CONFIG,
        help="Accelerate launcher config; defaults to the four-GPU ZeRO-1 setup.",
    )
    parser.add_argument("--joint-steps", type=int, default=JOINT_STEPS)
    parser.add_argument("--streaming-steps", type=int, default=STREAMING_STEPS)
    parser.add_argument(
        "--streaming-scheduler-steps",
        type=int,
        default=STREAMING_SCHEDULER_STEPS,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to runs/streamingwam_two_stage/<timestamp>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print both launch commands without starting training.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides applied to both stages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = REPO_ROOT / "runs" / "streamingwam_two_stage" / run_id
    else:
        run_dir = run_dir.expanduser().resolve()

    run_two_stage(
        nproc_per_node=args.nproc_per_node,
        run_dir=run_dir,
        accelerate_config=args.accelerate_config.expanduser().resolve(),
        joint_steps=args.joint_steps,
        streaming_steps=args.streaming_steps,
        streaming_scheduler_steps=args.streaming_scheduler_steps,
        extra_overrides=args.overrides,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
