from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .contracts import RESET_GRIPPER
from .contracts import RESET_JOINTS_RAD
from .contracts import TRAINED_REPLAN_K
from .controller import Controller
from .io import load_io_factory
from .policy import StreamingWAMPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plain StreamingWAM real-robot rolling execution")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--base-model-root", type=Path)
    parser.add_argument("--io-factory", required=True, help="package.module:function")
    parser.add_argument("--task", default="put the apple into the bowl")
    parser.add_argument("--replan-k", type=int, choices=TRAINED_REPLAN_K, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-timesteps", type=int, default=18_000)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shadow", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--confirm-real-robot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.execute and not args.confirm_real_robot:
        raise SystemExit("Real execution requires --confirm-real-robot")
    if args.reset and not args.execute:
        raise SystemExit("--reset is only valid with --execute")
    io = load_io_factory(args.io_factory, args)
    if args.reset:
        io.reset(np.asarray(RESET_JOINTS_RAD, dtype=np.float32), RESET_GRIPPER)
    policy = StreamingWAMPolicy(args.assets, base_model_root=args.base_model_root)
    controller = Controller(
        policy,
        io,
        task=args.task,
        replan_k=args.replan_k,
        seed=args.seed,
        execute=args.execute,
    )
    controller.run(args.max_timesteps)


if __name__ == "__main__":
    main()
