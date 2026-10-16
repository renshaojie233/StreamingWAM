from __future__ import annotations

import collections
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Protocol

import numpy as np

from .contracts import ACTION_DIM
from .contracts import ACTION_PREFIX_STEPS
from .contracts import ACTION_TO_DELTA_RAD
from .contracts import CONTROL_HZ
from .contracts import FUTURE_ACTION_HORIZON
from .contracts import IMAGE_HISTORY_FRAMES
from .contracts import IMAGE_HISTORY_SHAPE
from .contracts import TRAINED_REPLAN_K
from .io import RobotIO
from .preprocess import model_image
from .preprocess import model_state


class RollingPolicy(Protocol):
    def reset(self) -> None: ...

    def infer(
        self,
        task: str,
        image_history: np.ndarray,
        state: np.ndarray,
        action_prefix: np.ndarray,
        *,
        replan_k: int,
        seed: int,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class Request:
    sequence: int
    submit_model_clock: int | None
    image_history: np.ndarray
    state: np.ndarray
    action_prefix: np.ndarray


@dataclass(frozen=True)
class Plan:
    sequence: int
    submit_model_clock: int | None
    actions: np.ndarray
    warm_start: bool


def native_to_qdot(native_joint_action: np.ndarray) -> np.ndarray:
    value = np.asarray(native_joint_action, dtype=np.float32)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("native joint action must be finite shape (7,)")
    if float(np.max(np.abs(value))) > 1.0001:
        raise ValueError("native joint action is outside [-1,1]")
    return value * np.float32(ACTION_TO_DELTA_RAD * CONTROL_HZ)


def action_prefix(history: collections.deque[np.ndarray]) -> np.ndarray:
    if not history:
        raise ValueError("action history is empty")
    selected = list(history)
    while len(selected) < ACTION_PREFIX_STEPS:
        selected.insert(0, selected[0])
    result = np.stack(selected[-ACTION_PREFIX_STEPS:]).astype(np.float32)
    if result.shape != (ACTION_PREFIX_STEPS, ACTION_DIM):
        raise RuntimeError(f"invalid action prefix {result.shape}")
    return result


def image_history(history: collections.deque[np.ndarray]) -> np.ndarray:
    if not history:
        raise ValueError("image history is empty")
    selected = list(history)
    while len(selected) < IMAGE_HISTORY_FRAMES:
        selected.insert(0, selected[0])
    result = np.stack(selected[-IMAGE_HISTORY_FRAMES:], axis=1).astype(np.float32)
    if result.shape != IMAGE_HISTORY_SHAPE:
        raise RuntimeError(f"invalid image history {result.shape}")
    return result


def result_index(plan: Plan, model_clock: int | None) -> int:
    if plan.warm_start:
        if model_clock is not None or plan.submit_model_clock is not None:
            raise RuntimeError("misaligned warm-start plan")
        return 0
    if model_clock is None or plan.submit_model_clock is None:
        raise RuntimeError("rolling plan has no model-clock origin")
    index = model_clock - plan.submit_model_clock
    if not 0 <= index < FUTURE_ACTION_HORIZON:
        raise RuntimeError(f"rolling result expired at index {index}")
    return index


class Controller:
    """Plain 15 Hz rolling executor with no shift, blending, or smoothing."""

    def __init__(
        self,
        policy: RollingPolicy,
        io: RobotIO,
        *,
        task: str,
        replan_k: int,
        seed: int = 42,
        execute: bool = False,
    ) -> None:
        if replan_k not in TRAINED_REPLAN_K:
            raise ValueError(f"replan_k must be one of {TRAINED_REPLAN_K}")
        self.policy = policy
        self.io = io
        self.task = task
        self.replan_k = int(replan_k)
        self.seed = int(seed)
        self.execute = bool(execute)

    def _infer(self, request: Request) -> Plan:
        actions = self.policy.infer(
            self.task,
            request.image_history,
            request.state,
            request.action_prefix,
            replan_k=self.replan_k,
            seed=self.seed,
        )
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (FUTURE_ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid rolling plan {actions.shape}")
        if float(np.max(np.abs(actions[:, :7]))) > 1.0001:
            raise RuntimeError("model returned arm actions outside native [-1,1]")
        actions = actions.copy()
        actions[:, 7] = np.clip(actions[:, 7], 0.0, 1.0)
        return Plan(
            sequence=request.sequence,
            submit_model_clock=request.submit_model_clock,
            actions=actions,
            warm_start=request.submit_model_clock is None,
        )

    def run(self, max_timesteps: int) -> None:
        if max_timesteps <= 0:
            raise ValueError("max_timesteps must be positive")
        self.policy.reset()
        first = self.io.observe()
        latest_gripper = float(first.gripper_command)
        images: collections.deque[np.ndarray] = collections.deque(maxlen=IMAGE_HISTORY_FRAMES)
        images.append(model_image(first))
        zero = np.concatenate((np.zeros(7, dtype=np.float32), [latest_gripper])).astype(np.float32)
        actions: collections.deque[np.ndarray] = collections.deque(
            (zero.copy() for _ in range(ACTION_PREFIX_STEPS)), maxlen=ACTION_PREFIX_STEPS
        )
        warm_request = Request(0, None, image_history(images), model_state(first), action_prefix(actions))
        active = self._infer(warm_request)
        active_index = 0
        model_clock: int | None = 0
        last_submit_clock = 0
        sequence = 1
        future: Future[Plan] | None = None
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="streamingwam")
        next_tick = time.monotonic()

        if self.execute:
            self.io.start()
        try:
            for _tick in range(max_timesteps):
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)
                next_tick = max(next_tick + 1.0 / CONTROL_HZ, time.monotonic())

                observation = self.io.observe()
                images.append(model_image(observation))
                current_state = model_state(observation)

                if future is not None and future.done():
                    plan = future.result()
                    index = result_index(plan, model_clock)
                    if index > self.replan_k:
                        raise RuntimeError(
                            f"rolling update missed K boundary: index={index}, K={self.replan_k}"
                        )
                    active = plan
                    active_index = index
                    future = None

                due = model_clock is not None and model_clock - last_submit_clock >= self.replan_k
                if due and future is None:
                    request = Request(
                        sequence,
                        model_clock,
                        image_history(images),
                        current_state,
                        action_prefix(actions),
                    )
                    sequence += 1
                    last_submit_clock = model_clock
                    future = pool.submit(self._infer, request)

                waiting_at_boundary = future is not None and model_clock == last_submit_clock
                if active_index >= FUTURE_ACTION_HORIZON or waiting_at_boundary:
                    native = zero.copy()
                    native[7] = latest_gripper
                    executing_plan = False
                else:
                    native = active.actions[active_index].copy()
                    active_index += 1
                    latest_gripper = float(native[7])
                    executing_plan = True

                qdot = native_to_qdot(native[:7])
                if self.execute:
                    self.io.command(qdot, latest_gripper)
                actions.append(native)
                if executing_plan and model_clock is not None:
                    model_clock += 1
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            if self.execute:
                self.io.stop()
