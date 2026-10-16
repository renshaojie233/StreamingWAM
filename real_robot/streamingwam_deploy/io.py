from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Observation:
    """One synchronized real-robot observation.

    Images are RGB uint8 HWC arrays before resize.  The gripper field is the
    latest commanded continuous coordinate used by training: 0=closed,
    1=open.
    """

    camera_1: np.ndarray
    camera_3: np.ndarray
    camera_2: np.ndarray
    joint_position_rad: np.ndarray
    gripper_command: float
    timestamp_s: float


@runtime_checkable
class RobotIO(Protocol):
    def observe(self) -> Observation: ...

    def reset(self, joint_position_rad: np.ndarray, gripper_command: float) -> None: ...

    def start(self) -> None: ...

    def command(self, joint_velocity_rad_s: np.ndarray, gripper_command: float) -> None: ...

    def stop(self) -> None: ...


def load_io_factory(spec: str, args: Any) -> RobotIO:
    if ":" not in spec:
        raise ValueError("--io-factory must be package.module:function")
    module_name, factory_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    io = factory(args)
    if not isinstance(io, RobotIO):
        raise TypeError("I/O factory result does not implement RobotIO")
    return io


def validate_observation(observation: Observation) -> None:
    for name in ("camera_1", "camera_3", "camera_2"):
        image = np.asarray(getattr(observation, name))
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{name} must be RGB uint8 HWC, got {image.shape} {image.dtype}")
    joints = np.asarray(observation.joint_position_rad, dtype=np.float32)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError("joint_position_rad must be finite shape (7,)")
    if not np.isfinite(observation.gripper_command) or not 0.0 <= observation.gripper_command <= 1.0:
        raise ValueError("gripper_command must be finite in [0,1]")
    if not np.isfinite(observation.timestamp_s):
        raise ValueError("timestamp_s must be finite")
