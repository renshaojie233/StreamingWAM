from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from .contracts import CAMERA_ORDER
from .contracts import PER_CAMERA_HEIGHT
from .contracts import PER_CAMERA_WIDTH
from .io import Observation
from .io import validate_observation


def _resize_rgb(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float().unsqueeze(0)
    tensor = functional.interpolate(
        tensor,
        size=(PER_CAMERA_HEIGHT, PER_CAMERA_WIDTH),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]
    return tensor


def model_image(observation: Observation) -> np.ndarray:
    """Return the exact CHW three-camera composite in [-1,1]."""
    validate_observation(observation)
    resized = [_resize_rgb(getattr(observation, name)) for name in CAMERA_ORDER]
    composite = torch.cat(resized, dim=1).mul_(1.0 / 127.5).sub_(1.0)
    result = composite.numpy().astype(np.float32, copy=False)
    if result.shape != (3, 384, 224) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid composite image {result.shape}")
    return np.ascontiguousarray(result)


def model_state(observation: Observation) -> np.ndarray:
    validate_observation(observation)
    return np.concatenate(
        (
            np.asarray(observation.joint_position_rad, dtype=np.float32),
            np.asarray([observation.gripper_command], dtype=np.float32),
        )
    )
