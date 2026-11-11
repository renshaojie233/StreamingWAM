"""Shared, opt-in LIBERO streaming data contract (no model dependencies)."""

import numpy as np
import torch
import torch.nn.functional as F


HISTORY_VIDEO_DELTAS = (-4, -3, -2, -1, 0)
FUTURE_VIDEO_DELTAS = (2, 4, 6, 8)
VIDEO_DELTAS = HISTORY_VIDEO_DELTAS + FUTURE_VIDEO_DELTAS
ACTION_DELTAS = tuple(range(-8, 8))
ACTION_PREFIX_STEPS = 8


def streaming_enabled(data_cfg) -> bool:
    """Read streaming bool or streaming.enabled; absent means legacy behavior."""
    if data_cfg is None:
        return False
    streaming = data_cfg.get("streaming", None)
    if isinstance(streaming, bool):
        return streaming
    return bool(streaming.get("enabled", False)) if streaming is not None else False


def temporal_validity(base_index: int, episode_length: int, deltas) -> np.ndarray:
    """True for requested episode-local indices, BEFORE any clipping/padding."""
    indices = base_index + np.asarray(deltas, dtype=np.int64)
    return (indices >= 0) & (indices < episode_length)


def resize_rgb_frame(image, image_size=(224, 224)) -> torch.Tensor:
    """Return CPU float32 RGB CHW [0,1], resized to 224 by torch bilinear.

    Accept numpy/PIL HWC or torch CHW/HWC, uint8 [0,255] or float [0,1]/[0,255].
    Float inputs with maximum <= 1 are interpreted as [0,1]. No augmentation,
    flips or gripper/state transforms are performed here. Inference must call
    this on the original individual camera frames, not pre-resized mosaics.
    """
    if torch.is_tensor(image):
        value = image.detach().cpu()
        if value.ndim == 2:
            value = value.unsqueeze(0)
        elif value.ndim == 3 and value.shape[0] not in (1, 3, 4):
            if value.shape[-1] not in (1, 3, 4):
                raise ValueError(f"Unsupported RGB tensor shape: {tuple(value.shape)}")
            value = value.permute(2, 0, 1)
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = array[..., None]
        if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Unsupported RGB array shape: {array.shape}")
        value = torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1)
    if value.ndim != 3 or value.shape[0] not in (1, 3, 4):
        raise ValueError(f"Unsupported RGB frame shape: {tuple(value.shape)}")
    is_uint8 = value.dtype == torch.uint8
    value = value[:3].float()
    if value.shape[0] == 1:
        value = value.repeat(3, 1, 1)
    if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 255:
        raise ValueError("Expected finite RGB pixels in [0,1] or [0,255]")
    if is_uint8 or value.max() > 1:
        value = value / 255.0
    return F.interpolate(
        value.unsqueeze(0), size=image_size, mode="bilinear", align_corners=False,
    ).squeeze(0).contiguous()


def pack_rgb_views(primary, wrist, image_size=(224, 224)) -> torch.Tensor:
    """Shared train/infer mosaic: RGB float32 [3,H,2*W], primary on left.

    image_size is (height, width) for each individual camera view.
    """
    return torch.cat(
        [resize_rgb_frame(primary, image_size), resize_rgb_frame(wrist, image_size)], dim=-1,
    )


def encode_libero_state(state) -> np.ndarray:
    """Match training's interleaved CPU sin/cos and final FP16 dataset storage."""
    raw = torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(1, 8))
    return torch.stack((raw.sin(), raw.cos()), dim=-1).reshape(1, 16).half().numpy()
