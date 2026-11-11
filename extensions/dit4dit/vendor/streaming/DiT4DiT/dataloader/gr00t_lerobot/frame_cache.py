"""Lossless RGB/PTS cache; equivalent nearest-frame selection to torchvision PyAV."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


def cache_key(video):
    return hashlib.sha256(str(Path(video).resolve()).encode()).hexdigest()


@lru_cache(maxsize=32)
def open_cached_video(root, video, size, mtime_ns):
    folder = Path(root) / cache_key(video)
    metadata = json.loads((folder / "complete.json").read_text())
    if metadata["source"] != {"path": video, "size": size, "mtime_ns": mtime_ns}:
        raise ValueError(f"Stale frame cache for {video}")
    frames = np.memmap(folder / "rgb.uint8", mode="r", dtype=np.uint8,
                       shape=tuple(metadata["shape"]))
    timestamps = np.load(folder / "pts.npy", mmap_mode="r")
    return frames, timestamps


def cached_frames(video, timestamps, root):
    source = Path(video).resolve(strict=True)
    stat = source.stat()
    frames, pts = open_cached_video(str(Path(root).resolve()), str(source), stat.st_size, stat.st_mtime_ns)
    requested = np.asarray(timestamps, dtype=np.float64)
    right = np.searchsorted(pts, requested, side="left").clip(0, len(pts) - 1)
    left = (right - 1).clip(0, len(pts) - 1)
    # Earlier frame wins exact ties, matching the original strict '<' update.
    selected = np.where(np.abs(pts[left] - requested) <= np.abs(pts[right] - requested), left, right)
    return np.asarray(frames[selected])
