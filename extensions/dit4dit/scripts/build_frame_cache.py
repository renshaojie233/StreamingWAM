"""Decode each source video once, without resizing or lossy re-encoding."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import activate
activate("streaming")
from DiT4DiT.dataloader.gr00t_lerobot.frame_cache import cache_key


def decode(job):
    video, root = map(Path, job)
    video = video.resolve(strict=True)
    stat = video.stat()
    identity = {"path": str(video), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    target = root / cache_key(video)
    if (target / "complete.json").exists():
        info = json.loads((target / "complete.json").read_text())
        if info["source"] != identity:
            raise ValueError(f"Existing cache source changed: {video}")
        return info
    temporary = Path(tempfile.mkdtemp(prefix=".decode-", dir=root))
    timestamps = []
    shape = None
    try:
        with av.open(str(video)) as container, (temporary / "rgb.uint8").open("wb", buffering=8 * 1024 * 1024) as output:
            container.streams.video[0].thread_count = 2
            for frame in container.decode(video=0):
                rgb = frame.to_rgb().to_ndarray()
                if shape is None:
                    shape = rgb.shape
                if rgb.shape != shape:
                    raise ValueError(f"Video resolution changed: {video}")
                output.write(rgb.tobytes())
                timestamps.append(float(frame.pts * frame.time_base))
        pts = np.asarray(timestamps, dtype=np.float64)
        if not len(pts) or not (np.diff(pts) > 0).all():
            raise ValueError(f"Empty or nonmonotonic PTS: {video}")
        np.save(temporary / "pts.npy", pts)
        info = {"source": identity, "shape": [len(pts), *shape], "pixels": "pyav RGB24, no resize"}
        (temporary / "complete.json").write_text(json.dumps(info, indent=2))
        os.replace(temporary, target)
        return info
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    videos = sorted({path.resolve() for path in args.source_root.glob("*/videos/*/*/*.mp4")})
    if not videos:
        raise ValueError("No source videos found")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(decode, [(str(path), str(args.output_root)) for path in videos]):
            rows.append(result)
            print(json.dumps(result), flush=True)
    report = {"videos": len(rows), "frames": sum(row["shape"][0] for row in rows),
              "bytes": sum(int(np.prod(row["shape"])) for row in rows), "complete": True}
    (args.output_root / "manifest.json").write_text(json.dumps({**report, "files": rows}, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
