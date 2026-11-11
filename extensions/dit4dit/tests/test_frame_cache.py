import json

import numpy as np

from DiT4DiT.dataloader.gr00t_lerobot.frame_cache import cache_key, cached_frames


def test_cache_matches_nearest_timestamp_and_earlier_tie(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source identity for cache test")
    root = tmp_path / "cache"
    folder = root / cache_key(source)
    folder.mkdir(parents=True)
    images = np.arange(4 * 3 * 4 * 3, dtype=np.uint8).reshape(4, 3, 4, 3)
    images.tofile(folder / "rgb.uint8")
    pts = np.array([0., 1., 2., 3.])
    np.save(folder / "pts.npy", pts)
    stat = source.stat()
    (folder / "complete.json").write_text(json.dumps({
        "source": {"path": str(source.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "shape": list(images.shape),
    }))
    requested = np.array([-1., 0., 0.5, 1., 1.51, 2.5, 5.])
    indices = np.abs(pts[:, None] - requested).argmin(axis=0)
    np.testing.assert_array_equal(cached_frames(source, requested, root), images[indices])
