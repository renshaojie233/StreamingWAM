from __future__ import annotations

import collections
import hashlib
from pathlib import Path
import sys
import tarfile
import tempfile
import time
import unittest

import numpy as np

from real_robot.streamingwam_deploy.contracts import ACTION_PREFIX_STEPS
from real_robot.streamingwam_deploy.controller import Plan
from real_robot.streamingwam_deploy.controller import Controller
from real_robot.streamingwam_deploy.controller import action_prefix
from real_robot.streamingwam_deploy.controller import native_to_qdot
from real_robot.streamingwam_deploy.controller import result_index
from real_robot.streamingwam_deploy.io import Observation
from real_robot.streamingwam_deploy.preprocess import model_image


class ReferenceTests(unittest.TestCase):
    def test_release_downloader_reassembles_and_verifies(self) -> None:
        from real_robot import download_assets

        def digest(data: bytes) -> str:
            return hashlib.sha256(data).hexdigest()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            parts = {"weight.part-000": b"abc", "weight.part-001": b"def"}
            for name, data in parts.items():
                (source / name).write_bytes(data)
            runtime = source / "runtime.tar.gz"
            stats = root / "stats.json"
            stats.write_text("{}", encoding="utf-8")
            with tarfile.open(runtime, "w:gz") as archive:
                archive.add(stats, arcname="train_split_zscore_stats.json")

            original = (
                download_assets.RELEASE_BASE,
                download_assets.CHECKPOINT_SHA256,
                download_assets.RUNTIME_ARCHIVE,
                download_assets.RUNTIME_ARCHIVE_SHA256,
                download_assets.PART_SHA256,
                sys.argv,
            )
            try:
                download_assets.RELEASE_BASE = source.as_uri()
                download_assets.CHECKPOINT_SHA256 = digest(b"abcdef")
                download_assets.RUNTIME_ARCHIVE = runtime.name
                download_assets.RUNTIME_ARCHIVE_SHA256 = download_assets.sha256(runtime)
                download_assets.PART_SHA256 = {name: digest(data) for name, data in parts.items()}
                sys.argv = ["download_assets.py", "--output", str(output)]
                download_assets.main()
            finally:
                (
                    download_assets.RELEASE_BASE,
                    download_assets.CHECKPOINT_SHA256,
                    download_assets.RUNTIME_ARCHIVE,
                    download_assets.RUNTIME_ARCHIVE_SHA256,
                    download_assets.PART_SHA256,
                    sys.argv,
                ) = original
            self.assertEqual((output / "step_030000.pt").read_bytes(), b"abcdef")
            self.assertEqual((output / "train_split_zscore_stats.json").read_text(), "{}")

    def test_three_camera_order_and_shape(self) -> None:
        def image(value: int) -> np.ndarray:
            return np.full((20, 30, 3), value, dtype=np.uint8)

        observation = Observation(
            camera_1=image(10),
            camera_3=image(20),
            camera_2=image(30),
            joint_position_rad=np.zeros(7, dtype=np.float32),
            gripper_command=1.0,
            timestamp_s=1.0,
        )
        result = model_image(observation)
        self.assertEqual(result.shape, (3, 384, 224))
        means = [float(result[:, start : start + 128].mean()) for start in (0, 128, 256)]
        expected = [value / 127.5 - 1.0 for value in (10, 20, 30)]
        np.testing.assert_allclose(means, expected, atol=1e-6)

    def test_action_units(self) -> None:
        np.testing.assert_allclose(native_to_qdot(np.ones(7, dtype=np.float32)), 3.0)

    def test_prefix_repeats_earliest_action(self) -> None:
        sample = np.arange(8, dtype=np.float32)
        result = action_prefix(collections.deque([sample]))
        self.assertEqual(result.shape, (ACTION_PREFIX_STEPS, 8))
        np.testing.assert_array_equal(result, np.repeat(sample[None], ACTION_PREFIX_STEPS, axis=0))

    def test_raw_time_alignment(self) -> None:
        plan = Plan(1, 8, np.zeros((16, 8), dtype=np.float32), warm_start=False)
        self.assertEqual(result_index(plan, 11), 3)

    def test_no_shift_or_smoothing_surface(self) -> None:
        from real_robot.streamingwam_deploy import run

        source = Path(run.__file__).read_text(encoding="utf-8")
        forbidden = ("shift_steps", "lead_ticks", "smoothing_strength", "transition_duration")
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_end_to_end_control_clock(self) -> None:
        class FakePolicy:
            def __init__(self) -> None:
                self.calls = 0

            def reset(self) -> None:
                self.calls = 0

            def infer(self, *_args, **_kwargs) -> np.ndarray:
                self.calls += 1
                result = np.zeros((16, 8), dtype=np.float32)
                result[:, :7] = 0.1 * self.calls
                result[:, 7] = 0.75
                return result

        class FakeIO:
            def __init__(self) -> None:
                self.commands: list[tuple[np.ndarray, float]] = []
                self.started = False
                self.stopped = False

            def observe(self) -> Observation:
                image = np.zeros((16, 16, 3), dtype=np.uint8)
                return Observation(
                    camera_1=image,
                    camera_3=image,
                    camera_2=image,
                    joint_position_rad=np.zeros(7, dtype=np.float32),
                    gripper_command=1.0,
                    timestamp_s=time.monotonic(),
                )

            def reset(self, _joints: np.ndarray, _gripper: float) -> None:
                return None

            def start(self) -> None:
                self.started = True

            def command(self, qdot: np.ndarray, gripper: float) -> None:
                self.commands.append((qdot.copy(), float(gripper)))

            def stop(self) -> None:
                self.stopped = True

        policy = FakePolicy()
        io = FakeIO()
        Controller(
            policy,
            io,
            task="put the apple into the bowl",
            replan_k=2,
            execute=True,
        ).run(max_timesteps=6)
        self.assertTrue(io.started and io.stopped)
        self.assertEqual(len(io.commands), 6)
        self.assertGreaterEqual(policy.calls, 2)
        self.assertTrue(any(np.allclose(qdot, 0.3) for qdot, _ in io.commands))
        self.assertTrue(any(np.allclose(qdot, 0.0) for qdot, _ in io.commands))


if __name__ == "__main__":
    unittest.main()
