import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/train_libero_plus_two_stage.py"
SPEC = importlib.util.spec_from_file_location("train_libero_plus_two_stage", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TwoStageTrainingTest(unittest.TestCase):
    def test_commands_transfer_exact_joint_checkpoint(self):
        run_dir = Path("/tmp/example-two-stage")
        stage1, stage2, checkpoint = MODULE.build_stage_commands(
            nproc_per_node=4,
            run_dir=run_dir,
        )

        self.assertIn("task=fastwam_joint_libero_plus_stage1", stage1)
        self.assertIn("max_steps=34720", stage1)
        self.assertIn("lr_scheduler_total_steps=34720", stage1)
        self.assertIn("task=streamingwam_libero_plus", stage2)
        self.assertEqual(checkpoint.name, "step_034720.pt")
        self.assertIn(f"resume={checkpoint}", stage2)
        self.assertIn("max_steps=40000", stage2)
        self.assertIn("lr_scheduler_total_steps=60000", stage2)
        self.assertIn("lr_warmup_ratio=0.0", stage2)

    def test_stage_two_only_runs_after_checkpoint_exists(self):
        calls = []

        def fake_runner(command, *, cwd, check):
            calls.append(command)
            if len(calls) == 1:
                _, _, checkpoint = MODULE.build_stage_commands(
                    nproc_per_node=4,
                    run_dir=run_dir,
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.touch()
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            MODULE.run_two_stage(
                nproc_per_node=4,
                run_dir=run_dir,
                runner=fake_runner,
            )

        self.assertEqual(len(calls), 2)
        self.assertIn("task=fastwam_joint_libero_plus_stage1", calls[0])
        self.assertIn("task=streamingwam_libero_plus", calls[1])

    def test_missing_stage_boundary_stops_before_stage_two(self):
        calls = []

        def fake_runner(command, *, cwd, check):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                MODULE.run_two_stage(
                    nproc_per_node=4,
                    run_dir=Path(tmpdir) / "run",
                    runner=fake_runner,
                )

        self.assertEqual(len(calls), 1)

    def test_rejects_non_positive_process_count(self):
        with self.assertRaises(ValueError):
            MODULE.build_stage_commands(
                nproc_per_node=0,
                run_dir=Path("/tmp/example-two-stage"),
            )

    def test_short_smoke_schedule_updates_checkpoint_boundary(self):
        accelerate_config = Path("/tmp/accelerate-zero0.yaml")
        stage1, stage2, checkpoint = MODULE.build_stage_commands(
            nproc_per_node=1,
            run_dir=Path("/tmp/example-two-stage"),
            accelerate_config=accelerate_config,
            joint_steps=1,
            streaming_steps=1,
            streaming_scheduler_steps=1,
        )
        self.assertEqual(checkpoint.name, "step_000001.pt")
        self.assertIn(str(accelerate_config), stage1)
        self.assertIn("max_steps=1", stage1)
        self.assertIn("max_steps=1", stage2)
        self.assertIn("lr_scheduler_total_steps=1", stage2)


if __name__ == "__main__":
    unittest.main()
