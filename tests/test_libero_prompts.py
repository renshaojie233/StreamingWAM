"""CPU regressions for prompt handling and evaluation entry-point wiring."""
import ast
import concurrent.futures
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from experiments.libero.prompt_utils import (
    canonicalize_libero_plus_instruction as clean,
    existing_result_ok,
    instruction_for_model,
    prepare_prompt_output,
)

ROOT = Path(__file__).resolve().parents[1]


class PromptTests(unittest.TestCase):
    def test_full_recorded_run_matches_sanitizer_and_manifest(self):
        evidence = ROOT / "docs/evaluation/liberoplus_clean_20261016.csv"
        summary = json.loads(evidence.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(evidence.read_bytes()).hexdigest(), summary["per_task_csv_sha256"])
        with evidence.open() as handle:
            rows = list(csv.DictReader(handle))
        expected = {tuple(line.split(",")) for line in
                    (ROOT / "assets/libero_plus_full_10030.txt").read_text().splitlines()}
        self.assertEqual(len(rows), 10030)
        self.assertEqual({(r["suite"], r["task_id"]) for r in rows}, expected)
        self.assertEqual(sum(int(r["successes"]) for r in rows), 8237)
        self.assertEqual(sum(r["instruction_raw"] != r["instruction_model"] for r in rows), 5344)
        for row in rows:
            self.assertEqual(instruction_for_model(row["instruction_raw"], "clean"), row["instruction_model"])
            self.assertEqual(instruction_for_model(row["instruction_raw"], "legacy"), row["instruction_raw"])

    def test_metadata_suffixes(self):
        for suffix in ("table 17", "tb 12", "light 3", "add 2", "level2 sample17",
                       "view -1.2 2 3.4 -5 6 initstate 2 noise 4", "TABLE 17"):
            with self.subTest(suffix=suffix):
                self.assertEqual(clean("put the apple in the bowl " + suffix), "put the apple in the bowl")

    def test_preserves_language_perturbations_and_instruction_content(self):
        for text in ("please put the apple in the bowl", "move 2 apples from the table",
                     "turn on the light", "put the cup on table 2 and close the drawer",
                     "take the second cup", "place the red cup beside the blue bowl"):
            self.assertEqual(clean(text), text)
            self.assertEqual(clean(text + " tb 3"), text)
        self.assertEqual(clean("  put\n the apple\t in the bowl  "), "put the apple in the bowl")
        # Only the metadata formats in the verified run are recognized.
        self.assertEqual(clean("move the apple view 1 2 initstate 3 noise 4"),
                         "move the apple view 1 2 initstate 3 noise 4")

    def test_legacy_exact_passthrough(self):
        raw = " put\n the apple in the bowl table 17  "
        self.assertEqual(instruction_for_model(raw, "legacy"), raw)
        self.assertEqual(instruction_for_model(raw, "clean"), "put the apple in the bowl")

    def test_bad_configuration(self):
        with self.assertRaises(ValueError):
            instruction_for_model("put the apple", "cleen")
        with self.assertRaises(ValueError):
            instruction_for_model(" \n\t", "clean")

    def test_idempotent(self):
        text = "move the apple level1 sample2"
        self.assertEqual(clean(clean(text)), clean(text))


class ResumeTests(unittest.TestCase):
    def test_protocol_directories_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prepare_prompt_output(root, "clean")
            prepare_prompt_output(root, "clean")
            with self.assertRaisesRegex(ValueError, "separate EVALUATION.output_dir"):
                prepare_prompt_output(root, "legacy")

    def test_old_unlabelled_results_are_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path = root / "gpu7_task9_results.json"
            path.write_text(json.dumps({"successes": 1}))
            with self.assertRaises(ValueError):
                prepare_prompt_output(root, "clean")
            self.assertFalse((root / "prompt_protocol.json").exists())
            prepare_prompt_output(root, "legacy")
            self.assertTrue(existing_result_ok(path, "legacy"))
            with self.assertRaises(ValueError):
                existing_result_ok(path, "clean")

    def test_failed_results_retry_within_same_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "gpu0_task1_results.json"
            self.assertFalse(existing_result_ok(path, "clean"))
            path.write_text(json.dumps({"prompt_protocol": "clean", "error": "failed"}))
            self.assertFalse(existing_result_ok(path, "clean"))
            with self.assertRaises(ValueError):
                existing_result_ok(path, "legacy")
            path.write_text("incomplete {")
            self.assertFalse(existing_result_ok(path, "clean"))

    def test_concurrent_workers_publish_complete_marker(self):
        with tempfile.TemporaryDirectory() as d, concurrent.futures.ThreadPoolExecutor(8) as pool:
            self.assertEqual(list(pool.map(lambda _: prepare_prompt_output(Path(d), "clean"), range(32))),
                             ["clean"] * 32)
            self.assertEqual(json.loads((Path(d) / "prompt_protocol.json").read_text()),
                             {"prompt_protocol": "clean"})


class EvaluatorIntegrationTests(unittest.TestCase):
    def test_real_task_runner_passes_selected_instruction_to_episode(self):
        # Execute the actual task runner with only the simulator/rollout boundary mocked.
        path = ROOT / "experiments/libero/eval_libero_single.py"
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "run_single_task")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
        ast.fix_missing_locations(module)
        raw = "put the apple in the bowl table 17"
        class Config(dict):
            __getattr__ = dict.__getitem__
        for protocol, expected in (("clean", "put the apple in the bowl"), ("legacy", raw)):
            with self.subTest(protocol=protocol):
                rollout = Mock(return_value=(True, [], [], None))
                ns = {"get_libero_env": Mock(return_value=(object(), raw)),
                      "LIBERO_ENV_RESOLUTION": 256, "instruction_for_model": instruction_for_model,
                      "run_single_episode": rollout}
                exec(compile(module, str(path), "exec"), ns)
                cfg = Config(EVALUATION=Config(prompt_protocol=protocol, num_trials=1, save_rollout_video=False))
                result = ns["run_single_task"](None, [None], None, None, cfg, None, None,
                    action_horizon=32, input_w=224, input_h=224, model_device="cpu")
                self.assertEqual(rollout.call_args.kwargs["task_description"], expected)
                self.assertEqual(result["task_description_raw"], raw)
                self.assertEqual(result["task_description_model"], expected)
                self.assertEqual(result["prompt_protocol"], protocol)
                self.assertEqual(result["prompt_metadata_suffix_removed"], protocol == "clean")
                self.assertEqual(result["successes"], 1)

    def test_all_entrypoints_check_protocol_before_model_allocation(self):
        paths = list((ROOT / "experiments/libero").glob("eval_libero_task_list*.py"))
        paths.append(ROOT / "experiments/libero/eval_libero_single.py")
        for path in paths:
            with self.subTest(entrypoint=path.name):
                source = path.read_text()
                self.assertLess(source.index("    prepare_prompt_output(") if path.name == "eval_libero_single.py"
                                else source.index("    prompt_protocol = prepare_prompt_output("),
                                source.index("    model = instantiate(cfg.model"))
                if "task_list" in path.name:
                    self.assertIn("_existing_result_ok(output_file, prompt_protocol)", source)


if __name__ == "__main__":
    unittest.main()
