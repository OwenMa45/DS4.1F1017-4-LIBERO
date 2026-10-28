import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ds_libero.artifacts import Store, write_json
from ds_libero.benchmark import create_plan, load_plan, report_plan, run_plan_episode, wilson
from ds_libero.config import Config, digest
from ds_libero.errors import ExperimentError


class BenchmarkTests(unittest.TestCase):
    def make_plan(self, root, tasks=(0, 1), states=(0, 1)):
        return create_plan(Config.load("configs/direct_v2_high.json"), "libero_spatial",
                           list(tasks), list(states), 0, root)

    def save_result(self, root, entry, status, renderer="test"):
        store = Store(root / entry["run_directory"] / "episode", "run", Config.from_dict(entry["config"]))
        store.environment({"renderer": renderer, "task_name": f"task_{entry['config']['task_id']}",
                           "official_init_state_sha256": str(entry["config"]["init_state_id"])})
        scored = status in ("success", "failure", "invalid_output")
        store.finish({"status": status, "scored": scored,
                      "task_success": status == "success" if scored else None})
        return store

    def test_plan_creation_is_offline_and_each_identity_is_unique(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "plan"
            with patch("ds_libero.client.Client", side_effect=AssertionError("No API")):
                plan = self.make_plan(root)
            self.assertEqual(len(plan["episodes"]), 4)
            _, loaded = load_plan(root)
            self.assertEqual(loaded, plan)
            self.assertEqual(len(list((root / "configs").glob("*.json"))), 4)
            self.assertEqual(plan["max_http_requests_per_episode_including_api_check"], 303)

    def test_invalid_and_duplicate_indices_are_rejected_before_directory_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "plan"
            for tasks, states in (([0, 0], [0]), ([10], [0]), ([0], [-1]), ([], [0])):
                with self.subTest(tasks=tasks, states=states), self.assertRaises(ExperimentError):
                    self.make_plan(root, tasks, states)
                self.assertFalse(root.exists())

    def test_pilot_stays_fixed_while_benchmark_horizon_is_checked(self):
        values = Config.load("configs/direct_v2_high.json").to_dict()
        with self.assertRaises(ExperimentError):
            Config.from_dict({**values, "task_id": 1})
        cfg = Config.from_dict({**values, "evaluation_scope": "benchmark", "suite": "libero_10",
                                "max_steps": 520, "task_id": 9, "init_state_id": 49})
        self.assertEqual(cfg.max_steps, 520)
        with self.assertRaises(ExperimentError):
            Config.from_dict({**cfg.to_dict(), "max_steps": 220})

    def test_missing_and_infrastructure_trials_are_not_scored_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "plan"
            plan = self.make_plan(root)
            self.save_result(root, plan["episodes"][0], "success")
            self.save_result(root, plan["episodes"][1], "invalid_output")
            self.save_result(root, plan["episodes"][2], "api_error")
            report = report_plan(root)
            self.assertEqual((report["planned"], report["scored"], report["successes"]), (4, 2, 1))
            self.assertEqual(report["scored_success_rate"], .5)
            self.assertEqual(report["coverage"], .5)
            self.assertEqual(report["planned_success_bounds"], [.25, .75])
            self.assertIsNone(report["macro_task_success_rate"])
            self.assertFalse(report["complete"])
            self.assertEqual(report["status_counts"]["not_started"], 1)

    def test_complete_plan_reports_per_task_and_macro_rate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "plan"
            plan = self.make_plan(root)
            for entry, status in zip(plan["episodes"], ("success", "success", "success", "failure")):
                self.save_result(root, entry, status)
            report = report_plan(root)
            self.assertTrue(report["complete"])
            self.assertEqual(report["macro_task_success_rate"], .75)
            self.assertEqual(report["per_task"]["0"]["scored_success_rate"], 1.)
            self.assertEqual(report["per_task"]["1"]["scored_success_rate"], .5)
            self.assertTrue((root / "episodes.csv").is_file())

    def test_changed_config_and_changed_source_block_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "plan"
            plan = self.make_plan(root)
            with patch("ds_libero.benchmark.source_fingerprint", return_value="changed"):
                with self.assertRaises(ExperimentError):
                    run_plan_episode(root, 0)
            entry = plan["episodes"][0]
            write_json(root / entry["config_file"], {**entry["config"], "reasoning_effort": "low"})
            with self.assertRaises(ExperimentError):
                load_plan(root)

    def test_mixed_runtime_is_rejected_instead_of_pooled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "plan"
            plan = self.make_plan(root)
            self.save_result(root, plan["episodes"][0], "success")
            self.save_result(root, plan["episodes"][1], "success", renderer="different")
            with self.assertRaises(ExperimentError):
                report_plan(root)

    def test_preflight_failure_is_visible_and_not_model_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "plan"
            plan = self.make_plan(root, tasks=(0,), states=(0,))
            phase = root / plan["episodes"][0]["run_directory"] / "api-check"
            phase.mkdir(parents=True)
            write_json(phase / "result.json", {"status": "invalid_output"})
            report = report_plan(root)
            self.assertEqual(report["scored"], 0)
            self.assertEqual(report["episodes"][0]["preflight_status"], {"phase": "api-check", "status": "invalid_output"})

    def test_unfinished_request_is_counted_with_unknown_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "plan"
            plan = self.make_plan(root, tasks=(0,), states=(0,))
            store = self.save_result(root, plan["episodes"][0], "interrupted")
            store.event({"event": "api_started", "attempt": 1, "step": 0})
            report = report_plan(root)
            self.assertEqual(report["http_requests"], 1)
            self.assertTrue(report["usage_incomplete"])
            self.assertEqual(report["scored"], 0)

    def test_inconsistent_score_and_policy_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "plan"
            plan = self.make_plan(root)
            store = self.save_result(root, plan["episodes"][0], "success")
            store.finish({"status": "api_error", "scored": True, "task_success": True})
            with self.assertRaises(ExperimentError):
                report_plan(root)
            plan["episodes"][1]["config"]["reasoning_effort"] = "low"
            entry = plan["episodes"][1]
            entry["config_sha256"] = digest(entry["config"])
            write_json(root / entry["config_file"], entry["config"])
            plan["plan_sha256"] = digest({k: v for k, v in plan.items() if k != "plan_sha256"})
            write_json(root / "plan.json", plan)
            with self.assertRaises(ExperimentError):
                load_plan(root)

    def test_wilson_zero_small_and_all_success(self):
        self.assertIsNone(wilson(0, 0))
        self.assertAlmostEqual(wilson(0, 10)[1], .2775328, places=6)
        self.assertAlmostEqual(wilson(10, 10)[0], .7224672, places=6)
        self.assertLess(wilson(1, 2)[0], .1)
        self.assertGreater(wilson(1, 2)[1], .9)


if __name__ == "__main__":
    unittest.main()
