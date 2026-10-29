import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ds_libero.__main__ import parser
from ds_libero.accounting import account_attempts, pricing_snapshot
from ds_libero.artifacts import Store, write_json
from ds_libero.benchmark import create_plan, report_plan
from ds_libero.client import Client, HTTPResponse, ModelPolicy
from ds_libero.config import Config
from ds_libero.errors import ExperimentError
from ds_libero.protocol import request_payload
from ds_libero.replay import prepare_replay, replay_request
from ds_libero.report import build_campaign_report, build_report
from test_experiment import FakeEnv, observation, run_quiet
import test_v2
from test_v2 import ACTION


def usage():
    return {"prompt_tokens": 1227, "completion_tokens": 16384, "total_tokens": 17611,
            "prompt_cache_hit_tokens": 1024, "prompt_cache_miss_tokens": 203,
            "completion_tokens_details": {"reasoning_tokens": 16384}}


def attempt(number=1, raw=None):
    return {"event": "api_attempt", "attempt": number, "step": 0, "retry": number - 1,
            "returned_model": "deepseek-flash", "elapsed_seconds": 100., "finish_reason": "length",
            "usage": usage() if raw is None else raw}


class UnlimitedRequestTests(unittest.TestCase):
    def test_high_profile_omits_limit_in_actual_request(self):
        cfg = Config.load("configs/v2_base_high.json")
        body = request_payload(cfg, "task", observation(), [], 0)
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertNotIn("max_tokens", body)
        self.assertEqual(cfg.timeout_seconds, 1800)
        self.assertIsNone(Config.from_dict(cfg.to_dict()).max_tokens)
        for bad in (0, -1, True, "none", 393217):
            with self.subTest(bad=bad), self.assertRaises(ExperimentError):
                Config.from_dict({"max_tokens": bad})

    def test_replay_default_and_explicit_omit_verify_original_before_removing_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            original = test_v2.ReplayTests().legacy_episode(root)
            cfg, body, provenance = prepare_replay(root, 1, None, "high", 1800)
            original.pop("max_tokens")
            self.assertEqual(body, original)
            self.assertIsNone(cfg.max_tokens)
            self.assertEqual(cfg.timeout_seconds, 1800)
            self.assertTrue(provenance["verified_exact_original_request"])
            args = parser().parse_args(["replay-request", str(root), "--step", "1", "--out", "out"])
            self.assertIsNone(args.max_tokens)
            with patch("ds_libero.replay.Client", side_effect=AssertionError("No API")):
                report = replay_request(root, 1, None, Path(temp) / "prepared", "high", True, 1800)
            saved = json.loads((Path(temp) / "prepared/requests/step_00001.json").read_text())
            self.assertNotIn("max_tokens", saved)
            self.assertEqual(report["api_cost"]["total_cost_range_cny"], [0., 0.])
            self.assertEqual(report["http_requests"], 0)
            self.assertIn("command_wall_seconds", report)


class AccountingTests(unittest.TestCase):
    def test_truncated_paid_output_counts_reasoning_once(self):
        ledger, accounting = account_attempts([attempt()], [], pricing_snapshot())
        # The user's actual 16K truncation: output is paid even though no action is usable.
        expected = (1024 * .02 + 203 + 16384 * 4) / 1000000
        self.assertAlmostEqual(accounting["api_cost"]["total_cost_range_cny"][0], expected)
        self.assertAlmostEqual(accounting["api_cost"]["total_cost_range_cny"][1], expected * 2)
        self.assertEqual(accounting["reported_usage"]["total_tokens"], 17611)
        self.assertEqual(ledger[0]["finish_reason"], "length")

    def test_missing_cache_usage_is_not_free_and_inconsistent_cache_is_unpriced(self):
        for bad in ({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                    {**usage(), "prompt_cache_hit_tokens": 9999},
                    {**usage(), "total_tokens": 0}, {"completion_tokens": True}, {}):
            with self.subTest(bad=bad):
                _, totals = account_attempts([attempt(raw=bad)], [], pricing_snapshot())
                self.assertIsNone(totals["api_cost"]["total_cost_range_cny"])
                self.assertEqual(totals["api_cost"]["unknown_cost_attempts"], 1)

    def test_retries_timeout_and_hard_interrupt_leave_known_subtotal_and_unknown_total(self):
        events = [attempt(), {**attempt(2), "usage": None, "error": "timeout"}]
        started = [{"event": "api_started", "attempt": 3}]
        rows, totals = account_attempts(events, started, pricing_snapshot())
        self.assertEqual(totals["api_cost"]["unknown_cost_attempts"], 2)
        self.assertEqual(totals["api_cost"]["priced_attempts"], 1)
        self.assertIsNone(totals["api_cost"]["total_cost_range_cny"])
        self.assertGreater(totals["api_cost"]["known_cost_range_cny"][0], 0)
        self.assertIsNone(rows[-1]["completion_tokens"])
        self.assertFalse(rows[-1]["completed"])
        self.assertTrue(totals["usage_incomplete"])
        self.assertEqual(totals["usage_field_available_attempts"]["reasoning_tokens"], 1)

    def test_legacy_without_price_snapshot_and_unknown_model_remain_unpriced(self):
        for event, pricing in ((attempt(), None), ({**attempt(), "returned_model": "other"}, pricing_snapshot())):
            _, totals = account_attempts([event], [], pricing)
            self.assertIsNone(totals["api_cost"]["total_cost_range_cny"])

    def test_report_exports_attempt_ledger_with_unknown_cells_blank_and_frozen_prices(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / "run", "run", Config())
            store.event(attempt())
            store.event({"event": "api_started", "attempt": 2})
            with patch("ds_libero.accounting.PRICE_SNAPSHOT", {}):
                report = build_report(store.path, replay=False)
            self.assertEqual(report["api_cost"]["priced_attempts"], 1)
            self.assertTrue(report["http_latency_incomplete"])
            self.assertIsNone(report["task_success"])
            with (store.path / "api_calls.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["completion_tokens"], "")
            self.assertEqual(rows[0]["task_id"], "0")

    def test_full_task_latency_includes_api_and_retry_backoff_not_simulated_time(self):
        class Clock:
            value = 0.
            def now(self):
                return self.value
            def advance(self, seconds):
                self.value += seconds
        clock = Clock()
        class Env(FakeEnv):
            def reset(self):
                clock.advance(5.)
                return super().reset()
            def step(self, action):
                clock.advance(.25)
                return super().step(action)
        with tempfile.TemporaryDirectory() as temp:
            cfg = Config.load("configs/v2_base_high.json")
            store = Store(Path(temp) / "run", "run", cfg)
            replies = [HTTPResponse(503, b'{}'), HTTPResponse(200, json.dumps({"model": "deepseek-flash",
                "usage": usage(), "choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps({"actions": [ACTION]})}}]}).encode())]
            def send(endpoint, payload, key, timeout):
                self.assertEqual(timeout, 1800)
                self.assertNotIn("max_tokens", json.loads(payload))
                clock.advance(10.)
                return replies.pop(0)
            with patch("time.perf_counter", side_effect=clock.now), contextlib.redirect_stdout(io.StringIO()):
                policy = ModelPolicy(cfg, "task", Client(cfg, store, api_key="dummy", send=send,
                                                       sleep=clock.advance), store)
                result = run_quiet(Env(success_at=1), policy, store, cfg)
            store.finish(result)
            summary = build_report(store.path, replay=False)
            self.assertEqual(summary["task_execution_wall_seconds"], 22.25)
            self.assertEqual(summary["control_loop_wall_seconds"], 27.25)
            self.assertEqual(summary["reset_and_settle_seconds"], 5.)
            self.assertEqual(summary["http_latency_seconds"]["total"], 20.)
            self.assertEqual(summary["decision_latency_seconds"]["total"], 22.)
            self.assertEqual(summary["retry_backoff_seconds"], 2.)
            self.assertEqual(summary["simulated_control_seconds"], .05)
            self.assertTrue(summary["task_success"])

    def test_preflight_failure_cost_is_in_campaign_and_per_task_not_episode_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "plan"
            cfg = Config.load("configs/direct_v2_high.json")
            plan = create_plan(cfg, "libero_spatial", [0], [0, 1], 0, root)
            for entry in plan["episodes"]:
                folder = root / entry["run_directory"]
                store = Store(folder / "api-check", "api-check", Config.from_dict(entry["config"]))
                store.event(attempt())
                store.finish({"status": "invalid_output", "scored": False, "task_success": None,
                              "command_wall_seconds": 105.})
            report = report_plan(root)
            self.assertEqual(report["scored"], 0)
            self.assertEqual(report["http_requests"], 0)
            self.assertEqual(report["campaign_http_requests"], 2)
            self.assertEqual(report["per_task"]["0"]["campaign_reported_usage"]["total_tokens"], 35222)
            self.assertEqual(report["per_task"]["0"]["campaign_api_wait_seconds"], 200.)
            self.assertGreater(report["campaign_api_cost"]["total_cost_range_cny"][0], 0.)

    def test_campaign_allows_failed_setup_without_task_name_but_rejects_mixed_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calibration = Store(root / "calibration", "calibrate", Config())
            calibration.environment({"task_name": "bowl"})
            check = Store(root / "api-check", "api-check", Config())
            check.finish({"status": "runtime_error", "scored": False, "task_success": None})
            report = build_campaign_report(root)
            self.assertTrue(report["command_wall_incomplete"])
            check.manifest["config"]["max_tokens"] = None
            write_json(check.path / "manifest.json", check.manifest)
            with self.assertRaises(ExperimentError):
                build_campaign_report(root)


if __name__ == "__main__":
    unittest.main()
