import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ds_libero.__main__ import main
from ds_libero.accounting import account_attempts, aggregate_cost, pricing_snapshot
from ds_libero.artifacts import Store
from ds_libero.benchmark import create_five_plan, load_plan, report_plan, run_plan_all
from ds_libero.client import Client, HTTPResponse, ModelPolicy, resolve_api_key, response_content
from ds_libero.config import Config
from ds_libero.environment import verify_task_name
from ds_libero.errors import ExperimentError
from ds_libero.five_tasks import FIVE_TASKS, five_configs
from ds_libero.forecast import forecast_gpt
from ds_libero.protocol import request_payload
from ds_libero.replay import prepare_replay, replay_request
from ds_libero.report import build_report
from test_experiment import FakeEnv, MemoryStore, observation, run_quiet


ACTION = [0., 0., 0., 0., 0., 0., -1.]
JSON_INPUT_ERROR = {"message": "Response input messages must contain the word 'json' in some form "
                              "to use 'text.format' of type 'json_object'.",
                    "type": "invalid_request_error", "param": "input", "code": None}


def config(**changes):
    return Config.from_dict({**Config.load("configs/gpt61_sol_medium_220.json").to_dict(), **changes})


def observed(step=0):
    return replace(observation(step), geometry={"control_frame": {"schema": "panda-grip-site-v1"}})


class Env(FakeEnv):
    def reset(self):
        super().reset()
        return observed()

    def step(self, action):
        _, done = super().step(action)
        return observed(self.steps), done


def response(actions=None):
    return {"id": "resp_test", "model": "gpt-6.1-sol", "status": "completed", "created_at": 1234,
            "service_tier": "default", "output": [{"type": "reasoning", "summary": []},
                {"type": "message", "role": "assistant", "status": "completed", "content": [
                    {"type": "output_text", "text": json.dumps({"actions": actions or [ACTION]})}]}],
            "usage": {"input_tokens": 10000, "output_tokens": 4000, "total_tokens": 14000,
                      "input_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 2000,
                          "cache_write_tokens_details": {"ttl_30m_tokens": 2000, "ttl_12h_tokens": 0}},
                      "output_tokens_details": {"reasoning_tokens": 3900}}}


def http(data):
    return HTTPResponse(200, json.dumps(data).encode())


def event(data=None):
    data = response() if data is None else data
    return {"event": "api_attempt", "attempt": 1, "step": 0, "returned_model": data["model"],
            "service_tier": data["service_tier"], "usage": data["usage"], "elapsed_seconds": 30.}


class OpenAIProtocolTests(unittest.TestCase):
    def test_missing_key_fails_before_output_directory_or_simulator_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "must-not-exist"
            with patch("ds_libero.__main__.resolve_api_key", side_effect=ExperimentError("missing OpenAI key")), \
                    patch("ds_libero.__main__.Store", side_effect=AssertionError("No artifacts")), \
                    patch("ds_libero.environment.LiberoEnvironment", side_effect=AssertionError("No simulator")), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = main(["run", "--config", "configs/gpt61_sol_medium_220.json", "--out", str(out),
                             "--calibration", "missing", "--api-check", "missing"])
            self.assertEqual(code, 2)
            self.assertFalse(out.exists())

    def test_credential_provider_isolation_and_file_location(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-dummy", "DEEPSEEK_API_KEY": "ds-dummy"}):
            self.assertEqual(resolve_api_key(provider="openai"), "openai-dummy")
            self.assertEqual(resolve_api_key(), "ds-dummy")
        paths = []
        def read(path, **kwargs):
            paths.append(path)
            return " file-dummy\n"
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "DEEPSEEK_API_KEY": "must-not-use"}), \
                patch.object(Path, "read_text", read):
            self.assertEqual(resolve_api_key(provider="openai"), "file-dummy")
        self.assertEqual(paths, [Path(__file__).resolve().parents[1] / "gpt_API.txt"])
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "DEEPSEEK_API_KEY": "must-not-use"}), \
                patch.object(Path, "read_text", return_value=""):
            with self.assertRaisesRegex(ExperimentError, "OPENAI_API_KEY"):
                resolve_api_key(provider="openai")

    def test_profiles_pin_model_endpoint_limits_and_five_task_identities(self):
        cfg = config()
        trials = five_configs(cfg)
        self.assertEqual([(c.suite, c.task_id, c.expected_task_name) for c in trials], list(FIVE_TASKS))
        self.assertTrue(all(c.max_steps == 220 and c.init_state_id == c.seed == 0 for c in trials))
        for changes in ({"max_steps": 221}, {"max_http_requests": 221}, {"max_tokens": None},
                        {"endpoint": "https://example.com/v1/responses"}, {"model": "gpt-other"},
                        {"api_provider": "deepseek"}, {"max_retries": -1}):
            with self.subTest(changes=changes), self.assertRaises(ExperimentError):
                config(**changes)
        with self.assertRaises(ExperimentError):
            verify_task_name(trials[0], "wrong-task")

    def test_responses_payload_preserves_observations_prompt_images_and_remaining_steps(self):
        cfg = config()
        ds = replace(cfg, api_provider="deepseek", model="deepseek-flash")
        gpt_payload = request_payload(cfg, "same task", observed(219), [], 219,
                                      [(1, observed(1)), (2, observed(2))])
        ds_payload = request_payload(ds, "same task", observed(219), [], 219,
                                     [(1, observed(1)), (2, observed(2))])
        self.assertEqual(gpt_payload["instructions"], ds_payload["messages"][0]["content"])
        self.assertEqual(gpt_payload["reasoning"], {"effort": "medium"})
        self.assertEqual(gpt_payload["max_output_tokens"], 16384)
        self.assertFalse(gpt_payload["store"])
        self.assertEqual(gpt_payload["service_tier"], "default")
        self.assertEqual(gpt_payload["prompt_cache_options"], {"mode": "implicit", "ttl": "30m"})
        self.assertEqual(gpt_payload["text"], {"format": {"type": "json_object"}})
        for key in ("messages", "thinking", "max_tokens", "response_format", "temperature", "previous_response_id"):
            self.assertNotIn(key, gpt_payload)
        blocks = gpt_payload["input"][0]["content"]
        original = ds_payload["messages"][1]["content"]
        self.assertEqual(len(blocks), len(original) + 1)
        for block, previous in zip(blocks[:-1], original):
            self.assertEqual(block.get("text", block.get("image_url")),
                             previous.get("text", previous.get("image_url", {}).get("url")))
        self.assertEqual(sum(b["type"] == "input_image" for b in blocks), 6)
        self.assertEqual(json.loads(blocks[0]["text"])["max_actions_this_response"], 1)

    def test_json_mode_input_contract_accepts_all_five_tasks(self):
        # Reproduce the server's check of input messages, excluding top-level instructions.
        def send(endpoint, body, key, timeout):
            payload = json.loads(body)
            input_text = " ".join(block["text"] for message in payload["input"]
                                  for block in message["content"] if block["type"] == "input_text")
            if payload["text"]["format"]["type"] == "json_object" and "json" not in input_text.lower():
                return HTTPResponse(400, json.dumps({"error": JSON_INPUT_ERROR}).encode())
            return http(response())
        for cfg in five_configs(config()):
            with self.subTest(task=cfg.expected_task_name):
                env, store = Env(success_at=1), MemoryStore()
                client = Client(cfg, store, api_key="dummy", send=send)
                result = run_quiet(env, ModelPolicy(cfg, cfg.expected_task_name, client, store), store, cfg)
                self.assertEqual((result["status"], env.steps, client.attempts), ("success", 1, 1))

    def test_server_error_details_reach_summary_redacted_without_retry_or_action(self):
        key = "sensitive-test-key"
        for diagnostic in (JSON_INPUT_ERROR, {**JSON_INPUT_ERROR, "message": "Rejected key: " + key}):
            with self.subTest(message=diagnostic["message"]), tempfile.TemporaryDirectory() as temp:
                env, cfg = Env(), config(max_retries=2)
                store = Store(Path(temp) / "run", "run", cfg)
                client = Client(cfg, store, api_key=key,
                                send=lambda *a: HTTPResponse(400, json.dumps({"error": diagnostic}).encode()))
                result = run_quiet(env, ModelPolicy(cfg, "task", client, store), store, cfg)
                store.finish(result)
                summary = build_report(store.path, replay=False)
                self.assertEqual((summary["status"], env.steps, client.attempts), ("api_error", 0, 1))
                self.assertFalse(summary["scored"])
                self.assertIn(diagnostic["message"].replace(key, "[REDACTED]"), summary["reason"])
                self.assertIn('"param": "input"', summary["reason"])
                events = (store.path / "events.jsonl").read_text(encoding="utf-8")
                attempt = next(e for e in map(json.loads, events.splitlines()) if e["event"] == "api_attempt")
                self.assertEqual(attempt["api_error"]["type"], "invalid_request_error")
                self.assertIsNone(attempt["api_error"]["code"])
                self.assertNotIn(key, events + json.dumps(summary))

    def test_nonstandard_error_bodies_keep_http_failure_diagnostic(self):
        bodies = (b"", b"<html>Bad request</html>", b"null", b'{"error":"Bad request"}',
                  b'{"error":{"message":[],"type":123,"param":{},"code":false}}')
        for body in bodies:
            with self.subTest(body=body):
                env, store, cfg = Env(), MemoryStore(), config()
                client = Client(cfg, store, api_key="dummy", send=lambda *a: HTTPResponse(400, body))
                result = run_quiet(env, ModelPolicy(cfg, "task", client, store), store, cfg)
                self.assertEqual((result["status"], env.steps, client.attempts), ("api_error", 0, 1))
                self.assertIn("HTTP 400", result["reason"])

    def test_completed_output_after_reasoning_parses_and_failures_never_execute(self):
        self.assertEqual(json.loads(response_content(config(), response())[0]), {"actions": [ACTION]})
        cases = []
        truncated = response()
        truncated.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        cases.append((truncated, "invalid_output"))
        refused = response()
        refused["output"][1]["content"] = [{"type": "refusal", "refusal": "no"}]
        cases.append((refused, "invalid_output"))
        bad = response([ACTION, [2, 0, 0, 0, 0, 0, -1]])
        cases.append((bad, "invalid_output"))
        failed = response()
        failed["status"] = "failed"
        cases.append((failed, "api_error"))
        for data, status in cases:
            with self.subTest(status=status):
                env, store, cfg = Env(), MemoryStore(), config()
                client = Client(cfg, store, api_key="dummy", send=lambda *a: http(data))
                result = run_quiet(env, ModelPolicy(cfg, "task", client, store), store, cfg)
                self.assertEqual((env.steps, client.attempts, result["status"]), (0, 1, status))

    def test_real_control_limit_is_220_for_single_actions_and_chunks(self):
        for chunk in (1, 8):
            env, store, cfg = Env(), MemoryStore(), config()
            def send(endpoint, body, key, timeout):
                payload = json.loads(body)
                limit = json.loads(payload["input"][0]["content"][0]["text"])["max_actions_this_response"]
                return http(response([ACTION] * min(chunk, limit)))
            client = Client(cfg, store, api_key="dummy", send=send)
            result = run_quiet(env, ModelPolicy(cfg, "task", client, store), store, cfg)
            self.assertEqual((env.steps, result["status"]), (220, "failure"))
            self.assertEqual(client.attempts, 220 if chunk == 1 else 28)

    def test_default_does_not_retry_paid_timeout_and_budget_includes_enabled_retries(self):
        for cfg, status, count in ((config(), 429, 1), (config(), 503, 1),
                                    (config(max_http_requests=1, max_retries=2), 503, 1)):
            env, store = Env(), MemoryStore()
            client = Client(cfg, store, api_key="dummy", send=lambda *a: HTTPResponse(status, b'{}'), sleep=lambda _: None)
            result = run_quiet(env, ModelPolicy(cfg, "task", client, store), store, cfg)
            self.assertEqual((client.attempts, env.steps), (count, 0))
            self.assertFalse(result["scored"])
        env, store, cfg = Env(), MemoryStore(), config()
        def timeout(*args):
            raise TimeoutError("timed out")
        client = Client(cfg, store, api_key="dummy", send=timeout)
        result = run_quiet(env, ModelPolicy(cfg, "task", client, store), store, cfg)
        self.assertEqual((client.attempts, env.steps, result["status"]), (1, 0, "api_error"))


class OpenAIAccountingTests(unittest.TestCase):
    def test_long_context_price_multiplier_is_not_silently_omitted(self):
        data = response()
        data["usage"].update(input_tokens=300000, total_tokens=304000)
        _, totals = account_attempts([event(data)], [], pricing_snapshot("gpt-6.1-sol"))
        expected = (2 * (1000 * .1 + 2000 * 2.5 + 297000 * 2) + 4000 * 10 * 1.5) / 1e6
        self.assertAlmostEqual(totals["api_cost"]["total_cost_range_usd"][0], expected)

    def test_standard_cost_includes_cache_writes_and_reasoning_exactly_once(self):
        rows, totals = account_attempts([event()], [], pricing_snapshot("gpt-6.1-sol"))
        cost = totals["api_cost"]
        self.assertEqual(cost["currency"], "USD")
        self.assertNotIn("total_cost_range_cny", cost)
        expected = (1000 * .1 + 2000 * 2.5 + 7000 * 2 + 4000 * 10) / 1e6
        self.assertAlmostEqual(cost["total_cost_range_usd"][0], expected)
        self.assertEqual(rows[0]["reasoning_tokens"], 3900)
        long = response()
        long["usage"]["input_tokens_details"]["cache_write_tokens_details"] = {"ttl_30m_tokens": 1000, "ttl_12h_tokens": 1000}
        _, totals = account_attempts([event(long)], [], pricing_snapshot("gpt-6.1-sol"))
        self.assertAlmostEqual(totals["api_cost"]["total_cost_range"][0], expected + .0025)

    def test_unknown_usage_and_wrong_service_tier_never_look_free(self):
        samples = [event(), {**event(), "service_tier": "priority"}, {**event(), "returned_model": "different"}]
        samples[0]["usage"]["input_tokens_details"].pop("cache_write_tokens")
        for sample in samples:
            _, totals = account_attempts([sample], [{"attempt": 2}], pricing_snapshot("gpt-6.1-sol"))
            self.assertIsNone(totals["api_cost"]["total_cost_range"])
            self.assertEqual(totals["api_cost"]["unknown_cost_attempts"], 2)
        with self.assertRaises(ExperimentError):
            aggregate_cost([aggregate_cost([], currency="CNY"), aggregate_cost([], currency="USD")])

    def test_report_and_replay_use_responses_format_and_keep_finite_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = config()
            store = Store(Path(temp) / "source", "run", cfg)
            client = Client(cfg, store, api_key="dummy", send=lambda *a: http(response()))
            result = run_quiet(Env(success_at=1), ModelPolicy(cfg, "task", client, store), store, cfg)
            store.finish(result)
            report = build_report(store.path, replay=False)
            self.assertEqual(report["api_cost"]["currency"], "USD")
            self.assertEqual(report["api_cost"]["priced_attempts"], 1)
            self.assertEqual(report["http_latency_seconds"]["count"], 1)
            replay_cfg, body, provenance = prepare_replay(store.path, 0)
            self.assertEqual(body["max_output_tokens"], 16384)
            self.assertEqual(replay_cfg.api_provider, "openai")
            self.assertTrue(provenance["verified_exact_original_request"])
            with patch("ds_libero.replay.Client", side_effect=AssertionError("No network")):
                replayed = replay_request(store.path, 0, None, Path(temp) / "replay", dry_run=True)
            self.assertEqual(replayed["http_requests"], 0)
            self.assertEqual(replayed["api_cost"]["total_cost_range_usd"], [0., 0.])

    def test_five_plan_is_offline_and_campaign_costs_use_usd_even_before_episode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "plan"
            with patch("ds_libero.client.transport", side_effect=AssertionError("No network")):
                plan = create_five_plan(config(), root)
                self.assertEqual(load_plan(root)[1], plan)
                report = report_plan(root)
            self.assertEqual(report["planned"], 5)
            self.assertEqual(len(report["per_task"]), 5)
            self.assertEqual(report["campaign_api_cost"]["currency"], "USD")
            self.assertEqual(plan["max_http_requests_per_episode_including_api_check"], 221)
            entry = plan["episodes"][0]
            store = Store(root / entry["run_directory"] / "api-check", "api-check", Config.from_dict(entry["config"]))
            store.event(event())
            store.finish({"status": "invalid_output", "scored": False, "task_success": None})
            report = report_plan(root)
            self.assertEqual(report["campaign_http_requests"], 1)
            self.assertGreater(report["campaign_api_cost"]["known_cost_range_usd"][0], 0)
            self.assertEqual(report["scored"], 0)
            with patch("ds_libero.benchmark.run_plan_episode", return_value=2) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_plan_all(root), 2)
            self.assertEqual(run.call_args.args[1], 1)  # Never re-spend the failed preflight.

    def test_forecast_is_labeled_offline_and_matches_request_ceiling(self):
        with patch("ds_libero.client.Client", side_effect=AssertionError("No API")):
            report = forecast_gpt(config())
        self.assertEqual(report["maximum_total_http_attempts_including_preflight"], 1105)
        self.assertEqual(report["projected_total_http_attempts_including_preflight"], 225)
        self.assertEqual(report["scenarios"][1]["assumed_output_tokens_per_call_including_reasoning"], 4000)
        self.assertIn("Offline", report["notes"][0])


if __name__ == "__main__":
    unittest.main()
