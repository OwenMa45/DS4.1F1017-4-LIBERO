import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ds_libero.artifacts import Store
from ds_libero.client import Client, ModelPolicy
from ds_libero.config import Config
from ds_libero.errors import ExperimentError, InvalidAction
from ds_libero.protocol import parse_actions, request_payload
from ds_libero.replay import prepare_replay, replay_request
from ds_libero.report import build_report
from test_experiment import FakeEnv, FixedPolicy, MemoryStore, observation, reply, run_quiet


def v2(**kwargs):
    return replace(Config(), protocol_version="ds-libero-direct-v2", action_horizon=8, **kwargs)


ACTION = [0., 0., 0., 0., 0., 0., -1.]


class ChunkTests(unittest.TestCase):
    def test_bad_later_action_prevents_entire_sequence(self):
        env = FakeEnv()
        result = run_quiet(env, FixedPolicy([ACTION, [2.] + [0.] * 6]), MemoryStore(), v2())
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(env.steps, 0)

    def test_success_inside_chunk_discards_remainder(self):
        env = FakeEnv(success_at=3)
        result = run_quiet(env, FixedPolicy([ACTION] * 8), MemoryStore(), v2())
        self.assertEqual((result["status"], env.steps, result["policy_decisions"]), ("success", 3, 1))

    def test_environment_done_inside_chunk_is_not_success(self):
        env = FakeEnv(done_at=2)
        result = run_quiet(env, FixedPolicy([ACTION] * 8), MemoryStore(), v2())
        self.assertEqual(result["status"], "simulation_error")
        self.assertEqual(env.steps, 2)
        self.assertIsNone(result["task_success"])

    def test_step_limit_uses_executed_actions_not_api_calls(self):
        class Policy:
            def act(self, obs, history, step, refs):
                return [ACTION] * min(8, 10 - step)
        env = FakeEnv()
        result = run_quiet(env, Policy(), MemoryStore(), v2(max_steps=10))
        self.assertEqual((env.steps, result["policy_decisions"], result["status"]), (10, 2, "failure"))

    def test_oversized_final_chunk_is_rejected_without_partial_execution(self):
        env = FakeEnv()
        result = run_quiet(env, FixedPolicy([ACTION] * 8), MemoryStore(), v2(max_steps=10))
        self.assertEqual((env.steps, result["status"]), (8, "invalid_output"))

    def test_strict_chunk_shape(self):
        for value in ('{"actions":[]}', '{"actions":null}', '{"action":[0,0,0,0,0,0,-1]}',
                      '{"actions":[[0,0,0,0,0,0,-1]],"plan":"go"}',
                      '{"actions":[],"actions":[[0,0,0,0,0,0,-1]]}'):
            with self.subTest(value=value), self.assertRaises(InvalidAction):
                parse_actions(value, "stop", 8)
        with self.assertRaises(InvalidAction):
            parse_actions(json.dumps({"actions": [ACTION]}), "length", 8)

    def test_visual_history_is_bounded_and_has_matching_state_and_refs(self):
        cfg = v2(visual_history=2)
        store, requests = MemoryStore(), []
        def send(endpoint, body, key, timeout):
            requests.append(json.loads(body))
            return reply(json.dumps({"actions": [ACTION]}))
        policy = ModelPolicy(cfg, "task", Client(cfg, store, api_key="dummy", send=send), store)
        for step in range(4):
            obs = observation(step)
            policy.act(obs, [], step, store.observe(step, obs))
        blocks = requests[-1]["messages"][1]["content"]
        self.assertEqual(sum(b["type"] == "image_url" for b in blocks), 6)
        past = [json.loads(b["text"]) for b in blocks if b["type"] == "text" and b["text"].startswith('{"historical')]
        self.assertEqual([p["historical_control_step"] for p in past], [1, 2])
        self.assertEqual(past[0]["proprioception"], observation(1).state)
        decisions = [e for e in store.events if e["event"] == "decision"]
        self.assertEqual([e["step"] for e in decisions[-1]["visual_history_frames"]], [1, 2])

    def test_geometry_missing_fails_before_api(self):
        with self.assertRaises(ExperimentError):
            request_payload(v2(camera_geometry=True), "task", observation(), [], 0)

    def test_model_policy_chunks_integrate_with_reporting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            cfg = v2()
            store = Store(root, "run", cfg)
            response = reply(json.dumps({"actions": [ACTION] * 8}))
            policy = ModelPolicy(cfg, "task", Client(cfg, store, api_key="dummy", send=lambda *a: response), store)
            result = run_quiet(FakeEnv(success_at=3), policy, store, cfg)
            store.finish(result)
            report = build_report(root, replay=False)
            self.assertEqual(report["http_requests"], 1)
            self.assertEqual(report["executed_action_count"], 3)
            self.assertEqual(report["actions_per_valid_decision"]["mean"], 8)
            self.assertEqual(report["finish_reasons"], {"stop": 1})
            self.assertEqual(report["task_success"], True)

    def test_v2_retry_uses_identical_payload_without_advancing_simulation(self):
        from ds_libero.client import HTTPResponse
        env, store, requests = FakeEnv(success_at=2), MemoryStore(), []
        responses = [HTTPResponse(503, b'{}'), reply(json.dumps({"actions": [ACTION] * 3}))]
        def send(endpoint, body, key, timeout):
            self.assertEqual(env.steps, 0)
            requests.append(body)
            return responses.pop(0)
        cfg = v2()
        policy = ModelPolicy(cfg, "task", Client(cfg, store, api_key="dummy", send=send, sleep=lambda _: None), store)
        result = run_quiet(env, policy, store, cfg)
        self.assertEqual(result["status"], "success")
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(env.steps, 2)

    def test_v1_fingerprint_is_unchanged(self):
        self.assertEqual(Config().fingerprint, "911c4f579ce0bd922aee8e2472ddb658b58fdf42f3fd76cd00be3887017b37ff")

    def test_all_supplied_configs_load(self):
        for path in Path("configs").glob("*.json"):
            with self.subTest(path=path):
                Config.load(path)


class ReplayTests(unittest.TestCase):
    def legacy_episode(self, root):
        cfg = Config()
        store = Store(root, "run", cfg)
        store.environment({"task": "put bowl on plate"})
        obs = observation()
        refs = store.observe(1, obs)
        payload = request_payload(cfg, "put bowl on plate", obs, [], 1)
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        store.event({"event": "api_attempt", "step": 1,
                     "request_sha256": hashlib.sha256(encoded).hexdigest()})
        store.event({"event": "decision", "step": 1, "frames": refs,
                     "proprioception": obs.state, "recent_history": []})
        return payload

    def test_legacy_reconstruction_changes_only_token_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "original"
            expected = self.legacy_episode(root)
            cfg, payload, info = prepare_replay(root, 1, 16384)
            expected["max_tokens"] = 16384
            self.assertEqual(payload, expected)
            self.assertTrue(info["verified_exact_original_request"])
            self.assertEqual(cfg.max_tokens, 16384)

    def test_corrupt_frame_or_request_never_calls_api(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "original"
            self.legacy_episode(root)
            next((root / "frames").glob("*.png")).write_bytes(b"modified")
            with self.assertRaises(ExperimentError):
                prepare_replay(root, 1, 16384)

    def test_dry_run_does_not_construct_client_or_simulator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "original"
            self.legacy_episode(root)
            with patch("ds_libero.replay.Client", side_effect=AssertionError("API forbidden")):
                result = replay_request(root, 1, 16384, Path(temp) / "diagnostic", dry_run=True)
            self.assertEqual(result["status"], "prepared")
            self.assertEqual(result["model_actions_executed"], 0)
            self.assertFalse(result["scored"])

    def test_saved_payload_replay_keeps_historical_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "original"
            cfg = v2(visual_history=2)
            store = Store(root, "run", cfg)
            payload = request_payload(cfg, "task", observation(2), [], 2, [(1, observation(1))])
            Client(cfg, store, api_key="dummy", send=lambda *a: reply()).complete(payload, 2)
            _, prepared, _ = prepare_replay(root, 2, 16384, "low")
            self.assertEqual(prepared["messages"], payload["messages"])
            self.assertEqual(prepared["reasoning_effort"], "low")


class GeometryTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy
        except ImportError:
            self.skipTest("NumPy required")

    def test_projection_matches_upright_camera_convention(self):
        from ds_libero.geometry import camera_calibration, project_world
        cal = camera_calibration([0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 90, 256)
        self.assertEqual(project_world([0, 0, -1], cal), [128., 128.])
        self.assertAlmostEqual(project_world([.5, 0, -1], cal)[0], 192.)
        self.assertAlmostEqual(project_world([0, .5, -1], cal)[1], 64.)
        self.assertIsNone(project_world([0, 0, 1], cal))

    def test_camera_translation_changes_projection(self):
        from ds_libero.geometry import camera_calibration, project_world
        cal = camera_calibration([.5, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 90, 256)
        self.assertAlmostEqual(project_world([0, 0, -1], cal)[0], 64.)


if __name__ == "__main__":
    unittest.main()
