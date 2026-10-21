import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from ds_libero.artifacts import Store, verify_gate
from ds_libero.calibration import quat_product, rotation_delta
from ds_libero.client import Client, HTTPResponse, ModelPolicy
from ds_libero.config import Config
from ds_libero.errors import APIError, BudgetExhausted, ExperimentError, InvalidAction
from ds_libero.protocol import CAMERAS, Observation, parse_action, request_payload, validate_action
from ds_libero.report import build_report
from ds_libero.runner import run_episode

try:
    import numpy as np
    from PIL import Image
    HAS_IMAGES = True
except ImportError:
    HAS_IMAGES = False


def observation(step=0, images=None):
    return Observation({"eef_position_world_m": [step * .001, 0., 1.],
                        "eef_quaternion_xyzw": [0., 0., 0., 1.],
                        "gripper_joint_position_m": [.04, -.04]},
                       images or {key: b"fake PNG" for key in CAMERAS})


class MemoryStore:
    def __init__(self):
        self.events, self.bodies = [], []

    def event(self, event):
        self.events.append(copy.deepcopy(event))

    def response(self, attempt, body):
        self.bodies.append(body)
        return f"responses/{attempt}.txt"

    def observe(self, index, obs):
        return {key: {"path": f"{index}_{key}.png"} for key in CAMERAS}


class FakeEnv:
    def __init__(self, success_at=None, done_at=None, initial_success=False, raise_step=False):
        self.success_at, self.done_at = success_at, done_at
        self.initial_success, self.raise_step = initial_success, raise_step
        self.steps, self.actions = 0, []

    def reset(self):
        self.steps = 0
        self.actions = []
        return observation()

    def step(self, action):
        self.steps += 1
        self.actions.append(list(action))
        if self.raise_step:
            raise RuntimeError("simulator stopped after accepting an action")
        return observation(self.steps), self.steps == self.done_at

    def check_success(self):
        return self.initial_success or self.steps == self.success_at


class FixedPolicy:
    def __init__(self, action=None):
        self.action = [0., 0., 0., 0., 0., 0., -1.] if action is None else action
        self.histories = []

    def act(self, obs, history, step, refs):
        self.histories.append(copy.deepcopy(history))
        return self.action


def run_quiet(*args):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_episode(*args)


def reply(content='{"action":[0,0,0,0,0,0,-1]}', finish="stop"):
    return HTTPResponse(200, json.dumps({"model": "deepseek-flash", "id": "test-response",
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
                  "completion_tokens_details": {"reasoning_tokens": 15}}}).encode())


class ProtocolTests(unittest.TestCase):
    def test_strict_actions(self):
        good = '{"action":[-1,1,0.2,0,0,0,-1]}'
        self.assertEqual(parse_action(good), [-1., 1., .2, 0., 0., 0., -1.])
        bad = ['', 'null', '[]', '{"action":[0,0,0,0,0,0]}',
               '{"action":[true,0,0,0,0,0,-1]}', '{"action":["0",0,0,0,0,0,-1]}',
               '{"action":[NaN,0,0,0,0,0,-1]}', '{"action":[1e999,0,0,0,0,0,-1]}',
               '{"action":[1.01,0,0,0,0,0,-1]}', '{"action":[0,0,0,0,0,0,0.5]}',
               '{"action":[0,0,0,0,0,0,-1],"note":"go"}',
               '{"action":null,"action":[0,0,0,0,0,0,-1]}',
               '```json\n' + good + '\n```']
        for value in bad:
            with self.subTest(value=value), self.assertRaises(InvalidAction):
                parse_action(value)
        with self.assertRaises(InvalidAction):
            parse_action(good, "length")
        with self.assertRaises(InvalidAction):
            validate_action([10 ** 10000] + [0] * 6)

    def test_request_has_exact_visual_inputs_and_high_reasoning(self):
        payload = request_payload(Config(), "put bowl on plate", observation(), [], 0)
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertNotIn("temperature", payload)
        images = [c for c in payload["messages"][1]["content"] if c["type"] == "image_url"]
        self.assertEqual(len(images), 2)
        data = json.loads(payload["messages"][1]["content"][0]["text"])
        self.assertEqual(set(data["proprioception"]), set(observation().state))
        self.assertNotIn("objects", data)

    def test_json_config_matches_defaults(self):
        self.assertEqual(Config.load("configs/first_episode.json"), Config())

    def test_v1_cannot_silently_expand_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            for data in ({"max_steps": 1000}, {"max_tokens": True}, {"new_unknown_field": 1}):
                path.write_text(json.dumps(data))
                with self.assertRaises(ExperimentError):
                    Config.load(path)

    def test_world_rotation_delta_is_not_euler_difference(self):
        initial = [math.sin(.4), 0., 0., math.cos(.4)]
        world_z = [0., 0., math.sin(.1), math.cos(.1)]
        final = quat_product(world_z, initial)
        delta = rotation_delta(initial, final)
        for actual, expected in zip(delta, [0., 0., .2]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(rotation_delta(initial, [-x for x in initial]), [0., 0., 0.])


class ControlLoopTests(unittest.TestCase):
    def test_horizon_counts_only_executed_model_actions(self):
        env, store, policy = FakeEnv(), MemoryStore(), FixedPolicy()
        result = run_quiet(env, policy, store, replace(Config(), max_steps=7))
        self.assertEqual(env.steps, 7)
        self.assertEqual(result["status"], "failure")
        self.assertTrue(result["scored"])
        self.assertEqual(len(policy.histories[-1]), 4)
        self.assertEqual(policy.histories[-1][0]["step"], 2)

    def test_success_predicate_ends_episode_even_without_done(self):
        env = FakeEnv(success_at=2)
        result = run_quiet(env, FixedPolicy(), MemoryStore(), Config())
        self.assertEqual(result["status"], "success")
        self.assertEqual(env.steps, 2)

    def test_done_is_not_success(self):
        env = FakeEnv(done_at=1)
        result = run_quiet(env, FixedPolicy(), MemoryStore(), Config())
        self.assertEqual(result["status"], "simulation_error")
        self.assertFalse(result["scored"])
        self.assertIsNone(result["task_success"])

    def test_invalid_action_is_never_executed(self):
        env = FakeEnv()
        result = run_quiet(env, FixedPolicy([2.] + [0.] * 6), MemoryStore(), Config())
        self.assertEqual(env.steps, 0)
        self.assertEqual(result["status"], "invalid_output")
        self.assertTrue(result["scored"])
        self.assertFalse(result["task_success"])

    def test_initial_success_is_not_a_free_success(self):
        env, policy = FakeEnv(initial_success=True), FixedPolicy()
        result = run_quiet(env, policy, MemoryStore(), Config())
        self.assertFalse(result["scored"])
        self.assertEqual(policy.histories, [])

    def test_failed_step_is_marked_as_potentially_executed(self):
        env = FakeEnv(raise_step=True)
        result = run_quiet(env, FixedPolicy(), MemoryStore(), Config())
        self.assertFalse(result["scored"])
        self.assertTrue(result["last_step_may_have_executed"])


class HTTPTests(unittest.TestCase):
    def test_retry_never_advances_simulation_and_reuses_payload(self):
        env, store = FakeEnv(success_at=1), MemoryStore()
        requests, waits = [], []
        responses = [HTTPResponse(429, b'{}'), HTTPResponse(503, b'{}'), reply()]
        def send(endpoint, body, key, timeout):
            self.assertEqual(env.steps, 0)
            requests.append(body)
            return responses.pop(0)
        cfg = Config()
        client = Client(cfg, store, api_key="test-key", send=send, sleep=waits.append)
        result = run_quiet(env, ModelPolicy(cfg, "task", client, store), store, cfg)
        self.assertEqual(result["status"], "success")
        self.assertEqual(env.steps, 1)
        self.assertEqual(len(requests), 3)
        self.assertEqual(len(set(requests)), 1)
        self.assertEqual(waits, [2, 4])

    def test_auth_error_not_retried_and_key_not_logged(self):
        store = MemoryStore()
        client = Client(Config(), store, api_key="sensitive", send=lambda *a: HTTPResponse(401, b'{"error":"sensitive"}'))
        with self.assertRaises(APIError):
            client.complete({}, 0)
        self.assertEqual(client.attempts, 1)
        self.assertNotIn("sensitive", str(store.events) + str(store.bodies))

    def test_transport_failures_stop_after_three_attempts(self):
        store = MemoryStore()
        def fail(*args):
            raise TimeoutError("read timed out")
        client = Client(Config(), store, api_key="test", send=fail, sleep=lambda _: None)
        with self.assertRaises(APIError):
            client.complete({}, 0)
        self.assertEqual(client.attempts, 3)

    def test_http_budget_exhaustion_never_executes_an_action(self):
        env, store = FakeEnv(), MemoryStore()
        cfg = replace(Config(), max_http_requests=2)
        client = Client(cfg, store, api_key="test", send=lambda *a: HTTPResponse(503, b'{}'), sleep=lambda _: None)
        result = run_quiet(env, ModelPolicy(cfg, "task", client, store), store, cfg)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(client.attempts, 2)
        self.assertEqual(env.steps, 0)
        self.assertFalse(result["scored"])

    def test_truncated_response_not_retried_or_executed(self):
        env, store = FakeEnv(), MemoryStore()
        cfg = Config()
        client = Client(cfg, store, api_key="test", send=lambda *a: reply(finish="length"))
        result = run_quiet(env, ModelPolicy(cfg, "task", client, store), store, cfg)
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(client.attempts, 1)
        self.assertEqual(env.steps, 0)


class ArtifactTests(unittest.TestCase):
    def test_gate_checks_code_config_and_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calibrate"
            store = Store(path, "calibrate", Config())
            store.environment({"revision": "same"})
            store.finish({"status": "pass"})
            verify_gate(path, "calibrate", Config(), {"revision": "same"})
            with self.assertRaises(ExperimentError):
                verify_gate(path, "calibrate", Config(), {"revision": "changed"})
            with self.assertRaises(ExperimentError):
                verify_gate(path, "calibrate", replace(Config(), max_tokens=100), {"revision": "same"})
            with self.assertRaises(FileExistsError):
                Store(path, "calibrate", Config())

    def test_incomplete_run_and_partial_line_keep_null_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "episode"
            Store(root, "run", Config())
            (root / "events.jsonl").write_text('{"event":"step"}\n{"event":', encoding="utf-8")
            summary = build_report(root, replay=False)
            self.assertEqual(summary["status"], "incomplete")
            self.assertIsNone(summary["task_success"])
            self.assertTrue(summary["warnings"])

    def test_usage_retry_accounting_and_invalid_output_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "episode"
            store = Store(root, "run", Config())
            store.event({"event": "api_attempt", "elapsed_seconds": 1, "usage": None})
            store.event({"event": "api_attempt", "elapsed_seconds": 3, "usage": {
                "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 15}}})
            store.event({"event": "decision", "elapsed_seconds": 6, "valid": False, "error_status": "invalid_output"})
            store.finish({"status": "invalid_output", "scored": True, "task_success": False})
            summary = build_report(root, replay=False)
            self.assertEqual(summary["http_requests"], 2)
            self.assertEqual(summary["action_validity_rate"], 0)
            self.assertTrue(summary["usage_incomplete"])
            self.assertEqual(summary["reported_usage"]["total_tokens"], 30)
            self.assertEqual(summary["reported_usage"]["reasoning_tokens"], 15)
            self.assertEqual(summary["http_latency_seconds"]["p50"], 2)

    @unittest.skipUnless(HAS_IMAGES, "NumPy/Pillow required for image verification")
    def test_renderer_observation_is_flipped_exactly_once(self):
        from ds_libero.environment import LiberoEnvironment
        env = LiberoEnvironment.__new__(LiberoEnvironment)
        env.np = np
        env.cfg = replace(Config(), image_size=2)
        env.env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(7))))
        pixels = np.array([[[255, 0, 0], [255, 0, 0]], [[0, 0, 255], [0, 0, 255]]], dtype=np.uint8)
        raw = {"robot0_eef_pos": [0, 0, 1], "robot0_eef_quat": [0, 0, 0, 1], "robot0_gripper_qpos": [.04, -.04]}
        for camera in CAMERAS:
            raw[camera + "_image"] = pixels.copy()
        obs = env._observe(raw)
        with Image.open(io.BytesIO(obs.images["agentview"])) as image:
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))
            self.assertEqual(image.getpixel((0, 1)), (255, 0, 0))
        self.assertEqual(raw["agentview_image"][0, 0].tolist(), [255, 0, 0])

    @unittest.skipUnless(HAS_IMAGES, "NumPy/Pillow required for GIF verification")
    def test_report_builds_replay_and_keeps_raw_png(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "episode"
            store = Store(root, "run", Config())
            for step, color in enumerate(("red", "blue")):
                stream = io.BytesIO()
                Image.new("RGB", (32, 32), color).save(stream, format="PNG")
                store.observe(step, observation(step, {key: stream.getvalue() for key in CAMERAS}))
            store.finish({"status": "success", "scored": True, "task_success": True})
            summary = build_report(root)
            self.assertEqual(summary["replay"], "replay.gif")
            self.assertEqual(len(list((root / "frames").glob("*.png"))), 4)
            with Image.open(root / "replay.gif") as gif:
                self.assertEqual(gif.n_frames, 2)
                self.assertEqual(gif.info["duration"], 50)


if __name__ == "__main__":
    unittest.main()
