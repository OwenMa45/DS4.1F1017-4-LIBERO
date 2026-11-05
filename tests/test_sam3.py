import base64
import contextlib
import copy
import hashlib
import io
import json
import tempfile
import threading
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ds_libero.artifacts import Store, write_json
from ds_libero.client import Client, HTTPResponse, ModelPolicy
from ds_libero.config import Config, digest
from ds_libero.errors import APIError, ExperimentError, InvalidAction
from ds_libero.microbench import MicroBench
from ds_libero.protocol import CAMERAS, Observation, request_payload, system_prompt
from ds_libero.report import build_report
from ds_libero.sam3_server import (Sam3Backend, encode_predictions, load_local_image_weights,
                                  make_server, png_bytes, resolve_paths)
from ds_libero.sam3_tools import (Sam3Tool, WIRE_VERSION, complete_with_sam3, service_json,
                                settings, validate_arguments)
from test_experiment import HAS_IMAGES, FakeEnv, MemoryStore, observation, reply, run_quiet


ACTION = [0, 0, 0, 0, 0, 0, -1]


def config(**changes):
    return Config.from_dict({"protocol_version": "ds-libero-direct-v2", "max_steps": 3,
                             "action_horizon": 3, "sam3": {"max_tool_rounds": 2}, **changes})


def tool_reply(calls=None, finish="tool_calls", **fields):
    if calls is None:
        calls = [("call_1", "agentview", "black bowl")]
    message = {"role": "assistant", "content": None, "reasoning_content": "opaque test reasoning",
               "tool_calls": [{"id": ident, "type": "function", "function": {
                   "name": "sam3_segment", "arguments": json.dumps({"camera": cam, "prompt": prompt})}}
                   for ident, cam, prompt in calls], **fields}
    return HTTPResponse(200, json.dumps({"model": "deepseek-flash",
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}).encode())


class FakeTool:
    def __init__(self, options=None):
        self.options = settings(options or {})
        self.seen = []

    def execute(self, args, obs, step, call_id):
        self.seen.append((args, obs, step, call_id))
        return [{"type": "text", "text": '{"detections":[]}'}]


class ToolLoopTests(unittest.TestCase):
    def run_policy(self, responses, cfg=None, tool=None, store=None, send_check=None):
        cfg, tool, store = cfg or config(), tool or FakeTool(), store or MemoryStore()
        env = FakeEnv(success_at=1)
        sent = []

        def send(endpoint, body, key, timeout):
            self.assertEqual(env.steps, 0, "Neither tool rounds nor retries may advance simulation")
            sent.append(json.loads(body))
            if send_check:
                send_check(sent[-1])
            return responses.pop(0)

        with patch("ds_libero.sam3_tools.Sam3Tool", return_value=tool):
            policy = ModelPolicy(cfg, "put bowl on plate", Client(cfg, store, api_key="fake-key", send=send,
                                 sleep=lambda _: None), store)
        result = run_quiet(env, policy, store, cfg)
        return result, sent, tool, store

    def test_multiround_thinking_preserved_images_current_and_only_final_actions_execute(self):
        first = tool_reply([("a", "agentview", "black bowl"), ("b", "robot0_eye_in_hand", "plate")],
                           content="Inspecting the two visible objects.")
        second = tool_reply([("c", "agentview", "plate")], reasoning_content="second opaque trace")
        result, sent, tool, store = self.run_policy([first, second, reply(json.dumps({"actions": [ACTION]}))])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["control_steps"], 1)
        self.assertEqual(result["policy_decisions"], 1)
        self.assertEqual(len(tool.seen), 3)
        self.assertEqual([p["tool_choice"] for p in sent], ["auto", "auto", "none"])
        self.assertNotIn("response_format", sent[0])
        self.assertEqual(sent[-1]["response_format"], {"type": "json_object"})
        assistants = [m for m in sent[-1]["messages"] if m["role"] == "assistant"]
        self.assertEqual([m["reasoning_content"] for m in assistants], ["opaque test reasoning", "second opaque trace"])
        self.assertEqual(assistants[0]["content"], "Inspecting the two visible objects.")
        self.assertEqual([m["tool_call_id"] for m in sent[-1]["messages"] if m["role"] == "tool"], ["a", "b", "c"])
        for args, obs, step, _ in tool.seen:
            self.assertEqual(obs.images, observation().images)
            self.assertEqual(step, 0)
        self.assertEqual(len([e for e in store.events if e["event"] == "decision"]), 1)
        self.assertEqual(len([e for e in store.events if e["event"] == "api_attempt"]), 3)

    def test_retry_uses_identical_tool_context_and_shared_http_budget(self):
        result, sent, _, _ = self.run_policy([tool_reply(), HTTPResponse(503, b"{}"), reply('{"actions": [[0,0,0,0,0,0,-1]]}')])
        self.assertEqual(result["status"], "success")
        self.assertEqual(sent[1], sent[2])
        result, _, tool, _ = self.run_policy([tool_reply()], cfg=config(max_http_requests=1))
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertFalse(result["scored"])
        self.assertEqual(result["control_steps"], 0)
        self.assertEqual(len(tool.seen), 1)

    def test_required_perception_and_optional_mode(self):
        action = reply('{"actions": [[0,0,0,0,0,0,-1]]}')
        result, _, _, _ = self.run_policy([action])
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(result["control_steps"], 0)
        optional = FakeTool({"require_call": False})
        result, _, _, _ = self.run_policy([action], tool=optional)
        self.assertEqual(result["status"], "success")

    def test_truncated_mixed_missing_reasoning_unknown_and_overlarge_batches_rejected_before_tool(self):
        bad_name = json.loads(tool_reply().body)
        bad_name["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "exec"
        cases = [tool_reply(finish="length"), tool_reply(content={"actions": []}),
                 tool_reply(reasoning_content=None), HTTPResponse(200, json.dumps(bad_name).encode()),
                 tool_reply([(str(i), "agentview", "bowl") for i in range(5)]),
                 tool_reply([("a", "agentview", "bowl"), ("b", "../../file", "bowl")]),
                 tool_reply([("a", "agentview", "bowl"), ("a", "agentview", "bowl")])]
        for response in cases:
            with self.subTest(response=response.body[:80]):
                result, _, tool, _ = self.run_policy([response])
                self.assertEqual(result["status"], "invalid_output")
                self.assertEqual(tool.seen, [])
                self.assertEqual(result["control_steps"], 0)

    def test_duplicate_json_arguments_and_round_budget(self):
        data = json.loads(tool_reply().body)
        data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"camera":"agentview","prompt":"a","prompt":"b"}'
        result, _, tool, _ = self.run_policy([HTTPResponse(200, json.dumps(data).encode())])
        self.assertEqual(result["status"], "invalid_output")
        self.assertFalse(tool.seen)
        tool = FakeTool({"max_tool_rounds": 1})
        result, _, tool, _ = self.run_policy([tool_reply(), tool_reply([("b", "agentview", "plate")])], tool=tool)
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(len(tool.seen), 1)
        self.assertEqual(result["control_steps"], 0)

    def test_service_failure_unscored_and_illegal_final_actions_not_executed(self):
        tool = FakeTool()
        with patch.object(tool, "execute", side_effect=APIError("worker unavailable")):
            result, _, _, _ = self.run_policy([tool_reply()], tool=tool)
        self.assertEqual(result["status"], "api_error")
        self.assertFalse(result["scored"])
        result, _, _, _ = self.run_policy([tool_reply(), reply('{"actions": [[2,0,0,0,0,0,-1]]}')])
        self.assertTrue(result["scored"])
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(result["control_steps"], 0)

    def test_distinct_request_artifacts_and_all_round_costs(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = config()
            store = Store(Path(temp) / "episode", "run", cfg)
            result, _, _, _ = self.run_policy([tool_reply(), reply('{"actions": [[0,0,0,0,0,0,-1]]}')], cfg=cfg, store=store)
            store.finish(result)
            events = [json.loads(line) for line in (store.path / "events.jsonl").read_text().splitlines()]
            calls = [e for e in events if e["event"] == "api_attempt"]
            self.assertEqual(len({e["request_file"] for e in calls}), 2)
            for call in calls:
                self.assertEqual(hashlib.sha256((store.path / call["request_file"]).read_bytes()).hexdigest(), call["request_sha256"])
            report = build_report(store.path, replay=False)
            self.assertEqual(report["http_requests"], 2)
            self.assertEqual(report["reported_usage"]["total_tokens"], 60)
            self.assertEqual(report["model_outputs"], 1)


class ConfigurationTests(unittest.TestCase):
    def test_baseline_hash_prompt_and_payload_remain_unchanged(self):
        self.assertEqual(Config().fingerprint, "911c4f579ce0bd922aee8e2472ddb658b58fdf42f3fd76cd00be3887017b37ff")
        self.assertNotIn("sam3", Config().to_dict())
        plain = request_payload(Config(), "task", observation(), [], 0)
        self.assertNotIn("tools", plain)
        prompt = system_prompt(config())
        self.assertIn("SAM3 perception tool protocol", prompt)
        self.assertNotIn("You have no tools", prompt)
        self.assertNotIn("No object locations, depth, segmentation", prompt)

    def test_settings_reject_unknown_or_invalid_values(self):
        for value in ({"device": "cuda"}, {"timeout_seconds": True}, {"max_tool_rounds": 0},
                      {"max_instances": 100}, {"require_call": "yes"}, {"confidence_threshold": float("nan")},
                      {"service_url": "http://external.example:80"}, {"service_url": "http://x@y:80"}):
            with self.subTest(value=value), self.assertRaises(ExperimentError):
                settings(value)
        with self.assertRaises(ExperimentError):
            Config.from_dict({"sam3": {"require_call": True}})

    def test_model_cannot_choose_paths_camera_keys_or_blank_prompts(self):
        for value in ({"camera": "agentview", "prompt": " "}, {"camera": "all", "prompt": "bowl"},
                      {"camera": "agentview", "prompt": "bowl", "path": "/tmp/x"},
                      {"camera": "agentview", "prompt": "x" * 201}):
            with self.assertRaises(InvalidAction):
                validate_arguments(value)

    def test_planning_offline_comparable_protocol_but_distinct_policy_and_correct_budget(self):
        with tempfile.TemporaryDirectory() as temp, patch("ds_libero.sam3_tools.service_json", side_effect=AssertionError("network forbidden")):
            base = MicroBench("configs/deepseek_baseline_220.json").prepare(Path(temp) / "base")
            sam = MicroBench("configs/deepseek_sam3_220.json").prepare(Path(temp) / "sam", policy_id="ds-sam3")
            self.assertEqual(base["microbench"]["protocol_sha256"], sam["microbench"]["protocol_sha256"])
            self.assertNotEqual(base["episodes"][0]["config_sha256"], sam["episodes"][0]["config_sha256"])
            self.assertEqual(sam["max_http_requests_per_episode_including_api_check"], 1989)
            self.assertEqual(base["max_http_requests_per_episode_including_api_check"], 1983)
            self.assertEqual(len(sam["episodes"]), 5)

    def test_local_paths_package_parent_checkpoint_selection_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            # Match a sam3/ storage directory with separate checkout and weights.
            root = Path(temp) / "sam3" / "sam3"
            code, checkpoints = root / "sam3", root.parent / "checkpoints"
            (code / "assets").mkdir(parents=True)
            (code / "model_builder.py").write_text("")
            (code / "assets/bpe_simple_vocab_16e6.txt.gz").write_bytes(b"vocab")
            checkpoints.mkdir()
            with self.assertRaises(ExperimentError):
                resolve_paths(code, checkpoints)
            (checkpoints / "one.pt").write_bytes(b"weights")
            self.assertEqual(resolve_paths(root, checkpoints)[1].name, "one.pt")
            self.assertEqual(resolve_paths(root, checkpoints)[0], code.resolve())
            self.assertEqual(resolve_paths(code, checkpoints)[0], code.resolve())
            (checkpoints / "two.pt").write_bytes(b"weights")
            with self.assertRaises(ExperimentError):
                resolve_paths(code, checkpoints)
            (checkpoints / "sam3.pt").write_bytes(b"weights")
            self.assertEqual(resolve_paths(code, checkpoints)[1].name, "sam3.pt")


class BenchIntegrationTests(unittest.TestCase):
    def test_five_tasks_real_policy_loop_preflight_costs_and_runtime_freeze(self):
        from test_microbench import StubEnvironment
        health = {"schema": WIRE_VERSION, "identity": {"checkpoint": "fixture"}, "status": "ready"}
        health["runtime_id"] = digest(health["identity"])

        class AuditedTool(FakeTool):
            def __init__(self, options, store):
                super().__init__(options)
                self.store = store
                store.manifest["sam3_worker"] = health
                write_json(store.path / "manifest.json", store.manifest)

            def verify_runtime(self, expected):
                if expected["runtime_id"] != health["runtime_id"]:
                    raise ExperimentError("runtime mismatch")

            def execute(self, args, obs, step, call_id):
                self.store.event({"event": "sam3_call", "step": step, "call": 1, "status": "ok",
                                  "detections": 0, "elapsed_seconds": 0.01})
                return super().execute(args, obs, step, call_id)

        def send(endpoint, body, key, timeout):
            payload = json.loads(body)
            if any(m["role"] == "tool" for m in payload["messages"]):
                return reply('{"actions": [[0,0,0,0,0,0,-1]]}')
            return tool_reply()

        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            root = Path(temp) / "five"
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch("ds_libero.environment.LiberoEnvironment", side_effect=lambda cfg: StubEnvironment(cfg, success_at=1)))
            stack.enter_context(patch("ds_libero.calibration.calibrate", return_value={"status": "pass", "scored": False, "task_success": None}))
            stack.enter_context(patch("ds_libero.client.resolve_api_key", return_value="test-key"))
            stack.enter_context(patch("ds_libero.client.Client", side_effect=lambda cfg, store: Client(cfg, store, api_key="test-key", send=send)))
            stack.enter_context(patch("ds_libero.sam3_tools.Sam3Tool", AuditedTool))
            stack.enter_context(patch("ds_libero.sam3_tools.worker_health", return_value=health))
            bench = MicroBench(config())
            bench.prepare(root, policy_id="fixture-sam3")
            first = bench.run(root, episode=0, replay=False)
            self.assertEqual(first["scored"], 1)
            self.assertTrue((root / "sam3_runtime.json").is_file())
            with patch("ds_libero.sam3_tools.worker_health", return_value={**health, "runtime_id": "other"}):
                with self.assertRaises(ExperimentError):
                    bench.run(root, episode=1, replay=False)
            self.assertFalse((root / "runs/episode-0001").exists())
            final = bench.run(root, replay=False)
            self.assertEqual(final["score"], 1.0)
            self.assertEqual(final["scored"], 5)
            for row in final["episodes"]:
                self.assertEqual(row["http_requests"], 2)
                self.assertEqual(row["policy_decisions"], 1)
                self.assertEqual(row["sam3"]["successful_calls"], 1)
                summary = json.loads((root / row["artifacts"] / "campaign_summary.json").read_text())
                self.assertEqual(summary["http_requests"], 4)


class LoaderTests(unittest.TestCase):
    @staticmethod
    def image_model(keys=("weight",)):
        return SimpleNamespace(
            backbone=SimpleNamespace(vision_backbone=SimpleNamespace(sam2_convs=None)),
            inst_interactive_predictor=None, state_dict=lambda: dict.fromkeys(keys),
            load_state_dict=Mock())

    @staticmethod
    def sam2_neck_weights():
        # The 22 extra keys reported by the real checkpoint, not a model fixture.
        layers = [("dconv_2x2_0", "dconv_2x2_1", "conv_1x1", "conv_3x3"),
                  ("dconv_2x2", "conv_1x1", "conv_3x3"),
                  ("conv_1x1", "conv_3x3"), ("conv_1x1", "conv_3x3")]
        return {f"backbone.vision_backbone.sam2_convs.{index}.{layer}.{field}": "unused"
                for index, names in enumerate(layers) for layer in names for field in ("weight", "bias")}

    def test_full_checkpoint_and_image_state_ignore_only_disabled_sam2_neck(self):
        extras = self.sam2_neck_weights()
        self.assertEqual(len(extras), 22)
        active = {"backbone.vision_backbone.convs.0.conv_1x1.weight": "active", "weight": "active"}
        image = {**active, **extras}
        full = {"model": {**{"detector." + key: value for key, value in image.items()},
                          "tracker.weight": "unused tracker"}}
        for checkpoint in (full, image):
            with self.subTest(wrapped="model" in checkpoint):
                model = self.image_model(active)
                torch = SimpleNamespace(load=Mock(return_value=checkpoint))
                audit = load_local_image_weights(model, "unused.pt", torch)
                model.load_state_dict.assert_called_once_with(active, strict=True)
                self.assertEqual(audit, {"strict": True, "loaded_keys": len(active),
                                        "ignored_disabled_sam2_keys": sorted(extras)})
                self.assertEqual(len(image), len(active) + 22)  # Never mutate the checkpoint.

    def test_unknown_disabled_state_or_enabled_branch_does_not_filter(self):
        state = {"weight": "active", **self.sam2_neck_weights()}
        for branch in ("active_neck", "active_predictor", "unknown_neck", "unknown_predictor"):
            with self.subTest(branch=branch):
                model = self.image_model()
                if branch == "active_neck":
                    model.backbone.vision_backbone.sam2_convs = object()
                elif branch == "active_predictor":
                    model.inst_interactive_predictor = object()
                elif branch == "unknown_neck":
                    del model.backbone.vision_backbone.sam2_convs
                else:
                    del model.inst_interactive_predictor
                load_local_image_weights(model, "unused.pt", SimpleNamespace(load=lambda *a, **kw: state))
                model.load_state_dict.assert_called_once_with(state, strict=True)

    def test_registered_sam2_keys_are_never_discarded(self):
        state = {"weight": "active", **self.sam2_neck_weights()}
        model = self.image_model(state)
        audit = load_local_image_weights(model, "unused.pt", SimpleNamespace(load=lambda *a, **kw: state))
        model.load_state_dict.assert_called_once_with(state, strict=True)
        self.assertEqual(audit["ignored_disabled_sam2_keys"], [])

    def test_other_extra_missing_and_wrong_shape_weights_still_fail_strictly(self):
        for active, error in (({"weight": "active", "unrecognized.weight": 1}, "Unexpected key(s)"),
                              ({}, "Missing key(s)"), ({"weight": "wrong shape"}, "size mismatch")):
            with self.subTest(error=error):
                model = self.image_model()
                model.load_state_dict.side_effect = RuntimeError(error)
                state = {**active, **self.sam2_neck_weights()}
                with self.assertRaisesRegex(ExperimentError, "strict load") as raised:
                    load_local_image_weights(model, "unused.pt", SimpleNamespace(load=lambda *a, **kw: state))
                self.assertIn(error, str(raised.exception))
                model.load_state_dict.assert_called_once_with(active, strict=True)

    def test_builder_uses_supplied_package_local_checkpoint_and_no_hf(self):
        import types
        with tempfile.TemporaryDirectory() as temp:
            code = Path(temp) / "sam3"
            (code / "assets").mkdir(parents=True)
            (code / "model_builder.py").write_text("# fixture")
            (code / "__init__.py").write_text("")
            (code / "assets/bpe_simple_vocab_16e6.txt.gz").write_bytes(b"fixture vocabulary")
            checkpoint = Path(temp) / "sam3.pt"
            checkpoint.write_bytes(b"not a real checkpoint: constructor test only")
            package = types.ModuleType("sam3")
            package.__file__ = str(code / "__init__.py")
            builder = types.ModuleType("sam3.model_builder")
            calls = []
            loaded = []
            class Model:
                backbone = SimpleNamespace(vision_backbone=SimpleNamespace(sam2_convs=None))
                inst_interactive_predictor = None
                def state_dict(self): return {"weight": None}
                def load_state_dict(self, state, strict): loaded.append((state, strict))
                def to(self, device): return self
                def eval(self): return self
            def build(**kwargs):
                calls.append(kwargs)
                return Model()
            builder.build_sam3_image_model = build
            processor = types.ModuleType("sam3.model.sam3_image_processor")
            processor.Sam3Processor = lambda *args, **kwargs: object()
            torch = types.ModuleType("torch")
            torch.__version__ = "fixture"
            torch.cuda = types.SimpleNamespace(is_available=lambda: False)
            def load(path, **kwargs):
                self.assertEqual(path, str(checkpoint.resolve()))
                self.assertEqual(kwargs, {"map_location": "cpu", "weights_only": True})
                return {"model": {"detector.weight": "image weights", "tracker.weight": "not used",
                                  **{"detector." + key: value for key, value in self.sam2_neck_weights().items()}}}
            torch.load = load
            with patch.dict("sys.modules", {"torch": torch, "sam3": package, "sam3.model_builder": builder,
                                           "sam3.model.sam3_image_processor": processor}), patch("sys.path", list(__import__("sys").path)):
                backend = Sam3Backend(code, checkpoint, device="cpu", precision="float32")
            self.assertIsNone(calls[0]["checkpoint_path"])
            self.assertEqual(loaded, [({"weight": "image weights"}, True)])
            self.assertEqual(calls[0]["bpe_path"], str((code / "assets/bpe_simple_vocab_16e6.txt.gz").resolve()))
            self.assertFalse(calls[0]["load_from_HF"])
            self.assertFalse(calls[0]["enable_inst_interactivity"])
            self.assertEqual(backend.health()["identity"]["checkpoint_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            audit = backend.health()["identity"]["checkpoint_load"]
            self.assertEqual(audit["ignored_disabled_sam2_keys"], sorted(self.sam2_neck_weights()))
            self.assertTrue(audit["strict"])
            self.assertEqual(backend.runtime_id, digest(backend.health()["identity"]))

    def test_incompatible_weights_fail_instead_of_serving_partial_model(self):
        import types
        torch = types.SimpleNamespace(load=lambda *args, **kwargs: {"model": {"detector.bad": 1}})
        class Model:
            def state_dict(self): return {"weight": None}
            def load_state_dict(self, state, strict):
                if strict:
                    raise RuntimeError("Missing image weights")
        with self.assertRaisesRegex(ExperimentError, "strict load"):
            load_local_image_weights(Model(), "unused.pt", torch)


@unittest.skipUnless(HAS_IMAGES, "NumPy and Pillow required")
class WorkerTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from PIL import Image
        self.np = np
        self.image = Image.new("RGB", (256, 256), "white")
        self.image.putpixel((5, 9), (255, 0, 0))
        self.png = png_bytes(self.image)
        self.masks = np.zeros((2, 1, 256, 256), dtype=bool)
        self.masks[0, 0, 6:10, 4:8] = True
        self.masks[1, 0, 200:210, 180:190] = True
        self.boxes = np.array([[4, 6, 8, 10], [180, 200, 190, 210]], dtype=float)
        self.scores = np.array([0.9, 0.7])

    def test_original_pixel_coordinates_masks_overlay_order_and_empty_predictions(self):
        from PIL import Image
        result = encode_predictions(self.image, self.masks, self.boxes, self.scores, 1)
        first = result["detections"][0]
        self.assertEqual(first["bbox_xyxy"], [4, 6, 8, 10])
        self.assertEqual(first["centroid_uv"], [5.5, 7.5])
        self.assertEqual(first["mask_area_pixels"], 16)
        mask = Image.open(io.BytesIO(base64.b64decode(first["mask_png_base64"])))
        self.assertEqual(mask.getpixel((5, 7)), 255)
        self.assertEqual(mask.getpixel((5, 248)), 0)
        self.assertEqual(Image.open(io.BytesIO(base64.b64decode(result["overlay_png_base64"]))).size, (256, 256))
        empty = encode_predictions(self.image, self.masks[:0], self.boxes[:0], self.scores[:0], 6)
        self.assertEqual(empty["detections"], [])

    def backend(self):
        owner = self

        class Backend:
            identity = {"checkpoint_sha256": "fake-only", "device": "test"}
            runtime_id = digest(identity)

            def health(self):
                return {"schema": WIRE_VERSION, "status": "ready", "identity": self.identity, "runtime_id": self.runtime_id}

            def segment(self, request):
                owner.assertEqual(set(request), {"schema", "image_base64", "prompt", "confidence_threshold", "max_instances"})
                owner.assertEqual(base64.b64decode(request["image_base64"]), owner.png)
                data = encode_predictions(owner.image, owner.masks, owner.boxes, owner.scores, request["max_instances"])
                return {**data, "schema": WIRE_VERSION, "image_sha256": hashlib.sha256(owner.png).hexdigest(),
                        "runtime_id": self.runtime_id, "prompt": request["prompt"]}

        return Backend()

    @contextlib.contextmanager
    def server(self, backend):
        server = make_server(backend, port=0)
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.05), daemon=True)
        thread.start()
        try:
            yield settings({"service_url": f"http://127.0.0.1:{server.server_port}"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_loopback_service_result_reaches_deepseek_and_is_audited(self):
        backend = self.backend()
        with self.server(backend) as options, tempfile.TemporaryDirectory() as temp:
            cfg = config(sam3=options)
            store = Store(Path(temp) / "run", "run", cfg)
            tool = Sam3Tool(options, store)
            obs = Observation({}, {c: self.png for c in CAMERAS})
            content = tool.execute({"camera": "agentview", "prompt": "bowl"}, obs, 0, "../malicious-id")
            self.assertEqual([c["type"] for c in content], ["text", "image_url"])
            self.assertIn("data:image/png;base64,", content[1]["image_url"]["url"])
            text = json.loads(content[0]["text"])
            self.assertEqual(text["camera"], "agentview")
            self.assertNotIn("mask_png_base64", text["detections"][0])
            folder = store.path / "sam3/step_00000_call_0001"
            self.assertTrue((folder / "mask_01.png").is_file())
            self.assertTrue((folder / "overlay.png").is_file())
            self.assertEqual(store.manifest["sam3_worker"]["runtime_id"], backend.runtime_id)
            tool.verify_runtime(backend.health())
            with self.assertRaises(ExperimentError):
                tool.verify_runtime({"runtime_id": "changed"})
            store.finish({"status": "pass", "scored": False, "task_success": None})
            report = build_report(store.path, replay=False)
            self.assertEqual(report["sam3"]["successful_calls"], 1)
            self.assertIsNone(report["sam3"]["compute_cost"])

    def test_wrong_frame_or_restarted_worker_rejected_as_service_failure(self):
        backend = self.backend()
        with self.server(backend) as options, tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / "run", "run", config(sam3=options))
            tool = Sam3Tool(options, store)
            backend.runtime_id = "changed-after-health"
            with self.assertRaises(APIError):
                tool.execute({"camera": "agentview", "prompt": "bowl"}, Observation({}, {c: self.png for c in CAMERAS}), 0, "a")
            self.assertFalse((store.path / "sam3").exists())

    def test_backend_caches_images_not_prompts_and_never_flips(self):
        owner = self

        class Tensor:
            def __init__(self, value): self.value = value
            def detach(self): return self
            def cpu(self): return self
            def float(self): return self
            def numpy(self): return self.value

        class Processor:
            def __init__(self): self.encodes, self.prompts, self.resets = 0, [], 0
            def set_image(self, image):
                owner.assertEqual(image.getpixel((5, 9)), (255, 0, 0))
                self.encodes += 1
                return {"backbone_out": {}}
            def reset_all_prompts(self, state): self.resets += 1
            def set_confidence_threshold(self, threshold): self.threshold = threshold
            def set_text_prompt(self, prompt, state):
                self.prompts.append(prompt)
                return {"masks": Tensor(owner.masks), "boxes": Tensor(owner.boxes), "scores": Tensor(owner.scores)}

        backend = Sam3Backend.__new__(Sam3Backend)
        backend.processor, backend.cache = Processor(), OrderedDict()
        backend.torch = type("Torch", (), {"inference_mode": staticmethod(contextlib.nullcontext)})()
        backend.precision, backend.runtime_id = "float32", "fake"
        request = {"schema": WIRE_VERSION, "image_base64": base64.b64encode(self.png).decode(),
                   "prompt": "bowl", "confidence_threshold": 0.35, "max_instances": 6}
        first = backend.segment(request)
        second = backend.segment({**request, "prompt": "plate", "confidence_threshold": 0.5})
        self.assertFalse(first["image_embedding_cached"])
        self.assertTrue(second["image_embedding_cached"])
        self.assertEqual(backend.processor.encodes, 1)
        self.assertEqual(backend.processor.prompts, ["bowl", "plate"])
        self.assertEqual(backend.processor.resets, 4)
        self.assertEqual(backend.processor.threshold, 0.5)
        self.assertEqual(first["image_sha256"], hashlib.sha256(self.png).hexdigest())


if __name__ == "__main__":
    unittest.main()
