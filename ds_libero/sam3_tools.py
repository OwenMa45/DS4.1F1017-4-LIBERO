"""DeepSeek tool loop and a CPU-only client for the isolated SAM3 worker.

Only the current upright RGB PNG and a model-chosen phrase cross this boundary.
No simulator, object poses, depth, success predicates or model-controlled paths.
"""

import base64
import copy
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request

from .artifacts import write_json
from .config import digest
from .errors import APIError, ExperimentError, InvalidAction
from .protocol import CAMERAS, _unique_object


WIRE_VERSION = "sam3-rgb-v1"
DEFAULTS = {"service_url": "http://127.0.0.1:8765", "timeout_seconds": 120,
            "max_tool_rounds": 2, "max_calls_per_round": 4, "max_instances": 6,
            "confidence_threshold": 0.35, "require_call": True, "max_format_retries": 1}
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
TOOL = {"type": "function", "function": {
    "name": "sam3_segment",
    "description": "Segment a visible object or part in a CURRENT camera image using SAM3. "
                   "Returns predicted masks, pixel boxes/centroids, scores and a labeled overlay. "
                   "Use short English noun phrases; one concept and one camera per call. "
                   "This is RGB perception, not ground truth, depth or a grasp detector.",
    "parameters": {"type": "object", "properties": {
        "camera": {"type": "string", "enum": list(CAMERAS)},
        "prompt": {"type": "string", "description": "E.g. black bowl, plate, drawer handle, book"}},
        "required": ["camera", "prompt"], "additionalProperties": False}}}


def settings(values):
    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ExperimentError("Unknown SAM3 configuration field")
    result = {**DEFAULTS, **values}
    for key, lo, hi in (("timeout_seconds", 1, 3600), ("max_tool_rounds", 1, 4),
                        ("max_calls_per_round", 1, 8), ("max_instances", 1, 16),
                        ("max_format_retries", 0, 1)):
        if type(result[key]) is not int or not lo <= result[key] <= hi:
            raise ExperimentError(f"sam3.{key} must be an integer in {lo}..{hi}")
    threshold = result["confidence_threshold"]
    if type(threshold) not in (int, float) or not 0 < threshold < 1:
        raise ExperimentError("sam3.confidence_threshold must be in (0,1)")
    result["confidence_threshold"] = float(threshold)
    if type(result["require_call"]) is not bool:
        raise ExperimentError("sam3.require_call must be boolean")
    try:
        url = urllib.parse.urlsplit(result["service_url"])
        valid = (url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost", "::1")
                 and url.port is not None and not url.username and not url.password
                 and url.path in ("", "/") and not url.query and not url.fragment)
    except (TypeError, ValueError, AttributeError):
        valid = False
    if not valid:
        raise ExperimentError("sam3.service_url must be a loopback HTTP URL with a port (use an SSH tunnel for a remote GPU)")
    result["service_url"] = result["service_url"].rstrip("/")
    return result


def augment_prompt(prompt, options):
    options = settings(options)
    prompt = prompt.replace("You have no tools or scripted skills.",
                            "You can call sam3_segment for RGB perception; there are no scripted control skills.")
    prompt = prompt.replace("No object locations, depth, segmentation, demonstrations or scripted skills are\nprovided.",
                            "No ground-truth object locations, depth, simulator segmentation, demonstrations\nor scripted skills are provided. SAM3 supplies predicted RGB segmentation only.")
    prompt = prompt.replace('Return only a json object with the sole key "actions",',
                            'When returning robot actions (after any tool calls), return only a json object\nwith the sole key "actions",')
    prompt += f'''
SAM3 perception tool protocol:
Before alignment, contact, closure or placement, use sam3_segment to inspect the
target, destination or relevant part (e.g. a drawer handle) in the current views.
Choose the short English noun phrase yourself from the instruction and images.
You have at most {options['max_tool_rounds']} tool rounds, each with at most
{options['max_calls_per_round']} calls per control decision. Tool calls do NOT move
the robot. After tool results, return the usual JSON actions object. Only a final
response without tool calls can supply robot actions. When tools are disabled, choose
actions from the evidence already available; do not request additional tools.
Results are predictions with uncertainty. An empty detection list means no match,
not proof of absence. Try a simpler phrase or the other view within the budget.
Coordinates are in the original upright image: origin top-left, u right, v down.
Mask centroids are not necessarily grasp points. No depth, object pose, grasp
success or cross-camera correspondence is provided. Detection IDs are local to
one call, NOT tracking IDs. Compare masks, fingerpad projections and both RGB
views; retain the existing conservative feedback boundaries near contact.
Overlays have the same size and orientation as their original camera image.
The harness may start a separate action phase with completed SAM3 observations.
In that phase perception is already complete: use the supplied observations and
return actions directly, without attempting another tool call.
'''
    if options["require_call"]:
        prompt += ("Every control decision MUST call SAM3 at least once before returning actions,\n"
                   "unless the harness supplies completed SAM3 observations for its action phase.\n")
    return prompt


def service_json(options, route, body=None):
    # Loopback traffic must not be sent through an environment HTTP proxy.
    from .client import NoRedirect
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    request = urllib.request.Request(options["service_url"] + route, data=data,
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=options["timeout_seconds"]) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise APIError("SAM3 response exceeds size limit")
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("schema") != WIRE_VERSION:
            raise APIError("Incompatible SAM3 worker response")
        return value
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise APIError(f"SAM3 worker request failed: {type(exc).__name__}: {exc}") from exc


def worker_health(options):
    health = service_json(settings(options), "/health")
    identity = health.get("identity")
    if (health.get("status") != "ready" or not isinstance(identity, dict)
            or health.get("runtime_id") != digest(identity)):
        raise APIError("SAM3 worker is not ready or has invalid provenance")
    return health


def validate_arguments(value):
    if (not isinstance(value, dict) or set(value) != {"camera", "prompt"}
            or value["camera"] not in CAMERAS or not isinstance(value["prompt"], str)
            or not 1 <= len(value["prompt"].strip()) <= 200
            or any(ord(c) < 32 for c in value["prompt"])):
        raise InvalidAction("sam3_segment requires camera and a nonempty prompt of at most 200 characters")
    return {"camera": value["camera"], "prompt": value["prompt"].strip()}


def decode_png(encoded, size=None):
    from PIL import Image
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or max(image.size) > 2048 or (size and image.size != size):
                raise ValueError("wrong image format/size")
            image.load()
    except (ValueError, TypeError, OSError) as exc:
        raise APIError("SAM3 returned an invalid PNG") from exc
    return raw


class Sam3Tool:
    def __init__(self, options, store):
        self.options, self.store = settings(options), store
        self.health = worker_health(self.options)
        self.runtime_id = self.health["runtime_id"]
        self.calls = 0
        store.manifest["sam3_worker"] = self.health
        write_json(store.path / "manifest.json", store.manifest)

    def verify_runtime(self, expected):
        if expected is None or expected.get("runtime_id") != self.runtime_id:
            raise ExperimentError("SAM3 weights/code/runtime changed since preflight; start a new experiment")

    def execute(self, arguments, obs, step, call_id):
        from PIL import Image
        arguments = validate_arguments(arguments)
        image = obs.images[arguments["camera"]]
        with Image.open(io.BytesIO(image)) as source:
            size = source.size
        image_hash = hashlib.sha256(image).hexdigest()
        self.calls += 1
        event = {"event": "sam3_call", "step": step, "call": self.calls,
                 "tool_call_id": call_id, **arguments, "image_sha256": image_hash,
                 "runtime_id": self.runtime_id, "status": "error"}
        start = time.perf_counter()
        self.store.event({**event, "event": "sam3_started"})
        try:
            result = service_json(self.options, "/segment", {
                "schema": WIRE_VERSION, "image_base64": base64.b64encode(image).decode("ascii"),
                "prompt": arguments["prompt"], "confidence_threshold": self.options["confidence_threshold"],
                "max_instances": self.options["max_instances"]})
            if (result.get("runtime_id") != self.runtime_id or result.get("image_sha256") != image_hash
                    or result.get("prompt") != arguments["prompt"]
                    or (result.get("width"), result.get("height")) != size):
                raise APIError("SAM3 result does not match the current image, prompt or pinned runtime")
            detections = result.get("detections")
            if not isinstance(detections, list) or len(detections) > self.options["max_instances"]:
                raise APIError("Malformed SAM3 detections")
            overlay = decode_png(result.pop("overlay_png_base64", None), size)
            masks = []
            for index, detection in enumerate(detections):
                if not isinstance(detection, dict) or detection.get("id") != index + 1:
                    raise APIError("Malformed SAM3 detection IDs")
                score, box, center = detection.get("score"), detection.get("bbox_xyxy"), detection.get("centroid_uv")
                if (type(score) not in (int, float) or not 0 <= score <= 1
                        or not isinstance(box, list) or len(box) != 4
                        or not isinstance(center, list) or len(center) != 2
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in box + center)
                        or not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1])
                        or not (0 <= center[0] < size[0] and 0 <= center[1] < size[1])):
                    raise APIError("Malformed SAM3 pixel coordinates or score")
                masks.append(decode_png(detection.pop("mask_png_base64", None), size))
            # Never use model-generated call IDs/prompts as filesystem paths.
            folder = self.store.path / "sam3" / f"step_{step:05d}_call_{self.calls:04d}"
            folder.mkdir(parents=True, exist_ok=False)
            (folder / "overlay.png").write_bytes(overlay)
            for i, mask in enumerate(masks, 1):
                (folder / f"mask_{i:02d}.png").write_bytes(mask)
            result.update(camera=arguments["camera"], control_step=step)
            write_json(folder / "result.json", result)
            event.update(status="ok", detections=len(detections),
                         artifacts=folder.relative_to(self.store.path).as_posix())
            return [{"type": "text", "text": json.dumps(result, allow_nan=False)},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(overlay).decode("ascii"),
                        "detail": "original"}}]
        except ExperimentError as exc:
            event.update(error=str(exc), error_status=exc.status)
            raise
        finally:
            event["elapsed_seconds"] = time.perf_counter() - start
            self.store.event(event)


def _choice(reply):
    try:
        choice = reply["choices"][0]
        message, finish = choice["message"], choice["finish_reason"]
        if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
            raise TypeError()
        return message, finish
    except (KeyError, IndexError, TypeError) as exc:
        raise InvalidAction("Malformed DeepSeek tool response") from exc


def _format_problem(message, finish):
    # Retry serialization failures only. Valid JSON with illegal actions, duplicate
    # keys, truncated replies and unexpected tool calls still fail normal validation.
    if finish != "stop" or message.get("tool_calls"):
        return None
    content = message.get("content")
    if content is None or isinstance(content, str) and not content.strip():
        return "blank_action_content"
    if isinstance(content, str):
        try:
            json.loads(content, object_pairs_hook=_unique_object)
        except ValueError:
            return "non_json_action_content"
    return None


def _action_reply(client, base_payload, observations, step, options):
    # Start a fresh, tool-free JSON request. Do not mix JSON mode/tool_choice=none
    # with tool-protocol messages, or reinterpret reasoning/DSML as robot actions.
    payload = copy.deepcopy(base_payload)
    payload.pop("tools", None)
    payload.pop("tool_choice", None)
    payload["response_format"] = {"type": "json_object"}
    instruction = (
        "\nSAM3 action phase: perception calls for this decision are complete. "
        "The supplied SAM3 observations refer to the same current camera images. "
        "No tools are available in this request. Use those observations, the original "
        "images and robot state to choose actions under the existing control rules. "
        "Return only a JSON object with the sole key actions, containing 1.."
        "max_actions_this_response arrays of seven numbers. A shape-only example is "
        '{"actions":[[0,0,0,0,0,0,-1]]}; choose the actual values from the observations. '
        "Write the JSON directly in content, with no prose, code fence or tool markup.\n")
    payload["messages"][0]["content"] += instruction
    content = [{"type": "text", "text": f"Completed SAM3 observations for control step {step}:"}]
    for arguments, parts in observations:
        content.append({"type": "text", "text": json.dumps({"sam3_query": arguments}, allow_nan=False)})
        content.extend(copy.deepcopy(parts))
    payload["messages"].append({"role": "user", "content": content})
    client.store.event({"event": "sam3_action_phase", "step": step, "perception_calls": len(observations)})
    for retry in range(options["max_format_retries"] + 1):
        reply = client.complete(payload, step)
        message, finish = _choice(reply)
        if message.get("tool_calls"):
            raise InvalidAction("Tool call returned during the tool-free action phase")
        problem = _format_problem(message, finish)
        if problem is None or retry == options["max_format_retries"]:
            return reply
        client.store.event({"event": "sam3_action_format_retry", "step": step,
                            "retry": retry + 1, "reason": problem})
        payload["messages"].append({"role": "user", "content":
            "The previous response did not contain a complete JSON action object. "
            "Perception is complete and the robot has not moved. Return the action JSON "
            "directly in content, without tool calls, protocol markup, prose or markdown."})
    raise AssertionError("unreachable")


def complete_with_sam3(client, payload, obs, step, tool):
    """One decision, bounded inference rounds, zero simulator actions."""
    options = settings(tool.options)
    base_payload = payload
    payload = copy.deepcopy(base_payload)
    payload["tools"] = [copy.deepcopy(TOOL)]
    payload["tool_choice"] = "auto"
    payload.pop("response_format", None)
    seen_ids = set()
    calls_made = 0
    observations = []
    for turn in range(options["max_tool_rounds"]):
        # DeepSeek thinking rejects required/named choices. The prompt and
        # require_call validation enforce our experiment's perception contract.
        reply = client.complete(payload, step)
        message, finish = _choice(reply)
        calls = message.get("tool_calls")
        if not calls:
            if finish == "tool_calls":
                raise InvalidAction("Empty tool_calls response")
            if finish == "stop" and options["require_call"] and not calls_made:
                raise InvalidAction("SAM3 experiment requires a perception call before actions")
            if calls_made and _format_problem(message, finish):
                return _action_reply(client, base_payload, observations, step, options)
            return reply  # Existing action parser checks finish reason and all actions.
        if finish != "tool_calls":
            raise InvalidAction("Unexpected, truncated or over-budget tool call response")
        if (not isinstance(calls, list) or not 1 <= len(calls) <= options["max_calls_per_round"]
                or (message.get("content") is not None and not isinstance(message["content"], str))
                or not isinstance(message.get("reasoning_content"), str)):
            raise InvalidAction("Malformed tool batch/content or missing reasoning_content")
        parsed = []
        for call in calls:
            if (not isinstance(call, dict) or call.get("type") != "function"
                    or not isinstance(call.get("id"), str) or not 1 <= len(call["id"]) <= 200
                    or call["id"] in seen_ids or not isinstance(call.get("function"), dict)
                    or call["function"].get("name") != "sam3_segment"):
                raise InvalidAction("Unknown tool, malformed call or duplicate tool_call_id")
            seen_ids.add(call["id"])
            encoded = call["function"].get("arguments")
            try:
                if not isinstance(encoded, str) or len(encoded) > 4096:
                    raise ValueError()
                arguments = validate_arguments(json.loads(encoded, object_pairs_hook=_unique_object))
            except (ValueError, TypeError) as exc:
                raise InvalidAction("Invalid sam3_segment arguments JSON") from exc
            parsed.append((call["id"], arguments))
        # Preserve the complete reasoning content across all within-decision
        # tool rounds, as required by the thinking API. No artificial tool calls.
        payload["messages"].append({"role": "assistant", "content": message.get("content"),
            "reasoning_content": message["reasoning_content"], "tool_calls": copy.deepcopy(calls)})
        for call_id, arguments in parsed:
            content = tool.execute(arguments, obs, step, call_id)
            calls_made += 1
            observations.append((copy.deepcopy(arguments), copy.deepcopy(content)))
            payload["messages"].append({"role": "tool", "tool_call_id": call_id, "content": content})
    return _action_reply(client, base_payload, observations, step, options)
