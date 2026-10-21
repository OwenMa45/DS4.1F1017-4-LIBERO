import base64
import json
import math
from dataclasses import dataclass

from .errors import InvalidAction


CAMERAS = ("agentview", "robot0_eye_in_hand")


@dataclass(frozen=True)
class Observation:
    state: dict
    images: dict  # Upright PNG bytes, one per camera; no privileged scene state.


SYSTEM_PROMPT = """You control a simulated Franka Panda robot in LIBERO using visual feedback.
Complete the supplied natural-language task. You have no tools or scripted skills.
The two current images are an external agentview and a wrist-mounted camera.
Proprioception is end-effector position in world metres, end-effector quaternion
in [x,y,z,w] order, and the two gripper joint positions in metres. Previous
executed actions and proprioception are supplied as a short history.

Return exactly one json object with the sole key "action" and seven numeric items:
{"action":[0,0,0,0,0,0,-1]}
This example only illustrates the format; choose actions that accomplish the task.
The first three values are WORLD-frame normalized end-effector position deltas.
Each is in [-1,1]; 1 means a +0.05 metre target offset for that control step.
The next three are WORLD-frame normalized axis-angle rotation-vector deltas,
not Euler angles or an absolute orientation. Each is in [-1,1]; scale is 0.5 radians.
The final value must be -1 (open), 1 (close), or 0 (hold gripper command).
The underlying controller moves toward the target; actual motion can be smaller
than the requested offset. Each action executes for one 20 Hz control interval
(0.05 seconds of simulated time), then you receive fresh observations.
Inference latency does not advance the simulation. Use feedback to correct motion.
Do not return explanations, markdown, additional keys, absolute poses, or multiple actions.
"""


def validate_action(action):
    if not isinstance(action, list) or len(action) != 7:
        raise InvalidAction("action must be a JSON array of exactly seven numbers")
    if any(type(x) not in (int, float) for x in action):
        raise InvalidAction("action cannot contain booleans, strings, or nested arrays")
    try:
        if any(not math.isfinite(x) or not -1 <= x <= 1 for x in action):
            raise InvalidAction("action values must be finite and within [-1,1]")
    except OverflowError as exc:
        raise InvalidAction("action number is too large") from exc
    if action[6] not in (-1, 0, 1):
        raise InvalidAction("gripper must be exactly -1, 0, or 1")
    return [float(x) for x in action]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidAction(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_action(content, finish_reason="stop"):
    if finish_reason != "stop":
        raise InvalidAction(f"incomplete or unsupported response: finish_reason={finish_reason!r}")
    if not isinstance(content, str) or not content.strip():
        raise InvalidAction("model returned empty/non-text content")
    try:
        obj = json.loads(content, object_pairs_hook=_unique_object)
    except (ValueError, TypeError) as exc:
        raise InvalidAction("model content is not a complete JSON object") from exc
    if not isinstance(obj, dict) or set(obj) != {"action"}:
        raise InvalidAction("expected the sole top-level key 'action'")
    return validate_action(obj["action"])


def request_payload(cfg, task, obs, history, step):
    observation = {"task": task, "control_step": step, "max_control_steps": cfg.max_steps,
                   "proprioception": obs.state, "recent_history": history}
    content = [{"type": "text", "text": json.dumps(observation, allow_nan=False)}]
    for camera in CAMERAS:
        content.extend([
            {"type": "text", "text": f"Current camera: {camera}"},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(obs.images[camera]).decode("ascii"),
                "detail": "original"}},
        ])
    return {"model": cfg.model, "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}],
            "thinking": {"type": "enabled"}, "reasoning_effort": cfg.reasoning_effort,
            "response_format": {"type": "json_object"},
            "max_tokens": cfg.max_tokens, "stream": False}
