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
    geometry: dict = None


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


def system_prompt(cfg):
    if cfg.protocol_version == "ds-libero-direct-v1":
        return SYSTEM_PROMPT
    prompt = SYSTEM_PROMPT.split('Return exactly one json object')[0] + f'''Return only a json object with the sole key "actions", containing 1 to {cfg.action_horizon}
seven-number actions, for example {{"actions":[[0,0,0,0,0,0,-1]]}}.
The first three numbers are normalized WORLD-frame position deltas in [-1,1],
scaled to 0.05 metres. The next three are WORLD-frame axis-angle rotation-vector
deltas in [-1,1], scaled to 0.5 radians; they are not Euler angles.
The last number is exactly -1 (open), 1 (close), or 0 (hold gripper command).
Each action sets a NEW target relative to the actual EEF pose at that control step
and runs for 0.05 seconds. Repeating a delta MOVES the target again; it does not hold
a fixed destination. Actual motion in a step can be much smaller than its target
offset because the controller needs time to respond. Zero deltas hold near the
current pose, while the gripper command can continue closing or opening.
The entire sequence runs before another API call, except on task success or the
control-step limit. Choose short sequences near contact or when uncertain.
Remaining step budget limits the allowed number of actions in this response.
Only the simulator determines success. Waiting for this API never advances time.
Images are {cfg.image_size} by {cfg.image_size} pixels, already upright. Historical
images, if supplied, are labeled with their control step and matching state.
Camera calibration, if supplied, uses optical axes +X right, +Y down, +Z forward;
T_world_from_camera maps optical camera coordinates to world metres. The wrist
camera pose changes with the robot, so use the calibration paired with each image.
No object locations, depth, segmentation, demonstrations or scripted skills are
provided. Use measured motion and visual feedback; do not assume a fixed mapping
between world axes and image directions. Return actions, no explanations or keys.
'''
    if cfg.grasp_feedback:
        prompt += '''\nGripper feedback protocol:
Panda gripper commands change an actuator target gradually. +1 is NOT an instant
grasp. 0 preserves the actuator target; it does NOT keep incrementing it closed.
Opening and closing require multiple simulation steps, often more than one chunk.
Use gripper-only sequences (zero arm deltas) while closing, then inspect the new
joint separation and images. If still closing, you may choose another such sequence.
Do not combine approach, closure, lift and transport into a single blind sequence.
Align the fingerpads with the visible grasp region before closing; the wrist camera
image centre is not the grasp point. robot_gripper_geometry gives only your own
fingerpad centres and their projected pixels. It does not locate target objects.
Avoid descending far while still misaligned; use short corrections near contact.
After closure, choose a small vertical test lift and inspect both views before
transport. A nonzero separation alone cannot establish a grasp; an almost empty
closure or an object staying behind during lift is evidence to reassess/re-align.
Use measured EEF displacement instead of summing nominal deltas as actual motion.
Do not claim a grasp or continue transport merely because you commanded close.
'''
    if cfg.gripper_feedback_barrier:
        prompt = prompt.replace(
            "The entire sequence runs before another API call, except on task success or the\ncontrol-step limit.",
            "A validated prefix runs before another API call, stopping at the feedback boundary\ndescribed below, task success or the control-step limit.")
        prompt += '''\nExecution uses a feedback boundary: only the prefix before the first gripper
open/close reversal or switch between arm motion and gripper-only commands runs.
The remaining suffix is discarded. You receive a fresh observation and choose
again. Prefer one phase per response. No automatic grasp or recovery is executed.
'''
    if cfg.geometry_contract:
        prompt = prompt.replace(
            "Proprioception is end-effector position in world metres, end-effector quaternion\nin [x,y,z,w] order,",
            "Proprioception uses the Panda grip_site for BOTH world position and orientation\n(unit quaternion in [x,y,z,w] order),")
        prompt = prompt.replace(
            "Each action sets a NEW target relative to the actual EEF pose at that control step\nand runs for 0.05 seconds.",
            "Each action sets a NEW position target relative to the actual grip_site position\nand runs for 0.05 seconds. Nonzero rotation deltas left-multiply the actual site\nrotation about WORLD axes; zero rotation deltas retain the previous orientation goal.")
        prompt += '''\nGeometry contract (camera_geometry.control_frame):
The controlled point is the Panda grip_site near the fingerpads, NOT the mounting
flange. R_world_from_control maps site-local vectors into world coordinates.
The adapter corrects the legacy hand-body quaternion to the controller site frame;
legacy_hand_quaternion_xyzw is retained only for audit, not the action reference.

Gripper geometry (camera_geometry.robot_gripper_geometry):
points_control_frame_m and points_world_m give the SAME robot geom centres in two
frames. p_world = eef_position_world_m + R_world_from_control * p_control.
The pads oppose along site-local X; site-local +Z extends toward the fingertips.
Use the measured pad_midpoint, jaw_axis_world_left_to_right and matching pixels_uv
to align with the visible grasp region. Do not add a flange-length offset to the
grip_site position. Joint separation, pad-centre distance and inner-face clearance
are different quantities; these fields do not detect contacts or a successful grasp.

Camera geometry (camera_geometry.cameras):
K uses pixels of each individual upright image, not a side-by-side replay mosaic.
For a world point p, optical coordinates are R^T*(p-t), using R,t from
T_world_from_camera. With optical Z>0: u=fx*X/Z+cx, v=fy*Y/Z+cy.
position_control_frame_m, R_control_from_camera_optical and view_direction_world
describe the actual camera mounting/view; do not assume the wrist image centre is
the grasp point or import another robot's camera tilt. Use each historical image's
own calibration. A pixel gives a ray, not depth; compare both views and observed
motion. Pixel references outside the image are off-screen; null is behind the camera.
Choose corrections only in the existing 7D world-delta format, then check the new
observation. Geometry references are not object locations or absolute action targets.
'''
    if cfg.action_horizon > 8:
        prompt += f'''\nAction chunk scheduling:
Choose 1..{cfg.action_horizon} explicit 7D actions per response (at most
{cfg.action_horizon / cfg.control_freq:g} simulated seconds). This is a maximum, not a required length.
Each array entry executes once, in order; repeated commands need repeated entries.
No new reasoning or visual correction occurs inside the executed chunk. Its end
provides fresh images, measured state and executed-action feedback for your next
decision. Use longer chunks for predictable free-space motion or gripper-only
actuation within one phase; use shorter chunks for alignment, contact and test
lifts. Never pad a chunk to its maximum length when fresh feedback is needed.
Respect max_actions_this_response, which shrinks near the total control-step limit.
'''
    return prompt


def validate_actions(actions, limit):
    if not isinstance(actions, list) or not 1 <= len(actions) <= limit:
        raise InvalidAction(f"actions must contain 1..{limit} actions")
    # Validate the whole sequence before executing even its first element.
    return [validate_action(action) for action in actions]


def parse_actions(content, finish_reason, limit):
    if finish_reason != "stop":
        raise InvalidAction(f"incomplete or unsupported response: finish_reason={finish_reason!r}")
    if not isinstance(content, str) or not content.strip():
        raise InvalidAction("model returned empty/non-text content")
    try:
        obj = json.loads(content, object_pairs_hook=_unique_object)
    except (ValueError, TypeError) as exc:
        raise InvalidAction("model content is not a complete JSON object") from exc
    if not isinstance(obj, dict) or set(obj) != {"actions"}:
        raise InvalidAction("expected the sole top-level key 'actions'")
    return validate_actions(obj["actions"], limit)


def request_payload(cfg, task, obs, history, step, visual_history=(), control_feedback=None):
    if cfg.geometry_contract:
        from .errors import ExperimentError
        from .geometry_contract import SCHEMA
        for candidate in [obs] + [old_obs for _, old_obs in visual_history]:
            if (not candidate.geometry
                    or candidate.geometry.get("control_frame", {}).get("schema") != SCHEMA):
                raise ExperimentError("Current and historical observations require the checked Panda grip-site geometry")
    observation = {"task": task, "control_step": step, "max_control_steps": cfg.max_steps,
                   "proprioception": obs.state, "recent_history": history}
    if cfg.protocol_version == "ds-libero-direct-v2":
        observation["max_actions_this_response"] = min(cfg.action_horizon, cfg.max_steps - step)
        if cfg.camera_geometry:
            if not obs.geometry:
                from .errors import ExperimentError
                raise ExperimentError("Camera geometry requested but missing from observation")
            observation["camera_geometry"] = obs.geometry
        if cfg.grasp_feedback:
            if control_feedback is None:
                from .feedback import decision_feedback
                control_feedback = decision_feedback(obs, step, None, cfg.control_freq)
            observation["control_feedback"] = control_feedback
    content = [{"type": "text", "text": json.dumps(observation, allow_nan=False)}]
    for old_step, old_obs in visual_history:
        state = {"historical_control_step": old_step, "proprioception": old_obs.state}
        if cfg.camera_geometry:
            state["camera_geometry"] = old_obs.geometry
        content.append({"type": "text", "text": json.dumps(state, allow_nan=False)})
        _append_images(content, old_obs, f"Historical step {old_step}")
    _append_images(content, obs, "Current")
    payload = {"model": cfg.model, "messages": [
                {"role": "system", "content": system_prompt(cfg)},
                {"role": "user", "content": content}],
            "thinking": {"type": "enabled"}, "reasoning_effort": cfg.reasoning_effort,
            "response_format": {"type": "json_object"},
            "max_tokens": cfg.max_tokens, "stream": False}
    # Keep integer-budget v1 serialization byte-for-byte compatible with old replays.
    if cfg.max_tokens is None:
        del payload["max_tokens"]
    return payload


def _append_images(content, obs, label):
    for camera in CAMERAS:
        content.extend([
            {"type": "text", "text": f"{label} camera: {camera}"},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(obs.images[camera]).decode("ascii"),
                "detail": "original"}},
        ])
