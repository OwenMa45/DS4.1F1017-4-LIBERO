import json
import math
from pathlib import Path

from .artifacts import write_json
from .protocol import CAMERAS


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = fraction * (len(values) - 1)
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def distribution(values):
    return {"count": len(values), "mean": sum(values) / len(values) if values else None,
            "p50": percentile(values, .50), "p95": percentile(values, .95),
            "max": max(values) if values else None, "total": sum(values)}


def load_events(path):
    if not path.exists():
        return [], []
    lines = path.read_text(encoding="utf-8").splitlines()
    events, warnings = [], []
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except ValueError:
            if index != len(lines) - 1:
                raise ValueError(f"Corrupt event log at line {index + 1}")
            warnings.append("Ignored incomplete final log line after interruption")
    return events, warnings


def make_replay(root, events, fps):
    from PIL import Image, ImageDraw
    observations = [e for e in events if e.get("event") == "observation" and type(e.get("frame")) is int]
    frames = []
    try:
        for entry in observations:
            cameras = []
            for camera in CAMERAS:
                with Image.open(root / entry["frames"][camera]["path"]) as source:
                    cameras.append(source.convert("RGB"))
            width, height = cameras[0].size
            canvas = Image.new("RGB", (width * 2, height + 24), "white")
            draw = ImageDraw.Draw(canvas)
            for i, camera in enumerate(cameras):
                canvas.paste(camera, (width * i, 24))
                draw.text((width * i + 4, 5), f"{CAMERAS[i]} | step {entry['frame']}", fill="black")
                camera.close()
            frames.append(canvas)
        if not frames:
            return None
        path = root / "replay.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0)
        return path.name
    finally:
        for frame in frames:
            frame.close()


def build_report(path, replay=True):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result_path = root / "result.json"
    result = (json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else
              {"status": "incomplete", "scored": False, "task_success": None})
    events, warnings = load_events(root / "events.jsonl")
    attempts = [e for e in events if e.get("event") == "api_attempt"]
    completed_attempt_ids = {e.get("attempt") for e in attempts}
    unfinished_attempts = [e for e in events if e.get("event") == "api_started"
                           and e.get("attempt") not in completed_attempt_ids]
    request_count = len(attempts) + len(unfinished_attempts)
    decisions = [e for e in events if e.get("event") == "decision"]
    model_outputs = [e for e in decisions if e.get("valid") or e.get("error_status") == "invalid_output"]
    valid = sum(bool(e.get("valid")) for e in model_outputs)
    steps = [e for e in events if e.get("event") == "step"]
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
             "reasoning_tokens": 0, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}
    known_usage = 0
    for attempt in attempts:
        raw = attempt.get("usage")
        if not isinstance(raw, dict):
            continue
        known_usage += all(type(raw.get(k)) is int and raw[k] >= 0
                           for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
        for key in usage:
            value = raw.get(key)
            if key == "reasoning_tokens":
                value = (raw.get("completion_tokens_details") or {}).get("reasoning_tokens", value)
            if type(value) is int and value >= 0:
                usage[key] += value
    summary = {"schema_version": 1, "command": manifest["command"], **result,
               "protocol": manifest["config"], "config_sha256": manifest["config_sha256"],
               "source_sha256": manifest["source_sha256"],
               "environment_sha256": manifest.get("environment_sha256"),
               "scope": ("single benchmark episode; use the frozen plan for aggregate statistics"
                         if manifest["config"].get("evaluation_scope") == "benchmark"
                         else "single episode diagnostic; not an aggregate LIBERO benchmark score"),
               "http_requests": request_count, "completed_http_requests": len(attempts),
               "unfinished_http_attempts": [e["attempt"] for e in unfinished_attempts],
               "decision_requests": len(decisions),
               "model_outputs": len(model_outputs), "valid_model_outputs": valid,
               "action_validity_rate": valid / len(model_outputs) if model_outputs else None,
               "executed_action_count": len(steps),
               "actions_per_valid_decision": distribution([
                   len(e["actions"]) if "actions" in e else 1 for e in model_outputs if e.get("valid")]),
               "truncated_outputs": sum(e.get("finish_reason") == "length" for e in attempts),
               "finish_reasons": {reason: sum(e.get("finish_reason") == reason for e in attempts)
                                  for reason in sorted({e["finish_reason"] for e in attempts
                                                        if isinstance(e.get("finish_reason"), str)})},
               "system_fingerprints": sorted({e["system_fingerprint"] for e in attempts
                                              if isinstance(e.get("system_fingerprint"), str)}),
               "reported_usage": usage, "usage_available_attempts": known_usage,
               "usage_incomplete": known_usage != request_count,
               "usage_note": "Reasoning is a subset of completion tokens; unknown usage is not zero cost.",
               "returned_models": sorted({e["returned_model"] for e in attempts if isinstance(e.get("returned_model"), str)}),
               "http_latency_seconds": distribution([e["elapsed_seconds"] for e in attempts]),
               "decision_latency_seconds": distribution([e["elapsed_seconds"] for e in decisions]),
               "replay_timing": "20 Hz simulated control time; API waiting time is excluded",
               "warnings": warnings}
    if replay:
        try:
            summary["replay"] = make_replay(root, events, manifest["config"]["control_freq"])
        except Exception as exc:
            summary["warnings"].append(f"Replay generation failed; PNGs retained: {type(exc).__name__}: {exc}")
    write_json(root / "summary.json", summary)
    return summary
