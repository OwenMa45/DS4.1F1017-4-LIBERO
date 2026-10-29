import csv
import json
import math
from pathlib import Path

from .artifacts import write_json
from .accounting import TOKEN_FIELDS, account_attempts, aggregate_cost
from .config import digest
from .errors import ExperimentError
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
    ledger, accounting = account_attempts(attempts, unfinished_attempts, manifest.get("api_pricing"))
    http_times = [e["elapsed_seconds"] for e in attempts if e.get("elapsed_seconds") is not None]
    decision_times = [e["elapsed_seconds"] for e in decisions if e.get("elapsed_seconds") is not None]
    backoff = sum(e["elapsed_seconds"] for e in events if e.get("event") == "retry_backoff")
    identity = {k: manifest["config"][k] for k in ("suite", "task_id", "task_order_index", "init_state_id", "seed")}
    identity["task_name"] = manifest.get("environment", {}).get("task_name")
    summary = {"schema_version": 2, "command": manifest["command"], **result,
               "task_identity": identity,
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
               **accounting, "api_pricing": manifest.get("api_pricing"),
               "usage_note": "Reasoning is a subset of completion tokens; unknown usage is not zero cost.",
               "returned_models": sorted({e["returned_model"] for e in attempts if isinstance(e.get("returned_model"), str)}),
               "http_latency_seconds": distribution(http_times),
               "http_latency_incomplete": len(http_times) != request_count,
               "decision_latency_seconds": distribution(decision_times),
               "decision_latency_incomplete": bool(unfinished_attempts) or len(decision_times) != len(decisions),
               "retry_backoff_seconds": backoff,
               "latency_note": "Task execution wall time includes API, backoff, simulation and logging after reset. Command wall also includes setup/reset/cleanup, excludes report/GIF generation. HTTP/decision times overlap task time; do not add them. Replay has no task execution latency.",
               "replay_timing": "20 Hz simulated control time; API waiting time is excluded",
               "warnings": warnings}
    if replay:
        try:
            summary["replay"] = make_replay(root, events, manifest["config"]["control_freq"])
        except Exception as exc:
            summary["warnings"].append(f"Replay generation failed; PNGs retained: {type(exc).__name__}: {exc}")
    elif manifest["command"] == "run" and (root / "replay.gif").is_file():
        summary["replay"] = "replay.gif"
    write_json(root / "summary.json", summary)
    write_json(root / "api_calls.json", {"task_identity": identity, "api_pricing": manifest.get("api_pricing"),
                                        "api_cost": accounting["api_cost"], "attempts": ledger})
    rows = [{**identity, **r, "estimated_cost_min_cny": (r["estimated_cost_range_cny"] or [None, None])[0],
             "estimated_cost_max_cny": (r["estimated_cost_range_cny"] or [None, None])[1]} for r in ledger]
    fields = (*identity, "attempt", "step", "retry", "started_utc", "finished_utc", "response_created",
              "response_id", "completed", "status_code", "finish_reason", "returned_model", "system_fingerprint",
              "http_latency_seconds", *TOKEN_FIELDS, "usage_complete", "estimated_cost_min_cny", "estimated_cost_max_cny", "error")
    with (root / "api_calls.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary


def build_campaign_report(path, expected_config=None, expected_source=None):
    """Include paid preflight even when it prevented a scored episode from starting."""
    root = Path(path)
    summaries = {}
    configs, sources, identities = set(), set(), set()
    for phase, command in (("calibration", "calibrate"), ("api-check", "api-check"), ("episode", "run")):
        directory = root / phase
        if not (directory / "manifest.json").is_file():
            continue
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["command"] != command or digest(manifest["config"]) != manifest["config_sha256"]
                or (expected_config is not None and manifest["config_sha256"] != expected_config)
                or (expected_source is not None and manifest["source_sha256"] != expected_source)):
            raise ExperimentError(f"Campaign {phase} differs from the expected experiment")
        configs.add(manifest["config_sha256"])
        sources.add(manifest["source_sha256"])
        summaries[phase] = build_report(directory, replay=False)
        identities.add(digest({k: v for k, v in summaries[phase]["task_identity"].items() if k != "task_name"}))
    if len(configs) > 1 or len(sources) > 1 or len(identities) > 1:
        raise ExperimentError("Campaign phases use different configs, code or task identities")
    all_summaries = list(summaries.values())
    usage = {k: sum(s["reported_usage"][k] for s in all_summaries) for k in TOKEN_FIELDS}
    availability = {k: sum(s["usage_field_available_attempts"][k] for s in all_summaries) for k in TOKEN_FIELDS}
    cost = aggregate_cost([s["api_cost"] for s in all_summaries],
                          pricing_available=all(s["api_cost"]["pricing_available"] for s in all_summaries))
    wall = [s.get("command_wall_seconds") for s in all_summaries]
    report = {"schema_version": 1, "scope": "all recorded phases, including failed preflight and retries",
              "task_identity": all_summaries[-1]["task_identity"] if all_summaries else None,
              "phases": {p: {"status": s["status"], "http_requests": s["http_requests"],
                             "reported_usage": s["reported_usage"], "api_cost": s["api_cost"],
                             "usage_field_available_attempts": s["usage_field_available_attempts"],
                             "command_wall_seconds": s.get("command_wall_seconds")}
                         for p, s in summaries.items()},
              "http_requests": sum(s["http_requests"] for s in all_summaries),
              "reported_usage": usage, "usage_incomplete": any(s["usage_incomplete"] for s in all_summaries),
              "usage_field_available_attempts": availability,
              "api_cost": cost,
              "api_wait_seconds": sum(s["http_latency_seconds"]["total"] for s in all_summaries),
              "http_latency_incomplete": any(s["http_latency_incomplete"] for s in all_summaries),
              "recorded_command_wall_seconds": sum(w for w in wall if w is not None),
              "command_wall_incomplete": any(w is None for w in wall),
              "note": "Command wall sum excludes shell gaps and report/GIF generation; episode task latency is in episode/summary.json. No score is assigned here."}
    write_json(root / "campaign_summary.json", report)
    return report
