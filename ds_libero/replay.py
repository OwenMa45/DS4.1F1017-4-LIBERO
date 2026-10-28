"""Replay one recorded observation-to-action request, without importing LIBERO."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .artifacts import Store, write_json
from .client import Client
from .config import Config
from .errors import ExperimentError, InvalidAction
from .protocol import CAMERAS, Observation, parse_action, parse_actions, request_payload
from .report import build_report, load_events


def checked_file(root, relative, expected_hash=None):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ExperimentError("Source artifact path escapes the episode directory")
    data = path.read_bytes()
    if expected_hash and hashlib.sha256(data).hexdigest() != expected_hash:
        raise ExperimentError(f"Source artifact hash mismatch: {relative}")
    return data


def prepare_replay(directory, step, max_tokens, effort=None):
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cfg = Config.from_dict(manifest["config"])
    if cfg.fingerprint != manifest["config_sha256"]:
        raise ExperimentError("Source configuration fingerprint mismatch")
    events, warnings = load_events(root / "events.jsonl")
    if warnings:
        raise ExperimentError("Source events are incomplete; cannot verify an exact replay")
    attempts = [e for e in events if e.get("event") == "api_attempt" and e.get("step") == step]
    if not attempts:
        raise ExperimentError(f"No recorded API request at control step {step}")
    recorded = attempts[0]
    if recorded.get("request_file"):
        encoded = checked_file(root, recorded["request_file"], recorded["request_sha256"])
        payload = json.loads(encoded)
    else:
        if cfg.protocol_version != "ds-libero-direct-v1":
            raise ExperimentError("Legacy reconstruction supports v1 only")
        decisions = [e for e in events if e.get("event") == "decision" and e.get("step") == step]
        if len(decisions) != 1:
            raise ExperimentError("Expected one recorded decision for this step")
        decision = decisions[0]
        images = {camera: checked_file(root, decision["frames"][camera]["path"],
                                       decision["frames"][camera]["sha256"]) for camera in CAMERAS}
        obs = Observation(decision["proprioception"], images)
        payload = request_payload(cfg, manifest["environment"]["task"], obs, decision["recent_history"], step)
        prompt = (root / "prompt.txt").read_text(encoding="utf-8")
        if hashlib.sha256(prompt.encode()).hexdigest() != manifest["prompt_sha256"]:
            raise ExperimentError("Source prompt fingerprint mismatch")
        payload["messages"][0]["content"] = prompt
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        if hashlib.sha256(encoded).hexdigest() != recorded["request_sha256"]:
            raise ExperimentError("Reconstructed request does not match original SHA256; refusing approximate replay")
    if max_tokens <= 0 or max_tokens > 393216 or effort not in (None, "low", "high"):
        raise ExperimentError("Invalid replay max_tokens or reasoning effort")
    changes = {"max_tokens": {"before": payload["max_tokens"], "after": max_tokens}}
    payload["max_tokens"] = max_tokens
    if effort is not None:
        changes["reasoning_effort"] = {"before": payload["reasoning_effort"], "after": effort}
        payload["reasoning_effort"] = effort
    # Original protocol label is retained; diagnostic overrides are explicit in the manifest.
    cfg = replace(cfg, max_tokens=max_tokens, reasoning_effort=effort or cfg.reasoning_effort)
    return cfg, payload, {"source_directory": str(root), "source_step": step,
                          "source_request_sha256": recorded["request_sha256"],
                          "verified_exact_original_request": True, "overrides": changes}


def replay_request(directory, step, max_tokens, out, effort=None, dry_run=False):
    cfg, payload, provenance = prepare_replay(directory, step, max_tokens, effort)
    store = Store(out, "replay-request", cfg)
    prompt = payload["messages"][0]["content"]
    (store.path / "prompt.txt").write_text(prompt, encoding="utf-8")
    store.manifest.update(replay=provenance, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    write_json(store.path / "manifest.json", store.manifest)
    result = {"status": "incomplete", "scored": False, "task_success": None,
              "model_actions_executed": 0, "replay": provenance}
    decision = {"event": "decision", "step": step, "valid": False, "elapsed_seconds": 0.}
    import time
    started = time.perf_counter()
    try:
        if dry_run:
            store.request(step, json.dumps(payload, allow_nan=False, separators=(",", ":")).encode())
            result["status"] = "prepared"
        else:
            reply = Client(cfg, store).complete(payload, step)
            try:
                choice = reply["choices"][0]
                finish, content = choice["finish_reason"], choice["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise InvalidAction("API response has no complete assistant choice") from exc
            decision["finish_reason"] = finish
            action = (parse_action(content, finish) if cfg.protocol_version == "ds-libero-direct-v1"
                      else parse_actions(content, finish, min(cfg.action_horizon, cfg.max_steps - step)))
            decision.update(valid=True)
            decision["action" if cfg.protocol_version == "ds-libero-direct-v1" else "actions"] = action
            result.update(status="pass", proposed_action=action)
    except ExperimentError as exc:
        result.update(status=exc.status, reason=str(exc))
        decision.update(error_status=exc.status, error=str(exc))
    except KeyboardInterrupt:
        result.update(status="interrupted", reason="Interrupted by user")
    except Exception as exc:
        result.update(status="runtime_error", reason=f"{type(exc).__name__}: {exc}")
        decision.update(error_status="runtime_error", error=result["reason"])
    finally:
        decision["elapsed_seconds"] = time.perf_counter() - started
        if not dry_run:
            store.event(decision)
        store.finish(result)
    return build_report(store.path, replay=False)
