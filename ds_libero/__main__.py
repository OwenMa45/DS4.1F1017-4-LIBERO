import argparse
import json
import os
import sys
import time
from pathlib import Path

from .artifacts import Store, verify_gate
from .calibration import calibrate
from .client import Client, ModelPolicy
from .config import Config
from .errors import ExperimentError
from .report import build_report
from .runner import run_episode


def parser():
    p = argparse.ArgumentParser(description="Pinned DeepSeek / LIBERO single-episode experiment")
    commands = p.add_subparsers(dest="command", required=True)
    for command in ("calibrate", "api-check", "run"):
        child = commands.add_parser(command)
        child.add_argument("--config", type=Path, help="JSON config; defaults to the fixed v1 protocol")
        child.add_argument("--out", type=Path, required=True, help="New artifact directory (must not already exist)")
        if command in ("api-check", "run"):
            child.add_argument("--calibration", type=Path, required=True, help="Passing calibration directory")
        if command == "run":
            child.add_argument("--api-check", type=Path, required=True, help="Passing API-check directory")
    report = commands.add_parser("report")
    report.add_argument("directory", type=Path)
    report.add_argument("--no-replay", action="store_true", help="Summarize using only the Python standard library")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "report":
        try:
            summary = build_report(args.directory, replay=not args.no_replay)
        except (OSError, ValueError, KeyError) as exc:
            print(f"Report failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    try:
        cfg = Config.load(args.config)
        if args.command in ("api-check", "run") and not os.environ.get("DEEPSEEK_API_KEY", "").strip():
            raise ExperimentError("DEEPSEEK_API_KEY is not set; enter it in the remote shell")
        # Check gates' existence before spending time constructing the simulator.
        for gate in (getattr(args, "calibration", None), getattr(args, "api_check", None)):
            if gate is not None and not (gate / "result.json").is_file():
                raise ExperimentError(f"Gate artifacts not found: {gate}")
        store = Store(args.out, args.command, cfg)
    except (OSError, ValueError, ExperimentError) as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 2
    started = time.perf_counter()
    env = None
    result = {"status": "incomplete", "scored": False, "task_success": None}
    try:
        from .environment import LiberoEnvironment
        env = LiberoEnvironment(cfg)
        store.environment(env.provenance)
        print(f"Task: {env.task}\nArtifacts: {store.path}", flush=True)
        if args.command == "calibrate":
            result = calibrate(env, store)
        else:
            verify_gate(args.calibration, "calibrate", cfg, env.provenance)
            if args.command == "run":
                verify_gate(args.api_check, "api-check", cfg, env.provenance)
            policy = ModelPolicy(cfg, env.task, Client(cfg, store), store)
            if args.command == "api-check":
                obs = env.reset()
                refs = store.observe(0, obs)
                if env.check_success():
                    raise ExperimentError("Initial state already succeeds")
                action = policy.act(obs, [], 0, refs)
                result = {"status": "pass", "scored": False, "task_success": None,
                          "proposed_action": action, "model_actions_executed": 0,
                          "note": "Image request and JSON contract accepted; this is not a vision-accuracy or task-success test."}
            else:
                result = run_episode(env, policy, store, cfg)
    except ExperimentError as exc:
        result.update(status=exc.status, reason=str(exc))
    except KeyboardInterrupt:
        result.update(status="interrupted", reason="Interrupted by user")
    except Exception as exc:
        result.update(status="runtime_error", reason=f"{type(exc).__name__}: {exc}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["command_wall_seconds"] = time.perf_counter() - started
        store.finish(result)
    summary = build_report(store.path, replay=args.command == "run")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"RESULT: {result['status']} | {store.path / 'summary.json'}", flush=True)
    return 0 if result["status"] in ("pass", "success", "failure", "invalid_output") else 2


if __name__ == "__main__":
    raise SystemExit(main())
