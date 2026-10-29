import argparse
import json
import os
import sys
import time
from pathlib import Path

from .artifacts import Store, verify_gate
from .calibration import calibrate
from .client import Client, ModelPolicy
from .config import Config, SUITE_HORIZONS
from .errors import ExperimentError
from .report import build_report
from .runner import run_episode


def parser():
    p = argparse.ArgumentParser(description="Reproducible DeepSeek / LIBERO direct-policy experiments")
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
    replay = commands.add_parser("replay-request", help="Verified saved request; never starts the simulator")
    replay.add_argument("directory", type=Path)
    replay.add_argument("--step", type=int, required=True)
    budget = replay.add_mutually_exclusive_group()
    budget.add_argument("--max-tokens", type=int, default=None,
                        help="Optional explicit cap; default omits the API field (provider limit still applies)")
    budget.add_argument("--no-token-limit", dest="max_tokens", action="store_const", const=None)
    replay.add_argument("--reasoning-effort", choices=("low", "high"))
    replay.add_argument("--timeout-seconds", type=int, default=1800,
                        help="HTTP socket timeout, independent of token budget (default 1800)")
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--dry-run", action="store_true", help="Verify and save the payload without calling API")
    replay.add_argument("--prompt-api-key", action="store_true",
                        help="If the environment key is missing, read it without echo in this terminal; never save it")
    plan = commands.add_parser("plan", help="Predeclare a sample; no simulator or API calls")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--suite", choices=tuple(SUITE_HORIZONS), required=True)
    plan.add_argument("--task-ids", type=int, nargs="+", required=True)
    plan.add_argument("--init-state-ids", type=int, nargs="+", required=True)
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--out", type=Path, required=True)
    selected = commands.add_parser("run-plan-episode", help="Run one explicitly selected trial, including paid API calls")
    selected.add_argument("directory", type=Path)
    selected.add_argument("--episode", type=int, required=True, help="Zero-based plan episode index")
    aggregate = commands.add_parser("report-plan", help="Summarize the fixed sample without model calls")
    aggregate.add_argument("directory", type=Path)
    campaign = commands.add_parser("report-campaign", help="All phase costs, including failed preflight checks")
    campaign.add_argument("directory", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "report-campaign":
        from .report import build_campaign_report
        try:
            summary = build_campaign_report(args.directory)
        except (OSError, ValueError, KeyError, ExperimentError) as exc:
            print(f"Report failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command in ("plan", "run-plan-episode", "report-plan"):
        from .benchmark import create_plan, run_plan_episode, report_plan
        try:
            if args.command == "plan":
                plan = create_plan(Config.load(args.config), args.suite, args.task_ids,
                                   args.init_state_ids, args.seed, args.out)
                print(f"Prepared {len(plan['episodes'])} episodes; no API calls. Plan: {args.out / 'plan.json'}")
            elif args.command == "run-plan-episode":
                return run_plan_episode(args.directory, args.episode)
            else:
                report = report_plan(args.directory)
                print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, ensure_ascii=False, indent=2))
        except (OSError, ValueError, KeyError, ExperimentError) as exc:
            print(f"{args.command} failed: {exc}", file=sys.stderr)
            return 2
        return 0
    if args.command == "replay-request":
        from .replay import replay_request
        try:
            summary = replay_request(args.directory, args.step, args.max_tokens, args.out,
                                     args.reasoning_effort, args.dry_run, args.timeout_seconds,
                                     args.prompt_api_key)
        except (OSError, ValueError, KeyError, ExperimentError) as exc:
            print(f"Replay failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"RESULT: {summary['status']} | {args.out / 'summary.json'}", flush=True)
        return 0 if summary["status"] in ("pass", "prepared", "invalid_output") else 2
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
