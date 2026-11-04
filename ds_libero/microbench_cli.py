"""CLI for the public microbench API; all model imports are deferred to run."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .errors import ExperimentError
from .microbench import MicroBench


def add_arguments(parser):
    commands = parser.add_subparsers(dest="bench_command", required=True)
    commands.add_parser("list", help="List the five fixed tasks; offline")
    plan = commands.add_parser("plan", help="Freeze a new experiment; offline, does not import the policy")
    plan.add_argument("--out", type=Path, required=True)
    profile = plan.add_mutually_exclusive_group()
    profile.add_argument("--config", type=Path, help="Existing v2 API profile or external profile")
    profile.add_argument("--max-steps", type=int, default=220, help="External profile control steps (default: 220)")
    plan.add_argument("--policy", help="External factory reference: module:callable")
    plan.add_argument("--policy-id", help="Experiment policy label")
    plan.add_argument("--revision", help="External policy revision: code, prompt and checkpoint identifier")
    plan.add_argument("--policy-config", type=Path, help="Adapter JSON object; do not put credentials here")
    run = commands.add_parser("run", help="Run unstarted tasks; starts simulator and may call paid APIs")
    run.add_argument("directory", type=Path)
    run.add_argument("--episode", type=int, help="Explicitly select one unstarted task, index 0..4")
    run.add_argument("--no-replay", action="store_true")
    report = commands.add_parser("report", help="Rebuild scores from saved artifacts; offline")
    report.add_argument("directory", type=Path)


def dispatch(args):
    try:
        if args.bench_command == "list":
            result = [asdict(task) for task in MicroBench.tasks]
        elif args.bench_command == "plan":
            bench = MicroBench(args.config) if args.config else MicroBench(max_steps=args.max_steps)
            options = (json.loads(args.policy_config.read_text(encoding="utf-8"))
                       if args.policy_config else None)
            plan = bench.prepare(args.out, policy_id=args.policy_id, revision=args.revision,
                                 adapter=args.policy, policy_config=options)
            result = {"plan": str(args.out / "plan.json"), "tasks": len(plan["episodes"]),
                      "protocol_sha256": plan["microbench"]["protocol_sha256"], "model_calls": 0}
        elif args.bench_command == "run":
            result = MicroBench.run(args.directory, episode=args.episode, replay=not args.no_replay)
        else:
            result = MicroBench.report(args.directory)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result.get("batch_exit_code", 0) if isinstance(result, dict) else 0
    except (OSError, ValueError, TypeError, KeyError, ExperimentError) as exc:
        print(f"Microbench failed: {exc}", file=sys.stderr)
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description="LIBERO-Five v1 microbench")
    add_arguments(parser)
    return dispatch(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
