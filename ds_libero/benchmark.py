"""Predeclare trials and report only that fixed sample; no implicit API calls."""

import csv
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

from .artifacts import source_fingerprint, utc_now, write_json
from .accounting import TOKEN_FIELDS, aggregate_cost, pricing_snapshot
from .config import Config, SUITE_HORIZONS, digest
from .errors import ExperimentError
from .report import build_campaign_report, build_report, distribution, load_events


def create_plan(base, suite, task_ids, init_state_ids, seed, out):
    if base.protocol_version != "ds-libero-direct-v2":
        raise ExperimentError("Benchmark plans require an explicit v2 profile")
    if suite not in SUITE_HORIZONS:
        raise ExperimentError("Unsupported benchmark suite")
    for label, values, upper in (("task IDs", task_ids, 10), ("initial-state IDs", init_state_ids, 50)):
        if (not values or any(type(v) is not int or not 0 <= v < upper for v in values)
                or len(set(values)) != len(values)):
            raise ExperimentError(f"{label} must be nonempty, unique integers in 0..{upper - 1}")
    entries = []
    for task in task_ids:
        for init_state in init_state_ids:
            cfg = Config.from_dict({**base.to_dict(), "suite": suite, "task_id": task,
                                    "init_state_id": init_state, "seed": seed,
                                    "max_steps": SUITE_HORIZONS[suite], "evaluation_scope": "benchmark"})
            index = len(entries)
            entries.append({"index": index, "config": cfg.to_dict(), "config_sha256": cfg.fingerprint,
                            "config_file": f"configs/episode-{index:04d}.json",
                            "run_directory": f"runs/episode-{index:04d}"})
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    for entry in entries:
        write_json(root / entry["config_file"], entry["config"])
    plan = {"schema_version": 1, "created_utc": utc_now(), "source_sha256": source_fingerprint(),
            "suite": suite, "task_order_index": 0, "seed": seed, "episodes": entries,
            "scope": "predeclared sample; no official benchmark score is implied",
            "execution": "one explicitly selected episode at a time; no API call during plan creation",
            "initial_states_note": "Official state availability is verified by the remote runtime",
            "max_http_requests_per_episode_including_api_check": (base.max_http_requests +
                min(base.max_http_requests, base.max_http_attempts_per_decision))}
    plan["plan_sha256"] = digest(plan)
    write_json(root / "plan.json", plan)
    return plan


def create_five_plan(base, out):
    from .five_tasks import five_configs
    configs = five_configs(base)
    entries = [{"index": i, "config": cfg.to_dict(), "config_sha256": cfg.fingerprint,
                "config_file": f"configs/episode-{i:04d}.json", "run_directory": f"runs/episode-{i:04d}"}
               for i, cfg in enumerate(configs)]
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    for entry in entries:
        write_json(root / entry["config_file"], entry["config"])
    plan = {"schema_version": 2, "created_utc": utc_now(), "source_sha256": source_fingerprint(),
            "suite": "mixed", "task_order_index": 0, "seed": 0, "episodes": entries,
            "scope": "five-task diagnostic with custom control horizon; one initial state per task; not a standard LIBERO benchmark score",
            "execution": "sequential tasks, or one selected episode; no API during plan creation",
            "initial_states_note": "Official initial state 0 for each task, checked by the runtime",
            "max_control_steps_per_episode": base.max_steps,
            "max_http_requests_per_episode_including_api_check": base.max_http_requests + min(base.max_http_requests, base.max_http_attempts_per_decision)}
    plan["plan_sha256"] = digest(plan)
    write_json(root / "plan.json", plan)
    return plan


def child_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ExperimentError("Plan artifact escapes its root directory")
    return path


def load_plan(directory):
    root = Path(directory).resolve()
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if plan.get("schema_version") not in (1, 2) or plan.get("plan_sha256") != digest({
            k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ExperimentError("Plan hash/schema mismatch; create a new plan for changes")
    if not plan.get("episodes"):
        raise ExperimentError("Plan has no episodes")
    identities = set()
    paths = set()
    profiles = set()
    diagnostic = plan["schema_version"] == 2
    for index, entry in enumerate(plan["episodes"]):
        cfg = Config.from_dict(entry["config"])
        identity = (cfg.suite, cfg.task_order_index, cfg.task_id, cfg.init_state_id, cfg.seed)
        if entry["index"] != index or identity in identities or cfg.evaluation_scope != ("diagnostic" if diagnostic else "benchmark"):
            raise ExperimentError("Plan has invalid or repeated episode identities")
        identities.add(identity)
        if ((not diagnostic and cfg.suite != plan["suite"]) or cfg.seed != plan["seed"]
                or cfg.task_order_index != plan["task_order_index"]):
            raise ExperimentError("Episode does not match the declared suite, order and seed")
        varying = {"task_id", "init_state_id"} | ({"suite", "expected_task_name"} if diagnostic else set())
        profiles.add(digest({k: v for k, v in cfg.to_dict().items() if k not in varying}))
        if diagnostic and cfg.max_steps != plan["max_control_steps_per_episode"]:
            raise ExperimentError("Diagnostic horizon differs from the declared plan")
        config_path = child_path(root, entry["config_file"])
        run_path = child_path(root, entry["run_directory"])
        if run_path in paths:
            raise ExperimentError("Plan has repeated run directories")
        paths.add(run_path)
        if cfg.fingerprint != entry["config_sha256"] or Config.load(config_path).fingerprint != cfg.fingerprint:
            raise ExperimentError("Plan configuration changed; create a new plan")
    if len(profiles) != 1:
        raise ExperimentError("A plan must contain one frozen policy configuration")
    if diagnostic:
        from .five_tasks import FIVE_TASKS
        selected = [(e["config"]["suite"], e["config"]["task_id"], e["config"].get("expected_task_name")) for e in plan["episodes"]]
        if selected != list(FIVE_TASKS) or plan["suite"] != "mixed" or any(
                e["config"]["init_state_id"] != 0 or e["config"]["seed"] != 0 for e in plan["episodes"]):
            raise ExperimentError("Five-task selection, names or initial states changed")
    return root, plan


def run_plan_episode(directory, index):
    root, plan = load_plan(directory)
    if "microbench" in plan:
        raise ExperimentError("Use microbench run for this plan so its policy metadata is verified")
    if source_fingerprint() != plan["source_sha256"]:
        raise ExperimentError("Code changed since plan creation; freeze code and create a new plan")
    if not 0 <= index < len(plan["episodes"]):
        raise ExperimentError("Episode index is not in the plan")
    entry = plan["episodes"][index]
    script = Path(__file__).resolve().parent.parent / "scripts" / "run_pilot.sh"
    return subprocess.run(["bash", str(script), str(child_path(root, entry["config_file"])),
                           str(child_path(root, entry["run_directory"]))],
                          cwd=script.parent.parent).returncode


def run_plan_all(directory):
    """Continue only unstarted tasks; never silently retry an already paid trial."""
    root, plan = load_plan(directory)
    if "microbench" in plan:
        raise ExperimentError("Use microbench run for this plan so its policy metadata is verified")
    if source_fingerprint() != plan["source_sha256"]:
        raise ExperimentError("Code changed since plan creation; create a new plan")

    def completed(campaign):
        result = campaign / "episode" / "result.json"
        if result.is_file():
            return json.loads(result.read_text(encoding="utf-8"))["status"] in (
                "success", "failure", "invalid_output", "budget_exhausted")
        result = campaign / "api-check" / "result.json"
        return result.is_file() and json.loads(result.read_text(encoding="utf-8"))["status"] == "invalid_output"

    for entry in plan["episodes"]:
        campaign = child_path(root, entry["run_directory"])
        if completed(campaign):
            print(f"Episode {entry['index']} already recorded; skipping", flush=True)
            continue
        if campaign.exists():
            raise ExperimentError(f"Episode {entry['index']} has incomplete/infrastructure-error artifacts; inspect it or select a later unstarted episode explicitly")
        try:
            code = run_plan_episode(root, entry["index"])
        finally:
            report_plan(root)
        if not completed(campaign):
            return code or 2
    report_plan(root)
    return 0


def wilson(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p, zz = successes / total, z * z
    denominator = 1 + zz / total
    center = (p + zz / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + zz / (4 * total * total)) / denominator
    return [max(0., center - half), min(1., center + half)]


def counts(rows):
    n = len(rows)
    scored = sum(r["scored"] for r in rows)
    successes = sum(r["task_success"] is True for r in rows)
    return {"planned": n, "scored": scored, "successes": successes,
            "unscored": n - scored, "coverage": scored / n if n else None,
            "scored_success_rate": successes / scored if scored else None,
            "scored_wilson_95": wilson(successes, scored),
            "planned_success_bounds": [successes / n, (successes + n - scored) / n] if n else None,
            "status_counts": dict(Counter(r["status"] for r in rows))}


def report_plan(directory):
    root, plan = load_plan(directory)
    rows, api_latencies, runtime_signatures = [], [], set()
    for entry in plan["episodes"]:
        cfg = entry["config"]
        pricing = pricing_snapshot(cfg["model"])
        currency = pricing["currency"] if pricing else "CNY"
        row = {"index": entry["index"], "suite": cfg["suite"], "task_id": cfg["task_id"],
               "init_state_id": cfg["init_state_id"], "seed": cfg["seed"],
               "status": "not_started", "scored": False, "task_success": None,
               "http_requests": 0, "model_outputs": 0, "valid_model_outputs": 0,
               "truncated_outputs": 0, "reported_total_tokens": 0, "usage_incomplete": False,
               "command_wall_seconds": None, "simulated_control_seconds": None,
               "task_execution_wall_seconds": None, "api_wait_seconds": 0., "retry_backoff_seconds": 0.,
               "http_latency_incomplete": False,
               "usage_field_available_attempts": {k: 0 for k in TOKEN_FIELDS},
               "reported_usage": {k: 0 for k in TOKEN_FIELDS}, "api_cost": aggregate_cost([], currency=currency)}
        row.update(task_name=cfg.get("expected_task_name"), max_control_steps=cfg["max_steps"],
                   action_horizon=cfg["action_horizon"], control_steps=None, policy_decisions=None,
                   decision_latency_seconds=None, reason=None)
        campaign = child_path(root, entry["run_directory"])
        episode = campaign / "episode"
        if (episode / "manifest.json").is_file():
            manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
            if (manifest["command"] != "run" or manifest["source_sha256"] != plan["source_sha256"]
                    or manifest["config_sha256"] != entry["config_sha256"]
                    or digest(manifest["config"]) != entry["config_sha256"]):
                raise ExperimentError(f"Episode {entry['index']} differs from the frozen plan")
            environment = manifest.get("environment")
            if environment:
                # Task, BDDL and official state legitimately differ between trials.
                varying = {"task", "task_name", "bddl_sha256", "official_init_state_sha256", "initial_state_count"}
                runtime_signatures.add(digest({k: v for k, v in environment.items() if k not in varying}))
                row["task_name"] = environment.get("task_name")
                if cfg.get("expected_task_name") and row["task_name"] != cfg["expected_task_name"]:
                    raise ExperimentError(f"Episode {entry['index']} task name differs from its plan")
            summary = build_report(episode, replay=False)
            for key in ("status", "scored", "task_success", "http_requests", "model_outputs",
                        "valid_model_outputs", "truncated_outputs", "usage_incomplete",
                        "command_wall_seconds", "simulated_control_seconds", "task_execution_wall_seconds",
                        "retry_backoff_seconds", "reported_usage", "api_cost", "http_latency_incomplete",
                        "usage_field_available_attempts", "control_steps", "policy_decisions", "reason",
                        "decision_latency_seconds", "sam3"):
                if key in summary:
                    row[key] = summary[key]
            row["reported_total_tokens"] = summary["reported_usage"]["total_tokens"]
            row["api_wait_seconds"] = summary["http_latency_seconds"]["total"]
            row["system_fingerprints"] = summary["system_fingerprints"]
            row["summary_warnings"] = summary["warnings"]
            expected_score = row["status"] in ("success", "failure", "invalid_output")
            expected_success = (row["status"] == "success") if expected_score else None
            if row["scored"] is not expected_score or row["task_success"] is not expected_success:
                raise ExperimentError(f"Inconsistent score fields in episode {entry['index']}")
            events, _ = load_events(episode / "events.jsonl")
            api_latencies.extend(e["elapsed_seconds"] for e in events if e.get("event") == "api_attempt"
                                 and e.get("elapsed_seconds") is not None)
        elif campaign.exists():
            row["status"] = "preflight_incomplete"
            for phase in ("api-check", "calibration"):
                result = campaign / phase / "result.json"
                if result.is_file():
                    row["preflight_status"] = {"phase": phase, "status": json.loads(result.read_text(encoding="utf-8"))["status"]}
                    break
        row["campaign"] = (build_campaign_report(campaign, entry["config_sha256"], plan["source_sha256"], currency)
                           if campaign.exists() else None)
        for prefix, cost in (("episode", row["api_cost"]),
                             ("campaign", row["campaign"]["api_cost"] if row["campaign"] and row["campaign"]["phases"]
                              else aggregate_cost([], currency=currency))):
            row[f"{prefix}_known_cost_min"], row[f"{prefix}_known_cost_max"] = cost["known_cost_range"]
            unit = cost["currency"].lower()
            row[f"{prefix}_known_cost_min_{unit}"], row[f"{prefix}_known_cost_max_{unit}"] = cost["known_cost_range"]
            row[f"{prefix}_cost_currency"] = cost["currency"]
            row[f"{prefix}_cost_incomplete"] = cost["incomplete"]
        row["campaign_http_requests"] = row["campaign"]["http_requests"] if row["campaign"] else 0
        row["campaign_reported_total_tokens"] = row["campaign"]["reported_usage"]["total_tokens"] if row["campaign"] else 0
        row["campaign_usage_incomplete"] = row["campaign"]["usage_incomplete"] if row["campaign"] else False
        row.update({f"reported_{k}": v for k, v in row["reported_usage"].items()})
        rows.append(row)
    if len(runtime_signatures) > 1:
        raise ExperimentError("Episodes use different simulator/runtime contracts; report separate plans")
    total = counts(rows)
    def task_key(row):
        return f"{row['suite']}:{row['task_id']}" if plan["schema_version"] == 2 else str(row['task_id'])
    per_task = {task: counts([r for r in rows if task_key(r) == task])
                for task in sorted({task_key(r) for r in rows})}
    for task, statistics in per_task.items():
        trials = [r for r in rows if task_key(r) == task]
        statistics.update(resource_totals(trials))
        statistics.update(suite=trials[0]["suite"], task_id=trials[0]["task_id"], task_name=trials[0].get("task_name"))
    complete = total["scored"] == total["planned"]
    outputs = sum(r["model_outputs"] for r in rows)
    report = {"schema_version": 2, "plan_sha256": plan["plan_sha256"], "source_sha256": plan["source_sha256"],
              "scope": plan["scope"], "complete": complete, **total, "per_task": per_task,
              "macro_task_success_rate": (sum(t["scored_success_rate"] for t in per_task.values()) / len(per_task)
                                          if complete else None),
              "http_requests": sum(r["http_requests"] for r in rows), "model_outputs": outputs,
              "action_validity_rate": sum(r["valid_model_outputs"] for r in rows) / outputs if outputs else None,
              "truncated_outputs": sum(r["truncated_outputs"] for r in rows),
              "output_truncation_rate": sum(r["truncated_outputs"] for r in rows) / outputs if outputs else None,
              "reported_total_tokens": sum(r["reported_total_tokens"] for r in rows),
              "usage_incomplete": any(r["usage_incomplete"] for r in rows),
              **resource_totals(rows),
              "api_latency_seconds_all_attempts": distribution(api_latencies),
              "wall_seconds_all_recorded_episodes": distribution([r["command_wall_seconds"] for r in rows
                                                                   if r["command_wall_seconds"] is not None]),
              "system_fingerprints": sorted({s for r in rows for s in r.get("system_fingerprints", [])}),
              "notes": ["Episode metrics exclude preflight; campaign metrics include all recorded preflight, failed checks and retries.",
                        "Missing, interrupted and infrastructure-error trials remain in planned coverage.",
                        "Model invalid_output counts as a scored failure; repeated trials are not silently substituted.",
                        "Wilson intervals describe scored binary trials, not a claim about unseen tasks.",
                        "A changed provider system_fingerprint may indicate serving drift; weights are not pinned."],
              "episodes": rows}
    write_json(root / "aggregate.json", report)
    unit = rows[0]["api_cost"]["currency"].lower()
    fields = ("index", "suite", "task_id", "task_name", "init_state_id", "seed", "status", "reason", "scored", "task_success",
              "max_control_steps", "action_horizon", "control_steps", "policy_decisions",
              "http_requests", "model_outputs", "valid_model_outputs", "truncated_outputs",
              "reported_total_tokens", "usage_incomplete", "command_wall_seconds", "simulated_control_seconds",
              "task_execution_wall_seconds", "api_wait_seconds", "http_latency_incomplete", "retry_backoff_seconds",
              f"episode_known_cost_min_{unit}", f"episode_known_cost_max_{unit}", "episode_cost_incomplete",
              "episode_cost_currency", "episode_known_cost_min", "episode_known_cost_max",
              "campaign_http_requests", "campaign_reported_total_tokens", "campaign_usage_incomplete",
              f"campaign_known_cost_min_{unit}", f"campaign_known_cost_max_{unit}", "campaign_cost_incomplete",
              "campaign_cost_currency", "campaign_known_cost_min", "campaign_known_cost_max",
              *(f"reported_{k}" for k in TOKEN_FIELDS if k != "total_tokens"))
    with (root / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return report


def resource_totals(rows):
    campaigns = [r["campaign"] for r in rows if r.get("campaign") and r["campaign"]["phases"]]
    currency = rows[0]["api_cost"]["currency"] if rows else "CNY"
    return {"reported_usage": {k: sum(r["reported_usage"][k] for r in rows) for k in TOKEN_FIELDS},
            "usage_field_available_attempts": {k: sum(r["usage_field_available_attempts"][k] for r in rows)
                                                for k in TOKEN_FIELDS},
            "api_cost": aggregate_cost([r["api_cost"] for r in rows],
                pricing_available=all(r["api_cost"]["pricing_available"] for r in rows)),
            "api_wait_seconds": sum(r["api_wait_seconds"] for r in rows),
            "http_latency_incomplete": any(r["http_latency_incomplete"] for r in rows),
            "usage_incomplete": any(r["usage_incomplete"] for r in rows),
            "http_requests": sum(r["http_requests"] for r in rows),
            "task_execution_wall_seconds": distribution([r["task_execution_wall_seconds"] for r in rows
                                                          if r["task_execution_wall_seconds"] is not None]),
            "task_latency_available_episodes": sum(r["task_execution_wall_seconds"] is not None for r in rows),
            "successful_task_execution_wall_seconds": distribution([r["task_execution_wall_seconds"] for r in rows
                if r["task_success"] is True and r["task_execution_wall_seconds"] is not None]),
            "campaign_http_requests": sum(c["http_requests"] for c in campaigns),
            "campaign_reported_usage": {k: sum(c["reported_usage"][k] for c in campaigns) for k in TOKEN_FIELDS},
            "campaign_usage_field_available_attempts": {
                k: sum(c["usage_field_available_attempts"][k] for c in campaigns) for k in TOKEN_FIELDS},
            "campaign_usage_incomplete": any(c["usage_incomplete"] for c in campaigns),
            "campaign_api_cost": aggregate_cost([c["api_cost"] for c in campaigns],
                pricing_available=all(c["api_cost"]["pricing_available"] for c in campaigns), currency=currency),
            "campaign_api_wait_seconds": sum(c["api_wait_seconds"] for c in campaigns)}
