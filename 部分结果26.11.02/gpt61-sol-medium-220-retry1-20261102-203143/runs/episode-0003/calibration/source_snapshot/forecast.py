"""Offline planning scenarios, explicitly separate from measured GPT performance."""

import hashlib
import json
import math
from pathlib import Path

from .accounting import pricing_snapshot
from .benchmark import child_path, load_plan
from .errors import ExperimentError
from .five_tasks import FIVE_TASKS
from .report import load_events


def forecast_gpt(cfg, baseline=None):
    if cfg.api_provider != "openai":
        raise ExperimentError("GPT forecast requires an OpenAI comparison configuration")
    root, plan = load_plan(baseline) if baseline else (None, None)
    if plan and plan["schema_version"] != 2:
        raise ExperimentError("Use the matching five-task diagnostic plan as baseline")
    trials = []
    for index, (suite, task, name) in enumerate(FIVE_TASKS):
        average_actions = 5.0  # Fallback planning assumption, not a measured model result.
        evidence = {"basis": "assumed five executed actions per decision"}
        if plan:
            directory = child_path(root, plan["episodes"][index]["run_directory"]) / "episode"
            log = directory / "events.jsonl"
            events, warnings = load_events(log)
            # Measure the initial 220-step window, which is closest to the GPT budget.
            steps = [e for e in events if e.get("event") == "step" and e["step"] < cfg.max_steps]
            decisions = {e.get("decision_step", e["step"]) for e in steps}
            if steps and decisions:
                average_actions = len(steps) / len(decisions)
                evidence = {"basis": "executed actions per decision in baseline's initial control window",
                            "control_steps_observed": len(steps), "decisions_observed": len(decisions),
                            "events_sha256": hashlib.sha256(log.read_bytes()).hexdigest(), "warnings": warnings}
            evidence["source_directory"] = str(directory)
        calls = min(cfg.max_http_requests, math.ceil(cfg.max_steps / average_actions))
        trials.append({"index": index, "suite": suite, "task_id": task, "task_name": name,
                       "mean_executed_actions_per_decision": average_actions,
                       "projected_episode_calls": calls, "preflight_calls": 1, "evidence": evidence})
    projected = sum(t["projected_episode_calls"] + t["preflight_calls"] for t in trials)
    maximum = len(trials) * (cfg.max_http_requests + min(cfg.max_http_requests, cfg.max_retries + 1))
    pricing = pricing_snapshot(cfg.model)
    rates = pricing["per_million_tokens"]["standard"]
    scenarios = []
    # Inputs include images/history. Output includes hidden reasoning. These are assumptions,
    # not a transfer of DeepSeek's tokenization, reasoning length or tokens/sec to GPT.
    for label, inputs, outputs, latency in (("light", 8000, 2000, 15),
                                           ("central", 10000, 4000, 30),
                                           ("heavy", 14000, 8000, 60)):
        outputs = min(outputs, cfg.max_tokens)
        cost_low = projected * (inputs * float(rates["prompt_cache_miss_tokens"]) +
                                outputs * float(rates["completion_tokens"])) / 1e6
        cost_high = projected * (inputs * float(rates["prompt_cache_write_30m_tokens"]) +
                                 outputs * float(rates["completion_tokens"])) / 1e6
        scenarios.append({"scenario": label, "http_attempts": projected,
                          "assumed_input_tokens_per_call_including_images": inputs,
                          "assumed_output_tokens_per_call_including_reasoning": outputs,
                          "assumed_http_seconds_per_call": latency,
                          "assumed_total_setup_simulation_reporting_seconds": 600,
                          "estimated_cost_range_usd": [round(cost_low, 2), round(cost_high, 2)],
                          "estimated_sequential_wall_hours": round((projected * latency + 600) / 3600, 2)})
    # This is deliberately not called a cost cap: input usage is not bounded by this calculation.
    stress = maximum * (14000 * float(rates["prompt_cache_write_30m_tokens"]) +
                        cfg.max_tokens * float(rates["completion_tokens"])) / 1e6
    return {"model": cfg.model, "reasoning_effort": cfg.reasoning_effort, "api_pricing": pricing,
            "baseline_plan_sha256": plan["plan_sha256"] if plan else None,
            "control_steps_per_task": cfg.max_steps, "tasks": trials,
            "projected_total_http_attempts_including_preflight": projected,
            "maximum_total_http_attempts_including_preflight": maximum,
            "scenarios": scenarios,
            "stress_scenario": {"http_attempts": maximum, "input_tokens_per_call_assumed": 14000,
                                "output_tokens_per_call": cfg.max_tokens, "cost_usd": round(stress, 2)},
            "notes": ["Offline forecast; no API call, no GPT throughput or output-length measurement.",
                      "Projected calls extrapolate observed action chunk sizes; GPT may choose different chunks or stop early.",
                      "The cost range uses ordinary input vs all input charged as 30m cache writes, without assumed cache savings.",
                      "Latency/token scenarios are planning assumptions; rate limits, failures and retries can change them.",
                      "Stress cost assumes 14K input tokens per call; it is not a hard dollar limit.",
                      "The baseline originally allowed 1024 steps. A 220-step slice is descriptive, not a rerun with a 220-step prompt."]}
