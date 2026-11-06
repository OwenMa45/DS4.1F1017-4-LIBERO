"""LIBERO-Five v1: a fixed diagnostic suite with a model-independent policy API.

Importing, listing, planning and reporting do not load LIBERO or call a model.
Only ``MicroBench.run`` / ``evaluate`` start simulation and inference.
"""

import copy
import csv
import hashlib
import importlib
import inspect
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from .artifacts import Store, source_fingerprint, verify_gate, write_json
from .benchmark import child_path, create_five_plan, load_plan, report_plan
from .config import Config, LIBERO_COMMIT, digest
from .errors import ExperimentError
from .five_tasks import FIVE_TASKS, five_configs
from .protocol import Observation, system_prompt, validate_actions
from .report import build_report
from .runner import check_policy, run_episode


BENCH_ID = "libero-five-v1"
TASK_LABELS = ("黑碗放盘", "汤罐入篮", "开中层抽屉", "推盘", "书本入收纳格")
ENVIRONMENT_FIELDS = (
    "protocol_version", "task_order_index", "init_state_id", "seed", "settle_steps",
    "max_steps", "image_size", "history_steps", "control_freq", "action_horizon",
    "visual_history", "camera_geometry", "grasp_feedback", "gripper_feedback_barrier",
    "geometry_contract",
)


@dataclass(frozen=True)
class TaskSpec:
    index: int
    key: str
    label: str
    suite: str
    task_id: int
    task_name: str


TASKS = tuple(TaskSpec(i, f"{suite}:{task}", TASK_LABELS[i], suite, task, name)
              for i, (suite, task, name) in enumerate(FIVE_TASKS))


@dataclass(frozen=True)
class EpisodeContext:
    """A new factory call per episode prevents implicit cross-task policy state."""

    task: TaskSpec
    instruction: str
    max_steps: int
    action_horizon: int
    control_freq: int
    control_instructions: str
    policy_config: dict


@dataclass(frozen=True)
class PolicyInput:
    """Owned copies of visible data only. Images are upright RGB PNG bytes."""

    instruction: str
    observation: Observation
    history: list
    visual_history: tuple  # (control_step, Observation) pairs
    control_feedback: dict | None
    step: int
    remaining_steps: int
    max_actions: int


class Policy(Protocol):
    def act(self, request: PolicyInput) -> list[list[float]]:
        """Return 1..request.max_actions native 7D actions; no implicit clipping."""
        ...


PolicyFactory = Callable[[EpisodeContext], Policy]


def external_config(*, max_steps=220, action_horizon=8):
    """Geometry/feedback settings shared with the existing five-task runs."""
    return Config.from_dict(dict(
        protocol_version="ds-libero-direct-v2", api_provider="external",
        evaluation_scope="diagnostic", model="external-policy", expected_release="user-supplied",
        endpoint="", reasoning_effort="none", max_tokens=None, max_retries=0,
        max_steps=max_steps, action_horizon=action_horizon, visual_history=2,
        camera_geometry=True, grasp_feedback=True, gripper_feedback_barrier=True,
        geometry_contract=True,
    ))


def _spec(cfg):
    environment = {key: getattr(cfg, key) for key in ENVIRONMENT_FIELDS}
    if cfg.zero_on_dimension_error:
        environment["zero_on_dimension_error"] = True
    return {"id": BENCH_ID, "libero_commit": LIBERO_COMMIT,
            "tasks": [asdict(t) for t in TASKS],
            "environment": environment,
            "success": "env.check_success after each executed action",
            "score": "equal task weights; score is null until all five trials are scored",
            "scope": "five fixed tasks, official initial state 0 and seed 0; not a full LIBERO score"}


def _load(directory):
    root, plan = load_plan(directory)
    meta = plan.get("microbench", {})
    cfg = Config.from_dict(plan["episodes"][0]["config"])
    if (plan["schema_version"] != 2 or meta.get("schema_version") != 1
            or meta.get("spec") != _spec(cfg)
            or meta.get("protocol_sha256") != digest(_spec(cfg))):
        raise ExperimentError("Not a valid LIBERO-Five v1 microbench plan")
    return root, plan, cfg


def _import_factory(reference):
    module, separator, name = reference.partition(":")
    if not separator or not module or not name:
        raise ExperimentError("Policy factory must be module:callable")
    try:
        factory = importlib.import_module(module)
        for part in name.split("."):
            factory = getattr(factory, part)
    except (ImportError, AttributeError) as exc:
        raise ExperimentError(f"Cannot load policy factory {reference}: {exc}") from exc
    if not callable(factory):
        raise ExperimentError("Policy factory must be callable")
    return factory


def _factory_source(factory):
    try:
        path = inspect.getsourcefile(factory)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    except (OSError, TypeError):
        return None


class _PolicyAdapter:
    """Translate the public policy contract to the existing audited control loop."""

    def __init__(self, policy, context, cfg, store):
        if not callable(getattr(policy, "act", None)):
            raise ExperimentError("Policy factory must return an object with act(PolicyInput)")
        self.policy, self.context, self.cfg, self.store = policy, context, cfg, store
        self.previous = None
        self.visual_history = []

    def act(self, obs, history, step, refs):
        from .feedback import decision_feedback
        start = time.perf_counter()
        event = {"event": "decision", "step": step, "frames": refs, "valid": False}
        try:
            feedback = (decision_feedback(obs, step, self.previous, self.cfg.control_freq)
                        if self.cfg.grasp_feedback else None)
            limit = min(self.cfg.action_horizon, self.cfg.max_steps - step)
            request = PolicyInput(self.context.instruction, copy.deepcopy(obs), copy.deepcopy(history),
                                  copy.deepcopy(tuple(self.visual_history)), copy.deepcopy(feedback),
                                  step, self.cfg.max_steps - step, limit)
            actions = validate_actions(self.policy.act(request), limit)
            event.update(valid=True, actions=actions, control_feedback=feedback)
            self.previous = (step, copy.deepcopy(obs), copy.deepcopy(actions))
            if self.cfg.visual_history:
                self.visual_history.append((step, copy.deepcopy(obs)))
                self.visual_history = self.visual_history[-self.cfg.visual_history:]
            return actions
        except ExperimentError as exc:
            event.update(error=str(exc), error_status=exc.status)
            raise
        except KeyboardInterrupt:
            event.update(error_status="interrupted")
            raise
        except Exception as exc:
            event.update(error_status="runtime_error", error_type=type(exc).__name__)
            raise
        finally:
            event["elapsed_seconds"] = time.perf_counter() - start
            self.store.event(event)


def _phase(root, entry, phase, meta, factory, replay):
    """Use the same gates, environment, logs and scorer as the legacy commands."""
    from .calibration import calibrate
    from .client import Client, ModelPolicy
    from .environment import LiberoEnvironment

    cfg = Config.from_dict(entry["config"])
    campaign = child_path(root, entry["run_directory"])
    store = Store(campaign / phase, {"calibration": "calibrate", "api-check": "api-check",
                                    "episode": "run"}[phase], cfg)
    store.manifest["microbench"] = copy.deepcopy(meta)
    write_json(store.path / "manifest.json", store.manifest)
    started = time.perf_counter()
    env, custom_policy = None, None
    result = {"status": "incomplete", "scored": False, "task_success": None}
    try:
        env = LiberoEnvironment(cfg)
        store.environment(env.provenance)
        if phase == "calibration":
            result = calibrate(env, store)
        else:
            verify_gate(campaign / "calibration", "calibrate", cfg, env.provenance)
            if cfg.api_provider == "external":
                context = EpisodeContext(TASKS[entry["index"]], env.task, cfg.max_steps,
                    cfg.action_horizon, cfg.control_freq, system_prompt(cfg), copy.deepcopy(meta["policy"]["config"]))
                custom_policy = factory(context)
                policy = _PolicyAdapter(custom_policy, context, cfg, store)
            else:
                gate = {}
                if phase == "episode":
                    gate = verify_gate(campaign / "api-check", "api-check", cfg, env.provenance)
                policy = ModelPolicy(cfg, env.task, Client(cfg, store), store, gate.get("action_dimension_feedback"))
                if cfg.sam3:
                    policy.sam3.verify_runtime(json.loads((root / "sam3_runtime.json").read_text(encoding="utf-8")))
                    if phase == "episode":
                        checked = json.loads((campaign / "api-check" / "manifest.json").read_text(encoding="utf-8"))
                        policy.sam3.verify_runtime(checked.get("sam3_worker"))
            if phase == "api-check":
                result = check_policy(env, policy, store)
            else:
                result = run_episode(env, policy, store, cfg)
    except ExperimentError as exc:
        # InvalidAction during factory/setup is not an observed policy output.
        # run_episode itself already scores invalid act() returns correctly.
        status = "runtime_error" if phase == "episode" and exc.status == "invalid_output" else exc.status
        result.update(status=status, reason=str(exc))
    except KeyboardInterrupt:
        result.update(status="interrupted", reason="Interrupted by user")
    except Exception as exc:
        result.update(status="runtime_error", reason=f"{type(exc).__name__}: {exc}")
    finally:
        for label, resource in (("policy", custom_policy), ("environment", env)):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    result[f"{label}_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["command_wall_seconds"] = time.perf_counter() - started
        store.finish(result)
    build_report(store.path, replay=replay and phase == "episode")
    return result


def _recorded(campaign):
    # A rejected preflight is not a completed trial and must stop batch/resume.
    path = campaign / "episode" / "result.json"
    return path.is_file() and json.loads(path.read_text(encoding="utf-8"))["status"] in (
        "success", "failure", "invalid_output", "budget_exhausted")


class MicroBench:
    """Fixed five-task benchmark. The default is the 220-step geometry profile."""

    tasks = TASKS

    def __init__(self, config=None, *, max_steps=220, action_horizon=8):
        if config is not None and (max_steps != 220 or action_horizon != 8):
            raise ExperimentError("Pass a config OR explicit budgets, not both")
        base = (Config.load(config) if isinstance(config, (str, Path)) else config)
        if base is None:
            base = external_config(max_steps=max_steps, action_horizon=action_horizon)
        # Revalidate even dataclass instances constructed without Config.from_dict.
        self.config = five_configs(Config.from_dict(base.to_dict()))[0]

    @property
    def spec(self):
        return _spec(self.config)

    def prepare(self, out, *, policy_id=None, revision=None, adapter=None, policy_config=None):
        """Freeze the task list and experiment metadata, without importing an adapter."""
        external = self.config.api_provider == "external"
        if external:
            if not all(isinstance(v, str) and v.strip() for v in (policy_id, revision, adapter)):
                raise ExperimentError("External policies require policy_id, revision and module:callable adapter")
            if ":" not in adapter:
                raise ExperimentError("Policy adapter must be module:callable")
        elif adapter is not None or policy_config is not None:
            raise ExperimentError("Custom adapters require an external profile")
        if policy_config is not None and not isinstance(policy_config, dict):
            raise ExperimentError("policy_config must be a JSON object; keep credentials in the environment")
        policy = {"id": policy_id or self.config.model, "revision": revision or self.config.expected_release,
                  "adapter": adapter, "config": copy.deepcopy(policy_config or {})}
        if not all(isinstance(policy[key], str) and policy[key].strip() for key in ("id", "revision")):
            raise ExperimentError("policy_id and revision must be nonempty strings")
        digest(policy)  # Validate serializability/finiteness before creating files.
        plan = create_five_plan(self.config, out)
        plan["microbench"] = {"schema_version": 1, "spec": self.spec,
                              "protocol_sha256": digest(self.spec), "policy": policy}
        if external:
            plan["max_http_requests_per_episode_including_api_check"] = None
        plan["plan_sha256"] = digest({k: v for k, v in plan.items() if k != "plan_sha256"})
        write_json(Path(out) / "plan.json", plan)
        self.report(out)
        return plan

    def evaluate(self, policy_factory=None, *, out, policy_id=None, revision=None,
                 policy_config=None, replay=True):
        """Create a new plan and run it. Existing directories are never overwritten."""
        adapter = None
        if policy_factory is not None:
            if not callable(policy_factory):
                raise ExperimentError("policy_factory must be callable")
            adapter = f"{policy_factory.__module__}:{policy_factory.__qualname__}"
        self.prepare(out, policy_id=policy_id, revision=revision, adapter=adapter, policy_config=policy_config)
        return self.run(out, policy_factory=policy_factory, replay=replay)

    @staticmethod
    def run(directory, *, policy_factory=None, episode=None, replay=True):
        """Resume unstarted tasks only; never rerun a recorded or interrupted trial."""
        root, plan, cfg = _load(directory)
        if source_fingerprint() != plan["source_sha256"]:
            raise ExperimentError("Code changed since planning; create a new experiment directory")
        if episode is not None and (type(episode) is not int or not 0 <= episode < len(TASKS)):
            raise ExperimentError("episode must be an integer in 0..4")
        selected = plan["episodes"] if episode is None else [plan["episodes"][episode]]
        pending = []
        for entry in selected:
            campaign = child_path(root, entry["run_directory"])
            if _recorded(campaign):
                continue
            if campaign.exists():
                raise ExperimentError(f"Episode {entry['index']} has incomplete/error artifacts; inspect it or select a later unstarted episode")
            pending.append(entry)
        if not pending:
            report = MicroBench.report(root)
            report["batch_exit_code"] = 0
            write_json(root / "microbench.json", report)
            return report
        meta = plan["microbench"]
        external = cfg.api_provider == "external"
        if external:
            factory = policy_factory if policy_factory is not None else _import_factory(meta["policy"]["adapter"])
            if not callable(factory):
                raise ExperimentError("policy_factory must be callable")
            reference = f"{factory.__module__}:{factory.__qualname__}"
            if policy_factory is not None and reference != meta["policy"]["adapter"]:
                raise ExperimentError("Supplied factory differs from the adapter frozen in the plan")
            identity = {"reference": reference, "source_sha256": _factory_source(factory),
                        "policy_sha256": digest(meta["policy"])}
            identity_path = root / "policy_source.json"
            if identity_path.exists():
                if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                    raise ExperimentError("Policy factory changed; create a new experiment directory")
            else:
                write_json(identity_path, identity)
        else:
            if policy_factory is not None:
                raise ExperimentError("A custom factory requires an external profile")
            from .client import resolve_api_key
            resolve_api_key(provider=cfg.api_provider)
            if cfg.sam3:
                from .sam3_tools import worker_health
                health = worker_health(cfg.sam3)
                runtime_path = root / "sam3_runtime.json"
                if runtime_path.exists():
                    old = json.loads(runtime_path.read_text(encoding="utf-8"))
                    if old.get("runtime_id") != health["runtime_id"]:
                        raise ExperimentError("SAM3 runtime changed across this five-task experiment")
                else:
                    write_json(runtime_path, health)
            factory = None
        exit_code = 0
        for entry in pending:
            campaign = child_path(root, entry["run_directory"])
            campaign.mkdir(parents=True, exist_ok=False)
            try:
                for phase in (("calibration", "episode") if external else ("calibration", "api-check", "episode")):
                    result = _phase(root, entry, phase, meta, factory, replay)
                    if phase != "episode" and result["status"] != "pass":
                        break
            finally:
                MicroBench.report(root)
            if not _recorded(campaign):
                exit_code = 2
                break
        report = MicroBench.report(root)
        report["batch_exit_code"] = exit_code
        write_json(root / "microbench.json", report)
        return report

    @staticmethod
    def report(directory):
        """Stable public score schema; unknown external inference costs stay null."""
        root, plan, cfg = _load(directory)
        for entry in plan["episodes"]:
            campaign = child_path(root, entry["run_directory"])
            for phase in ("calibration", "api-check", "episode"):
                path = campaign / phase / "manifest.json"
                if path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("microbench") != plan["microbench"]:
                    raise ExperimentError(f"Episode {entry['index']} {phase} differs from the frozen microbench policy")
        aggregate = report_plan(root)
        external = cfg.api_provider == "external"
        rows = []
        for task, row in zip(TASKS, aggregate["episodes"]):
            outputs = row["model_outputs"]
            rows.append({"index": task.index, "task": task.key, "label": task.label,
                         "task_name": task.task_name, "status": row["status"],
                         "scored": row["scored"], "success": row["task_success"],
                         "reason": row["reason"], "control_steps": row["control_steps"],
                         "policy_decisions": row["policy_decisions"],
                         "dimension_fallback_steps": row.get("dimension_fallback_steps"),
                         "action_validity_rate": row["valid_model_outputs"] / outputs if outputs else None,
                         "decision_latency_seconds": row["decision_latency_seconds"],
                         "task_wall_seconds": row["task_execution_wall_seconds"],
                         "http_requests": None if external else row["http_requests"],
                         "campaign_http_requests": None if external else row["campaign_http_requests"],
                         "usage": None if external else row["reported_usage"],
                         "api_cost": None if external else row["api_cost"],
                         "campaign_api_cost": None if external or not row["campaign"] else row["campaign"]["api_cost"],
                         "preflight_status": row.get("preflight_status"),
                         "sam3": row.get("sam3"),
                         "artifacts": plan["episodes"][task.index]["run_directory"]})
        report = {"schema_version": 1, "benchmark": BENCH_ID, **copy.deepcopy(plan["microbench"]),
                  "plan_sha256": plan["plan_sha256"], "source_sha256": plan["source_sha256"],
                  "complete": aggregate["complete"], "planned": aggregate["planned"],
                  "scored": aggregate["scored"], "successes": aggregate["successes"],
                  "score": aggregate["macro_task_success_rate"], "coverage": aggregate["coverage"],
                  "planned_success_bounds": aggregate["planned_success_bounds"],
                  "status_counts": aggregate["status_counts"], "episodes": rows,
                  "campaign_http_requests": None if external else sum(r["campaign_http_requests"] for r in rows),
                  "inference_accounting": "not_instrumented" if external else "built_in_client",
                  "notes": ["Score is 0..1; null means at least one of the five tasks is unscored.",
                            "http_requests counts the control episode; campaign_http_requests also includes paid preflight.",
                            "Compare protocol_sha256 before comparing experiments; horizons and feedback settings affect the protocol.",
                            "External inference usage/cost is unknown, not zero. Detailed legacy logs contain recorded events only.",
                            "Policy revision is user-declared; adapter file hashing does not pin weights or transitive dependencies."]}
        write_json(root / "microbench.json", report)
        fields = ("index", "task", "label", "status", "scored", "success", "reason", "control_steps",
                  "policy_decisions", "action_validity_rate", "dimension_fallback_steps", "task_wall_seconds", "http_requests", "campaign_http_requests", "artifacts")
        with (root / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return report
