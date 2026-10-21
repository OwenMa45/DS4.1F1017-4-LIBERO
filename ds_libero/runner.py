"""The control loop is independent of MuJoCo and the HTTP implementation."""

import copy
import time

from .errors import ExperimentError, InvalidAction, SimulationError
from .protocol import validate_action


def run_episode(env, policy, store, cfg):
    start = time.perf_counter()
    steps = 0
    sim_seconds = 0.0
    reset_seconds = 0.0
    pending_step = False
    result = {"status": "incomplete", "scored": False, "task_success": None}
    try:
        reset_start = time.perf_counter()
        obs = env.reset()
        reset_seconds = time.perf_counter() - reset_start
        refs = store.observe(0, obs)
        if env.check_success():
            raise ExperimentError("Initial state already satisfies the task; episode is not scoreable")
        history = []
        for step in range(cfg.max_steps):
            # Defensive validation here also protects adapters other than ModelPolicy.
            action = validate_action(policy.act(obs, copy.deepcopy(history), step, refs))
            before = copy.deepcopy(obs.state)
            store.event({"event": "step_started", "step": step, "action": action})
            pending_step = True
            sim_start = time.perf_counter()
            next_obs, done = env.step(action)
            steps += 1
            pending_step = False
            elapsed = time.perf_counter() - sim_start
            sim_seconds += elapsed
            success = bool(env.check_success())
            store.event({"event": "step", "step": step, "action": action, "before": before,
                         "after": next_obs.state, "task_success": success, "environment_done": bool(done),
                         "simulation_step_seconds": elapsed})
            refs = store.observe(steps, next_obs)
            history.append({"step": step, "before": before, "action": action,
                            "after": copy.deepcopy(next_obs.state)})
            history = history[-cfg.history_steps:]
            obs = next_obs
            print(f"step={steps}/{cfg.max_steps} success={success}", flush=True)
            if success:
                result.update(status="success", scored=True, task_success=True)
                break
            if done:
                raise SimulationError("Environment terminated without check_success(); not scored as success/failure")
        else:
            result.update(status="failure", scored=True, task_success=False, reason="control_step_limit")
    except InvalidAction as exc:
        result.update(status=exc.status, scored=True, task_success=False, reason=str(exc))
    except ExperimentError as exc:
        result.update(status=exc.status, reason=str(exc))
    except KeyboardInterrupt:
        result.update(status="interrupted", reason="Interrupted by user")
    except Exception as exc:
        result.update(status="runtime_error", reason=f"{type(exc).__name__}: {exc}")
    result.update(control_steps=steps, settle_steps=cfg.settle_steps,
                  simulated_control_seconds=steps / cfg.control_freq,
                  simulation_step_seconds=sim_seconds, reset_and_settle_seconds=reset_seconds,
                  control_loop_wall_seconds=time.perf_counter() - start,
                  last_step_may_have_executed=pending_step)
    return result
