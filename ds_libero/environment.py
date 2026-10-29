"""Adapter for the existing, pinned CPU installation. Imports are intentionally lazy."""

import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import os
import random
import subprocess
import sys
from pathlib import Path

from .config import LIBERO_COMMIT
from .errors import ExperimentError, SimulationError
from .protocol import CAMERAS, Observation


def git_output(root, *args):
    try:
        process = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                 timeout=15, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExperimentError(f"Cannot inspect LIBERO source revision: {exc}") from exc
    if process.returncode:
        raise ExperimentError("LIBERO source must be an accessible Git checkout")
    return process.stdout


class LiberoEnvironment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = None
        work_value = os.environ.get("LIBERO_CPU_WORK")
        if not work_value:
            raise ExperimentError("Start with /mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh, not system Python")
        work = Path(work_value).resolve()
        if sys.version_info[:2] != (3, 10) or Path(sys.prefix).resolve() != (work / ".venv").resolve():
            raise ExperimentError("Expected the existing Python 3.10 .venv selected by run_cpu.sh")
        for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
            if os.environ.get(key) != "osmesa":
                raise ExperimentError(f"{key} must be osmesa; use run_cpu.sh")
        config_path = Path(os.environ.get("LIBERO_CONFIG_PATH", "")) / "config.yaml"
        if not config_path.is_file():
            raise ExperimentError("Existing LIBERO_CONFIG_PATH/config.yaml is required; no config is auto-created")
        versions = {}
        for name, expected in (("robosuite", "1.4.1"), ("mujoco", "2.3.7"), ("torch", "2.5.1+cpu")):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise ExperimentError(f"Missing {name}; use the existing CPU runtime") from exc
            if versions[name] != expected:
                raise ExperimentError(f"Expected {name}=={expected}, got {versions[name]}")
        for name in ("numpy", "Pillow"):
            versions[name] = importlib.metadata.version(name)
        spec = importlib.util.find_spec("libero")
        expected_package = (work / "LIBERO" / "libero").resolve()
        if spec is None or expected_package not in {Path(p).resolve() for p in spec.submodule_search_locations or []}:
            raise ExperimentError("Imported LIBERO does not come from LIBERO_CPU_WORK/LIBERO")
        source = work / "LIBERO"
        commit = git_output(source, "rev-parse", "HEAD").decode().strip()
        if commit != LIBERO_COMMIT:
            raise ExperimentError(f"LIBERO commit mismatch: expected {LIBERO_COMMIT}, got {commit}")
        tracked_diff = git_output(source, "diff", "HEAD", "--", "libero", "setup.py")
        import numpy as np
        import torch
        from OpenGL import GL
        import robosuite.utils.binding_utils as binding
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        if torch.version.cuda is not None or "osmesa" not in binding.GLContext.__module__.lower():
            raise ExperimentError("CPU torch and OSMesa binding are required")
        self.np = np
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.suite = benchmark.get_benchmark_dict()[cfg.suite](task_order_index=cfg.task_order_index)
        task = self.suite.get_task(cfg.task_id)
        self.task = task.language
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        if not bddl.is_file():
            raise ExperimentError(f"Missing task BDDL: {bddl}")
        states = self.suite.get_task_init_states(cfg.task_id)
        if not 0 <= cfg.init_state_id < len(states):
            raise ExperimentError("Official initial-state index is unavailable")
        self.initial_state = np.asarray(states[cfg.init_state_id], dtype=np.float64).copy()
        try:
            self.env = OffScreenRenderEnv(
                bddl_file_name=str(bddl), robots=["Panda"], controller="OSC_POSE",
                camera_names=list(CAMERAS), camera_heights=cfg.image_size, camera_widths=cfg.image_size,
                control_freq=cfg.control_freq, horizon=1000, ignore_done=False)
            self.contract = self._controller_contract()
            renderer = GL.glGetString(GL.GL_RENDERER)
            self.provenance = {
                "python": sys.version.split()[0], "versions": versions, "libero_commit": commit,
                "libero_tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
                "libero_has_tracked_changes": bool(tracked_diff),
                "libero_root": str(source), "libero_config": str(config_path),
                "libero_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "backend": binding.GLContext.__module__,
                "renderer": renderer.decode() if renderer else "not reported",
                "task": self.task, "task_name": task.name, "bddl_sha256": hashlib.sha256(bddl.read_bytes()).hexdigest(),
                "official_init_state_sha256": hashlib.sha256(self.initial_state.tobytes()).hexdigest(),
                "initial_state_count": len(states), "controller": self.contract,
            }
        except BaseException:
            self.close()
            raise

    def _controller_contract(self):
        np = self.np
        robot = self.env.robots[0]
        controller = robot.controller
        expected = {"input_min": [-1.] * 6, "input_max": [1.] * 6,
                    "output_min": [-.05] * 3 + [-.5] * 3,
                    "output_max": [.05] * 3 + [.5] * 3}
        actual = {}
        if controller.name != "OSC_POSE" or not controller.use_delta or controller.impedance_mode != "fixed":
            raise ExperimentError("Expected delta OSC_POSE with fixed impedance")
        for name, wanted in expected.items():
            value = np.asarray(getattr(controller, name), dtype=float)
            if value.shape != (6,) or not np.allclose(value, wanted, rtol=0, atol=1e-10):
                raise ExperimentError(f"Unexpected controller scaling for {name}: {value}")
            actual[name] = value.tolist()
        low, high = self.env.env.action_spec
        if np.shape(low) != (7,) or not np.allclose(low, -1) or not np.allclose(high, 1):
            raise ExperimentError("Expected seven native action dimensions bounded by [-1,1]")
        if type(robot.gripper).__name__ != "PandaGripper" or robot.gripper.dof != 1:
            raise ExperimentError("Expected the one-dimensional Panda gripper")
        if self.env.env.control_freq != self.cfg.control_freq:
            raise ExperimentError("Unexpected simulation control frequency")
        actual.update(name=controller.name, use_delta=True, impedance_mode=controller.impedance_mode,
                      control_freq=self.env.env.control_freq, gripper=type(robot.gripper).__name__,
                      controller_source_sha256=hashlib.sha256(Path(inspect.getfile(type(controller))).read_bytes()).hexdigest(),
                      gripper_source_sha256=hashlib.sha256(Path(inspect.getfile(type(robot.gripper))).read_bytes()).hexdigest())
        if self.cfg.geometry_contract:
            from robosuite import macros
            from .geometry_contract import SCHEMA, check_image_convention
            try:
                check_image_convention(macros.IMAGE_CONVENTION)
            except ValueError as exc:
                raise ExperimentError(str(exc)) from exc
            site = robot.gripper.important_sites["grip_site"]
            if controller.eef_name != site or robot.eef_site_id != self.env.sim.model.site_name2id(site):
                raise ExperimentError("Controller and observable must use the same Panda grip_site")
            actual.update(geometry_schema=SCHEMA, eef_position_source=site, eef_orientation_source=site,
                          legacy_orientation_source=robot.robot_model.eef_name,
                          raw_image_convention=macros.IMAGE_CONVENTION,
                          observation_refresh="sim.forward then force_update; no simulation step")
        return actual

    def _observe(self, raw):
        from PIL import Image
        np = self.np
        if self.cfg.geometry_contract:
            # robosuite observables are sampled/cached. Align image, proprioception
            # and direct MuJoCo geometry at one current state without taking a step.
            simulation_time = float(self.env.sim.data.time)
            self.env.sim.forward()
            raw = self.env.env._get_observations(force_update=True)
            if float(self.env.sim.data.time) != simulation_time:
                raise SimulationError("Geometry observation refresh unexpectedly advanced simulation time")
        state = {}
        for source, target, size in (("robot0_eef_pos", "eef_position_world_m", 3),
                                     ("robot0_eef_quat", "eef_quaternion_xyzw", 4),
                                     ("robot0_gripper_qpos", "gripper_joint_position_m", 2)):
            value = np.asarray(raw[source], dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise SimulationError(f"Invalid proprioception: {source}")
            state[target] = value.tolist()
        if not np.isfinite(self.env.sim.data.qpos).all():
            raise SimulationError("Non-finite simulator state")
        images = {}
        for camera in CAMERAS:
            pixels = np.asarray(raw[camera + "_image"])
            expected_shape = (self.cfg.image_size, self.cfg.image_size, 3)
            if pixels.shape != expected_shape or pixels.dtype != np.uint8 or float(pixels.std()) <= 1:
                raise SimulationError(f"Invalid/blank camera frame: {camera}, {pixels.shape}, {pixels.dtype}")
            stream = io.BytesIO()
            Image.fromarray(pixels[::-1].copy()).save(stream, format="PNG")
            images[camera] = stream.getvalue()
        geometry = None
        if self.cfg.camera_geometry:
            from .geometry import observe_geometry
            geometry = observe_geometry(self.env.sim, CAMERAS, self.cfg.image_size,
                                        state["eef_position_world_m"])
            if self.cfg.grasp_feedback:
                from .geometry import observe_gripper_geometry
                geometry["robot_gripper_geometry"] = observe_gripper_geometry(
                    self.env.sim, self.env.robots[0].gripper, geometry)
            if self.cfg.geometry_contract:
                from .geometry_contract import observe_control_frame
                try:
                    state, geometry = observe_control_frame(self.env.sim, self.env.robots[0], state, geometry)
                except (ValueError, KeyError) as exc:
                    raise SimulationError(f"Robot/camera geometry contract failed: {exc}") from exc
        return Observation(state, images, geometry)

    def reset(self):
        # Always discard calibration/API-check state and start from the official state.
        self.env.seed(self.cfg.seed)
        self.env.reset()
        raw = self.env.set_init_state(self.initial_state.copy())
        settle = self.np.array([0., 0., 0., 0., 0., 0., -1.], dtype=self.np.float32)
        for _ in range(self.cfg.settle_steps):
            raw, _, done, _ = self.env.step(settle)
            if done and not self.env.check_success():
                raise SimulationError("Environment terminated during settling")
        return self._observe(raw)

    def step(self, action):
        raw, _, done, _ = self.env.step(self.np.asarray(action, dtype=self.np.float32))
        return self._observe(raw), bool(done)

    def check_success(self):
        return bool(self.env.check_success())

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
