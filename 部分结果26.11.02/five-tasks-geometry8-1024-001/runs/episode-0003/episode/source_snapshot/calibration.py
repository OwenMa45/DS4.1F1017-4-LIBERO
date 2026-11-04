import math

from .errors import ExperimentError


def quat_product(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def rotation_delta(before, after):
    # q_after * inverse(q_before): world-frame relative rotation vector.
    norm = math.sqrt(sum(x * x for x in before))
    before = [x / norm for x in before]
    norm = math.sqrt(sum(x * x for x in after))
    after = [x / norm for x in after]
    q = quat_product(after, [-before[0], -before[1], -before[2], before[3]])
    if q[3] < 0:
        q = [-x for x in q]
    sine = math.sqrt(sum(x * x for x in q[:3]))
    if sine < 1e-12:
        return [0., 0., 0.]
    scale = 2 * math.atan2(sine, q[3]) / sine
    return [x * scale for x in q[:3]]


def aperture(obs):
    joints = obs.state["gripper_joint_position_m"]
    return abs(joints[0] - joints[1])


def calibrate(env, store):
    probes = []
    geometry_checks = []
    def record(label, obs):
        store.observe(label, obs)
        frame = (obs.geometry or {}).get("control_frame")
        if frame:
            from .geometry_overlay import draw_geometry_overlay
            folder = store.path / "geometry_overlays"
            folder.mkdir(exist_ok=True)
            for camera in obs.images:
                (folder / f"{label}_{camera}.png").write_bytes(draw_geometry_overlay(obs, camera))
            geometry_checks.append({"frame": label, **frame["checks"]})
    for axis in range(6):
        for sign in (1, -1):
            label = f"{'xyz'[axis % 3]}_{'position' if axis < 3 else 'rotation'}_{'positive' if sign > 0 else 'negative'}"
            before = env.reset()
            if env.check_success():
                raise ExperimentError("Official initial state already succeeds")
            record(label + "_before", before)
            action = [0.] * 6 + [-1.]
            action[axis] = sign * .1
            after = before
            for _ in range(3):
                after, done = env.step(action)
                if done:
                    raise ExperimentError("Environment ended during calibration")
            record(label + "_after", after)
            translation = [b - a for a, b in zip(before.state["eef_position_world_m"], after.state["eef_position_world_m"])]
            rotation = rotation_delta(before.state["eef_quaternion_xyzw"], after.state["eef_quaternion_xyzw"])
            measured = translation if axis < 3 else rotation
            threshold = 0.00005 if axis < 3 else 0.0001
            passed = sign * measured[axis % 3] > threshold
            probe = {"name": label, "action": action, "steps": 3,
                     "delta_position_m": translation, "delta_rotation_world_rad": rotation,
                     "minimum_signed_motion": threshold, "passed": passed}
            probes.append(probe)
            store.event({"event": "calibration_probe", **probe})
            print(f"{label}: {'PASS' if passed else 'FAIL'} measured={measured}", flush=True)
    before = env.reset()
    opened_initially = aperture(before)
    close_trace = []
    for index in range(20):
        closed, done = env.step([0.] * 6 + [1.])
        close_trace.append({"steps": index + 1, "joint_separation_m": aperture(closed)})
        if done:
            raise ExperimentError("Environment ended during gripper calibration")
    record("gripper_closed", closed)
    open_trace = []
    for index in range(20):
        opened, done = env.step([0.] * 6 + [-1.])
        open_trace.append({"steps": index + 1, "joint_separation_m": aperture(opened)})
        if done:
            raise ExperimentError("Environment ended during gripper calibration")
    record("gripper_open", opened)
    probe = {"name": "gripper_close_then_open", "initial_aperture_m": opened_initially,
             "close_trace": close_trace, "open_trace": open_trace,
             "closed_aperture_m": aperture(closed), "opened_aperture_m": aperture(opened),
             "passed": opened_initially - aperture(closed) > .001 and aperture(opened) - aperture(closed) > .001}
    probes.append(probe)
    store.event({"event": "calibration_probe", **probe})
    passed = all(p["passed"] for p in probes)
    result = {"status": "pass" if passed else "calibration_failed", "scored": False, "task_success": None,
            "probes": probes, "controller": env.contract,
            "note": "Diagnostic probes only; sign thresholds test motion, not target-position accuracy."}
    if geometry_checks:
        result["geometry_contract_checks"] = geometry_checks
        result["geometry_overlays"] = "geometry_overlays/ (diagnostic images only; not policy inputs)"
    return result
