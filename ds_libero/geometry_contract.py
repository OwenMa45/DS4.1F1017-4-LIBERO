"""Checked Panda grip-site geometry for the opt-in v2 profile.

Uses only named robot sites/bodies, robot fingerpads and camera calibration.
Pinned robosuite 1.4.1 sources and derived constants: docs/geometry_contract.md.
"""
import math

SCHEMA = "panda-grip-site-v1"
POSITION_TOLERANCE_M = 1e-5
ROTATION_TOLERANCE = 1e-5


def rotation_matrix(value):
    import numpy as np
    value = np.asarray(value, dtype=float)
    if (value.shape != (3, 3) or not np.isfinite(value).all()
            or not np.allclose(value.T @ value, np.eye(3), atol=ROTATION_TOLERANCE, rtol=0)
            or not math.isclose(float(np.linalg.det(value)), 1., abs_tol=ROTATION_TOLERANCE)):
        raise ValueError("Expected a finite, right-handed orthonormal rotation matrix")
    return value


def quaternion_matrix(quaternion):
    import numpy as np
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not math.isclose(float(np.linalg.norm(q)), 1., abs_tol=1e-4):
        raise ValueError("Expected a unit xyzw quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def matrix_quaternion(matrix):
    """Convert a checked rotation to canonical xyzw, including rotations near pi."""
    import numpy as np
    r = rotation_matrix(matrix)
    if np.trace(r) > 0:
        s = math.sqrt(1. + float(np.trace(r))) * 2
        q = [(r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s, (r[1,0]-r[0,1])/s, s/4]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i+1) % 3, (i+2) % 3
        s = math.sqrt(1. + r[i,i]-r[j,j]-r[k,k]) * 2
        q = [0.] * 4
        q[i], q[j], q[k], q[3] = s/4, (r[j,i]+r[i,j])/s, (r[k,i]+r[i,k])/s, (r[k,j]-r[j,k])/s
    q = np.array(q) / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q.tolist()


def check_image_convention(convention):
    if convention != "opengl":
        raise ValueError("geometry_contract requires robosuite IMAGE_CONVENTION=opengl before the adapter's vertical flip")


def observe_control_frame(sim, robot, raw_state, geometry):
    """Check the source-frame relationship, then return a unified site state and geometry.

    raw_state.eef_position_world_m is a site position, but its legacy quaternion
    is the right_hand BODY orientation. Never combine those as a site pose.
    """
    import numpy as np
    site_name = robot.gripper.important_sites["grip_site"]
    if robot.controller.eef_name != site_name:
        raise ValueError("OSC controller and gripper observation must reference the same grip_site")
    site_id = sim.model.site_name2id(site_name)
    if robot.eef_site_id != site_id:
        raise ValueError("Robot observable and controller refer to different sites")
    hand_name = robot.robot_model.eef_name
    hand_id = sim.model.body_name2id(hand_name)
    p = np.asarray(sim.data.site_xpos[site_id], dtype=float)
    hand_p = np.asarray(sim.data.body_xpos[hand_id], dtype=float)
    r = rotation_matrix(np.asarray(sim.data.site_xmat[site_id]).reshape(3, 3))
    hand_r = rotation_matrix(np.asarray(sim.data.body_xmat[hand_id]).reshape(3, 3))
    observed_p = np.asarray(raw_state["eef_position_world_m"], dtype=float)
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (p, hand_p, observed_p)):
        raise ValueError("Invalid robot reference position")
    position_error = float(np.max(np.abs(p - observed_p)))
    hand_error = float(np.max(np.abs(hand_r - quaternion_matrix(raw_state["eef_quaternion_xyzw"]))))
    # panda_gripper.xml: root Rz(-pi/2), eef body at [0,0,.097].
    expected_rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    mount_rotation = hand_r.T @ r
    mount_offset = hand_r.T @ (p-hand_p)
    rotation_error = float(np.max(np.abs(mount_rotation - expected_rotation)))
    mount_error = float(np.max(np.abs(mount_offset - [0., 0., .097])))
    if max(position_error, mount_error) > POSITION_TOLERANCE_M or max(hand_error, rotation_error) > ROTATION_TOLERANCE:
        raise ValueError("Panda site/body geometry differs from the pinned contract or observations are not synchronized")

    gripper = dict(geometry["robot_gripper_geometry"])
    points = gripper["points_world_m"]
    local = {key: r.T @ (np.asarray(value, dtype=float)-p) for key, value in points.items()}
    joints = np.asarray(raw_state["gripper_joint_position_m"], dtype=float)
    if joints.shape != (2,) or not np.isfinite(joints).all():
        raise ValueError("Invalid gripper joints")
    # Derived from the actual finger body, tip and box-centre offsets, NOT DROID.
    expected_pads = {"left_pad": [-joints[0]-.0035, 0., -.0036],
                     "right_pad": [-joints[1]+.0035, 0., -.0036]}
    pad_error = max(float(np.max(np.abs(local[key]-value))) for key, value in expected_pads.items())
    if not math.isfinite(pad_error) or pad_error > POSITION_TOLERANCE_M:
        raise ValueError("Panda fingerpad references differ from the pinned geometry contract")
    jaw = np.asarray(points["right_pad"]) - np.asarray(points["left_pad"])
    if float(np.linalg.norm(jaw)) < 1e-8:
        raise ValueError("Degenerate fingerpad separation")
    gripper.update(points_control_frame_m={key: value.tolist() for key, value in local.items()},
                   jaw_axis_world_left_to_right=(jaw/np.linalg.norm(jaw)).tolist(),
                   finger_extension_axis_world=r[:, 2].tolist())
    cameras = {}
    for name, camera in geometry["cameras"].items():
        t = np.asarray(camera["T_world_from_camera"], dtype=float)
        k = np.asarray(camera["K"], dtype=float)
        if (t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0,0,0,1])
                or k.shape != (3, 3) or not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0):
            raise ValueError("Invalid camera transform or intrinsics")
        camera_r = rotation_matrix(t[:3, :3])
        cameras[name] = {**camera,
                         "position_control_frame_m": (r.T @ (t[:3, 3]-p)).tolist(),
                         "R_control_from_camera_optical": (r.T @ camera_r).tolist(),
                         "view_direction_world": camera_r[:, 2].tolist()}
    frame = {"schema": SCHEMA, "position_source": site_name, "orientation_source": site_name,
             "legacy_orientation_source": hand_name,
             "legacy_hand_quaternion_xyzw": raw_state["eef_quaternion_xyzw"],
             "R_world_from_control": r.tolist(), "R_hand_from_control": mount_rotation.tolist(),
             "checks": {"observation_position_error_m": position_error, "legacy_hand_rotation_error": hand_error,
                        "mount_position_error_m": mount_error, "mount_rotation_error": rotation_error,
                        "fingerpad_position_error_m": pad_error}}
    return ({**raw_state, "eef_quaternion_xyzw": matrix_quaternion(r)},
            {**geometry, "control_frame": frame, "robot_gripper_geometry": gripper, "cameras": cameras})
