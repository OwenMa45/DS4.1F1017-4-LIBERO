"""Robot/camera calibration only. Never reads object poses, depth or task state."""

import math


def camera_calibration(position, rotation, fovy, size):
    import numpy as np
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    position = np.asarray(position, dtype=float)
    # MuJoCo camera looks along -Z with +Y up; upright PNG uses optical +Y down.
    optical_rotation = rotation @ np.diag([1., -1., -1.])
    transform = np.eye(4)
    transform[:3, :3] = optical_rotation
    transform[:3, 3] = position
    focal = size / (2 * math.tan(math.radians(float(fovy)) / 2))
    intrinsic = np.array([[focal, 0., size / 2], [0., focal, size / 2], [0., 0., 1.]])
    if not np.isfinite(transform).all() or not np.isfinite(intrinsic).all() or focal <= 0:
        raise ValueError("Non-finite or invalid camera calibration")
    return {"image_width": size, "image_height": size,
            "K": intrinsic.tolist(), "T_world_from_camera": transform.tolist()}


def project_world(point, calibration):
    import numpy as np
    transform = np.asarray(calibration["T_world_from_camera"])
    camera = transform[:3, :3].T @ (np.asarray(point) - transform[:3, 3])
    if camera[2] <= 0:
        return None
    pixel = np.asarray(calibration["K"]) @ camera
    return (pixel[:2] / pixel[2]).tolist()


def observe_geometry(sim, cameras, size, eef_position):
    result = {"convention": "upright PNG: pixel u right, v down; optical X right Y down Z forward",
              "source": "simulator robot and camera calibration; no object state", "cameras": {}}
    for name in cameras:
        index = sim.model.camera_name2id(name)
        camera = camera_calibration(sim.data.cam_xpos[index], sim.data.cam_xmat[index],
                                    sim.model.cam_fovy[index], size)
        # A projected robot-axis reference helps interpret matrices; not object localization.
        camera["eef_pixel_uv"] = project_world(eef_position, camera)
        camera["eef_plus_5cm_world_axes_pixel_uv"] = {
            axis: project_world([x + (.05 if j == i else 0.) for j, x in enumerate(eef_position)], camera)
            for i, axis in enumerate("xyz")}
        result["cameras"][name] = camera
    return result
