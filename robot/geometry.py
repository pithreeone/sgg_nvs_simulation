"""
geometry.py -- the frame maths, with no controller attached.

AI2-THOR uses a left-handed Y-up frame; yaw is degrees clockwise from +Z, so the
forward vector is (sin(yaw), cos(yaw)) in XZ.

These lived in `robot_controller.py` and were imported from it by `eval_move`,
`viz/show_tasks` and three `build/sgg/` scripts -- none of which want a
`RobotController`, and two of which do not open a THOR scene at all.  Pulling
them out means "convert a pose to a heading" no longer costs a 430-line module
that owns a simulator connection.  `robot_controller` re-exports them, so the
`build/sgg/` and `archive/` call sites that ask it for them still work.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

def _xz(p: Dict[str, float]) -> np.ndarray:
    """Project a THOR position dict onto the horizontal plane."""
    return np.array([p["x"], p["z"]], dtype=float)


def _xyz(p: Dict[str, float]) -> np.ndarray:
    return np.array([p["x"], p["y"], p["z"]], dtype=float)


def yaw_towards(from_xz: np.ndarray, to_xz: np.ndarray) -> float:
    """Yaw (deg, THOR convention) that points from one XZ point at another."""
    delta = to_xz - from_xz
    return math.degrees(math.atan2(delta[0], delta[1])) % 360.0


def wrap_deg(angle: float) -> float:
    """Wrap an angle into [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def horizon_towards(camera_xyz: np.ndarray, target_xyz: np.ndarray) -> float:
    """
    Camera pitch that centres `target_xyz`, in THOR's cameraHorizon convention
    (POSITIVE = looking down).
    """
    horizontal = math.hypot(
        target_xyz[0] - camera_xyz[0], target_xyz[2] - camera_xyz[2]
    )
    drop = camera_xyz[1] - target_xyz[1]
    return math.degrees(math.atan2(drop, max(horizontal, 1e-6)))


def unproject(rc, pixel, depth: float, fov: float) -> np.ndarray:
    """
    A pixel plus its depth, in world coordinates.  Real camera, real depth.

    The inverse of what `horizon_towards`/`yaw_towards` above do for the
    optical axis, generalised to an arbitrary pixel.  Unity's conventions, the
    same ones `nvs_lemniscate.look_at_point` uses: yaw 0 faces +z, +pitch looks
    down, y is up.  THOR's `fieldOfView` is VERTICAL, so the focal length comes
    from the image height.
    """
    height, width = rc.event.frame.shape[:2]
    focal = (height / 2.0) / math.tan(math.radians(fov) / 2.0)
    yaw = math.radians(rc.agent_yaw)
    pitch = math.radians(rc.camera_horizon)

    forward = np.array([math.sin(yaw) * math.cos(pitch), -math.sin(pitch),
                        math.cos(yaw) * math.cos(pitch)])
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    up = np.array([math.sin(yaw) * math.sin(pitch), math.cos(pitch),
                   math.cos(yaw) * math.sin(pitch)])

    u = (pixel[0] - width / 2.0) / focal
    v = (pixel[1] - height / 2.0) / focal
    return rc.camera_xyz + depth * (forward + u * right - v * up)


def point_in_box(rc, box, fov: float) -> Optional[np.ndarray]:
    """
    The 3D point a detection box sits at, from the depth frame.

    Exists so a viewpoint search can be centred on something the robot SAW
    rather than on ground-truth object metadata.  The MEDIAN depth inside the
    box, not the depth at its centre: a detection box for a table contains a
    good deal of what is behind and on top of it, and the centre pixel is as
    likely to land on a plate as on the table.
    """
    depth = getattr(rc.event, "depth_frame", None)
    if depth is None:
        return None
    height, width = depth.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return unproject(rc, ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                     float(np.median(depth[y0:y1, x0:x1])), fov)
