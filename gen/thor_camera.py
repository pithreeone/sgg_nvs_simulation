"""
thor_camera.py -- AI2-THOR camera poses as w2c matrices, verified by reprojection.

Produces the pose format an NVS model expects:
    w2c   4x4 world-to-camera, OpenCV convention (X right, Y down, Z forward)
    K     3x3 intrinsics
    look_at, up_direction, fov (radians)

AI2-THOR runs on Unity: left-handed, Y up, and yaw measured clockwise from +Z.
Converting that to OpenCV is exactly the kind of thing where one sign error
produces plausible-looking garbage, so nothing here is taken on trust --
`verify_reprojection()` projects known object centroids through the matrix and
compares against `instance_detections2D`, and the exporter refuses to write poses
whose reprojection error is large.

Conventions established empirically (see verify_reprojection):
    forward = ( sin(yaw)cos(pitch), -sin(pitch), cos(yaw)cos(pitch) )
    right   = ( cos(yaw),            0,         -sin(yaw)          )
    down    = right x forward                     (OpenCV Y points down)
    pitch   = cameraHorizon, POSITIVE meaning looking down
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def camera_basis(yaw_deg: float, pitch_deg: float) -> Tuple[np.ndarray, ...]:
    """Right / down / forward unit vectors in Unity world coordinates."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    forward = np.array([
        math.sin(yaw) * math.cos(pitch),
        -math.sin(pitch),                 # cameraHorizon > 0 means looking down
        math.cos(yaw) * math.cos(pitch),
    ], dtype=float)
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=float)
    # right x forward, NOT forward x right.  Determined by search over all sign
    # and ordering combinations, scored by reprojection IoU of 3D bounding-box
    # corners against THOR's own instance_detections2D: this ordering scored
    # 0.525 mean IoU against 0.101 for the runner-up.  Deriving it by reasoning
    # about Unity's left-handedness got the sign wrong.
    down = np.cross(right, forward)

    for v in (forward, right, down):
        v /= max(np.linalg.norm(v), 1e-12)
    return right, down, forward


def w2c_matrix(
    camera_position: Sequence[float], yaw_deg: float, pitch_deg: float
) -> np.ndarray:
    """
    4x4 world-to-camera matrix in OpenCV convention.

    Rows of R are the camera axes expressed in world coordinates, so
    R @ (p_world - c) gives camera-space coordinates.
    """
    right, down, forward = camera_basis(yaw_deg, pitch_deg)
    centre = np.asarray(camera_position, dtype=float)

    rotation = np.stack([right, down, forward], axis=0)     # 3x3
    translation = -rotation @ centre

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def intrinsics(width: int, height: int, fov_vertical_deg: float) -> np.ndarray:
    """
    Pinhole intrinsics from the VERTICAL field of view.

    THOR passes `fieldOfView` straight to Unity's Camera.fieldOfView, which is
    vertical -- treating it as horizontal makes fx wrong by the aspect ratio.
    """
    fy = (height / 2.0) / math.tan(math.radians(fov_vertical_deg) / 2.0)
    fx = fy                                  # square pixels
    return np.array([
        [fx, 0.0, width / 2.0],
        [0.0, fy, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def project(points_world: np.ndarray, w2c: np.ndarray, K: np.ndarray) -> np.ndarray:
    """World points (N,3) -> pixel coordinates (N,2); z<=0 becomes NaN."""
    points = np.asarray(points_world, dtype=float).reshape(-1, 3)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    camera = (w2c @ homogeneous.T).T[:, :3]
    z = camera[:, 2:3]
    pixels = (K @ camera.T).T
    with np.errstate(invalid="ignore", divide="ignore"):
        pixels = pixels[:, :2] / pixels[:, 2:3]
    pixels[np.repeat(z <= 1e-6, 2, axis=1)] = np.nan
    return pixels


def look_at_point(
    camera_position: Sequence[float], yaw_deg: float, pitch_deg: float,
    distance: float = 2.0,
) -> np.ndarray:
    """A point along the optical axis, for NVS APIs that want an explicit target."""
    _, _, forward = camera_basis(yaw_deg, pitch_deg)
    return np.asarray(camera_position, dtype=float) + distance * forward


def up_direction(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    _, down, _ = camera_basis(yaw_deg, pitch_deg)
    return -down


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _bbox_corners(obj: Dict[str, Any]) -> Optional[np.ndarray]:
    box = obj.get("axisAlignedBoundingBox") or {}
    centre, size = box.get("center"), box.get("size")
    if not centre or not size:
        return None
    cx, cy, cz = centre["x"], centre["y"], centre["z"]
    hx, hy, hz = size["x"] / 2, size["y"] / 2, size["z"] / 2
    return np.array([[cx + dx * hx, cy + dy * hy, cz + dz * hz]
                     for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)])


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def verify_reprojection(
    robot, min_fill: float = 0.7, edge_margin: int = 4, min_box_px: int = 3000
) -> Dict[str, Any]:
    """
    Project each object's 3D bounding-box corners and compare the resulting 2D
    extent with THOR's `instance_detections2D`, scoring by IoU.

    Objects are excluded when the comparison would be unfair rather than wrong:

    * partly occluded (`min_fill`) -- THOR's 2D box bounds the VISIBLE mask, a
      projected 3D box bounds the whole object including hidden parts
    * touching the frame edge -- THOR clips its box at the image border while the
      projected 3D box runs off screen, so IoU collapses even for a perfect
      projection.  This alone was scoring Windows at 0.000.
    * small (`min_box_px`) -- an axis-aligned 3D box is a good shape proxy for a
      cabinet or a shelf and a poor one for a vase or a saltshaker, and on a small
      object a few pixels of discrepancy is a large fraction of the box.  Measured
      with a correct projection: Cabinet 0.916, Shelf 0.935, Towel 0.900 versus
      SaltShaker 0.109, Vase 0.347 -- the spread is about object shape, not about
      the projection.

    A correct convention scores high here.  The ordering was chosen by exhaustive
    search over sign and cross-product combinations: 0.525 for this one against
    0.101 for the runner-up and ~0.00-0.04 for the rest, so the identification is
    unambiguous even though the absolute value depends on these filters.
    """
    event = robot.event
    detections = getattr(event, "instance_detections2D", None) or {}
    masks = getattr(event, "instance_masks", None) or {}
    if not detections:
        return {"ok": False, "reason": "instance segmentation not rendered"}

    width, height = robot.width, robot.height
    w2c = w2c_matrix(robot.camera_xyz, robot.agent_yaw, robot.camera_horizon)
    K = intrinsics(width, height, robot.fov_vertical)
    by_id = {o["objectId"]: o for o in event.metadata["objects"]}

    ious, details, skipped = [], [], {"occluded": 0, "edge": 0, "behind": 0,
                                      "small": 0}
    for object_id, box in detections.items():
        obj = by_id.get(object_id)
        corners = _bbox_corners(obj) if obj else None
        if corners is None:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)

        if (x2 - x1) * (y2 - y1) < min_box_px:
            skipped["small"] += 1
            continue

        if (x1 <= edge_margin or y1 <= edge_margin
                or x2 >= width - edge_margin or y2 >= height - edge_margin):
            skipped["edge"] += 1
            continue

        mask = masks.get(object_id)
        if mask is not None:
            bbox_px = max((x2 - x1) * (y2 - y1), 1.0)
            if np.count_nonzero(mask) / bbox_px < min_fill:
                skipped["occluded"] += 1
                continue

        pixels = project(corners, w2c, K)
        if np.isnan(pixels).any():
            skipped["behind"] += 1
            continue

        projected = [pixels[:, 0].min(), pixels[:, 1].min(),
                     pixels[:, 0].max(), pixels[:, 1].max()]
        value = _iou(projected, [x1, y1, x2, y2])
        ious.append(value)
        details.append((obj["objectType"], round(value, 3)))

    if not ious:
        return {"ok": False, "reason": f"no scorable objects (skipped {skipped})"}

    median = float(np.median(ious))
    return {
        "ok": median > 0.75,
        "median_iou": round(median, 3),
        "mean_iou": round(float(np.mean(ious)), 3),
        "n": len(ious),
        "skipped": skipped,
        "worst": sorted(details, key=lambda d: d[1])[:3],
        "best": sorted(details, key=lambda d: -d[1])[:3],
        "w2c": w2c,
        "K": K,
    }


def pose_record(robot, distance: float = 2.0) -> Dict[str, Any]:
    """Everything an NVS model needs about the current viewpoint, JSON-ready."""
    camera = robot.camera_xyz
    yaw, pitch = robot.agent_yaw, robot.camera_horizon
    w2c = w2c_matrix(camera, yaw, pitch)
    K = intrinsics(robot.width, robot.height, robot.fov_vertical)
    return {
        "camera_position": [round(float(v), 5) for v in camera],
        "yaw_deg": round(float(yaw), 3),
        "pitch_deg": round(float(pitch), 3),
        "fov_vertical_rad": round(math.radians(robot.fov_vertical), 6),
        "fov_vertical_deg": round(float(robot.fov_vertical), 3),
        "fov_horizontal_deg": round(float(robot.fov_horizontal), 3),
        "w2c": [[round(float(v), 6) for v in row] for row in w2c],
        "c2w": [[round(float(v), 6) for v in row]
                for row in np.linalg.inv(w2c)],
        "K": [[round(float(v), 4) for v in row] for row in K],
        "look_at": [round(float(v), 5)
                    for v in look_at_point(camera, yaw, pitch, distance)],
        "up_direction": [round(float(v), 5) for v in up_direction(yaw, pitch)],
        "convention": "opencv_x_right_y_down_z_forward",
    }


def relative_pose(w2c_from: np.ndarray, w2c_to: np.ndarray) -> np.ndarray:
    """
    Transform taking camera `from` to camera `to`.

    This is what a pose-conditioned NVS model consumes: the target view expressed
    relative to the reference view rather than in world coordinates.
    """
    return np.asarray(w2c_to) @ np.linalg.inv(np.asarray(w2c_from))


def baseline_metrics(w2c_ref: np.ndarray, w2c_tgt: np.ndarray,
                     look_at: Sequence[float]) -> Dict[str, float]:
    """
    How far apart two views are -- the independent variable for a baseline sweep.

    `azimuth_deg` / `elevation_deg` are measured about the look-at point, so they
    are directly comparable with an NVS model's --max_az / --max_el settings.
    """
    c_ref = -np.asarray(w2c_ref)[:3, :3].T @ np.asarray(w2c_ref)[:3, 3]
    c_tgt = -np.asarray(w2c_tgt)[:3, :3].T @ np.asarray(w2c_tgt)[:3, 3]
    target = np.asarray(look_at, dtype=float)

    v_ref, v_tgt = c_ref - target, c_tgt - target
    n_ref = max(np.linalg.norm(v_ref), 1e-9)
    n_tgt = max(np.linalg.norm(v_tgt), 1e-9)

    # azimuth in the horizontal plane, elevation out of it
    az_ref = math.degrees(math.atan2(v_ref[0], v_ref[2]))
    az_tgt = math.degrees(math.atan2(v_tgt[0], v_tgt[2]))
    el_ref = math.degrees(math.asin(np.clip(v_ref[1] / n_ref, -1, 1)))
    el_tgt = math.degrees(math.asin(np.clip(v_tgt[1] / n_tgt, -1, 1)))

    return {
        "translation_m": round(float(np.linalg.norm(c_tgt - c_ref)), 4),
        "azimuth_deg": round(float((az_tgt - az_ref + 180) % 360 - 180), 3),
        "elevation_deg": round(float(el_tgt - el_ref), 3),
        "range_ratio": round(float(n_tgt / n_ref), 4),
        "angle_between_deg": round(float(math.degrees(math.acos(
            np.clip(float(v_ref @ v_tgt) / (n_ref * n_tgt), -1, 1)))), 3),
    }


if __name__ == "__main__":
    # Self-check: does the w2c convention actually reproject correctly?
    from robot.robot_controller import RobotController

    robot = RobotController(scene="FloorPlan1", width=1024, height=768,
                            headless=True, verbose=False)
    print(f"{'scene':<14}{'n':>4}{'median IoU':>12}{'mean':>8}  worst")
    try:
        for scene in ("FloorPlan1", "FloorPlan201", "FloorPlan301", "FloorPlan401"):
            robot.reset(scene)
            reachable = robot.get_reachable_positions(refresh=True)
            spot = reachable[len(reachable) // 3]
            robot.teleport(
                position={"x": float(spot[0]), "y": robot.agent_position["y"],
                          "z": float(spot[1])},
                yaw=135.0, horizon=20.0, standing=True,
            )
            result = verify_reprojection(robot)
            worst = ", ".join(f"{t}:{v}" for t, v in result.get("worst", []))
            print(f"{scene:<14}{result.get('n', 0):>4}"
                  f"{result.get('median_iou', float('nan')):>12.3f}"
                  f"{result.get('mean_iou', float('nan')):>8.3f}  {worst}")
            print(f"{'':<14}{'OK' if result['ok'] else 'FAILED'}"
                  f"   skipped={result.get('skipped')}  best={result.get('best')}")
    finally:
        robot.stop()
