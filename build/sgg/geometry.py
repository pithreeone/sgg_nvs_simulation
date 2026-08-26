"""
geometry.py -- camera maths for the scene builders, with no scene attached.

These were split between `thor_camera.py` and `build_occlusion_dataset.py`, both
of which are runnable tools: `export_multiview` wanted six functions out of the
first and `nvs_lemniscate` wanted `bbox_of` out of the second, so asking for a
projection meant importing 300 or 600 lines of argparse and render loop.

`intrinsics` returns the 3x3 matrix; `intrinsics_record` is the same maths as a
JSON-serialisable dict, which is what goes into a dataset record.  They were two
functions of one name in two files, and only the return type differed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


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


def intrinsics_record(width: int, height: int, fov_deg: float) -> Dict[str, float]:
    f = (height / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return {"fx": f, "fy": f, "cx": width / 2.0, "cy": height / 2.0,
            "width": width, "height": height, "fov_vertical_deg": fov_deg}


def bbox_of(mask: np.ndarray) -> Optional[List[int]]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2) + 1, int(y2) + 1]


def pick_focus(controller, seed: int, min_area: float = 0.15) -> Dict[str, float]:
    """
    The surface this seed builds around, rotated so seeds cover different rooms.

    `focus` fixes the whole region of interest: which surfaces get clutter, where
    the furniture may stand, and what every camera aims at.  It used to be
    whichever surface happened to be nearest THOR's default agent spawn, which is
    a property of the scene and not of the seed -- so three seeds of FloorPlan201
    rearranged the same dining table and photographed it from the same side,
    while the other five surfaces in the room were never visited.  That is the
    likely upstream cause of both the clustered viewpoints and `chair` taking 33%
    of all annotations: the focus landed on a dining table, and a dining table
    comes surrounded by chairs.

    Tiny surfaces are skipped -- a 0.2 m shelf can hold two objects and gives the
    cameras nothing to orbit.
    """
    surfaces = P.surfaces_near(controller.last_event,
                               controller.last_event.metadata["agent"]["position"])
    usable = []
    for surface in surfaces:
        size = surface["axisAlignedBoundingBox"]["size"]
        area = size["x"] * size["z"]
        if area >= min_area:
            usable.append((area, surface))
    if not usable:
        usable = [(0.0, s) for s in surfaces]
    if not usable:
        return dict(controller.last_event.metadata["agent"]["position"])
    # Largest first, then rotate by seed, so consecutive seeds land on different
    # surfaces rather than re-sampling the same one.
    usable.sort(key=lambda t: -t[0])

    # Deduplicate by ground position first.  Rotating over the raw list looked
    # like it spread the seeds and did not: a ShelvingUnit exposes several
    # `Shelf` receptacles stacked at the SAME (x, z), so `usable[1]`, `[2]` and
    # `[3]` were three shelves of one unit.  Everything downstream keys off the
    # focus's ground position -- `surfaces_near` orders clutter by it and the
    # cameras stand in a ring around it -- so those seeds rebuilt the same corner
    # of the room and their samples are correlated, which quietly overstates n.
    #
    # Measured over the 30 candidate scenes: 11 gave fewer than 3 distinct focus
    # regions across seeds 1-3, including FloorPlan206, which has 15 usable
    # surfaces and was yielding exactly one.
    seen, distinct = set(), []
    for area, surface in usable:
        key = (round(surface["position"]["x"], 2), round(surface["position"]["z"], 2))
        if key not in seen:
            seen.add(key)
            distinct.append(surface)
    return dict(distinct[seed % len(distinct)]["position"])
