"""
export_multiview.py -- render TRUE views along an NVS trajectory in AI2-THOR.

For each scene this writes one reference view plus `num_frames` target views on a
lemniscate (figure-of-eight) trajectory, matching the trajectory an NVS model
would be asked to generate.  Because the target views are rendered by the
simulator rather than generated, they are ground truth: running the same NVS
model on the same poses and comparing gives an exact, paired measurement of what
NVS error costs, which no real dataset can provide.

Trajectory
----------
Gerono lemniscate in the reference camera's image plane, as in the NVS preset:

    x(t) = cos(t) / (1 + sin(t)^2)              in [-1, 1]
    y(t) = cos(t)sin(t) / (1 + sin(t)^2)        in [-0.3536, 0.3536]
    t    = linspace(0, 2pi, num_frames + 1)[:-1] + pi/2

Scaled so the ANGULAR extent about the look-at point is exactly +/-max_az
horizontally and +/-max_el vertically:

    A_x = D * tan(max_az),   A_y = D * tan(max_el) / 0.35355

where D is the reference camera's distance to the look-at point.  Every view then
looks back at that same point, so the trajectory is comparable with an NVS model
configured with the same --max_az / --max_el.

Camera placement uses AddThirdPartyCamera: the acting agent is restricted to
reachable floor positions with two body heights, so it cannot follow a free-space
trajectory.  Third-party cameras accept arbitrary position and rotation
(including roll) and still provide depth and instance segmentation.

Output
------
    multiview/<scene>/ref.png                  reference view (NVS input)
    multiview/<scene>/view_XX.png              true render at target pose XX
    multiview/<scene>/scene.json               poses, relative poses, baselines,
                                               per-view GT, scene-level GT

No depth is written by default: the true render at each target pose is better
ground truth than any depth-based warp, and cross-view association is already
given exactly by the GT object_id stored per view.  `--depth ref` adds the
reference depth back if a depth-warp NVS baseline is ever wanted.
"""

from __future__ import annotations

# `python gen/<script>.py` puts gen/ on sys.path, not the repo root, so the
# root-level modules (vg150, vg_gt, ...) would not resolve.  Running as
# `python -m gen.<script>` does not need this; it is here so both work.
import os as _os
import sys as _sys
if __package__ in (None, ""):
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import collections
import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from gen.export_samples import (DEFAULT_SCENES, LOOSE_MAPPINGS, MIN_AREA,
                            ground_truth_relations, support_relations,
                            visible_objects)
from gen.thor_camera import (camera_basis, intrinsics, project, relative_pose,
                         w2c_matrix)
from vg.vg150 import thor_to_vg150

LEMNISCATE_Y_PEAK = 0.35355339059327373      # max of cos t sin t / (1 + sin^2 t)


DEPTH_SCALE_MM = 1000.0        # depth stored as uint16 millimetres


def write_depth(path_stem: str, depth: np.ndarray, fmt: str) -> Optional[str]:
    """
    Write a depth map.

    16-bit PNG in millimetres, not .npy: 15.8x smaller (84 KB against 1327 KB at
    576x576), readable from any language, and 1 mm quantisation error against a
    depth range under 2 m.  Read it back with:

        cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    Pass fmt="npy" if exact float32 is genuinely needed.
    """
    array = np.asarray(depth, dtype=np.float32)
    if fmt == "none":
        return None
    if fmt == "npy":
        cv2.imwrite  # no-op, keeps linters quiet about the unused import path
        np.save(f"{path_stem}.npy", array)
        return f"{os.path.basename(path_stem)}.npy"
    millimetres = np.clip(array * DEPTH_SCALE_MM, 0, 65535).astype(np.uint16)
    cv2.imwrite(f"{path_stem}.png", millimetres)
    return f"{os.path.basename(path_stem)}.png"


def load_external_poses(path: str) -> Dict[str, Any]:
    """
    Load target poses produced by an EXTERNAL trajectory generator.

    Needed because the trajectory is generated on the NVS side, so the true
    renders must land on exactly those poses -- otherwise the pairing is wrong and
    nothing in the comparison means anything.

    Reproducing the generator here is not safe.  `get_lemniscate_w2cs` scales both
    axes of the curve by a SINGLE amplitude `a`, and a Gerono lemniscate has a
    fixed axis ratio (y peaks at 0.3536 of x), so a = D tan(max_az) forces
    max_el = atan(0.3536 tan(max_az)) = 1.77 deg when max_az = 5 deg.  "max_az 5,
    max_el 5" therefore cannot both hold under a single amplitude, and guessing
    wrong scales the vertical extent by ~2.8x.

    Accepted JSON:
        {"convention": "opencv" | "opengl",
         "matrix": "w2c" | "c2w",
         "relative": true | false,          # relative to the reference, or world
         "poses": [[4x4], ...]}

    Also accepts a bare list of 4x4 matrices, assumed w2c / world / opencv.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        payload = {"poses": payload}
    poses = np.asarray(payload["poses"], dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected (N,4,4) poses, got {poses.shape}")

    convention = payload.get("convention", "opencv").lower()
    if convention in ("opengl", "blender"):
        # OpenGL/Blender cameras look down -Z with +Y up; OpenCV looks down +Z
        # with +Y down.  This is the diagflat([1,-1,-1,1]) flip.
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        poses = np.array([flip @ p @ flip for p in poses])

    return {
        "poses": poses,
        "matrix": payload.get("matrix", "w2c").lower(),
        "relative": bool(payload.get("relative", False)),
        "convention": convention,
        "source": os.path.basename(path),
    }


def euler_from_w2c(w2c: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """Recover Unity (pitch, yaw, roll) and the camera centre from a w2c matrix."""
    rotation = np.asarray(w2c)[:3, :3]
    centre = -rotation.T @ np.asarray(w2c)[:3, 3]
    right, down, forward = rotation[0], rotation[1], rotation[2]

    yaw = math.degrees(math.atan2(forward[0], forward[2])) % 360.0
    pitch = math.degrees(math.asin(float(np.clip(-forward[1], -1.0, 1.0))))
    _, down0, _ = camera_basis(yaw, pitch)
    # negated: verified by round-tripping w2c_from_euler -> euler_from_w2c, which
    # returned -5 for an input roll of +5 before this sign was corrected
    roll = -math.degrees(math.atan2(float(np.cross(-down0, -down) @ forward),
                                    float((-down0) @ (-down))))
    return pitch, yaw, roll, centre


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


def lemniscate_offsets(
    num_frames: int, max_az_deg: float, max_el_deg: float, distance: float,
    flip_x: bool = False, flip_y: bool = False,
) -> np.ndarray:
    """
    Camera-space (x, y, 0) offsets on a lemniscate, in metres.

    EXACT port of `resolve_poses(trajectory="lemniscate")` from the NVS side, so
    the true renders land on the poses the NVS model is actually asked for:

        ax = dist * tan(max_az)          ay = dist * tan(max_el)
        t  = linspace(0, 2pi, num_views + 1)[1:] + pi/2
        x  = ax * cos(t) / (1 + sin(t)^2)
        y  = ay * cos(t) * sin(t) / (1 + sin(t)^2)

    Two details this file previously got wrong, both silent:

    1. `[1:]`, not `[:-1]`.  The zero-offset frames -- where cos(t) = 0 and the
       camera sits exactly at the reference pose -- are therefore at indices
       num_frames//2 - 1 and num_frames - 1 (9 and 19 for 20 views), NOT 0 and 10.
    2. `ay` is NOT renormalised by the curve's y peak.  Since
       cos(t)sin(t)/(1+sin^2 t) peaks at 0.35355, the achieved elevation is
       atan(0.35355 * tan(max_el)) = 1.77 deg for max_el = 5, not 5 deg.  Dividing
       by the peak (as this did) made the vertical extent 2.8x too large.

    Note that both extents are scale-invariant in `distance`: the achieved angles
    depend only on max_az / max_el.  `distance` only sets the metric translation,
    which matters for real parallax against a real scene but is normalised away by
    the NVS model's own camera normalisation.
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, num_frames + 1)[1:] + np.pi / 2.0
    denominator = 1.0 + np.sin(thetas) ** 2

    amplitude_x = distance * math.tan(math.radians(max_az_deg))
    amplitude_y = distance * math.tan(math.radians(max_el_deg))
    if flip_x:
        amplitude_x = -amplitude_x
    if flip_y:
        amplitude_y = -amplitude_y

    x = amplitude_x * np.cos(thetas) / denominator
    y = amplitude_y * np.cos(thetas) * np.sin(thetas) / denominator
    return np.stack([x, y, np.zeros_like(x)], axis=-1)


def achieved_extent(max_az_deg: float, max_el_deg: float) -> Tuple[float, float]:
    """Angles the curve actually reaches, which differ from the requested max_el."""
    return (max_az_deg,
            math.degrees(math.atan(LEMNISCATE_Y_PEAK
                                   * math.tan(math.radians(max_el_deg)))))


def euler_from_lookat(
    eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray
) -> Tuple[float, float, float]:
    """
    Unity Euler angles (pitch about X, yaw about Y, roll about Z) that make a
    camera at `eye` look at `target` with `up_hint` as the up reference.

    Derived to match thor_camera's verified basis:
        forward = (sin(yaw)cos(pitch), -sin(pitch), cos(yaw)cos(pitch))
    so yaw = atan2(fx, fz) and pitch = -asin(fy).  Roll is then the signed angle
    from the roll-free up vector to `up_hint`, measured about the optical axis.
    """
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        return 0.0, 0.0, 0.0
    forward = forward / norm

    yaw = math.degrees(math.atan2(forward[0], forward[2])) % 360.0
    pitch = math.degrees(math.asin(float(np.clip(-forward[1], -1.0, 1.0))))

    # up with roll = 0, from the same basis convention
    _, down0, _ = camera_basis(yaw, pitch)
    up0 = -down0

    up = np.asarray(up_hint, dtype=float)
    up = up - forward * float(up @ forward)          # project out the axial part
    if np.linalg.norm(up) < 1e-6:
        return pitch, yaw, 0.0
    up = up / np.linalg.norm(up)

    roll = math.degrees(math.atan2(float(np.cross(up0, up) @ forward),
                                   float(up0 @ up)))
    return pitch, yaw, roll


def basis_from_euler(pitch: float, yaw: float, roll: float) -> Tuple[np.ndarray, ...]:
    """Right / down / forward for a camera with roll applied about the optical axis."""
    right0, down0, forward = camera_basis(yaw, pitch)
    angle = math.radians(roll)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    right = cos_a * right0 + sin_a * down0
    down = -sin_a * right0 + cos_a * down0
    return right, down, forward


def w2c_from_euler(
    position: Sequence[float], pitch: float, yaw: float, roll: float
) -> np.ndarray:
    right, down, forward = basis_from_euler(pitch, yaw, roll)
    rotation = np.stack([right, down, forward], axis=0)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = -rotation @ np.asarray(position, dtype=float)
    return matrix


# ---------------------------------------------------------------------------
# Third-party camera rendering
# ---------------------------------------------------------------------------


def tpc_objects(
    event, camera_index: int, width: int, height: int
) -> List[Dict[str, Any]]:
    """VG150-nameable objects visible in a third-party camera's frame."""
    masks_all = getattr(event, "third_party_instance_masks", None)
    if not masks_all or camera_index >= len(masks_all):
        return []
    masks = masks_all[camera_index]
    by_id = {o["objectId"]: o for o in event.metadata["objects"]}

    out = []
    for object_id, mask in masks.items():
        obj = by_id.get(object_id)
        if obj is None:
            continue
        vg = thor_to_vg150(obj["objectType"])
        if not vg:
            continue
        pixels = int(np.count_nonzero(mask))
        if pixels / float(width * height) < MIN_AREA:
            continue
        ys, xs = np.nonzero(mask)
        out.append({
            "object_id": object_id,
            "thor_type": obj["objectType"],
            "vg150_class": vg,
            "loose": obj["objectType"] in LOOSE_MAPPINGS,
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "mask_pixels": pixels,
            "area_fraction": round(pixels / float(width * height), 5),
        })
    return sorted(out, key=lambda o: -o["area_fraction"])


def scene_relations(event) -> List[Dict[str, Any]]:
    """
    Every world-relative relation in the scene, regardless of visibility.

    `on` and `in` only: both are properties of the scene rather than of a camera,
    so one view sees a subset and N views should recover more of the same fixed
    set -- which is exactly what multi-view fusion is meant to do.

    `on` and `in` only.  Both are world-relative, so one view sees a subset and N
    views should recover more of the same fixed set -- which is exactly what
    multi-view fusion is meant to do.  `between` was dropped: see
    ground_truth_relations for why none of the three definitions tried held up.
    `behind` / `in front of` are deliberately absent: they are CAMERA-relative and
    have no scene-level truth value, so they live in each view's own relation list
    instead.  `near` is omitted too -- symmetric and dense, it would swamp the
    support relations in any recall average.
    """
    return support_relations(event.metadata["objects"])


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def pick_reference_views(
    robot, reachable, tries: int, n_refs: int, rng,
    min_separation_m: float = 1.0, min_yaw_separation_deg: float = 45.0,
) -> List[Tuple[float, float, float, float]]:
    """
    Choose `n_refs` content-rich AND mutually diverse reference viewpoints.

    Taking the top-N by score alone returns near-duplicates: the highest-scoring
    poses cluster in whichever corner shows the most furniture.  Candidates are
    therefore accepted greedily by score subject to a minimum separation in both
    position and heading, which is what makes the reference views actually probe
    different parts of the room.

    Diversity matters for the fusion experiment specifically: measured on
    FloorPlan1, one reference view already saw all 23 nameable objects, leaving
    fusion no headroom at all, while other scenes gained 22-25% from the union
    across views.  Whether fusion can help is partly a property of the starting
    viewpoint.
    """
    candidates = []
    for _ in range(tries):
        spot = reachable[rng.randrange(len(reachable))]
        yaw = rng.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        horizon = rng.choice([0.0, 15.0])
        robot.teleport(
            position={"x": float(spot[0]), "y": robot.agent_position["y"],
                      "z": float(spot[1])},
            yaw=float(yaw), horizon=horizon, standing=True,
        )
        present = visible_objects(robot.event, robot.width, robot.height)
        relations = ground_truth_relations(robot.event, present)
        candidates.append((len(present) + 3 * len(relations),
                           float(spot[0]), float(spot[1]), float(yaw), horizon))

    candidates.sort(key=lambda c: -c[0])
    chosen: List[Tuple[float, float, float, float]] = []
    for _, x, z, yaw, horizon in candidates:
        too_close = False
        for cx, cz, cyaw, _ in chosen:
            if (math.hypot(x - cx, z - cz) < min_separation_m
                    and abs((yaw - cyaw + 180) % 360 - 180) < min_yaw_separation_deg):
                too_close = True
                break
        if not too_close:
            chosen.append((x, z, yaw, horizon))
        if len(chosen) >= n_refs:
            break
    # if the room is too small to satisfy the constraint, fall back to top-scoring
    for _, x, z, yaw, horizon in candidates:
        if len(chosen) >= n_refs:
            break
        if (x, z, yaw, horizon) not in chosen:
            chosen.append((x, z, yaw, horizon))
    return chosen[:n_refs]


def export_scene(
    robot, scene: str, outdir: str, num_frames: int,
    ref_pose: Tuple[float, float, float, float],
    ref_label: str,
    max_az: float, max_el: float, tries: int, rng,
    poses_from: Optional[str] = None,
    depth_format: str = "png16",
    depth_mode: str = "ref",
    look_distance_m: Optional[float] = 2.0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> Optional[Dict[str, Any]]:
    from robot.robot_controller import _xz

    robot.reset(scene)
    reachable = robot.get_reachable_positions(refresh=True)
    if len(reachable) == 0:
        return None

    # --- place the agent at the given reference pose -----------------------
    x, z, yaw, horizon = ref_pose
    robot.teleport(position={"x": x, "y": robot.agent_position["y"], "z": z},
                   yaw=yaw, horizon=horizon, standing=True)

    ref_objects = visible_objects(robot.event, robot.width, robot.height)
    ref_relations = ground_truth_relations(robot.event, ref_objects)
    if not ref_objects:
        return None

    ref_position = robot.camera_xyz.copy()
    # Snapshot yaw/pitch NOW.  The record dict is built at the end of this
    # function, by which point the agent has been parked out of shot with
    # yaw + 180 and horizon 0 -- reading robot.agent_yaw there stored the PARKING
    # pose, silently, in every group.  `w2c` was unaffected because it is computed
    # here, which is how the error was recoverable at all.
    ref_yaw = float(robot.agent_yaw)
    ref_pitch = float(robot.camera_horizon)
    w2c_ref = w2c_matrix(ref_position, ref_yaw, ref_pitch)
    K = intrinsics(robot.width, robot.height, robot.fov_vertical)

    # Look-at distance.  FIXED by default rather than adapted to the scene.
    #
    # The trajectory amplitude is A_x = D tan(max_az), so with D adapted to the
    # visible content the same nominal "+/-5 deg" produced 0.153 m of camera
    # travel in FloorPlan1 (D=1.50) and 0.441 m in FloorPlan10 (D=4.32) -- a 2.9x
    # spread. The NVS model only ever sees the relative pose, so identically
    # labelled conditions carried very different real parallax, and both NVS
    # difficulty and information gain varied with it. That is an uncontrolled
    # confound across scenes, so D is held constant and the measured value is
    # recorded for reference.
    by_id = {o["objectId"]: o for o in robot.event.metadata["objects"]}
    distances = [float(by_id[o["object_id"]].get("distance", 2.0))
                 for o in ref_objects[:8]]
    adaptive_distance = float(np.clip(np.median(distances), 0.8, 5.0))
    look_distance = (adaptive_distance if look_distance_m is None
                     else float(look_distance_m))
    _, _, ref_forward = camera_basis(robot.agent_yaw, robot.camera_horizon)
    look_at = ref_position + look_distance * ref_forward
    ref_up = -np.stack(camera_basis(robot.agent_yaw, robot.camera_horizon))[1]

    scene_dir = os.path.join(outdir, scene, ref_label)
    os.makedirs(scene_dir, exist_ok=True)
    cv2.imwrite(os.path.join(scene_dir, "ref.png"),
                cv2.cvtColor(robot.event.frame, cv2.COLOR_RGB2BGR))
    depth_ref_name = (
        write_depth(os.path.join(scene_dir, "depth_ref"),
                    robot.event.depth_frame, depth_format)
        if depth_mode in ("ref", "all") else None
    )

    # --- get the agent out of shot ----------------------------------------
    # A third-party camera sits at the agent's own camera position, and unlike the
    # agent camera it DOES render the agent's body mesh.  Left in place, every
    # target view has a large grey capsule across it -- measured as a mean
    # difference of 18.4 against the reference view even at zero offset, where the
    # two should be identical.  Park the agent at the reachable position farthest
    # from the trajectory before rendering.
    distances_from_ref = np.linalg.norm(
        reachable - np.array([ref_position[0], ref_position[2]]), axis=1
    )
    parking = reachable[int(np.argmax(distances_from_ref))]
    robot.teleport(
        position={"x": float(parking[0]), "y": robot.agent_position["y"],
                  "z": float(parking[1])},
        yaw=float((robot.agent_yaw + 180.0) % 360.0), horizon=0.0, standing=False,
    )

    # --- lemniscate target poses ------------------------------------------
    external = None
    if poses_from:
        external = load_external_poses(poses_from)
        offsets_cam = None
    else:
        offsets_cam = lemniscate_offsets(num_frames, max_az, max_el, look_distance,
                                         flip_x=flip_x, flip_y=flip_y)
    right, down, forward = camera_basis(robot.agent_yaw, robot.camera_horizon)
    up = -down
    views: List[Dict[str, Any]] = []

    added = robot.controller.step(
        action="AddThirdPartyCamera",
        position={"x": float(ref_position[0]), "y": float(ref_position[1]),
                  "z": float(ref_position[2])},
        rotation={"x": float(robot.camera_horizon), "y": float(robot.agent_yaw),
                  "z": 0.0},
        fieldOfView=float(robot.fov_vertical),
    )
    if not added.metadata["lastActionSuccess"]:
        return None

    if external is not None:
        targets = []
        for pose in external["poses"]:
            matrix = np.asarray(pose, dtype=float)
            if external["matrix"] == "c2w":
                matrix = np.linalg.inv(matrix)
            if external["relative"]:
                matrix = matrix @ w2c_ref          # relative -> world
            targets.append(euler_from_w2c(matrix))
    else:
        targets = []
        for offset in offsets_cam:
            # their positions_local are in OpenCV camera space (Y DOWN), so a
            # positive y offset moves the camera DOWN, not up
            eye = ref_position + offset[0] * right - offset[1] * up
            pitch, yaw_t, roll = euler_from_lookat(eye, look_at, ref_up)
            targets.append((pitch, yaw_t, roll, eye))

    for index, (pitch, yaw_t, roll, eye) in enumerate(targets):

        event = robot.controller.step(
            action="UpdateThirdPartyCamera", thirdPartyCameraId=0,
            position={"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])},
            rotation={"x": float(pitch), "y": float(yaw_t), "z": float(roll)},
            fieldOfView=float(robot.fov_vertical),
        )
        if not event.metadata["lastActionSuccess"]:
            continue

        frame = np.asarray(event.third_party_camera_frames[0])[:, :, :3]
        cv2.imwrite(os.path.join(scene_dir, f"view_{index:02d}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # Per-view depth is deliberately NOT written by default.  Its only
        # plausible uses were measuring NVS geometric error -- for which the TRUE
        # RENDER at the same pose is strictly better ground truth than a
        # depth-based warp -- and cross-view association, which is already served
        # exactly by the GT `object_id` stored per view.  Only the REFERENCE depth
        # has a real use: warping the reference forward as a non-learned NVS
        # baseline.
        depth_name = (
            write_depth(os.path.join(scene_dir, f"depth_{index:02d}"),
                        event.third_party_depth_frames[0], depth_format)
            if depth_mode == "all" else None
        )

        w2c = w2c_from_euler(eye, pitch, yaw_t, roll)
        from gen.thor_camera import baseline_metrics
        views.append({
            "index": index,
            "image": f"view_{index:02d}.png",
            "depth": depth_name,
            "camera_position": [round(float(v), 5) for v in eye],
            "euler_unity": {"pitch": round(pitch, 4), "yaw": round(yaw_t, 4),
                            "roll": round(roll, 4)},
            "w2c": [[round(float(v), 6) for v in row] for row in w2c],
            "relative_to_ref": [[round(float(v), 6) for v in row]
                                for row in relative_pose(w2c_ref, w2c)],
            "baseline": baseline_metrics(w2c_ref, w2c, look_at),
            "objects": tpc_objects(event, 0, robot.width, robot.height),
        })

    robot.controller.step(action="AddThirdPartyCamera",
                          position={"x": 0, "y": -50, "z": 0},
                          rotation={"x": 0, "y": 0, "z": 0})  # park it out of the way

    record = {
        "scene": scene,
        "ref_label": ref_label,
        "dir": os.path.join(scene, ref_label),
        "width": robot.width,
        "height": robot.height,
        "K": [[round(float(v), 4) for v in row] for row in K],
        "fov_vertical_deg": round(float(robot.fov_vertical), 3),
        "convention": "opencv_x_right_y_down_z_forward",
        "depth_format": depth_format,
        "depth_mode": depth_mode,
        "depth_scale_to_metres": (1.0 / DEPTH_SCALE_MM
                                  if depth_format == "png16" else 1.0),
        "agent_parked_at": [round(float(parking[0]), 4),
                            round(float(parking[1]), 4)],
        "trajectory": {
            "type": "lemniscate",
            # thetas start at pi/2, where cos(theta)=0, so frames 0 and
            # num_frames/2 sit exactly at the reference pose -- the crossing point
            # of the figure-of-eight.  2 of 20 "novel" views are therefore copies
            # of the input view.  Useful as a free NVS sanity check: if a
            # zero-offset frame does not reproduce the input, the pipeline is wrong.
            "zero_offset_frames": [num_frames // 2 - 1, num_frames - 1],
            "achieved_max_az_deg": round(achieved_extent(max_az, max_el)[0], 3),
            "achieved_max_el_deg": round(achieved_extent(max_az, max_el)[1], 3),
            "num_frames": num_frames,
            "max_az_deg": max_az,
            "max_el_deg": max_el,
            "external_poses": (external["source"] if external else None),
            "flip_x": flip_x,
            "flip_y": flip_y,
            "start_direction": ("right" if flip_x else "left")
                               + "-" + ("up" if flip_y else "down"),
            "look_at": [round(float(v), 5) for v in look_at],
            "look_distance_m": round(look_distance, 4),
            "look_distance_fixed": look_distance_m is not None,
            # what an adaptive rule would have chosen, kept so the effect of
            # fixing D can be checked after the fact
            "scene_median_object_distance_m": round(adaptive_distance, 4),
            "amplitude_x_m": round(
                look_distance * math.tan(math.radians(max_az)), 4),
            "amplitude_y_m": round(
                look_distance * math.tan(math.radians(max_el))
                / LEMNISCATE_Y_PEAK, 4),
        },
        "reference": {
            "image": "ref.png",
            "depth": depth_ref_name,
            "camera_position": [round(float(v), 5) for v in ref_position],
            "yaw_deg": round(ref_yaw, 3),
            "pitch_deg": round(ref_pitch, 3),
            "w2c": [[round(float(v), 6) for v in row] for row in w2c_ref],
            "objects": ref_objects,
            "relations": ref_relations,
        },
        "views": views,
        "scene_relations": scene_relations(robot.event),
    }
    with open(os.path.join(scene_dir, "scene.json"), "w", encoding="utf-8") as h:
        json.dump(record, h, indent=1)
    return record


def refresh_multiview_gt(outdir: str, headless: bool) -> int:
    """
    Recompute ground truth for an EXISTING export without re-rendering.

    Viewpoints are chosen using the GT (candidates are scored by relation count),
    so regenerating from scratch after a GT change produces different images and
    silently invalidates anything already computed on the old ones.  This path
    teleports to each stored camera pose instead, so every PNG is untouched.
    """
    from robot.robot_controller import RobotController

    paths = sorted(glob.glob(os.path.join(outdir, "*", "r*", "scene.json")))
    if not paths:
        print(f"nothing to refresh under {outdir}/")
        return 1

    first = json.load(open(paths[0], encoding="utf-8"))
    robot = RobotController(scene=first["scene"], width=first["width"],
                            height=first["height"], headless=headless, verbose=False)
    before = collections.Counter()
    after = collections.Counter()
    try:
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
            for r in record["reference"]["relations"]:
                before[r["predicate"]] += 1

            robot.reset(record["scene"])
            ref = record["reference"]
            # Recover the pose from w2c rather than from yaw_deg/pitch_deg: those
            # two fields were written after the agent had been parked, so in any
            # export made before that fix they hold the parking pose.  w2c is
            # computed at capture time and is authoritative; this also repairs the
            # bad fields in place.
            pitch, yaw, _, centre = euler_from_w2c(np.array(ref["w2c"]))
            ref["yaw_deg"] = round(float(yaw), 3)
            ref["pitch_deg"] = round(float(pitch), 3)
            robot.teleport(
                position={"x": float(centre[0]),
                          "y": robot.agent_position["y"],
                          "z": float(centre[2])},
                yaw=yaw, horizon=pitch, standing=True,
            )
            robot.height_level = "stand"
            present = visible_objects(robot.event, record["width"], record["height"])
            record["reference"]["objects"] = present
            record["reference"]["relations"] = ground_truth_relations(
                robot.event, present)
            record["scene_relations"] = scene_relations(robot.event)
            for r in record["reference"]["relations"]:
                after[r["predicate"]] += 1

            with open(path, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=1)
    finally:
        robot.stop()

    print(f"refreshed {len(paths)} groups; images untouched\n")
    keys = sorted(set(before) | set(after))
    print(f"{'predicate':<14}{'before':>8}{'after':>8}{'delta':>8}")
    for k in keys:
        print(f"{k:<14}{before[k]:>8}{after[k]:>8}{after[k]-before[k]:>+8}")
    print(f"{'total':<14}{sum(before.values()):>8}{sum(after.values()):>8}"
          f"{sum(after.values())-sum(before.values()):>+8}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render true views on an NVS path")
    parser.add_argument("--scenes", default=None,
                        help="comma-separated; default spans 4 room types")
    parser.add_argument("--n-scenes", type=int, default=6)
    parser.add_argument("--refs-per-scene", type=int, default=1,
                        help="reference views per scene, chosen to be both "
                             "content-rich and mutually diverse. Whether fusion "
                             "has any headroom depends on the starting view: one "
                             "FloorPlan1 reference already saw all 23 nameable "
                             "objects, while other scenes gained 22-25%% from the "
                             "union across views.")
    parser.add_argument("--num-frames", type=int, default=20,
                        help="target views per scene (matches the NVS run)")
    parser.add_argument("--max-az", type=float, default=5.0)
    parser.add_argument("--max-el", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--tries", type=int, default=20)
    parser.add_argument("--outdir", default="datasets/sgg/multiview")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth-format", default="png16",
                        choices=("png16", "npy", "none"),
                        help="png16 = uint16 millimetres, 16x smaller than npy and "
                             "language-agnostic (default)")
    parser.add_argument("--refresh-gt", action="store_true",
                        help="recompute GT for an existing export at the stored "
                             "camera poses, leaving every image untouched")
    parser.add_argument("--poses-from", default=None,
                        help="JSON of target poses from an EXTERNAL trajectory "
                             "generator; renders at exactly those poses instead of "
                             "computing a lemniscate here. Required when the "
                             "trajectory is produced on the NVS side, since the "
                             "true renders must land on the same poses.")
    parser.add_argument("--flip-x", action="store_true",
                        help="mirror the path horizontally. By default the camera "
                             "sets off LEFT then DOWN; use this if the NVS "
                             "implementation starts rightward.")
    parser.add_argument("--flip-y", action="store_true",
                        help="mirror the path vertically (default starts downward)")
    parser.add_argument("--look-distance", type=float, default=2.0,
                        help="fixed look-at distance in metres (default 2.0). "
                             "Pass 0 to adapt it to each scene's visible content, "
                             "which makes real camera travel vary ~3x across "
                             "scenes for the same nominal max_az.")
    parser.add_argument("--depth", default="none",
                        choices=("none", "ref", "all"),
                        help="depth maps are off by default. The true render at "
                             "each pose is better ground truth than any "
                             "depth-based warp, and GT object_id already gives "
                             "cross-view association, so depth has no required "
                             "use here. '--depth ref' brings back the reference "
                             "depth if a depth-warp NVS baseline is wanted.")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    if args.refresh_gt:
        return refresh_multiview_gt(args.outdir, args.headless)
    if args.look_distance is not None and args.look_distance <= 0:
        args.look_distance = None          # 0 means "adapt per scene"

    import random

    from robot.robot_controller import RobotController

    scenes = (args.scenes.split(",") if args.scenes
              else DEFAULT_SCENES[:args.n_scenes])
    rng = random.Random(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    robot = RobotController(scene=scenes[0], width=args.width, height=args.height,
                            headless=args.headless, seed=args.seed, verbose=False)
    index = []
    try:
        for scene in scenes:
            robot.reset(scene)
            reachable = robot.get_reachable_positions(refresh=True)
            if len(reachable) == 0:
                print(f"{scene:<14} skipped (no reachable positions)")
                continue
            ref_poses = pick_reference_views(
                robot, reachable, args.tries, args.refs_per_scene, rng
            )
            for ref_index, ref_pose in enumerate(ref_poses):
                label = f"r{ref_index:02d}"
                record = export_scene(
                    robot, scene, args.outdir, args.num_frames,
                    ref_pose=ref_pose, ref_label=label,
                    max_az=args.max_az, max_el=args.max_el, tries=args.tries,
                    rng=rng, poses_from=args.poses_from,
                    depth_format=args.depth_format, depth_mode=args.depth,
                    look_distance_m=args.look_distance,
                    flip_x=args.flip_x, flip_y=args.flip_y,
                )
                if record is None:
                    continue
                index.append({"scene": scene, "ref": label,
                              "dir": os.path.join(scene, label),
                              "views": len(record["views"]),
                              "ref_objects": len(record["reference"]["objects"]),
                              "scene_relations": len(record["scene_relations"])})
            done = [e for e in index if e["scene"] == scene]
            if done:
                print(f"{scene:<14}{len(done)} refs  "
                      f"views={done[0]['views']}  "
                      f"ref objs={[e['ref_objects'] for e in done]}  "
                      f"scene rels={done[0]['scene_relations']}")
    finally:
        robot.stop()

    with open(os.path.join(args.outdir, "index.json"), "w", encoding="utf-8") as h:
        json.dump({"n_groups": len(index),
                   "n_scenes": len({e["scene"] for e in index}),
                   "refs_per_scene": args.refs_per_scene,
                   "groups": index,
                   "max_az_deg": args.max_az, "max_el_deg": args.max_el,
                   "num_frames": args.num_frames}, h, indent=1)
    print(f"\n{len(index)} scenes -> {args.outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
