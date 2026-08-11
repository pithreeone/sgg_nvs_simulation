"""
occlusion.py -- build controlled occlusions in iTHOR by moving scene objects.

Motivation: EGTR cannot report a relation whose endpoint it cannot see, and the
failure decomposition on `multiview/` put 59% of the loss in detection and 20%
in pair proposal.  To measure how much of that multi-view fusion recovers, we
need scenes where the occlusion is known and dialled to a chosen level rather
than whatever the stock scene happens to provide.

Measured against AI2-THOR 5.0.0.  What works and what does not:

    SetObjectPoses         arbitrary position, INCLUDING mid-air -- gravity is
                           not applied, so an object stays where it is put.
                           This is the only action that gives free placement.
    PlaceObjectAtPoint     refuses anything that is not a valid receptacle spawn
                           point.  A 0.3 m nudge along a counter failed outright,
                           and a request 0.5 m above the surface reported success
                           while leaving the object exactly where it was -- so its
                           return value cannot be trusted as evidence of movement.
    GetSpawnCoordinatesAboveReceptacle, InitialRandomSpawn, TeleportObject   work
    DisableObject / EnableObject                                            fail

Two properties of `SetObjectPoses` shape this module's API:

  * It takes the pose of EVERY moveable object at once; anything omitted is
    dropped from the scene.  So each call rebuilds the full list and only edits
    the entry being moved.
  * **It changes objectId.**  Ids encode position (`Mug|-01.76|+00.90|-00.62`
    becomes `Mug|-01.36|+01.50|-00.62`), so any id captured before the call is
    stale afterwards.  `name` (`Mug_e7fad100`) is stable and is what this module
    keys on throughout.  This matters beyond the module: `multiview/` uses
    `object_id` for cross-view association, so an export that poses objects must
    re-read ids after every pose change or switch to `name`.

Only moveable or pickupable objects can be posed -- 38 of 77 in FloorPlan1.
Cabinets, counters and fridges are static, so an occluder has to come from the
moveable set.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Occlusion above which an object stops being annotated at all.  A silhouette
#: this hidden is not a hard detection but an impossible one -- too little of it
#: is left for any method to name it, so annotating the relations it takes part
#: in only inflates the denominator with cases nothing can recover.
#:
#: It lives here rather than in `build_occlusion_dataset` because the EVALUATOR
#: has to agree with it: `eval_occlusion.BANDS` tops out just above this value,
#: and if the two drift the relations between the band edge and the build ceiling
#: are silently dropped from every metric -- exactly the samples raising the
#: ceiling was meant to add.  Both sides import this.
#:
#: Raised 0.80 -> 0.90.  On `occlusion_ds3` only 55 of 4604 object-views sat
#: above 0.75, so the old ceiling was binding on very few objects; the band it
#: opens is the one the benchmark most wants and has least of.
MAX_OCCLUSION = 0.90

#: Objects that make poor occluders regardless of size: too thin to block much,
#: or so scene-defining that moving them makes the render obviously wrong.
POOR_OCCLUDERS = {"Floor", "Painting", "Mirror", "LightSwitch", "Window", "Curtains"}

#: Fraction of the camera-to-target distance at which the occluder is placed.
#: Nearer the camera means a larger subtended angle and a sharper transition:
#: measured at t=0.45 on a 1.02 m target, a Pot went from 100% to 0% occlusion
#: over 8 cm of lateral offset.  Two thirds of the way to the target keeps the
#: response gradual enough for the search below to land on a requested level.
DEFAULT_DEPTH_FRACTION = 0.67


def moveable(event) -> List[Dict[str, Any]]:
    return [o for o in event.metadata["objects"] if o["moveable"] or o["pickupable"]]


def by_name(event, name: str) -> Optional[Dict[str, Any]]:
    return next((o for o in event.metadata["objects"] if o["name"] == name), None)


def _size(entry: Dict[str, Any]) -> Tuple[float, float, float]:
    s = entry["axisAlignedBoundingBox"]["size"]
    return s["x"], s["y"], s["z"]


def visible_pixels(event, name: str) -> int:
    """
    Rendered area of an object, by instance mask.

    Keyed on `name` rather than the objectId the caller happens to be holding,
    because a pose change silently invalidates the latter.  Requires the
    controller to have been started with `renderInstanceSegmentation=True`.
    """
    entry = by_name(event, name)
    if entry is None:
        return 0
    masks = event.instance_masks
    object_id = entry["objectId"]
    return int(masks[object_id].sum()) if object_id in masks else 0


def set_pose(controller, name: str, position: Sequence[float],
             rotation: Optional[Dict[str, float]] = None) -> bool:
    """
    Move one object, leaving every other moveable object where it is.

    Rebuilding the whole list on each call is not redundant: `SetObjectPoses`
    treats its argument as the complete set, and an object left out of it
    disappears from the scene.
    """
    poses = []
    for entry in moveable(controller.last_event):
        if entry["name"] == name:
            where = {"x": float(position[0]), "y": float(position[1]),
                     "z": float(position[2])}
            spin = rotation if rotation is not None else dict(entry["rotation"])
        else:
            where, spin = dict(entry["position"]), dict(entry["rotation"])
        poses.append({"objectName": entry["name"], "position": where, "rotation": spin})
    event = controller.step(action="SetObjectPoses", objectPoses=poses)
    return event.metadata["lastActionSuccess"]


def pose_snapshot(event) -> List[Dict[str, Any]]:
    """Every moveable object's world pose, in `SetObjectPoses` format."""
    return [{"objectName": o["name"], "position": dict(o["position"]),
             "rotation": dict(o["rotation"])} for o in moveable(event)]


def set_poses(controller, poses: Sequence[Dict[str, Any]]) -> bool:
    """
    Restore a whole `pose_snapshot`, pinning the moveable scene exactly.

    `SetObjectPoses` treats its argument as the COMPLETE set, so this is one
    call and anything omitted would vanish -- which is also why restoring the
    whole snapshot is the right granularity.  Setting one object leaves the
    others wherever THOR's own settle put them on this particular load, and
    that settle is not reproducible: measured on FloorPlan211, replaying only
    the occluder still moved the TARGET enough to change its visible box IoU
    between 0.810 and 0.846 across three runs of the same case.

    No gravity and no collision resolution are applied, so a restored snapshot
    is exact rather than approximately right.
    """
    have = {o["name"] for o in moveable(controller.last_event)}
    missing = [p["objectName"] for p in poses if p["objectName"] not in have]
    if missing:
        print(f"  ! snapshot names not in this scene: {missing[:3]}")
        return False
    event = controller.step(action="SetObjectPoses",
                            objectPoses=[dict(p) for p in poses])
    return event.metadata["lastActionSuccess"]


def ray_frame(camera: Dict[str, float], target: Dict[str, float]):
    """Unit vector camera->target, a horizontal perpendicular, and the distance."""
    c = np.array([camera["x"], camera["y"], camera["z"]])
    t = np.array([target["x"], target["y"], target["z"]])
    delta = t - c
    distance = float(np.linalg.norm(delta))
    forward = delta / distance
    side = np.cross(forward, [0.0, 1.0, 0.0])
    norm = np.linalg.norm(side)
    # Looking straight down leaves no horizontal perpendicular; any direction in
    # the ground plane will do in that degenerate case.
    side = side / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    return c, forward, side, distance


def pick_occluder(event, target_name: str, camera: Dict[str, float],
                  min_extent: float = 0.10) -> Optional[Dict[str, Any]]:
    """
    A moveable object big enough to hide the target, preferring the smallest
    that will do.

    Picking the largest is the obvious choice and the wrong one: a ShelvingUnit
    (0.94 x 1.48 m) placed on the ray took a Mug to 100% occlusion at every
    offset tested, which is a binary switch rather than a control.  The search
    below needs an occluder whose apparent size is comparable to the target's.
    """
    target = by_name(event, target_name)
    if target is None:
        return None
    want = max(_size(target)[0], _size(target)[2])
    candidates = []
    for entry in moveable(event):
        if entry["name"] == target_name or entry["objectType"] in POOR_OCCLUDERS:
            continue
        width = max(_size(entry)[0], _size(entry)[2])
        if width < max(min_extent, want * 0.8):
            continue
        candidates.append((width, entry))
    if not candidates:
        return None
    return min(candidates, key=lambda t: t[0])[1]


def occlude(controller, target_name: str, fraction: float,
            occluder_name: Optional[str] = None,
            depth_fraction: float = DEFAULT_DEPTH_FRACTION,
            tolerance: float = 0.05, max_iters: int = 12
            ) -> Dict[str, Any]:
    """
    Hide `fraction` of the target's rendered area behind another object.

    Occlusion is not a closed-form function of the offset -- it depends on both
    silhouettes and on whatever else is already in the way -- so this bisects on
    lateral offset and measures the mask each time, in the same spirit as
    `approach_until_resolvable()` in `robot_controller.py`.  An open-loop
    placement from a size table was tried there and was wrong by 8x.

    Returns a report including the achieved fraction, which will not always be
    the requested one: an occluder that is too large saturates and the search
    reports the closest it reached rather than pretending to have hit the mark.
    """
    event = controller.last_event
    baseline = visible_pixels(event, target_name)
    target = by_name(event, target_name)
    if target is None or baseline == 0:
        return {"ok": False, "reason": "target not visible before occlusion",
                "baseline_px": baseline}

    if occluder_name is None:
        chosen = pick_occluder(event, target_name,
                               event.metadata["cameraPosition"])
        if chosen is None:
            return {"ok": False, "reason": "no suitable moveable occluder",
                    "baseline_px": baseline}
        occluder_name = chosen["name"]

    origin = by_name(event, occluder_name)
    origin_position = dict(origin["position"])
    camera = event.metadata["cameraPosition"]
    c, forward, side, distance = ray_frame(camera, target["position"])
    centre = c + forward * distance * depth_fraction

    def achieved(offset: float) -> float:
        set_pose(controller, occluder_name, centre + side * offset)
        return 1.0 - visible_pixels(controller.last_event, target_name) / baseline

    # On the ray the occluder is at its most effective; far to the side it is out
    # of the way.  Bracket between those, then bisect.  The high end grows from
    # the occluder's own width rather than a fixed constant so a wide occluder
    # does not need extra iterations to escape the frame.
    lo, hi = 0.0, max(_size(origin)[0], _size(origin)[2]) + max(_size(target)[0], _size(target)[2])
    best = (abs(achieved(lo) - fraction), lo, achieved(lo))
    for _ in range(max_iters):
        mid = (lo + hi) / 2.0
        value = achieved(mid)
        error = abs(value - fraction)
        if error < best[0]:
            best = (error, mid, value)
        if error <= tolerance:
            break
        # More offset means less occlusion, so the comparison runs backwards.
        if value > fraction:
            lo = mid
        else:
            hi = mid

    error, offset, value = best
    achieved(offset)                      # leave the scene at the best placement
    return {
        "ok": error <= max(tolerance, 0.15),
        "target": target_name,
        "occluder": occluder_name,
        "requested": fraction,
        "achieved": round(value, 3),
        "offset_m": round(offset, 3),
        "baseline_px": baseline,
        "remaining_px": visible_pixels(controller.last_event, target_name),
        "occluder_origin": origin_position,
    }


def restore(controller, name: str, position: Dict[str, float]) -> bool:
    """Put an occluder back where `occlude()` found it."""
    return set_pose(controller, name, (position["x"], position["y"], position["z"]))
