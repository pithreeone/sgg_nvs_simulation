"""
build_occlusion_dataset.py -- render multi-view scenes annotated with occlusion.

For every (object, view) it records the object's full silhouette, the part of it
that is actually visible, and the ratio between them; for every (relation, view)
it records the occlusion of both endpoints.  Relations themselves are AMODAL --
a relation that is true of the scene is annotated whether or not this particular
camera can see it, because the point of the dataset is to ask whether extra views
recover what one view cannot.  Nothing is filtered by occlusion at build time:
the unoccluded relations are the control the occluded ones are measured against,
and stratifying recall by occlusion needs the whole range present.

    scene.json
      views[i].objects[j]   bbox_amodal, bbox_visible, occlusion, distance, ...
      views[i].relations[k] subject_occlusion, object_occlusion, predicate, ...
      views[i].image        view_00.png

Two decisions worth stating.

Relation geometry is computed from the AMODAL boxes.  `vg_gt` decides a predicate
partly from `cover` and `area_ratio`, and the visible box of a heavily occluded
object is a fragment, so using it would have the same relation classified
differently from different viewpoints -- an artefact of the occlusion rather than
a fact about the scene.

Both boxes are stored because a detector shown an 80%-occluded object will
predict roughly its visible extent, and scoring that against the amodal box fails
at any sensible IoU.  Scoring against the visible box alone has the opposite
problem: it shrinks as occlusion grows, so IoU gets EASIER the more hidden the
object is, which rewards exactly what the benchmark is meant to penalise.
Evaluation should take whichever of the two gives the better IoU.
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
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import gen.occlusion_pipeline as P
from gen.occlusion import MAX_OCCLUSION
from gen.export_samples import support_relations
from vg.vg150 import THOR_TO_VG150
from vg.vg_gt import VG_ANNOTATION_RATE, support_pairs, vg_calibrated_relations

#: Ten scenes per family x 3 seeds.  Ranked by `survey_scenes.py`
#: (nameable moveable objects x placement surfaces, spread over room size), then
#: filtered on a criterion the earlier five-scene lists did not need:
#:
#:   DISTINCT FOCUS REGIONS >= seeds.  A seed picks a different surface to build
#:   around, but `pick_focus` rotating over the raw surface list was not giving
#:   different regions -- a ShelvingUnit exposes several `Shelf` receptacles at
#:   the same (x, z), so consecutive seeds rebuilt the same corner.  Deduping by
#:   ground position fixed most of it; the scenes below are the ones that still
#:   yield 3 genuinely separate regions.  Measured, not assumed: of the first 30
#:   candidates, 11 gave fewer than 3 and FloorPlan301 gave exactly 1, which
#:   would have made all three of its seeds the same sample.
#:
#: This matters statistically, not just aesthetically: seeds sharing a focus are
#: correlated, so counting them as independent overstates n -- the same trap
#: TASKS.md records for poses within a target.
LIVING_ROOMS = ["FloorPlan201", "FloorPlan215", "FloorPlan230", "FloorPlan204",
                "FloorPlan203", "FloorPlan227", "FloorPlan218", "FloorPlan224",
                "FloorPlan223", "FloorPlan228"]

KITCHENS = ["FloorPlan7", "FloorPlan1", "FloorPlan10", "FloorPlan5",
            "FloorPlan16", "FloorPlan17", "FloorPlan18", "FloorPlan21",
            "FloorPlan8", "FloorPlan6"]

#: Bedrooms are small, so four of these (326, 330, 323, 321) offer only 6-7
#: camera positions in the 1.2-2.0 m band for 10 views and will repeat positions;
#: the build prints a note when that happens.  Preferred anyway over bedrooms
#: with more room but only 2 focus regions -- a repeated position still varies
#: height, yaw and pitch, whereas a repeated focus duplicates the whole sample.
BEDROOMS = ["FloorPlan311", "FloorPlan326", "FloorPlan330", "FloorPlan307",
            "FloorPlan323", "FloorPlan321", "FloorPlan328", "FloorPlan302",
            "FloorPlan303", "FloorPlan305"]

#: EXCLUDED from `--rooms all`, kept so the decision is reversible.
#:
#: Bathrooms are thin by construction -- 9-11 nameable moveable objects against a
#: kitchen's 20-27, and 3.6 relations per view against a kitchen's 7.0.  They were
#: originally included for `sink`/`toilet`/`towel` coverage, but on `occlusion_ds3`
#: their 150 views bought 540 relations of which 297 are `on` and only 14 `under`,
#: and the sampler is degenerate there: FloorPlan420 has 33 reachable positions in
#: total and just 14 inside the camera band, so all 10 views come from nearly the
#: same spot.  A room whose viewpoints cannot vary is not a multi-view occlusion
#: sample.  Dropping them loses `toilet` and `towel` from the vocabulary entirely.
#:
#: Build them explicitly with `--rooms bathroom` or `--scenes ...`.
BATHROOMS = ["FloorPlan429", "FloorPlan430", "FloorPlan420", "FloorPlan419",
             "FloorPlan427"]

ROOM_FAMILIES = {"kitchen": KITCHENS, "living": LIVING_ROOMS,
                 "bedroom": BEDROOMS, "bathroom": BATHROOMS}

#: What `--rooms all` expands to.  Not simply every key of `ROOM_FAMILIES`.
DEFAULT_FAMILIES = ["kitchen", "living", "bedroom"]

#: Camera height band, in metres, sampled UNIFORMLY per view.  Below the agent's
#: fixed standing eye height of 1.576 m, which is the only height `Teleport`
#: offers and is why the sweep uses a third-party camera instead.
#:
#: This was three discrete levels (0.85, 1.10, 1.35).  Two reasons it is a range
#: now: a continuous band removes the quantisation -- with 10 views drawn from
#: three levels the height was replicated 3-4 times per scene and could not be
#: used as a covariate -- and 1.0-1.5 m is the band a mobile robot's sensor
#: actually sits in, which is the height the downstream tasks care about.
#:
#: Trade-off to keep in view: raising the floor from 0.85 m REDUCES inter-object
#: occlusion, because a lower camera sees the room closer to the plane the
#: furniture occupies and objects overlap in projection rather than being viewed
#: from above.  The occlusion band distribution should be re-checked after any
#: change here -- it is the quantity the dataset exists to measure.
HEIGHT_RANGE = (1.0, 1.5)

#: Vertical field of view.  The whole intrinsic matrix follows from this and the
#: resolution -- fx = fy = (h/2)/tan(vfov/2), the principal point is the centre,
#: and there is no skew.  THOR exposes no way to set fx != fy or to move the
#: principal point.
DEFAULT_FOV = 60.0




def bbox_of(mask: np.ndarray) -> Optional[List[int]]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2) + 1, int(y2) + 1]


def intrinsics(width: int, height: int, fov_deg: float) -> Dict[str, float]:
    f = (height / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return {"fx": f, "fy": f, "cx": width / 2.0, "cy": height / 2.0,
            "width": width, "height": height, "fov_vertical_deg": fov_deg}


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


def camera_poses(controller, focus: Dict[str, float], count: int,
                 height_range: Tuple[float, float], seed: int,
                 radius: Tuple[float, float] = (1.2, 2.0)) -> List[Dict[str, Any]]:
    # `radius` is the distance band the camera stands in, measured horizontally
    # to the FOCUS surface -- not to each annotated object, so the realised
    # camera-to-object distances run past the cap.
    #
    # Pulling it in makes targets bigger, which matters because the size floor
    # rejects anything under 0.06 of the frame -- a 0.15 m object only clears
    # that inside about 1.5 m at 60 degrees vertical.  It also raises occlusion:
    # a nearer camera gives the flanking objects more angle, which is the effect
    # recorded for Task 1 (`fill` dropped 0.43 -> 0.17 on approach).  That
    # partly offsets the occlusion lost by raising `HEIGHT_RANGE` off the floor.
    #
    # The cost is viewpoint diversity: fewer reachable positions satisfy a
    # tighter band, and if none do the sampler falls back to `reachable[:count]`
    # and ignores the band entirely.  Check the fallback is not firing before
    # tightening further.
    """
    Positions around the arrangement, each aimed at it.

    Aiming rather than sampling yaw freely: a random heading from a random
    position usually faces a wall, and a view with nothing in it costs the same
    to render as a useful one.  Jitter keeps the framing from being identical
    every time.
    """
    rng = random.Random(seed)
    reachable = controller.step(
        action="GetReachablePositions").metadata["actionReturn"] or []
    near = [p for p in reachable
            if radius[0] <= math.dist((p["x"], p["z"]), (focus["x"], focus["z"]))
            <= radius[1]]
    rng.shuffle(near)
    if not near:
        # Silent until now.  When it fires the band is not applied at all, so the
        # scene's cameras stand wherever `GetReachablePositions` happened to list
        # first -- which is not the configuration anything downstream assumes.
        print(f"    WARNING: no reachable position in {radius[0]:g}-{radius[1]:g} m "
              f"of the focus; falling back to unfiltered positions")
        near = reachable[:count] or [dict(focus)]
    elif len(near) < count:
        print(f"    note: only {len(near)} distinct positions in "
              f"{radius[0]:g}-{radius[1]:g} m for {count} views; positions repeat")

    poses = []
    for index in range(count):
        spot = near[index % len(near)]
        height = rng.uniform(*height_range)
        dx, dz = focus["x"] - spot["x"], focus["z"] - spot["z"]
        flat = math.hypot(dx, dz)
        yaw = math.degrees(math.atan2(dx, dz)) + rng.uniform(-12.0, 12.0)
        # Positive pitch looks down in Unity.  Aimed at the focus surface, so the
        # tilt follows from the height the view happened to draw.
        pitch = math.degrees(math.atan2(height - focus["y"], max(flat, 1e-3)))
        poses.append({"position": {"x": spot["x"], "y": height, "z": spot["z"]},
                      "yaw": yaw % 360.0,
                      "pitch": max(-40.0, min(50.0, pitch + rng.uniform(-6.0, 6.0)))})
    return poses


def look(controller, pose: Dict[str, Any], fov: float, first: bool = False,
         reachable: Optional[Sequence[Dict[str, float]]] = None):
    """
    Aim the third-party camera, and get the agent out of its way first.

    Parking is folded in here rather than done once per scene because "behind the
    camera" moves with the camera.  It costs one extra step per camera update,
    which is the cheap half of the loop.
    """
    P.park_agent(controller, pose["position"], pose["yaw"], reachable)
    action = "AddThirdPartyCamera" if first else "UpdateThirdPartyCamera"
    kwargs = {} if first else {"thirdPartyCameraId": 0}
    controller.step(action=action, position=dict(pose["position"]),
                    rotation={"x": pose["pitch"], "y": pose["yaw"], "z": 0.0},
                    fieldOfView=fov, **kwargs)


def masks_and_boxes(event) -> Dict[str, Tuple[int, List[int]]]:
    names = {o["objectId"]: o["name"] for o in event.metadata["objects"]}
    out = {}
    for object_id, mask in event.third_party_instance_masks[0].items():
        name = names.get(object_id)
        if name is None:
            continue
        box = bbox_of(mask)
        if box is not None:
            out[name] = (int(mask.sum()), box)
    return out


def build_scene(controller, scene: str, seed: int, poses: Sequence[Dict[str, Any]],
                plan: Dict[str, Any], fov: float, outdir: str,
                width: int, height: int,
                max_occlusion: float = MAX_OCCLUSION) -> Dict[str, Any]:
    """
    One sweep with everything present, then one per target with it alone.

    The disabled state survives a camera move, so the whole viewpoint list is
    swept once per target rather than toggling per viewpoint.
    """
    from PIL import Image

    os.makedirs(outdir, exist_ok=True)
    targets = plan["targets"]
    reachable = controller.step(
        action="GetReachablePositions").metadata["actionReturn"] or []

    observed: List[Dict[str, Tuple[int, List[int]]]] = []
    for index, pose in enumerate(poses):
        look(controller, pose, fov, first=(index == 0), reachable=reachable)
        Image.fromarray(controller.last_event.third_party_camera_frames[0]).save(
            os.path.join(outdir, f"view_{index:02d}.png"))
        observed.append(masks_and_boxes(controller.last_event))

    # Metadata is read here, with everything present, because `distance` and
    # `position` are properties of the scene and must not be read while half of
    # it is disabled.
    entries = {o["name"]: o for o in controller.last_event.metadata["objects"]}

    # Everything nameable is disabled, not just the moveable half.  An amodal box
    # is only amodal if NOTHING else is in the frame, so leaving the fixed
    # fittings standing truncated the reference box of anything behind a counter
    # and silently understated its occlusion.
    ids = {o["name"]: o["objectId"] for o in controller.last_event.metadata["objects"]
           if o["name"] in set(targets)}
    for object_id in ids.values():
        controller.step(action="DisableObject", objectId=object_id)

    amodal: Dict[str, List[Optional[Tuple[int, List[int]]]]] = {}
    for name in targets:
        if name not in ids:
            continue
        controller.step(action="EnableObject", objectId=ids[name])
        per_view = []
        for index, pose in enumerate(poses):
            look(controller, pose, fov, reachable=reachable)
            per_view.append(masks_and_boxes(controller.last_event).get(name))
        controller.step(action="DisableObject", objectId=ids[name])
        amodal[name] = per_view

    for object_id in ids.values():
        controller.step(action="EnableObject", objectId=object_id)

    # Below this the ratio is mask aliasing rather than occlusion: an unfiltered
    # first run called a 359 px plate "88% occluded", which says more about its
    # distance than about anything in front of it.  The fraction is the one VG's
    # annotators worked to, so it scales with resolution on its own.
    min_px = int(P.MIN_EXTENT_FRACTION * width * height)

    views = []
    for index, pose in enumerate(poses):
        camera = dict(pose["position"])
        objects = []
        for name in targets:
            entry = entries.get(name)
            reference = amodal.get(name, [None] * len(poses))[index]
            if entry is None or reference is None:
                continue
            ref_px, ref_box = reference
            if ref_px < min_px:
                continue
            seen_px, seen_box = observed[index].get(name, (0, None))
            if 1.0 - seen_px / ref_px > max_occlusion:
                continue
            objects.append({
                "name": name,
                "thor_type": entry["objectType"],
                "vg150_class": THOR_TO_VG150.get(entry["objectType"]),
                "object_id": entry["objectId"],
                "position": dict(entry["position"]),
                "distance": math.dist(
                    (camera["x"], camera["y"], camera["z"]),
                    (entry["position"]["x"], entry["position"]["y"],
                     entry["position"]["z"])),
                "parent_receptacles": entry.get("parentReceptacles") or [],
                "pickupable": bool(entry["pickupable"]),
                # Stored so the relations can be re-derived from the JSON alone.
                # Support depends on `moveable` too: a Television is moveable but
                # not pickupable, and leaving it out is what mislabelled 78
                # `on` relations as `above`.
                "moveable": bool(entry.get("moveable")),
                "bbox_amodal": ref_box,
                "bbox_visible": seen_box,
                "reference_px": ref_px,
                "visible_px": seen_px,
                "occlusion": round(max(0.0, 1.0 - seen_px / ref_px), 4),
            })
        views.append({"index": index, "image": f"view_{index:02d}.png",
                      "camera": {**pose}, "objects": objects})

    return {"scene": scene, "seed": seed, "min_reference_px": min_px,
            "max_occlusion": max_occlusion,
            "intrinsics": intrinsics(width, height, fov),
            "convention": "Unity left-handed, Y up, yaw clockwise from +Z; "
                          "fieldOfView is VERTICAL",
            "added": plan["placed"], "dropped": plan["dropped"],
            "views": views}


def annotate_relations(record: Dict[str, Any], entries: Dict[str, Any],
                       rate: Optional[float] = VG_ANNOTATION_RATE) -> None:
    """
    Attach VG-convention relations to each view, tagged with endpoint occlusion.

    `vg_gt` keys on `object_id` and reads `bbox_xyxy`, so each view is handed a
    shim whose id is the stable `name` and whose box is the AMODAL one.

    `rate` is the share of all object pairs a human would have bothered to
    relate, and it is what keeps the annotation to the ~10% of pairs VG's own
    density implies.  `None` keeps every geometric proposal: useful for asking
    why a particular triplet is absent, NOT for building a dataset -- the
    recall denominators stop being comparable to anything published here.
    """
    width = record["intrinsics"]["width"]
    height = record["intrinsics"]["height"]

    for view in record["views"]:
        shim = [{"object_id": o["name"], "vg150_class": o["vg150_class"],
                 "bbox_xyxy": o["bbox_amodal"], "position": o["position"],
                 "distance": o["distance"], "loose": False}
                for o in view["objects"] if o["vg150_class"]]
        present = {o["name"] for o in view["objects"]}
        occlusion = {o["name"]: o["occlusion"] for o in view["objects"]}

        # `on`/`in` come from `parentReceptacles`, which THOR asserts directly and
        # which is the one relation family whose 3D definition already agreed
        # with VG's annotation convention (44% recall against 0% for the rest).
        by_id = {entries[n]["objectId"]: n for n in present if n in entries}
        support = []
        for rel in support_relations([entries[n] for n in present if n in entries],
                                     allowed_ids=set(by_id)):
            subject = by_id.get(rel["subject_id"])
            obj = by_id.get(rel["object_id"])
            if subject and obj:
                support.append({**rel, "subject_id": subject, "object_id": obj})

        geometric = vg_calibrated_relations(
            shim, width, height, support_pairs(support), rate=rate)

        relations = []
        for rel in support + geometric:
            subject, obj = rel["subject_id"], rel["object_id"]
            if subject not in occlusion or obj not in occlusion:
                continue
            relations.append({
                "subject": rel["subject"], "predicate": rel["predicate"],
                "object": rel["object"],
                "subject_name": subject, "object_name": obj,
                "subject_occlusion": occlusion[subject],
                "object_occlusion": occlusion[obj],
                "annotation": rel["annotation"],
            })
        view["relations"] = relations


def build(scene: str, seed: int, n_views: int, width: int, height: int,
          fov: float, root: str, surfaces: int, spacing: float,
          max_occlusion: float = MAX_OCCLUSION,
          radius: Tuple[float, float] = (1.2, 2.0),
          height_range: Tuple[float, float] = HEIGHT_RANGE,
          small_extra: int = 16, large_extra: int = 6) -> Dict[str, Any]:
    from ai2thor.controller import Controller

    controller = Controller(scene=scene, width=width, height=height,
                            renderInstanceSegmentation=True,
                            visibilityDistance=15.0)
    try:
        if not P.surfaces_near(controller.last_event,
                               controller.last_event.metadata["agent"]["position"]):
            return {"error": f"{scene} has no placement surface"}
        focus = pick_focus(controller, seed)
        plan = P.arrange(controller, seed=seed, focus=focus,
                         n_surfaces=surfaces, spacing=spacing,
                         small_extra=small_extra, large_extra=large_extra)
        poses = camera_poses(controller, focus, n_views, height_range, seed,
                             radius=radius)
        outdir = os.path.join(root, f"{scene}_s{seed}")
        record = build_scene(controller, scene, seed, poses, plan, fov, outdir,
                             width, height, max_occlusion)
        entries = {o["name"]: o for o in controller.last_event.metadata["objects"]}
        annotate_relations(record, entries)
        with open(os.path.join(outdir, "scene.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1)
        return record
    finally:
        controller.stop()


HARD = 0.5
CLEAR = 0.25


def recoverable(record: Dict[str, Any]) -> Dict[str, int]:
    """
    Count the relations this dataset actually exists to measure.

    A relation is HARD in a view when an endpoint is at least half hidden, and
    RECOVERABLE when some other view sees both endpoints nearly clear.  Hard and
    recoverable is the set extra views can rescue and one view cannot; hard
    everywhere is unrecoverable by any method and only inflates the denominator.
    """
    best: Dict[tuple, float] = {}
    for view in record["views"]:
        for rel in view["relations"]:
            key = (rel["subject_name"], rel["predicate"], rel["object_name"])
            worst = max(rel["subject_occlusion"], rel["object_occlusion"])
            best[key] = min(best.get(key, 1.0), worst)

    counts = {"hard": 0, "recoverable": 0, "unrecoverable": 0, "clear": 0}
    for view in record["views"]:
        for rel in view["relations"]:
            key = (rel["subject_name"], rel["predicate"], rel["object_name"])
            worst = max(rel["subject_occlusion"], rel["object_occlusion"])
            if worst < HARD:
                counts["clear"] += 1
            else:
                counts["hard"] += 1
                if best[key] < CLEAR:
                    counts["recoverable"] += 1
                else:
                    counts["unrecoverable"] += 1
    return counts


def summarise(record: Dict[str, Any]) -> str:
    views = record["views"]
    objects = sum(len(v["objects"]) for v in views)
    relations = sum(len(v["relations"]) for v in views)
    c = recoverable(record)
    return (f"{record['scene']} s{record['seed']}: {len(views)} views, "
            f"{objects} object-views, {relations} relations "
            f"[clear {c['clear']}, hard {c['hard']} of which "
            f"recoverable {c['recoverable']}]")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--rooms", nargs="*", default=["living"],
                        choices=sorted(ROOM_FAMILIES) + ["all"],
                        help="room families to build; ignored if --scenes is given")
    # Two seeds, not three.  A seed rotates `pick_focus` onto a different
    # surface, so seeds buy within-scene coverage rather than repetition --
    # but measured over `occlusion_ds3` the three are interchangeable in
    # yield (64 / 63 / 60 relations per scene), so dropping one costs data
    # proportionally and nothing structurally.  Build time falls by a third.
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--views", type=int, default=10)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--fov", type=float, default=DEFAULT_FOV)
    parser.add_argument("--surfaces", type=int, default=3)
    parser.add_argument("--spacing", type=float, default=0.18)
    parser.add_argument("--heights", type=float, nargs=2,
                        default=list(HEIGHT_RANGE), metavar=("MIN", "MAX"),
                        help="camera height band in metres, sampled uniformly")
    parser.add_argument("--radius", type=float, nargs=2, default=[1.2, 2.0],
                        metavar=("MIN", "MAX"),
                        help="camera distance band from the focus, in metres")
    parser.add_argument("--small-extra", type=int, default=16,
                        help="counter-top duplicates to add across all types")
    parser.add_argument("--large-extra", type=int, default=6,
                        help="floor-standing duplicates to add")
    parser.add_argument("--max-occlusion", type=float, default=MAX_OCCLUSION,
                        help="drop targets hidden beyond this; nothing recovers "
                             "an object with almost no pixels left")
    parser.add_argument("--out", default="datasets/occlusion_ds")
    args = parser.parse_args(argv)

    if not args.scenes:
        families = list(DEFAULT_FAMILIES) if "all" in args.rooms else args.rooms
        args.scenes = [s for f in families for s in ROOM_FAMILIES[f]]
        print(f"rooms: {', '.join(families)}")

    print(f"{len(args.scenes)} scenes x {len(args.seeds)} seeds x {args.views} views"
          f"   {args.width}x{args.height}  vfov {args.fov:g}")
    print(f"camera height {args.heights[0]:g}-{args.heights[1]:g} m, sampled uniformly\n"
          f"(agent standing height 1.576 is not used)\n")

    ok = 0
    for scene in args.scenes:
        for seed in args.seeds:
            # Keyword from `max_occlusion` on: these are all optional and a new
            # one inserted mid-signature silently shifted `small_extra` into
            # `height_range` once already.
            record = build(scene, seed, args.views, args.width, args.height,
                           args.fov, args.out, args.surfaces, args.spacing,
                           max_occlusion=args.max_occlusion,
                           radius=tuple(args.radius),
                           height_range=tuple(args.heights),
                           small_extra=args.small_extra,
                           large_extra=args.large_extra)
            if "error" in record:
                print(f"  {record['error']}")
                continue
            print(f"  {summarise(record)}")
            ok += 1
    print(f"\n{ok} scene records under {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
