"""
export_samples.py -- export AI2-THOR frames plus ground-truth scene graphs.

Output layout:
    samples/rgb/<name>.png      raw RGB frames -- THIS is what goes into EGTR
    samples/gt/<name>_gt.png    preview with GT boxes and relation arrows drawn
    samples/meta/<name>.json    GT objects, relations, camera pose, intrinsics
    samples/index.json          every sample in one file

Viewpoints are chosen to maximise the number of VG150-nameable objects visible at
a usable pixel size, so the frames are a fair test of what EGTR could possibly
predict rather than random walls.

Ground-truth relations
----------------------
  on     -- from `parentReceptacles` (THOR's support relation)
  under  -- derived geometrically: the host's XZ footprint covers the target and
            the host's bottom face sits 0.15-1.0 m above the target's top face

Only objects whose THOR type maps to a VG150 class are included, since anything
else is unnameable by the model and would count as a false positive unfairly.

Usage
-----
    python export_samples.py                        # 12 samples, 4 room types
    python export_samples.py --n 24 --width 1024 --height 768
    python export_samples.py --scenes FloorPlan1,FloorPlan201 --headless
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
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vg.vg150 import thor_to_vg150

DEFAULT_SCENES = [
    "FloorPlan1", "FloorPlan5", "FloorPlan10",        # kitchens
    "FloorPlan201", "FloorPlan205", "FloorPlan210",   # living rooms
    "FloorPlan301", "FloorPlan305", "FloorPlan310",   # bedrooms
    "FloorPlan401", "FloorPlan405", "FloorPlan410",   # bathrooms
]

MIN_AREA = 0.0015          # ignore objects smaller than this fraction of pixels
MIN_OBJECTS = 3            # a usable viewpoint shows at least this many

#: THOR types whose VG150 mapping is only approximate.  A model that "gets these
#: wrong" may in fact be right and the mapping wrong, so anything involving them
#: is flagged `loose` and can be excluded from strict scoring.
#: THOR receptacles with a genuine interior, as opposed to a support surface.
#: `parentReceptacles` does not distinguish "on" from "in", so without this a Cup
#: inside a Cabinet was labelled `on cabinet`.  Measured over 40 scenes, 9% of
#: support relations are really containment, and a purely geometric test is too
#: aggressive -- a Vase on a shelf board sits inside the shelf UNIT's bounding
#: box, which would wrongly label it `in shelf`.
CONTAINER_TYPES = {
    "SinkBasin", "Sink", "GarbageCan", "Bowl", "Pot", "Pan", "Box", "Toilet",
    "Bathtub", "BathtubBasin", "Mug", "Cup", "Cart", "Safe", "LaundryHamper",
}

#: Thresholds for the geometrically derived predicates.  Deliberately tight: at
#: overlap 0.15 and 0.9 m, a single 21-object view produced 209 relations, and
#: since EGTR emits only 100 triplets that alone caps R@100 at 0.48 however good
#: the model is.  VG annotators also label a sparse subset rather than every
#: overlapping pair, so a dense ground truth is a different distribution from the
#: one the model was trained on.
NEAR_THRESHOLD_M = 0.5
BEHIND_MIN_OVERLAP = 0.4
UNDER_MIN_GAP_M = 0.15
UNDER_MAX_GAP_M = 1.0

#: `between` is extremely sensitive to how loosely it is defined.  Any three
#: roughly collinear objects qualify under a naive rule, which yielded 672
#: instances over 40 scenes -- against only 26 `between` examples in the whole
#: VG150 test split, so the generated distribution would be nothing like the one
#: the model was trained on.  These constraints bring it to ~11, the right order
#: of magnitude, and the surviving examples read correctly
#: ("apple between potato and lettuce", "cup between egg and bowl").
#: `between` is judged in IMAGE space, not world space.  A VG annotator looks at a
#: photograph and cannot see depth, so "A between B and C" means A appears between
#: them in the picture -- and specifically that B and C are A's immediate left and
#: right neighbours.  Measured over 200 reference views:
#:     3D collinearity (strict)          1 relation    -- unusable
#:     any object horizontally between   7530          -- "shelf between shelf/vase"
#:     immediate left/right neighbours     94          -- this one
#: Because it depends on the camera, `between` belongs to a view, not to the scene.
BETWEEN_MIN_VOVERLAP = 0.35     # all three must sit in the same "row"
BETWEEN_MAX_SIZE_RATIO = 4.0    # flankers not much larger than the target

#: THOR types treated as structure rather than as objects a relation is "about".
#: A `between` whose subject is a CounterTop or a Shelf is not a useful example.
STRUCTURAL_TYPES = {
    "Cabinet", "Drawer", "CounterTop", "Shelf", "ShelvingUnit", "Sink", "SinkBasin",
    "Fridge", "Microwave", "Toilet", "Bathtub", "BathtubBasin", "Window", "Door",
    "DiningTable", "CoffeeTable", "SideTable", "Desk", "Bed", "Sofa", "Chair",
    "ArmChair", "Stool", "TVStand", "Television", "Curtains", "Blinds", "Faucet",
    "StoveBurner", "Floor", "Wall", "Ceiling", "Painting", "Mirror", "LightSwitch",
    "HousePlant",
}

BETWEEN_NEAREST_K = 3          # 3D variant: flankers among the target's nearest few
BETWEEN_MAX_FLANKER_SIZE = 3.0  # x target's own size; excludes furniture flankers
BETWEEN_MAX_OFFSET = 0.10      # fraction of the flanker span
BETWEEN_MAX_DY_M = 0.15
BETWEEN_MIN_SPAN_M = 0.20
BETWEEN_MAX_SPAN_M = 1.20
BETWEEN_MAX_COSINE = -0.5      # flankers on genuinely opposite sides (>120 deg)

LOOSE_MAPPINGS = {
    "Sofa", "Stool", "Footstool",          # -> chair (VG150 has no sofa/couch)
    "Television",                          # -> screen
    "Pan",                                 # -> pot
    "GarbageCan",                          # -> box
    "ShelvingUnit",                        # -> shelf
    "LightSwitch",                         # -> light
    "AlarmClock", "Watch",                 # -> clock
    "Newspaper", "PaperTowelRoll", "ToiletPaper",   # -> paper
    "Lettuce", "Tomato", "Potato",         # -> vegetable
    "Bread", "Egg",                        # -> food
    "Apple",                               # -> fruit
    "TeddyBear",                           # -> animal
    "SoapBottle", "SprayBottle",           # -> bottle
    "TissueBox",                           # -> box
    "HousePlant",                          # -> plant (THOR fuses plant AND pot)
    "DeskLamp", "FloorLamp",               # -> lamp
    "TVStand",                             # -> stand
}


def _bbox3d(o: Dict[str, Any]) -> Optional[Dict[str, float]]:
    b = o.get("axisAlignedBoundingBox") or {}
    c, s = b.get("center"), b.get("size")
    if not c or not s:
        return None
    return {"cx": c["x"], "cy": c["y"], "cz": c["z"],
            "sx": s["x"], "sy": s["y"], "sz": s["z"]}


def visible_objects(event, width: int, height: int) -> List[Dict[str, Any]]:
    """VG150-nameable objects present in this frame at a usable pixel size."""
    detections = getattr(event, "instance_detections2D", None) or {}
    masks = getattr(event, "instance_masks", None) or {}
    out = []
    for obj in event.metadata["objects"]:
        vg = thor_to_vg150(obj["objectType"])
        if not vg:
            continue
        box = detections.get(obj["objectId"])
        mask = masks.get(obj["objectId"])
        if box is None or mask is None:
            continue
        pixels = int(np.count_nonzero(mask))
        area = pixels / float(width * height)
        if area < MIN_AREA:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        out.append({
            "object_id": obj["objectId"],
            "thor_type": obj["objectType"],
            "vg150_class": vg,
            "loose": obj["objectType"] in LOOSE_MAPPINGS,
            "bbox_xyxy": [x1, y1, x2, y2],
            "area_fraction": round(area, 5),
            "mask_pixels": pixels,
            "position": {k: round(v, 4) for k, v in obj["position"].items()},
            "distance": round(float(obj.get("distance", -1)), 3),
        })
    return sorted(out, key=lambda o: -o["area_fraction"])


def _is_container(host: Dict[str, Any]) -> bool:
    return bool(host.get("openable")) or host["objectType"] in CONTAINER_TYPES


def _inside_bbox(target: Dict[str, Any], host: Dict[str, Any]) -> bool:
    tb, hb = _bbox3d(target), _bbox3d(host)
    if tb is None or hb is None:
        return False
    return (abs(tb["cx"] - hb["cx"]) <= hb["sx"] / 2
            and abs(tb["cz"] - hb["cz"]) <= hb["sz"] / 2
            and (hb["cy"] - hb["sy"] / 2) <= tb["cy"] <= (hb["cy"] + hb["sy"] / 2))


def support_relations(
    objects: Sequence[Dict[str, Any]],
    allowed_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    `on` and `in` from `parentReceptacles`, split by whether the host is a
    container.  World-relative, so valid for the whole scene, not per view.

    The subject need only have a parent receptacle; it does NOT need to be
    `pickupable`.  Requiring that discarded 206 of the 384 support facts in the
    occlusion dataset, because THOR marks a HousePlant, Television, Microwave or
    GarbageCan `moveable` but not `pickupable` -- they rest on things constantly,
    and `parentReceptacles` says so.  Those pairs then fell through to the
    geometric derivation and came out as `above`, which is how 78 of 172 `above`
    labels came to describe a television standing on its stand.  EGTR called them
    `on` and was right; the annotation was wrong.  Support is a fact about
    contact, not about whether a robot could lift the thing.
    """
    by_id = {o["objectId"]: o for o in objects}
    out, seen = [], set()
    for obj in objects:
        if not (obj.get("pickupable") or obj.get("moveable")):
            continue
        subject_vg = thor_to_vg150(obj["objectType"])
        if not subject_vg:
            continue
        if allowed_ids is not None and obj["objectId"] not in allowed_ids:
            continue
        for parent_id in (obj.get("parentReceptacles") or []):
            host = by_id.get(parent_id)
            if host is None:
                continue
            if allowed_ids is not None and parent_id not in allowed_ids:
                continue
            host_vg = thor_to_vg150(host["objectType"])
            if not host_vg or host_vg == subject_vg:
                continue
            predicate = ("in" if (_is_container(host) and _inside_bbox(obj, host))
                         else "on")
            key = (obj["objectId"], predicate, parent_id)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "subject": subject_vg, "predicate": predicate, "object": host_vg,
                "subject_id": obj["objectId"], "object_id": parent_id,
                "annotation": "parentReceptacles",
                "loose": bool(LOOSE_MAPPINGS & {obj["objectType"],
                                                host["objectType"]}),
            })
    return out


def occlusion_relations(
    objects: Sequence[Dict[str, Any]],
    present: Sequence[Dict[str, Any]],
    min_overlap: float = BEHIND_MIN_OVERLAP,
    max_overlap: float = 0.9,
    exclude_pairs: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    `behind` / `in front of` for one view.

    CAMERA-RELATIVE, so these belong to a view and never to the scene: A is
    `behind` B when their 2D boxes overlap appreciably and A is the farther of the
    two.  That matches what an annotator reads off a single image, and is the only
    family here that exercises horizontal occlusion.
    """
    by_id = {o["objectId"]: o for o in objects}
    out = []
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            if a["vg150_class"] == b["vg150_class"]:
                continue
            # A pair already joined by on/in is not an occlusion example: a vase
            # sitting on a shelf scores overlap 1.0 simply because it lies inside
            # the shelf unit's box, which is containment rather than occlusion.
            if exclude_pairs and frozenset((a["object_id"], b["object_id"])) in exclude_pairs:
                continue
            ax1, ay1, ax2, ay2 = a["bbox_xyxy"]
            bx1, by1, bx2, by2 = b["bbox_xyxy"]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter <= 0:
                continue
            smaller = min(max((ax2 - ax1) * (ay2 - ay1), 1),
                          max((bx2 - bx1) * (by2 - by1), 1))
            ratio = inter / smaller
            # Near-total overlap means one box encloses the other, which is a
            # containment configuration, not one object occluding another.
            if not (min_overlap <= ratio <= max_overlap):
                continue
            da = by_id[a["object_id"]].get("distance", 0.0)
            db = by_id[b["object_id"]].get("distance", 0.0)
            far, near = (a, b) if da > db else (b, a)
            loose = bool(LOOSE_MAPPINGS & {by_id[far["object_id"]]["objectType"],
                                           by_id[near["object_id"]]["objectType"]})
            out.append({
                "subject": far["vg150_class"], "predicate": "behind",
                "object": near["vg150_class"],
                "subject_id": far["object_id"], "object_id": near["object_id"],
                "annotation": "geometric", "loose": loose,
                "bbox_overlap": round(ratio, 3),
            })
            out.append({
                "subject": near["vg150_class"], "predicate": "in front of",
                "object": far["vg150_class"],
                "subject_id": near["object_id"], "object_id": far["object_id"],
                "annotation": "geometric", "loose": loose,
                "bbox_overlap": round(ratio, 3),
            })
    return out


def proximity_relations(
    objects: Sequence[Dict[str, Any]],
    present: Sequence[Dict[str, Any]],
    threshold: float = NEAR_THRESHOLD_M,
    exclude_pairs: Optional[set] = None,
    require_pickupable: bool = True,
) -> List[Dict[str, Any]]:
    """
    `near` by 3D centre distance.  Symmetric, so emitted once per pair.

    At least one endpoint must be pickupable.  Without that constraint the most
    frequent relation in the whole set was `cabinet near drawer` -- the door and
    the drawer of the same cabinet unit, which is a part-whole configuration
    rather than proximity, and not something an annotator would label.  Requiring
    one small object removes 36% of pairs and specifically the structural ones
    (`cabinet near drawer`, `counter near drawer`, `table near chair`), while
    keeping `drawer near pot`, `shelf near vase`, `drawer near lamp`.
    """
    by_id = {o["objectId"]: o for o in objects}
    out = []
    for i, a in enumerate(present):
        pa = by_id[a["object_id"]]["position"]
        for b in present[i + 1:]:
            if a["vg150_class"] == b["vg150_class"]:
                continue
            if exclude_pairs and frozenset((a["object_id"], b["object_id"])) in exclude_pairs:
                continue
            if require_pickupable and not (by_id[a["object_id"]].get("pickupable")
                                           or by_id[b["object_id"]].get("pickupable")):
                continue
            pb = by_id[b["object_id"]]["position"]
            d = math.dist((pa["x"], pa["y"], pa["z"]), (pb["x"], pb["y"], pb["z"]))
            if d > threshold:
                continue
            out.append({
                "subject": a["vg150_class"], "predicate": "near",
                "object": b["vg150_class"],
                "subject_id": a["object_id"], "object_id": b["object_id"],
                "annotation": "geometric", "distance_m": round(d, 3),
                "loose": bool(LOOSE_MAPPINGS
                              & {by_id[a["object_id"]]["objectType"],
                                 by_id[b["object_id"]]["objectType"]}),
            })
    return out


def vertical_relations(
    objects: Sequence[Dict[str, Any]],
    present: Sequence[Dict[str, Any]],
    min_gap: float = UNDER_MIN_GAP_M,
    max_gap: float = UNDER_MAX_GAP_M,
    exclude_pairs: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    `under` and `above`, derived geometrically and emitted as an inverse pair.

    The host's XZ footprint must cover the target's centre and the host's bottom
    face must sit `min_gap`..`max_gap` above the target's top face.  Below the
    lower bound the target is really resting ON the host; above the upper bound
    the host does not occlude it.

    Worth knowing when reading results: EGTR scored 0.000 on `under` over the
    sample set, and that zero is not attributable -- the model's own VG150 recall
    for `under` is only 0.071, and a geometric rule may also label pairs a human
    would not ("pot under cabinet" is geometrically true but not obviously
    annotation-worthy).  `above` additionally supplies the only pitch-UP examples
    in the set; every other predicate here looks level or down.
    """
    by_id = {o["objectId"]: o for o in objects}
    out = []
    for entry in present:
        target = by_id[entry["object_id"]]
        if not target.get("pickupable"):
            continue
        tb = _bbox3d(target)
        if tb is None:
            continue
        for other in present:
            if other["object_id"] == entry["object_id"]:
                continue
            if other["vg150_class"] == entry["vg150_class"]:
                continue
            if (exclude_pairs and frozenset((entry["object_id"],
                                             other["object_id"])) in exclude_pairs):
                continue
            host = by_id[other["object_id"]]
            # The overhead object must be structural.  Without this, a bottle
            # standing on a higher shelf became the "host" of a vase below it --
            # geometrically true, semantically empty.
            if host.get("pickupable"):
                continue
            hb = _bbox3d(host)
            if hb is None:
                continue
            if (abs(tb["cx"] - hb["cx"]) > hb["sx"] / 2
                    or abs(tb["cz"] - hb["cz"]) > hb["sz"] / 2):
                continue
            gap = (hb["cy"] - hb["sy"] / 2) - (tb["cy"] + tb["sy"] / 2)
            if not (min_gap <= gap <= max_gap):
                continue
            loose = bool(LOOSE_MAPPINGS & {target["objectType"],
                                           host["objectType"]})
            out.append({
                "subject": entry["vg150_class"], "predicate": "under",
                "object": other["vg150_class"],
                "subject_id": entry["object_id"], "object_id": other["object_id"],
                "annotation": "geometric", "loose": loose,
                "gap_metres": round(gap, 3),
            })
            out.append({
                "subject": other["vg150_class"], "predicate": "above",
                "object": entry["vg150_class"],
                "subject_id": other["object_id"], "object_id": entry["object_id"],
                "annotation": "geometric", "loose": loose,
                "gap_metres": round(gap, 3),
            })
    return out


def between_relations(
    objects: Sequence[Dict[str, Any]],
    present: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    `between`, with the object slot holding TWO flankers.

    Deliberately strict.  The naive reading -- any target near the midpoint of any
    two objects -- produced 672 instances over 40 scenes, versus 26 in the entire
    VG150 test split, and the examples were nonsense ("bottle between drawer and
    vase").  Five constraints fix that:

      * flankers must be among the target's `BETWEEN_NEAREST_K` nearest objects
      * flankers must be comparable in size to the target, not furniture
      * the target must sit within 10% of the flanker span from its midpoint
      * the flankers must lie on genuinely opposite sides (angle > 120 deg)
      * the span itself must be 0.2-1.2 m

    Result: ~11 instances over 40 scenes, the same order as VG, with examples like
    "apple between potato and lettuce".

    Worth noting for interpretation: EGTR scored 0.000 on `between` both before
    and after the improvement being evaluated, so this predicate is expected to
    contribute nothing to recall -- it is here for coverage of the GAP family, not
    because the model is expected to get it.
    """
    import itertools

    # `between` is WORLD-relative: the configuration does not depend on the camera,
    # so it belongs to the scene.  Restricting it to one view's visible objects
    # produced zero instances -- the strict nearest-neighbour and opposed-sides
    # constraints almost never line up within a single view's object subset.
    # Called with present=None it runs over the whole scene.
    if present is None:
        present = [
            {"object_id": o["objectId"], "vg150_class": thor_to_vg150(o["objectType"])}
            for o in objects if thor_to_vg150(o["objectType"])
        ]

    by_id = {o["objectId"]: o for o in objects}
    boxes = {}
    for entry in present:
        box = _bbox3d(by_id[entry["object_id"]])
        if box is not None:
            boxes[entry["object_id"]] = box

    out = []
    for entry in present:
        target = by_id[entry["object_id"]]
        if not target.get("pickupable"):
            continue
        tb = boxes.get(entry["object_id"])
        if tb is None:
            continue
        target_centre = np.array([tb["cx"], tb["cy"], tb["cz"]])
        target_size = max(tb["sx"], tb["sy"], tb["sz"])

        neighbours = sorted(
            (o for o in present
             if o["object_id"] != entry["object_id"] and o["object_id"] in boxes),
            key=lambda o: np.linalg.norm(
                np.array([boxes[o["object_id"]]["cx"], boxes[o["object_id"]]["cy"],
                          boxes[o["object_id"]]["cz"]]) - target_centre),
        )[:BETWEEN_NEAREST_K]
        neighbours = [
            o for o in neighbours
            if max(boxes[o["object_id"]]["sx"], boxes[o["object_id"]]["sy"],
                   boxes[o["object_id"]]["sz"])
            < BETWEEN_MAX_FLANKER_SIZE * target_size
        ]

        for a, b in itertools.combinations(neighbours, 2):
            if a["vg150_class"] == b["vg150_class"] == entry["vg150_class"]:
                continue
            ba, bb_ = boxes[a["object_id"]], boxes[b["object_id"]]
            ca = np.array([ba["cx"], ba["cy"], ba["cz"]])
            cb = np.array([bb_["cx"], bb_["cy"], bb_["cz"]])
            midpoint = (ca + cb) / 2.0
            span = float(np.linalg.norm((ca - cb)[[0, 2]]))
            if not (BETWEEN_MIN_SPAN_M < span < BETWEEN_MAX_SPAN_M):
                continue
            if np.linalg.norm((target_centre - midpoint)[[0, 2]]) > BETWEEN_MAX_OFFSET * span:
                continue
            if abs(target_centre[1] - midpoint[1]) > BETWEEN_MAX_DY_M:
                continue
            v1, v2 = (ca - target_centre)[[0, 2]], (cb - target_centre)[[0, 2]]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            if float(v1 @ v2) / (n1 * n2) > BETWEEN_MAX_COSINE:
                continue
            out.append({
                "subject": entry["vg150_class"],
                "predicate": "between",
                # two flankers -- the only predicate here with a 2-element object
                "object": [a["vg150_class"], b["vg150_class"]],
                "subject_id": entry["object_id"],
                "object_id": [a["object_id"], b["object_id"]],
                "annotation": "geometric",
                "span_m": round(span, 3),
                "loose": bool(LOOSE_MAPPINGS & {
                    target["objectType"],
                    by_id[a["object_id"]]["objectType"],
                    by_id[b["object_id"]]["objectType"]}),
            })
    return out


def between_relations_2d(
    objects: Sequence[Dict[str, Any]],
    present: Sequence[Dict[str, Any]],
    min_voverlap: float = BETWEEN_MIN_VOVERLAP,
    max_size_ratio: float = BETWEEN_MAX_SIZE_RATIO,
) -> List[Dict[str, Any]]:
    """
    `between` from image geometry: B and C are the target's immediate left and
    right neighbours in the frame.

    This replaces a 3D-collinearity rule that produced one usable relation in 200
    views.  The image-space reading is also the more faithful one -- VG is
    annotated from photographs, where depth is not available, so "between" is a
    statement about apparent arrangement.  Requiring the flankers to be the
    IMMEDIATE neighbours is what keeps it honest: without that, any three objects
    roughly in a row qualify and the count explodes to 7530 with subjects like
    "shelf between shelf and vase".

    At most one relation per subject, since each has a single nearest neighbour on
    either side.

    Caveat worth stating in any write-up: this is "looks between", not "is
    physically between".  A pot on a counter with a distant cabinet to its left and
    a distant plant to its right qualifies.  That is what an annotator would mark,
    but it is a choice, not a fact.
    """
    by_id = {o["objectId"]: o for o in objects}
    out = []
    for entry in present:
        if by_id[entry["object_id"]]["objectType"] in STRUCTURAL_TYPES:
            continue
        tx1, ty1, tx2, ty2 = entry["bbox_xyxy"]
        target_area = entry["area_fraction"]
        left = right = None

        for other in present:
            if other["object_id"] == entry["object_id"]:
                continue
            if other["area_fraction"] > max_size_ratio * target_area:
                continue
            oy1, oy2 = other["bbox_xyxy"][1], other["bbox_xyxy"][3]
            overlap = min(ty2, oy2) - max(ty1, oy1)
            if overlap <= min_voverlap * min(ty2 - ty1, oy2 - oy1):
                continue          # not in the same row
            centre = (other["bbox_xyxy"][0] + other["bbox_xyxy"][2]) / 2
            if centre < tx1:
                if left is None or centre > (left["bbox_xyxy"][0]
                                             + left["bbox_xyxy"][2]) / 2:
                    left = other
            elif centre > tx2:
                if right is None or centre < (right["bbox_xyxy"][0]
                                              + right["bbox_xyxy"][2]) / 2:
                    right = other

        if left is None or right is None:
            continue
        if left["vg150_class"] == right["vg150_class"] == entry["vg150_class"]:
            continue
        out.append({
            "subject": entry["vg150_class"],
            "predicate": "between",
            "object": [left["vg150_class"], right["vg150_class"]],
            "subject_id": entry["object_id"],
            "object_id": [left["object_id"], right["object_id"]],
            "annotation": "geometric_2d",
            "loose": bool(LOOSE_MAPPINGS & {
                by_id[entry["object_id"]]["objectType"],
                by_id[left["object_id"]]["objectType"],
                by_id[right["object_id"]]["objectType"]}),
        })
    return out


def ground_truth_relations(
    event, present: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    All derivable relations for ONE view.

    Five predicates over three action families, which is as far as THOR ground
    truth honestly reaches:

      on / in            parentReceptacles, split by container  (high confidence)
      behind / in front  2D overlap + depth order, camera-relative
      under / above      host footprint covers target, with vertical clearance
      near               centre distance under a threshold
    `between` is deliberately NOT included.  Three definitions were tried and none
    was usable: 3D collinearity gave 1 relation over 200 views; "anything
    horizontally between" gave 7530 with subjects like "shelf between shelf and
    vase"; immediate left/right image neighbours gave 92 but with no distance
    bound, so a vase on a shelf and a pillow on a sofa several metres apart
    counted as neighbours purely because nothing else sat between them in the
    frame.  EGTR also scores 0.000 on `between` both before and after the
    improvement under evaluation, so more samples of a questionable relation would
    buy nothing.  `between_relations` and `between_relations_2d` remain below,
    unused, in case a defensible definition turns up.

    The remaining 43 VG150 predicates are not derivable: holding/wearing need an
    agent and iTHOR has none, has/part of need part decomposition and THOR objects
    are atomic, and says/made of have no geometry at all.
    """
    objects = event.metadata["objects"]
    allowed = {o["object_id"] for o in present}
    support = support_relations(objects, allowed_ids=allowed)
    # Pairs already joined by on/in are excluded from the geometric predicates:
    # an object resting on its support trivially overlaps it and sits near it, and
    # neither reading is a useful occlusion or proximity example.
    supported = {frozenset((r["subject_id"], r["object_id"])) for r in support}
    return (support
            + occlusion_relations(objects, present, exclude_pairs=supported)
            + vertical_relations(objects, present, exclude_pairs=supported)
            + proximity_relations(objects, present, exclude_pairs=supported))


def draw_ground_truth(frame_rgb, present, relations) -> np.ndarray:
    """Preview image with GT boxes, VG150 labels, and relation arrows."""
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    centres = {}
    for entry in present:
        x1, y1, x2, y2 = entry["bbox_xyxy"]
        centres[entry["object_id"]] = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (90, 220, 90), 2)
        label = f'{entry["vg150_class"]} ({entry["thor_type"]})'
        cv2.putText(img, label, (x1 + 2, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (x1 + 2, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 240, 90), 1, cv2.LINE_AA)

    for rel in relations:
        a = centres.get(rel["subject_id"])
        # `between` carries two flanker ids; draw an arrow to each
        targets = (rel["object_id"] if isinstance(rel["object_id"], list)
                   else [rel["object_id"]])
        if not a:
            continue
        for target_id in targets:
            b = centres.get(target_id)
            if not b:
                continue
            colour = {"on": (240, 170, 60), "in": (90, 240, 240),
                      "behind": (240, 90, 200), "in front of": (200, 120, 255),
                      "under": (120, 255, 160), "above": (60, 200, 90),
                      "near": (160, 160, 160), "between": (60, 90, 250),
                      }.get(rel["predicate"], (200, 200, 200))
            cv2.arrowedLine(img, a, b, colour, 2, tipLength=0.05)
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            cv2.putText(img, rel["predicate"], mid, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, rel["predicate"], mid, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colour, 1, cv2.LINE_AA)
    return img


def refresh_ground_truth(args) -> int:
    """
    Recompute GT for samples that already exist, without re-rendering.

    Viewpoint selection depends on the GT (it scores candidates by relation
    count), so regenerating from scratch after a GT change produces DIFFERENT
    images -- which silently invalidates any predictions already computed on the
    old ones.  Measured: 10 of 12 images changed.  This path teleports to each
    stored camera pose instead, so the images are bit-identical and existing
    inference results remain usable.
    """
    from robot.robot_controller import RobotController

    index_path = os.path.join(args.outdir, "index.json")
    with open(index_path, encoding="utf-8") as handle:
        old = json.load(handle)["samples"]
    if not old:
        print("nothing to refresh")
        return 1

    gt_dir = os.path.join(args.outdir, "gt")
    meta_dir = os.path.join(args.outdir, "meta")
    robot = RobotController(scene=old[0]["scene"], width=old[0]["width"],
                            height=old[0]["height"], headless=args.headless,
                            seed=args.seed, verbose=False)
    updated = []
    try:
        for record in old:
            cam = record["camera"]
            robot.reset(record["scene"])
            robot.teleport(position=cam["position"], yaw=cam["yaw"],
                           horizon=cam["horizon"], standing=True)
            robot.height_level = "stand"

            present = visible_objects(robot.event, record["width"], record["height"])
            relations = ground_truth_relations(robot.event, present)

            cv2.imwrite(os.path.join(gt_dir, f"{record['name']}_gt.png"),
                        draw_ground_truth(robot.event.frame, present, relations))
            record["objects"] = present
            record["relations"] = relations
            with open(os.path.join(meta_dir, f"{record['name']}.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(record, handle, indent=1)
            updated.append(record)

            kinds = {}
            for r in relations:
                kinds[r["predicate"]] = kinds.get(r["predicate"], 0) + 1
            print(f"{record['name']:<18} {len(present):>2} objects, "
                  f"{len(relations):>2} relations {kinds}")
    finally:
        robot.stop()

    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump({"n": len(updated), "samples": updated}, handle, indent=1)
    print(f"\nrefreshed GT for {len(updated)} samples; images untouched")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export THOR frames + GT graphs")
    parser.add_argument("--n", type=int, default=12, help="samples to export")
    parser.add_argument("--scenes", default=None,
                        help="comma-separated scene list; default spans 4 room types")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--tries", type=int, default=25,
                        help="candidate viewpoints scored per scene")
    parser.add_argument("--outdir", default="samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--refresh-gt", action="store_true",
                        help="recompute ground truth for EXISTING samples at their "
                             "stored camera poses, leaving the images untouched. "
                             "Use after changing the GT logic so predictions "
                             "already computed on those images stay valid.")
    args = parser.parse_args(argv)

    if args.refresh_gt:
        return refresh_ground_truth(args)

    scenes = (args.scenes.split(",") if args.scenes else DEFAULT_SCENES)
    # Keep raw frames in their own directory so the whole folder can be handed
    # straight to an inference script without filtering out previews and JSON.
    rgb_dir = os.path.join(args.outdir, "rgb")
    gt_dir = os.path.join(args.outdir, "gt")
    meta_dir = os.path.join(args.outdir, "meta")
    for d in (rgb_dir, gt_dir, meta_dir):
        os.makedirs(d, exist_ok=True)

    import random

    from robot.robot_controller import RobotController, _xz, yaw_towards

    rng = random.Random(args.seed)
    robot = RobotController(scene=scenes[0], width=args.width, height=args.height,
                            headless=args.headless, seed=args.seed, verbose=False)

    index: List[Dict[str, Any]] = []
    per_scene = max(1, args.n // len(scenes))

    try:
        for scene in scenes:
            if len(index) >= args.n:
                break
            robot.reset(scene)
            reachable = robot.get_reachable_positions(refresh=True)
            if len(reachable) == 0:
                continue

            # Score candidate viewpoints by how much nameable content they show,
            # so the samples test EGTR rather than test empty walls.
            candidates = []
            for _ in range(args.tries):
                spot = reachable[rng.randrange(len(reachable))]
                yaw = rng.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270,
                                  300, 330])
                horizon = rng.choice([0.0, 15.0, 30.0])
                robot.teleport(
                    position={"x": float(spot[0]), "y": robot.agent_position["y"],
                              "z": float(spot[1])},
                    yaw=float(yaw), horizon=horizon, standing=True,
                )
                present = visible_objects(robot.event, args.width, args.height)
                if len(present) < MIN_OBJECTS:
                    continue
                relations = ground_truth_relations(robot.event, present)
                # prefer views rich in BOTH objects and relations
                score = len(present) + 3 * len(relations)
                candidates.append((score, float(spot[0]), float(spot[1]),
                                   float(yaw), horizon))

            candidates.sort(key=lambda c: -c[0])
            for score, x, z, yaw, horizon in candidates[:per_scene]:
                if len(index) >= args.n:
                    break
                robot.teleport(
                    position={"x": x, "y": robot.agent_position["y"], "z": z},
                    yaw=yaw, horizon=horizon, standing=True,
                )
                present = visible_objects(robot.event, args.width, args.height)
                relations = ground_truth_relations(robot.event, present)

                name = f"{scene}_{len(index):02d}"
                cv2.imwrite(os.path.join(rgb_dir, f"{name}.png"),
                            cv2.cvtColor(robot.event.frame, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(gt_dir, f"{name}_gt.png"),
                            draw_ground_truth(robot.event.frame, present, relations))

                record = {
                    "name": name,
                    "scene": scene,
                    "image": f"rgb/{name}.png",
                    "width": args.width,
                    "height": args.height,
                    "camera": {
                        "position": {k: round(v, 4)
                                     for k, v in robot.agent_position.items()},
                        "yaw": round(robot.agent_yaw, 2),
                        "horizon": round(robot.camera_horizon, 2),
                        "eye_height_m": robot.HEIGHT_LEVELS[robot.height_level],
                        "fov_vertical_deg": round(robot.fov_vertical, 2),
                        "fov_horizontal_deg": round(robot.fov_horizontal, 2),
                    },
                    "objects": present,
                    "relations": relations,
                }
                with open(os.path.join(meta_dir, f"{name}.json"), "w",
                          encoding="utf-8") as handle:
                    json.dump(record, handle, indent=1)
                index.append(record)
                print(f"{name:<18} {len(present):>2} objects, "
                      f"{len(relations):>2} relations   "
                      f"classes: {sorted({o['vg150_class'] for o in present})}")
    finally:
        robot.stop()

    with open(os.path.join(args.outdir, "index.json"), "w", encoding="utf-8") as h:
        json.dump({"n": len(index), "samples": index}, h, indent=1)

    classes: Dict[str, int] = {}
    predicates: Dict[str, int] = {}
    for record in index:
        for o in record["objects"]:
            classes[o["vg150_class"]] = classes.get(o["vg150_class"], 0) + 1
        for r in record["relations"]:
            predicates[r["predicate"]] = predicates.get(r["predicate"], 0) + 1

    print(f"\n{len(index)} samples -> {args.outdir}/  "
          f"(rgb/ = raw frames for EGTR, gt/ = previews, meta/ = ground truth)")
    print(f"objects total: {sum(classes.values())} over "
          f"{len(classes)} VG150 classes")
    print(f"  {dict(sorted(classes.items(), key=lambda kv: -kv[1]))}")
    print(f"relations total: {predicates}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
