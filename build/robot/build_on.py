"""
build_on.py -- `Find the cup on the chair`, with something in the way.

The two existing lists both say `behind`, and `behind` is the predicate EGTR is
WORST at: measured over 4528 instructions on `datasets/sgg/occlusion_ds4`
(`analysis/eval_grounding.py`), `on` grounds at 45.9% against `behind`'s 22.2%.
A viewpoint result on `behind` alone cannot separate "moving helps" from "the
relation was unreadable to begin with".  This list removes that confound.

    [target] on the [chair]        <- the answer, and the instruction's B
         |
         |     [distract]          the same asset, on the floor
    [occluder]                     floor-standing, in the way, NEVER NAMED
         |
      [camera]                     1.5-2.5 m back, standing

Exactly one copy is on the chair; the other is on the floor beside it.  With
both on the seat the relation would do no work, which is the failure
`build_tabletop.py` fixed for `behind` by putting the distractor at the
occluder's depth.

FLOOR GEOMETRY, NOT TABLETOP.  Everything in the other two lists sits on one
table and the robot stands 0.55-1.15 m away, so the occluder has to be a small
object squeezed onto the same surface.  Here the occluder stands on the FLOOR
between the robot and the chair, which is what a thing in the way actually is,
and lifts the size and standoff limits with it -- a house plant at 2 m reads as
an obstacle rather than as a prop balanced beside the target.

WHY A STOOL AND NOT A DINING CHAIR, and why the generator MEASURES rather than
trusting a list.  `place_on` stands a thing on a surface whose height it is
told, and that height is the axis-aligned box's top.  For a chair with a back
that top is the BACKREST -- measured sag 0.22 to 0.76 m -- so the target would
balance on the top rail.  Every `box` asset in THOR fails the same way for a
different reason: all 59 are OPEN containers, sag equal to their whole height,
and an object placed at the box top floats over the opening.  This list was
first built on `box` and the pictures showed cups hovering over cardboard.

So `flat_topped` looks straight down with a depth camera and keeps only assets
whose surface IS the box top.  Of the `chair` pool that leaves the stools and
footstools: 13 of 52 flat, seats 0.45-0.75 m.

AND THEN `nameable` ASKS EGTR.  A landmark the model does not call a chair
cannot be grounded however good the geometry is, and the flat-top screen alone
left `Footstool_1` (argmaxes to `table`) and `Stool_1_1` (to `sink`) in the
pool.  Both screens are run here rather than written down as a list, because a
list drifts silently when the asset pool or the checkpoint changes -- which is
the failure `build_tabletop`'s own hand-typed asset list had.

Selection ranks by p(class) and not by argmax, so a landmark that argmaxes
elsewhere is still SHORTLISTED -- all 13 rank 1st for `chair`.  The screen is
therefore stricter than the pipeline needs, and deliberately: a scene a person
cannot read as `on the chair` is not a fair question to ask a robot.

    python build/robot/build_on.py --n 12 --out datasets/robot/cases_on.json
"""

from __future__ import annotations

# `python build/robot/<script>.py` puts this directory on sys.path, not
# the repo root.  Same shim as its siblings; `-m` does not need it.
import os as _os
import sys as _sys
if __package__ in (None, ""):
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from build.robot.build_tabletop import (MIN_PIXELS, TARGET_CLASSES,
                                               TARGET_SIZE, catalogue,
                                               clear_of, occluder_yaw, reject,
                                               stage_occluder)
from robot.world.proc_scene import (ROOM, aim_at, look_from, open_room, place_on,
                              spawn, surface_top, visible_box, visible_pixels)

#: Classes the landmark may be -- what the target stands on, and the only object
#: the sentence names besides the target.  `chair` is the one VG150 class with
#: THOR assets that are both flat-topped and seat-height; see the module note.
LANDMARK_CLASSES = ("chair",)

#: Landmark size band, metres: tall enough that its seat is a surface in its own
#: right, short enough that a standing camera can look down onto it.
LANDMARK_SIZE = (0.40, 0.80)

#: Classes the occluder may be.  FLOOR-STANDING AND TALL, which is what makes
#: this list's obstacle an obstacle rather than a prop: it has to cover a target
#: sitting ~0.5 m up, seen from a camera 1.5-2.5 m back.
OCCLUDER_CLASSES_ON = ("plant", "lamp", "box", "chair", "stand")
OCCLUDER_SIZE_ON = (0.70, 1.60)

#: How far below the bounding box's top the real surface may be before an asset
#: is rejected as not flat.  Two centimetres is the depth camera's own noise on
#: a flat seat; every asset that fails does so by 0.2 m or more, so the cut sits
#: nowhere near anything.
FLAT_SAG = 0.02

#: The overhead camera `flat_topped` measures from: above the tallest landmark,
#: below the ceiling.
PROBE_Y = 2.2

#: How far `nameable` stands back to photograph a candidate landmark, metres.
#: The middle of `STANDOFF`, so the screen sees roughly what a case will.
NAME_STANDOFF = 1.8

#: Where the robot stands, metres from the chair -- NEAREST FIRST, first legal
#: pose wins.  Further out than `build_tabletop`'s 0.55-1.15: there the target
#: is a 10 cm object on a table and distance is the only lever on how many
#: pixels it gets, here there is a floor-standing obstacle between the two and
#: standing inside it is not a pose.
STANDOFF = (1.5, 1.8, 2.1, 2.4)

#: Where the occluder goes, as a fraction of the way from the chair to the
#: camera.  Swept nearest-first by `stage_occluder`; nearer the camera means a
#: given asset covers more, which is the second lever the bisection needs.
ALONG_ON = (0.35, 0.50, 0.65)

#: How far to the side the floor copy stands, metres from the chair's centre.
#: Far enough that the two copies do not overlap in the frame, near enough that
#: both are in it -- the sweep is +-30 degrees and a copy outside the frame
#: cannot be compared with one inside.
FLOOR_OFFSET = (0.55, 0.85)

#: How much of the floor copy the occluder may hide.  Its job is to be nameable
#: and NOT be the answer; a hidden one makes the case a naming problem again.
MAX_DISTRACTOR_OCCLUSION = 0.10

#: How far in from the seat's edge the target must sit, as a fraction of the
#: seat's half-width.  A round stool's axis-aligned box has air in its corners,
#: the same reason `build_tabletop` has `TABLE_MARGIN`.
SEAT_INSET = 0.55


def flat_topped(controller, assets: Sequence[str],
                sag: float = FLAT_SAG) -> List[Tuple[str, float, float]]:
    """`[(asset, seat height, seat half-width)]` for the ones you can stand on.

    MEASURED, NOT LISTED.  Looks straight down at the asset's centre and
    compares the depth it reads against the top of the axis-aligned box.  An
    open box or a chair back shows a sag of the object's whole height; a seat
    shows the camera's own noise.  See the module note for what this caught.
    """
    keep = []
    centre = ROOM / 2.0
    for asset in sorted(assets):
        controller.reset()
        try:
            entry = spawn(controller, asset, "probe", centre, 0.0, centre)
        except Exception:                                    # noqa: BLE001
            continue
        box = entry["axisAlignedBoundingBox"]
        top = surface_top(entry)
        controller.step(action="AddThirdPartyCamera",
                        position={"x": box["center"]["x"], "y": PROBE_Y,
                                  "z": box["center"]["z"]},
                        rotation={"x": 90, "y": 0, "z": 0}, fieldOfView=60.0)
        frames = controller.step(action="Pass").third_party_depth_frames
        if not frames:
            continue
        depth = frames[0]
        h, w = depth.shape[:2]
        patch = depth[int(h * .45):int(h * .55), int(w * .45):int(w * .55)]
        if top - (PROBE_Y - float(np.median(patch))) < sag:
            keep.append((asset, top,
                         min(box["size"]["x"], box["size"]["z"]) / 2.0))
    return keep


def class_rank(egtr, frame, want: str) -> int:
    """Where the best query for `want` sits in the p(want) ranking, 1-based.

    THE QUANTITY THE PIPELINE SELECTS ON.  `grounding.candidates` sorts every
    query by p(class) and keeps the top K, so this number IS whether the object
    has a candidate; the class the query argmaxes to never enters.
    """
    from robot.policy.grounding import class_mass
    from robot.sgg_live import raw_predict

    probs = raw_predict(egtr, frame)["probs_softmax"].detach().cpu().float()
    mass = class_mass(probs, egtr, want)
    if mass is None:
        return 10 ** 6
    mass = np.asarray(mass)
    return int(np.where(np.argsort(-mass) == int(mass.argmax()))[0][0]) + 1


def nameable(controller, egtr, seats: Sequence[Tuple[str, float, float]],
             want: str) -> List[Tuple[str, float, float]]:
    """Keep the landmarks EGTR actually CALLS `want` -- argmax, not rank.

    STRICTER THAN THE PIPELINE, ON PURPOSE, and this is the one screen here that
    is.  Rank is what `grounding.candidates` selects on, and by rank all 13
    flat-topped stools pass: every one of them is the top query for `chair`.  A
    rank screen therefore filters nothing and the list keeps `Footstool_1`, which
    the model calls a `table`, and `Stool_1_1`, which it calls a `sink`.

    Those are black cuboids.  `Find the cup on the chair` in a room whose only
    seat is a black cuboid is a question with no answer a person would agree
    with either, and grading a viewpoint rule on it measures the asset library.
    Argmax keeps 5 of 13 and every one of them reads as a stool.

    Photographed alone, from `NAME_STANDOFF`: an asset that needs its scene to
    be recognised is not the asset to build a landmark from.
    """
    from robot.policy.grounding import class_mass
    from robot.sgg_live import raw_predict

    keep = []
    centre = ROOM / 2.0
    for asset, top, half in seats:
        controller.reset()
        spawn(controller, asset, "seat", centre, 0.0, centre + 0.9)
        event, _ = aim_at(controller, centre, centre + 0.9 - NAME_STANDOFF, 0.0,
                          (centre, top, centre + 0.9))
        probs = raw_predict(egtr, event.frame)["probs_softmax"].detach().cpu()
        probs = probs.float()
        query = int(np.asarray(class_mass(probs, egtr, want)).argmax())
        if egtr["obj_names"].get(int(probs[query].argmax()) + 1) == want:
            keep.append((asset, top, half))
    return keep


def nameable_targets(controller, egtr, pools: Dict[str, List[str]],
                     seat: Tuple[str, float, float],
                     rank: int) -> Dict[str, List[str]]:
    """The target assets that still rank for their class ON A SEAT, this far off.

    AT THIS LIST'S OWN GEOMETRY.  `build_tabletop` screened the same classes on a
    table at 0.55-1.15 m; here the object sits on a seat 1.5-2.4 m away and is a
    quarter the pixels.  A pool screened at the wrong scale is a pool of objects
    that have no candidate, and no ranking rule recovers those.
    """
    centre = ROOM / 2.0
    seat_asset, top, _ = seat
    keep: Dict[str, List[str]] = {}
    for want in sorted(pools):
        for asset in pools[want]:
            controller.reset()
            spawn(controller, seat_asset, "seat", centre, 0.0, centre + 0.9)
            place_on(controller, asset, "target", centre, top, centre + 0.9)
            event, _ = aim_at(controller, centre, centre + 0.9 - STANDOFF[0],
                              0.0, (centre, top + 0.08, centre + 0.9))
            if class_rank(egtr, event.frame, want) <= rank:
                keep.setdefault(want, []).append(asset)
    return keep


def stand_back(controller, x: float, tz: float) -> Optional[float]:
    """The nearest z the robot can legally stand at, looking down +z at `tz`.

    `STANDOFF` nearest-first, keeping the first pose THOR accepts WITHOUT
    `forceAction`.  That is the whole guarantee a case can be replayed and
    walked: `rebuild` needs no force, and a refused step can be reported as a
    refusal rather than teleported through the furniture.
    """
    for standoff in STANDOFF:
        z = tz - standoff
        event = controller.step(action="Teleport",
                                position={"x": float(x), "y": 0.9,
                                          "z": float(z)},
                                rotation={"x": 0, "y": 0.0, "z": 0},
                                horizon=0.0, standing=True, forceAction=False)
        if event.metadata["lastActionSuccess"]:
            return z
    return None


def one_case(controller, rng: random.Random, index: int,
             band: Tuple[float, float, float], catalogues,
             gap: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Build one scene and return its case record, or None if it will not stage."""
    targets, occluders, seats = catalogues
    target_class = sorted(targets)[index % len(targets)]
    asset = rng.choice(targets[target_class])
    # THE OCCLUDER MUST BE NEITHER THE TARGET'S CLASS NOR THE LANDMARK'S: it is
    # never named, but a third copy competes in the same p(class) ranking, and
    # the shortlist is built by p(class) alone.
    choices = [c for c in occluders if c not in (target_class, "chair")]
    if not choices or not seats:
        return None
    occ_class = rng.choice(choices)
    occ_asset = rng.choice(occluders[occ_class])
    seat_asset, _, seat_half = seats[rng.randrange(len(seats))]

    controller.reset()
    centre = ROOM / 2.0
    cz = centre + 0.9
    chair = spawn(controller, seat_asset, "landmark", centre, 0.0, cz)
    top = surface_top(chair)
    inset = seat_half * SEAT_INSET

    # THE TARGET FIRST, ALONE: the bisection measures against its unoccluded
    # pixel count, so nothing else may be in the picture yet.
    tx = centre + rng.uniform(-inset * 0.4, inset * 0.4)
    tz = cz + rng.uniform(-inset * 0.3, inset * 0.3)
    sign = rng.choice((-1.0, 1.0))
    _, target_pos = place_on(controller, asset, "target", tx, top, tz)

    z = stand_back(controller, tx, tz)
    if z is None:
        return reject("nowhere legal to stand and see it")
    cam = np.array([tx, z])
    yaw = 0.0
    event, horizon = aim_at(controller, cam[0], cam[1], yaw,
                            (tx, top + 0.08, tz))
    clear = visible_pixels(event, "target")
    if clear < MIN_PIXELS:
        return reject(f"target only {clear} px unoccluded")

    # THE SECOND COPY, ON THE FLOOR, on the side the occluder is NOT slid
    # towards, so the thing that hides the answer does not also hide it.
    dx = tx - sign * rng.uniform(*FLOOR_OFFSET)
    _, distract_pos = place_on(controller, asset, "distract", dx, 0.0, cz)

    # THE OCCLUDER, ON THE FLOOR between camera and chair, bisected sideways
    # against the target exactly as `build_tabletop` does it -- the machinery is
    # surface-agnostic, it only needs the resting height.
    occ_yaw = occluder_yaw(controller, occ_asset, occ_class)
    _, occ_pos = place_on(controller, occ_asset, "occluder", tx, 0.0,
                          tz - (gap if gap is not None else 0.7), occ_yaw)
    staged = stage_occluder(controller, "occluder", occ_pos["y"],
                            np.array([tx, tz]), cam, clear, band, sense=sign,
                            yaw=occ_yaw, gap=gap)
    if staged is None:
        return reject(f"{occ_asset} never landed in the band")

    event = look_from(controller, cam[0], cam[1], yaw, horizon, force=True)

    # VERIFY WHAT THE PLACEMENT WAS MEANT TO GUARANTEE.  Everything above is
    # arithmetic on bounding boxes; these are pixels.
    target_px = visible_pixels(event, "target")
    hidden_target = 1.0 - target_px / max(clear, 1)
    if not band[0] <= hidden_target <= band[2]:
        return reject(f"target ended at {hidden_target:.0%} hidden, outside "
                      f"the band")
    distract_px = visible_pixels(event, "distract")
    if distract_px < MIN_PIXELS:
        return reject(f"the floor copy is only {distract_px} px")
    hidden_distract = 1.0 - distract_px / max(clear_of(
        controller, cam, yaw, horizon, "distract", "occluder",
        staged["position"]), 1)
    if hidden_distract > MAX_DISTRACTOR_OCCLUSION:
        return reject(f"occluder hides {hidden_distract:.0%} of the floor copy, "
                      f"so the case is a naming problem again")
    # THE LANDMARK HAS TO BE IN THE PICTURE.  It is the instruction's B, and a
    # sentence whose landmark is off-frame cannot be grounded from this pose by
    # any rule.  The occluder stands in front of both it and the target.
    landmark_px = visible_pixels(event, "landmark")
    if landmark_px < MIN_PIXELS:
        return reject(f"the chair is only {landmark_px} px")

    return {
        "scene": f"on|{index}",
        "instruction": f"Find the {target_class} on the chair",
        "predicate": "on",
        "subject_class": target_class, "object_class": "chair",
        "target_name": "target",
        # THE LANDMARK IS NOT THE OCCLUDER, which is the first list where those
        # two differ: the sentence names what the target STANDS ON, and what
        # hides it is never mentioned.  `fuse_live.task_for` reads
        # `landmark_name`, falling back to `occluder_name` for the older lists.
        "landmark_name": "landmark",
        "distractor_name": "distract", "occluder_name": "occluder",
        "occluder_position": staged["position"], "occluder_yaw": occ_yaw,
        "staged_occlusion": round(hidden_target, 3),
        "bisected_occlusion": staged["occlusion"],
        "clear_px": clear, "target_px": target_px,
        "distractor_px": distract_px, "landmark_px": landmark_px,
        "distractor_occlusion": round(hidden_distract, 3),
        "seat_height": round(top, 3),
        "standoff": round(float(tz - z), 3),
        "target_box": visible_box(event, "target"),
        # NO TABLE.  `robot.world.proc_scene.rebuild` spawns whatever `objects` names
        # and nothing else, so the chair being the only furniture is enough.
        "objects": [{"name": "landmark", "asset": seat_asset,
                     "position": {"x": centre, "y": 0.0, "z": cz}},
                    {"name": "target", "asset": asset, "position": target_pos},
                    {"name": "distract", "asset": asset,
                     "position": distract_pos},
                    {"name": "occluder", "asset": occ_asset,
                     "position": staged["position"], "yaw": occ_yaw}],
        "start": f"{cam[0]:.4f},{cam[1]:.4f},{yaw:.2f},{horizon:.2f}",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-occlusion", type=float, default=0.25)
    ap.add_argument("--target-occlusion", type=float, default=0.35)
    ap.add_argument("--max-occlusion", type=float, default=0.50)
    # SQUARE, to match the experiment; see `build_slot`.
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--target-classes", nargs="+", default=list(TARGET_CLASSES),
                    metavar="C")
    ap.add_argument("--occluder-classes", nargs="+",
                    default=list(OCCLUDER_CLASSES_ON), metavar="C",
                    help="floor-standing classes the obstacle may be")
    ap.add_argument("--landmark-classes", nargs="+",
                    default=list(LANDMARK_CLASSES), metavar="C",
                    help="what the target stands on -- the instruction's "
                         "landmark.  Screened for a flat top; see `flat_topped`.")
    ap.add_argument("--gap", type=float, default=None, metavar="M",
                    help="separation between target and obstacle in metres; "
                         "omitted, `ALONG_ON`'s fractions are swept")
    ap.add_argument("--occluder-width", type=float, default=0.55, metavar="M",
                    help="cap on the obstacle's horizontal extent.  0.80 let a "
                         "floor lamp's shade swallow the chair it was supposed "
                         "to stand in front of.")
    ap.add_argument("--no-name-screen", dest="name_screen",
                    action="store_false",
                    help="skip asking EGTR whether each end of the sentence has "
                         "a candidate.  For inspecting the geometry without "
                         "loading 5 GB of weights; the list it writes is not "
                         "one to measure on.")
    ap.add_argument("--name-rank", type=int, default=10, metavar="K",
                    help="how far down the p(class) ranking a TARGET asset's "
                         "own query may sit and still be used.  10 is "
                         "`--condition`'s default, so this is exactly `the "
                         "object has a candidate`.  The LANDMARK is screened "
                         "more strictly; see `nameable`.")
    ap.add_argument("--out", default="datasets/robot/cases_on.json")
    args = ap.parse_args(argv)

    band = (args.min_occlusion, args.target_occlusion, args.max_occlusion)
    rng = random.Random(args.seed)
    controller = open_room(args.width, args.height, args.fov)
    cases: List[Dict[str, Any]] = []
    try:
        pool = catalogue(controller, tuple(args.landmark_classes),
                         LANDMARK_SIZE, 1.00)
        flat = flat_topped(controller, [a for v in pool.values() for a in v])
        seats, want = flat, args.landmark_classes[0]
        targets = catalogue(controller, tuple(args.target_classes), TARGET_SIZE)
        in_band = sum(len(v) for v in targets.values())
        if args.name_screen:
            from robot.sgg_live import load_egtr

            egtr = load_egtr()
            seats = nameable(controller, egtr, flat, want)
            if seats:
                targets = nameable_targets(controller, egtr, targets, seats[0],
                                           args.name_rank)
        if not seats:
            print("  landmarks: none survived the screens")
            return 1
        if not targets:
            print("  no target asset ranks for its own class at this distance")
            return 1
        catalogues = (targets,
                      catalogue(controller, tuple(args.occluder_classes),
                                OCCLUDER_SIZE_ON, args.occluder_width),
                      seats)
        for label, got in zip(("targets:  ", "occluders:"), catalogues[:2]):
            print("  " + label + ", ".join(f"{k}x{len(v)}"
                                           for k, v in got.items()))
        if args.name_screen:
            print(f"             {in_band} in the size band -> "
                  f"{sum(len(v) for v in targets.values())} rank within "
                  f"{args.name_rank} on a seat at {STANDOFF[0]} m")
        print(f"  landmarks: {sum(len(v) for v in pool.values())} in the size "
              f"band -> {len(flat)} flat-topped -> {len(seats)} EGTR calls "
              f"`{want}`, seats "
              f"{min(s for _, s, _ in seats):.2f}-"
              f"{max(s for _, s, _ in seats):.2f} m")
        print("             " + ", ".join(a for a, _, _ in seats))
        attempt = 0
        while len(cases) < args.n and attempt < args.n * 6:
            case = None
            try:
                case = one_case(controller, rng, attempt, band, catalogues,
                                args.gap)
            except Exception as error:                       # noqa: BLE001
                print(f"  ! {type(error).__name__}: {error}", flush=True)
            attempt += 1
            if case:
                cases.append(case)
                print(f"[{len(cases)}/{args.n}] {case['instruction']:32s} "
                      f"{case['staged_occlusion']:.0%} hidden, target "
                      f"{case['target_px']} px on a "
                      f"{case['seat_height']:.2f} m seat, standoff "
                      f"{case['standoff']:.1f} m", flush=True)
    finally:
        controller.stop()

    if not cases:
        print("no cases built")
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"predicate_families": ["on"], "occlusion_band": list(band),
                   "procedural": True, "cases": cases}, fh, indent=1)
    occ = sorted(c["staged_occlusion"] for c in cases)
    print(f"\n{len(cases)}/{attempt} attempts -> {args.out}")
    print(f"  occlusion {occ[0]:.0%}-{occ[-1]:.0%}, median "
          f"{occ[len(occ) // 2]:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
