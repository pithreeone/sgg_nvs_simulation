"""
build_pairview_sweep.py -- two objects on a table, both fully visible, photographed from an arc.

The companion scene to `build_sideview_sweep.py`, and its complement.  That one stages an occlusion
and asks what hiding an object does to the detector.  This one stages NOTHING that is hidden: two
objects stand on a table, both completely visible from every frame, the camera flies a +-30 degree
arc about the pair, and the question is whether EGTR still says the same thing about them.

WHAT THE SCENE IS FOR.  Three ground-truth relations come out of one render and all three are
CONSTANT over the arc:

    back  on      table          physical support, cannot change with the camera
    front on      table          same
    back  behind  front          the depth ordering, and +-30 degrees is not enough to flip it

So every disagreement the model produces across the arc is the model's, not the scene's.  That is
the whole design constraint, and it is why the arc is short: at +-90 the pair would genuinely go
side-by-side and at +-180 `behind` would genuinely invert, and a figure containing a real inversion
cannot also be a figure about instability.  +-30 is also the span `fuse_live.py` actually orbits, so
the figure and the method sweep the same wedge.

`behind` IS THE PREDICATE THIS PROJECT RUNS ON.  The instruction template is `Find the {A} behind
the {B}` and the robot's stop rule is literally "the first triplet in the list labelled `behind`"
(`../sgg_nvs/../simulation/PIPELINE.md`, section 6).  A demonstration that the word survives or does
not survive a 30 degree step is a statement about the pipeline's floor, not about a curiosity.

THE CAMERA STANDS, and that is a reversal of the sideview sweep's choice.  There the camera had to
crouch or the sightline passed over the occluder.  Here there is no occluder, and crouching costs
the figure its subject: measured on the existing sideview renders, a camera at 0.90 m and 0.85 m
radius reduces a 0.834 m tabletop to an untextured expanse with no legs and no far edge, EGTR calls
it `room` or `tile` at p = 0.03-0.11, and the support predicate collapses to `in` / `made of` along
with it.  That is a picture of a table not being detected, not of a relation being unstable.
Standing at 1.575 m and 1.3 m out, the table is a table.

NOTHING IS ALLOWED TO OCCLUDE ANYTHING, and it is measured rather than assumed.  The two objects are
offset sideways as well as in depth, and how far sideways is not a constant: asset widths run from a
mug to a laptop and one lateral separation cannot serve both.  So the generator tries a ladder of
separations and keeps the first at which BOTH objects stay above --min_visible of their own
unobstructed pixel count at every azimuth tested, with each object's unobstructed count measured by
lifting the other one out of the room (`build_tabletop.clear_of`'s trick).  A case that never clears
the ladder is dropped.  The per-frame fractions are written into the sweep so the figure can print
them: a strip claiming "both objects are fully visible" has to show its evidence.

THE DEPTH MARGIN IS RECORDED PER FRAME for the same reason.  `vg_gt.py` derives `behind` from
`subject.distance > object.distance` with the smaller box as subject, so the sweep stores both
camera distances and the signed margin at every azimuth.  The margin shrinks as the arc swings
towards the lateral offset -- at lateral 0.28 and gap 0.32 it runs about 0.14 m to 0.42 m over
+-30 -- and a reader who wants to check that the ordering never flipped can read the numbers
instead of trusting the caption.

    python build/sgg/build_pairview_sweep.py --n 12
    python build/sgg/build_pairview_sweep.py --pairs cup:laptop bowl:bottle --span 30 --frames 13

Output, per case, under --out/<case_id>/:
    frame_00.png .. frame_NN.png     the arc, centre frame at azimuth 0
    sweep.json                       per frame: azimuth, camera, and for each of back/front/table
                                     the visible pixels, the unobstructed count, the box and the
                                     camera distance
"""

from __future__ import annotations

# `python build/sgg/<script>.py` puts build/sgg/ on sys.path, not the repo root.  Same shim as its siblings.
import os as _os
import sys as _sys
if __package__ in (None, ""):
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import itertools
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from build.robot.build_tabletop import (MIN_PIXELS, OCCLUDER_CLASSES,
                                               OCCLUDER_SIZE, TABLE,
                                               TARGET_CLASSES, TARGET_SIZE,
                                               catalogue, occluder_yaw,
                                               on_table, settle)
from robot.world.proc_scene import (ROOM, aim_at, look_from, open_room, place_on,
                              spawn, surface_top, visible_box, visible_pixels)
from vg.vg_gt import vg_calibrated_relations

#: Where the PAIR's midpoint sits, metres in from the table's near edge.  Both objects straddle it,
#: so the deeper one needs `INSET + gap/2` of table behind it and the near one must not hang off
#: the front; 0.60 leaves room for both against `TABLE_MARGIN` at every rung of the ladder.
INSET = 0.60

#: Smallest separation along the camera ray at azimuth 0, metres -- this is what makes one object
#: `behind` the other.  A FLOOR and not the value used: see `gap_for`.
GAP_MIN = 0.32

#: Depth bought on top of what `gap_for` needs, metres.  See that function: without it the ladder
#: lands every rung exactly on the acceptance threshold and asset placement error decides the case.
MARGIN_SLACK = 0.06

#: Sideways separations tried, in order, until both objects come out unoccluded.  The ladder starts
#: narrow because a narrow pair reads more clearly as ONE arrangement rather than two unrelated
#: objects, and widens only as far as the table allows -- 0.485 m either side of centre, so 0.90 is
#: the last rung that keeps both objects on the wood.
#:
#: IT HAS TO REACH THIS WIDE, and the arithmetic says why.  The two objects' separation across the
#: image is `lateral * cos(a) - gap * sin(a)`: the depth offset that makes one of them `behind` the
#: other also drags the near one ACROSS the far one as the camera swings towards its side.  At
#: gap 0.32 and 30 degrees that costs 0.16 m of the separation, and a laptop 0.30 m wide standing
#: nearer to the camera than a cup needs most of the rest.  Measured: 0.24 leaves the cup 45%
#: hidden at +15 degrees.
LATERAL_LADDER = (0.30, 0.44, 0.58, 0.72, 0.86)

#: Azimuths the ladder is tested at, as a fraction of --span.  The ends and the middle: whichever
#: separation survives those survives the frames between them, and testing all of them would triple
#: the generator's cost for a check the final pass repeats anyway.
LADDER_PROBES = (-1.0, -0.5, 0.0, 0.5, 1.0)

#: See the docstring: the camera stands, which is the opposite of the sideview sweep's choice.
STANDING = True


def gap_for(lateral: float, span_deg: float, min_margin: float) -> float:
    """How far apart in depth the pair must stand, given how far apart it stands sideways.

    A FIXED GAP IS WRONG, and the first version of this generator was wrong in exactly the way the
    arithmetic predicts.  The camera-frame depth difference at azimuth `a` is

        margin(a) = lateral * sin(a) + gap * cos(a)

    -- the sideways offset that keeps the two objects from overlapping ALSO swings one of them
    forward as the camera goes negative.  At lateral 0.58 and gap 0.32 that reaches -0.037 m at
    -30 degrees: the deeper object is genuinely the nearer one there, the ground truth flips from
    `behind` to `in front of` mid-arc, and the case is not a stability figure any more.  Measured,
    on `05_bottle_behind_laptop`, before this function existed.

    Inverting the worst case (`a = -span`) for `gap` gives the floor below.  It is why the ladder
    can widen without limit: every rung buys its own depth.

    THE SLACK IS NOT DECORATION.  Solved exactly, the formula puts the margin at -span at exactly
    `min_margin` for every rung -- so the ladder stops helping, and whether a case passes comes
    down to the difference between where an asset was placed and where its bounding-box centre
    ended up.  Measured that way, seven of sixteen cases failed at +0.028 to +0.054 m against a
    0.06 floor, at every rung identically.  `MARGIN_SLACK` buys the placement error a margin of
    its own.
    """
    span = math.radians(span_deg)
    return max(GAP_MIN, (min_margin + MARGIN_SLACK + lateral * math.sin(span)) / math.cos(span))


def arc_pose(centre_xz, radius: float, azimuth_deg: float) -> Tuple[float, float, float]:
    """Camera (x, z, yaw) on a circle about `centre_xz`.  Azimuth 0 is the staged viewpoint.

    THOR yaw is clockwise from +z with forward `(sin yaw, ., cos yaw)`, so a camera placed at
    `centre + R(sin a, -cos a)` looks back at the centre with yaw `-a` exactly.  Identical to
    `build_sideview_sweep.arc_pose`; it is four lines and importing across generators to save them
    would couple two scripts that are deliberately independent.
    """
    a = math.radians(azimuth_deg)
    x = float(centre_xz[0] + radius * math.sin(a))
    z = float(centre_xz[1] - radius * math.cos(a))
    return x, z, -float(azimuth_deg)


def unobstructed(controller, cam, yaw: float, horizon: float, name: str,
                 blocker: str, restore: Dict[str, Any]) -> int:
    """Pixels of `name` with `blocker` lifted out of the room -- the denominator for occlusion.

    `build_tabletop.clear_of` does this, but it is wrong here twice over.  It calls `look_from`
    without `standing`, so on a standing sweep the denominator would come from a crouched camera.
    And it restores the blocker with `settle`'s default yaw of ZERO, which turns the object back
    to face north -- `settle`'s own docstring warns about exactly that, having been bitten by it
    once already.  Measured before the fix: the first call straightened Laptop_7 out of its
    `occluder_yaw` half turn and every occlusion after it was against a differently-shaped
    laptop, which is what pinned the ladder at a mysterious 19-20% however wide the pair was
    pushed apart.
    """
    controller.step(action="TeleportObject", objectId=blocker,
                    position={"x": 0.0, "y": -5.0, "z": 0.0},
                    rotation={"x": 0, "y": 0, "z": 0}, forceAction=True)
    event = look_from(controller, cam[0], cam[1], yaw, horizon, force=True, standing=STANDING)
    seen = visible_pixels(event, name)
    settle(controller, blocker, restore["position"], restore["yaw"])
    return seen


def look(controller, centre_xz, radius: float, azimuth: float, aim_xyz):
    """One frame on the arc: teleport, pitch onto the pair's midpoint, return the event."""
    cx, cz, yaw = arc_pose(centre_xz, radius, azimuth)
    event, horizon = aim_at(controller, cx, cz, yaw, aim_xyz, standing=STANDING)
    return event, (cx, cz), yaw, horizon


def measure(controller, event, cam, yaw: float, horizon: float,
            placed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-object pixels, unobstructed pixels, box and camera distance in the current view."""
    camera = event.metadata["cameraPosition"]
    by_name = {o["name"]: o for o in event.metadata["objects"]}
    out = {}
    for name, other in (("back", "front"), ("front", "back")):
        seen = visible_pixels(event, name)
        clear = unobstructed(controller, cam, yaw, horizon, name, other, placed[other])
        entry = by_name.get(name)
        box = entry["axisAlignedBoundingBox"]["center"] if entry else None
        out[name] = {
            "visible_px": seen,
            "clear_px": clear,
            # Against the count with the OTHER OBJECT REMOVED, so this is the fraction of the
            # object that its partner hides -- not the fraction the table's own edge crops, which
            # is a property of the framing and identical in both counts.
            "occlusion": round(max(0.0, 1.0 - seen / clear), 4) if clear else 1.0,
            "box": visible_box(event, name),
            "distance": (round(math.dist((camera["x"], camera["y"], camera["z"]),
                                         (box["x"], box["y"], box["z"])), 4)
                         if box else None),
        }
    # The table carries no occlusion figure on purpose: the two objects DO stand on it and do cover
    # a few of its pixels, which is what `on` means.  A number that goes up when the scene is
    # working would only invite the wrong reading.
    entry = by_name.get("table")
    box = entry["axisAlignedBoundingBox"]["center"] if entry else None
    out["table"] = {
        "visible_px": visible_pixels(event, "table"),
        "box": visible_box(event, "table"),
        "distance": (round(math.dist((camera["x"], camera["y"], camera["z"]),
                                     (box["x"], box["y"], box["z"])), 4) if box else None),
    }
    for name in out:
        entry = by_name.get(name)
        out[name]["position"] = (entry["axisAlignedBoundingBox"]["center"] if entry else None)
    return out


def derived_relations(seen: Dict[str, Dict[str, Any]], classes: Dict[str, str],
                      width: int, height: int) -> List[Dict[str, Any]]:
    """What THIS PROJECT'S OWN convention says about the frame, not what this file thinks.

    The staged arrangement is `back behind front`, and the temptation is to assert that from the
    geometry and move on.  But the paper's ground truth is `vg_gt.vg_calibrated_relations`, which
    is not a depth comparison: it gates every family on a band of 2D descriptors fitted to what VG
    annotators actually wrote, so a pair far enough apart across the image comes out `near` however
    cleanly one of them is deeper.  A figure whose caption says `behind` while the paper's own
    ground truth would have said `near` is a figure that contradicts its own tables.

    Called per frame rather than once, because the descriptors it reads are 2D and every azimuth
    gives it different boxes.  `on` is excluded the same way the dataset builder excludes it: THOR
    asserts support directly through `parentReceptacles` and VG almost never double-labels a
    supported pair, so leaving the pairs in would have the derivation restate the tabletop.
    """
    shim = [{"object_id": name, "vg150_class": classes[name],
             "bbox_xyxy": seen[name]["box"], "position": seen[name]["position"],
             "distance": seen[name]["distance"], "loose": False}
            for name in ("back", "front", "table")
            if seen[name]["box"] and seen[name]["position"]]
    exclude = {frozenset(("back", "table")), frozenset(("front", "table"))}
    keep = lambda rs: [{"subject": r["subject"], "predicate": r["predicate"],
                        "object": r["object"]} for r in rs]
    # TWO PASSES, AND THE SECOND ONE IS A DEFENCE.  Single-label is the ground truth the paper is
    # scored against: one relation per pair, the family with the highest likelihood, because that
    # is what an annotator writes.  But the losing families are not FALSE -- two objects 0.3 m
    # apart really are `near` each other as well as one being behind the other, and `near` is
    # exactly what EGTR says about these pairs most often.  Scoring the model as wrong for saying
    # a true thing that the ground truth happened not to pick would be a figure a reviewer
    # dismantles in one sentence.  `also_true` is that sentence, answered in advance.
    #
    # `rate=None` ON THE SECOND PASS, and it is not a detail.  The default keeps only
    # `n_pairs * VG_ANNOTATION_RATE` relations, because in a VG photograph an annotator labels
    # about a tenth of the available pairs -- a salience model, and the right one for a room.  In
    # a staged scene with three objects that budget rounds down to ONE relation total, so the
    # multi-label pass returned exactly what the single-label pass did and the defence it exists to
    # mount was silently empty.  Salience is not the question here; truth is.
    return {"gt": keep(vg_calibrated_relations(shim, width, height, exclude_pairs=exclude)),
            "also_true": keep(vg_calibrated_relations(shim, width, height, exclude_pairs=exclude,
                                                      multi_label=True, rate=None))}


def probe_ladder(controller, centre_xz, radius: float, azimuths: Sequence[float],
                 placed, aim_y: float, mid_xz, classes, size, min_visible: float,
                 min_margin: float):
    """Is this separation usable?  Four conditions, all of them measured, none of them assumed.

    1. NEITHER OBJECT HIDES THE OTHER.  The figure's whole premise is that everything is visible,
       so `min_visible` is checked against each object's count with its partner lifted out.
    2. BOTH ARE BIG ENOUGH FOR THE GROUND TRUTH TO SEE THEM.  `vg_gt.candidates` drops anything
       whose box is smaller than `MIN_LINEAR_EXTENT` of the frame, because VG annotators did not
       label objects that small.  A Cup_5 at 1.3 m measures 0.054 against a floor of 0.06 and is
       dropped -- so the first version of this generator staged a perfectly clean scene about
       which the project's own convention had nothing whatever to say.
    3. THE DEPTH ORDERING HAS ROOM TO SPARE.  `vg_gt` derives `behind` from a bare comparison of
       camera distances, so it will happily call one object behind another that is 14 mm further
       away -- measured, on `bottle_behind_laptop` at -30 degrees.  A relation that true by 14 mm
       is a relation no reader will see in the picture, and a figure asking why the model does not
       see it either has answered its own question.  `min_margin` is the answer to "how far behind
       is behind".
    4. THE GROUND TRUTH IS THE SAME AT EVERY AZIMUTH TESTED.  This is the condition the figure
       rests on and the only one that cannot be repaired afterwards: if `vg_calibrated_relations`
       calls the pair `behind` at one end of the arc and `near` at the other, then the relation
       really did change and any instability the model shows is not the model's fault.

    Tested at `LADDER_PROBES` fractions of the span rather than every frame -- the full arc is
    re-checked once a separation is accepted, and running it here would triple the cost of a
    search that mostly fails on its first probe.
    """
    from vg.vg_gt import MIN_LINEAR_EXTENT

    span = max(abs(azimuths[0]), abs(azimuths[-1]))
    floor = math.sqrt(size[0] * size[1]) * MIN_LINEAR_EXTENT
    said = set()
    for fraction in LADDER_PROBES:
        azimuth = fraction * span
        event, cam, yaw, horizon = look(controller, centre_xz, radius, azimuth,
                                        (mid_xz[0], aim_y, mid_xz[1]))
        seen = measure(controller, event, cam, yaw, horizon, placed)
        for name in ("back", "front"):
            if seen[name]["clear_px"] < MIN_PIXELS:
                return {"ok": False, "why": f"{name} only {seen[name]['clear_px']} px "
                                            f"at az {azimuth:+.0f}"}
            if seen[name]["occlusion"] > 1.0 - min_visible:
                return {"ok": False, "why": f"{name} {seen[name]['occlusion']:.0%} hidden "
                                            f"at az {azimuth:+.0f}"}
            box = seen[name]["box"]
            extent = math.sqrt((box[2] - box[0]) * (box[3] - box[1])) if box else 0.0
            if extent < floor:
                return {"ok": False, "why": f"{name} too small for the ground truth to see "
                                            f"({extent / math.sqrt(size[0] * size[1]):.3f} "
                                            f"< {MIN_LINEAR_EXTENT}) at az {azimuth:+.0f}"}
        margin = seen["back"]["distance"] - seen["front"]["distance"]
        if margin < min_margin:
            return {"ok": False, "why": f"depth margin only {margin:+.3f} m at az {azimuth:+.0f}"}
        # The WHOLE triplet, not just the predicate.  `vg_gt` makes the SMALLER box the subject,
        # so a large box staged behind a small bottle comes back as `bottle in front of box` --
        # the same fact, named from the other end.  Recording only the word would have this file
        # write `box in front of bottle` into the ground truth, which is false.
        pair = {(r["subject"], r["predicate"], r["object"])
                for r in derived_relations(seen, classes, size[0], size[1])["gt"]
                if {r["subject"], r["object"]} == {classes["back"], classes["front"]}}
        if not pair:
            return {"ok": False, "why": f"vg_gt derives no relation for the pair at "
                                        f"az {azimuth:+.0f}"}
        said |= pair
        if len(said) > 1:
            return {"ok": False, "why": "vg_gt changes its mind over the arc: "
                                        + "; ".join(" ".join(t) for t in sorted(said))}
    triplet = sorted(said)[0]
    return {"ok": True, "why": f"clear, and vg_gt says `{' '.join(triplet)}` at every probe",
            "triplet": list(triplet)}


def one_case(controller, index: int, back_class: str, back_asset: str,
             front_class: str, front_asset: str, radius: float,
             azimuths: Sequence[float], min_visible: float, min_margin: float,
             size: Tuple[int, int], out_dir: str) -> Optional[Dict[str, Any]]:
    """Stage one pair, find a separation at which neither hides the other, fly the arc."""
    case_id = f"{index:02d}_{back_class}_behind_{front_class}"
    print(f"[{case_id}] back={back_asset} front={front_asset}", flush=True)

    controller.reset()
    centre = ROOM / 2.0
    table = spawn(controller, TABLE, "table", centre, 0.0, centre + 0.6)
    top = surface_top(table)
    box = table["axisAlignedBoundingBox"]
    edge = box["center"]["z"] - box["size"]["z"] / 2.0
    mid_x, mid_z = centre, edge + INSET
    centre_xz = np.array([mid_x, mid_z])

    back_yaw = occluder_yaw(controller, back_asset, back_class)
    front_yaw = occluder_yaw(controller, front_asset, front_class)

    chosen: Optional[Dict[str, Any]] = None
    for lateral in LATERAL_LADDER:
        # Deeper object to -x, nearer object to +x.  The sense is fixed rather than drawn: it
        # decides which half of the arc squeezes the pair together in the image, and a strip whose
        # geometry mirrors between cases cannot be read as a series.
        gap = gap_for(lateral, max(abs(azimuths[0]), abs(azimuths[-1])), min_margin)
        spots = {"back": (mid_x - lateral / 2.0, mid_z + gap / 2.0, back_yaw),
                 "front": (mid_x + lateral / 2.0, mid_z - gap / 2.0, front_yaw)}
        if not all(on_table(table, x, z) for x, z, _ in spots.values()):
            print(f"    lateral {lateral:.2f}: off the table")
            continue
        placed, entries = {}, {}
        for name, asset in (("back", back_asset), ("front", front_asset)):
            x, z, yaw = spots[name]
            entries[name], position = place_on(controller, asset, name, x, top, z, yaw)
            # Position AND yaw, together: every restore in this file goes through this dict, and
            # keeping the two apart is how the half turn got lost the first time.
            placed[name] = {"position": position, "yaw": yaw}
        aim_y = float(np.mean([entries[n]["axisAlignedBoundingBox"]["center"]["y"]
                               for n in ("back", "front")]))

        verdict = probe_ladder(controller, centre_xz, radius, azimuths, placed, aim_y,
                               (mid_x, mid_z), {"back": back_class, "front": front_class,
                                                "table": "table"}, size, min_visible, min_margin)
        print(f"    lateral {lateral:.2f}: {verdict['why']}", flush=True)
        if verdict["ok"]:
            chosen = {"lateral": lateral, "gap": gap, "placed": placed, "aim_y": aim_y,
                      "probe": verdict}
            break

    if chosen is None:
        print("    - no separation kept both objects clear")
        return None

    placed, aim_y = chosen["placed"], chosen["aim_y"]
    frames: List[Dict[str, Any]] = []
    os.makedirs(os.path.join(out_dir, case_id), exist_ok=True)
    for i, azimuth in enumerate(azimuths):
        event, cam, yaw, horizon = look(controller, centre_xz, radius, azimuth,
                                        (mid_x, aim_y, mid_z))
        name = f"frame_{i:02d}.png"
        Image.fromarray(event.frame).save(os.path.join(out_dir, case_id, name))
        seen = measure(controller, event, cam, yaw, horizon, placed)
        back_d, front_d = seen["back"]["distance"], seen["front"]["distance"]
        frames.append({
            "index": i, "image": name, "azimuth_deg": round(float(azimuth), 2),
            "camera": {"x": round(cam[0], 4), "z": round(cam[1], 4),
                       "yaw": round(yaw, 2), "horizon": round(horizon, 2)},
            "objects": seen,
            # The ground truth this frame carries, stated as the geometry rather than as a word:
            # `vg_gt.derive` calls the deeper of two objects `behind` the nearer one, so a positive
            # margin at every azimuth IS the claim that the relation never flipped over the arc.
            "depth_margin_m": (round(back_d - front_d, 4)
                               if back_d is not None and front_d is not None else None),
            **derived_relations(seen, {"back": back_class, "front": front_class,
                                       "table": "table"}, size[0], size[1]),
        })
        said = ", ".join(f"{r['subject']} {r['predicate']} {r['object']}"
                         for r in frames[-1]["gt"]) or "-"
        print(f"  az={azimuth:+6.1f}  back {seen['back']['occlusion']:.1%} hidden / "
              f"front {seen['front']['occlusion']:.1%}  margin "
              f"{frames[-1]['depth_margin_m']:+.3f}  vg_gt: {said}", flush=True)

    record = {
        "case_id": case_id,
        "back": {"class": back_class, "asset": back_asset, "yaw": back_yaw,
                 "position": placed["back"]["position"]},
        "front": {"class": front_class, "asset": front_asset, "yaw": front_yaw,
                  "position": placed["front"]["position"]},
        "table": {"asset": TABLE, "x": centre, "z": centre + 0.6, "top": top},
        "lateral_m": chosen["lateral"], "gap_m": round(chosen["gap"], 4),
        "radius_m": radius, "standing": STANDING,
        "min_visible": min_visible, "min_margin_m": min_margin,
        # The three relations the scene asserts, all of them constant over the arc.  Written down
        # so the figure does not have to reconstruct them from the geometry.  The depth one carries
        # the word `vg_calibrated_relations` actually emitted rather than the word this file
        # staged for: the two agree here by construction (`probe_ladder` refuses the case
        # otherwise) and saying so twice differently is how they would drift apart.
        "ground_truth": [
            {"subject": chosen["probe"]["triplet"][0],
             "predicate": chosen["probe"]["triplet"][1],
             "object": chosen["probe"]["triplet"][2],
             "roles": ["back", "front"],
             "basis": "vg_gt.vg_calibrated_relations, identical at every frame"},
            {"subject": back_class, "predicate": "on", "object": "table",
             "roles": ["back", "table"], "basis": "placed on the tabletop"},
            {"subject": front_class, "predicate": "on", "object": "table",
             "roles": ["front", "table"], "basis": "placed on the tabletop"},
        ],
        "frames": frames,
    }
    # THE LADDER TESTED FIVE AZIMUTHS; THE ARC HAS THIRTEEN.  Re-checking here is not belt and
    # braces: the probes are the ends, the middle and the quarters, and a ground truth that
    # wobbles between two of them would ship unnoticed.  A case that fails is written out anyway
    # with the disagreement recorded, because a silently dropped case is indistinguishable from
    # one that was never staged.
    over_arc = {(r["subject"], r["predicate"], r["object"]) for f in frames for r in f["gt"]
                if {r["subject"], r["object"]} == {back_class, front_class}}
    record["vg_gt_over_arc"] = sorted(" ".join(t) for t in over_arc)
    record["usable"] = over_arc == {tuple(chosen["probe"]["triplet"])}
    if not record["usable"]:
        print(f"    ! ground truth is not constant over the full arc: "
              f"{record['vg_gt_over_arc']}")

    with open(os.path.join(out_dir, case_id, "sweep.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    return record


def pairings(catalogues, rng: random.Random, wanted: Optional[Sequence[str]], n: int):
    """(back_class, back_asset, front_class, front_asset) tuples to stage.

    Explicit `--pairs back:front` wins.  Otherwise walk the class product in a fixed order rather
    than drawing at random: a sweep whose class pairs change between runs cannot be compared with
    the one before it, and the assets inside a class are drawn instead, which is where the variety
    that matters lives.
    """
    catalogue_back, catalogue_front = catalogues
    if wanted:
        combos = []
        for spec in wanted:
            back, _, front = spec.partition(":")
            if back not in catalogue_back or front not in catalogue_front:
                raise SystemExit(f"--pairs {spec}: no assets for "
                                 f"{back if back not in catalogue_back else front}")
            combos.append((back, front))
    else:
        # ROUND ROBIN OVER THE BACK CLASS, not the raw product.  `itertools.product` walks
        # `bottle x everything` first, and with `--n 12` and a staging failure rate around half,
        # the twelve attempts never reached the second back class at all: three of the five cases
        # that survived were a bottle.  Interleaving spends the same budget across the catalogue.
        by_back = {b: [f for f in sorted(catalogue_front) if f != b]
                   for b in sorted(catalogue_back)}
        combos = [(b, fronts[i]) for i in range(max(map(len, by_back.values())))
                  for b, fronts in by_back.items() if i < len(fronts)]
    for index, (back, front) in enumerate(combos[:n] if not wanted else combos):
        yield index, back, rng.choice(catalogue_back[back]), front, rng.choice(catalogue_front[front])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=12, help="how many cases to stage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", nargs="+", default=None, metavar="BACK:FRONT",
                    help="stage exactly these class pairs instead of walking the product")
    ap.add_argument("--span", type=float, default=30.0,
                    help="the arc runs from -span to +span degrees.  30 is deliberate: it is what "
                         "fuse_live.py orbits, and it is short enough that the depth ordering "
                         "cannot genuinely invert")
    ap.add_argument("--frames", type=int, default=13, help="frames on the arc, odd so one sits at 0")
    ap.add_argument("--radius", type=float, default=1.1,
                    help="camera distance from the pair's midpoint, m.  Further out than the "
                         "sideview sweep's 0.85 so the table reads as a table, but not so far "
                         "that the deeper object falls under the ground truth's own size floor: "
                         "a cup at 1.3 m measures 0.054 against `vg_gt.MIN_LINEAR_EXTENT` 0.06")
    ap.add_argument("--min_visible", type=float, default=0.98,
                    help="each object must show at least this fraction of its unobstructed pixel "
                         "count at every azimuth, or the case is dropped")
    ap.add_argument("--min_margin", type=float, default=0.06,
                    help="the deeper object must be at least this much further from the camera at "
                         "every azimuth.  Guards against a `behind` that is true by millimetres "
                         "and invisible in the picture -- see probe_ladder")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--classes", nargs="+",
                    default=sorted(set(TARGET_CLASSES) | set(OCCLUDER_CLASSES)),
                    help="classes either object may be drawn from")
    ap.add_argument("--out", default="datasets/sgg/pairview_demo")
    args = ap.parse_args(argv)

    if args.frames % 2 == 0:
        raise SystemExit("--frames must be odd so one frame lands on azimuth 0")
    azimuths = np.linspace(-args.span, args.span, args.frames)

    rng = random.Random(args.seed)
    controller = open_room(args.width, args.height, args.fov)
    # One size band for both roles.  The sideview sweep needs its occluder to be the taller thing;
    # here neither object hides anything, so the only requirement is that both are tabletop-sized.
    # One size band for both roles, and its FLOOR is raised above `TARGET_SIZE`'s 0.10.  Neither
    # object hides anything here, so the sideview sweep's reason for two bands is gone; what
    # replaces it is the ground truth's size gate, which a 0.10 m object at this radius fails.
    band = (0.14, max(TARGET_SIZE[1], OCCLUDER_SIZE[1]))
    shared = catalogue(controller, args.classes, band)
    print(f"classes {[f'{k}:{len(v)}' for k, v in sorted(shared.items())]}")

    os.makedirs(args.out, exist_ok=True)
    built = []
    for index, back, back_asset, front, front_asset in pairings((shared, shared), rng,
                                                                args.pairs, args.n):
        record = one_case(controller, index, back, back_asset, front, front_asset,
                          args.radius, azimuths, args.min_visible, args.min_margin,
                          (args.width, args.height), args.out)
        if record is not None:
            worst = max(max(f["objects"][r]["occlusion"] for r in ("back", "front"))
                        for f in record["frames"])
            built.append({"case_id": record["case_id"], "lateral_m": record["lateral_m"],
                          "usable": record["usable"], "worst_occlusion": round(worst, 4),
                          "min_depth_margin_m": min(f["depth_margin_m"] for f in record["frames"]
                                                    if f["depth_margin_m"] is not None),
                          # What the paper's own convention calls this pair over the arc.  One
                          # entry means the ground truth held; more than one means the case is
                          # not usable as a stability figure and has to be seen, not hidden.
                          "vg_gt": record["vg_gt_over_arc"]})
    controller.stop()

    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump({"span_deg": args.span, "frames": args.frames, "radius_m": args.radius,
                   "gap_min_m": GAP_MIN, "standing": STANDING, "cases": built}, fh, indent=1)
    print(f"\n{len(built)} staged -> {args.out}")
    for case in built:
        print(f"  {'ok ' if case['usable'] else 'BAD'} {case['case_id']:<34} "
              f"lateral {case['lateral_m']:.2f}  "
              f"worst occlusion {case['worst_occlusion']:.1%}  "
              f"min depth margin {case['min_depth_margin_m']:+.3f} m  "
              f"vg_gt {' | '.join(case['vg_gt']) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
