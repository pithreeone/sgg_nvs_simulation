"""
probe_behind.py -- which class pairs will EGTR actually call `behind`?

The tabletop cases are built so that exactly one copy of the target is behind
the landmark, and the model is then asked which one.  It mostly declines to use
the word: over 40 cases the true pair's argmax predicate was `on` 13 times, `in`
11, `near` 5, and `behind` never, in any vocabulary.  Broken down by class pair
the six successes had a bottle or a lamp as the landmark and a small round thing
as the subject, while `bowl behind the box` failed 4 times out of 4 -- which
reads like a language prior, not a perception failure: two containers in Visual
Genome are `in`/`on`/`near` each other, and things are `behind` bottles and
lamps.

Rather than infer that from 40 cases with one to four per pair, measure it
directly.  One staged scene per (subject, landmark) pair, the same geometry
every time, and the question is only where `behind` ranks for the pair that
really is in that relation.

This is a screen, not an experiment: no distractor, no occlusion band, no
fusion.  It says which instructions are worth building a dataset out of.

    python probe_behind.py --assets 3
"""

from __future__ import annotations

# `python archive/<script>.py` puts archive/ on sys.path, not the repo root.
# These were moved here without it, so they could not import the generators at
# all; same shim as `build/sgg/` and `build/robot/`.
import os as _os
import sys as _sys
if __package__ in (None, ""):
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import collections
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from build.robot.build_tabletop import (NEAR_EDGE_INSET, OCCLUDER_CLASSES, OCCLUDER_SIZE,
                            TABLE, TARGET_CLASSES, TARGET_SIZE, catalogue,
                            on_table, stand_back)
from robot.world.proc_scene import (ROOM, aim_at, open_room, place_on, spawn,
                              surface_top, visible_box, visible_pixels)

#: The seven VG150 spatial predicates, `in` dropped: nothing on a table is
#: inside anything else here, so leaving it in only lets it absorb probability
#: that belongs to a relation the scene does support.
SPATIAL = ("on", "near", "above", "under", "behind", "in front of")

#: How far in front of the target the landmark sits, metres.  Fixed, so the
#: comparison across pairs is about the OBJECTS rather than about geometry.
GAP = 0.20

#: The staged pair has to actually be occluded, or `behind` is not the relation
#: the picture shows and a low rank says nothing.
OCCLUSION = (0.15, 0.75)


def rank_of_behind(egtr, frame, gt_target, gt_landmark) -> Optional[Dict[str, Any]]:
    """Where `behind` ranks among the spatial predicates for the true pair."""
    from probe_corr import iou
    from robot.sgg_live import raw_predict

    if gt_target is None or gt_landmark is None:
        return None
    raw = raw_predict(egtr, frame)
    boxes = raw["boxes"].numpy()
    rel = raw["rel"].detach().cpu().numpy()
    slots = []
    for gt in (gt_target, gt_landmark):
        ious = np.array([iou(boxes[q].tolist(), gt) for q in range(len(boxes))])
        if ious.max() < 0.5:
            return None
        slots.append(int(ious.argmax()))

    names = egtr["rel_names"]
    keep = np.array([i for i, n in enumerate(names) if n in SPATIAL])
    row = rel[slots[0], slots[1]]
    order = keep[np.argsort(-row[keep])]
    return {"rank": int(np.where(order == names.index("behind"))[0][0]) + 1,
            "argmax": names[int(order[0])]}


def sweep_ranks(controller, egtr, rc, views: int) -> List[Dict[str, Any]]:
    """
    The same measurement from every swept view.

    This is the question the reference frame cannot answer.  R relabels a pair
    by consensus across views, so what matters is not whether the robot's own
    frame says `behind` -- it does not -- but whether ANY viewpoint does, and
    how many.  A predicate that never wins from any angle is one no amount of
    multi-view agreement can recover.
    """
    from robot.world.nvs_lemniscate import camera_for, lemniscate, orbit_centre
    from robot.world.proc_scene import look_from
    from probe_corr import SKIP_VIEWS, seg_box

    camera = rc.camera_xyz.copy()
    centre = orbit_centre(rc)
    corner = min(((0.3, 0.3), (0.3, ROOM - 0.3), (ROOM - 0.3, 0.3),
                  (ROOM - 0.3, ROOM - 0.3)),
                 key=lambda p: -math.dist(p, (camera[0], camera[2])))
    look_from(controller, corner[0], corner[1], 0.0, 0.0, force=True)

    out, first = [], True
    for index, (az, el) in enumerate(lemniscate(views, 30.0, 15.0)):
        if index in SKIP_VIEWS:
            continue
        pose = camera_for(centre, camera, az, el)
        controller.step(
            action="AddThirdPartyCamera" if first else "UpdateThirdPartyCamera",
            position=dict(pose["position"]),
            rotation={"x": pose["pitch"], "y": pose["yaw"], "z": 0.0},
            fieldOfView=60.0,
            **({} if first else {"thirdPartyCameraId": 0}))
        first = False
        event = controller.last_event
        seg = event.third_party_instance_segmentation_frames[0]
        colour = {n: c for c, n in event.color_to_object_id.items()}
        got = rank_of_behind(
            egtr, np.array(event.third_party_camera_frames[0]),
            seg_box(seg, colour["target"]) if "target" in colour else None,
            seg_box(seg, colour["occluder"]) if "occluder" in colour else None)
        if got:
            out.append(got)
    return out


def one(controller, egtr, target_asset: str, occ_asset: str,
        views: int = 0) -> Optional[Dict[str, Any]]:
    """Stage one pair and return where `behind` ranks for it."""
    from robot.world.proc_scene import View

    controller.reset()
    centre = ROOM / 2.0
    table = spawn(controller, TABLE, "table", centre, 0.0, centre + 0.6)
    top = surface_top(table)
    box = table["axisAlignedBoundingBox"]
    edge = box["center"]["z"] - box["size"]["z"] / 2.0

    tz = edge + sum(NEAR_EDGE_INSET) / 2.0
    place_on(controller, target_asset, "target", centre, top, tz)
    z = stand_back(controller, centre, tz)
    if z is None:
        return None
    event, horizon = aim_at(controller, centre, z, 0.0, (centre, top + 0.08, tz))
    clear = visible_pixels(event, "target")
    if clear < 300:
        return None

    if not on_table(table, centre, tz - GAP):
        return None
    place_on(controller, occ_asset, "occluder", centre, top, tz - GAP)
    event = controller.step(action="Pass")
    hidden = 1.0 - visible_pixels(event, "target") / clear
    if not OCCLUSION[0] <= hidden <= OCCLUSION[1]:
        return None

    reference = rank_of_behind(egtr, event.frame,
                               visible_box(event, "target"),
                               visible_box(event, "occluder"))
    if reference is None:
        return None
    out = {"rank": reference["rank"], "argmax": reference["argmax"],
           "hidden": hidden, "views": []}
    if views:
        rc = View(event)
        rc.controller = controller
        out["views"] = sweep_ranks(controller, egtr, rc, views)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--assets", type=int, default=3,
                    help="how many asset combinations to try per class pair")
    ap.add_argument("--views", type=int, default=20,
                    help="lemniscate views to ask the same question from; "
                         "0 measures the reference frame only")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    args = ap.parse_args(argv)

    from robot.sgg_live import load_egtr

    egtr = load_egtr()
    controller = open_room(args.width, args.height, args.fov)
    rows: List[Dict[str, Any]] = []
    try:
        targets = catalogue(controller, TARGET_CLASSES, TARGET_SIZE)
        occluders = catalogue(controller, OCCLUDER_CLASSES, OCCLUDER_SIZE)
        for subject in sorted(targets):
            for landmark in sorted(occluders):
                if subject == landmark:
                    continue
                got = []
                for k in range(args.assets):
                    ta = sorted(targets[subject])[k % len(targets[subject])]
                    oa = sorted(occluders[landmark])[k % len(occluders[landmark])]
                    try:
                        result = one(controller, egtr, ta, oa, args.views)
                    except Exception as error:                  # noqa: BLE001
                        print(f"  ! {type(error).__name__}: {error}", flush=True)
                        result = None
                    if result:
                        got.append(result)
                if not got:
                    continue
                ranks = [g["rank"] for g in got]
                views = [v for g in got for v in g["views"]]
                rows.append({"pair": f"{subject} behind {landmark}",
                             "n": len(got),
                             "first": sum(1 for r in ranks if r == 1),
                             "median": float(np.median(ranks)),
                             "views": len(views),
                             # The question R depends on: does ANY viewpoint put
                             # `behind` first, and how many of them.
                             "view_first": sum(1 for v in views
                                               if v["rank"] == 1),
                             "best": min([v["rank"] for v in views],
                                         default=None),
                             "argmax": collections.Counter(
                                 g["argmax"] for g in got).most_common(1)[0][0]})
                print(f"  {rows[-1]['pair']:30s} n={len(got)} "
                      f"reference rank {ranks}"
                      + (f"   views: {rows[-1]['view_first']}/{len(views)} "
                         f"first, best {rows[-1]['best']}" if views else ""),
                      flush=True)
    finally:
        controller.stop()

    if not rows:
        print("nothing staged")
        return 1
    print(f"\n  {'instruction':30s} {'ref 1st':>8s} {'median':>7s} "
          f"{'views 1st':>10s} {'best':>5s}   usually called")
    for row in sorted(rows, key=lambda r: (-r["view_first"], -r["first"],
                                           r["median"])):
        print(f"  {row['pair']:30s} {row['first']:>3d}/{row['n']:<4d} "
              f"{row['median']:>7.0f} {row['view_first']:>6d}/{row['views']:<3d} "
              f"{str(row['best']):>5s}   {row['argmax']}")
    total = sum(r["views"] for r in rows)
    won = sum(r["view_first"] for r in rows)
    print(f"\n  `behind` is the top spatial predicate in {won} of {total} "
          f"novel views ({won / max(total, 1):.1%}), and in "
          f"{sum(r['first'] for r in rows)} of {sum(r['n'] for r in rows)} "
          f"reference frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
