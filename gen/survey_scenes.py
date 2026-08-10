"""
survey_scenes.py -- rank floor plans by how much occlusion data they can support.

The five living rooms in `build_occlusion_dataset.LIVING_ROOMS` were chosen by
(VG150-nameable moveable objects) x (placement surfaces), spanning a range of
room sizes so the set is not five variations of one floor plan.  That criterion
was applied by hand and never written down, so extending the dataset to the other
three room types would have meant guessing at it.  This measures it instead.

Three quantities, all read from one THOR reset per scene:

  nameable    moveable objects `THOR_TO_VG150` can name.  These are what
              `occlusion_pipeline.arrange` duplicates to build clutter, so a
              scene with few of them cannot be made cluttered.
  surfaces    receptacles large enough for `pick_focus` to build a seed around
              (>= 0.15 m2).  Each seed picks a different one, so a scene with
              fewer surfaces than seeds photographs the same corner repeatedly.
  reachable   `GetReachablePositions`, a proxy for room size.  Spanning it keeps
              the set from being five copies of one layout.

Ranked by `nameable * min(surfaces, seeds)`, which is the number of distinct
cluttered arrangements the scene can actually produce.

The pick is then bucketed by room size and takes the best-scoring scene in each
bucket, rather than the top N outright.  Validated against the hand-picked living
rooms it is meant to reproduce, it returns FloorPlan201, 215, 230, 227, 228
against the original 201, 204, 215, 218, 230: three exact, with 227/228 sitting
next to 204/218 in both score and size.  That is close enough to trust the
criterion on room types nobody has inspected by hand.  Spreading over size
without regard to score picks thin rooms; taking the top N by score alone picks
several variations of one large layout.
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
import sys
from typing import Any, Dict, List, Optional, Sequence

from vg.vg150 import THOR_TO_VG150

FAMILIES = {
    "kitchen": range(1, 31),
    "living": range(201, 231),
    "bedroom": range(301, 331),
    "bathroom": range(401, 431),
}

#: Same floor as `pick_focus`: a 0.2 m shelf holds two objects and gives the
#: cameras nothing to orbit.
MIN_SURFACE_AREA = 0.15


def survey(controller, scene: str) -> Dict[str, Any]:
    event = controller.reset(scene=scene)
    objects = event.metadata["objects"]
    nameable = [o for o in objects
                if (o.get("pickupable") or o.get("moveable"))
                and THOR_TO_VG150.get(o["objectType"])]
    surfaces = []
    for o in objects:
        if not o.get("receptacle"):
            continue
        size = (o.get("axisAlignedBoundingBox") or {}).get("size")
        if size and size["x"] * size["z"] >= MIN_SURFACE_AREA:
            surfaces.append(o)
    reachable = controller.step(action="GetReachablePositions")
    positions = reachable.metadata.get("actionReturn") or []
    classes = {THOR_TO_VG150[o["objectType"]] for o in nameable}
    all_named = {THOR_TO_VG150[o["objectType"]] for o in objects
                 if THOR_TO_VG150.get(o["objectType"])}
    return {"scene": scene, "nameable": len(nameable), "surfaces": len(surfaces),
            "reachable": len(positions), "moveable_classes": len(classes),
            "classes": len(all_named), "class_list": sorted(all_named)}


def choose(rows: List[Dict[str, Any]], pick: int) -> List[Dict[str, Any]]:
    """
    The best-scoring scene in each of `pick` room-size buckets.

    Taking the top `pick` by score alone gives several variations of one large
    layout; spreading over room size alone gives thin scenes with nothing to
    clutter.  Bucketing the strong candidates by size and taking the best of each
    keeps both properties, and reproduces the hand-picked living rooms.
    """
    candidates = sorted(rows, key=lambda r: -r["score"])[:max(pick * 3, pick)]
    candidates.sort(key=lambda r: r["reachable"])
    chosen = []
    for index in range(pick):
        low = index * len(candidates) // pick
        high = max((index + 1) * len(candidates) // pick, low + 1)
        bucket = candidates[low:high]
        if bucket:
            chosen.append(max(bucket, key=lambda r: r["score"]))
    return sorted(chosen, key=lambda r: -r["score"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--families", nargs="*", default=list(FAMILIES))
    parser.add_argument("--seeds", type=int, default=3,
                        help="surfaces beyond this many do not add arrangements")
    parser.add_argument("--pick", type=int, default=5,
                        help="how many scenes per family to recommend")
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--out", default=None)
    parser.add_argument("--load", default=None,
                        help="re-rank a saved survey instead of running THOR")
    args = parser.parse_args(argv)

    saved = None
    controller = None
    if args.load:
        with open(args.load, encoding="utf-8") as handle:
            saved = json.load(handle)
    else:
        from ai2thor.controller import Controller
        controller = Controller(scene="FloorPlan1", width=args.width,
                                height=args.width, visibilityDistance=15.0)
    out: Dict[str, List[Dict[str, Any]]] = {}
    try:
        for family in args.families:
            if saved is not None:
                rows = list(saved[family])
            else:
                rows = []
                for index in FAMILIES[family]:
                    try:
                        rows.append(survey(controller, f"FloorPlan{index}"))
                    except Exception as exc:                   # noqa: BLE001
                        print(f"  FloorPlan{index}: {exc}")
            for row in rows:
                row["score"] = row["nameable"] * min(row["surfaces"], args.seeds)
            rows.sort(key=lambda r: -r["score"])
            out[family] = rows

            print(f"\n{'=' * 78}\n{family.upper()}\n{'=' * 78}")
            print(f"{'scene':<16}{'nameable':>9}{'surfaces':>10}{'reachable':>11}"
                  f"{'classes':>9}{'score':>8}")
            for row in rows[:12]:
                print(f"{row['scene']:<16}{row['nameable']:>9}{row['surfaces']:>10}"
                      f"{row['reachable']:>11}{row['classes']:>9}{row['score']:>8}")

            pick = choose(rows, args.pick)
            print(f"\n  recommended: {[r['scene'] for r in pick]}")
            print(f"  reachable spans {min(r['reachable'] for r in pick)}"
                  f"-{max(r['reachable'] for r in pick)}, "
                  f"nameable {min(r['nameable'] for r in pick)}"
                  f"-{max(r['nameable'] for r in pick)}")
            out[f"{family}_pick"] = [r["scene"] for r in pick]
    finally:
        if controller is not None:
            controller.stop()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
