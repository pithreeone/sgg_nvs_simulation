"""
probe_class.py -- which VG150 class should the SUBJECT be?

`probe_argmax_cases_easy.json` says the landmark endpoint is solved and the
subject endpoint is the whole bottleneck:

    K=1   landmark  prob rank 1 in 40/40      subject  12/40      joint 12/40

so the joint number IS the subject number.  The landmark was fixed the same way
-- lamp/clock survived argmax 0/18 and 0/5, box/laptop 17/17 and 23/23 -- and
that single swap moved the correct pair's median rank from 21.5 to 2.

The paired twin control in `probe_occl.json` says the remaining subject loss is
mostly NOT occlusion:

    both copies named 17,  only the unoccluded twin 7,  only the target 4,
    NEITHER 12   (n = 40)

`neither` is the class failing at this size in a clean view, and it is larger
than the occlusion term.  A viewpoint cannot fix it, so it is dataset noise
sitting on top of the effect the NVS sweep is supposed to demonstrate.  Removing
it makes occlusion the only remaining failure mode -- which is the experiment.

This probe measures the CEILING for each candidate class: one object, staged at
the dataset's own geometry (gap 0.35 m behind where the landmark would be,
camera 1.00 m back), WITH NO OCCLUDER AT ALL.  Whatever a class scores here is
the best any decision rule could do on it, because nothing is hidden.

    python probe_class.py --assets 5 --out nvs_pilot/probe_class.json
"""

from __future__ import annotations

# `python archive/<script>.py` puts archive/ on sys.path, not the repo root.
# These were moved here without it, so they could not import the generators at
# all; same shim as `gen/` and `build_robotic_task/`.
import os as _os
import sys as _sys
if __package__ in (None, ""):
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import collections
import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from build_robotic_task.build_tabletop import (NEAR_EDGE_INSET, TABLE, TABLE_MARGIN, TARGET_SIZE,
                            catalogue, on_table)
from robot.proc_scene import (ROOM, aim_at, open_room, place_on, spawn,
                              surface_top, visible_box, visible_pixels)

#: Every VG150 class with an asset in `TARGET_SIZE` narrow enough to be hidden
#: by a 0.30-0.40 m landmark.  Measured against the asset database, not guessed.
#: `clock` is carried despite its landmark-side failure so the probe reproduces
#: that result and can be trusted on the classes it has no prior for.
CANDIDATES = ("cup", "bowl", "bottle", "vase", "fruit", "vegetable", "food",
              "clock", "paper", "box", "plant")

#: The staging the current dataset uses, so the ceiling is measured at the size
#: and depth the cases actually put the subject at -- a class that is nameable
#: at 0.5 m and not at 1.0 m would otherwise look usable.
GAP = 0.35
STANDOFF = 1.00
IOU_HIT = 0.5


def name_scores(egtr, frame, truth: Sequence[float],
                vg_class: str) -> Optional[Dict[str, Any]]:
    """How well EGTR names the one staged object, by both selection rules.

    `prob_rank` is what `conditioned` uses -- rank among ALL 200 queries by
    p(class), top-K kept -- so it is the number that predicts the pipeline.
    `argmax_ok` is the stricter rule, reported because the two disagreed by a
    wide margin on the landmark classes.
    """
    from robot.sgg_live import raw_predict
    from robot.task_find import iou

    raw = raw_predict(egtr, frame)
    probs = raw["probs_softmax"].cpu().numpy()
    boxes = raw["boxes"].numpy()
    index = {v: k - 1 for k, v in egtr["obj_names"].items()}.get(vg_class)
    if index is None:
        return None
    covering = {q for q in range(len(boxes))
                if iou(boxes[q].tolist(), truth) >= IOU_HIT}
    if not covering:
        return {"prob_rank": None, "argmax_ok": False, "best_p": 0.0,
                "covering": 0, "said": None}
    order = list(np.argsort(-probs[:, index]))
    argmax = probs.argmax(-1)
    names = {k - 1: v for k, v in egtr["obj_names"].items()}
    best = max(covering, key=lambda q: float(probs[q].max()))
    return {
        "prob_rank": next((r for r, q in enumerate(order, 1) if q in covering),
                          None),
        "argmax_ok": any(argmax[q] == index for q in covering),
        "best_p": max(float(probs[q][index]) for q in covering),
        "covering": len(covering),
        # What the model calls it instead.  A class that loses to a consistent
        # alternative is a relabelling problem; one that loses to noise is not.
        "said": names.get(int(argmax[best])),
    }


def one(controller, egtr, asset: str, vg_class: str) -> Optional[Dict[str, Any]]:
    """Stage one unoccluded object at the dataset's geometry and name it."""
    controller.reset()
    centre = ROOM / 2.0
    table = spawn(controller, TABLE, "table", centre, 0.0, centre + 0.6)
    top = surface_top(table)
    box = table["axisAlignedBoundingBox"]
    edge = box["center"]["z"] - box["size"]["z"] / 2.0
    tz = edge + max(sum(NEAR_EDGE_INSET) / 2.0, GAP + TABLE_MARGIN + 0.05)
    if not on_table(table, centre, tz):
        return None
    place_on(controller, asset, "target", centre, top, tz)
    event, _ = aim_at(controller, centre, tz - STANDOFF, 0.0,
                      (centre, top + 0.08, tz))
    px = visible_pixels(event, "target")
    if px < 300:
        return None
    truth = visible_box(event, "target")
    if truth is None:
        return None
    got = name_scores(egtr, event.frame, truth, vg_class)
    return got and {**got, "asset": asset, "cls": vg_class, "px": px}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--assets", type=int, default=5,
                    help="assets per class.  Classes with fewer are staged once "
                         "each; the geometry is deterministic so repeats of the "
                         "same asset would inflate n without adding evidence.")
    ap.add_argument("--classes", nargs="+", default=list(CANDIDATES))
    ap.add_argument("--width-cap", type=float, default=0.22, metavar="M",
                    help="horizontal extent cap.  The default is the width a "
                         "0.30-0.40 m landmark can actually hide; the GENERATOR "
                         "caps the target side at 0.30, so assets between the "
                         "two are in the dataset but not in the default sweep.")
    ap.add_argument("--band", type=float, nargs=2, default=list(TARGET_SIZE),
                    metavar=("MIN", "MAX"),
                    help="height band.  Pass the occluder band to measure the "
                         "landmark side with the same code.")
    ap.add_argument("--only", nargs="+", metavar="ASSET",
                    help="stage exactly these asset ids, whatever the size "
                         "filters say.  Used to measure the ceiling of the "
                         "assets a BUILT case list actually put on the table, "
                         "which is the number that explains its results.")
    ap.add_argument("--out", help="write the per-object rows as JSON")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    args = ap.parse_args(argv)

    from robot.sgg_live import load_egtr

    egtr = load_egtr()
    controller = open_room(args.width, args.height, args.fov)
    rows: List[Dict[str, Any]] = []
    try:
        # The same width cap the target side of the generator uses, so a class
        # that only qualifies via a wide asset is not credited here.
        pool = catalogue(controller, args.classes, tuple(args.band),
                         args.width_cap)
        if args.only:
            keep = set(args.only)
            pool = {c: [a for a in v if a in keep] for c, v in pool.items()}
            pool = {c: v for c, v in pool.items() if v}
            missing = keep - {a for v in pool.values() for a in v}
            if missing:
                print(f"  ! outside --band/--width-cap, not staged: "
                      f"{sorted(missing)}", flush=True)
        print(f"  {sum(len(v) for v in pool.values())} assets over "
              f"{len(pool)} classes\n", flush=True)
        for vg_class in sorted(pool):
            for asset in sorted(pool[vg_class])[:args.assets]:
                try:
                    got = one(controller, egtr, asset, vg_class)
                except Exception as error:                      # noqa: BLE001
                    print(f"  ! {type(error).__name__}: {error}", flush=True)
                    continue
                if got is None:
                    continue
                rows.append(got)
                print(f"  {vg_class:10s} {asset:32s} rank "
                      f"{str(got['prob_rank']):<5s} argmax "
                      f"{'Y' if got['argmax_ok'] else '.'} "
                      f"p {got['best_p']:.3f}  px {got['px']:>6d}  "
                      f"said {got['said']}", flush=True)
    finally:
        controller.stop()

    if not rows:
        print("nothing staged")
        return 1
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"gap": GAP, "standoff": STANDOFF, "rows": rows}, handle,
                      indent=1)
        print(f"\n  wrote {args.out}")

    by: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by[row["cls"]].append(row)

    def at(got: Sequence[Dict[str, Any]], k: int) -> int:
        return sum(1 for g in got
                   if g["prob_rank"] is not None and g["prob_rank"] <= k)

    print(f"\n  UNOCCLUDED CEILING at gap {GAP} m, standoff {STANDOFF} m\n")
    print(f"  {'class':10} {'n':>3} {'rank 1':>9} {'rank<=5':>9} "
          f"{'rank<=10':>9} {'argmax':>8} {'p med':>7} {'px med':>7}  "
          f"usually called")
    for cls, got in sorted(by.items(), key=lambda kv: -at(kv[1], 1) / len(kv[1])):
        m = len(got)
        said = collections.Counter(g["said"] for g in got).most_common(2)
        print(f"  {cls:10} {m:>3} {at(got,1):>5}/{m:<3} {at(got,5):>5}/{m:<3} "
              f"{at(got,10):>5}/{m:<3} "
              f"{sum(1 for g in got if g['argmax_ok']):>4}/{m:<3} "
              f"{np.median([g['best_p'] for g in got]):>7.3f} "
              f"{np.median([g['px'] for g in got]):>7.0f}  "
              + ", ".join(f"{k} {v}" for k, v in said))

    print("\n  `rank 1` is the ceiling on the current pipeline: `conditioned` "
          "ranks all\n  200 queries by p(class) and the joint top-1 cannot beat "
          "its weaker end.\n  The classes now in use are cup, bowl and box; "
          "their joint rank-1 is 12/40.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
