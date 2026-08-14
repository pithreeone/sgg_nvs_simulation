"""
probe_size.py -- does EGTR fail on these tasks because the target is TINY?

The task sheets showed the graded instance covering a median 0.55% of the frame
-- roughly 50x50 px at 800x600 -- because `find_cases` stands the robot 1.2-2.5 m
off a table and a mug is small at that range.  Every predicate we have tried
returns near-zero recall, and "the predicate is wrong" and "the object is too
small to detect" predict the same near-zero.  This separates them.

For each frozen case, at the reference pose and with the staging replayed:

    share       the graded instance's visible box, as a fraction of the frame
    detected    does ANY of EGTR's queries put its box on that instance
                (IoU >= 0.5) AND call it the right VG150 class
    best_prob   the highest class probability among the queries that overlap it

`detected` is a strictly easier question than the experiment's own: no relation,
no ranking, no second endpoint.  If it fails at small sizes and succeeds at
large ones, the case list is mis-framed rather than the predicate mis-chosen,
and no amount of fusion will help until the robot stands closer.

    python probe_size.py --cases nvs_pilot/cases/cases_easy2.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

IOU_HIT = 0.5


def probe(rc, case: Dict[str, Any], egtr, args) -> Optional[Dict[str, Any]]:
    from eval_nvs_pointer import geometry
    from robot.sgg_live import raw_predict
    from robot.task_find import iou, stage_at
    from vg.vg150 import THOR_TO_VG150

    graded = case["target_name"]
    hidden = case.get("landmark_name") or graded
    names = sorted({o["name"] for o in rc.event.metadata["objects"]
                    if o["objectType"] in THOR_TO_VG150
                    or o.get("moveable") or o.get("pickupable")})
    if stage_at(rc, hidden, case["occluder_name"], case["occluder_position"],
                case.get("scene_poses")) is None:
        return None

    box = geometry(rc, [graded], names, False).get(graded, {}).get("bbox_visible")
    if not box:
        return None
    height, width = rc.event.frame.shape[:2]
    share = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)

    raw = raw_predict(egtr, rc.event.frame)
    classes = {v: k - 1 for k, v in egtr["obj_names"].items()}
    want = classes.get(case["subject_class"])
    probs = raw["probs_softmax"].cpu().numpy()
    boxes = raw["boxes"].numpy()

    overlapping = [q for q in range(len(boxes))
                   if iou(boxes[q].tolist(), box) >= IOU_HIT]
    detected = any(int(probs[q].argmax()) == want for q in overlapping)
    best = max((float(probs[q][want]) for q in overlapping), default=0.0)
    return {"scene": case["scene"], "instruction": case["instruction"],
            "subject_class": case["subject_class"], "share": float(share),
            "queries_on_it": len(overlapping), "detected": bool(detected),
            "best_prob": best}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from robot import drive
    from robot.sgg_live import load_egtr

    cases = json.load(open(args.cases))["cases"]
    if args.n:
        cases = cases[:args.n]
    egtr = load_egtr()

    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        rc = drive.open_scene(case["scene"], args.width, args.height, args.fov,
                              case["start"])
        try:
            row = probe(rc, case, egtr, args)
        except Exception as error:                              # noqa: BLE001
            print(f"    ! {type(error).__name__}: {error}", flush=True)
            row = None
        finally:
            rc.stop()
        if row:
            rows.append(row)
            print(f"[{index}/{len(cases)}] {row['share'] * 100:5.2f}%  "
                  f"{'DET' if row['detected'] else '   '}  "
                  f"p={row['best_prob']:.3f}  {row['instruction']}", flush=True)

    if not rows:
        print("nothing measured")
        return 1

    print(f"\n{len(rows)} cases   detected {sum(r['detected'] for r in rows)}"
          f" ({sum(r['detected'] for r in rows) / len(rows):.0%})")
    print("\n  size band        n   detected   median p(class)")
    bands = [(0.0, 0.005, "< 0.5%"), (0.005, 0.01, "0.5-1%"),
             (0.01, 0.02, "1-2%"), (0.02, 0.05, "2-5%"), (0.05, 1.0, "> 5%")]
    for low, high, label in bands:
        group = [r for r in rows if low <= r["share"] < high]
        if not group:
            continue
        hit = sum(r["detected"] for r in group)
        median = float(np.median([r["best_prob"] for r in group]))
        print(f"  {label:12s} {len(group):5d}   {hit:4d} ({hit / len(group):4.0%})"
              f"      {median:.3f}")

    out = args.out or os.path.splitext(args.cases)[0] + "_size.json"
    with open(out, "w") as fh:
        json.dump({"iou_hit": IOU_HIT, "cases": rows}, fh, indent=1)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
