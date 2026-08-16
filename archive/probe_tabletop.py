"""
probe_tabletop.py -- can EGTR see these objects at all, in the clean scenes?

The same question `probe_size.py` asked of the iTHOR lists, where the answer was
31 of 71.  It is the ceiling on everything downstream: a relation cannot be
fused onto an object the detector never found, and a robot cannot walk to one.

Asked here because the tabletop scenes remove every excuse.  No clutter, no
L-shaped counters, occlusion set to a measured 35%, and a distractor that is a
byte-identical copy of the target -- so a failure now is the detector's, not the
scene's.

Reported per role, because they fail differently:

  target      35% hidden by the landmark; the thing the instruction asks for
  distractor  the same asset, unoccluded; if THIS is missed the case cannot
              test disambiguation at all
  occluder    the landmark, unoccluded and larger

    python probe_tabletop.py --cases nvs_pilot/cases/cases_tabletop.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

IOU_HIT = 0.5


def probe(controller, case: Dict[str, Any], egtr) -> Dict[str, Any]:
    from robot.proc_scene import rebuild, visible_box
    from robot.sgg_live import raw_predict
    from robot.task_find import iou

    event = rebuild(controller, case)
    raw = raw_predict(egtr, event.frame)
    classes = {v: k - 1 for k, v in egtr["obj_names"].items()}
    probs = raw["probs_softmax"].cpu().numpy()
    boxes = raw["boxes"].numpy()

    out: Dict[str, Any] = {"scene": case["scene"],
                           "instruction": case["instruction"],
                           "occlusion": case["staged_occlusion"]}
    for role, want in (("target", case["subject_class"]),
                       ("distract", case["subject_class"]),
                       ("occluder", case["object_class"])):
        truth = visible_box(event, role)
        index = classes.get(want)
        if truth is None or index is None:
            out[role] = {"detected": False, "best_prob": 0.0, "queries": 0}
            continue
        on_it = [q for q in range(len(boxes))
                 if iou(boxes[q].tolist(), truth) >= IOU_HIT]
        out[role] = {
            "detected": any(int(probs[q].argmax()) == index for q in on_it),
            "best_prob": max((float(probs[q][index]) for q in on_it),
                             default=0.0),
            "queries": len(on_it),
            "px": int((event.instance_segmentation_frame is not None)
                      and abs(truth[2] - truth[0]) * abs(truth[3] - truth[1])),
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="nvs_pilot/cases/cases_tabletop.json")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from robot.proc_scene import open_room
    from robot.sgg_live import load_egtr

    cases = json.load(open(args.cases))["cases"]
    egtr = load_egtr()
    controller = open_room(args.width, args.height, args.fov)
    rows: List[Dict[str, Any]] = []
    try:
        for index, case in enumerate(cases, 1):
            row = probe(controller, case, egtr)
            rows.append(row)
            flags = " ".join(f"{r}:{'Y' if row[r]['detected'] else '.'}"
                             f"{row[r]['best_prob']:.2f}"
                             for r in ("target", "distract", "occluder"))
            print(f"[{index}/{len(cases)}] {flags}   {row['instruction']}",
                  flush=True)
    finally:
        controller.stop()

    print()
    for role in ("target", "distract", "occluder"):
        hit = sum(1 for r in rows if r[role]["detected"])
        median = float(np.median([r[role]["best_prob"] for r in rows]))
        print(f"  {role:10s} detected {hit:3d}/{len(rows)}   "
              f"median p(class) {median:.3f}")
    both = sum(1 for r in rows
               if r["target"]["detected"] and r["distract"]["detected"])
    print(f"\n  both copies detected in {both}/{len(rows)} -- the cases where "
          f"disambiguation is even askable")
    out = args.out or os.path.splitext(args.cases)[0] + "_probe.json"
    json.dump({"iou_hit": IOU_HIT, "cases": rows}, open(out, "w"), indent=1)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
