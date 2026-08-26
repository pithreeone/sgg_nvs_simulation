"""
show_candidates.py -- draw the boxes `conditioned()` actually hands the decision.

The pipeline never uses a box from a synthesised view: `triplets_of` takes boxes
and classes from the REFERENCE slots for every arm, because a novel view's boxes
live in a frame the grader has no ground truth for.  So these ARE the boxes the
robot decides and acts on, and what they look like is not a detail.

Two panels per case, because the two endpoints behave very differently:

    left    the top-K queries by p(instructed SUBJECT class)   -- the hard end
    right   the top-K queries by p(instructed OBJECT class)    -- the easy end

Thick green is the ground truth, read from the segmentation frame, so it is the
VISIBLE extent: a 35%-occluded target has a green box around the third of it
that shows.  A prediction covering the whole object is therefore expected to
look too big, which is the first thing to check before calling a box loose.

    python show_candidates.py --cases results/cases/cases_easy2.json --n 4 --topk 5
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from robot.world.proc_scene import open_room, rebuild, visible_box
from robot.task.task_find import iou

#: BGR.  Rank 1 is the brightest; later ranks fade, so the eye reads the
#: ordering without having to parse the labels.
RANKS = [(60, 60, 255), (60, 140, 255), (60, 200, 255), (90, 230, 200),
         (120, 220, 120), (170, 200, 100), (200, 170, 90), (210, 140, 90),
         (200, 110, 110), (180, 90, 140)]
TRUTH = (60, 230, 60)
HIT_IOU = 0.5


def panel(frame, boxes, ranked, probs_col, truth, title: str, topk: int):
    """One frame with the top-`topk` candidate boxes and the ground truth."""
    import cv2

    img = frame.copy()
    if truth:
        x0, y0, x1, y1 = (int(v) for v in truth)
        cv2.rectangle(img, (x0, y0), (x1, y1), TRUTH, 3)
    for rank, q in enumerate(ranked[:topk], 1):
        x0, y0, x1, y1 = (int(v) for v in boxes[q])
        colour = RANKS[(rank - 1) % len(RANKS)]
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, 1)
        hit = truth and iou(boxes[q].tolist(), truth) >= HIT_IOU
        cv2.putText(img, f"{rank}:{probs_col[q]:.2f}{'*' if hit else ''}",
                    (x0 + 2, max(11, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    colour, 1, cv2.LINE_AA)
    bar = np.zeros((22, img.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="results/cases/cases_easy2.json")
    ap.add_argument("--n", type=int, default=4, help="how many cases to draw")
    ap.add_argument("--topk", type=int, default=5,
                    help="candidates per side to draw.  The pipeline's own K is "
                         "10; more than about 5 is unreadable, and the point of "
                         "the picture is what the boxes LOOK like, not the count")
    ap.add_argument("--nms", type=float, default=0.0, metavar="IOU",
                    help="drop a candidate whose box overlaps a higher-ranked "
                         "one by more than this, WITHIN the top-K window -- the "
                         "same rule `fuse_live --cand-nms` applies.  0 draws the "
                         "list as the published pipeline forms it, in which the "
                         "top 10 covers a median of five objects.")
    ap.add_argument("--out", default="results/candidates")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    args = ap.parse_args(argv)

    import cv2

    from robot.sgg_live import load_egtr, raw_predict

    cases = json.load(open(args.cases))["cases"][:args.n]
    egtr = load_egtr()
    controller = open_room(args.width, args.height, args.fov)
    os.makedirs(args.out, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    try:
        for case in cases:
            event = rebuild(controller, case)
            raw = raw_predict(egtr, event.frame)
            probs = raw["probs_softmax"].cpu().numpy()
            boxes = raw["boxes"].numpy()
            index = {v: k - 1 for k, v in egtr["obj_names"].items()}
            frame = cv2.cvtColor(np.array(event.frame), cv2.COLOR_RGB2BGR)
            sides = []
            for role, key in (("target", "subject_class"),
                              ("occluder", "object_class")):
                col = probs[:, index[case[key]]]
                ranked = list(np.argsort(-col))
                if args.nms:
                    window, ranked = ranked[:args.topk * 2], []
                    for q in window:
                        if all(iou(boxes[q].tolist(), boxes[k].tolist())
                               <= args.nms for k in ranked):
                            ranked.append(q)
                truth = visible_box(event, role)
                first = next((r for r, q in enumerate(ranked, 1)
                              if truth and iou(boxes[q].tolist(), truth)
                              >= HIT_IOU), None)
                sides.append(panel(frame, boxes, ranked, col, truth,
                                   f"{case[key]}  top-{args.topk} by p(class)"
                                   f"   green = visible truth"
                                   f"   first hit at rank {first}", args.topk))
                rows.append({"scene": case["scene"], "role": role,
                             "first_hit_rank": first,
                             "p_top": float(col[ranked[0]])})
            name = f"{case['scene'].replace('|', '_')}.png"
            cv2.imwrite(os.path.join(args.out, name), np.hstack(sides))
            t, o = rows[-2], rows[-1]
            print(f"  {case['scene']:13} {case['instruction'][:34]:34} "
                  f"subject p1 {t['p_top']:.3f} first hit #{t['first_hit_rank']}"
                  f"   landmark p1 {o['p_top']:.3f} "
                  f"first hit #{o['first_hit_rank']}", flush=True)
    finally:
        controller.stop()
    print(f"\n  wrote {len(cases)} images to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
