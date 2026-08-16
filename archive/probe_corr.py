"""
probe_corr.py -- does the correspondence in channel A land on the right copy?

Channel A repairs a query's object score by taking the best score among the
views that CORRESPOND to it, where corresponding means cosine floor plus mutual
nearest neighbour on the EGTR query embeddings.  `channels.object_scores` states
the limitation in its own docstring: mutual-NN verifies mutual PREFERENCE, not
identity.  The tabletop cases are the adversarial input for exactly that -- the
target and the distractor are two instances of the SAME asset, so their query
embeddings are as close as two embeddings get, and a correspondence that is only
"nearest available" has no way to tell them apart.

If that is what happens, A does not merely fail to help: it pools the wrong
instance's score into the target and the target's into the wrong instance, which
is worse than leaving the reference score alone.  On these 40 cases A moved the
correct pair DOWN the ranking 11 times and UP 4, so something is costing it.

What this measures, per case: take the reference slot that best localises the
TARGET, follow its correspondence into each swept view, and ask which object the
matched slot's box actually covers there.

    python probe_corr.py --cases nvs_pilot/cases/cases_tabletop.json --n 12
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from robot.nvs_lemniscate import LOOKAT_DIST

#: The lemniscate's duplicates of the reference pose, which `object_scores`
#: skips; counting them here would inflate agreement with free copies of the
#: very frame the correspondence starts from.
SKIP_VIEWS = (9, 19)

#: `object_scores`' own default.  Read from there rather than retyped, so this
#: probe cannot drift away from the channel it is meant to describe.
GATE_COS = 0.80


def seg_box(frame: np.ndarray, colour) -> Optional[List[float]]:
    """xyxy of one instance in a segmentation frame, or None if it draws none."""
    ys, xs = np.where((frame == np.array(colour, dtype=frame.dtype)).all(-1))
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()),
            float(xs.max() + 1), float(ys.max() + 1)]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / (area - inter))


def mutual_match(h_ref: np.ndarray, h_view: np.ndarray, q: int,
                 cos: float = GATE_COS) -> Optional[int]:
    """
    The view slot that mutually prefers reference slot `q`, or None.

    The same rule `object_scores` uses: cosine floor, then mutual nearest
    neighbour.  Written out here rather than imported because the channel
    returns only the pooled score and never says which slot it pooled from --
    which is precisely the thing under examination.
    """
    a = h_ref / (np.linalg.norm(h_ref, axis=-1, keepdims=True) + 1e-8)
    b = h_view / (np.linalg.norm(h_view, axis=-1, keepdims=True) + 1e-8)
    sim = a @ b.T
    m = int(sim[q].argmax())
    if sim[q, m] < cos:
        return None
    return m if int(sim[:, m].argmax()) == q else None


def one(controller, case: Dict[str, Any], egtr, args) -> Optional[Dict[str, Any]]:
    """Sweep one case and tally where the target's correspondences land."""
    from robot.nvs_lemniscate import camera_for, lemniscate, orbit_centre
    from robot.proc_scene import ROOM, View, look_from, rebuild, visible_box
    from robot.sgg_live import raw_predict

    from fuse_live import record

    event = rebuild(controller, case)
    rc = View(event)
    rc.controller = controller

    raw = raw_predict(egtr, event.frame)
    boxes = raw["boxes"].numpy()
    # BOTH copies, because the question is comparative.  A repairing the target
    # in a third of views is only a problem if it repairs the unoccluded twin in
    # far more of them: the ranking is a product of the two endpoints' scores,
    # so what moves the correct pair down the list is the OTHER slot going up.
    slots = {}
    for name in ("target", "distract"):
        gt = visible_box(event, name)
        if gt is None:
            return None
        ious = np.array([iou(boxes[q].tolist(), gt) for q in range(len(boxes))])
        if ious.max() < 0.5:
            return None
        slots[name] = int(ious.argmax())
    if slots["target"] == slots["distract"]:
        return None                       # one box over both; nothing to compare

    centre = orbit_centre(rc, args.lookat)
    reference = event.frame.copy()
    camera = rc.camera_xyz.copy()
    corner = min(((0.3, 0.3), (0.3, ROOM - 0.3), (ROOM - 0.3, 0.3),
                  (ROOM - 0.3, ROOM - 0.3)),
                 key=lambda p: -math.dist(p, (camera[0], camera[2])))
    look_from(controller, corner[0], corner[1], 0.0, 0.0, force=True)

    colours = {name: col for col, name in event.color_to_object_id.items()
               if name in ("target", "distract")}
    frames, truth, first = [], [], True
    for index, (az, el) in enumerate(lemniscate(args.views, args.max_az,
                                                args.max_el)):
        pose = camera_for(centre, camera, az, el)
        controller.step(
            action="AddThirdPartyCamera" if first else "UpdateThirdPartyCamera",
            position=dict(pose["position"]),
            rotation={"x": pose["pitch"], "y": pose["yaw"], "z": 0.0},
            fieldOfView=args.fov,
            **({} if first else {"thirdPartyCameraId": 0}))
        first = False
        ev = controller.last_event
        frames.append((index, np.array(ev.third_party_camera_frames[0])))
        seg = ev.third_party_instance_segmentation_frames[0]
        truth.append({name: seg_box(seg, col) for name, col in colours.items()})

    built = record(egtr, reference, frames)
    h_ref = built["rec"]["h_ref"].float().numpy()

    tally = {f"{who} {what}": 0
             for who in ("target", "distract")
             for what in ("right", "swapped", "neither", "no match")}
    for entry, boxes_gt in zip(built["rec"]["views"], truth):
        if int(entry["v"]) in SKIP_VIEWS:
            continue
        h_view = entry["h"].float().numpy()
        for who, other in (("target", "distract"), ("distract", "target")):
            matched = mutual_match(h_ref, h_view, slots[who])
            if matched is None:
                tally[f"{who} no match"] += 1
                continue
            box = entry["boxes"][matched].tolist()
            mine = iou(box, boxes_gt[who]) if boxes_gt.get(who) else 0.0
            theirs = iou(box, boxes_gt[other]) if boxes_gt.get(other) else 0.0
            if max(mine, theirs) < 0.3:
                tally[f"{who} neither"] += 1
            elif theirs > mine:
                tally[f"{who} swapped"] += 1
            else:
                tally[f"{who} right"] += 1
    return tally


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="nvs_pilot/cases/cases_tabletop.json")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--max-az", type=float, default=30.0)
    ap.add_argument("--max-el", type=float, default=15.0)
    ap.add_argument("--lookat", type=float, default=LOOKAT_DIST,
                    help="metres down the optical axis the sweep orbits")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    args = ap.parse_args(argv)

    from robot.proc_scene import open_room
    from robot.sgg_live import load_egtr

    cases = json.load(open(args.cases))["cases"][:args.n]
    egtr = load_egtr()
    controller = open_room(args.width, args.height, args.fov)
    totals: Dict[str, int] = {}
    try:
        for index, case in enumerate(cases, 1):
            try:
                tally = one(controller, case, egtr, args)
            except Exception as error:                          # noqa: BLE001
                print(f"  ! {type(error).__name__}: {error}", flush=True)
                tally = None
            if not tally:
                continue
            for key, value in tally.items():
                totals[key] = totals.get(key, 0) + value
            print(f"  [{index}/{len(cases)}] {case['instruction']:36s} "
                  f"target matched "
                  f"{tally['target right'] + tally['target swapped'] + tally['target neither']:>2d}"
                  f", twin matched "
                  f"{tally['distract right'] + tally['distract swapped'] + tally['distract neither']:>2d}",
                  flush=True)
    finally:
        controller.stop()

    if not totals:
        print("nothing measured")
        return 1
    print(f"\n  over {len(cases)} cases, per swept view:")
    print(f"    {'slot':10s} {'right':>7s} {'swapped':>8s} {'neither':>8s} "
          f"{'no match':>9s}   matched at all")
    for who, label in (("target", "target (hidden)"),
                       ("distract", "twin (in clear)")):
        row = [totals.get(f"{who} {w}", 0)
               for w in ("right", "swapped", "neither", "no match")]
        n = sum(row)
        print(f"    {label:16s} {row[0]:>5d} {row[1]:>8d} {row[2]:>8d} "
              f"{row[3]:>9d}   {(n - row[3]) / n:>6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
