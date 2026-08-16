"""
show_views.py -- the swept views of one case, with the correspondence drawn on.

`probe_corr.py` reports that the occluded target is corresponded in 32% of views
and its unoccluded twin in 56%, which is the mechanism by which channel A pushes
the correct pair DOWN the ranking.  That is a number about pictures nobody has
looked at.  This draws them.

Per view:

    SOLID green    the target's true box, from the third-party segmentation
    DASHED white   the twin's true box
    MAGENTA        the box of the slot that mutual-NN matched to the target's
                   REFERENCE slot -- absent when the correspondence found none

So a view captioned `no match` with the target plainly visible says the cosine
floor is the binding constraint; a magenta box sitting on the twin says the
embeddings cannot tell two copies of one asset apart; and a view where the
target has genuinely left the frame says the sweep is too wide, which is a
different problem with a different fix.

    python show_views.py --cases nvs_pilot/cases/cases_tabletop.json --case 5
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from probe_corr import GATE_COS, SKIP_VIEWS, iou, mutual_match, seg_box
from robot.nvs_lemniscate import LOOKAT_DIST

#: BGR, matching `show_tasks.py` so the two sheets read the same way.
TARGET = (140, 255, 140)
TWIN = (245, 245, 245)
MATCH = (230, 120, 230)


def sweep_views(controller, case: Dict[str, Any], egtr, args):
    """Render the lemniscate and return per-view frames, truth and matches."""
    from robot.nvs_lemniscate import camera_for, lemniscate, orbit_centre
    from robot.proc_scene import ROOM, View, look_from, rebuild, visible_box
    from robot.sgg_live import raw_predict

    from fuse_live import record

    event = rebuild(controller, case)
    rc = View(event)
    rc.controller = controller

    raw = raw_predict(egtr, event.frame)
    boxes = raw["boxes"].numpy()
    slots = {}
    for name in ("target", "distract", "occluder"):
        gt = visible_box(event, name)
        ious = np.array([iou(boxes[q].tolist(), gt) for q in range(len(boxes))]
                        ) if gt else np.zeros(len(boxes))
        slots[name] = int(ious.argmax()) if ious.max() >= 0.5 else None

    centre = orbit_centre(rc, args.lookat)
    reference = event.frame.copy()
    camera = rc.camera_xyz.copy()
    corner = min(((0.3, 0.3), (0.3, ROOM - 0.3), (ROOM - 0.3, 0.3),
                  (ROOM - 0.3, ROOM - 0.3)),
                 key=lambda p: -math.dist(p, (camera[0], camera[2])))
    look_from(controller, corner[0], corner[1], 0.0, 0.0, force=True)

    colours = {name: col for col, name in event.color_to_object_id.items()
               if name in ("target", "distract", "occluder")}
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
        truth.append({name: seg_box(seg, col) for name, col in colours.items()}
                     | {"az": az, "el": el})

    built = record(egtr, reference, frames)
    h_ref = built["rec"]["h_ref"].float().numpy()
    panels = []
    for (index, frame), boxes_gt, entry in zip(frames, truth,
                                               built["rec"]["views"]):
        hv = entry["h"].float().numpy()
        matched = (None if slots["target"] is None
                   else mutual_match(h_ref, hv, slots["target"], args.cos))
        # THE LANDMARK TOO.  A pair enters the ranking only if BOTH endpoints
        # correspond (`ok[i] and ok[j]` in `_pair_field`), so a view that finds
        # the target and loses the laptop contributes nothing -- and the caption
        # has to say which of the two failed.
        lmk = (None if slots["occluder"] is None
               else mutual_match(h_ref, hv, slots["occluder"], args.cos))
        box = None if matched is None else entry["boxes"][matched].tolist()
        lmk_box = None if lmk is None else entry["boxes"][lmk].tolist()
        verdict = "no match"
        if box is not None:
            on_t = iou(box, boxes_gt["target"]) if boxes_gt.get("target") else 0
            on_w = (iou(box, boxes_gt["distract"]) if boxes_gt.get("distract")
                    else 0)
            verdict = ("neither" if max(on_t, on_w) < 0.3
                       else "TWIN" if on_w > on_t else "target")
        both = box is not None and lmk_box is not None
        panels.append({"frame": frame, "index": index, "truth": boxes_gt,
                       "box": box, "lmk_box": lmk_box,
                       "verdict": f"{verdict}{' +lmk' if both else ' -LMK'}",
                       "skipped": index in SKIP_VIEWS})
    return reference, panels


def draw(cv2, panel: Dict[str, Any]) -> np.ndarray:
    from show_tasks import dashed

    canvas = np.ascontiguousarray(panel["frame"][:, :, ::-1])
    if panel["truth"].get("target"):
        b = [int(v) for v in panel["truth"]["target"]]
        cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), TARGET, 2)
    if panel["truth"].get("distract"):
        dashed(cv2, canvas, panel["truth"]["distract"], TWIN)
    if panel.get("lmk_box"):
        b = [int(v) for v in panel["lmk_box"]]
        cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), (90, 200, 255), 1)
    if panel["box"]:
        b = [int(v) for v in panel["box"]]
        cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), MATCH, 2)
    return canvas


def sheet(cv2, reference, panels: Sequence[Dict[str, Any]], columns: int,
          tile=(320, 240)) -> np.ndarray:
    bar, pad = 20, 4
    cell = (tile[0] + 2 * pad, tile[1] + bar + 2 * pad)
    cards = [{"frame": reference, "index": -1, "truth": {}, "box": None,
              "verdict": "reference", "skipped": False}] + list(panels)
    rows = (len(cards) + columns - 1) // columns
    out = np.full((rows * cell[1], columns * cell[0], 3), 22, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, panel in enumerate(cards):
        row, column = divmod(i, columns)
        x, y = column * cell[0], row * cell[1]
        out[y + pad:y + pad + tile[1], x + pad:x + pad + tile[0]] = \
            cv2.resize(draw(cv2, panel), tile)
        label = ("reference" if panel["index"] < 0
                 else f"view {panel['index']:>2d}"
                      + ("  [skipped]" if panel["skipped"]
                         else f"  {panel['verdict']}"))
        colour = ((150, 220, 150) if panel["verdict"] == "target"
                  else (120, 120, 255) if panel["verdict"] == "TWIN"
                  else (235, 235, 235) if panel["index"] < 0
                  else (140, 140, 140))
        cv2.putText(out, label, (x + pad + 2, y + pad + tile[1] + 14), font,
                    0.42, colour, 1, cv2.LINE_AA)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="nvs_pilot/cases/cases_tabletop.json")
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--max-az", type=float, default=30.0)
    ap.add_argument("--max-el", type=float, default=15.0)
    ap.add_argument("--cos", type=float, default=GATE_COS)
    ap.add_argument("--lookat", type=float, default=LOOKAT_DIST,
                    help="metres down the optical axis the sweep orbits")
    ap.add_argument("--columns", type=int, default=6)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import cv2

    from robot.proc_scene import open_room
    from robot.sgg_live import load_egtr

    case = json.load(open(args.cases))["cases"][args.case]
    egtr = load_egtr()
    controller = open_room(args.width, args.height, args.fov)
    try:
        reference, panels = sweep_views(controller, case, egtr, args)
    finally:
        controller.stop()

    out = args.out or f"nvs_pilot/views_case{args.case}.png"
    cv2.imwrite(out, sheet(cv2, reference, panels, args.columns))
    counts: Dict[str, int] = {}
    for panel in panels:
        if not panel["skipped"]:
            counts[panel["verdict"]] = counts.get(panel["verdict"], 0) + 1
    print(f"\n  {case['instruction']}  ->  {out}")
    print("  " + "   ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
