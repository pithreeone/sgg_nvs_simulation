"""
show_slot.py -- look at what `build_slot.py` built, before trusting the numbers.

The whole claim about a slot case is a claim about ANGLES: that the best view is
interior to the sweep and both ends are worse.  `show_tasks.py` draws the
reference view only, which is the one frame that cannot show it.  So this
replays the case's own swept poses -- the same `look_along` the generator
measured with, so the pictures and the recorded curve cannot disagree -- and
tiles them.

    filmstrip   one panel per azimuth, target in GREEN, the named landmark in
                BLUE, the unnamed blocker in RED, captioned with the hidden
                fraction.  REF marks 0 degrees and BEST marks the argmin.
    curves      hidden-against-azimuth for every case at once, with the
                reference, the best view and the two ends marked.  This is the
                one that says whether "walk to the end" loses on every case or
                only on the ones that were looked at.

No EGTR, no fusion: this asks whether the GEOMETRY is what was asked for, and
mixing perception into that picture would make a bad slot and a bad detection
look the same.

    python viz/show_slot.py --cases datasets/robot/cases_slot.json --case 0
    python viz/show_slot.py --cases datasets/robot/cases_slot.json --curves
"""

from __future__ import annotations

# `python viz/<script>.py` puts viz/ on sys.path, not the repo root, so the
# top-level modules would not import.  Same bootstrap as analysis/ and gen/.
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)


import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from build_robotic_task.build_slot import look_along
from robot.proc_scene import open_room, rebuild, visible_box

#: BGR, and chosen so the two OCCLUDERS are told apart at a glance: the blue one
#: is in the instruction, the red one never is.
COLOURS = {"target": (80, 220, 80), "occluder": (240, 170, 40),
           "blocker": (60, 60, 230)}


def draw(frame, boxes: Dict[str, Optional[Sequence[float]]], caption: str,
         tag: str, scale: float) -> np.ndarray:
    """One panel: the frame, the three boxes, and a caption bar."""
    canvas = frame.copy()
    for name, box in boxes.items():
        if box is None:
            continue
        x0, y0, x1, y1 = (int(v) for v in box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOURS[name], 3)
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA)
    bar = 22
    canvas = cv2.copyMakeBorder(canvas, 0, bar, 0, 0, cv2.BORDER_CONSTANT,
                                value=(24, 24, 24))
    cv2.putText(canvas, caption, (6, canvas.shape[0] - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1,
                cv2.LINE_AA)
    if tag:
        # The tag goes at the TOP so it cannot be confused with the number, and
        # in the target's own colour so REF/BEST read as being about the target.
        cv2.putText(canvas, tag, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLOURS["target"], 2, cv2.LINE_AA)
    return canvas


def tile(panels: Sequence[np.ndarray], columns: int) -> np.ndarray:
    """Grid the panels, padding the last row so the rows stack."""
    rows = []
    for start in range(0, len(panels), columns):
        row = list(panels[start:start + columns])
        while len(row) < columns:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def filmstrip(case: Dict[str, Any], args) -> np.ndarray:
    """The case's own swept poses, replayed and tiled."""
    controller = open_room(args.width, args.height, args.fov)
    try:
        rebuild(controller, case)
        table = case["table"]
        target = next(o for o in case["objects"] if o["name"] == "target")
        target_xz = np.array([target["position"]["x"], target["position"]["z"]])
        rows = [r for r in case["sweep"]
                if abs(r["azimuth"]) <= args.reach][::args.every]
        panels = []
        for row in rows:
            event, _ = look_along(controller, target_xz, table["top"],
                                  case["radius"], row["azimuth"])
            boxes = {name: visible_box(event, name) for name in COLOURS}
            tag = ("REF" if row["azimuth"] == 0.0 else
                   "BEST" if row["azimuth"] == case["best_azimuth"] else
                   "" if row["reachable"] else "unreachable")
            panels.append(draw(event.frame[:, :, ::-1], boxes,
                               f"{row['azimuth']:+.1f} deg   "
                               f"{row['hidden']:.0%} hidden   {row['px']} px",
                               tag, args.scale))
        return tile(panels, args.columns)
    finally:
        controller.stop()


def contact(cases: Sequence[Dict[str, Any]], args) -> np.ndarray:
    """Every case's REFERENCE view, one panel each, on one sheet.

    The frame comes from `rebuild`, which is the case's own stored start pose --
    so this is exactly what a runner sees at step 0, and nothing about the sweep
    enters.  What it is for is judging the SCENES rather than the angles: whether
    the instruction names something a reader would name, whether the landmark
    dominates, whether the target is a recognisable object at all.
    """
    controller = open_room(args.width, args.height, args.fov)
    try:
        panels = []
        for index, case in enumerate(cases):
            event = rebuild(controller, case)
            boxes = {name: visible_box(event, name) for name in COLOURS}
            panels.append(draw(
                event.frame[:, :, ::-1], boxes,
                f"{index}  {case['scene']}  {case['staged_occlusion']:.0%} hid"
                f"  best {case['best_hidden']:.0%}@{case['best_azimuth']:+.0f}",
                "", args.scale))
        return tile(panels, args.columns)
    finally:
        controller.stop()


def curves(cases: Sequence[Dict[str, Any]], data: Dict[str, Any], out: str):
    """hidden(azimuth) for every case, with what each acceptance test reads."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reach = data.get("reach", 30.0)
    wide = min(len(cases), 4)
    tall = (len(cases) + wide - 1) // wide
    fig, axes = plt.subplots(tall, wide, figsize=(3.6 * wide, 2.7 * tall),
                             squeeze=False, sharex=True, sharey=True)
    for ax, case in zip(axes.ravel(), cases):
        rows = [r for r in case["sweep"] if abs(r["azimuth"]) <= reach]
        az = [r["azimuth"] for r in rows]
        hid = [r["hidden"] * 100 for r in rows]
        ax.plot(az, hid, "-o", ms=2.5, lw=1.4, color="#3b6ea5")
        ax.axhspan(0, data.get("open_max", 0.30) * 100, color="#3b6ea5",
                   alpha=0.10, lw=0)
        ax.axvline(0, color="#888", lw=0.8, ls=":")
        ax.plot([case["best_azimuth"]], [case["best_hidden"] * 100], "*",
                ms=13, color="#d1495b", zorder=5)
        # The two ends are what --falloff reads, so mark them rather than
        # leaving the reader to find the endpoints of the line.
        for edge in (-reach, reach):
            hit = next((r for r in rows if r["azimuth"] == edge), None)
            if hit:
                ax.plot([edge], [hit["hidden"] * 100], "s", ms=5,
                        color="#2a2a2a", zorder=5)
        ax.set_title(f"{case['scene']}  ref {case['staged_occlusion']:.0%}"
                     f"  best {case['best_hidden']:.0%} at "
                     f"{case['best_azimuth']:+.1f}", fontsize=8)
        ax.set_ylim(-4, 104)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("azimuth, degrees", fontsize=8)
    for row in axes:
        row[0].set_ylabel("target hidden, %", fontsize=8)
    fig.suptitle("star = best view, squares = the ends --falloff compares it "
                 "against, band = --open-max", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"  -> {out}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="datasets/robot/cases_slot.json")
    ap.add_argument("--case", type=int, default=None,
                    help="which case to draw a filmstrip of.  Omitted with "
                         "--curves, every case is plotted")
    ap.add_argument("--curves", action="store_true",
                    help="draw hidden(azimuth) for every case instead of frames")
    ap.add_argument("--refs", action="store_true",
                    help="one sheet of every case's REFERENCE view -- what a "
                         "runner sees at step 0, for judging the scenes rather "
                         "than the angles")
    ap.add_argument("--every", type=int, default=2, metavar="N",
                    help="keep every Nth swept azimuth; 25 panels is a wall")
    ap.add_argument("--columns", type=int, default=7)
    ap.add_argument("--scale", type=float, default=0.42)
    ap.add_argument("--reach", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    data = json.load(open(args.cases))
    cases = data["cases"]
    if not data.get("slot"):
        print(f"  ! {args.cases} was not built by build_slot.py -- no `sweep`")
        return 1

    stem = os.path.splitext(os.path.basename(args.cases))[0]
    if args.curves:
        out = args.out or f"nvs_pilot/{stem}_curves.png"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        curves(cases, data, out)
        return 0

    if args.refs:
        out = args.out or f"nvs_pilot/{stem}_refs.png"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        print(f"  {len(cases)} reference views, {args.columns} per row")
        cv2.imwrite(out, contact(cases, args))
        print(f"  -> {out}")
        return 0

    index = args.case or 0
    case = cases[index]
    print(f"[{index}] {case['scene']}  {case['instruction']}")
    print(f"    landmark {case['objects'][1]['asset']}  "
          f"blocker {case['objects'][2]['asset']} ({case['blocker_class']})")
    out = args.out or f"nvs_pilot/{stem}_{index}.png"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cv2.imwrite(out, filmstrip(case, args))
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
