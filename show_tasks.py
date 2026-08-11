"""
show_tasks.py -- render what a frozen case list actually asks, as one sheet.

Every number in these experiments rests on a claim about a picture: that the
staged occluder really hides the landmark, that the instruction names something
a person would name, and that the instance being graded is the one the sentence
points at.  None of that is visible in a JSON file, and two of the three have
already been wrong once -- `in front of` moved the graded instance to the
occluder and three call sites went on measuring the landmark.

So this draws, on the robot's own reference frame:

    SOLID box    the instance being GRADED -- `target_name`, whatever the
                 rewrite moved it to
    DASHED box   the landmark, the other endpoint of the instruction
    caption      the instruction, and how much of the landmark is hidden

It re-stages each case exactly as a run does (`stage_at` with the stored scene
snapshot), so the picture is the scene the experiment scores, not a fresh
settle that happens to look similar.

    python show_tasks.py --cases nvs_pilot/cases_behind.json --n 12
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: BGR.  Green for the graded instance, amber for the landmark: the two roles
#: are the thing most easily confused, so they never share a colour.
GRADED = (140, 255, 140)
LANDMARK = (110, 190, 255)


def dashed(cv2, canvas, box, colour, thickness: int = 2, dash: int = 9) -> None:
    """A dashed rectangle, so the landmark cannot be mistaken for the target."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    for x in range(x0, x1, dash * 2):
        cv2.line(canvas, (x, y0), (min(x + dash, x1), y0), colour, thickness)
        cv2.line(canvas, (x, y1), (min(x + dash, x1), y1), colour, thickness)
    for y in range(y0, y1, dash * 2):
        cv2.line(canvas, (x0, y), (x0, min(y + dash, y1)), colour, thickness)
        cv2.line(canvas, (x1, y), (x1, min(y + dash, y1)), colour, thickness)


def one(rc, case: Dict[str, Any], args) -> Optional[Dict[str, Any]]:
    """Stage the case and return its frame with both endpoints drawn."""
    import cv2

    from eval_nvs_pointer import geometry
    from robot.task_find import stage_at
    from vg.vg150 import THOR_TO_VG150

    # The OTHER endpoint of the sentence, which is not the same field for
    # every predicate: `on` names the receptacle, `behind` names the occluder,
    # and `in front of` -- which moved grading to the occluder -- names the
    # hidden object it stored as `landmark_name`.
    graded = case["target_name"]
    other = (case.get("landmark_name")
             or (case.get("occluder_name") if case.get("predicate") == "behind"
                 else case.get("receptacle_name")))
    landmark = other or case["receptacle_name"]
    # Occlusion is always measured on the hidden object.
    hidden = case.get("landmark_name") or case["target_name"]
    names = sorted({o["name"] for o in rc.event.metadata["objects"]
                    if o["objectType"] in THOR_TO_VG150
                    or o.get("moveable") or o.get("pickupable")})

    staged = stage_at(rc, hidden, case["occluder_name"],
                      case["occluder_position"], case.get("scene_poses"))
    if staged is None:
        return None

    geo = geometry(rc, sorted({graded, landmark}), names, False)
    canvas = np.ascontiguousarray(rc.event.frame[:, :, ::-1])
    for name, colour, solid in ((graded, GRADED, True),
                                (landmark, LANDMARK, False)):
        box = geo.get(name, {}).get("bbox_visible")
        if not box:
            continue
        if solid:
            cv2.rectangle(canvas, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), colour, 2)
        else:
            dashed(cv2, canvas, box, colour)
    box = geo.get(graded, {}).get("bbox_visible")
    frame_px = rc.event.frame.shape[0] * rc.event.frame.shape[1]
    share = (((box[2] - box[0]) * (box[3] - box[1])) / frame_px) if box else 0.0
    return {"frame": canvas, "instruction": case["instruction"],
            "occlusion": staged["occlusion"], "scene": case["scene"],
            "share": share,
            "graded_visible": bool(box)}


def sheet(cv2, panels: Sequence[Dict[str, Any]], columns: int = 4,
          tile=(400, 300)) -> np.ndarray:
    bar, pad = 34, 5
    cell = (tile[0] + 2 * pad, tile[1] + bar + 2 * pad)
    rows = (len(panels) + columns - 1) // columns
    canvas = np.full((rows * cell[1], columns * cell[0], 3), 22, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        x, y = column * cell[0], row * cell[1]
        canvas[y + pad:y + pad + tile[1], x + pad:x + pad + tile[0]] = \
            cv2.resize(panel["frame"], tile)
        cv2.putText(canvas, panel["instruction"][:46],
                    (x + pad + 2, y + pad + tile[1] + 15), font, 0.44,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{panel['scene']}   hidden "
                            f"{panel['occlusion']:.0%}   target "
                            f"{panel['share'] * 100:.2f}% of frame"
                            + ("" if panel["graded_visible"]
                               else "   [GRADED INSTANCE NOT VISIBLE]"),
                    (x + pad + 2, y + pad + tile[1] + 29), font, 0.40,
                    (150, 150, 150) if panel["graded_visible"] else (120, 120, 255),
                    1, cv2.LINE_AA)
    return canvas


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--n", type=int, default=12, help="how many to draw")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--columns", type=int, default=4)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--out", default=None,
                    help="default: <cases stem>_tasks.png beside the case file")
    args = ap.parse_args(argv)

    import cv2

    from robot import drive

    cases = json.load(open(args.cases))["cases"][args.start:args.start + args.n]
    panels: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['scene']}  {case['instruction']}")
        rc = drive.open_scene(case["scene"], args.width, args.height, args.fov,
                              case["start"])
        try:
            panel = one(rc, case, args)
        except Exception as error:                              # noqa: BLE001
            print(f"    ! {type(error).__name__}: {error}")
            panel = None
        finally:
            rc.stop()
        if panel:
            panels.append(panel)

    if not panels:
        print("nothing to draw")
        return 1
    out = args.out or os.path.splitext(args.cases)[0] + "_tasks.png"
    cv2.imwrite(out, sheet(cv2, panels, args.columns))
    import statistics as st
    missing = sum(1 for p in panels if not p["graded_visible"])
    shares = sorted(p["share"] for p in panels)
    print(f"\n{len(panels)} panels -> {out}"
          + (f"   ({missing} with the graded instance not visible)" if missing
             else ""))
    print(f"  graded instance covers {st.median(shares) * 100:.2f}% of the "
          f"frame at the median, {shares[0] * 100:.2f}-{shares[-1] * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
