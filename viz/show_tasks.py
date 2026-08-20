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

    python viz/show_tasks.py --cases datasets/robot/cases_slot.json --n 12
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
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: BGR.  Green for the graded instance, amber for the landmark: the two roles
#: are the thing most easily confused, so they never share a colour.
GRADED = (140, 255, 140)
LANDMARK = (110, 190, 255)
#: Magenta: the DETECTION whose box and depth produce P-hat, the point the
#: motion policy orbits and aims at.  Drawn because the anchor was measured at a
#: median 0.48 m from the object it is supposed to sit on, with a worst case of
#: 2.28 m, and a box picture is the only way to see whether the detection or the
#: depth is at fault.
ANCHOR = (230, 120, 230)
#: White, dashed: the DISTRACTOR -- a second object of the target's own class,
#: placed off the sightline so the relation has to say WHICH one.  Drawn because
#: the only way to see that a distractor is doing its job is to check it is
#: visible, near the target, and not occluding it.
DISTRACTOR = (245, 245, 245)


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

    from robot.drive_triplet_scene import geometry
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

    distractor = case.get("distractor_name")
    wanted = sorted({graded, landmark} | ({distractor} if distractor else set()))
    geo = geometry(rc, wanted, names, False)
    canvas = np.ascontiguousarray(rc.event.frame[:, :, ::-1])

    anchor_box = anchor_err = None
    if args.anchor and case.get("object_class"):
        from robot.robot_controller import point_in_box
        from robot.sgg_live import raw_predict

        raw = raw_predict(args.egtr, rc.event.frame)
        classes = {v: k - 1 for k, v in args.egtr["obj_names"].items()}
        want = classes.get(case["object_class"])
        if want is not None:
            query = int(raw["probs_softmax"][:, want].argmax())
            anchor_box = raw["boxes"][query].tolist()
            point = point_in_box(rc, anchor_box, args.fov)
            truth = next((o["position"] for o in rc.event.metadata["objects"]
                          if o["name"] == landmark), None)
            if point is not None and truth:
                anchor_err = float(np.hypot(point[0] - truth["x"],
                                            point[2] - truth["z"]))
            cv2.rectangle(canvas, (int(anchor_box[0]), int(anchor_box[1])),
                          (int(anchor_box[2]), int(anchor_box[3])), ANCHOR, 2)
    for name, colour, solid in ([(graded, GRADED, True),
                                 (landmark, LANDMARK, False)]
                                + ([(distractor, DISTRACTOR, False)]
                                   if distractor else [])):
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
            "share": share, "anchor_err": anchor_err,
            "distractor": distractor,
            "distractor_visible": bool(distractor
                                       and geo.get(distractor, {})
                                       .get("bbox_visible")),
            "graded_visible": bool(box)}


def one_proc(controller, case: Dict[str, Any], args) -> Optional[Dict[str, Any]]:
    """
    The same picture for a procedural tabletop case.

    Separate from `one` because almost nothing it does applies here.  There is
    no `stage_at` -- `rebuild` restores the whole scene from the record, exactly,
    because the scene IS four spawns.  There is no `geometry` -- every box comes
    from the segmentation frame by instance name, so a second copy of the target
    asset gets its own box instead of being merged with the first, which is the
    entire reason these cases exist.  And the controller is opened ONCE and
    reused: `rebuild` resets it, so the per-case Unity restart the iTHOR path
    pays for is unnecessary here.
    """
    import cv2

    from robot.proc_scene import View, rebuild, visible_box

    event = rebuild(controller, case)
    graded = case["target_name"]
    landmark = case["occluder_name"]
    distractor = case.get("distractor_name")

    canvas = np.ascontiguousarray(event.frame[:, :, ::-1])
    anchor_err = None
    if args.anchor and case.get("object_class"):
        from robot.robot_controller import point_in_box
        from robot.sgg_live import raw_predict

        raw = raw_predict(args.egtr, event.frame)
        classes = {v: k - 1 for k, v in args.egtr["obj_names"].items()}
        want = classes.get(case["object_class"])
        if want is not None:
            box = raw["boxes"][int(raw["probs_softmax"][:, want].argmax())]
            point = point_in_box(View(event), box.tolist(), args.fov)
            truth = next((o["position"] for o in event.metadata["objects"]
                          if o["name"] == landmark), None)
            if point is not None and truth:
                anchor_err = float(np.hypot(point[0] - truth["x"],
                                            point[2] - truth["z"]))
            cv2.rectangle(canvas, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), ANCHOR, 2)

    boxes = {name: visible_box(event, name)
             for name in (graded, landmark, distractor) if name}
    for name, colour, solid in ([(graded, GRADED, True),
                                 (landmark, LANDMARK, False)]
                                + ([(distractor, DISTRACTOR, False)]
                                   if distractor else [])):
        box = boxes.get(name)
        if not box:
            continue
        if solid:
            cv2.rectangle(canvas, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), colour, 2)
        else:
            dashed(cv2, canvas, box, colour)

    box = boxes.get(graded)
    frame_px = event.frame.shape[0] * event.frame.shape[1]
    share = (((box[2] - box[0]) * (box[3] - box[1])) / frame_px) if box else 0.0
    return {"frame": canvas, "instruction": case["instruction"],
            "occlusion": case["staged_occlusion"], "scene": case["scene"],
            "share": share, "anchor_err": anchor_err,
            "distractor": distractor,
            "distractor_visible": bool(boxes.get(distractor)),
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
                            + ("" if panel.get("anchor_err") is None
                               else f"   P-hat off by "
                                    f"{panel['anchor_err']:.2f} m")
                            + ("" if panel["graded_visible"]
                               else "   [GRADED INSTANCE NOT VISIBLE]")
                            + ("" if not panel.get("distractor")
                               else ("   +distractor"
                                     if panel["distractor_visible"]
                                     else "   [DISTRACTOR NOT VISIBLE]")),
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
    ap.add_argument("--anchor", action="store_true",
                    help="also draw the detection that produces P-hat, and how "
                         "far the resulting 3D point lands from the landmark")
    ap.add_argument("--out", default=None,
                    help="default: <cases stem>_tasks.png beside the case file")
    args = ap.parse_args(argv)

    import cv2

    from robot import drive

    args.egtr = None
    if args.anchor:
        from robot.sgg_live import load_egtr
        args.egtr = load_egtr()

    data = json.load(open(args.cases))
    cases = data["cases"][args.start:args.start + args.n]
    panels: List[Dict[str, Any]] = []

    # A procedural list carries its own scenes; `case["scene"]` is a label like
    # `tabletop|3` and passing it to `open_scene` fails with the build's full
    # list of floor plans, which reads as a missing asset rather than as the
    # wrong loader.
    if data.get("procedural"):
        from robot.proc_scene import open_room

        controller = open_room(args.width, args.height, args.fov)
        try:
            for index, case in enumerate(cases, 1):
                print(f"[{index}/{len(cases)}] {case['scene']}  "
                      f"{case['instruction']}")
                try:
                    panel = one_proc(controller, case, args)
                except Exception as error:                      # noqa: BLE001
                    print(f"    ! {type(error).__name__}: {error}")
                    panel = None
                if panel:
                    panels.append(panel)
        finally:
            controller.stop()
    else:
        for index, case in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {case['scene']}  "
                  f"{case['instruction']}")
            rc = drive.open_scene(case["scene"], args.width, args.height,
                                  args.fov, case["start"])
            try:
                panel = one(rc, case, args)
            except Exception as error:                          # noqa: BLE001
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
