"""
plot_viewgrid.py -- the control arm's action space, from above.

`viewgrid.feasible` returns a count of what it rejected and why, which says how
many poses were lost but not WHERE.  A polar grid that looks full can still have
a hole exactly where the interesting views are, and a table of rejection counts
cannot show that.  This draws the floor plane -- AI2-THOR's x-z, metres -- with
one dot per surviving position.

SAME SYMBOLS AS `viewpoints.py`, so a figure from the simulator and one from the
real robot's map can be read side by side without a key each:

    blue dot + arrow   a pose that survives: where the robot stands and where it
                       looks.  One arrow per yaw offset, so a position that
                       survives all three carries three
    red star           each object the instruction names
    red ring           round the NEAR one -- which end of the red line the
                       sector is meant to sit on
    red line           the pair's own axis, which the lattice is laid along

    faint hollow dot   rejected, and not in `viewpoints.py`: it plots survivors
                       only, and a lattice that looks full can still have a hole
                       exactly where the interesting views are
    green dot + arrow  the pose the robot starts the case at

    python viz/plot_viewgrid.py --cases datasets/robot/cases_slot.json --case 0
"""

from __future__ import annotations

# `python viz/<script>.py` puts viz/ on sys.path, not the repo root, so the
# top-level modules would not import.  Same bootstrap as analysis/ and build/sgg/.
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import json
import math

import numpy as np
from PIL import Image

from robot.world.proc_scene import ROOM, Robot, open_room, rebuild
from robot.world.viewgrid import (R_MAX, R_MIN, SPACING, SPAN,
                                  WALL_MARGIN, feasible, lattice)

#: The rejected candidates are recomputed here rather than returned by
#: `feasible`, which counts them and moves on: the plot is the only consumer
#: that wants each one's position, and threading that through the experiment
#: path would put a figure's needs in the policy's way.  They are drawn in one
#: faint grey, not one colour per reason -- the counts in the caption say WHICH
#: test dropped how many, and the figure's job is to say WHERE.
#: Arrow length in metres.  Shorter than the lattice spacing, so a heading is
#: readable without the arrow reaching the next position and looking like a
#: path between the two.
ARROW = 0.22


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="datasets/robot/cases_slot.json")
    ap.add_argument("--case", type=int, nargs="+", default=[0])
    ap.add_argument("--fov", type=float, default=60.0)
    # SQUARE, because SEVA works on a square latent grid and a 4:3 input is
    # letterboxed into it.  THOR's `fieldOfView` is VERTICAL, so 60 degrees at
    # 800x600 was 75.6 degrees WIDE and at 600x600 is 60 -- the robot sees a
    # narrower slice of the room, and the case lists were staged at the old
    # aspect.  Recorded in the output either way.
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--frames", type=int, default=0, metavar="N",
                    help="also render what the robot sees from the first N "
                         "surviving poses, beside the plot")
    ap.add_argument("--span", type=float, default=SPAN,
                    help="degrees of azimuth kept about the pair's axis")
    ap.add_argument("--out", default="results/figs_viewgrid/grid.png")
    args = ap.parse_args(argv)

    cases = json.load(open(args.cases))["cases"]
    controller = open_room(args.width, args.height, args.fov)
    panels = []
    try:
        for index in args.case:
            case = cases[index]
            rc = Robot(controller, rebuild(controller, case))
            # Same two points `eval_move.run_case_proc` hands the episode.
            at = {o["name"]: o["position"] for o in case["objects"]}
            target = [at["target"]["x"], at["target"]["z"]]
            landmark = [at["occluder"]["x"], at["occluder"]["z"]]
            poses, centre, why = feasible(rc, landmark, target, args.fov,
                                          args.width, args.height, ROOM,
                                          span=args.span)
            panels.append({
                "index": index, "case": case, "why": why, "centre": centre,
                "span": args.span,
                "kept": np.array([[p["position"]["x"], p["position"]["z"],
                                   p["yaw"]] for p in poses],
                                 float).reshape(-1, 3),
                "start": [float(rc.agent_position["x"]),
                          float(rc.agent_position["z"])],
                "start_yaw": float(rc.agent_yaw),
                "target": target, "landmark": landmark})
            print(f"case {index}: {len(poses)} poses, rejected {why}",
                  flush=True)
            # WHAT THE POSE ACTUALLY LOOKS LIKE.  A dot on a floor plan says a
            # position survived the filters, not that the frame is usable.
            # THE START POSE FIRST, since that is the one every arm is read
            # against and the only one that is not a candidate.
            if args.frames:
                where = _os.path.join(_os.path.dirname(
                    _os.path.abspath(args.out)), f"case{index:02d}_start.png")
                Image.fromarray(
                    np.asarray(rc.event.frame)[..., :3]).save(where)
                print(f"    start                            -> {where}",
                      flush=True)
            for k, pose in enumerate(poses[:args.frames]):
                rc.teleport(position=pose["position"], yaw=pose["yaw"],
                            horizon=0.0)
                where = _os.path.join(_os.path.dirname(
                    _os.path.abspath(args.out)), f"case{index:02d}_view{k}.png")
                Image.fromarray(
                    np.asarray(rc.event.frame)[..., :3]).save(where)
                print(f"    view {k}  r {pose['radius']:.2f} m  "
                      f"az {pose['azimuth']:+.0f}  yaw_off "
                      f"{pose['yaw_offset']:+.0f}  -> {where}", flush=True)
    finally:
        controller.stop()

    cols = min(len(panels), 3)
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows),
                             squeeze=False)
    for ax, panel in zip(axes.ravel(), panels):
        centre = panel["centre"]
        # EVERY LATTICE POINT, so the holes are visible; `feasible` keeps
        # only the survivors, and a plot of those alone cannot show what is
        # missing.
        axis_deg = math.degrees(math.atan2(
            panel["landmark"][0] - panel["target"][0],
            panel["landmark"][1] - panel["target"][1]))
        for spot, _r, _az in lattice(centre, axis_deg, SPACING, R_MIN, R_MAX,
                                     panel["span"]):
            ax.plot(spot[0], spot[1], "o", ms=2.5, mfc="none", mec="0.75",
                    alpha=0.6, lw=0.5, zorder=1)
        if len(panel["kept"]):
            ax.plot(panel["kept"][:, 0], panel["kept"][:, 1], "o", ms=4,
                    color="tab:blue", zorder=3, lw=0)
            # THOR yaw is degrees clockwise from +z, so the heading is
            # (sin, cos) -- not the (cos, sin) a maths convention would give.
            yaw = np.radians(panel["kept"][:, 2])
            ax.quiver(panel["kept"][:, 0], panel["kept"][:, 1],
                      np.sin(yaw) * ARROW, np.cos(yaw) * ARROW,
                      angles="xy", scale_units="xy", scale=1.0, width=0.005,
                      color="tab:blue", alpha=0.55, zorder=2)

        # The pair's axis, its two ends, and a ring on the NEAR one.
        ax.plot([panel["target"][0], panel["landmark"][0]],
                [panel["target"][1], panel["landmark"][1]], "-",
                color="tab:red", lw=1.2, zorder=4)
        for point in (panel["target"], panel["landmark"]):
            ax.plot(point[0], point[1], "*", color="tab:red", ms=16, zorder=5)
        ax.plot(panel["landmark"][0], panel["landmark"][1], "o", mfc="none",
                mec="tab:red", ms=22, mew=1.2, zorder=5)

        ax.plot(*panel["start"], "o", ms=5, color="tab:green", zorder=6)
        start_yaw = math.radians(panel["start_yaw"])
        ax.quiver(panel["start"][0], panel["start"][1],
                  math.sin(start_yaw) * ARROW * 1.8,
                  math.cos(start_yaw) * ARROW * 1.8,
                  angles="xy", scale_units="xy", scale=1.0, width=0.010,
                  color="tab:green", zorder=6)
        ax.add_patch(plt.Rectangle((WALL_MARGIN, WALL_MARGIN),
                                   ROOM - 2 * WALL_MARGIN,
                                   ROOM - 2 * WALL_MARGIN,
                                   fill=False, ls=":", ec="#999999"))
        ax.set_xlim(0, ROOM); ax.set_ylim(0, ROOM)
        ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        # `viewpoints.py`'s caption line: name the set that was generated.
        ax.set_title(
            f"case {panel['index']}   {len(panel['kept'])} poses\n"
            f"lattice {SPACING:g} m, r {R_MIN:g}..{R_MAX:g} m, "
            f"az +/- {panel['span'] / 2:g} deg\n"
            + "  ".join(f"{k} {v}" for k, v in panel["why"].items()),
            fontsize=7)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    _os.makedirs(_os.path.dirname(_os.path.abspath(args.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
