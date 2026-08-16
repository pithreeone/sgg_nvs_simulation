"""
probe_viewpoint.py -- is there a viewpoint from which the triplet IS extractable?

`eval_move.py` answers a policy question: does the heading the sweep chooses beat
a random one.  This answers the prior question, which no policy can exceed -- put
the robot at a FIXED set of azimuths around the landmark, score top-1 at each,
and report both the per-angle rate and the any-angle ceiling.

Two things it settles.  If the ceiling is barely above the start, motion is done
and the effort belongs in the stop rule or the dataset.  If some angle is
systematically better, the bearing policy has a target to aim at rather than a
vote to average.

The arc is the SAME one `eval_move.walk` flies -- about P-hat, at
`--radius-scale` times the current radius -- so an angle that scores here is an
angle the robot could actually reach.

    python probe_viewpoint.py --cases nvs_pilot/cases/cases_hard.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: Azimuths tried, degrees.  0 is the start pose and is the baseline every other
#: column is read against.  Beyond about 45 degrees the arc leaves the room on
#: these scenes, so the sweep is deliberately not wider than the geometry allows.
ANGLES = (-45.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 45.0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="nvs_pilot/cases/cases_hard.json")
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--condition", type=int, default=10)
    ap.add_argument("--radius-scale", type=float, default=1.2)
    ap.add_argument("--angles", type=float, nargs="+", default=list(ANGLES))
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--max-az", type=float, default=30.0)
    ap.add_argument("--max-el", type=float, default=15.0)
    ap.add_argument("--trust", type=int, default=10)
    # `perceive` reads it; this probe never uses the heading it produces, only
    # the P-hat that comes with it.
    ap.add_argument("--bearing", default="bin")
    ap.add_argument("--cand-nms", type=float, default=0.0)
    ap.add_argument("--save-trail", default=None)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--out", default="nvs_pilot/probe_viewpoint.json")
    args = ap.parse_args(argv)

    from eval_move import look, perceive, walk
    from fuse_live import task_for
    from robot.proc_scene import Robot, open_room, rebuild
    from robot.sgg_live import load_egtr

    cases = json.load(open(args.cases))["cases"]
    if args.n:
        cases = cases[:args.n]
    egtr = load_egtr()
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        controller = open_room(args.width, args.height, args.fov)
        try:
            rc = Robot(controller, rebuild(controller, case))
            task = task_for(case)
            start_pose = {"position": dict(rc.agent_position),
                          "yaw": rc.agent_yaw, "horizon": rc.camera_horizon}
            # P-hat once, from the start pose, exactly as an episode does it.
            seed = perceive(rc, case, task, egtr, args)
            if seed is None:
                print(f"  [{index}] {case['scene']} unreadable", flush=True)
                continue
            centre = seed["centre"]
            row = {"scene": case["scene"], "instruction": case["instruction"],
                   "ranks": {}}
            for az in args.angles:
                rc.teleport(position=start_pose["position"],
                            yaw=start_pose["yaw"],
                            horizon=start_pose["horizon"])
                # TWO REASONS a pose yields nothing, and they must not share a
                # value.  `walk` refuses the pose (geometry), or the robot got
                # there and the instructed pair reached no rank at all
                # (perception).  Writing both as None made every "unreachable"
                # count in this file's own summary unreadable, and made a
                # policy that skips Nones look like it was avoiding bad
                # geometry when it was really reading the answer.
                if az and walk(rc, centre, az, args) is None:
                    row["ranks"][str(az)] = "unreachable"
                    continue
                seen = look(rc, task, egtr, args)
                seen.pop("frame", None)
                row["ranks"][str(az)] = seen.get("recall")   # int, or None
                # THE TOP-1'S CONFIDENCE AT A REAL POSE, beside its correctness,
                # which is what a stopping threshold would have to be cut from.
                row.setdefault("score", {})[str(az)] = seen.get("top1_score")
                row.setdefault("margin", {})[str(az)] = seen.get("top1_margin")
            rows.append(row)
            got = row["ranks"]
            print(f"  [{index}/{len(cases)}] {case['scene']:13} "
                  + " ".join(f"{a:+.0f}:{got[str(a)]}" for a in args.angles),
                  flush=True)
        except Exception as error:                              # noqa: BLE001
            print(f"  [{index}] ! {type(error).__name__}: {error}", flush=True)
        finally:
            controller.stop()
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)),
                        exist_ok=True)
            json.dump({"angles": args.angles, "cases": rows},
                      open(args.out, "w"), indent=1)

    if not rows:
        return 1
    n = len(rows)
    print(f"\n  {n} cases.  top-1 correct (rank 1) at each azimuth:\n")
    print(f"  {'azimuth':>9} {'top-1':>9} {'unreachable':>13} {'no rank':>13}")
    for az in args.angles:
        got = [r["ranks"][str(az)] for r in rows]
        hit = sum(1 for v in got if v == 1)
        dead = sum(1 for v in got if v == "unreachable")
        blind = sum(1 for v in got if v is None)
        print(f"  {az:>+9.0f} {hit:>6}/{n:<3} {dead:>12} {blind:>13}")
    ceiling = sum(1 for r in rows
                  if any(v == 1 for v in r["ranks"].values()))
    print(f"\n  `unreachable` = walk refused the pose; `no rank` = the robot got"
          f"\n  there and the instructed pair reached no rank at all.")
    start = sum(1 for r in rows if r["ranks"]["0.0"] == 1)
    moved = sum(1 for r in rows
                if any(v == 1 for a, v in r["ranks"].items()
                       if float(a) != 0.0))
    print(f"\n  start pose alone            {start:>3}/{n}")
    print(f"  SOME non-zero angle works   {moved:>3}/{n}")
    print(f"  ANY angle works (ceiling)   {ceiling:>3}/{n}")
    print(f"\n  The ceiling is what no bearing policy and no stop rule can "
          f"exceed.\n  `start` is what standing still already gets.")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
