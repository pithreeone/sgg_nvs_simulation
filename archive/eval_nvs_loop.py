"""
eval_nvs_loop.py -- keep asking.  If the robot walks there and still cannot see
the triplet, sweep again from where it now stands.

`eval_nvs_pointer.py` takes one step and stops.  This iterates:

    pose_0 = the start
    for k in 0..K-1:
        frame  = the robot's own camera, here
        graded = does this ONE REAL FRAME contain the instructed triplet,
                 grounded on the instructed instance?      -> yes: done at step k
        sweep  = 20 oracle-NVS views centred on pose_k
        pose_{k+1} = the pose of the best recovering view, projected onto the
                     floor;  or, if no view recovers it, a random step

The arrival frame is BOTH things at once: the success test for the step that
produced it, and the reference the next sweep orbits.  So an iteration costs one
extra sweep and nothing else -- no re-render, no re-measure.

Why iterate rather than widen.  Measured on the same ten tasks, going from the
NVS pipeline's own +-10 deg cone to +-30/15 doubled how often SOME novel view
recovered the triplet (2/8 -> 4/8 of the reference-miss cases).  So the
recovering viewpoint is usually OUTSIDE the cone a narrow-baseline generator can
synthesise.  Widening the generator is not the fix -- `0806_progress.md`
measures its geometric fidelity falling from 0.36 to 0.28 rigidity when the
baseline is widened, and every realizable fusion rule scoring the same on both.
Three iterations of a narrow cone reach the same viewpoints while every single
generation stays inside what the model is good at.  That is the claim this file
exists to test.

Three arms, one grader, the same budget:

  stay      step 0 only.  The no-NVS floor, and the shared origin of the others.
  random    K steps, random bearing each time, re-aiming at the target after
            each one.
  nvs       K steps, guided by the sweep, FALLING BACK to a random step when no
            view recovers the triplet.

The fallback matters twice over.  It keeps the budgets equal -- an arm that
stops early is not comparable to one that keeps moving -- and it makes the
degenerate case exact: both arms draw from the SAME bearing sequence, so if
every sweep comes up empty the `nvs` trajectory is not merely similar to
`random`, it is identical.  Any difference between the two arms is then
attributable to the guided steps alone, which is why each step records whether
it was `pointer` or `random` and why a success is credited to NVS only when the
step that produced it was guided.

Elevation is discarded, deliberately.  A lemniscate pose is 3D and the base is
not: only (x, z) is executed, the body never changes height (crouching
overshoots a +-15 deg request fourfold -- see `eval_nvs_pointer`), and the
camera ends 0.18-0.37 m below the view that earned the pointer.  Synthesising a
viewpoint the robot cannot occupy is a capability, not a defect; what this
measures is whether the AZIMUTH component alone still carries the recovery.

    python eval_nvs_loop.py --cases results/cases10.json \
        --max-az 30 --max-el 15 --steps 3 --out results/loop_k10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from robot.world import proc_scene
from robot.task.measure import measure, pose_of
from eval_nvs_pointer import (DEFAULT_STEP, contact_sheet, geometry, occ,
                              regrade, sweep_at, walk)
from robot.task.task_find import build_tasks, draw, put_in_front
from vg.vg150 import THOR_TO_VG150

#: How close a candidate pose may come to one the robot has already stood in
#: before it stops counting as somewhere new.  Without this the loop oscillates:
#: the cone centred on pose_1 contains the direction of pose_0, and pose_0's
#: viewpoint may well rank highly -- it is only known to be a failure because
#: the robot already went and looked.
VISITED_RADIUS = 0.20


def pick_pointer(rc, views: Sequence[Dict[str, Any]],
                 visited: Sequence[np.ndarray]) -> Optional[Dict[str, Any]]:
    """
    The best-ranked recovering view that is somewhere new and reachable.

    Falling through to the next-best rather than giving up matters: the views
    that recover a target most convincingly are the ones with the most
    elevation, and those are exactly the ones whose ground footprint can land on
    a tabletop.
    """
    from eval_nvs_pointer import MAX_SNAP

    recovered = [v for v in views if v["result"]["grounded_rank"]]
    for view in sorted(recovered, key=lambda v: v["result"]["grounded_rank"]):
        xz = np.array([view["pose"]["position"]["x"],
                       view["pose"]["position"]["z"]])
        if any(float(np.linalg.norm(xz - seen)) < VISITED_RADIUS
               for seen in visited):
            continue
        spot = rc.nearest_reachable(xz)
        if spot is not None and float(np.linalg.norm(spot - xz)) <= MAX_SNAP:
            return view
    return None


def step_length(views: Sequence[Dict[str, Any]], origin: np.ndarray) -> float:
    """
    How far a random step goes: the median ground displacement of the sweep.

    Taken from the whole sweep rather than from whichever view the pointer
    happened to pick, so the control's stride is a property of the trajectory
    geometry and not of the guided arm's choices.
    """
    offsets = [float(np.linalg.norm(
        np.array([v["pose"]["position"]["x"], v["pose"]["position"]["z"]])
        - origin)) for v in views]
    return max(float(np.median(offsets)), DEFAULT_STEP)


def bearing_step(origin: np.ndarray, bearing: float, distance: float) -> np.ndarray:
    return origin + distance * np.array([math.sin(bearing), math.cos(bearing)])


#: How many times a step may be re-aimed when THOR refuses the move.  A random
#: bearing can point into a wall, and the first version treated that as the end
#: of the rollout -- the arm stopped walking with most of its budget unspent.
#: Measured on the K=10 run: 5 of 58 cases ended early this way, and in one the
#: control stopped at step 1 while the guided arm went on, which is not a
#: comparison.  Retries draw from a per-arm stream so the SHARED bearings, and
#: with them the exact `nvs == random` identity when every sweep is empty, are
#: untouched on every case that does not need one.
MOVE_RETRIES = 6


def rollout(rc, task: Dict[str, Any], names: Sequence[str],
            target_xyz: np.ndarray, start_pose: Dict[str, Any],
            start_step: Dict[str, Any], guided: bool,
            bearings: Sequence[float], stride: Optional[float],
            retry: random.Random,
            args, egtr, predict, cv2, outdir: Optional[str]
            ) -> Dict[str, Any]:
    """
    One policy, from the start pose, for at most `args.steps` moves.

    `start_step` is the already-graded step 0 -- both arms share it, so neither
    pays for it twice and both are measured against the same origin.
    """
    rc.teleport(position=start_pose["position"], yaw=start_pose["yaw"],
                horizon=start_pose["horizon"],
                standing=start_pose["body"] == "stand")
    origin = np.array([start_pose["position"]["x"], start_pose["position"]["z"]])
    here = origin
    visited = [origin.copy()]
    trail = [start_step]
    sweeps: List[Dict[str, Any]] = []

    for k in range(args.steps):
        if trail[-1]["result"]["grounded_rank"]:
            break                              # already answered; stop walking

        source, pointer = "random", None
        if guided:
            views, present, _ = sweep_at(
                rc, task, names, target_xyz, egtr, predict, args.views,
                args.max_az, args.max_el, args.radius, args.fov, args.topk,
                args.iou, args.amodal)
            pointer = pick_pointer(rc, views, visited)
            if stride is None:
                stride = step_length(views, here)
            recovered = sum(bool(v["result"]["grounded_rank"]) for v in views)
            sweeps.append({"step": k, "recovered": recovered,
                           "pointer": pointer["view"] if pointer else None,
                           "views": views})
            if outdir:
                os.makedirs(outdir, exist_ok=True)
                cv2.imwrite(os.path.join(outdir, f"sweep_{k}.png"),
                            contact_sheet(cv2, trail[-1]["frame"],
                                          [p["frame"] for p in present], views,
                                          pointer["view"] if pointer else None))
            # The sweep parked the robot across the room; go back before moving.
            rc.teleport(position={"x": float(here[0]),
                                  "y": float(start_pose["position"]["y"]),
                                  "z": float(here[1])},
                        yaw=trail[-1]["pose"]["yaw"],
                        horizon=trail[-1]["pose"]["horizon"], standing=True)

        if pointer is not None:
            goal = np.array([pointer["pose"]["position"]["x"],
                             pointer["pose"]["position"]["z"]])
            source = "pointer"
        else:
            goal = bearing_step(here, bearings[k], stride or DEFAULT_STEP)

        moved = walk(rc, goal, task, names, target_xyz, egtr, predict,
                     args.topk, args.iou, here, args.amodal)
        # THOR refused the pose -- a wall, or a viewpoint over furniture.  Aim
        # somewhere else rather than abandoning the remaining budget.
        for _ in range(MOVE_RETRIES if moved is None else 0):
            source = "random"
            goal = bearing_step(here, retry.uniform(0.0, 2.0 * math.pi),
                                stride or DEFAULT_STEP)
            moved = walk(rc, goal, task, names, target_xyz, egtr, predict,
                         args.topk, args.iou, here, args.amodal)
            if moved is not None:
                break
        if moved is None:
            print(f"      step {k + 1} unreachable after "
                  f"{MOVE_RETRIES} re-aims; stopping")
            break
        guided_step = source == "pointer" and pointer is not None
        moved.update(step=k + 1, source=source,
                     view=pointer["view"] if guided_step else None,
                     elevation_lost=(pointer["pose"]["elevation"]
                                     if guided_step else 0.0))
        here = np.array([moved["pose"]["position"]["x"],
                         moved["pose"]["position"]["z"]])
        visited.append(here.copy())
        trail.append(moved)
        print(f"      step {k + 1} {source:<8} moved {moved['displacement']:.2f} m  "
              f"occ {occ(moved['occlusion'])}  "
              f"grounded {moved['result']['grounded_rank']}")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        for entry in trail:
            stem = os.path.join(outdir,
                                f"step_{entry['step']}_{entry['source']}")
            frame = entry.pop("frame", None)
            if frame is not None:
                cv2.imwrite(stem + ".png", frame[:, :, ::-1])
            drawn = entry.pop("triplet", None)
            if drawn is not None:
                cv2.imwrite(stem + "_triplet.png", drawn)
    for entry in trail:
        entry.pop("frame", None)
        entry.pop("triplet", None)

    won = next((e for e in trail if e["result"]["grounded_rank"]), None)
    return {"trail": trail, "sweeps": sweeps, "stride": stride,
            "success_step": won["step"] if won else None,
            "success_source": won["source"] if won else None,
            "metres": sum(e.get("displacement", 0.0) for e in trail)}


def run_case(case: Dict[str, Any], args, egtr, predict, cv2,
             rng: random.Random) -> Optional[Dict[str, Any]]:
    rc = proc_scene.open_scene(case["scene"], args.width, args.height, args.fov,
                          case["start"])
    try:
        # Two lists, and the difference matters.  Tasks are built only from what
        # VG150 can name, but the AMODAL reference has to disable everything
        # that could be in the way -- and a staged occluder need not be
        # nameable.  Measured on FloorPlan230, a `Statue` (no VG150 class) was
        # left standing through the amodal sweep, so the newspaper's "unoccluded"
        # reference was its OCCLUDED pixels and the run reported 0.00 occlusion
        # for a target `put_in_front` had just measured as 54% hidden.
        nameable = sorted(o["name"] for o in rc.event.metadata["objects"]
                          if o["objectType"] in THOR_TO_VG150)
        names = sorted({o["name"] for o in rc.event.metadata["objects"]
                        if o["objectType"] in THOR_TO_VG150
                        or o.get("moveable") or o.get("pickupable")})
        state = measure(rc, nameable, args.fov)
        entries = {o["name"]: o for o in rc.event.metadata["objects"]}
        task = next((t for t in build_tasks(state["objects"], entries)
                     if t["target_name"] == case["target_name"]), None)
        if task is None:
            print(f"  ! {case['target_name']} no longer supports an "
                  f"unambiguous instruction here")
            return None
        print(f"  {task['instruction']}   target {task['target_name']}")
        staged = put_in_front(rc, task["target_name"], case["occluder_type"],
                                 task["receptacle_name"])
        if staged is None:
            return None

        start_pose = pose_of(rc)
        target_entry = next(o for o in rc.event.metadata["objects"]
                            if o["name"] == task["target_name"])
        target_xyz = np.array([target_entry["position"]["x"],
                               target_entry["position"]["y"],
                               target_entry["position"]["z"]])

        # Step 0, graded once and handed to both arms.
        geo = geometry(rc, [task["target_name"]]
                       + [d["name"] for d in task.get("distractors") or []],
                       names, args.amodal)
        start_triplets = predict(egtr, rc.event.frame, args.topk)
        here, result = regrade(task, geo, start_triplets, args.iou)
        start_step = {"step": 0, "source": "start", "pose": start_pose,
                      "occlusion": here["target_occlusion"], "result": result,
                      "displacement": 0.0, "frame": rc.event.frame,
                      "triplet": draw(rc.event.frame, here, result,
                                      start_triplets)}
        print(f"    step 0 stay     occ {occ(here['target_occlusion'])}  "
              f"class {result['class_rank']}  grounded {result['grounded_rank']}")

        # One bearing sequence, drawn once and used by BOTH arms, so an
        # all-empty guided run reproduces the random run exactly.
        bearings = [rng.uniform(0.0, 2.0 * math.pi) for _ in range(args.steps)]
        retries = {arm: random.Random(f"{case['scene']}/{case['target_name']}/{arm}")
                   for arm in ("nvs", "random")}
        outdir = os.path.join(args.out, f"{case['scene']}_{task['target_name']}")

        print("    -- nvs")
        nvs = rollout(rc, task, names, target_xyz, start_pose,
                      dict(start_step), True, bearings, None, retries["nvs"],
                      args, egtr, predict, cv2, os.path.join(outdir, "nvs"))
        print("    -- random")
        control = rollout(rc, task, names, target_xyz, start_pose,
                          dict(start_step), False, bearings, nvs["stride"],
                          retries["random"], args, egtr, predict, cv2,
                          os.path.join(outdir, "random"))

        start_step.pop("frame", None)
        start_step.pop("triplet", None)
        return {"case": case, "instruction": task["instruction"],
                "target": task["target_name"], "staged": staged,
                "start": start_step, "nvs": nvs, "random": control,
                "outdir": outdir}
    finally:
        rc.stop()


def solved_by(arm: Dict[str, Any], k: int, guided_only: bool = False) -> bool:
    """Did this arm answer the instruction within `k` moves?"""
    step = arm["success_step"]
    if step is None or step > k:
        return False
    return not guided_only or arm["success_source"] == "pointer"


def table(rows: Sequence[Dict[str, Any]], steps: int) -> str:
    header = (f"{'policy':<16}{'n':>4}"
              + "".join(f"{'@' + str(k):>8}" for k in range(steps + 1))
              + f"{'metres':>9}")
    lines = [header]
    for label, key, guided in (("stay", None, False), ("random", "random", False),
                               ("nvs", "nvs", False),
                               ("nvs (guided win)", "nvs", True)):
        cells = ""
        for k in range(steps + 1):
            if key is None:
                got = sum(bool(r["start"]["result"]["grounded_rank"]) for r in rows)
            else:
                got = sum(solved_by(r[key], k, guided) for r in rows)
            cells += f"{str(got) + '/' + str(len(rows)):>8}"
        metres = (0.0 if key is None
                  else sum(r[key]["metres"] for r in rows) / max(len(rows), 1))
        lines.append(f"{label:<16}{len(rows):>4}{cells}{metres:>9.2f}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="results/cases10.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3, help="the move budget K")
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--max-az", type=float, default=30.0)
    ap.add_argument("--max-el", type=float, default=15.0)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--amodal", action="store_true",
                    help="also measure each target's unoccluded extent; costs "
                         "~2N THOR steps per frame and changes no verdict")
    ap.add_argument("--sgg-root", default=None)
    ap.add_argument("--out", default="results/loop_k10")
    args = ap.parse_args(argv)

    import cv2

    from robot.sgg_live import SGG_ROOT, load_egtr, predict

    with open(args.cases) as fh:
        cases = json.load(fh)["cases"]
    if args.limit:
        cases = cases[:args.limit]

    egtr = load_egtr(args.sgg_root or SGG_ROOT)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for index, case in enumerate(cases):
        print(f"\n[{index + 1}/{len(cases)}] {case['scene']}  "
              f"{case['instruction']}")
        try:
            row = run_case(case, args, egtr, predict, cv2,
                           random.Random(args.seed * 1000 + index))
        except Exception as error:                       # noqa: BLE001
            print(f"  ! {type(error).__name__}: {error}")
            row = None
        if row:
            rows.append(row)

    print(f"\n{len(rows)}/{len(cases)} cases completed\n\n"
          + table(rows, args.steps))
    hard = [r for r in rows if not r["start"]["result"]["grounded_rank"]]
    if hard:
        print(f"\nstratum M -- the reference MISSED it ({len(hard)} cases)\n"
              + table(hard, args.steps))

    # How often the guided arm had nothing to point at, which is how often it
    # was really just walking randomly.
    fell_back = sum(s["pointer"] is None
                    for r in rows for s in r["nvs"]["sweeps"])
    total = sum(len(r["nvs"]["sweeps"]) for r in rows)
    print(f"\nsweeps with no recovering view (nvs fell back to random): "
          f"{fell_back}/{total}")

    def plain(value):
        """Anything that slipped through as an array is a rendering, not data.

        A leaked frame used to abort the dump AFTER every case had run, losing
        an hour and a half of measurement to a serialisation error.  Degrading
        it to a note keeps the run.
        """
        if isinstance(value, np.ndarray):
            return f"<ndarray {value.shape}>"
        raise TypeError(f"not JSON serialisable: {type(value).__name__}")

    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump({"steps": args.steps, "views": args.views,
                   "max_az": args.max_az, "max_el": args.max_el,
                   "iou": args.iou, "topk": args.topk, "seed": args.seed,
                   "rows": rows}, fh, indent=1, default=plain)
    print(f"wrote {os.path.join(args.out, 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
