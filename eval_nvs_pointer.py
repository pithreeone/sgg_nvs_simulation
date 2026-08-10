"""
eval_nvs_pointer.py -- does an oracle NVS sweep tell the robot WHERE TO STAND?

The question is not whether a synthesised view can see the triplet.  Success is
defined here as a SINGLE REAL FRAME, taken at a pose the robot actually stands
in, containing the instructed triplet grounded on the instructed instance.  A
novel view therefore never answers anything: its only job is to point.

Three policies, one grader, one scene each:

  stay      do not move.  This is the no-NVS lower bound and is literally the
            reference frame's own result.
  random    move the SAME DISTANCE the NVS policy would have moved, in a random
            direction, then re-aim at the target.  This is the control that
            matters.  `moves/fp203_left` showed occlusion falling 0.74 -> 0.49
            over three left steps, so in a cluttered room almost any sideways
            move improves the view -- without this control the table measures
            "moving helps", not "the NVS suggestion helps".
  nvs       render the lemniscate with THOR, grade every view, walk to the pose
            of the view that recovered the triplet.  If no view recovers it,
            the policy does not move and scores whatever `stay` scored.

Both moving policies are handed the target's position to AIM at.  That is a
shared oracle and it is deliberate: the comparison isolates where to STAND,
which is the only thing NVS is being asked for.  Re-aiming is also not optional
-- `drive_triplet_scene.py` records that reusing a recorded camera pitch points
the agent at the floorboards.

Why the NVS is rendered rather than generated.  Stable Virtual Camera's frames
have no THOR geometry, so the target's box in one is undefined and a grounded
hit cannot be scored there; and `0806_progress.md` measures their rigidity at
0.28-0.36, so it should not be assumed either.  Rendering the same lemniscate
poses in THOR gives an ORACLE NVS: an upper bound on what a perfect generative
model could point at.  A null result here is therefore informative -- it says
the recovering viewpoint is not in the +-10 deg cone at all, which no
improvement to the generator could fix.

Everything is re-measured at every pose.  Boxes, occlusion and the graph belong
to the frame in front of the robot; nothing is carried from the reference.

    python find_cases.py --n 20 --out nvs_pilot/cases.json
    python eval_nvs_pointer.py --cases nvs_pilot/cases.json --out nvs_pilot
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from robot import drive
from robot.drive_triplet_scene import agent_masks, measure, pose_of
from robot.nvs_lemniscate import camera_for, lemniscate, look_at_point, park_once, sweep
from robot.robot_controller import horizon_towards, yaw_towards
from robot.task_find import build_tasks, draw, grade, put_in_front
from vg.vg150 import THOR_TO_VG150

#: The body height NEVER changes when projecting a lemniscate pose, and the
#: measurement that settled it: crouching to serve a -5.9 deg view took
#: FloorPlan201's book from 0.34 occlusion to 0.85 and destroyed a class hit at
#: rank 2, while the random control moved the same 0.25 m standing and went to
#: 0.19 / rank 4.  The arithmetic says why.  At a 1.5 m standoff -5.9 deg asks
#: the camera to drop 0.15 m; the agent's only downward option is Stand ->
#: Crouch, which drops it 0.63 m (1.576 -> 0.95, TASKS.md).  That is a 4x
#: overshoot, and inside a +-10 deg cone every elevation request is this small.
#: Crouching is therefore never the right projection here -- it would need a
#: cone wide enough to ask for a ~0.6 m drop, i.e. past +-20 deg.
CROUCH_IS_NEVER_RIGHT = True

#: Fallback displacement for the random control on cases where the NVS policy
#: does not move.  0.30 m is what +10 deg of orbit azimuth costs at the 1.7 m
#: standoff measured on FloorPlan203, i.e. the whole width of the NVS cone.
DEFAULT_STEP = 0.30



def occ(value: Optional[float]) -> str:
    """Occlusion for a printout.  `--amodal` off means there is no denominator."""
    return "  n/a" if value is None else f"{value:5.2f}"

def visible_only(rc, wanted: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """
    Just the boxes the camera actually draws.  No clearing, no extra renders.

    This is the fast path and, measured, it is also the whole of the grading
    signal.  Over twelve staged frames EGTR's box agreed with the target's
    VISIBLE extent 6 times and with its AMODAL extent 0 times (6 ties, all at
    low occlusion), and the agreement gap widens with occlusion -- 0.41 amodal
    vs 0.89 visible on a half-hidden lamp.  So `max(amodal, visible)` was
    always just `visible`: dropping the amodal branch changed the hit count
    from 10 to 10.  `0806_progress.md` reports the same thing independently
    ("bbox_visible is the evaluator's box ... amodal checks are systematically
    pessimistic for occluded objects").

    Clearing the scene is also what the pipeline was actually spending its time
    on: `DisableObject`/`EnableObject` are one THOR step EACH, and a living room
    holds 30-60 nameable or moveable objects, so an amodal reference cost ~2N
    steps against the 20 renders of the sweep itself.
    """
    names_by_id = {o["objectId"]: o["name"]
                   for o in rc.event.metadata["objects"]}
    seen = agent_masks(rc.event, names_by_id)
    out = {}
    for name in wanted:
        seen_px, seen_box = seen.get(name, (0, None))
        out[name] = {"bbox_amodal": None, "bbox_visible": seen_box,
                     "reference_px": None, "visible_px": seen_px,
                     "occlusion": None}
    return out


def geometry(rc, wanted: Sequence[str], names: Sequence[str],
             amodal: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Amodal and visible boxes of `wanted`, at the pose the robot is standing in.

    With `amodal=False` this is `visible_only` -- see there for why that is the
    default in the runners.

    The same protocol as `drive_triplet_scene.measure` -- pause physics, disable
    everything nameable, enable one object at a time -- but for a named few
    rather than for the whole scene, and WITHOUT that function's
    `MAX_OCCLUSION` filter.  The filter is why this exists: a staged target sits
    at 0.7-0.8 occlusion, past the ceiling, so `measure` drops it and there is
    nothing left to grade.  A dropped target is the outcome under test, not a
    reason to lose the row.

    Pausing is not optional; `measure` documents the failure it prevents (12 of
    15 objects come back with no instance mask on the `EnableObject` event).
    Ids are read fresh from the current event because staging goes through
    `SetObjectPoses`, which renumbers every objectId.
    """
    if not amodal:
        return visible_only(rc, wanted)

    entries = {o["name"]: o for o in rc.event.metadata["objects"]}
    names_by_id = {o["objectId"]: o["name"] for o in rc.event.metadata["objects"]}
    seen = agent_masks(rc.event, names_by_id)

    ids = {n: entries[n]["objectId"] for n in names if n in entries}
    alone: Dict[str, Any] = {}
    rc.controller.step(action="PausePhysicsAutoSim")
    try:
        for object_id in ids.values():
            rc.controller.step(action="DisableObject", objectId=object_id)
        for name in wanted:
            if name not in ids:
                continue
            event = rc.controller.step(action="EnableObject",
                                       objectId=ids[name])
            alone[name] = agent_masks(event, names_by_id).get(name)
            rc.controller.step(action="DisableObject", objectId=ids[name])
        for object_id in ids.values():
            rc.controller.step(action="EnableObject", objectId=object_id)
    finally:
        rc.controller.step(action="UnpausePhysicsAutoSim")
    # A no-op step to get a frame rendered with every object back in the scene.
    # `measure` documents that a just-re-enabled object is missing from the
    # segmentation of that same event; the frame EGTR is asked about must not be
    # one where half the room has yet to reappear.
    rc.controller.step(action="Done")
    rc.event = rc.controller.last_event

    out = {}
    for name in wanted:
        reference = alone.get(name)
        ref_px, ref_box = reference if reference else (0, None)
        seen_px, seen_box = seen.get(name, (0, None))
        out[name] = {
            "bbox_amodal": ref_box,
            "bbox_visible": seen_box,
            "reference_px": ref_px,
            "visible_px": seen_px,
            "occlusion": round(max(0.0, 1.0 - seen_px / ref_px), 4)
                         if ref_px else 1.0,
        }
    return out


def regrade(task: Dict[str, Any], geo: Dict[str, Dict[str, Any]],
            triplets: Sequence[Dict[str, Any]], threshold: float
            ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Grade `task` against boxes measured at the CURRENT pose, not the reference."""
    here = dict(task)
    target = geo.get(task["target_name"], {})
    here["target_box_amodal"] = target.get("bbox_amodal")
    here["target_box_visible"] = target.get("bbox_visible")
    here["target_occlusion"] = target.get("occlusion", 1.0)
    here["distractors"] = [
        {**d, "bbox_amodal": geo.get(d["name"], {}).get("bbox_amodal"),
         "bbox_visible": geo.get(d["name"], {}).get("bbox_visible")}
        for d in (task.get("distractors") or [])]
    return here, grade(here, triplets, [], threshold, False)


def stand_at(rc, xz: np.ndarray, target_xyz: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Put the robot at `xz`, looking at the target.  Exact first, navmesh second.

    Snapping to `GetReachablePositions` was destroying the signal this
    experiment exists to measure.  That grid is 0.25 m, and +-10 deg of orbit
    at a 1.5 m standoff is a 0.26 m sidestep -- the whole NVS cone is about one
    grid cell wide, so the nearest reachable point was usually the one the robot
    was already standing on.  Measured on the first 20-case run: 6 of the 9
    cases that had a pointer moved 0.00 m.

    `snapToGrid` is off (see `drive.py`), so THOR accepts an arbitrary position
    and reports whether it was legal.  Trying the exact spot first and keeping
    the navmesh only as the fallback preserves sub-grid moves without letting
    the agent be teleported into a wall.
    """
    xz = np.asarray(xz, dtype=float)
    # Standing is not a choice here; see CROUCH_IS_NEVER_RIGHT.
    rc.set_height_level("stand")
    y = float(rc.agent_position["y"])
    yaw = yaw_towards(xz, np.array([target_xyz[0], target_xyz[2]]))

    landed, snapped = None, False
    if rc.teleport(position={"x": float(xz[0]), "y": y, "z": float(xz[1])},
                   yaw=yaw, horizon=0.0):
        landed = xz
    else:
        spot = rc.nearest_reachable(xz)
        if spot is None:
            return None
        yaw = yaw_towards(spot, np.array([target_xyz[0], target_xyz[2]]))
        if not rc.teleport(position={"x": float(spot[0]), "y": y,
                                     "z": float(spot[1])},
                           yaw=yaw, horizon=0.0):
            return None
        landed, snapped = spot, True

    horizon = horizon_towards(rc.camera_xyz, target_xyz)
    rc.teleport(yaw=yaw, horizon=horizon)
    return {"pose": pose_of(rc), "snapped": snapped,
            "snap_error": float(np.linalg.norm(landed - xz)),
            "requested_xz": [float(xz[0]), float(xz[1])]}


def walk(rc, xz: np.ndarray, task: Dict[str, Any],
         names: Sequence[str], target_xyz: np.ndarray, egtr, predict,
         topk: int, iou: float, home_xz: np.ndarray,
         amodal: bool = True) -> Optional[Dict[str, Any]]:
    """Move, re-measure, re-ask.  One policy step, from the robot's own camera."""
    placed = stand_at(rc, xz, target_xyz)
    if placed is None:
        return None
    geo = geometry(rc, [task["target_name"]]
                   + [d["name"] for d in task.get("distractors") or []], names,
                   amodal)
    triplets = predict(egtr, rc.event.frame, topk)
    here, result = regrade(task, geo, triplets, iou)
    position = placed["pose"]["position"]
    return {**placed,
            "displacement": float(math.dist(
                (position["x"], position["z"]), tuple(home_xz))),
            "occlusion": here["target_occlusion"],
            "visible_px": geo[task["target_name"]]["visible_px"],
            "result": result,
            "frame": rc.event.frame,
            "triplet": draw(rc.event.frame, here, result, triplets)}


#: How far the navmesh may drag a suggested viewpoint before the suggestion has
#: stopped being that viewpoint.  An orbit pose can land inside the table it is
#: orbiting, and `nearest_reachable` will happily hand back a spot on the far
#: side of it.
MAX_SNAP = 0.50


def pick_pointer(rc, recovered: Sequence[Dict[str, Any]]
                 ) -> Optional[Dict[str, Any]]:
    """
    The best-ranked recovering view the robot can actually go and stand at.

    Falling through to the next-best view rather than giving up matters: the
    lemniscate's strongest views are the ones with the most elevation, and those
    are exactly the ones whose footprint can sit over a tabletop.
    """
    for view in sorted(recovered, key=lambda v: v["result"]["grounded_rank"]):
        xz = np.array([view["pose"]["position"]["x"],
                       view["pose"]["position"]["z"]])
        spot = rc.nearest_reachable(xz)
        if spot is not None and float(np.linalg.norm(spot - xz)) <= MAX_SNAP:
            return view
    return None


def contact_sheet(cv2, ref_frame: np.ndarray, frames: Sequence[np.ndarray],
                  views: Sequence[Dict[str, Any]], pointer: Optional[int],
                  columns: int = 5, tile: Tuple[int, int] = (320, 240)
                  ) -> np.ndarray:
    """
    The robot's own frame and every synthesised view, on one sheet.

    Border colour is the verdict on that view -- green grounded, blue the class
    triple only, grey nothing -- so which viewpoints recover the instruction is
    readable without opening twenty files.  The pointer is the one drawn white.
    """
    bar, pad = 26, 4
    cell = (tile[0] + 2 * pad, tile[1] + bar + 2 * pad)
    rows = (len(frames) + 1 + columns - 1) // columns
    sheet = np.full((rows * cell[1], columns * cell[0], 3), 24, np.uint8)

    def place(index, frame, caption, colour, thick):
        row, column = divmod(index, columns)
        x, y = column * cell[0], row * cell[1]
        small = cv2.resize(frame[:, :, ::-1], tile)
        sheet[y + pad:y + pad + tile[1], x + pad:x + pad + tile[0]] = small
        cv2.rectangle(sheet, (x + pad - 2, y + pad - 2),
                      (x + pad + tile[0] + 1, y + pad + tile[1] + 1),
                      colour, thick)
        cv2.putText(sheet, caption, (x + pad + 2, y + pad + tile[1] + bar - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

    place(0, ref_frame, "REF  the robot's own frame", (200, 200, 200), 1)
    for index, (frame, view) in enumerate(zip(frames, views), start=1):
        result, pose = view["result"], view["pose"]
        colour = ((150, 255, 150) if result["grounded_rank"]
                  else (255, 170, 110) if result["class_rank"]
                  else (140, 140, 140))
        rank = (f"r{result['grounded_rank']}" if result["grounded_rank"]
                else f"(c{result['class_rank']})" if result["class_rank"]
                else "-")
        place(index, frame,
              f"v{view['view']:02d} az{pose['azimuth']:+.0f} "
              f"el{pose['elevation']:+.0f} occ{occ(view['occlusion'])} {rank}",
              (255, 255, 255) if view["view"] == pointer else colour,
              3 if view["view"] == pointer else 1)
    return sheet


def random_target(rng: random.Random, home_xz: np.ndarray,
                  distance: float) -> np.ndarray:
    bearing = rng.uniform(0.0, 2.0 * math.pi)
    return home_xz + distance * np.array([math.sin(bearing), math.cos(bearing)])


def sweep_at(rc, task: Dict[str, Any], names: Sequence[str],
             target_xyz: np.ndarray, egtr, predict, views_n: int,
             max_az: float, max_el: float, radius: Optional[float],
             fov: float, topk: int, iou: float, amodal: bool = True
             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    """
    The oracle NVS sweep from wherever the robot is standing right now.

    Leaves the robot PARKED where `park_once` put it -- the caller decides where
    to go next and must teleport out.  Returns the graded views, the raw sweep
    rows (for their frames), and the orbit radius that was used.

    Two passes are unavoidable.  The first renders the scene as it is, for the
    frames EGTR sees and the target's visible pixels; the second disables every
    nameable object and re-enables only the target, which is what makes the
    amodal box -- and hence the occlusion and the grading box -- well defined in
    a frame the agent camera never took.
    """
    camera = rc.camera_xyz
    radius = radius or float(np.linalg.norm(camera - target_xyz))
    centre = look_at_point(camera, rc.agent_yaw, rc.camera_horizon, radius)
    poses = [camera_for(centre, camera, az, el)
             for az, el in lemniscate(views_n, max_az, max_el)]
    reachable = rc.controller.step(
        action="GetReachablePositions").metadata["actionReturn"] or []
    # Park away from the REFERENCE camera, not from a lemniscate pose: the robot
    # casts a shadow and `park_once` exists to keep that shadow identical across
    # every frame the sweep compares.
    park_once(rc, {"x": float(camera[0]), "y": float(camera[1]),
                   "z": float(camera[2])}, reachable)

    present = sweep(rc, poses, task["target_name"], fov, reachable,
                    keep_frames=True)
    alone: List[Optional[Dict[str, Any]]] = [None] * len(present)
    if amodal:
        # Re-read ids HERE: staging goes through `SetObjectPoses`, which
        # renumbers every objectId, and `DisableObject` on a stale id fails with
        # an EMPTY error message.
        current = {o["name"]: o
                   for o in rc.controller.last_event.metadata["objects"]}
        ids = {n: current[n]["objectId"] for n in names if n in current}
        rc.controller.step(action="PausePhysicsAutoSim")
        try:
            for object_id in ids.values():
                rc.controller.step(action="DisableObject", objectId=object_id)
            rc.controller.step(action="EnableObject",
                               objectId=ids[task["target_name"]])
            alone = sweep(rc, poses, task["target_name"], fov, reachable,
                          keep_frames=False)
            for object_id in ids.values():
                rc.controller.step(action="EnableObject", objectId=object_id)
        finally:
            rc.controller.step(action="UnpausePhysicsAutoSim")

    views = []
    for index, (seen, bare) in enumerate(zip(present, alone)):
        in_view = dict(task)
        if bare is None:
            in_view["target_occlusion"] = None
            in_view["target_box_amodal"] = None
        else:
            occlusion = (1.0 - seen["pixels"] / bare["pixels"]
                         if bare["pixels"] else 1.0)
            in_view["target_occlusion"] = round(max(0.0, occlusion), 3)
            in_view["target_box_amodal"] = bare["box"]
        in_view["target_box_visible"] = seen["box"]
        # Distractor boxes belong to the agent camera and are meaningless in a
        # third-party frame; the sweep is only ever asked to POINT, and
        # grounding on the target is the whole of that signal.
        in_view["distractors"] = []
        result = grade(in_view, predict(egtr, seen["frame"], topk), [], iou,
                       False)
        views.append({"view": index, "pose": seen["pose"],
                      "occlusion": in_view["target_occlusion"],
                      "visible_px": seen["pixels"],
                      # Kept so a `best_subject_iou` of 0.0 can be told apart
                      # from a missing box without re-running THOR.
                      "bbox_amodal": bare["box"] if bare else None,
                      "bbox_visible": seen["box"], "result": result})
    return views, present, radius


def run_case(case: Dict[str, Any], args, egtr, predict, cv2,
             rng: random.Random) -> Optional[Dict[str, Any]]:
    rc = drive.open_scene(case["scene"], args.width, args.height, args.fov,
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

        # The instruction is rebuilt at the reference pose BEFORE staging, so
        # the question asked is the one the case names.
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

        home = dict(rc.agent_position)
        home_xz = np.array([home["x"], home["z"]])
        reference_pose = pose_of(rc)
        target_entry = next(o for o in rc.event.metadata["objects"]
                            if o["name"] == task["target_name"])
        target_xyz = np.array([target_entry["position"]["x"],
                               target_entry["position"]["y"],
                               target_entry["position"]["z"]])

        # --- stay -------------------------------------------------------
        geo = geometry(rc, [task["target_name"]]
                       + [d["name"] for d in task.get("distractors") or []],
                       names, args.amodal)
        ref_triplets = predict(egtr, rc.event.frame, args.topk)
        here, ref_result = regrade(task, geo, ref_triplets, args.iou)
        ref_frame = rc.event.frame
        ref_drawn = draw(ref_frame, here, ref_result, ref_triplets)
        print(f"    stay      occ {occ(here['target_occlusion'])}  "
              f"class {ref_result['class_rank']}  "
              f"grounded {ref_result['grounded_rank']}")

        # --- the oracle NVS sweep ---------------------------------------
        views, present, radius = sweep_at(
            rc, task, names, target_xyz, egtr, predict, args.views,
            args.max_az, args.max_el, args.radius, args.fov, args.topk,
            args.iou, args.amodal)

        recovered = [v for v in views if v["result"]["grounded_rank"]]
        pointer = pick_pointer(rc, recovered)
        print(f"    sweep     recovered in {len(recovered)}/{len(views)} views"
              + (f"; pointing at view {pointer['view']:02d} "
                 f"(az {pointer['pose']['azimuth']:+.1f}, "
                 f"el {pointer['pose']['elevation']:+.1f}, "
                 f"rank {pointer['result']['grounded_rank']})" if pointer else ""))

        rc.teleport(position=reference_pose["position"],
                    yaw=reference_pose["yaw"], horizon=reference_pose["horizon"],
                    standing=reference_pose["body"] == "stand")

        # --- nvs-guided move --------------------------------------------
        nvs_move = None
        if pointer is not None:
            elevation = pointer["pose"]["elevation"]
            nvs_move = walk(rc,
                            np.array([pointer["pose"]["position"]["x"],
                                      pointer["pose"]["position"]["z"]]),
                            task, names, target_xyz,
                            egtr, predict, args.topk, args.iou, home_xz,
                            args.amodal)
            if nvs_move is not None:
                nvs_move["view"] = pointer["view"]
                nvs_move["elevation_requested"] = elevation
                # The whole elevation is lost, by design.  The body has no
                # continuous height and its one downward step overshoots a
                # +-10 deg request fourfold -- see CROUCH_IS_NEVER_RIGHT.  This
                # is the cost of executing an NVS pose on a real base, and it is
                # reported rather than hidden.
                nvs_move["elevation_lost"] = elevation
                print(f"    nvs       moved {nvs_move['displacement']:.2f} m  "
                      f"occ {occ(nvs_move['occlusion'])}  "
                      f"grounded {nvs_move['result']['grounded_rank']}")
            else:
                print(f"    nvs       view {pointer['view']:02d} has no "
                      f"reachable footprint")

        # --- random control ---------------------------------------------
        # Floored, not inherited.  On the first run the NVS arm's displacement
        # was 0.00 m on 6 of 9 cases, so `0.5 * distance` was 0 and the control
        # accepted standing still -- six rows of "random" that were a second
        # copy of `stay`.
        distance = max(nvs_move["displacement"] if nvs_move else 0.0,
                       DEFAULT_STEP)
        random_move = None
        for _ in range(args.random_tries):
            rc.teleport(position=reference_pose["position"],
                        yaw=reference_pose["yaw"],
                        horizon=reference_pose["horizon"],
                        standing=reference_pose["body"] == "stand")
            candidate = walk(rc, random_target(rng, home_xz, distance),
                             task, names, target_xyz, egtr, predict,
                             args.topk, args.iou, home_xz, args.amodal)
            # A bearing that snaps back onto the reference is not a move; retry
            # rather than credit the control with a free `stay`.
            if candidate and candidate["displacement"] >= 0.5 * distance:
                random_move = candidate
                break
        if random_move:
            print(f"    random    moved {random_move['displacement']:.2f} m  "
                  f"occ {occ(random_move['occlusion'])}  "
                  f"grounded {random_move['result']['grounded_rank']}")

        outdir = os.path.join(args.out, f"{case['scene']}_{task['target_name']}")
        os.makedirs(outdir, exist_ok=True)
        cv2.imwrite(os.path.join(outdir, "stay.png"), ref_frame[:, :, ::-1])
        cv2.imwrite(os.path.join(outdir, "stay_triplet.png"), ref_drawn)
        # The contact sheet carries every view already; the full-resolution
        # copies are 16 MB a case and only earn that when someone is going to
        # look at one closely.
        if args.save_views:
            os.makedirs(os.path.join(outdir, "views"), exist_ok=True)
            for view, seen in zip(views, present):
                cv2.imwrite(os.path.join(outdir, "views",
                                         f"view_{view['view']:02d}.png"),
                            seen["frame"][:, :, ::-1])
        cv2.imwrite(os.path.join(outdir, "sweep.png"),
                    contact_sheet(cv2, ref_frame,
                                  [s["frame"] for s in present], views,
                                  pointer["view"] if pointer else None))
        for name, move in (("nvs", nvs_move), ("random", random_move)):
            if move:
                cv2.imwrite(os.path.join(outdir, f"{name}.png"),
                            move.pop("frame")[:, :, ::-1])
                cv2.imwrite(os.path.join(outdir, f"{name}_triplet.png"),
                            move.pop("triplet"))

        return {"case": case, "instruction": task["instruction"],
                "target": task["target_name"], "staged": staged,
                "reference_pose": reference_pose,
                "orbit_radius": radius,
                "stay": {"occlusion": here["target_occlusion"],
                         "result": ref_result},
                "views": views,
                "pointer": pointer["view"] if pointer else None,
                "n_recovered_views": len(recovered),
                "nvs": nvs_move, "random": random_move, "outdir": outdir}
    finally:
        rc.stop()


def hit(entry: Optional[Dict[str, Any]], k: int) -> bool:
    return bool(entry and entry["result"]["grounded_rank"]
                and entry["result"]["grounded_rank"] <= k)


def table(rows: Sequence[Dict[str, Any]], ks: Sequence[int]) -> str:
    """One line per policy.  `nvs` is the POLICY: it stays when it has no pointer."""
    def arm(row, policy):
        if policy == "stay":
            return row["stay"]
        if policy == "random":
            return row["random"]
        return row["nvs"] or row["stay"]      # no pointer -> the policy stays

    lines = [f"{'policy':<10}{'n':>4}" + "".join(f"{'hit@'+str(k):>9}" for k in ks)
             + f"{'moved':>8}{'mean m':>9}"]
    for policy in ("stay", "random", "nvs"):
        arms = [arm(r, policy) for r in rows]
        alive = [a for a in arms if a]
        moved = [a for a in alive if a.get("displacement")]
        cells = "".join(f"{sum(hit(a, k) for a in alive):>4}/{len(arms):<4}"
                        for k in ks)
        mean = (sum(a["displacement"] for a in moved) / len(moved)) if moved else 0.0
        lines.append(f"{policy:<10}{len(arms):>4}{cells}{len(moved):>8}{mean:>9.2f}")

    # Not a policy: the share of cases where SOME novel view grounded the
    # triplet.  It bounds what any pointer could have achieved, and separates
    # "the recovering viewpoint is not in the cone" from "the pointer missed it".
    ceiling = sum(r["n_recovered_views"] > 0 for r in rows)
    lines.append(f"{'ceiling':<10}{len(rows):>4}   some novel view grounded it "
                 f"in {ceiling}/{len(rows)} cases")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="nvs_pilot/cases.json")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--max-az", type=float, default=10.0,
                    help="the NVS pipeline's own cone; run_nvs_occlusion.sh "
                         "uses --max_az 10 --max_el 10")
    ap.add_argument("--max-el", type=float, default=10.0)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--random-tries", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--amodal", action="store_true",
                    help="also measure each target's unoccluded extent, which "
                         "costs ~2N THOR steps per frame for N objects.  "
                         "Measured to change no verdict (see `visible_only`); "
                         "off by default")
    ap.add_argument("--save-views", action="store_true",
                    help="also write each synthesised view at full resolution "
                         "(~16 MB per case); the contact sheet is always written")
    ap.add_argument("--sgg-root", default=None)
    ap.add_argument("--out", default="nvs_pilot")
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
        # A fresh stream per case, so re-running one case reproduces its own
        # bearings rather than every later case's.
        rng = random.Random(args.seed * 1000 + index)
        try:
            row = run_case(case, args, egtr, predict, cv2, rng)
        except Exception as error:                       # noqa: BLE001
            print(f"  ! {type(error).__name__}: {error}")
            row = None
        if row:
            rows.append(row)

    ks = (10, 20, 100)
    report = table(rows, ks)
    print(f"\n{len(rows)}/{len(cases)} cases completed\n\n{report}")

    hard = [r for r in rows if not hit(r["stay"], 100)]
    easy = [r for r in rows if hit(r["stay"], 100)]
    if hard:
        print(f"\nstratum M -- the reference MISSED it ({len(hard)} cases)\n"
              + table(hard, ks))
    if easy:
        print(f"\nstratum H -- the reference already had it ({len(easy)} cases)\n"
              + table(easy, ks))

    # The head-to-head.  Over all cases the `nvs` policy is diluted by the ones
    # where it declines to move and scores `stay`, so `random` -- which always
    # moves -- can win the whole table while losing every case where the two
    # actually both moved.  This subset is where the pointer is on trial.
    pointed = [r for r in rows if r["nvs"]]
    if pointed:
        print(f"\nwhere a pointer existed and the robot walked to it "
              f"({len(pointed)} cases)\n" + table(pointed, ks))

    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump({"views": args.views, "max_az": args.max_az,
                   "max_el": args.max_el, "iou": args.iou, "topk": args.topk,
                   "seed": args.seed, "rows": rows}, fh, indent=1)
    print(f"\nwrote {os.path.join(args.out, 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
