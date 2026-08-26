"""
eval_move.py -- let the robot walk on the fusion's own evidence, and measure it.

Two arms, identical perception (A+C+R in both), differing only in the heading:
`evidence` takes it from the sweep, `random` draws it.  So this ablates NVS from
NAVIGATION; `fuse_live.py` ablates it from PERCEPTION.  The numbers are not
interchangeable.  See PIPELINE.md.

SUCCESS IS ONE REAL FRAME, JUDGED WITHOUT NVS: at the pose it walked to, the
robot runs one EGTR pass on its own camera and succeeds when the first entry
matching the instruction is the right instance.  A robot needing twenty
synthesised views to confirm what it is looking at has not found it.  Fusion is
spent on the DECISION, never on the ANSWER -- which also keeps the control
honest, since both arms are graded by the same single-view check.

    python eval_move.py --cases datasets/robot/cases_slot.json --steps 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from robot.world.proc_scene import Robot as ProcRobot

def random_pose(rc, azimuth: float, elevation: float):
    """A pose from the family the sweep samples, WITHOUT synthesising it.

    THE CONTROL ARM'S MOVE.  It draws (azimuth, elevation) from the same box the
    lemniscate spans and walks the pose that implies -- so both arms have the
    same action space and the only difference between them is WHICH pose was
    chosen.  The control used to fly an arc about P-hat instead, a different
    motion entirely, which left "moving helps" and "the rule chooses well"
    entangled in every comparison.

    No sweep and no detector: the orbit is a depth reading on the optical axis,
    the same one `perceive` centres its sweep on.
    """
    from robot.world.nvs_lemniscate import camera_for, look_at_point, orbit_depth

    radius = orbit_depth(getattr(rc.event, "depth_frame", None))
    if radius is None:
        return None, None
    camera = rc.camera_xyz.copy()
    orbit = look_at_point(camera, rc.agent_yaw, rc.camera_horizon, radius)
    return camera_for(orbit, camera, azimuth, elevation), orbit


def truth_boxes(rc, task) -> Dict[str, Any]:
    """The two boxes at the CURRENT pose, either backend.

    Procedural rooms read the segmentation frame BY INSTANCE NAME: `geometry`
    merges same-class instances, which would score the robot correct for walking
    to either copy of the target -- the one thing these cases exist to tell apart.
    """
    if isinstance(rc, ProcRobot):
        from robot.world.proc_scene import visible_box

        return {"target": visible_box(rc.event, task["target_name"]),
                "landmark": visible_box(rc.event, task["receptacle_name"])}

    from robot.task.measure import geometry
    from vg.vg150 import THOR_TO_VG150

    names = sorted({o["name"] for o in rc.event.metadata["objects"]
                    if o["objectType"] in THOR_TO_VG150
                    or o.get("moveable") or o.get("pickupable")})
    geo = geometry(rc, [task["target_name"], task["receptacle_name"]], names,
                   False)
    return {"target": geo.get(task["target_name"], {}).get("bbox_visible"),
            "landmark": geo.get(task["receptacle_name"], {}).get("bbox_visible")}


def look(rc, task, egtr, args) -> Dict[str, Any]:
    """THE ANSWER.  One EGTR pass on the robot's own frame, no synthesised views.

    THE VERDICT IS `top1_correct`: rank the instruction's candidate pairs by
    `rel * s_i * s_j` and ask whether the FIRST one is the instructed instance,
    both endpoints.  That is `fuse_live.decide`'s rule exactly, so a walked-to
    pose is judged by what the static experiment reports, and it is a rule a
    robot can execute -- it acts on its top-1 without needing to know it is
    right.

    THE LEXICAL STOP RULE IS STILL COMPUTED AND NO LONGER DECIDES ANYTHING.
    `stops` fired when some pair within `--trust` had `rel.argmax() == predicate`,
    and it measured the wrong thing twice over: EGTR utters `behind` for 2.7% of
    pairs, and the match landed at rank 4-10, never rank 1, so in 8 of 13 stops
    the correct pair was ALREADY first and the robot acted on a lower one.  The
    fields survive as diagnostics because "what would a lexical stop rule have
    done" is still a question worth answering; they are not the score.

      `top1_correct`  the verdict.  Ground truth, but only as the GRADE -- the
                      robot's own choice is the top-1, made without it.
      `recall`        where the correct pair sits in the ranking.  `top1_correct`
                      is `recall == 1`; the rest of the curve says how close a
                      miss was.
      `stops`, `matched_at`   diagnostics only.  See above.
    """
    from robot.policy.grounding import single_frame
    from robot.sgg_live import raw_predict
    from robot.task.task_find import iou

    raw = raw_predict(egtr, rc.event.frame)
    probs = raw["probs_softmax"].detach().cpu()
    rel = raw["rel"].detach().cpu()
    boxes = raw["boxes"].detach().cpu()
    # THE METRIC, and the SAME CALL `move_once` grades a photograph with.
    read = single_frame(probs.float(), boxes, rel, egtr, task,
                        width=args.condition, nms=args.cand_nms,
                        weight=args.weight, pair_iou=args.pair_iou)
    if read is None:
        return {"top1_correct": False, "stops": False, "stop_correct": False,
                "iou": 0.0, "matched_at": None, "recall": None,
                "candidates": 0,
                "frame": rc.event.frame.copy()
                         if args.figures else None}
    order = read["order"]
    predicate = egtr["rel_names"].index(task["predicate"])

    # Ground truth enters HERE and nowhere else: the ORDER above, and the
    # top-1 the robot acts on, are computed without it.
    truth = truth_boxes(rc, task)

    def grounded(a: int, b: int) -> bool:
        return (bool(truth["target"]) and bool(truth["landmark"])
                and iou(boxes[a].tolist(), truth["target"]) >= args.iou
                and iou(boxes[b].tolist(),
                        truth["landmark"]) >= args.iou)

    recall = next((position for position, (_, i, j) in enumerate(order, 1)
                   if grounded(i, j)), None)

    # THE TOP-1 PAIR'S BOXES, which is what the metric is about.  `chosen` below
    # exists at only 15% of poses, so rendering that instead left most panels
    # with no box and a caption reporting a different quantity.
    def box_of(q: int) -> List[float]:
        return [round(v, 1) for v in boxes[int(q)].tolist()]

    # THE TOP-1'S OWN SCORE, so "is the robot confident here?" can be asked of a
    # REAL frame.  The same quantity on NVS-mapped views separates a correct
    # top-1 from a wrong one by 0.33 of a decade with the quartiles overlapping,
    # but those values drift ~17 orders of magnitude across viewpoints for
    # numerical rather than photometric reasons; a real frame has no such drift,
    # so the question is open there and cannot be answered without recording it.
    # `margin` is the ratio to the best pair naming a DIFFERENT subject -- EGTR
    # emits several boxes per object, so the raw rank 2 is usually the same thing
    # again.  Both in log10, and None when there is nothing to compare.
    rival = next((v for v, a, _ in order[1:]
                  if iou(boxes[a].tolist(),
                         boxes[order[0][1]].tolist()) < args.iou),
                 None) if order else None
    top1 = {"top1_box": box_of(order[0][1]),
            "top1_object_box": box_of(order[0][2]),
            "top1_score": round(math.log10(order[0][0]), 3)
                          if order[0][0] > 0 else None,
            "top1_margin": round(math.log10(order[0][0]) - math.log10(rival), 3)
                           if rival and order[0][0] > 0 else None} if order else {}

    chosen, at = None, None
    for position, (_, i, j) in enumerate(order, 1):
        if args.trust and position > args.trust:
            break
        if int(rel[i, j].argmax()) == predicate:
            chosen, at = (i, j), position
            break
    if chosen is None:
        return {"top1_correct": recall == 1, "stops": False,
                "stop_correct": False, "iou": 0.0,
                "matched_at": None, "recall": recall,
                "candidates": len(order), **top1,
                "truth_box": None if not truth["target"] else
                             [round(float(v), 1) for v in truth["target"]],
                "frame": rc.event.frame.copy()
                         if args.figures else None}

    overlap = (iou(boxes[chosen[0]].tolist(), truth["target"])
               if truth["target"] else 0.0)
    on_landmark = (iou(boxes[chosen[1]].tolist(), truth["landmark"])
                   if truth["landmark"] else 0.0)
    return {"top1_correct": recall == 1,
            "stops": True,
            "stop_correct": overlap >= args.iou and on_landmark >= args.iou,
            "iou": round(overlap, 3), "iou_object": round(on_landmark, 3),
            "matched_at": at, "recall": recall, "candidates": len(order),
            **top1,
            # Kept so a stop can be RENDERED: "IoU 0.03" does not say whether the
            # robot mistook another bottle, a plant, or boxed the right one badly.
            "chosen_box": [round(v, 1) for v in
                           boxes[chosen[0]].tolist()],
            "object_box": [round(v, 1) for v in
                           boxes[chosen[1]].tolist()],
            "truth_box": None if not truth["target"] else
                         [round(float(v), 1) for v in truth["target"]],
            "frame": rc.event.frame.copy()
                     if args.figures else None}


def perceive(rc, case, task, egtr, args, want_record=False):
    """THE DECISION.  A sweep, fused with A+C+R, read for WHERE TO GO.

    Whether the robot has ARRIVED is `look`'s job, from one real frame.  Returns
    None when the scene cannot be read.
    """
    from lib.fusion import channels as ch
    from fuse_live import (OBJSCORE, OBJSCORE_CLASS, OBJSCORE_COS, conditioned,
                           consensus_relabel, record)
    from robot.policy import evidence, viewpick
    from robot.policy.grounding import pair_cells, rank_pairs
    from robot.world.nvs_lemniscate import (camera_for, lemniscate, look_at_point,
                                      orbit_depth, park_once, sweep)

    reference, camera = rc.event.frame.copy(), rc.camera_xyz.copy()

    # THE SWEEP'S CENTRE must lie on the reference camera's optical axis -- that
    # is what makes az = el = 0 reproduce the reference frame -- so it is the
    # pose plus a scalar depth read at the image centre, with NO DETECTION IN
    # THE LOOP.  It is also the point the step is executed about, so nothing
    # that can name the wrong object decides where the robot walks.
    radius = orbit_depth(getattr(rc.event, "depth_frame", None))
    if radius is None:
        return None
    orbit = look_at_point(camera, rc.agent_yaw, rc.camera_horizon, radius)

    poses = [camera_for(orbit, camera, az, el)
             for az, el in lemniscate(args.views, args.max_az, args.max_el)]
    reachable = rc.controller.step(
        action="GetReachablePositions").metadata["actionReturn"] or []
    if reachable:
        park_once(rc, {"x": float(camera[0]), "y": float(camera[1]),
                       "z": float(camera[2])}, reachable)
    else:
        # A procedural house has no navmesh, so `park_once` has nothing to pick
        # from.  Its point is only that the robot's shadow stays put across the
        # sweep, which a fixed far corner satisfies just as well.
        from robot.world.proc_scene import ROOM, look_from

        corner = min(((0.3, 0.3), (0.3, ROOM - 0.3), (ROOM - 0.3, 0.3),
                      (ROOM - 0.3, ROOM - 0.3)),
                     key=lambda p: -math.dist(p, (camera[0], camera[2])))
        look_from(rc.controller, corner[0], corner[1], 0.0, 0.0, force=True)
    # `--synth seva` hands the whole trajectory to the model in one pass, with
    # the robot's own frame as the only input; `reference` is that frame, read
    # at the top of this function before the sweep moves anything.
    model = getattr(args, "synth_model", None)
    if model is not None:
        model.tag = str(case["scene"]).replace("|", "_")
    rendered = sweep(rc, poses, task["target_name"], args.fov, reachable,
                     keep_frames=True,
                     synth=model.for_reference(reference) if model else None)
    # The sweep parks the robot across the room; come back before measuring.
    rc.teleport(position={"x": float(camera[0]),
                          "y": float(rc.agent_position["y"]),
                          "z": float(camera[2])})

    built = record(egtr, reference, [(i, r["frame"])
                                     for i, r in enumerate(rendered)])
    cand = conditioned(built, egtr, task, args.condition, args.cand_nms)
    if cand is None:
        return None
    which, margin, ballots = consensus_relabel(built, egtr, "argmax",
                                               per_view=True)
    s = ch.object_scores(built["rec"], mode=OBJSCORE,
                         class_mode=OBJSCORE_CLASS, cos=OBJSCORE_COS)
    rel = built["rel"]
    boxes = built["boxes"]
    predicate = egtr["rel_names"].index(task["predicate"])

    # THE PAIR THE SWEEP IS AIMED AT -- `move_once.decide_step`'s ranking, the
    # same call.  It is NOT the answer: `look` grades the pose this walks to,
    # from one real frame.  All it does here is tell `edge`, `acr` and `attrib`
    # which pair to ask the views about.
    #
    # `--w 0` and `--pair-iou 0` reproduce the predicate-free `rel.max()` ranking
    # this replaced closely but not exactly, so numbers from before the change
    # are not comparable pair-for-pair.
    cells = pair_cells(cand, boxes=boxes, pair_iou=args.pair_iou)
    ev = None
    if args.w:
        _, contrib, spoke = evidence.pair_contributions(built, cells, predicate)
        ev = evidence.pooled(cells, contrib, spoke, args.pool)
    order = rank_pairs(rel, cand, predicate, s, s, boxes=boxes,
                       pair_iou=args.pair_iou, mix=ev, weight=args.w)
    if not order:
        return None
    chosen, at = None, None
    for position, (_, i, j) in enumerate(order, 1):
        label = int(which[i, j]) if float(margin[i, j]) >= 0.0 \
            else int(rel[i, j].argmax())
        if label == predicate:
            chosen, at = (i, j), position
            break

    # THE VIEW THE RULE PICKED, when it can name one -- `viewpick.pick_view` is
    # the policy, shared with `move_once` so the simulator and the robot cannot
    # run different rules.  A sweep view is already a full pose, so naming one
    # needs nothing predicted to execute: the relative transform is that pose
    # minus this one.  `votes` -- the older scalar-azimuth channel -- had only
    # one consumer, the arc about P-hat, and went with it.  The per-view scores
    # are `move_once`'s to print.
    _, view, _ = viewpick.pick_view(
        args.bearing, args.side_step, built, rendered, cand, egtr, task,
        order, chosen, ballots, reference=reference,
        vlm=getattr(args, "vlm_model_obj", None))
    # BOTH HALVES OF A POSE-EXECUTING STEP: the view to go to, and the point the
    # sweep itself orbits.  `orbit` is a depth reading at the image centre and
    # passes through no detector, so `step_to` needs nothing that can name the
    # wrong object.
    out = {"matched_at": at, "view": view, "orbit": orbit}
    if want_record:
        # `probe_sideview.py` re-scores the reference's candidates through each
        # view's own field, and `save_sweep` draws them; both need the record,
        # the poses and the frame they came from.  Off by default: an episode
        # would otherwise carry 20 frames per step.
        out.update({"built": built, "rendered": rendered,
                    "reference": reference})
    return out


#: How far out to back off when the chosen view's own pose is refused, as a
#: factor on its distance from the sweep's hub.  RADIAL, so the bearing about the
#: hub -- the only thing the sweep actually chose -- is untouched, and the robot
#: stands the same direction away, further off.
#:
#: SMALL, AND TRIED IN ORDER, because backing off compounds.  An unconditional
#: 1.5 put the robot 1.15 -> 1.98 -> 2.72 m from the target over three steps,
#: since each step re-measures the hub from wherever it now stands and multiplies
#: again: on `slot|11` the bearing was right at every step and the answer decayed
#: anyway, the object having shrunk out of reach.
BACKOFF = (1.0, 1.08, 1.16, 1.25, 1.35, 1.5)


def step_to(rc, view: Dict[str, Any], orbit: np.ndarray,
            args) -> Optional[float]:
    """Walk to a rendered view: its own pose, relative to this one.

    THERE IS NOTHING TO PREDICT.  `nvs_lemniscate.camera_for` already returned
    position, yaw and pitch for every sweep view, so the motion is that pose
    minus the current one.  The arc this replaced kept only a scalar azimuth and
    swung about P-hat, a DIFFERENT centre from the one the sweep orbits, so the
    pose it reached was not the pose that had been rendered and scored.  On a
    list whose viewpoint windows are 15 degrees wide that gap is the
    measurement.

    THREE THINGS THE ROBOT CANNOT COPY.  The sweep's elevation lifts the virtual
    camera off the floor and a body cannot follow, so only the ground-plane
    translation is executed and the height is dropped -- which is the honest
    statement of what elevation is for here: evidence, never an action.  The yaw
    is RECOMPUTED from wherever the robot ends up rather than copied, so a
    shortened step still aims at the same point; at the full step it equals the
    view's own yaw, `camera_for` having defined it the same way.

    AND THE RADIUS, BUT ONLY WHEN IT HAS TO.  A sweep view sits at the reference
    camera's own distance from `orbit`, and that circle is sometimes illegal --
    the reference camera is already near the closest standoff the generator could
    find.  The fallback is to back off RADIALLY, in small steps, which keeps the
    bearing the sweep chose and only stands further away.  It is a fallback and
    not a policy: applying it unconditionally compounds, and did -- see `BACKOFF`.
    """
    from robot.geometry import horizon_towards, yaw_towards

    here = rc.camera_xyz[[0, 2]]
    hub = orbit[[0, 2]]
    seen_at = np.array([float(view["position"]["x"]),
                        float(view["position"]["z"])])
    y = float(rc.agent_position["y"])
    for backoff in BACKOFF:
        spot = hub + (seen_at - hub) * backoff
        # THE POSE'S OWN AIM, when it has one.  `camera_for`'s views all look at
        # the orbit centre, so for those the offset is absent and the yaw is
        # exactly what recomputing gives.  `viewgrid`'s carry a `yaw_offset`,
        # which is a real degree of freedom -- where the robot LOOKS is a
        # separate variable from where it STANDS -- and dropping it collapsed
        # every position's three candidates onto one.
        aim = yaw_towards(spot, hub) + float(view.get("yaw_offset", 0.0))
        if rc.teleport(position={"x": float(spot[0]), "y": y,
                                 "z": float(spot[1])},
                       yaw=aim, horizon=0.0):
            rc.teleport(yaw=aim,
                        horizon=horizon_towards(rc.camera_xyz, orbit))
            return float(np.linalg.norm(spot - here))
    return None


#: `--no-control` drops `random`, so consumers ask the ROW which arms it has
#: rather than assuming both.
ARMS = ("evidence", "random")

#: The view rules `--policy` accepts, in `viewpick.pick_view`'s own order.
BEARINGS = ("reveal", "attrib", "bin", "visible", "side", "volatility",
            "node", "edge", "acr", "vlm")


def arms_in(row) -> tuple:
    """The arms this result row actually carries, in ARMS order."""
    return tuple(a for a in ARMS if a in row)


def rollout(rc, case, task, egtr, args, guided: bool, bearings, start_pose,
            start, seed_view=None, seed_orbit=None, drawn=None,
            grid_centre=None) -> Dict[str, Any]:
    """One policy from the start pose.

    FIXED LENGTH: the episode always walks `--steps` steps and is graded at each
    pose by `top1_correct`.  Nothing stops it early.  The lexical rule that used
    to end an episode is still recorded per pose and decides nothing -- see
    `look`.  A robot that always acts on its top-1 needs no stopping rule to be
    scored; whether one could be BUILT is a separate question, and answering it
    inside the policy is what made the two unreadable together.

    `start` is the shared step-0 single-view answer, so both arms begin from the
    same frame and verdict, and `seed_view` is the sweep already taken there --
    re-sweeping would ask the same question from the same pose.
    """
    rc.teleport(position=start_pose["position"], yaw=start_pose["yaw"],
                horizon=start_pose["horizon"])

    def here():
        return [round(float(rc.agent_position["x"]), 3),
                round(float(rc.agent_position["z"]), 3)]

    stops: List[Dict[str, Any]] = []
    sheet: List[Dict[str, Any]] = []
    start = dict(start)
    start_frame = start.pop("frame", None)
    if start_frame is not None:
        sheet.append({"frame": start_frame, "step": 0, **start})
    trail = [{"step": 0, **start, "moved": 0.0, "xz": here()}]
    if start["stops"] and start_frame is not None:
        stops.append({"frame": start_frame, "step": 0, **start})

    # THE STOPPING RULE, when one is asked for: end the episode at the first pose
    # whose top-1 scores at least `--stop-score`.  On 360 real poses a correct
    # top-1 scored a median 1.05 decades above a wrong one (AUC 0.77), which is
    # the only quantity measured here that separates them at all -- the sweep's
    # own scores separate by 0.33 with the quartiles overlapping, and comparing
    # the current pose against the sweep's views fired on 2 of 13 correct poses
    # and 2 of 25 wrong ones.
    def confident(seen) -> bool:
        return (args.stop_score is not None
                and seen.get("top1_score") is not None
                and seen["top1_score"] >= args.stop_score)

    metres, sweeps = 0.0, 0
    stopped_at = 0 if confident(start) else None
    for k in range(args.steps if stopped_at is None else 0):
        # BOTH ARMS WALK TO A POSE, and it is the same executor -- `step_to`,
        # about the sweep's own orbit.  The guided arm's pose is the rule's
        # choice, the control's is drawn from the same box the sweep spans, so
        # the comparison isolates the CHOICE and nothing else.
        if guided:
            if k > 0:                       # step 0 reuses the start-pose sweep
                # ONLY THE GUIDED ARM SWEEPS, so only it has figures to write;
                # the control never asks the question.
                want = bool(args.figures)
                reading = perceive(rc, case, task, egtr, args, want_record=want)
                sweeps += 1
                if want and reading:
                    save_sweep(reading, case, task, egtr, args, step=k)
                seed_view, seed_orbit = ((reading["view"], reading["orbit"])
                                         if reading else (None, None))
            view, orbit, source = seed_view, seed_orbit, "evidence"
        elif drawn:
            view, orbit = drawn[k], grid_centre
            source = "random"
        else:
            view, orbit = random_pose(rc, *bearings[k])
            source = "random"

        moved = (None if view is None or orbit is None
                 else step_to(rc, view, orbit, args))
        if moved is None:
            trail.append({"step": k + 1, "unreachable": True})
            break
        metres += moved
        seen = look(rc, task, egtr, args)
        frame = seen.pop("frame", None)
        if frame is not None:
            sheet.append({"frame": frame, "step": k + 1, **seen})
            if seen["stops"]:
                stops.append({"frame": frame, "step": k + 1, **seen})
        trail.append({"step": k + 1, "source": source,
                      "azimuth": round(float(view["azimuth"]), 1),
                      "elevation": round(float(view["elevation"]), 1),
                      "moved": round(moved, 3), "xz": here(), **seen})
        if confident(seen):
            stopped_at = k + 1
            break
    # `per_step[k]` is the verdict AFTER k steps, so index 0 is the start pose
    # and the table in `main` reads straight off it.
    per_step = [bool(s.get("top1_correct")) for s in trail
                if "top1_correct" in s]
    return {"trail": trail, "per_step": per_step,
            # WHERE THE EPISODE ENDED.  With no stopping rule the episode is
            # fixed length and this is the last pose; with one it is the pose the
            # rule chose, which is the number a deployed robot would live with.
            "correct": per_step[-1] if per_step else False,
            "stopped_at": stopped_at,
            "best": any(per_step[1:]),
            # RETURN TO THE BEST POSE VISITED, which needs no constant.  A
            # threshold has to say "good enough" about a pose in isolation and
            # the top-1 score is bad at that -- at the cut that scores best on
            # both lists it is right 74-77% of the time it fires, and firing at
            # the wrong MOMENT costs more than firing at a wrong pose.  The same
            # quantity RANKS poses well (AUC 0.81 and 0.75), so comparing the
            # poses actually visited asks it only what it can answer.
            #
            # The robot walks back and looks again; the verdict it would reach
            # there is the one already recorded, so this is bookkeeping.
            "revisit": max((s for s in trail if s.get("top1_score") is not None),
                           key=lambda s: s["top1_score"],
                           default={}).get("top1_correct", False),
            "steps": len(per_step) - 1, "metres": round(metres, 3),
            "sweeps": sweeps, "stops": stops, "sheet": sheet}


def case_dir(case: Dict[str, Any], args, step: Optional[int] = None) -> str:
    """`<--figures>/<scene>_<target>[/stepN]/`, made on demand.

    `move_once`'s layout: one folder per case, one subfolder per step holding
    the sweep taken there.  A real episode is a sequence of `move_once` runs
    laid out exactly this way, so the two are readable side by side.
    """
    out = os.path.join(args.figures,
                       f"{case['scene']}_{case['target_name']}"
                       .replace("/", "_").replace("|", "_"))
    if step is not None:
        out = os.path.join(out, f"step{step}")
    os.makedirs(out, exist_ok=True)
    return out


def save_sweep(read: Dict[str, Any], case: Dict[str, Any], task, egtr, args,
               step: int) -> None:
    """`move_once`'s two sweep figures, for the sweep taken at one pose.

    The same call the real robot makes, so a rendered sweep and a synthesised
    one can be put side by side: `sweep.png` is what the views look like,
    `sweep_pred.png` is what each of them alone grounds the instruction to.
    """
    from robot.policy.grounding import per_view_answers
    from viz import sweep as figures

    built, rendered = read["built"], read["rendered"]
    out = case_dir(case, args, step)
    reference = read["reference"]
    # THE RAW SHEET ONLY WHEN A MODEL MADE THE VIEWS.  It exists to show what the
    # synthesiser INVENTED; under `--synth none` the frames are THOR's own
    # renders, so there is nothing to judge and `sweep_pred` shows the same
    # pictures with the answer drawn on them.
    if args.synth != "none":
        figures.contact_sheet(reference, [r["frame"] for r in rendered],
                              [r["pose"] for r in rendered],
                              os.path.join(out, "sweep.png"))
    rows, ref_row = per_view_answers(built, rendered, task, egtr,
                                     width=args.condition, nms=args.cand_nms,
                                     weight=args.weight,
                                     pair_iou=args.pair_iou)
    figures.answer_sheet(reference, rendered, rows, ref_row,
                         task["subject_class"], task["object_class"],
                         os.path.join(out, "sweep_pred.png"))


def render_trail(out: Dict[str, Any], case: Dict[str, Any], args) -> None:
    """One row per arm, one panel per pose: what the robot saw as it walked.

    GREEN the instructed instance, MAGENTA the subject this pose would walk to,
    thin ORANGE its object endpoint.  `rank` is the correct pair's position in
    this pose's own ranking, so `rank 1  no stop` is the robot looking straight
    at the answer and declining it.
    """
    import cv2

    from viz import sheets

    rows = []
    for arm in arms_in(out):
        panels = out[arm].get("sheet", [])
        if not panels:
            continue
        tiles = []
        for pose in panels:
            # Verdict by the METRIC, which is top-1.  THE LEXICAL `stops` FLAG IS
            # NOT DRAWN: it decides nothing in the policy -- `--stop-score` does
            # -- so captioning a panel `STOP(wrong)` next to an episode that
            # walks on reads as a contradiction in the figure rather than as the
            # dead diagnostic it is.
            hit = pose.get("recall") == 1
            # SPELLED OUT.  `evid s2 rank 3 miss` needs a key to read and a
            # figure that needs a key does not get read.
            place = pose.get("recall")
            label = (f"{arm}  step {pose['step']}  "
                     + (f"correct pair ranked {place}" if place
                        else "correct pair not ranked")
                     + ("  CORRECT" if hit else "  wrong"))
            # OVER THE FRAME, not above it: these tiles are a fixed 400x300 so a
            # row of them lines up across arms, and there is nowhere to put the
            # extra band.  `sheets.tile` letters it after the resize either way.
            tiles.append(sheets.tile(
                pose["frame"], [label],
                colours=[sheets.TRUTH if hit else sheets.DIM],
                size=(400, 300), over=True,
                boxes=[(pose.get(key), colour, thick) for key, colour, thick in
                       (("truth_box", sheets.TRUTH, 2),
                        ("top1_box", sheets.SUBJECT, 2),
                        ("top1_object_box", sheets.OBJECT, 1))]))
        rows.append(sheets.row(tiles))
    if not rows:
        return
    # The arms can walk different numbers of steps, so `stack` pads to the wider.
    cv2.imwrite(os.path.join(case_dir(case, args), "trail.png"),
                sheets.stack(rows, f"{case['scene']}   {case['instruction']}"))


def run_case(case: Dict[str, Any], args, egtr) -> Optional[Dict[str, Any]]:
    from robot.world import proc_scene
    from robot.task.measure import measure
    from robot.task.task_find import build_tasks, stage_at
    from vg.vg150 import THOR_TO_VG150

    rc = proc_scene.open_scene(case["scene"], args.width, args.height, args.fov,
                          case["start"])
    try:
        nameable = sorted(o["name"] for o in rc.event.metadata["objects"]
                          if o["objectType"] in THOR_TO_VG150)
        state = measure(rc, nameable, args.fov)
        entries = {o["name"]: o for o in rc.event.metadata["objects"]}
        base = case.get("landmark_name") or case["target_name"]
        task = next((t for t in build_tasks(state["objects"], entries)
                     if t["target_name"] == base), None)
        if task is None:
            return None
        task = {**task, "instruction": case["instruction"],
                "predicate": case["predicate"],
                "subject_class": case["subject_class"],
                "object_class": case["object_class"],
                "target_name": case["target_name"]}
        if stage_at(rc, base, case["occluder_name"], case["occluder_position"],
                    case.get("scene_poses")) is None:
            return None

        return run_episode(rc, case, task, egtr, args,
                           [float(entries[case["target_name"]]["position"][k])
                            for k in ("x", "z")],
                           [float(entries[base]["position"][k])
                            for k in ("x", "z")])
    finally:
        rc.stop()


def run_case_proc(case: Dict[str, Any], args, egtr) -> Optional[Dict[str, Any]]:
    """The same episode in a procedural tabletop room.

    Only the setup differs; everything from the start pose on is `run_episode`,
    shared, so the two backends cannot drift into measuring different things.
    """
    from robot.world.proc_scene import Robot, open_room, rebuild

    from fuse_live import task_for

    controller = open_room(args.width, args.height, args.fov)
    try:
        rc = Robot(controller, rebuild(controller, case))
        task = task_for(case)
        at = {o["name"]: o["position"] for o in case["objects"]}
        return run_episode(rc, case, task, egtr, args,
                           [at["target"]["x"], at["target"]["z"]],
                           [at["occluder"]["x"], at["occluder"]["z"]])
    finally:
        controller.stop()


def run_episode(rc, case: Dict[str, Any], task: Dict[str, Any], egtr, args,
                target_xz: Sequence[float], landmark_xz: Sequence[float]
                ) -> Optional[Dict[str, Any]]:
    """Both arms from one start pose, on a scene somebody else has staged."""
    start_pose = {"position": dict(rc.agent_position),
                  "yaw": rc.agent_yaw, "horizon": rc.camera_horizon}
    # The ANSWER at step 0, from the single real frame.
    start = look(rc, task, egtr, args)
    # The start-pose sweep.  ONLY THE EVIDENCE ARM ASKS FOR IT -- a control that
    # draws its pose at random never reads the answer, and taking it anyway cost
    # one EGTR pass over 20 views per case for nothing.  `orbit` it also returns
    # is needed either way, so when there is no sweep it is read straight off
    # the depth frame, which is where `perceive` gets it too.
    seed_view = seed_orbit = None
    if args.arm == "evidence":
        seed_read = perceive(rc, case, task, egtr, args,
                             want_record=bool(args.figures))
        if seed_read is None:
            return None
        seed_view, seed_orbit = seed_read["view"], seed_read["orbit"]
        if args.figures:
            save_sweep(seed_read, case, task, egtr, args, step=0)
        rc.teleport(position=start_pose["position"], yaw=start_pose["yaw"],
                    horizon=start_pose["horizon"])
    else:
        from robot.world.nvs_lemniscate import LOOKAT_DIST, look_at_point

        seed_orbit = look_at_point(rc.camera_xyz.copy(), rc.agent_yaw,
                                   rc.camera_horizon, LOOKAT_DIST)

    # THE CONTROL'S POSES, seeded on the CASE so a rerun walks the same control
    # trajectory.  TWO CONTROLS, answering two questions:
    #
    #   wedge  the same box the lemniscate spans, so both arms choose from one
    #          action space and the comparison isolates the CHOICE.  This is the
    #          ablation: does scoring beat picking at random from the same menu.
    #   grid   anywhere in the room the robot can stand with the pair in frame.
    #          A larger space than the method's own, on purpose -- the method
    #          decides where to synthesise and that decision is part of it.  So
    #          this reads as system against an aimless robot, and the gap is NOT
    #          attributable to scoring alone.  Report it beside `wedge`.
    rng = random.Random(f"{case['scene']}/{case['target_name']}")
    grid: List[Dict[str, Any]] = []
    grid_centre = None
    if args.control == "grid":
        from robot.world.proc_scene import ROOM
        from robot.world.viewgrid import feasible

        from robot.world import viewgrid

        # LANDMARK FIRST: it is the object nearer the camera in
        # "find the {target} behind the {landmark}", which is what fixes
        # azimuth 0 -- see `viewgrid`.
        grid, centre_xz, why = feasible(
            rc, landmark_xz, target_xz, args.fov, args.width, args.height,
            ROOM, span=(viewgrid.SPAN if args.control_span is None
                        else args.control_span))
        grid_centre = np.array([centre_xz[0], float(rc.agent_position["y"]),
                                centre_xz[1]], float)
        print(f"    grid {len(grid)} poses  rejected {why}", flush=True)
    bearings = [(rng.uniform(-args.max_az, args.max_az),
                 rng.uniform(-args.max_el, args.max_el))
                for _ in range(args.steps)]
    drawn = [rng.choice(grid) for _ in range(args.steps)] if grid else []

    out = {"scene": case["scene"], "instruction": case["instruction"],
           # The verdict at the START pose, which is the "do not move" control
           # every arm is read against.
           "start_correct": bool(start["top1_correct"]),
           "start_stops": start["stops"], "start_iou": start["iou"],
           # For the top-down plot: the point the sweep orbits, and the two
           # objects the instruction names.
           "centre_xz": [round(float(seed_orbit[0]), 3),
                         round(float(seed_orbit[2]), 3)],
           "target_xz": [round(v, 3) for v in target_xz],
           "landmark_xz": [round(v, 3) for v in landmark_xz]}
    arms = ((args.arm, args.arm == "evidence"),)
    for arm, guided in arms:
        out[arm] = rollout(rc, case, task, egtr, args, guided, bearings,
                           start_pose, start, seed_view, seed_orbit,
                           drawn=drawn, grid_centre=grid_centre)
    if args.figures:
        render_trail(out, case, args)
    for arm in arms_in(out):
        for stop in out[arm].get("stops", []):
            stop.pop("frame", None)
        for pose in out[arm].get("sheet", []):
            pose.pop("frame", None)
    if isinstance(rc, ProcRobot) and rc.refused:
        out["refused"] = f"{rc.refused}/{rc.steps}"

    # One line per case: top-1 correct at every pose, start first.  `.` is wrong
    # and `1` is right, so a walk that finds the answer and walks off it reads
    # `.1.` at a glance -- the failure the fixed-length episode exists to show.
    def track(r):
        return "".join("1" if ok else "." for ok in r["per_step"])

    print(f"    start {'OK ' if start['top1_correct'] else 'no '}"
          + "".join(f"   {arm} {track(out[arm])} "
                    f"({out[arm]['metres']} m)" for arm in arms_in(out)),
          flush=True)
    return out


#: `report_recall` lived here and printed recall@K of the correct pair.  It
#: existed because the OLD verdict was the lexical stop rule, which fires for
#: 2.7% of pairs, so a walk needed a threshold-free way to show that moving
#: improved the ranking at all.  The verdict is now top-1 itself, which IS
#: recall@1, and the rest of the curve answered a question nobody was asking.
#: `recall` is still written to the JSON per pose, so the analysis is one script
#: away if a near-miss distribution is ever wanted again.


def main(argv: Optional[Sequence[str]] = None) -> int:
    from robot.policy.grounding import WEIGHTS

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="datasets/robot/cases_slot.json")
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--condition", type=int, default=10)
    ap.add_argument("--no-reap", dest="reap", action="store_false",
                    help="do not kill orphaned THOR instances before starting")
    ap.add_argument("--trust", type=int, default=10, metavar="K",
                    help="how far down its own ranking the robot believes a "
                         "match.  0 = the whole conditioned list, which stops on "
                         "rank-74 matches and is not recognition.")
    ap.add_argument("--figures", metavar="DIR", default=None,
                    help="one folder per case, laid out as `move_once` lays out "
                         "a step:  <DIR>/<scene>_<target>/{sweep.png, "
                         "sweep_pred.png, trail.png}.  `sweep*` are the start "
                         "pose's views and what each grounds the instruction "
                         "to; `trail` is every pose the robot then looked from, "
                         "one row per arm.")
    ap.add_argument("--stop-score", type=float, default=None, metavar="LOG10",
                    help="stop at the first pose whose top-1 scores at least "
                         "this, in log10 of `rel * s * s` on a REAL frame.  "
                         "Omit for a fixed-length episode.  Around -5.0 on "
                         "cases_hard; a correct top-1 sits a median 1.05 decades "
                         "above a wrong one there (AUC 0.77), so the cut is real "
                         "but it is a fitted constant, not a calibrated one.")
    ap.add_argument("--side-step", type=float, default=10.0, metavar="DEG",
                    help="how far `--bearing side` and `volatility` walk per "
                         "step.  Small on "
                         "purpose: the side is re-measured every step, so an "
                         "overshoot flips the next reading and the robot comes "
                         "back, which a 30-degree step cannot do -- three of "
                         "those compound to 90 and top-1 correct went 32 -> 9 "
                         "-> 0.  Reaching 30 degrees in three 10-degree steps "
                         "scores what one 30-degree step does, with two "
                         "chances to correct the side on the way.")
    ap.add_argument("--cand-nms", type=float, default=0.0, metavar="IOU",
                    help="deduplicate the candidate list by box overlap before "
                         "taking the top-K.  See `grounding.candidates`.")
    ap.add_argument("--weight", choices=WEIGHTS, default="class",
                    help="what the METRIC's ranking weights a query by.  `class` "
                         "is p(the instruction's own noun); `s` is the query's "
                         "confidence over ALL classes, so it can be high for a "
                         "box the detector is merely sure is a chair.  Measured "
                         "over 4528 instructions on datasets/sgg/occlusion_ds4 "
                         "(analysis/eval_grounding.py): 30.6%% against 29.9%%, and "
                         "`s` degrades faster as the shortlist grows -- 23.1%% "
                         "against 28.2%% at K=40.  `s` was the default until "
                         "then, so numbers from before that are not comparable.")
    ap.add_argument("--pair-iou", type=float, default=0.0, metavar="IOU",
                    help="reject a pair whose two boxes overlap this much: it is "
                         "one object related to itself.  Applies to BOTH the "
                         "metric and the pair the sweep is aimed at.  0 keeps "
                         "every published number reproducible; `move_once` uses "
                         "0.15.  See `grounding.pair_cells`.")
    ap.add_argument("--w", type=float, default=0.0, metavar="W",
                    help="mixing weight on channel B -- whether the other views "
                         "agree this PAIR stands in the instructed relation -- "
                         "in the ranking the sweep is aimed at.  0 = A+C only. "
                         "Does not touch the metric, which is one real frame.")
    ap.add_argument("--pool", choices=("mean", "max"), default="max",
                    help="how channel B pools a pair's evidence across views.")
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--max-az", type=float, default=30.0)
    ap.add_argument("--max-el", type=float, default=15.0)
    # WHERE THE SWEEP COMES FROM, and the only thing that separates a bound from
    # a measurement.  `none` moves THOR's own camera to each pose -- the ground
    # truth a synthesiser is trying to produce, so every number is an upper
    # bound on the same pipeline driven by a model.  `seva` runs Stable Virtual
    # Camera on the robot's single frame instead.  The ANSWER is unaffected
    # either way: `look` grades on one real frame, never on a synthesised one.
    ap.add_argument("--synth", choices=("none", "seva"), default="none",
                    help="synthesise the sweep with a real NVS model instead of "
                         "rendering it in THOR.  See robot/nvs_seva.py.")
    ap.add_argument("--synth-steps", type=int, default=10, metavar="N",
                    help="denoising steps per sweep.  10 and single-stage is "
                         "the VG pipeline's own setting for a single "
                         "conditioning image; see script/run_nvs_multiview_"
                         "single.sh in ../sgg_nvs.")
    ap.add_argument("--synth-camera-scale", type=float, default=1.0,
                    metavar="S",
                    help="SEVA normalises the baseline away and rescales to "
                         "this, so it -- not the 0.5 m orbit radius -- is what "
                         "sets how much parallax the model is asked for")
    ap.add_argument("--synth-cfg", type=float, default=3.0)
    ap.add_argument("--synth-T", type=int, default=21, metavar="N",
                    help="frames SEVA denoises per chunk.  The sweep is 20 "
                         "views, so 21 is one pass; smaller chunks the "
                         "trajectory and is the knob that fits it on a small "
                         "card, at the cost of cross-chunk consistency.")
    ap.add_argument("--policy", default="reveal",
                    choices=tuple(BEARINGS) + ("random", "random-grid"),
                    help="THE ONE ARM THIS RUN WALKS.  A rule name runs the "
                         "method; `random` draws from the box the sweep spans "
                         "and `random-grid` from every pose in the room that "
                         "keeps the pair in frame.  One run, one row -- the two "
                         "arms used to walk together, which made a run that "
                         "changed one of them re-measure the other.")
    ap.add_argument("--control-span", type=float, default=None, metavar="DEG",
                    help="degrees of azimuth `--control grid` keeps about the "
                         "pair's own axis (default: viewgrid.SPAN, 150).  360 "
                         "keeps the whole floor, including the far side where "
                         "the instructed predicate is no longer true.")
    # THE DEFAULT IS READ FROM THE MODULE, not repeated here.  Repeating it meant
    # a run asking for one model silently got the other: `vlm.MODEL` moved to 7B
    # and this string did not, so two runs an hour apart produced byte-identical
    # results and both were 3B.
    ap.add_argument("--vlm-model", default=None,
                    help="the VLM `--policy vlm` reads the frame with "
                         "(default: robot.policy.vlm.MODEL)")
    ap.add_argument("--vlm-bits", type=int, default=4, choices=(4, 8, 16),
                    help="4-bit NF4 is ~2.5 GB and is what fits beside THOR "
                         "and EGTR on an 8 GB card; 16 is bf16 at 7.5 GB")
    ap.add_argument("--synth-fp16", action="store_true",
                    help="build SEVA's three modules in fp16 (~3.5 GB instead "
                         "of 6.8).  Needed to load at all on an 8 GB card; "
                         "watch for black frames, which is the SD-2.1 VAE "
                         "going NaN.")
    ap.add_argument("--synth-two-pass", action="store_true",
                    help="add the trajectory prior.  Off by default: its anchor "
                         "pass is for long trajectories, not for one image over "
                         "a small baseline, and it costs 5x.")
    ap.add_argument("--synth-dump", metavar="DIR", default=None,
                    help="write every synthesised sweep here, to look at")
    ap.add_argument("--iou", type=float, default=0.5)
    # SQUARE, because SEVA works on a square latent grid and a 4:3 input is
    # letterboxed into it.  THOR's `fieldOfView` is VERTICAL, so 60 degrees at
    # 800x600 was 75.6 degrees WIDE and at 600x600 is 60 -- the robot sees a
    # narrower slice of the room, and the case lists were staged at the old
    # aspect.  Recorded in the output either way.
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--out", default="results/move.json")
    args = ap.parse_args(argv)

    from robot.sgg_live import load_egtr

    if args.reap:
        # THOR's Unity child does not die with its python parent, so a run killed
        # from outside leaves an instance holding GPU memory and an X window.
        # They accumulate silently; five were once found alive at once.
        import subprocess
        # The server build is thor-CloudRendering, so the desktop-only pattern
        # reaped nothing there and orphans accumulated across jobs.
        stray = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f",
                                "thor-(Linux64|CloudRendering)"],
                               capture_output=True, text=True).stdout.split()
        if stray:
            print(f"reaping {len(stray)} orphaned THOR instances", flush=True)
            subprocess.run(["kill", "-9", *stray])

    frozen = json.load(open(args.cases))
    cases = frozen["cases"]
    if args.n:
        cases = cases[:args.n]
    egtr = load_egtr()

    # Five GB of weights and the best part of a minute, so it is built ONCE for
    # the whole run and carried on `args`; `perceive` binds it to the reference
    # frame per sweep.  None keeps the old path exactly as it was.
    args.synth_model = None
    if args.synth == "seva":
        from robot.world.nvs_seva import Synthesiser
        from robot.world.nvs_lemniscate import LOOKAT_DIST

        args.synth_model = Synthesiser(
            steps=args.synth_steps, cfg=args.synth_cfg,
            camera_scale=args.synth_camera_scale,
            two_pass=args.synth_two_pass, lookat_dist=LOOKAT_DIST,
            fov=args.fov, dump=args.synth_dump, T=args.synth_T,
            fp16=args.synth_fp16)

    # ONE FLAG, THREE THINGS IT USED TO TAKE.  `--policy` names the arm this run
    # walks; `--bearing`, `--control` and `--no-control` were three flags whose
    # legal combinations were not all meaningful (a `--bearing` with
    # `--no-control` off silently measured a second arm nobody asked for).
    args.arm = "random" if args.policy.startswith("random") else "evidence"
    args.control = "grid" if args.policy == "random-grid" else "wedge"
    args.bearing = args.policy if args.arm == "evidence" else None

    # ONE MODEL FOR THE WHOLE RUN, like the synthesiser above: loading is tens
    # of seconds and this arm decides once per step.
    args.vlm_model_obj = None
    if args.policy == "vlm":
        from robot.policy.vlm import MODEL, Director

        args.vlm_model = args.vlm_model or MODEL
        args.vlm_model_obj = Director(model=args.vlm_model, bits=args.vlm_bits)

    results: List[Dict[str, Any]] = []

    def save():
        if not args.out:
            return
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"steps": args.steps, "condition": args.condition,
                   "policy": args.policy,
                   "control_span": args.control_span,
                   "vlm_model": args.vlm_model if args.policy == "vlm" else None,
                   "vlm_bits": args.vlm_bits if args.policy == "vlm" else None,
                   "synth": args.synth, "synth_T": args.synth_T,
                   "synth_fp16": args.synth_fp16,
                   "synth_steps": args.synth_steps, "cases": results},
                  open(args.out, "w"), indent=1)

    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['scene']}  {case['instruction']}",
              flush=True)
        try:
            out = (run_case_proc(case, args, egtr) if frozen.get("procedural")
                   else run_case(case, args, egtr))
        except Exception as error:                              # noqa: BLE001
            # The line matters more than the message: a bare type and text sent
            # the last three of these to the wrong place.
            import traceback
            print(f"    ! {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
            out = None
        if out:
            results.append(out)
            save()

    if results:
        n = len(results)
        # THE TABLE.  Top-1 correct after k steps, per arm, against standing
        # still -- one criterion, one denominator, every case in it.
        print(f"\n{n} cases.  Top-1 correct -- `fuse_live.decide`'s rule, on one "
              f"real frame:\n")
        depth = max(len(r[a]["per_step"])
                    for r in results for a in arms_in(r))
        print("  " + " " * 16
              + "".join(f"{('stay' if k == 0 else f'{k} step'):>9}"
                        for k in range(depth))
              + f"{'best pose':>11}")
        for arm in arms_in(results[0]):
            row = []
            for k in range(depth):
                row.append(sum(1 for r in results
                               if len(r[arm]["per_step"]) > k
                               and r[arm]["per_step"][k]))
            print(f"  {arm:16s}"
                  + "".join(f"{v:>6}/{n:<3}" for v in row)
                  + f"{sum(1 for r in results if r[arm]['best']):>7}/{n:<3}")
        print("\n  `stay` is the start pose, which every policy shares, so that "
              "column is\n  the do-not-move control and is comparable across "
              "runs.\n  `best pose` is any pose after moving -- the ceiling a "
              "stopping rule could reach.")
        print("\n  Walking the fixed steps, then returning to the "
              "highest-scoring pose:\n")
        for arm in arms_in(results[0]):
            print(f"  {arm:16s}"
                  f"{sum(1 for r in results if r[arm]['revisit']):>6}/{n:<3}")
        # HOW FAR IT WALKED FOR THAT.  Two policies at the same accuracy are not
        # the same policy if one crossed the room to get there, and the action
        # spaces differ by design: `random` draws from the wedge the sweep
        # spans, `random-grid` from the whole feasible floor.  The median as
        # well as the mean, because one case that backed off across the room
        # moves a 40-case mean by more than it should.
        print("\n  Metres walked over the fixed steps:\n")
        for arm in arms_in(results[0]):
            walked = sorted(float(r[arm]["metres"]) for r in results)
            mid = (walked[len(walked) // 2] if len(walked) % 2
                   else (walked[len(walked) // 2 - 1]
                         + walked[len(walked) // 2]) / 2.0)
            print(f"  {arm:16s}{sum(walked) / n:>6.2f} m mean"
                  f"{mid:>8.2f} m median"
                  f"{walked[-1]:>8.2f} m max")
        if args.stop_score is not None:
            # WHERE THE ROBOT ACTUALLY STOPPED, which is the only column a
            # deployed robot gets.  The columns above are diagnostics: they
            # report every pose, and nobody can pick among them at run time.
            print(f"\n  Stopping at the first pose scoring >= "
                  f"{args.stop_score:+.2f}:\n")
            for arm in arms_in(results[0]):
                ended = [r[arm] for r in results]
                steps = [e["stopped_at"] if e["stopped_at"] is not None
                         else e["steps"] for e in ended]
                print(f"  {arm:16s}{sum(1 for e in ended if e['correct']):>6}/{n:<3}"
                      f"   stopped in {sum(steps) / n:.2f} steps on average, "
                      f"{sum(1 for e in ended if e['stopped_at'] is not None)}/{n} "
                      f"because the rule fired")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
