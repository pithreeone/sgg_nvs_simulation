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

    python eval_move.py --cases nvs_pilot/cases/cases_easy2.json --steps 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from robot.proc_scene import Robot as ProcRobot

#: Bins for the bearing vote, degrees.  Votes are often bimodal -- the target is
#: recoverable from either side -- and the mean of "25 left" and "25 right" is
#: "do not move".
BIN = 10.0


#: The smoothed version of `--bearing side` -- a Gaussian kernel regression of
#: the same counts over azimuth -- lives in `plot_viewdist.py`, where it is the
#: FIGURE.  It is not here because it is not the policy: its argmax landed on the
#: +-30 boundary in 35 of 40 cases, so the curve was only ever answering "which
#: side", and comparing the two halves directly scores 36 of 40 against its 35.


def bearing_from(votes: Sequence[float], bin_width: float = BIN
                 ) -> Optional[float]:
    """Bin the voting azimuths, take the heaviest bin, average inside it."""
    if not votes:
        return None
    bins: Dict[int, List[float]] = {}
    for az in votes:
        bins.setdefault(int(round(az / bin_width)), []).append(az)
    best = max(bins.values(), key=len)
    return sum(best) / len(best)


def truth_boxes(rc, task) -> Dict[str, Any]:
    """The two boxes at the CURRENT pose, either backend.

    Procedural rooms read the segmentation frame BY INSTANCE NAME: `geometry`
    merges same-class instances, which would score the robot correct for walking
    to either copy of the target -- the one thing these cases exist to tell apart.
    """
    if isinstance(rc, ProcRobot):
        from robot.proc_scene import visible_box

        return {"target": visible_box(rc.event, task["target_name"]),
                "landmark": visible_box(rc.event, task["receptacle_name"])}

    from eval_nvs_pointer import geometry
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
    from fuse_live import conditioned
    from robot.sgg_live import raw_predict
    from robot.task_find import iou

    raw = raw_predict(egtr, rc.event.frame)
    built = {"probs_ref": raw["probs_softmax"].detach().cpu().numpy(),
             "rel": raw["rel"].detach().cpu(),
             "boxes": raw["boxes"].detach().cpu(),
             "s": raw["probs_softmax"].detach().cpu().max(-1).values}
    cand = conditioned(built, egtr, task, args.condition, args.cand_nms)
    if cand is None:
        return {"top1_correct": False, "stops": False, "stop_correct": False,
                "iou": 0.0, "matched_at": None, "recall": None,
                "candidates": 0,
                "frame": rc.event.frame.copy()
                         if args.save_trail else None}

    rel, s = built["rel"], built["s"].float()
    subjects, objects = cand
    predicate = egtr["rel_names"].index(task["predicate"])
    # `fuse_live.decide`'s score exactly, so a walked-to pose is judged by the
    # rule the static experiment reports.  Weighting by p(instructed class)
    # instead of `s` moved 2 of 40 rankings: `rel` spans 20 orders of magnitude
    # against `s`'s 1.5, so anything multiplied in is a rounding error.
    order = sorted(((float(rel[i, j, predicate]) * float(s[i]) * float(s[j]),
                     i, j)
                    for i in subjects for j in objects if i != j), reverse=True)

    # Ground truth enters HERE and nowhere else: the ORDER above, and the
    # top-1 the robot acts on, are computed without it.
    truth = truth_boxes(rc, task)

    def grounded(a: int, b: int) -> bool:
        return (bool(truth["target"]) and bool(truth["landmark"])
                and iou(built["boxes"][a].tolist(), truth["target"]) >= args.iou
                and iou(built["boxes"][b].tolist(),
                        truth["landmark"]) >= args.iou)

    recall = next((position for position, (_, i, j) in enumerate(order, 1)
                   if grounded(i, j)), None)

    # THE TOP-1 PAIR'S BOXES, which is what the metric is about.  `chosen` below
    # exists at only 15% of poses, so rendering that instead left most panels
    # with no box and a caption reporting a different quantity.
    def box_of(q: int) -> List[float]:
        return [round(v, 1) for v in built["boxes"][int(q)].tolist()]

    top1 = {"top1_box": box_of(order[0][1]),
            "top1_object_box": box_of(order[0][2])} if order else {}

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
                         if args.save_trail else None}

    overlap = (iou(built["boxes"][chosen[0]].tolist(), truth["target"])
               if truth["target"] else 0.0)
    on_landmark = (iou(built["boxes"][chosen[1]].tolist(), truth["landmark"])
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
                           built["boxes"][chosen[0]].tolist()],
            "object_box": [round(v, 1) for v in
                           built["boxes"][chosen[1]].tolist()],
            "truth_box": None if not truth["target"] else
                         [round(float(v), 1) for v in truth["target"]],
            "frame": rc.event.frame.copy()
                     if args.save_trail else None}


def perceive(rc, case, task, egtr, args, centre=None, want_record=False):
    """THE DECISION.  A sweep, fused with A+C+R, read for WHERE TO GO.

    Whether the robot has ARRIVED is `look`'s job, from one real frame.  Returns
    None when the scene cannot be read.  `centre` is P-hat: derived here on the
    first call and returned for the caller to carry.
    """
    from lib.fusion import channels as ch
    from fuse_live import (CORR, GATE_COS, OBJSCORE, OBJSCORE_CLASS,
                           OBJSCORE_COS, conditioned, consensus_relabel, record)
    from robot.nvs_lemniscate import (camera_for, lemniscate, look_at_point,
                                      park_once, sweep)
    from robot.robot_controller import point_in_box

    import torch

    reference, camera = rc.event.frame.copy(), rc.camera_xyz.copy()

    # TWO CENTRES, different jobs.  The SWEEP's must lie on the reference
    # camera's optical axis -- that is what makes az = el = 0 reproduce the
    # reference frame -- so it is the pose plus a scalar depth read at the image
    # centre, with no detection in the loop.  The ARC centre, refined below from
    # the top-ranked pair, is the one that decides where the robot walks.
    depth = getattr(rc.event, "depth_frame", None)
    if depth is None:
        return None
    height, width = depth.shape[:2]
    patch = depth[int(height * 0.4):int(height * 0.6),
                  int(width * 0.4):int(width * 0.6)]
    orbit = look_at_point(camera, rc.agent_yaw, rc.camera_horizon,
                          float(np.median(patch)))

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
        from robot.proc_scene import ROOM, look_from

        corner = min(((0.3, 0.3), (0.3, ROOM - 0.3), (ROOM - 0.3, 0.3),
                      (ROOM - 0.3, ROOM - 0.3)),
                     key=lambda p: -math.dist(p, (camera[0], camera[2])))
        look_from(rc.controller, corner[0], corner[1], 0.0, 0.0, force=True)
    rendered = sweep(rc, poses, task["target_name"], args.fov, reachable,
                     keep_frames=True)
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
    subjects, objects = cand
    order = sorted(((float(rel[i, j].max()) * float(s[i]) * float(s[j]), i, j)
                    for i in subjects for j in objects if i != j), reverse=True)

    predicate = egtr["rel_names"].index(task["predicate"])
    chosen, at = None, None
    for position, (_, i, j) in enumerate(order, 1):
        label = int(which[i, j]) if float(margin[i, j]) >= 0.0 \
            else int(rel[i, j].argmax())
        if label == predicate:
            chosen, at = (i, j), position
            break

    if centre is None:
        # `order[0]`, never `chosen`: a pair passing the lexical gate fires the
        # stop rule, so the `chosen` branch was dead for motion in 32 of 32
        # walked episodes.  The OBJECT endpoint either way -- the landmark ranks
        # 1st in 40 of 40, and P-hat lands a median 0.07 m from it against 0.48 m
        # for an independent argmax over p(landmark class).
        endpoint = order[0][2]
        centre = point_in_box(rc, built["boxes"][endpoint].tolist(), args.fov)
        if centre is None:
            return None

    votes: List[float] = []
    if args.bearing == "side":
        # HOW MANY OF THE INSTRUCTION'S OWN CANDIDATES THIS VIEW STILL SEES.
        #
        # No predicate, no identity, no P-hat: a candidate that does not
        # correspond in a view can never be selected FROM that view, so this is
        # the necessary condition, and answering it needs no idea which triplet
        # is the right one.  That is why it beats the rules that do -- agreement
        # with a voted pair was right in 4 of 10, and the class-probability gain
        # is carried by the unoccluded twin, which every viewpoint sees.
        #
        # ONLY THE SUBJECT SIDE, which is an open choice and not a claim.  The
        # landmark corresponds from nearly everywhere (it ranks 1st in 40 of 40),
        # so its count is near constant across azimuth and a constant cannot move
        # an argmax -- multiplying it in measured 34 of 40 against 35 here.
        # Taking the `min` of the two sides instead scored 37, and would drop the
        # assumption that the SUBJECT is the hard end, which is `behind`'s
        # property rather than a general one; 2 cases in 40 is not evidence for
        # it, so it is left as the next thing to try, not adopted.
        #
        # The count is at SLOT level, NOT deduplicated to objects.  Dedup was
        # tried on the belief that duplicate boxes are a detector habit: 31 of 40
        # against 36.  How many of an object's boxes survive is a graded measure
        # of how clearly it is seen, and dedup throws that gradation away.
        #
        # A SIDE, THEN A SMALL STEP -- there is no angle in this rule, and that is
        # the honest shape of what was measured.  The smoothed estimator in
        # `plot_viewdist.py` reads the same counts as a curve over azimuth and
        # picks its argmax; it scores 35 of 40 against 36 for comparing the two
        # halves and stepping a fixed amount.  The curve's SHAPE carries nothing:
        # its argmax sat on the +-30 boundary in 35 of 40 cases, so it was only
        # ever answering "which side".  Reporting it as a side is not a
        # simplification of the result, it IS the result.
        #
        # The step is small and the side is re-measured every step, so the walk
        # self-corrects: overshoot flips the sign of the next reading and the
        # robot comes back.  At 30 degrees a step it could not -- three of those
        # compound to 90, past anything the sweep ever saw, and top-1 correct
        # went 32 -> 9 -> 0 over three steps.
        hn = torch.nn.functional.normalize(
            built["rec"]["h_ref"].float(), dim=-1)
        # POSITIVE AZIMUTH IS THE ROBOT'S LEFT; see `nvs_lemniscate.camera_for`.
        left: List[float] = []
        right: List[float] = []
        for index, view in enumerate(built["rec"]["views"]):
            if int(view["v"]) in ch.SKIP_VIEWS:
                continue
            _, ok = ch.correspond(hn, view["h"], CORR, GATE_COS)
            # A view that corresponds to nothing counts as 0, never skipped:
            # that is the reading on the side the target is hidden on, and
            # dropping it is the one line that pre-sorted this measurement
            # toward the answer once already.  So both halves always have a
            # value, and the guided arm never falls back to the control.
            count = (0.0 if ok is None
                     else float(sum(1 for q in cand[0] if bool(ok[q]))))
            azimuth = float(rendered[index]["pose"]["azimuth"])
            (left if azimuth > 0 else right).append(count)
        if left and right:
            votes = [args.side_step
                     if sum(left) / len(left) >= sum(right) / len(right)
                     else -args.side_step]
    elif args.bearing == "visible":
        # WHICH VIEW SEES THE TARGET, not which view says `behind`.  A visibility
        # question needs no predicate and only the SUBJECT endpoint to
        # correspond, where the `bin` rule's lexical gate left 39% of guided
        # steps with no NVS input at all.  It also avoids `rel`, whose cross-view
        # variation is numerical rather than photometric (~17 orders for the same
        # pair, equally for an occluded target, its unoccluded twin, and the
        # symmetric `near`); a class probability is a softmax output, so a
        # difference between views is a difference in what was visible.
        #
        # THE GAIN, NOT THE MAXIMUM: raw p(cup) picks whichever view sees the
        # DISTRACTOR best, since it is unoccluded from every angle.
        index = {v: k - 1 for k, v in egtr["obj_names"].items()}.get(
            task["subject_class"])
        if index is not None:
            hn = torch.nn.functional.normalize(
                built["rec"]["h_ref"].float(), dim=-1)
            ref_p = built["probs_ref"][:, index]
            az_list = [r["pose"]["azimuth"] for r in rendered]
            best, top = None, 0.0
            for view in built["rec"]["views"]:
                if int(view["v"]) in ch.SKIP_VIEWS:
                    continue
                match, ok = ch.correspond(hn, view["h"], "mutual", OBJSCORE_COS)
                probs = view["probs"].float()
                for q in cand[0]:
                    if not bool(ok[q]):
                        continue
                    gain = float(probs[int(match[q]), index]) - float(ref_p[q])
                    if gain > top:
                        best, top = int(view["v"]), gain
            if best is not None:
                votes = [az_list[best]]
    elif chosen is not None:
        si, sj = chosen
        az_list = [r["pose"]["azimuth"] for r in rendered]
        votes = [az_list[b["v"]] for b in ballots
                 if bool(b["spoke"][si, sj])
                 and int(b["named"][si, sj]) == predicate]
        if not votes:
            # A's attribution: one endpoint is enough, so it survives where R's
            # two-endpoint gate does not.
            hn = torch.nn.functional.normalize(
                built["rec"]["h_ref"].float(), dim=-1)
            best, top = None, -1.0
            for view in built["rec"]["views"]:
                if int(view["v"]) in ch.SKIP_VIEWS:
                    continue
                match, ok = ch.correspond(hn, view["h"], "mutual", OBJSCORE_COS)
                if not bool(ok[si]):
                    continue
                value = float(view["probs"].float()[int(match[si])].max())
                if value > top:
                    best, top = int(view["v"]), value
            if best is not None:
                votes = [az_list[best]]
    out = {"votes": votes, "matched_at": at, "centre": centre}
    if want_record:
        # `probe_sideview.py` re-scores the reference's candidates through each
        # view's own field, which needs the record and the poses.  Off by
        # default: an episode would otherwise carry 20 frames per step.
        out.update({"built": built, "rendered": rendered})
    return out


#: How much further out than the current standoff the arc is flown, as a factor
#: on the radius.  See `walk` for the arithmetic that fixes it above 1.13.
RADIUS_SCALE = 1.2

#: Multipliers ON TOP of `--radius-scale`, tried in order until THOR accepts the
#: pose, because a refused pose would otherwise be scored as a policy failure.
#: The last rung is 2.5x nominal, past which the object is too small to read and
#: failing is the honest answer.
RADIUS_LADDER = (1.0, 1.25, 1.5, 1.8, 2.1, 2.5)


def walk(rc, centre: np.ndarray, azimuth: float, args) -> Optional[float]:
    """Swing about P-hat by `azimuth` and re-aim.  Returns metres moved.

    THE ARC IS FLOWN AT A LARGER RADIUS THAN THE ROBOT STANDS AT, because the
    nearest one is illegal by construction: an arc about a point ON THE TABLE
    closes depth by `r (1 - cos theta)`, and the generator already spent that
    margin.  `(RADIUS_SCALE - 1) * r` must cover it; the r cancels, so the factor
    must exceed 1.13 at 30 degrees whatever the radius.  At 1.0 THOR refused 17
    of 71 guided steps and 14 of 35 episodes could not move at all.  The trade is
    pixel area: the target sits ~1.15 m away instead of 1.00.

    ONE FACTOR IS NOT ENOUGH -- which pose is legal depends on where the walls
    and furniture fall, and a refused angle is indistinguishable, in the result,
    from an angle the robot chose badly.  So the scale ESCALATES until THOR
    accepts one.  Nothing here reads ground truth.
    """
    from robot.nvs_lemniscate import arc_step
    from robot.robot_controller import horizon_towards, yaw_towards

    here = rc.camera_xyz[[0, 2]]
    hub = centre[[0, 2]]
    turned = arc_step(hub, here, azimuth)
    y = float(rc.agent_position["y"])
    for scale in RADIUS_LADDER:
        goal = hub + (turned - hub) * args.radius_scale * scale
        if rc.teleport(position={"x": float(goal[0]), "y": y,
                                 "z": float(goal[1])},
                       yaw=yaw_towards(goal, hub), horizon=0.0):
            break
    else:
        # Normally useless in a procedural room -- the navmesh this reads is
        # empty -- but the iTHOR path has one, so it stays.
        goal = hub + (turned - hub) * args.radius_scale
        spot = rc.nearest_reachable(goal)
        if spot is None or not rc.teleport(
                position={"x": float(spot[0]), "y": y, "z": float(spot[1])},
                yaw=yaw_towards(spot, hub), horizon=0.0):
            return None
        goal = spot
    rc.teleport(yaw=yaw_towards(goal, hub),
                horizon=horizon_towards(rc.camera_xyz, centre))
    return float(np.linalg.norm(goal - here))


#: `--no-control` drops `random`, so consumers ask the ROW which arms it has
#: rather than assuming both.
ARMS = ("evidence", "random")


def arms_in(row) -> tuple:
    """The arms this result row actually carries, in ARMS order."""
    return tuple(a for a in ARMS if a in row)


def rollout(rc, case, task, egtr, args, guided: bool, bearings, start_pose,
            centre, start, seed_votes) -> Dict[str, Any]:
    """One policy from the start pose.

    FIXED LENGTH: the episode always walks `--steps` steps and is graded at each
    pose by `top1_correct`.  Nothing stops it early.  The lexical rule that used
    to end an episode is still recorded per pose and decides nothing -- see
    `look`.  A robot that always acts on its top-1 needs no stopping rule to be
    scored; whether one could be BUILT is a separate question, and answering it
    inside the policy is what made the two unreadable together.

    `start` is the shared step-0 single-view answer, so both arms begin from the
    same frame and verdict.  `seed_votes` is the sweep already taken there to
    establish P-hat: re-sweeping would ask the same question from the same pose,
    and when it came back empty the guided arm left along the control's heading.
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

    metres, sweeps, votes = 0.0, 0, list(seed_votes)
    for k in range(args.steps):
        source, azimuth = "random", bearings[k]
        reading = None
        if guided:
            if k > 0:                       # step 0 reuses the start-pose sweep
                reading = perceive(rc, case, task, egtr, args, centre)
                sweeps += 1
                votes = reading["votes"] if reading else []
            voted = bearing_from(votes)
            if voted is not None:
                azimuth, source = voted, "evidence"

        moved = walk(rc, centre, azimuth, args)
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
                      "azimuth": round(azimuth, 1), "moved": round(moved, 3),
                      "votes": len(votes), "xz": here(), **seen})
    # `per_step[k]` is the verdict AFTER k steps, so index 0 is the start pose
    # and the table in `main` reads straight off it.
    per_step = [bool(s.get("top1_correct")) for s in trail
                if "top1_correct" in s]
    return {"trail": trail, "per_step": per_step,
            "correct": per_step[-1] if per_step else False,
            "best": any(per_step[1:]),
            "steps": len(per_step) - 1, "metres": round(metres, 3),
            "sweeps": sweeps, "stops": stops, "sheet": sheet}


def render_trail(out: Dict[str, Any], case: Dict[str, Any], args) -> None:
    """One row per arm, one panel per pose: what the robot saw as it walked.

    GREEN the instructed instance, MAGENTA the subject this pose would walk to,
    thin ORANGE its object endpoint.  `rank` is the correct pair's position in
    this pose's own ranking, so `rank 1  no stop` is the robot looking straight
    at the answer and declining it.
    """
    import cv2

    rows = []
    for arm in arms_in(out):
        panels = out[arm].get("sheet", [])
        if not panels:
            continue
        tiles = []
        for pose in panels:
            canvas = np.ascontiguousarray(pose["frame"][:, :, ::-1])
            for key, colour, thick in (("truth_box", (140, 255, 140), 2),
                                       ("top1_box", (230, 120, 230), 2),
                                       ("top1_object_box", (90, 200, 255), 1)):
                if pose.get(key):
                    x0, y0, x1, y1 = (int(v) for v in pose[key])
                    cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thick)
            # Verdict by the METRIC, which is top-1.  Whether the robot STOPPED
            # is appended, not substituted: `rank 1  miss-stop` is the stop rule
            # failing, not perception.
            hit = pose.get("recall") == 1
            label = (f"{arm[:4]} s{pose['step']} rank {pose.get('recall')} "
                     f"{'OK' if hit else 'miss'}"
                     + (f"  STOP{'' if pose.get('stop_correct') else '(wrong)'}"
                        if pose.get("stops") else ""))
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (20, 20, 20), -1)
            cv2.putText(canvas, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (140, 255, 140) if hit else (200, 200, 200),
                        1, cv2.LINE_AA)
            tiles.append(cv2.resize(canvas, (400, 300)))
        rows.append(np.hstack(tiles))
    if not rows:
        return
    # The arms can walk different numbers of steps, so pad to the wider row.
    width = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == width else
            np.hstack([r, np.zeros((r.shape[0], width - r.shape[1], 3),
                                   dtype=r.dtype)]) for r in rows]
    os.makedirs(args.save_trail, exist_ok=True)
    stem = f"{case['scene']}_{case['target_name']}".replace("/", "_")
    cv2.imwrite(os.path.join(args.save_trail, f"{stem}.png"), np.vstack(rows))


def run_case(case: Dict[str, Any], args, egtr) -> Optional[Dict[str, Any]]:
    from robot import drive
    from robot.drive_triplet_scene import measure
    from robot.task_find import build_tasks, stage_at
    from vg.vg150 import THOR_TO_VG150

    rc = drive.open_scene(case["scene"], args.width, args.height, args.fov,
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
    from robot.proc_scene import Robot, open_room, rebuild

    controller = open_room(args.width, args.height, args.fov)
    try:
        rc = Robot(controller, rebuild(controller, case))
        task = {"instruction": case["instruction"],
                "predicate": case["predicate"],
                "subject_class": case["subject_class"],
                "object_class": case["object_class"],
                "target_name": case["target_name"],
                "receptacle_name": case["occluder_name"],
                "distractors": [{"name": case["distractor_name"]}]}
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
    # P-hat, which the arc turns about, from the pair the fusion settled on.
    seed_read = perceive(rc, case, task, egtr, args)
    if seed_read is None:
        return None
    centre, seed_votes = seed_read["centre"], seed_read["votes"]
    rc.teleport(position=start_pose["position"], yaw=start_pose["yaw"],
                horizon=start_pose["horizon"])

    # One bearing sequence, both arms, seeded on the CASE rather than consumed
    # from a shared stream: an episode where the evidence never speaks then
    # reproduces the control exactly, and dropping the control does not shift the
    # guided arm's fallback headings.
    rng = random.Random(f"{case['scene']}/{case['target_name']}")
    bearings = [rng.uniform(-args.max_az, args.max_az)
                for _ in range(args.steps)]

    out = {"scene": case["scene"], "instruction": case["instruction"],
           # The verdict at the START pose, which is the "do not move" control
           # every arm is read against.
           "start_correct": bool(start["top1_correct"]),
           "start_stops": start["stops"], "start_iou": start["iou"],
           # For the top-down plot: the point both arms orbit, and the two
           # objects the instruction names.
           "centre_xz": [round(float(centre[0]), 3),
                         round(float(centre[2]), 3)],
           "target_xz": [round(v, 3) for v in target_xz],
           "landmark_xz": [round(v, 3) for v in landmark_xz]}
    arms = (("evidence", True),) if args.no_control else (("evidence", True),
                                                          ("random", False))
    for arm, guided in arms:
        out[arm] = rollout(rc, case, task, egtr, args, guided, bearings,
                           start_pose, centre, start, seed_votes)
    if args.save_trail:
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="nvs_pilot/cases/cases_easy2.json")
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--condition", type=int, default=10)
    ap.add_argument("--no-reap", dest="reap", action="store_false",
                    help="do not kill orphaned THOR instances before starting")
    ap.add_argument("--trust", type=int, default=10, metavar="K",
                    help="how far down its own ranking the robot believes a "
                         "match.  0 = the whole conditioned list, which stops on "
                         "rank-74 matches and is not recognition.")
    ap.add_argument("--save-trail", metavar="DIR", default=None,
                    help="write EVERY pose the robot looked from, one row per "
                         "arm, with the boxes and the correct pair's rank")
    ap.add_argument("--no-control", action="store_true",
                    help="run only the `evidence` arm, halving the wall clock.  "
                         "For RENDERS, not for claims: with no control there is "
                         "nothing to attribute a success to.")
    ap.add_argument("--bearing", choices=("bin", "visible", "side"),
                    default="bin",
                    help="how the sweep becomes a heading.  `bin` averages the "
                         "azimuths of views that named the instructed predicate "
                         "and is gated on the lexical match -- 39%% of guided "
                         "steps then had no NVS input.  `visible` heads for the "
                         "single view that most INCREASED p(subject class).  "
                         "`side` compares how many of the instruction's "
                         "candidates the LEFT half of the sweep still sees "
                         "against the right half, and steps `--side-step` "
                         "degrees toward the better one: 36 of 40 on "
                         "cases_hard against 20 for the best fixed angle, 17.4 "
                         "for random and 13 for standing still.  It always "
                         "speaks, so no step falls back to the control.")
    ap.add_argument("--side-step", type=float, default=10.0, metavar="DEG",
                    help="how far `--bearing side` walks per step.  Small on "
                         "purpose: the side is re-measured every step, so an "
                         "overshoot flips the next reading and the robot comes "
                         "back, which a 30-degree step cannot do -- three of "
                         "those compound to 90 and top-1 correct went 32 -> 9 "
                         "-> 0.  Reaching 30 degrees in three 10-degree steps "
                         "scores what one 30-degree step does, with two "
                         "chances to correct the side on the way.")
    ap.add_argument("--cand-nms", type=float, default=0.0, metavar="IOU",
                    help="deduplicate the candidate list by box overlap before "
                         "taking the top-K.  See `fuse_live.conditioned`.")
    ap.add_argument("--radius-scale", type=float, default=RADIUS_SCALE,
                    metavar="K",
                    help="fly the arc at K times the current radius.  Must "
                         "exceed 1 / cos(max azimuth) ~ 1.13 at 30 degrees, or "
                         "the arc closes depth the generator already spent and "
                         "40%% of episodes cannot move at all.  See `walk`.")
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--max-az", type=float, default=30.0)
    ap.add_argument("--max-el", type=float, default=15.0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--out", default="nvs_pilot/move.json")
    args = ap.parse_args(argv)

    from robot.sgg_live import load_egtr

    if args.reap:
        # THOR's Unity child does not die with its python parent, so a run killed
        # from outside leaves an instance holding GPU memory and an X window.
        # They accumulate silently; five were once found alive at once.
        import subprocess
        stray = subprocess.run(["pgrep", "-f", "thor-Linux64"],
                               capture_output=True, text=True).stdout.split()
        if stray:
            print(f"reaping {len(stray)} orphaned THOR instances", flush=True)
            subprocess.run(["kill", "-9", *stray])

    frozen = json.load(open(args.cases))
    cases = frozen["cases"]
    if args.n:
        cases = cases[:args.n]
    egtr = load_egtr()

    results: List[Dict[str, Any]] = []

    def save():
        if not args.out:
            return
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"steps": args.steps, "condition": args.condition,
                   "cases": results}, open(args.out, "w"), indent=1)

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
        print("\n  `stay` is the same pose for both arms, so that column is the "
              "control.\n  `best pose` is any pose after moving -- the ceiling a "
              "stopping rule could reach.")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
