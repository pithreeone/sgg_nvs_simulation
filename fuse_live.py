"""
fuse_live.py -- run the paper's map-back fusion on one live THOR frame + its sweep.

The offline study feeds `lib/fusion/channels.py` out of a cache that
`scratch/mapback_cache.py` wrote from PNGs on disk.  Nothing about the fusion
needs that: the cache is a schema, not a mechanism.  This file assembles the
SAME record in memory from a frame the robot just took plus the novel views
synthesised around it, and calls the very same functions.  So what runs here is
the paper's fusion, not a reimplementation of it -- which is the only way a
robotics result can be evidence about the paper's method.

The channel set is the one `logs/occlusion_ds4_refpred_assign.log` was run at:

    B (predCond)   beta, pool=mean, endpoint gate cos>=0.80, corr=assign,
                   predcond_mode=refpred -- each view is read at the predicate
                   the REFERENCE proposes, so `on` needs no hand-set predicate
                   list.  Our instructions are all `on`, and the shipped
                   5-predicate list excludes it.
    A (objscore)   mode=max, class gate, cos 0.80 -- `run_occlusion_ds4_objscore.sh`
                   and the A+C+R row of `logs/occlusion_ds4_acr.log`.  Reported
                   as its OWN row here rather than folded into the fused one:
                   offline A is worth +0.17 R@20 standalone and +0.0 on top of
                   B, so a single "fused" number would hide which channel moved.
    C (dedup)      OFF here, deliberately: with C on, a single-view-vs-fused
                   delta bundles deduplication into a number reported as
                   multi-view.  C is a fair thing to add later, to BOTH arms.
    D (predicate calibration)  OFF: it is a prior fitted on labels, and it
                   moves both arms identically.

Views 9 and 19 are dropped by `channels.SKIP_VIEWS`, and that is not a detail:
the lemniscate returns to zero offset at t=pi and t=2pi, so those two frames
reproduce the reference pose.  Counting them manufactures cross-view agreement
out of a copy of the input.

With `--beta 0` the fused ranking is `max_p(rel) * s_i*s_j`, which is exactly
what `sgg_live.predict` computes -- so that setting is a self-test of the
assembly, not an experiment, and the runner checks it.

Three rows come out of one render pass, so the channels are separable at no
extra cost: `single` (neither), `A` (object-score repair only), `A+B` (both).

    python fuse_live.py --case 0 1 2 --beta 1.0 --topk 100
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Sparse width of a view's own relation field.  `assign` correspondence reads
#: the UN-mapped field, which the cache stores at 500 view pairs
#: (`az30el15_v20_um500`); measured equivalent to 2000 for B's recall.
VIEW_TOPK = 500

#: `logs/occlusion_ds4_refpred_assign.log`.
GATE_COS = 0.80
CORR = "assign"
POOL = "mean"

#: Channel A, from `my_script/run_occlusion_ds4_objscore.sh` and the A+C+R row
#: of `logs/occlusion_ds4_acr.log`.  `max` pools the single best view, which is
#: where channels.py measures its gain; `gate` requires the view's argmax class
#: to equal the reference's, so a view that re-detects a DIFFERENT object cannot
#: donate its score.
OBJSCORE = "max"
OBJSCORE_CLASS = "gate"
OBJSCORE_COS = 0.80

#: Channel C: two slots are the same object at box IoU above this AND the same
#: argmax class, and then only one triplet survives per (group, group, argmax
#: predicate).  `occlusion_channels.py --dedup_iou`.
DEDUP_IOU = 0.90

#: Channel R, the consensus relabel.  `rl_margin` is the AGREEMENT FRACTION --
#: votes for the winning predicate over views that said anything -- so tau is a
#: fraction and 0 means "whenever any view spoke, take the majority".  The
#: fraction rather than the raw count is load-bearing: a pair many views can see
#: is an EASY pair where the reference is already right, so ranking by count
#: selects exactly the wrong pairs (19.2% already-correct against 2.7%).
RELABEL_MODE = "consensus"
RELABEL_TAU = 0.0

#: Which predicates a view may vote FOR.  `occlusion_channels.py` defaults this
#: to the predicates the ground truth happens to contain, and says so: that
#: "peeks at test labels".  Taking `vg_conventions.TARGET_PREDICATES` instead is
#: the no-peeking option the same file offers, and it is the only defensible one
#: here -- our instructions are all `on`, so a vocabulary derived from our own
#: labels would be the answer.
from vg.vg_conventions import TARGET_PREDICATES as RELABEL_VOCAB  # noqa: E402

#: Channel V: zero every predicate the generator cannot produce, so `max_p(rel)`
#: competes only among the admissible ones.  Measured reason to want it here:
#: on the `behind` list EGTR detects and correctly classifies the target object
#: in 44% of cases (probe_size.py) while the instructed triplet reaches the
#: top-100 in 1 of 71 -- the object end is fine and the PREDICATE end loses,
#: every time, to `on`.
#:
#: What V is NOT, in their words: "it is NOT a multi-view result and must never
#: be quoted as one; it is a benchmark mismatch being corrected, and it moves
#: the baseline every other channel is measured against."  So it is switched on
#: for EVERY row including the single-view one, which keeps it out of every
#: difference between rows.
#:
#: The vocabulary is `vg_conventions.TARGET_PREDICATES`, a property of the THOR
#: export knowable without opening a label file -- not the predicates this
#: ground truth happens to contain, which `occlusion_channels.py` flags as
#: peeking at test labels.
VOCAB = tuple(RELABEL_VOCAB)


def sparse_pairs(torch, rel, keep: int = VIEW_TOPK):
    """`mapback_cache.topk_sparse`: keep the strongest `keep` pairs of a field."""
    n = rel.shape[0]
    strength = rel.max(-1)[0].clone()
    strength.fill_diagonal_(0.0)
    top = torch.topk(strength.flatten(), min(keep, n * n)).indices
    ii, jj = top // n, top % n
    return torch.stack([ii, jj], 1).short().cpu(), rel[ii, jj].half().cpu()


def record(egtr, reference: np.ndarray,
           views: Sequence[Tuple[int, np.ndarray]]) -> Dict[str, Any]:
    """
    One in-memory `mapback_cache` record: the reference, then every view.

    Field for field what `--layout dirs --unmapped_topk` writes, because
    `channels.view_evidence_perpred` reads it by name.  `probs` is the class
    SOFTMAX and `s_ref`/`c_ref` its max, matching `mapback_cache.py`; the
    sigmoid EGTR also emits is not used: `object_scores` reads `s_ref`/`probs`,
    which are the softmax ones.
    """
    from robot.sgg_live import raw_predict

    torch = egtr["torch"]

    def one(frame):
        raw = raw_predict(egtr, frame)
        return {"probs": raw["probs_softmax"].detach().cpu().float(),
                "h": raw["query"].detach().cpu().float(),
                "rel": raw["rel"].detach().cpu().float(),
                "boxes": raw["boxes"].detach().cpu().float()}

    ref = one(reference)
    s_ref, c_ref = ref["probs"].max(-1)
    entries = []
    for index, frame in views:
        view = one(frame)
        vidx, vval = sparse_pairs(torch, view["rel"])
        entries.append({"v": int(index), "h": view["h"].half(),
                        "probs": view["probs"].half(),
                        "vidx": vidx, "vval": vval,
                        "boxes": view["boxes"]})
    return {"rec": {"s_ref": s_ref.half(), "c_ref": c_ref.short(),
                    "h_ref": ref["h"].half(), "views": entries},
            "rel": ref["rel"], "s": s_ref, "c": c_ref, "boxes": ref["boxes"],
            "probs_ref": ref["probs"].numpy()}


def vocab_mask(egtr, vocab: Sequence[str] = VOCAB):
    """[50] multipliers: 1.0 for an admissible predicate, 0.0 for the rest."""
    from lib.fusion import channels as ch

    names = egtr["rel_names"]
    return ch.weight_vector(names, {p: 0.0 for p in names if p not in vocab})


def fuse(built: Dict[str, Any], beta: float, objscore: str = "none",
         weights=None,
         pool: str = POOL, gate_cos: float = GATE_COS, corr: str = CORR,
         class_mode: str = OBJSCORE_CLASS, objscore_cos: float = OBJSCORE_COS):
    """
    -> (rank [NQ,NQ], coverage stats).  Both channels, either one switchable off.

    Straight through `lib.fusion.channels`:

      A  `object_scores` repairs the per-slot object score s~ from the views
         that re-detect the same slot, pooled by `max` -- the single best view,
         which is where its gain was measured to come from.
      B  per-predicate view evidence, read at the predicate the REFERENCE
         proposes, added to the reference's own ranking with weight `beta`.

    `objscore="none", beta=0` is the plain single-view ranking, and is the
    self-test: it must reproduce `sgg_live.predict`.
    """
    from lib.fusion import channels as ch

    # V, applied before anything reads `rel` -- the ranking, the relabel's
    # reference predicate, and B's `evidence_at` all follow it.
    rel = built["rel"] if weights is None else built["rel"] * weights
    s = (ch.object_scores(built["rec"], mode=objscore, class_mode=class_mode,
                          cos=objscore_cos)
         if objscore != "none" else built["s"].float())
    if beta <= 0:
        return ch.pair_rank(rel, s, None, 0.0)[1], {}
    agg, cnt, stats = ch.view_evidence_perpred(built["rec"], pool=pool,
                                               gate_cos=gate_cos, corr=corr)
    if agg is None:
        return ch.pair_rank(rel, s, None, 0.0)[1], {}
    evidence = ch.evidence_at(agg.float(), cnt.float(), rel, pool)
    return ch.pair_rank(rel, s, evidence, beta)[1], stats


def consensus_relabel(built: Dict[str, Any], egtr, corr: str = "argmax",
                      gate_cos: float = GATE_COS):
    """
    R -> (rl_pred [NQ,NQ] predicate index, rl_margin [NQ,NQ] agreement fraction).

    For every reference pair, each view that speaks about it names one
    predicate; the majority wins and the margin is the fraction of speakers who
    agreed.  A view speaks only if its mapped field is non-zero there AND both
    endpoints have a correspondence at `gate_cos`.  Ported from
    `occlusion_channels.py`'s consensus branch, which is not importable -- it is
    inline in that script's per-image prep, not a function in `channels.py`.

    `-1` where nobody spoke, so `frac >= tau` cannot fire there even at tau = 0.
    """
    import torch

    from lib.fusion import channels as ch

    torch_ = egtr["torch"]
    names = egtr["rel_names"]
    mask = torch.zeros(len(names))
    for predicate in RELABEL_VOCAB:
        if predicate in names:
            mask[names.index(predicate)] = 1.0

    rec = built["rec"]
    nq = rec["s_ref"].shape[0]
    hn = torch.nn.functional.normalize(rec["h_ref"].float(), dim=-1)
    votes = torch.zeros(nq, nq, len(names))
    spoke_n = torch.zeros(nq, nq)
    for view in rec["views"]:
        if int(view["v"]) in ch.SKIP_VIEWS:
            continue
        field, ok = ch._pair_field(view, nq, hn, corr, gate_cos)
        field = field * mask
        spoke = field.max(-1)[0] > 0
        if ok is not None:
            spoke = spoke & (ok[:, None] & ok[None, :])
        votes.scatter_add_(-1, field.argmax(-1).unsqueeze(-1),
                           spoke.float().unsqueeze(-1))
        spoke_n += spoke.float()
    top, which = votes.max(-1)
    return which, torch.where(spoke_n > 0, top / spoke_n.clamp(min=1.0),
                              torch.full_like(spoke_n, -1.0))


def conditioned(built: Dict[str, Any], egtr, task: Dict[str, Any],
                width: int = 10):
    """
    The candidate pairs an INSTRUCTION licenses, and nothing else.

    "Find the bottle behind the plant" hands the robot two class names and a
    predicate.  That is information the task provides, so a candidate set built
    from it uses no oracle: the model still has to say WHICH query is the
    bottle, which is the plant, and how strongly they stand in that relation.
    Ground truth enters only when the answer is graded.

    What this changes, and it is the whole point: a triplet no longer has to
    outrank every other triplet in the room.  Under free-form SGDet the
    instructed pair loses to `laptop on table` and the case contributes nothing
    measurable -- 70 of 71 `behind` cases reached no rank at all.  Conditioned,
    every case yields a rank, and the difficulty sits where the task actually
    puts it: which bottle, and is it really behind the plant.

    NOT comparable to any free-form SGG number, including this project's own
    benchmark tables.  It is a different question asked of the same model.

    THE PREDICATE IS NOT CONDITIONED, and that is the line between a fair
    experiment and a rigged one.  The instruction supplies two class names, so
    restricting candidates by class costs nothing the task did not give.  It
    also supplies the predicate -- but the predicate is the QUANTITY UNDER
    TEST: whether multi-view evidence makes the model say `behind` about this
    pair rather than `on` or `near` is the entire question B and R exist to
    answer.  Fixing it would hand over the answer and leave those two channels
    with nothing to do.

    -> ([subject queries], [object queries])
    """
    import numpy as _np

    classes = {v: k - 1 for k, v in egtr["obj_names"].items()}
    probs = built["probs_ref"]
    subject_id = classes.get(task["subject_class"])
    object_id = classes.get(task["object_class"])
    if subject_id is None or object_id is None:
        return None

    # Ranked by the probability of the INSTRUCTED class, not by argmax: a query
    # sitting on the bottle often argmaxes to something else -- measured median
    # p(class) on these targets is 0.07 -- and dropping it would throw away the
    # detection this experiment depends on.
    subjects = _np.argsort(-probs[:, subject_id])[:width]
    objects = _np.argsort(-probs[:, object_id])[:width]
    return [int(q) for q in subjects], [int(q) for q in objects]


def decide(built: Dict[str, Any], egtr, task: Dict[str, Any], rank,
           candidates, geo, iou_hit: float = 0.5) -> Dict[str, Any]:
    """
    The ONE pair a robot would act on, and whether acting on it succeeds.

    The instruction is `find the bottle behind the plant`.  A robot cannot use
    a ranked list: it walks to one object.  So the instruction is spent where a
    decision needs it -- in the SCORE -- rather than as a filter applied after
    the fact:

        candidates   top-N queries by p(bottle) x top-N by p(plant)
        score        rel[i, j, behind] * s~_i * s~_j
        decision     argmax, one pair
        success      that subject's box lands on the instructed instance

    `behind the plant` is doing the job it does in English: picking out WHICH
    bottle.  The robot does not need the model to utter the word, it needs the
    model's `behind` field to be right about which pair stands in that relation
    -- so a good relation representation shows up as picking the right bottle,
    and a bad one as picking the wrong one.  This is what A (repairing s~) and B
    (adding view evidence to the relation term) both feed into.

    WHAT THIS STOPS MEASURING: whether the model would SAY `behind`.  Channels R
    and V are about which label wins a ranking, so they have nothing to act on
    here and are not reported.
    """
    from robot.task_find import iou

    subjects, objects = candidates
    predicate = egtr["rel_names"].index(task["predicate"])
    rel, s = built["rel"], built["s"].float()
    scored = [(float(rel[i, j, predicate]) * float(s[i]) * float(s[j]), i, j)
              for i in subjects for j in objects if i != j]
    if not scored:
        return {"chosen": None, "success": False, "iou": 0.0}
    score, i, j = max(scored)

    truth = geo.get(task["target_name"], {}).get("bbox_visible")
    box = built["boxes"][i].tolist()
    overlap = iou(box, truth) if truth else 0.0
    # Where the right answer sat, so a miss can be told apart from a near miss.
    ranked = sorted(scored, reverse=True)
    correct = next((r for r, (_, a, _b) in enumerate(ranked, 1)
                    if truth and iou(built["boxes"][a].tolist(), truth) >= iou_hit),
                   None)
    return {"chosen": [int(i), int(j)], "score": score,
            "iou": round(overlap, 3), "success": overlap >= iou_hit,
            "best_rank": correct, "candidates": len(scored)}


def triplets_of(built: Dict[str, Any], rank, egtr, topk: int = 10,
                groups=None, relabel=None, weights=None,
                candidates=None, as_classes=None,
                tau: float = RELABEL_TAU) -> List[Dict[str, Any]]:
    """
    A ranking matrix as the triplet list the rest of this repo speaks.

    Boxes and classes come from the REFERENCE slots for every arm, which is
    what makes single-view and fused directly comparable and what lets
    `task_find.grade` score both against the robot's own frame -- a novel
    view's boxes are in a frame the grader has no ground truth for.
    """
    from lib.pytorch_misc import argsort_desc

    rel = built["rel"] if weights is None else built["rel"] * weights
    c_ref, boxes = built["c"], built["boxes"]
    s = built["s"].float()
    out, seen = [], set()
    if candidates is not None:
        # Only the instruction's own class pair competes, and the predicate is
        # fixed to the one it names.
        subjects, objects = candidates
        scored = sorted(((float(rank[i, j]), i, j)
                         for i in subjects for j in objects if i != j),
                        reverse=True)
        order = [(i, j) for _, i, j in scored]
    else:
        order = argsort_desc(rank.numpy())
    for sub_q, obj_q in order:
        if len(out) >= topk:
            break
        sub_q, obj_q = int(sub_q), int(obj_q)
        predicate = int(rel[sub_q, obj_q].argmax())
        if groups is not None:
            # C.  One triplet per (group, group, predicate); the signature uses
            # the REFERENCE's predicate and is computed BEFORE R, which is what
            # `to_prediction` does.  channels.py flags the consequence -- a
            # relabel can make two survivors collide -- and accepts it rather
            # than re-running dedup, which would change the top-K.
            key = (int(groups[sub_q]), int(groups[obj_q]), predicate)
            if key in seen:
                continue
            seen.add(key)
        if relabel is not None:
            rl_pred, rl_margin = relabel
            if float(rl_margin[sub_q, obj_q]) >= tau:
                predicate = int(rl_pred[sub_q, obj_q])
        # Under conditioning the class pair is GIVEN by the instruction, so
        # the endpoints carry the instructed names rather than each query's
        # argmax.  Selecting a query by p(bottle) and then grading it as
        # whatever its argmax happens to be would make the conditioning inert:
        # the candidate is chosen for one reason and judged by another.  What
        # is left to judge is the PREDICATE and the INSTANCE.
        names = (as_classes if as_classes else
                 (egtr["obj_names"].get(int(c_ref[sub_q]) + 1,
                                        f"obj_{int(c_ref[sub_q])}"),
                  egtr["obj_names"].get(int(c_ref[obj_q]) + 1,
                                        f"obj_{int(c_ref[obj_q])}")))
        out.append({
            "subject": names[0],
            "predicate": egtr["rel_names"][predicate],
            "object": names[1],
            "score": float(rank[sub_q, obj_q]),
            "subject_score": float(s[sub_q]), "object_score": float(s[obj_q]),
            "subject_box": [float(v) for v in boxes[sub_q].tolist()],
            "object_box": [float(v) for v in boxes[obj_q].tolist()],
            "subject_query": sub_q, "object_query": obj_q,
        })
    return out


def run_case(case: Dict[str, Any], args, egtr,
             band: Optional[Sequence[float]] = None
             ) -> Optional[Dict[str, Any]]:
    """Stage the case, sweep it, fuse it, and score both arms on the SAME frame."""
    from robot import drive
    from robot.drive_triplet_scene import measure
    from eval_nvs_pointer import geometry, regrade
    from robot.nvs_lemniscate import camera_for, lemniscate, park_once, sweep
    from robot.robot_controller import point_in_box
    from robot.sgg_live import predict
    from robot.task_find import (STAGE_GRID, build_tasks, put_in_front,
                                 stage_at)
    from vg.vg150 import THOR_TO_VG150

    rc = drive.open_scene(case["scene"], args.width, args.height, args.fov,
                          case["start"])
    try:
        nameable = sorted(o["name"] for o in rc.event.metadata["objects"]
                          if o["objectType"] in THOR_TO_VG150)
        names = sorted({o["name"] for o in rc.event.metadata["objects"]
                        if o["objectType"] in THOR_TO_VG150
                        or o.get("moveable") or o.get("pickupable")})
        state = measure(rc, nameable, args.fov)
        entries = {o["name"]: o for o in rc.event.metadata["objects"]}
        # `build_tasks` keys its `on` proposals by the object that RESTS on
        # something, which for `in front of` is the landmark rather than the
        # graded instance -- `target_name` moved to the occluder when the case
        # was frozen.  Look the base task up by whichever the case says.
        base = case.get("landmark_name") or case["target_name"]
        task = next((t for t in build_tasks(state["objects"], entries)
                     if t["target_name"] == base), None)
        if task is None:
            print("  ! the instruction is no longer unambiguous here")
            return None
        if case.get("predicate") in ("behind", "in front of"):
            # The frozen record already names the relation; rebuilding it from
            # `build_tasks` would hand back the `on` version.
            task = {**task, "instruction": case["instruction"],
                    "predicate": case["predicate"],
                    "subject_class": case["subject_class"],
                    "object_class": case["object_class"],
                    # `in front of` moves the graded instance to the occluder.
                    "target_name": case["target_name"],
                    "distractors": [] if case["predicate"] == "in front of"
                                   else task.get("distractors", []),
                    "receptacle_name": case.get("occluder_name",
                                                task["receptacle_name"])}
        print(f"  {task['instruction']}   target {task['target_name']}")
        # The BAND THE CASE FILE WAS FROZEN AT, not `put_in_front`'s defaults.
        # Staging is redone here because `SetObjectPoses` renumbers objectIds
        # and THOR's settle is not bit-reproducible -- but redoing it without
        # the band silently re-stages every case at the default target of 0.50,
        # which made a 0.25-0.50 case list produce a measured median of 0.48,
        # identical to the list it was supposed to differ from.
        # The occlusion is always measured on the LANDMARK -- the object the
        # occluder hides.  For `on`/`behind` that is the graded instance too,
        # but `in front of` moves grading to the occluder, and measuring the
        # occluder against itself reports 0% hidden and stages nothing.
        hidden = case.get("landmark_name") or task["target_name"]
        if case.get("occluder_position"):
            staged = stage_at(rc, hidden, case["occluder_name"],
                              case["occluder_position"],
                              case.get("scene_poses"))
        else:
            staged = put_in_front(rc, hidden,
                                  case["occluder_type"],
                                  task["receptacle_name"],
                                  *( () if band is None else
                                     (STAGE_GRID, band[0], band[1], band[2]) ))
        if staged is None:
            print("  ! staging failed")
            return None

        reference = rc.event.frame.copy()
        camera = rc.camera_xyz.copy()

        # The orbit centre, from the OBJECT anchor's box and the real depth
        # frame.  No ground truth -- see `robot_controller.point_in_box`.
        from robot.sgg_live import raw_predict
        object_id = {v: k - 1 for k, v in egtr["obj_names"].items()}[
            task["object_class"]]
        ref_raw = raw_predict(egtr, reference)
        anchor = int(ref_raw["probs_softmax"][:, object_id].argmax())
        centre = point_in_box(rc, ref_raw["boxes"][anchor].tolist(), args.fov)
        if centre is None:
            print("  ! no depth at the object anchor")
            return None

        poses = [camera_for(centre, camera, az, el)
                 for az, el in lemniscate(args.views, args.max_az, args.max_el)]
        reachable = rc.controller.step(
            action="GetReachablePositions").metadata["actionReturn"] or []
        park_once(rc, {"x": float(camera[0]), "y": float(camera[1]),
                       "z": float(camera[2])}, reachable)
        rendered = sweep(rc, poses, task["target_name"], args.fov, reachable,
                         keep_frames=True)

        built = record(egtr, reference,
                       [(i, r["frame"]) for i, r in enumerate(rendered)])

        # Grade on the robot's own frame, which is the only frame the ground
        # truth is defined in.  Both arms share it.
        rc.teleport(position={"x": float(camera[0]),
                              "y": float(rc.agent_position["y"]),
                              "z": float(camera[2])})
        geo = geometry(rc, [task["target_name"]]
                       + [d["name"] for d in task.get("distractors") or []],
                       names, False)

        import torch
        from lib.fusion import channels as ch

        groups = ch.slot_groups_xyxy(built["boxes"], built["c"].long(),
                                     args.dedup_iou)
        relabel = consensus_relabel(built, egtr, args.corr)
        weights = (vocab_mask(egtr, [task["predicate"]])
                   if args.vocab_instructed
                   else vocab_mask(egtr) if args.vocab else None)
        cand = (conditioned(built, egtr, task, args.condition)
                if args.condition else None)
        if args.condition and cand is None:
            print("  ! the instruction names a class or predicate EGTR has no "
                  "index for")
            return None

        # `-` is the single-view baseline; the rest are the method's channels
        # switched on one at a time, all off the SAME render pass.
        rows = {}
        for label, objscore, beta, use_c, use_r in (
                ("single", "none", 0.0, False, False),
                ("A", args.objscore, 0.0, False, False),
                ("A+C+R", args.objscore, 0.0, True, True)):
            rank, stats = fuse(built, beta, objscore, weights)
            if cand:
                built_row = dict(built)
                if objscore != "none":
                    from lib.fusion import channels as ch
                    built_row["s"] = ch.object_scores(
                        built["rec"], mode=objscore,
                        class_mode=OBJSCORE_CLASS, cos=OBJSCORE_COS)
                act = decide(built_row, egtr, task, rank, cand, geo, args.iou)
            triplets = triplets_of(built, rank, egtr, args.topk,
                                   groups if use_c else None,
                                   relabel if use_r else None, weights, cand,
                                   (task["subject_class"],
                                    task["object_class"]) if cand else None)
            _, result = regrade(task, geo, triplets, args.iou)
            # The MATCHED triplet, not the top-ranked one.  Printing the
            # highest-scoring candidate next to a rank of 17 reads as a
            # contradiction -- "bottle near plant, grounded 17" -- when what
            # happened is that `near` outranked `behind` and the `behind` entry
            # sitting at 17 is the one that scored.
            # THE ROBOT'S CRITERION.  It scans the list for the instruction's
            # triplet and walks to the first one it finds, so success is not
            # "the right pair appears somewhere" -- it is "the FIRST entry that
            # matches the instruction is the right instance".  class_rank is
            # where the first match sits, grounded_rank where the first CORRECT
            # match sits; the robot succeeds exactly when they coincide.
            first = result["class_rank"]
            acts = (first is not None and result["grounded_rank"] == first)
            matched = (result.get("matches") or [{}])[0]
            rows[label] = {"result": result, "stats": stats,
                           "acts": acts, "first_match": first,
                           "act": act if cand else None,
                           "top": [f"{t['subject']} {t['predicate']} "
                                   f"{t['object']}" for t in triplets[:3]],
                           "matched": (f"#{matched.get('rank')} "
                                       f"{matched.get('predicate')} "
                                       f"iou {matched.get('subject_iou')}"
                                       if matched else None)}
            print(f"    {label:7s} class {str(result['class_rank']):>5}  "
                  f"grounded {str(result['grounded_rank']):>5}  "
                  f"best IoU {result['best_subject_iou']:.3f}   "
                  f"top1 {rows[label]['top'][0] if rows[label]['top'] else '-'}"
                  f"   {'ACTS OK' if acts else 'acts wrong'}"
                  + (f"\n              top-1 decision: "
                     f"{'HIT ' if act['success'] else 'miss'} "
                     f"iou {act['iou']:.2f}   correct pair at "
                     f"{act['best_rank']} of {act['candidates']}"
                     if cand else ""))
            if stats:
                print(f"            gate: {stats['views_per_pair']:.2f} of "
                      f"{stats['n_views']} views speak per pair, "
                      f"{stats['pairs_covered']:.1%} of pairs covered")

        # Self-test: beta=0 must reproduce `sgg_live.predict` exactly, or the
        # record assembly is wrong somewhere upstream of the fusion.
        direct = predict(egtr, reference, args.topk)
        rank0, _ = fuse(built, 0.0, "none")
        mine = triplets_of(built, rank0, egtr, args.topk)
        agree = sum(a["subject_query"] == b["subject_query"]
                    and a["object_query"] == b["object_query"]
                    for a, b in zip(direct, mine))
        print(f"    [self-test] beta=0 reproduces predict(): "
              f"{agree}/{len(direct)} of the top-{args.topk} pairs")

        return {"scene": case["scene"], "instruction": task["instruction"],
                "staged_occlusion": staged["occlusion"],
                "self_test": f"{agree}/{len(direct)}",
                **{k: {"class_rank": v["result"]["class_rank"],
                       "grounded_rank": v["result"]["grounded_rank"],
                       "best_subject_iou": v["result"]["best_subject_iou"],
                       "matched": v["matched"], "act": v["act"],
                       "acts": v["acts"], "first_match": v["first_match"],
                       "top3": v["top"][:3]} for k, v in rows.items()}}
    finally:
        rc.stop()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="nvs_pilot/cases_frozen.json")
    ap.add_argument("--case", type=int, nargs="*", default=[0])
    ap.add_argument("--beta", type=float, default=1.0, help="channel B weight")
    ap.add_argument("--condition", type=int, default=0, metavar="N",
                    help="restrict candidates to the top-N queries per "
                         "instructed class on each side, with the predicate "
                         "fixed to the instruction's.  Uses only what the "
                         "instruction provides; ground truth still enters only "
                         "at grading.  0 = free-form SGDet, the default.")
    ap.add_argument("--vocab-instructed", action="store_true",
                    help="condition on the predicate the INSTRUCTION names: "
                         "zero all 49 others, so pairs compete only on how "
                         "`behind` they are.  Stronger than --vocab, which "
                         "keeps `on` and therefore keeps whatever beat the "
                         "instructed predicate in the first place.  This "
                         "changes the question from `does the graph contain the "
                         "triplet` to `given the relation, can the model "
                         "localise it` -- defensible for a referring expression, "
                         "where the relation IS given, but not comparable to "
                         "any free-form SGG number.")
    ap.add_argument("--vocab", action="store_true",
                    help="channel V: zero the predicates the THOR export cannot "
                         "produce.  ON for every row, so it never appears in a "
                         "between-row difference -- it moves the baseline, it is "
                         "not a multi-view gain.")
    ap.add_argument("--dedup-iou", type=float, default=DEDUP_IOU,
                    help="channel C: box IoU + same class -> one triplet")
    ap.add_argument("--corr", default="argmax",
                    help="correspondence R reads the views through; the method's "
                         "default, not `assign`")
    ap.add_argument("--objscore", default=OBJSCORE,
                    choices=("max", "mean", "max_lowconf", "const", "randslot",
                             "none"),
                    help="channel A pooling; const and randslot are its controls")
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--max-az", type=float, default=30.0)
    ap.add_argument("--max-el", type=float, default=15.0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--out", default="nvs_pilot/fuse_live.json")
    args = ap.parse_args(argv)

    from robot.sgg_live import load_egtr

    frozen = json.load(open(args.cases))
    cases = frozen["cases"]
    band = frozen.get("occlusion_band")
    if band:
        print(f"staging band from {args.cases}: {band}")
    egtr = load_egtr()

    def save() -> None:
        """Rewrite the results file after EVERY case, not at the end.

        A seventy-case run is a quarter of an hour of THOR and EGTR, and an
        end-of-run write turns any crash into a total loss -- which happened
        twice on this machine while staging.  The file is small; the write is
        far cheaper than one case.
        """
        if not args.out or not results:
            return
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"beta": args.beta, "objscore": args.objscore,
                       "vocab": list(VOCAB) if args.vocab else None,
                       "dedup_iou": args.dedup_iou, "corr_relabel": args.corr,
                       "relabel_mode": RELABEL_MODE, "relabel_tau": RELABEL_TAU,
                       "gate": GATE_COS, "corr": CORR, "pool": POOL,
                       "objscore_class": OBJSCORE_CLASS,
                       "cases": results}, fh, indent=1)

    results = []
    for index in args.case:
        print(f"\n[{index}] {cases[index]['scene']}")
        try:
            outcome = run_case(cases[index], args, egtr, band)
        except Exception as error:                              # noqa: BLE001
            print(f"  ! {type(error).__name__}: {error}")
            outcome = None
        if outcome:
            results.append({"case": index, **outcome})
            save()

    if args.out and results:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"beta": args.beta, "objscore": args.objscore,
                       "vocab": list(VOCAB) if args.vocab else None,
                       "dedup_iou": args.dedup_iou, "corr_relabel": args.corr,
                       "relabel_mode": RELABEL_MODE, "relabel_tau": RELABEL_TAU,
                       "gate": GATE_COS, "corr": CORR, "pool": POOL,
                       "objscore_class": OBJSCORE_CLASS,
                       "cases": results}, fh, indent=1)
        print(f"\n{len(results)} cases  ->  {args.out}")
        for row in ("single", "A", "A+C+R"):
            print(f"  {row:7s} robot succeeds "
                  f"{sum(1 for r in results if r[row]['acts']):3d}/{len(results)}"
                  f"   instruction matched at all "
                  f"{sum(1 for r in results if r[row]['class_rank']):3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
