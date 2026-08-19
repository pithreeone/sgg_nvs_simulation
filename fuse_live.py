"""
fuse_live.py -- run the paper's map-back fusion on one live THOR frame + its sweep.

The offline study feeds `lib/fusion/channels.py` from a cache of PNGs on disk.
The cache is a schema, not a mechanism, so this file assembles the SAME record
in memory and calls the very same functions.

    A (objscore)   pool each query's object score over the views.  Its OWN row,
                   so a single number cannot hide which channel moved.
    C (dedup)      box IoU + same argmax class -> one triplet.
    R (relabel)    cross-view majority over the predicate.  Reaches
                   `triplets_of` ONLY, never `decide`, so it can move
                   `robot succeeds` and not `top-1 decision`.
    B, D           not in use.  B's weight was hardcoded to 0 for every arm;
                   D is a prior fitted on labels.

Views 9 and 19 are dropped by `channels.SKIP_VIEWS`: the lemniscate returns to
zero offset at t=pi and t=2pi, so counting them manufactures cross-view
agreement out of a copy of the input.

Two metrics per arm, answering different questions -- see PIPELINE.md.
`top-1 decision` ranks the conditioned pairs by `rel[behind] * s * s`;
`robot succeeds` additionally demands the model UTTER the instructed predicate.

    python fuse_live.py --cases datasets/robot/cases_easy2.json \
        --condition 10 --case $(seq 0 39)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Sparse width of a view's own relation field.  The disk cache stores 500; the
#: pair an INSTRUCTION names is nowhere near any view's 500 strongest (measured:
#: 0, 0, 0 and 1 views speaking about the chosen pair).  Nothing is cached here,
#: so keeping every pair costs 4 MB per view.
VIEW_TOPK = 40000

#: `logs/occlusion_ds4_refpred_assign.log`.  Recorded in the output; `CORR` is
#: also imported by `probe_sideview.py` and `probe_viewdist.py`.
GATE_COS = 0.80
CORR = "assign"
POOL = "mean"

#: Channel A.  `max` pools the single best view, where channels.py measures its
#: gain.
OBJSCORE = "max"
#: `gate` requires a view's argmax class to equal the reference's before it may
#: donate a score.  On `cases_hard` it NEVER FIRES for an occluded target (s
#: stays at 0.066 in 5 of 8) while lifting the unoccluded twin every time, so A
#: widens the gap it exists to close; `anymax` lifts the target 4.7x instead.
#: It also contradicts `conditioned`, which ranks by p(instructed class)
#: precisely because the argmax is often something else.
OBJSCORE_CLASS = "gate"

OBJSCORE_COS = 0.80

#: Channel C: same object at box IoU above this AND the same argmax class, then
#: one triplet per (group, group, argmax predicate).  `--dedup_iou` upstream.
DEDUP_IOU = 0.90

#: Channel R.  `rl_margin` is the AGREEMENT FRACTION, not the raw count: a pair
#: many views can see is an EASY pair the reference already gets right, so
#: ranking by count selects the wrong pairs (19.2% already-correct against
#: 2.7%).  tau = 0 means "whenever any view spoke, take the majority".
RELABEL_MODE = "consensus"
RELABEL_TAU = 0.0

#: Which predicates a view may vote FOR.  `TARGET_PREDICATES`, not the ones the
#: ground truth happens to contain -- that would be peeking at test labels.
from vg.vg_conventions import TARGET_PREDICATES as RELABEL_VOCAB  # noqa: E402

from robot.nvs_lemniscate import LOOKAT_DIST  # noqa: E402


def task_for(case: Dict[str, Any]) -> Dict[str, Any]:
    """The instruction as the pipeline needs it, from a case of any list.

    THE DISTRACTOR IS OPTIONAL.  A list built without a same-class competitor
    says so with `has_distractor: False` and carries no `distractor_name`, and
    `cases_slot` is that list on purpose: with nothing else of the target's class
    in the room, "the laptop behind the box" and "any laptop" have the same
    answer.  A number measured there says nothing about relational grounding and
    must not be pooled with a list that has distractors.

    This used to be four identical literals, each indexing `distractor_name`
    unconditionally, so the first case without one raised `KeyError` before a
    single frame was rendered.
    """
    name = case.get("distractor_name")
    return {"instruction": case["instruction"],
            "predicate": case["predicate"],
            "subject_class": case["subject_class"],
            "object_class": case["object_class"],
            "target_name": case["target_name"],
            "receptacle_name": case["occluder_name"],
            "distractors": [{"name": name}] if name else []}


def sparse_pairs(torch, rel, keep: int = VIEW_TOPK):
    """`mapback_cache.topk_sparse`: keep the strongest `keep` pairs of a field."""
    n = rel.shape[0]
    strength = rel.max(-1)[0].clone()
    strength.fill_diagonal_(0.0)
    top = torch.topk(strength.flatten(), min(keep, n * n)).indices
    ii, jj = top // n, top % n
    # float32, NOT half: float16's smallest subnormal is 5.96e-08 against a
    # median relation value near 1e-18, so 74% of the instructed pairs
    # underflowed to EXACTLY ZERO and R -- which reads `field.max(-1) > 0` as
    # "this view spoke" -- was silently mute on most of the evidence.
    return torch.stack([ii, jj], 1).short().cpu(), rel[ii, jj].float().cpu()


def record(egtr, reference: np.ndarray,
           views: Sequence[Tuple[int, np.ndarray]]) -> Dict[str, Any]:
    """One in-memory `mapback_cache` record: the reference, then every view.

    Field for field what `--layout dirs --unmapped_topk` writes, because
    `channels.view_evidence_perpred` reads it by name.  `probs` is the class
    SOFTMAX and `s_ref`/`c_ref` its max; the sigmoid EGTR also emits is unused.
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


def fuse(built: Dict[str, Any], objscore: str = "none",
         class_mode: str = OBJSCORE_CLASS, objscore_cos: float = OBJSCORE_COS):
    """-> (rank [NQ, NQ], coverage stats).  Channel A, switchable off.

    `objscore="none"` is the plain single-view ranking, and the self-test: it
    must reproduce `sgg_live.predict`.
    """
    from lib.fusion import channels as ch

    s = (ch.object_scores(built["rec"], mode=objscore, class_mode=class_mode,
                          cos=objscore_cos)
         if objscore != "none" else built["s"].float())
    return ch.pair_rank(built["rel"], s, None, 0.0)[1], {}


def consensus_relabel(built: Dict[str, Any], egtr, corr: str = "argmax",
                      gate_cos: float = GATE_COS, per_view: bool = False):
    """
    R -> (rl_pred [NQ,NQ] predicate index, rl_margin [NQ,NQ] agreement fraction).

    Each view that speaks about a reference pair names one predicate; the
    majority wins and the margin is the fraction of speakers who agreed.  A view
    speaks only if its mapped field is non-zero there AND both endpoints
    correspond at `gate_cos`.  `-1` where nobody spoke, so `frac >= tau` cannot
    fire there even at tau = 0.  Ported from `occlusion_channels.py`'s consensus
    branch, which is inline in that script rather than a function in channels.py.

    `per_view=True` also returns, per view, WHICH predicate it named and whether
    it spoke.  R discards that after the vote, but a view index is an azimuth on
    the lemniscate, so it is a DIRECTION -- what `eval_move.perceive` needs.
    """
    import torch

    from lib.fusion import channels as ch

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
    ballots = []
    for view in rec["views"]:
        if int(view["v"]) in ch.SKIP_VIEWS:
            continue
        field, ok = ch._pair_field(view, nq, hn, corr, gate_cos)
        field = field * mask
        spoke = field.max(-1)[0] > 0
        if ok is not None:
            spoke = spoke & (ok[:, None] & ok[None, :])
        named = field.argmax(-1)
        votes.scatter_add_(-1, named.unsqueeze(-1), spoke.float().unsqueeze(-1))
        spoke_n += spoke.float()
        if per_view:
            ballots.append({"v": int(view["v"]), "named": named, "spoke": spoke,
                            "strength": field.max(-1)[0]})
    top, which = votes.max(-1)
    margin = torch.where(spoke_n > 0, top / spoke_n.clamp(min=1.0),
                         torch.full_like(spoke_n, -1.0))
    return (which, margin, ballots) if per_view else (which, margin)


#: VG150 classes that name the same thing, so the instruction's word covers all
#: of them.  A softmax splits its mass across mutually exclusive labels, and
#: where two labels are not really exclusive that split is a loss the pipeline
#: pays twice -- once when the slot fails to reach the candidate list, and again
#: when its score is halved.  Summing the columns undoes it: the classes are
#: disjoint events, so `p(laptop or screen)` IS the sum.
#:
#: MEASURED ON `cases_slot`: over the poses where a slot covers the laptop, EGTR
#: called it `laptop` twice and `screen` twice out of seven, and the instructed
#: class fell outside the slot's own top TEN classes at three of them -- so the
#: target could not enter a candidate list gated on p(laptop) at all.
#:
#: THE TABLE IS SEMANTIC AND FIXED IN ADVANCE.  `screen` is the part of a laptop
#: that faces the camera; that is the whole argument, and it is the only kind
#: admitted here.  The other things EGTR called the landmark -- `board`, `tile`,
#: `bottle`, `door` -- are plain mistakes and are NOT aliased, however much
#: adding them would help.  Aliases chosen by which ones raise a score are the
#: same error as tuning on the test set.
#:
#: GRADING NEVER SEES THIS.  `top1_correct` overlaps boxes with the ground truth
#: and reads no class at all, so the metric is not loosened; and both arms and
#: every rule share `conditioned`, so no comparison is tilted.
CLASS_ALIASES: Dict[str, Sequence[str]] = {
    "laptop": ("laptop", "screen"),
}


def conditioned(built: Dict[str, Any], egtr, task: Dict[str, Any],
                width: int = 10, nms: float = 0.0):
    """The candidate pairs an INSTRUCTION licenses -> ([subjects], [objects]).

    Two class names is all the task provides, so this uses no oracle: the model
    still has to say WHICH query is the bottle.  Free-form SGDet reached no rank
    at all in 70 of 71 cases; conditioned, every case yields one.  NOT comparable
    to any free-form SGG number.

    THE PREDICATE IS NOT CONDITIONED HERE -- it is the quantity under test, so
    fixing it would hand over the answer.  `decide` does spend it, in the SCORE.
    """
    import numpy as _np

    classes = {v: k - 1 for k, v in egtr["obj_names"].items()}
    probs = built["probs_ref"]

    def mass(name: str):
        """p(the instruction's word), summed over the classes that mean it."""
        columns = [classes[c] for c in CLASS_ALIASES.get(name, (name,))
                   if c in classes]
        return probs[:, columns].sum(-1) if columns else None

    subject_p = mass(task["subject_class"])
    object_p = mass(task["object_class"])
    if subject_p is None or object_p is None:
        return None

    # By p(INSTRUCTED class), not by argmax: a query sitting on the bottle often
    # argmaxes to something else (median p(class) on these targets is 0.07).
    if not nms:
        subjects = _np.argsort(-subject_p)[:width]
        objects = _np.argsort(-object_p)[:width]
        return [int(q) for q in subjects], [int(q) for q in objects]

    # ONE SLOT PER OBJECT: the subject side's top 10 covers a median of FIVE
    # distinct objects, so the 100-pair set is ~25 pairs padded out.  By BOX
    # OVERLAP ALONE -- `slot_groups_xyxy` also requires the same argmax class,
    # and two queries on one cup can argmax to `cup` and `bottle`.
    #
    # MEASURED NEGATIVE (fuse_easy2_nms.json): it does not promote the target,
    # because the query outranking it is the DISTRACTOR, a different object.
    # Rank 1 stays 22/40, rank <= 3 goes 37 to 38/40, top-1 loses 1.
    from robot.task_find import iou

    boxes = built["boxes"]

    def pick(column) -> List[int]:
        # WITHIN the top-`width` window, never backfilling past it: with 200
        # queries there are always more distinct boxes on the wall, so a
        # count-preserving dedup would swap duplicates of the right object for
        # lower-ranked boxes on wrong ones.  Shrinking the list is the point.
        keep: List[int] = []
        for q in _np.argsort(-column)[:width]:
            box = boxes[int(q)].tolist()
            if all(iou(box, boxes[k].tolist()) <= nms for k in keep):
                keep.append(int(q))
        return keep

    return pick(subject_p), pick(object_p)


def decide(built: Dict[str, Any], egtr, task: Dict[str, Any], rank,
           candidates, geo, iou_hit: float = 0.5, groups=None
           ) -> Dict[str, Any]:
    """The ONE pair a robot would act on, and whether acting on it succeeds.

        candidates   top-N by p(bottle) x top-N by p(plant)
        score        rel[i, j, behind] * s_i * s_j
        decision     argmax, one pair
        success      BOTH boxes land where the sentence says

    The instruction is spent in the SCORE, not as a filter afterwards, so
    `behind the plant` does its English job of picking out WHICH bottle.  The
    robot needs the `behind` field to be right about which pair stands in the
    relation, not for the model to UTTER the word -- that is `robot succeeds`,
    via `triplets_of`.  So R and V, which decide the LABEL, do nothing here.

    `groups` is channel C, and omitting it cost 30% of all cases: `i != j`
    rejects identical QUERY INDICES, not identical OBJECTS, EGTR emits several
    boxes per object, and `rel` scores such a self-pair highly -- a thing is
    trivially `behind` itself.  The top-1 landed on the LANDMARK in 12 of 40.
    """
    subjects, objects = candidates
    predicate = egtr["rel_names"].index(task["predicate"])
    rel = built["rel"]
    sub_w = obj_w = built["s"].float()
    scored = [(float(rel[i, j, predicate]) * float(sub_w[i]) * float(obj_w[j]),
               i, j)
              for i in subjects for j in objects
              if i != j and (groups is None or groups[i] != groups[j])]
    if not scored:
        return {"chosen": None, "success": False, "iou": 0.0}
    return _grade_choice(built, task, geo, scored, iou_hit)


def _grade_choice(built: Dict[str, Any], task: Dict[str, Any], geo,
                  scored: List[Tuple[float, int, int]],
                  iou_hit: float) -> Dict[str, Any]:
    """Grade the argmax of `scored`, whatever produced it.

    BOTH ENDPOINTS, because this is triplet detection.  Grading the subject alone
    passed 4 of 40 pairs whose object endpoint was on nothing at all -- the right
    bowl reached by way of a relation to a box that does not exist.
    """
    from robot.task_find import iou

    score, i, j = max(scored)
    truth = geo.get(task["target_name"], {}).get("bbox_visible")
    landmark = geo.get(task["receptacle_name"], {}).get("bbox_visible")
    if landmark is None:
        raise KeyError(f"no ground-truth box for the landmark "
                       f"{task['receptacle_name']!r}; grading both endpoints "
                       f"needs it, and falling back to the subject alone would "
                       f"report a weaker criterion under the same name")

    def grounded(a: int, b: int) -> bool:
        return (bool(truth)
                and iou(built["boxes"][a].tolist(), truth) >= iou_hit
                and iou(built["boxes"][b].tolist(), landmark) >= iou_hit)

    subject_iou = iou(built["boxes"][i].tolist(), truth) if truth else 0.0
    object_iou = iou(built["boxes"][j].tolist(), landmark)
    # Where the right answer sat, so a miss can be told apart from a near miss.
    ranked = sorted(scored, reverse=True)
    correct = next((r for r, (_, a, b) in enumerate(ranked, 1)
                    if grounded(a, b)), None)

    # HOW MUCH THE WINNER WON BY -- the only confidence signal left after the
    # converse ratio was ruled out, and weak at AUC 0.61.  `distinct` skips
    # runners-up that are another box on the same object.
    def other_subject(a: int) -> bool:
        return iou(built["boxes"][a].tolist(),
                   built["boxes"][i].tolist()) < iou_hit

    runner = ranked[1][0] if len(ranked) > 1 else None
    distinct = next((s for s, a, _ in ranked[1:] if other_subject(a)), None)
    return {"chosen": [int(i), int(j)], "score": score,
            "iou": round(subject_iou, 3), "iou_object": round(object_iou, 3),
            "success": grounded(i, j),
            "score_runner_up": runner, "score_other_subject": distinct,
            "best_rank": correct, "candidates": len(scored)}


def triplets_of(built: Dict[str, Any], rank, egtr, topk: int = 10,
                groups=None, relabel=None, candidates=None, as_classes=None,
                tau: float = RELABEL_TAU, order=None) -> List[Dict[str, Any]]:
    """
    A ranking matrix as the triplet list the rest of this repo speaks.

    Boxes and classes come from the REFERENCE slots for every arm.  That is what
    makes single-view and fused comparable, and what lets `task_find.grade` score
    both against the robot's own frame -- a novel view's boxes are in a frame the
    grader has no ground truth for.
    """
    from lib.pytorch_misc import argsort_desc

    rel = built["rel"]
    c_ref, boxes = built["c"], built["boxes"]
    s = built["s"].float()
    out, seen = [], set()
    if candidates is not None:
        # Only the instruction's own class pair competes.  `order` breaks ties
        # between two pairs BOTH LABELLED `behind`, which the default `rank`
        # would settle by their `on` scores.  The LABEL is untouched, so this is
        # a tie-break, not a way to force the word.
        subjects, objects = candidates
        key = rank if order is None else order
        scored = sorted(((float(key[i, j]), i, j)
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
            # C.  One triplet per (group, group, predicate), signed with the
            # REFERENCE's predicate BEFORE R, as `to_prediction` does.
            # channels.py notes that a relabel can then make two survivors
            # collide, and accepts it rather than changing the top-K.
            key = (int(groups[sub_q]), int(groups[obj_q]), predicate)
            if key in seen:
                continue
            seen.add(key)
        if relabel is not None:
            rl_pred, rl_margin = relabel
            if float(rl_margin[sub_q, obj_q]) >= tau:
                predicate = int(rl_pred[sub_q, obj_q])
        # Under conditioning the class pair is GIVEN, so the endpoints carry the
        # instructed names rather than each query's argmax: selecting a query by
        # p(bottle) and grading it as whatever it argmaxes to would make the
        # conditioning inert.  What is left to judge is the PREDICATE and the
        # INSTANCE.
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
    from eval_nvs_pointer import geometry
    from robot.nvs_lemniscate import (camera_for, lemniscate, orbit_centre,
                                      park_once, sweep)
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
        # something, which for `in front of` is the landmark, not the graded
        # instance.  Look the base task up by whichever the case says.
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
        # Staging is redone because `SetObjectPoses` renumbers objectIds and
        # THOR's settle is not bit-reproducible -- but it must use THE BAND THE
        # CASE FILE WAS FROZEN AT, not `put_in_front`'s default of 0.50, which
        # made a 0.25-0.50 case list measure a median of 0.48.
        #
        # Occlusion is always measured on the LANDMARK, the object the occluder
        # hides.  For `on`/`behind` that is the graded instance too, but `in
        # front of` moves grading to the occluder, and measuring the occluder
        # against itself reports 0% hidden and stages nothing.
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

        # A GIVEN trajectory: a fixed distance down the optical axis, no
        # detection in the loop.  See `orbit_centre`.
        centre = orbit_centre(rc, args.lookat)

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

        # Grade on the robot's own frame, the only frame the ground truth is
        # defined in.  Both arms share it.
        rc.teleport(position={"x": float(camera[0]),
                              "y": float(rc.agent_position["y"]),
                              "z": float(camera[2])})
        # The landmark is in here because `decide` grades both endpoints.
        geo = geometry(rc, [task["target_name"], task["receptacle_name"]]
                       + [d["name"] for d in task.get("distractors") or []],
                       names, False)

        return score_case(built, egtr, task, geo, reference, args,
                          case["scene"], staged["occlusion"])
    finally:
        rc.stop()


def run_case_proc(case: Dict[str, Any], args, egtr) -> Optional[Dict[str, Any]]:
    """
    The same experiment on a procedural tabletop case.  Three substitutions:

      the scene   `rebuild` restores the record exactly -- no `stage_at`, no
                  settle, no scene snapshot to diff.
      the task    written from the case fields.  `build_tasks` exists to FIND an
                  unambiguous instruction in a room somebody else furnished;
                  here the instruction is what the scene was built to satisfy.
      the boxes   `visible_box` per instance name, which unlike `geometry` keeps
                  the target and its identical twin apart.  Merging them would
                  grade the robot correct for walking to either.

    The sweep drives a third-party camera, so it never needed the navmesh this
    room lacks.  `park_once` did, and is replaced by parking in a corner -- the
    point is only that the robot's shadow stays put across all views.
    """
    from robot.nvs_lemniscate import (camera_for, lemniscate, orbit_centre,
                                      sweep)
    from robot.proc_scene import (ROOM, View, look_from, open_room, rebuild,
                                  visible_box)

    controller = open_room(args.width, args.height, args.fov)
    try:
        event = rebuild(controller, case)
        rc = View(event)
        rc.controller = controller

        task = task_for(case)
        print(f"  {task['instruction']}   target {task['target_name']}")

        reference = event.frame.copy()
        camera = rc.camera_xyz.copy()
        centre = orbit_centre(rc, args.lookat)

        poses = [camera_for(centre, camera, az, el)
                 for az, el in lemniscate(args.views, args.max_az, args.max_el)]
        corner = min(((0.3, 0.3), (0.3, ROOM - 0.3), (ROOM - 0.3, 0.3),
                      (ROOM - 0.3, ROOM - 0.3)),
                     key=lambda p: -math.dist(p, (camera[0], camera[2])))
        look_from(controller, corner[0], corner[1], 0.0, 0.0, force=True)
        rendered = sweep(rc, poses, task["target_name"], args.fov, [], True)

        built = record(egtr, reference,
                       [(i, r["frame"]) for i, r in enumerate(rendered)])

        # Back to the robot's own pose: grading happens on the frame the ground
        # truth is defined in, and both arms share it.
        event = rebuild(controller, case)
        geo = {name: {"bbox_visible": visible_box(event, name)}
               for name in (case["target_name"], case["occluder_name"],
                            *(d["name"] for d in task["distractors"]))}
        return score_case(built, egtr, task, geo, reference, args,
                          case["scene"], case["staged_occlusion"])
    finally:
        controller.stop()


def score_case(built, egtr, task: Dict[str, Any], geo: Dict[str, Any],
               reference, args, scene: str, occlusion: float
               ) -> Optional[Dict[str, Any]]:
    """
    Everything downstream of the render pass: fuse, ground, and score the arms.

    The iTHOR and procedural paths differ ONLY in how they get here.  From
    `built` onward they must stay the same computation, or the tabletop numbers
    stop being comparable to the ones already reported.
    """
    from eval_nvs_pointer import regrade
    from lib.fusion import channels as ch
    from robot.sgg_live import predict

    groups = ch.slot_groups_xyxy(built["boxes"], built["c"].long(),
                                 args.dedup_iou)
    relabel = consensus_relabel(built, egtr, args.corr)
    cand = (conditioned(built, egtr, task, args.condition, args.cand_nms)
            if args.condition else None)
    if args.condition and cand is None:
        print("  ! the instruction names a class or predicate EGTR has no "
              "index for")
        return None

    # `single` is the single-view baseline; the rest are the method's channels
    # switched on one at a time, all off the SAME render pass.
    arms = [("single", "none", False, False),
            ("A", args.objscore, False, False),
            ("A+C+R", args.objscore, True, True)]

    rows = {}
    for label, objscore, use_c, use_r in arms:
        rank, stats = fuse(built, objscore)
        act, order = None, None
        built_row = dict(built)
        if objscore != "none":
            built_row["s"] = ch.object_scores(
                built["rec"], mode=objscore,
                class_mode=args.objscore_class, cos=OBJSCORE_COS)
        if cand:
            # C only where the arm claims C, or the arms stop being comparable.
            act = decide(built_row, egtr, task, rank, cand, geo, args.iou,
                         groups if use_c else None)
            if args.rank_by == "instructed":
                # Same s the arm's own ranking used, so switching the ordering
                # does not quietly switch the object scores too.
                s_used = built_row["s"].float()
                order = (built["rel"][:, :,
                                      egtr["rel_names"].index(task["predicate"])]
                         * s_used[:, None] * s_used[None, :])
        triplets = triplets_of(built, rank, egtr, args.topk,
                               groups if use_c else None,
                               relabel if use_r else None, cand,
                               (task["subject_class"],
                                task["object_class"]) if cand else None,
                               order=order)
        _, result = regrade(task, geo, triplets, args.iou)
        # THE ROBOT'S CRITERION: it walks to the FIRST entry matching the
        # instruction, so success is not "the right pair appears somewhere".
        # class_rank is where the first match sits, grounded_rank where the first
        # CORRECT match sits; the robot succeeds exactly when they coincide.
        first = result["class_rank"]
        acts = (first is not None and result["grounded_rank"] == first)
        # The MATCHED triplet, not the top-ranked one: printing the
        # highest-scoring candidate beside a rank of 17 reads as a contradiction
        # when what happened is that `near` outranked `behind`.
        matched = (result.get("matches") or [{}])[0]
        rows[label] = {"result": result, "stats": stats,
                       "acts": acts, "first_match": first, "act": act,
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

    # Self-test: with no channels on, the assembly must reproduce
    # `sgg_live.predict` exactly.
    direct = predict(egtr, reference, args.topk)
    rank0, _ = fuse(built, "none")
    mine = triplets_of(built, rank0, egtr, args.topk)
    agree = sum(a["subject_query"] == b["subject_query"]
                and a["object_query"] == b["object_query"]
                for a, b in zip(direct, mine))
    print(f"    [self-test] beta=0 reproduces predict(): "
          f"{agree}/{len(direct)} of the top-{args.topk} pairs")

    return {"scene": scene, "instruction": task["instruction"],
            "staged_occlusion": occlusion,
            "self_test": f"{agree}/{len(direct)}",
            **{k: {"class_rank": v["result"]["class_rank"],
                   "grounded_rank": v["result"]["grounded_rank"],
                   "best_subject_iou": v["result"]["best_subject_iou"],
                   "matched": v["matched"], "act": v["act"],
                   "acts": v["acts"], "first_match": v["first_match"],
                   "top3": v["top"][:3]} for k, v in rows.items()}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="datasets/robot/cases_easy2.json")
    ap.add_argument("--case", type=int, nargs="*", default=[0])
    ap.add_argument("--condition", type=int, default=0, metavar="N",
                    help="restrict candidates to the top-N queries per "
                         "instructed class on each side.  Ground truth still "
                         "enters only at grading.  0 = free-form SGDet.")
    ap.add_argument("--dedup-iou", type=float, default=DEDUP_IOU,
                    help="channel C: box IoU + same class -> one triplet")
    ap.add_argument("--corr", default="argmax",
                    help="correspondence R reads the views through; the "
                         "method's default, not `assign`")
    ap.add_argument("--objscore-class", default=OBJSCORE_CLASS,
                    choices=("gate", "anymax", "refprob"),
                    help="whether a view must agree on the argmax class before "
                         "channel A takes its object score.  See OBJSCORE_CLASS.")
    ap.add_argument("--objscore", default=OBJSCORE,
                    choices=("max", "mean", "max_lowconf", "const", "randslot",
                             "none"),
                    help="channel A pooling; const and randslot are its controls")
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--max-az", type=float, default=30.0)
    ap.add_argument("--max-el", type=float, default=15.0)
    ap.add_argument("--rank-by", choices=("best", "instructed"), default="best",
                    help="which score orders the conditioned candidates: each "
                         "pair's BEST predicate, or the INSTRUCTED predicate's "
                         "own slice.  Labels are unaffected.")
    ap.add_argument("--cand-nms", type=float, default=0.0, metavar="IOU",
                    help="deduplicate the candidate list by box overlap before "
                         "taking the top-K.  0 reproduces the published "
                         "behaviour; see `conditioned` for the measured "
                         "negative.")
    ap.add_argument("--lookat", type=float, default=LOOKAT_DIST,
                    help="metres down the optical axis the sweep orbits")
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

    results: List[Dict[str, Any]] = []

    def save() -> None:
        """Rewrite the results file after EVERY case.  A seventy-case run is a
        quarter of an hour of THOR and EGTR, and an end-of-run write turns any
        crash into a total loss -- which happened twice here while staging."""
        if not args.out or not results:
            return
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"objscore": args.objscore,
                       "dedup_iou": args.dedup_iou, "corr_relabel": args.corr,
                       "relabel_mode": RELABEL_MODE, "relabel_tau": RELABEL_TAU,
                       "gate": GATE_COS, "corr": CORR, "pool": POOL,
                       "objscore_class": args.objscore_class,
                       "cases": results}, fh, indent=1)

    for index in args.case:
        print(f"\n[{index}] {cases[index]['scene']}")
        try:
            outcome = (run_case_proc(cases[index], args, egtr)
                       if frozen.get("procedural")
                       else run_case(cases[index], args, egtr, band))
        except Exception as error:                              # noqa: BLE001
            print(f"  ! {type(error).__name__}: {error}")
            outcome = None
        if outcome:
            results.append({"case": index, **outcome})
            save()

    if args.out and results:
        print(f"\n{len(results)} cases  ->  {args.out}")
        for row in ("single", "A", "A+C+R"):
            print(f"  {row:7s} robot succeeds "
                  f"{sum(1 for r in results if r[row]['acts']):3d}/{len(results)}"
                  f"   instruction matched at all "
                  f"{sum(1 for r in results if r[row]['class_rank']):3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
