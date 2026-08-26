"""
grounding.py -- one instruction, one image, the pair it names.  The shared rule.

Two halves, and every caller needs both:

    candidates()   two class names -> a shortlist of queries per side
    rank_pairs()   rel[i, j, predicate] * w_subject[i] * w_object[j]

`eval_move.look`'s verdict, `fuse_live.decide`'s choice and `move_once`'s step
are this ranking read at different places.  The differences between them --
a pair filter, the choice of weight, channel B -- are arguments here, not four
copies of the body.

Not here: whether the answer is RIGHT (that needs ground truth, and lives in
`eval_move.look`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Words VG150 splits that an instruction does not.  p(class) sums over the set.
CLASS_ALIASES: Dict[str, Sequence[str]] = {
    "laptop": ("laptop", "screen"),
}


def class_columns(egtr, name: str) -> List[int]:
    """The probability columns that spell `name`, aliases included.

    Empty when VG150 has no word for it -- a real answer, not an error.
    """
    columns = {v: k - 1 for k, v in egtr["obj_names"].items()}
    return [columns[c] for c in CLASS_ALIASES.get(name, (name,))
            if c in columns]


def max_prob(probs):
    """`s`: each query's confidence over all classes.  numpy or torch."""
    top = probs.max(-1)
    return top.values if hasattr(top, "values") else top


def class_mass(probs, egtr, name: str):
    """p(the instruction's word) per query -- None when VG150 cannot spell it.

    `probs` may be numpy or torch; `[:, cols].sum(-1)` means the same in both.
    """
    columns = class_columns(egtr, name)
    return probs[:, columns].sum(-1) if columns else None


def candidates(probs, boxes, egtr, task: Dict[str, Any],
               width: int = 10, nms: float = 0.0):
    """The queries an INSTRUCTION licenses -> ([subjects], [objects]).

    ONE FRAME'S probabilities and boxes, never a fused record: the reference's
    for a decision, a synthesised view's own for "what would a robot standing
    THERE ground this to".  `fuse_live.conditioned` is the record-shaped wrapper.

    Two class names is all the task gives, so no oracle enters -- the model still
    has to say WHICH query is the bottle.  Free-form SGDet reached no rank at all
    in 70 of 71 cases; conditioned, every case yields one.  The two numbers are
    not comparable.

    THE PREDICATE IS NOT CONDITIONED ON -- it is the quantity under test.
    `rank_pairs` spends it, in the score.
    """
    import numpy as _np

    subject_p = class_mass(probs, egtr, task["subject_class"])
    object_p = class_mass(probs, egtr, task["object_class"])
    if subject_p is None or object_p is None:
        return None

    # By p(INSTRUCTED class), not by argmax: a query sitting on the bottle often
    # argmaxes to something else (median p(class) on these targets is 0.07).
    if not nms:
        return ([int(q) for q in _np.argsort(-subject_p)[:width]],
                [int(q) for q in _np.argsort(-object_p)[:width]])

    # `nms` deduplicates by box overlap first -- the top 10 covers a median of
    # five distinct objects.  MEASURED NEGATIVE (fuse_easy2_nms.json): rank 1
    # stays 22/40, rank <= 3 goes 37 -> 38/40, top-1 loses 1.  The query
    # outranking the target is the DISTRACTOR, which dedup cannot remove.
    from robot.task.task_find import iou

    def pick(column) -> List[int]:
        # Within the top-`width` window, never backfilling past it: shrinking
        # the list is the point.
        keep: List[int] = []
        for q in _np.argsort(-column)[:width]:
            box = boxes[int(q)].tolist()
            if all(iou(box, boxes[k].tolist()) <= nms for k in keep):
                keep.append(int(q))
        return keep

    return pick(subject_p), pick(object_p)


def pair_cells(candidates, *, boxes=None, pair_iou: float = 0.0,
               groups=None) -> List[Tuple[int, int]]:
    """The candidate pairs that are two DIFFERENT objects.

    `i != j` rejects only the same QUERY, and EGTR emits several boxes per
    object -- so without one of the tests below a pair can be one thing related
    to itself, which `rel` scores highly.  Two ways to say "different":

      groups     channel C's slot grouping, when the caller has one.
      pair_iou   box overlap.  THE CUT IS LOW BECAUSE THE BOXES NEST: a label
                 inside its own carton reads only 0.23, IoU dividing by the union.

    Neither given, the test is `i != j` -- which is what `eval_move.look` does.
    """
    subjects, objects = candidates

    def distinct(i: int, j: int) -> bool:
        if i == j:
            return False
        if groups is not None and groups[i] == groups[j]:
            return False
        if pair_iou and boxes is not None:
            from robot.task.task_find import iou

            return iou(boxes[i].tolist(), boxes[j].tolist()) < pair_iou
        return True

    return [(i, j) for i in subjects for j in objects if distinct(i, j)]


def rank_pairs(rel, candidates, predicate: int, subject_w, object_w, *,
               boxes=None, pair_iou: float = 0.0, groups=None,
               mix=None, weight: float = 0.0
               ) -> List[Tuple[float, int, int]]:
    """`[(score, subject, object)]`, best first.  The ranking, everywhere.

    WHICH WEIGHT IS THE CALLER'S CHOICE:

      s          the query's confidence over ALL classes.  `fuse_live.decide`'s,
                 and what the simulator's numbers report.  On rendered frames the
                 swap is worth 2 of 40 rankings -- `rel` spans 20 orders of
                 magnitude against `s`'s 1.5.
      p(class)   the instruction's own noun.  On a real photograph the two
                 disagree: a target at p 0.07 / s 0.09 loses to a distractor at
                 s 0.21.

    `mix` is channel B, a per-pair evidence field pooled across a sweep.  BOTH
    TERMS ARE NORMALISED WITHIN THIS CANDIDATE SET, so `weight` is a mixing
    weight and not a scale correction: 0.5 counts the channels equally, 0
    reproduces the un-mixed ranking exactly.  (Raw `rel + beta * ev` needs
    beta ~ 30 purely because one confident reference value outruns a cross-view
    mean by a decade and a half.)
    """
    cells = pair_cells(candidates, boxes=boxes, pair_iou=pair_iou,
                       groups=groups)
    if not cells:
        return []

    def base(i: int, j: int) -> float:
        return float(rel[i, j, predicate])

    if mix is None:
        scored = [(base(i, j) * float(subject_w[i]) * float(object_w[j]), i, j)
                  for i, j in cells]
    else:
        # `or 1.0` for the case every candidate reads zero.
        rel_top = max(base(i, j) for i, j in cells) or 1.0
        mix_top = max(float(mix[i, j]) for i, j in cells) or 1.0
        scored = [(((1.0 - weight) * base(i, j) / rel_top
                    + weight * float(mix[i, j]) / mix_top)
                   * float(subject_w[i]) * float(object_w[j]), i, j)
                  for i, j in cells]
    return sorted(scored, reverse=True)


#: How a single frame's ranking weights a query.  See `rank_pairs`.
#:
#:   s        its confidence over ALL classes.  Every simulator number in this
#:            repo was measured with it.
#:   class    p(the instruction's own noun), the quantity that BUILT the
#:            shortlist, spent again on the ordering.
#:   both     their product.  `s` says "the detector is sure of this box" and
#:            `class` says "it looks like the word"; the two are not the same
#:            claim and a query needs both to be the answer.
WEIGHTS = ("s", "class", "both")


def single_frame(probs, boxes, rel, egtr, task: Dict[str, Any], *,
                 width: int = 10, nms: float = 0.0, weight: str = "s",
                 pair_iou: float = 0.0) -> Optional[Dict[str, Any]]:
    """THE METRIC.  One frame, one instruction, the ranking it licenses.

    -> {"order", "candidates", "p"} or None when VG150 cannot spell a class.

    THIS IS WHAT A ROBOT IS GRADED ON, in the simulator and on a photograph
    alike: `eval_move.look` asks whether `order[0]` is the instructed instance,
    and `move_once` asks each swept view the same question.  No fusion, no
    sweep, no correspondence -- a single frame answering "what would a robot
    standing here ground this sentence to".  A robot needing twenty synthesised
    views to confirm what it is looking at has not found it.

    Ground truth never enters: grading `order` is the caller's job.
    """
    cand = candidates(probs, boxes, egtr, task, width, nms)
    if cand is None:
        return None
    subject_p = class_mass(probs, egtr, task["subject_class"])
    object_p = class_mass(probs, egtr, task["object_class"])
    if weight == "class":
        sub_w, obj_w = subject_p, object_p
    elif weight == "both":
        s = max_prob(probs)
        sub_w, obj_w = subject_p * s, object_p * s
    else:
        sub_w = obj_w = max_prob(probs)
    order = rank_pairs(rel, cand, egtr["rel_names"].index(task["predicate"]),
                       sub_w, obj_w, boxes=boxes, pair_iou=pair_iou)
    return {"order": order, "candidates": cand, "p": (subject_p, object_p)}


def per_view_answers(built, rendered, task, egtr, *, width: int = 10,
                     nms: float = 0.0, weight: str = "s",
                     pair_iou: float = 0.0
                     ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """WHAT EACH FRAME WOULD ANSWER ON ITS OWN -> (per-view rows, the reference's).

    Every frame is treated as a photograph somebody standing there took: its OWN
    candidates, its OWN `rel[i, j, predicate]`, its OWN boxes.  No fusion and no
    correspondence -- each answers "what would a robot standing HERE ground this
    to", so the reference row is the answer before moving.

    THE REFERENCE IS SCORED BY THE SAME CODE AS THE VIEWS, which it was not: it
    used to be handed `decide_step`'s pair, weighted by `ch.object_scores` and so
    already carrying all 20 views' evidence.  The "before" panel was then the
    best informed tile on the sheet.

    THE ROWS ARE THE INTERFACE: they go into `step.json` and they are all
    `viz.sweep.answer_sheet` needs, so the figure can show nothing the record
    does not.
    """
    predicate = egtr["rel_names"].index(task["predicate"])
    import torch

    from lib.fusion import channels as ch

    def best_pair(probs, boxes, rel):
        """The frame's own top-1 -> (score, i, j, p_subject, p_object) or None.

        `single_frame` is THE METRIC, the same call `eval_move.look` grades a
        walked-to pose with.
        """
        read = single_frame(probs, boxes, rel, egtr, task, width=width,
                            nms=nms, weight=weight, pair_iou=pair_iou)
        if read is None or not read["order"]:
            return None
        best = read["order"][0]
        p_sub, p_obj = read["p"]
        return (*best, float(p_sub[best[1]]), float(p_obj[best[2]]))

    def row(own, boxes, **fields) -> Dict[str, Any]:
        """One frame's answer, or its absence, in the shape `step.json` carries."""
        return {**fields,
                "own_top1": None if own is None else own[1:],
                "own_subject_box": None if own is None else
                [round(t, 1) for t in boxes[own[1]].tolist()],
                "own_object_box": None if own is None else
                [round(t, 1) for t in boxes[own[2]].tolist()],
                "own_p": None if own is None else
                [round(own[3], 3), round(own[4], 3)]}

    ref_own = best_pair(torch.as_tensor(built["probs_ref"]).float(),
                        built["boxes"], built["rel"])
    reference_row = row(ref_own, built["boxes"], v=None)

    report = []
    for entry in built["rec"]["views"]:
        v = int(entry["v"])
        probs = entry["probs"].float()
        boxes = entry["boxes"]
        # `record` stores each view's relation field sparsely; VIEW_TOPK is
        # 200 x 200, so nothing is actually dropped and this is the full field.
        nq = probs.shape[0]
        rel_v = torch.zeros(nq, nq, entry["vval"].shape[-1])
        idx = entry["vidx"].long()
        rel_v[idx[:, 0], idx[:, 1]] = entry["vval"].float()

        pose = rendered[v]["pose"]
        report.append(row(best_pair(probs, boxes, rel_v), boxes, v=v,
                          azimuth=round(float(pose["azimuth"]), 1),
                          elevation=round(float(pose["elevation"]), 1),
                          # A view that re-renders the pose the robot is already
                          # at.  The figure says so; nothing else reads it.
                          duplicate=v in ch.SKIP_VIEWS))
    return report, reference_row
