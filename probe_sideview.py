"""
probe_sideview.py -- can the sweep say WHICH SIDE to walk to?

`probe_viewpoint.py` established the prize.  On `cases_hard`, at radius 1.5,
39 of 40 cases are "one side reaches rank 1, the other side reaches nothing",
both sides are reachable (measured by pure teleport, no perception involved), and
the oracle side is 17 left / 22 right.  So choosing the side is a binary decision
worth 20.5/40 against 40/40 -- nothing else in this project has that leverage.
(`left` means POSITIVE azimuth throughout this file; see `side_of`.)

The sweep already looks BOTH WAYS: 18 views spanning +-30 degrees of azimuth,
half on each side.  It does not have to predict an unseen viewpoint, only compare
two sets of pictures it already holds.  This probe asks whether any GT-free score
over those views recovers the oracle side.

SCORED AT THE TRIPLET, NOT THE OBJECT, because the triplet is the object of
study: for each view, the reference frame's OWN conditioned candidates are
re-scored through that view's mapped relation field, and the view is judged by
what it does to the instructed pair.

    raw      the view's top-1 score.  Expected to be noise and included as the
             control: the same pair's `rel` varies by ~17 orders of magnitude
             ACROSS views, equally for an occluded target, its unoccluded twin,
             and the symmetric `near`, so a cross-view comparison of absolute
             values reads numerical scale rather than what was visible.
    margin   top-1 over the best candidate naming a DIFFERENT subject, WITHIN
             the view.  A ratio inside one view, so that scale factor cancels --
             which is the whole reason to prefer it.

Chance is 51% (20.5/40).  A rule at 75% would be worth wiring into the policy;
a rule at 55% would say the sweep does not carry side information at all.

    python probe_sideview.py --cases datasets/robot/cases_hard.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def side_of(azimuth: float) -> Optional[str]:
    """`left` / `right`, or None for the two views that reproduce the reference.

    POSITIVE AZIMUTH IS THE ROBOT'S LEFT.  Checked against
    `nvs_lemniscate.camera_for`: at az +30 the camera lands at -x, and a robot
    facing +z has its left hand there.  This file had the two names swapped
    until 2026-08-14.  Nothing it ever REPORTED was wrong -- every figure counts
    whether a rule's side matches the oracle's, and both were named by the same
    swapped convention, so the agreement counts are unchanged -- but the words
    printed next to them were, which is worse than useless in a log somebody
    reads a month later.
    """
    if abs(azimuth) < 1e-6:
        return None
    return "left" if azimuth > 0 else "right"


def score_views(built, egtr, cand, task, rendered, args,
                truth=None) -> List[Dict[str, Any]]:
    """Per view: its top-1 among the reference's candidates, and the margin.

    `truth` grades that top-1.  The mapped field lives in REFERENCE slot space,
    so the reference frame's own ground-truth boxes are the right yardstick, and
    `hit` answers the question that matters more than the side: read through this
    view, IS the instructed triplet the top-ranked pair?
    """
    import torch

    from lib.fusion import channels as ch
    from fuse_live import GATE_COS, CORR
    from robot.task.task_find import iou
    from typing import Dict as _D

    subjects, objects = cand
    predicate = egtr["rel_names"].index(task["predicate"])

    # SLOTS MERGED INTO OBJECTS, by box overlap alone.  A detector emits many
    # mutually-containing boxes per object, so a vote over raw slot pairs splits
    # one answer across several: on `tabletop|1` the correct triplet arrived as
    # (181,54), (181,83) and (43,83) -- three "different" pairs naming one cup
    # and one laptop -- and lost 1-1-1 to a single wrong pair with 4 votes.
    #
    # NOT `slot_groups_xyxy`, which also requires the same argmax class: two
    # slots on one cup can argmax to `cup` and `bottle`, and that rule keeps them
    # apart, which is exactly the case this has to merge.
    group: _D[int, int] = {}
    for q in list(subjects) + list(objects):
        for seen, g in group.items():
            if iou(built["boxes"][q].tolist(),
                   built["boxes"][seen].tolist()) >= args.iou:
                group[q] = g
                break
        else:
            group[q] = len(set(group.values()))
    s = built["s"].float()
    boxes = built["boxes"]
    nq = built["rec"]["s_ref"].shape[0]
    hn = torch.nn.functional.normalize(built["rec"]["h_ref"].float(), dim=-1)
    # The two class columns the instruction names.  `min` over the two endpoints
    # is deliberate: whichever end is the hard one binds the pair's score, so the
    # rule does not need to be told that `behind` hides the SUBJECT.
    classes = {v: k - 1 for k, v in egtr["obj_names"].items()}
    col_s = classes.get(task["subject_class"])
    col_o = classes.get(task["object_class"])
    ref_p = built["probs_ref"]

    out = []
    for index, view in enumerate(built["rec"]["views"]):
        if int(view["v"]) in ch.SKIP_VIEWS:
            continue
        field, ok = ch._pair_field(view, nq, hn, CORR, GATE_COS)
        # `_pair_field` keeps `ok` but not `match`, and not the cosine itself.
        # Both are recorded here so GATE_COS becomes an OFFLINE knob: re-scoring
        # at 0.75 must not cost another sweep.
        match, _ = ch.correspond(hn, view["h"], CORR, GATE_COS)
        cos_all = hn @ torch.nn.functional.normalize(view["h"].float(), dim=-1).T
        cos_max = cos_all.max(1).values

        def evidence(q: int, column: Optional[int]) -> float:
            """p(instructed class) at `q` READ THROUGH THIS VIEW.

            A slot that did not correspond scores 0, never dropped: failing to
            correspond means the object is not findable from here, which is the
            quantity being measured, not missing data.  Scored as the WORST
            value it can take -- measured to matter (21/40 -> 29/40 when the
            same rule treated missing as neutral instead).
            """
            if column is None or ok is not None and not bool(ok[q]):
                return 0.0
            return float(view["probs"].float()[int(match[q]), column])

        def at_ref(q: int, column: Optional[int]) -> float:
            return 0.0 if column is None else float(ref_p[q, column])

        pair_gain = max(
            (min(evidence(i, col_s), evidence(j, col_o))
             - min(at_ref(i, col_s), at_ref(j, col_o))
             for i in subjects for j in objects if i != j), default=None)
        scored = []
        for i in subjects:
            for j in objects:
                if i == j:
                    continue
                if ok is not None and not (bool(ok[i]) and bool(ok[j])):
                    continue
                value = float(field[i, j, predicate]) * float(s[i]) * float(s[j])
                if value > 0:
                    scored.append((value, int(i), int(j)))
        # A VIEW THAT SCORED NOTHING STILL GETS A ROW.  It corresponded to none
        # of the reference's candidates, which is a fact ABOUT that viewpoint --
        # and it is the fact that holds on the side the target is hidden on.
        # `continue` here is the one line that pre-sorted the sample toward the
        # answer once already; see PIPELINE.md.
        scored.sort(reverse=True)
        top = ti = rival = None
        if scored:
            top, ti, _ = scored[0]
            # The runner-up must name a DIFFERENT subject: EGTR emits several
            # boxes per object, so the raw rank 2 is usually the same thing again
            # and its score is nearly identical whatever rank 1 is.
            rival = next((v for v, a, _ in scored[1:]
                          if iou(boxes[a].tolist(),
                                 boxes[ti].tolist()) < args.iou), None)
        graded = bool(truth and truth.get("target") and truth.get("landmark"))

        def covers(a: int, b: int) -> bool:
            return (iou(boxes[a].tolist(), truth["target"]) >= args.iou
                    and iou(boxes[b].tolist(), truth["landmark"]) >= args.iou)

        # Is this view's top-1 the instructed triplet?  Both endpoints, as
        # everywhere else.  A view that named nothing is a MISS, not a None:
        # it did not find the triplet.
        hit = (covers(scored[0][1], scored[0][2]) if scored else False) \
            if graded else None
        # And where the correct pair sits in this view's own ranking.
        rank = next((r for r, (_, a, b) in enumerate(scored, 1)
                     if covers(a, b)), None) if graded else None
        out.append({
            "v": int(view["v"]),
            "azimuth": float(rendered[index]["pose"]["azimuth"]),
            "raw": top,
            "margin": (math.log10(top) - math.log10(rival))
                      if (top and rival) else None,
            "pairs": len(scored), "hit": hit, "rank": rank,
            # Per ROLE, so a rule need not assume which end is the hard one.
            "subj_ok": sum(1 for q in subjects if ok is None or bool(ok[q])),
            "obj_ok": sum(1 for q in objects if ok is None or bool(ok[q])),
            # THE GATE ITSELF, per candidate, in the candidate list's own order.
            # The counts above are one summary of it; storing the vector means a
            # rule over any SUBSET -- the top-5 scored pairs, one slot per
            # object, the tail of the class ranking -- is an offline re-score
            # rather than another sweep.  Three 40-case runs went on adding one
            # scalar at a time before this.
            "ok_subj": [bool(ok[q]) if ok is not None else True
                        for q in subjects],
            "ok_obj": [bool(ok[q]) if ok is not None else True
                       for q in objects],
            # THE COUNT DEDUPLICATED TO OBJECTS, which is the one with a reading:
            # how many DISTINCT candidate triplets are still in play from here.
            # The raw 100-pair count is ~8 distinct triplets padded out by EGTR's
            # duplicate boxes (measured on the reference frame), so it weights an
            # object by how many queries the detector spent on it.
            #
            # NOT a measure of the answer's quality -- a candidate absent from
            # this view can never be selected FROM this view, so this is the
            # necessary condition, and it needs no idea which triplet is right.
            "live_triplets": len({(group[i], group[j])
                                  for i in subjects for j in objects
                                  if group[i] != group[j]
                                  and (ok is None
                                       or (bool(ok[i]) and bool(ok[j])))}),
            "subj_obj": len({group[q] for q in subjects
                             if ok is None or bool(ok[q])}),
            "obj_obj": len({group[q] for q in objects
                            if ok is None or bool(ok[q])}),
            # THE SAME COUNT DEDUPLICATED BY OBJECT.  The top 10 slots per role
            # cover a median of five distinct objects, so a raw slot count
            # weights an object by how many duplicate queries EGTR spent on it
            # -- a detector habit, not visibility.  If the signal survives here
            # it is about what was VISIBLE; if it does not, the raw count was
            # riding on duplicates.
            "subj_obj": len({group[q] for q in subjects
                             if ok is None or bool(ok[q])}),
            "obj_obj": len({group[q] for q in objects
                            if ok is None or bool(ok[q])}),
            # The cosine itself, per candidate slot: GATE_COS offline.
            "cos_subj": [round(float(cos_max[q]), 3) for q in subjects],
            "cos_obj": [round(float(cos_max[q]), 3) for q in objects],
            "pair_gain": pair_gain,
            # WHICH pair this view chose.  The consensus over these is the only
            # GT-free proxy for `hit` we have: on the side where the target is
            # visible every view picks the same slot, and on the side where it is
            # hidden each picks its own noise.  Reported as the pair, so the
            # agreement can be turned into a DISTRIBUTION over azimuth rather
            # than a binary side.
            "top1": [int(scored[0][1]), int(scored[0][2])] if scored else None,
            # The same triplet at OBJECT level, which is what a vote must use.
            "top1_group": [group[scored[0][1]], group[scored[0][2]]]
                          if scored else None,
        })
    # THE INVARIANT: every non-skipped view has a row.  A missing row means some
    # filter is deciding which viewpoints get a say, which is how this
    # measurement went wrong before.
    wanted = sum(1 for v in built["rec"]["views"]
                 if int(v["v"]) not in ch.SKIP_VIEWS)
    assert len(out) == wanted, f"{wanted - len(out)} views vanished"
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="datasets/robot/cases_hard.json")
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--oracle", default="results/probe_viewpoint_r15.json",
                    help="which side reaches rank 1, from probe_viewpoint.py")
    ap.add_argument("--condition", type=int, default=10)
    ap.add_argument("--cand-nms", type=float, default=0.0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--max-az", type=float, default=30.0)
    ap.add_argument("--max-el", type=float, default=15.0)
    ap.add_argument("--bearing", default="bin")
    ap.add_argument("--trust", type=int, default=10)
    ap.add_argument("--radius-scale", type=float, default=1.5)
    ap.add_argument("--save-trail", default=None)
    # SQUARE, because SEVA works on a square latent grid and a 4:3 input is
    # letterboxed into it.  THOR's `fieldOfView` is VERTICAL, so 60 degrees at
    # 800x600 was 75.6 degrees WIDE and at 600x600 is 60 -- the robot sees a
    # narrower slice of the room, and the case lists were staged at the old
    # aspect.  Recorded in the output either way.
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--out", default="results/probe_sideview.json")
    args = ap.parse_args(argv)

    from eval_move import perceive
    from fuse_live import conditioned, task_for
    from robot.world.proc_scene import Robot, open_room, rebuild
    from robot.sgg_live import load_egtr

    # The oracle side, from the viewpoint sweep.
    truth: Dict[str, Optional[str]] = {}
    if os.path.exists(args.oracle):
        ora = json.load(open(args.oracle))["cases"]
        for row in ora:
            # Same convention as `side_of`: positive azimuth is the LEFT.
            left = any(v == 1 for a, v in row["ranks"].items()
                       if float(a) > 0)
            right = any(v == 1 for a, v in row["ranks"].items()
                        if float(a) < 0)
            truth[row["scene"]] = ("left" if left and not right else
                                   "right" if right and not left else None)

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
            # `perceive` renders and records the sweep; it also returns P-hat,
            # which this probe does not need but the call does.
            got = perceive(rc, case, task, egtr, args, want_record=True)
            if got is None or got.get("built") is None:
                print(f"  [{index}] {case['scene']} unreadable", flush=True)
                continue
            built, rendered = got["built"], got["rendered"]
            cand = conditioned(built, egtr, task, args.condition, args.cand_nms)
            if cand is None:
                continue
            from eval_move import truth_boxes
            views = score_views(built, egtr, cand, task, rendered, args,
                                truth_boxes(rc, task))
            # THE REFERENCE FRAME'S OWN RANKING, once per case, so a "top-N"
            # subset is an offline choice.  `fuse_live.decide`'s score exactly.
            # GT-free: the instruction supplies two class names and nothing else.
            predicate = egtr["rel_names"].index(task["predicate"])
            rel, s = built["rel"], built["s"].float()
            scored_ref = sorted(
                ((float(rel[i, j, predicate]) * float(s[i]) * float(s[j]),
                  int(i), int(j))
                 for i in cand[0] for j in cand[1] if i != j), reverse=True)
            row = {"scene": case["scene"], "oracle": truth.get(case["scene"]),
                   # The candidate lists themselves, so `ok_subj[k]` can be
                   # traced back to a slot without re-deriving `conditioned`.
                   "subjects": [int(q) for q in cand[0]],
                   "objects": [int(q) for q in cand[1]],
                   # THE CANDIDATE BOXES, so the grouping IoU is an offline
                   # knob.  Merging slots into objects at 0.5 is what split the
                   # `p_hat` vote three ways on `tabletop|1`; re-deciding that
                   # threshold should not cost another sweep.  Grading stays at
                   # `--iou` -- the truth's yardstick is not a knob.
                   "subj_boxes": [[round(v, 1) for v in
                                   built["boxes"][q].tolist()]
                                  for q in cand[0]],
                   "obj_boxes": [[round(v, 1) for v in
                                  built["boxes"][q].tolist()]
                                 for q in cand[1]],
                   # Top 20 pairs is enough for any top-N rule worth trying and
                   # keeps the file readable.
                   "ref_pairs": [[v, i, j] for v, i, j in scored_ref[:20]],
                   "views": views}
            for rule in ("raw", "margin"):
                usable = [v for v in views
                          if v[rule] is not None and side_of(v["azimuth"])]
                best = max(usable, key=lambda v: v[rule]) if usable else None
                row[rule] = side_of(best["azimuth"]) if best else None
            rows.append(row)
            hits = sum(1 for v in views if v.get("hit"))
            print(f"  [{index}/{len(cases)}] {case['scene']:13} "
                  f"oracle {str(row['oracle']):5} raw {str(row['raw']):5} "
                  f"margin {str(row['margin']):5}  "
                  f"views whose top-1 is right: {hits}/{len(views)}", flush=True)
        except Exception as error:                              # noqa: BLE001
            import traceback
            print(f"  [{index}] ! {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
        finally:
            controller.stop()
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)),
                        exist_ok=True)
            json.dump({"cases": rows}, open(args.out, "w"), indent=1)

    graded = [r for r in rows if r["oracle"]]
    if not graded:
        return 1
    print(f"\n  {len(graded)} cases with a one-sided oracle "
          f"({sum(1 for r in graded if r['oracle'] == 'left')} left, "
          f"{sum(1 for r in graded if r['oracle'] == 'right')} right)\n")
    print(f"  {'rule':8} {'agrees':>10} {'rate':>7}")
    for rule in ("raw", "margin"):
        hit = sum(1 for r in graded if r[rule] == r["oracle"])
        said = sum(1 for r in graded if r[rule])
        print(f"  {rule:8} {hit:>7}/{said:<3} {hit / max(said, 1):>6.0%}")
    # THE PRIOR QUESTION: read through a view, is the triplet the top pair?
    allv = [v for r in rows for v in r["views"] if v.get("hit") is not None]
    if allv:
        hit = sum(1 for v in allv if v["hit"])
        any_view = sum(1 for r in rows
                       if any(v.get("hit") for v in r["views"]))
        print(f"\n  ONE VIEW'S OWN TOP-1, graded against the reference frame:")
        print(f"    per view                {hit}/{len(allv)} "
              f"({hit / len(allv):.0%})")
        print(f"    SOME view gets it       {any_view}/{len(rows)}")
        print(f"    (reference frame alone is 13/40 on this list)")
    print(f"\n  chance is 50%; picking one side always would score "
          f"{max(sum(1 for r in graded if r['oracle'] == s) for s in ('left', 'right'))}"
          f"/{len(graded)}")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
