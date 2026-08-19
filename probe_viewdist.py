"""
probe_viewdist.py -- the distribution over viewpoints, and where it says to walk.

The idea under test.  Each sweep view scores the reference frame's OWN candidate
pairs through its mapped relation field and names a top-1.  Merge slots into
objects, take the most-voted pair as `p_hat`, and mark each view 1 or 0 for
whether it agrees.  Smooth that 0/1 sequence ON THE AZIMUTH AXIS and you have a
distribution over poses: argmax it to choose a heading, or sample it for a
multi-step policy.

TWO HALVES, KEPT APART ON PURPOSE
---------------------------------
    build_*   sees only what the robot sees.  No `truth` argument reaches these
              functions at all, so a ground-truth test CANNOT decide which views
              or which candidates enter the distribution.
    grade_*   sees the truth, and may only ADD a column.  It may never remove a
              row.

That split is not tidiness, it is the fix for a specific bug.  The earlier
version of this measurement inherited one line from `probe_sideview.py` --

    if not parts or not any(p["ok"] for p in parts):   # p["ok"] IS the truth
        continue

-- which deleted every view where the instructed pair failed to correspond.
Those are exactly the views looking from the side the target is HIDDEN on, so
the surviving sample was pre-sorted toward the answer.  It carried the entire
result: on the same 10 cases, removing it took views per case 12.9 -> 18.0,
per-view top-1 47% -> 34%, the vote 6/10 -> 4/10, and the aligned oracle-side
margin +0.060 -> +0.017.  A distribution that had looked like a policy became
indistinguishable from picking an angle at random (4/10 against 4.2/10).

THE INVARIANT THAT ENFORCES IT
------------------------------
Every non-skipped sweep view gets a row, always.  A view where nothing survived
the correspondence gate is recorded with `top1_group = None`, not dropped, and
`build_views` asserts the count.  So "a row went missing" is now a crash rather
than a quietly better number.

    python probe_viewdist.py --cases datasets/robot/cases_hard.json --n 10
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Sigmas tried for the azimuth smoothing, degrees.  The sweep spans +-30, so a
#: sigma of 20 is barely more than a left/right split and is included to show
#: how much of any result is really just the side.
SIGMAS = (6.0, 10.0, 14.0, 20.0)

#: Candidate headings the distribution is evaluated at.  These are angles
#: `probe_viewpoint.py` walks to, so a heading chosen here has a measured
#: outcome.
#:
#: 0 IS IN THE SET, and it earns its place only in a LOOP.  For a single step it
#: is the baseline being beaten, not a choice -- which is why it was excluded
#: until now.  But a loop needs to be able to elect to STAY, and `argmax == 0` is
#: the only stopping signal available that reads no ground truth.  The comparison
#: costs nothing: the sweep's az = 0 view reproduces the reference frame exactly,
#: so "the value of staying" is measured on the same scale as "the value of
#: going".  Read single-step tables with this in mind -- a policy that may answer
#: "do not move" is not being scored against the same denominator as one that
#: must move.
#:
#: CAPPED AT THE SWEEP'S OWN RANGE (--max-az, 30 degrees).  The kernel is a
#: normalised average, so outside the views' support it just copies the nearest
#: extreme outward -- +45 carried no evidence of its own, and because the curve
#: is usually monotone the argmax then slid to the boundary in 25 of 40 cases.
#: Dropping +-45 costs one case (36 -> 35 of 40) and leaves side agreement
#: untouched at 39/39, so nothing was gained by extrapolating.  The LADDER in
#: `probe_viewpoint.py` still walks to +-45: it measures the ceiling, and what a
#: policy may CHOOSE is a different question from what is worth measuring.
ANGLES = (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0)

#: Which score ranks the pairs within a view.  `s` is the reference frame's
#: object score and is VIEW-INVARIANT, so it shifts every view by the same
#: factor; it is kept as a control because on the reference frame alone it
#: actively hurts (median log10 gap -0.13 -> -0.68 over 10 hard cases).
SCORES = ("rel", "rel_s")


# --------------------------------------------------------------------- build

def build_groups(built, cand, iou_hit: float) -> Dict[int, int]:
    """slot -> object id, by BOX OVERLAP ALONE.

    A detector emits many mutually-containing boxes per object, so a vote over
    raw slot pairs splits one answer across several: on `tabletop|1` the correct
    triplet arrived as (181,54), (181,83) and (43,83) -- three "different" pairs
    naming one cup and one laptop -- and lost 1-1-1 to a single wrong pair.

    NOT `channels.slot_groups_xyxy`, which also requires the same argmax class:
    two slots on one cup can argmax to `cup` and `bottle`, and that rule keeps
    them apart, which is the case this has to merge.
    """
    from robot.task_find import iou

    boxes = built["boxes"]
    group: Dict[int, int] = {}
    for q in list(cand[0]) + list(cand[1]):
        for seen, g in group.items():
            if iou(boxes[q].tolist(), boxes[seen].tolist()) >= iou_hit:
                group[q] = g
                break
        else:
            group[q] = len(set(group.values()))
    return group


def build_views(built, egtr, task, cand, rendered, groups,
                gate_cos: float, corr: str) -> List[Dict[str, Any]]:
    """Per view: which pair it ranks first, at slot and at object level.

    NO TRUTH ARGUMENT.  Whatever is wrong with this function, it cannot be
    reading the answer.

    A view whose candidates do not correspond still gets a row, with
    `top1_group = None`.  Dropping it would make the row set depend on the
    correspondence gate, and the gate fails hardest on precisely the occluded
    target this experiment is about.
    """
    import torch

    from lib.fusion import channels as ch

    subjects, objects = cand
    predicate = egtr["rel_names"].index(task["predicate"])
    s = built["s"].float()
    nq = built["rec"]["s_ref"].shape[0]
    hn = torch.nn.functional.normalize(built["rec"]["h_ref"].float(), dim=-1)

    wanted = [(i, v) for i, v in enumerate(built["rec"]["views"])
              if int(v["v"]) not in ch.SKIP_VIEWS]
    out: List[Dict[str, Any]] = []
    for index, view in wanted:
        field, ok = ch._pair_field(view, nq, hn, corr, gate_cos)
        scored: Dict[str, List[Tuple[float, int, int]]] = {k: [] for k in SCORES}
        for i in subjects:
            for j in objects:
                if i == j or groups[i] == groups[j]:
                    continue
                if ok is not None and not (bool(ok[i]) and bool(ok[j])):
                    continue
                rel = float(field[i, j, predicate])
                if rel <= 0:
                    continue
                scored["rel"].append((rel, int(i), int(j)))
                scored["rel_s"].append((rel * float(s[i]) * float(s[j]),
                                        int(i), int(j)))
        row: Dict[str, Any] = {
            "v": int(view["v"]),
            "azimuth": float(rendered[index]["pose"]["azimuth"]),
            "elevation": float(rendered[index]["pose"]["elevation"]),
            "pairs": len(scored["rel"]),
        }
        for key in SCORES:
            if not scored[key]:
                row[key] = {"top1_slots": None, "top1_group": None,
                            "score": None}
                continue
            best = max(scored[key])
            row[key] = {"top1_slots": [best[1], best[2]],
                        "top1_group": [groups[best[1]], groups[best[2]]],
                        "score": best[0]}
        out.append(row)

    # THE INVARIANT.  Every non-skipped view has a row; nothing was filtered.
    assert len(out) == len(wanted), (
        f"{len(wanted) - len(out)} views vanished -- selection must not depend "
        f"on anything, least of all the truth")
    return out


def build_vote(views: Sequence[Dict[str, Any]], key: str):
    """The most-voted object-level pair, or None if no view named one."""
    tally = collections.Counter(tuple(v[key]["top1_group"]) for v in views
                               if v[key]["top1_group"] is not None)
    return tally.most_common(1)[0][0] if tally else None


def build_distribution(views: Sequence[Dict[str, Any]], phat, key: str,
                       sigma: float, angles: Sequence[float]) -> np.ndarray:
    """Local agreement rate with `phat` at each of `angles`.

    Gaussian kernel over azimuth, DIVIDED BY THE KERNEL SUM.  The sweep is a
    Gerono lemniscate (az = A sin t, el = B sin 2t), so views bunch up near the
    azimuth extremes; an unnormalised kernel sum would read that density as
    agreement.  Views that named nothing count as disagreement, which is the
    honest reading -- they did not support `phat`.

    Smoothing is on the AZIMUTH AXIS, never on view index: the lemniscate
    visits 0 -> +A -> 0 -> -A -> 0, so index order is not spatial order, and
    each azimuth is seen twice at different elevations.  Pooling those two is a
    feature.
    """
    az = np.array([v["azimuth"] for v in views], float)
    ind = np.array([1.0 if (phat is not None
                            and v[key]["top1_group"] is not None
                            and tuple(v[key]["top1_group"]) == tuple(phat))
                    else 0.0 for v in views])
    grid = np.asarray(angles, float)
    k = np.exp(-0.5 * ((grid[:, None] - az[None, :]) / sigma) ** 2)
    return (k @ ind) / np.maximum(k.sum(1), 1e-9)


# --------------------------------------------------------------------- grade

def grade_views(views: List[Dict[str, Any]], built, truth,
                iou_hit: float) -> None:
    """ADD `hit` to each view's each score.  Never removes a row.

    `hit` asks the only question that matters per view: read through this view,
    is the instructed triplet the top-ranked pair?  Both endpoints, judged in
    REFERENCE box space, because the mapped field lives in reference slot space
    and a synthesised view has no ground truth of its own.
    """
    from robot.task_find import iou

    boxes = built["boxes"]
    for row in views:
        for key in SCORES:
            slots = row[key]["top1_slots"]
            row[key]["hit"] = (
                None if not (truth and truth.get("target")
                             and truth.get("landmark"))
                else bool(slots
                          and iou(boxes[slots[0]].tolist(),
                                  truth["target"]) >= iou_hit
                          and iou(boxes[slots[1]].tolist(),
                                  truth["landmark"]) >= iou_hit))


def grade_vote(views: Sequence[Dict[str, Any]], phat, key: str
               ) -> Optional[bool]:
    """Is the voted pair the right OBJECT pair?

    An object-level group is coarser than the IoU test: two slots in one group
    can differ on whether they cover the target at IoU 0.5.  So a group pair
    counts as correct when the MAJORITY of the views that voted for it had a
    truly correct top-1.  Object-level correct is a weaker bar than the robot
    acting correctly, and is reported as such.
    """
    if phat is None:
        return None
    votes = [v for v in views if v[key]["top1_group"] is not None
             and tuple(v[key]["top1_group"]) == tuple(phat)]
    graded = [v[key]["hit"] for v in votes if v[key]["hit"] is not None]
    return None if not graded else sum(graded) * 2 >= len(graded)


# ---------------------------------------------------------------------- main

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="datasets/robot/cases_hard.json")
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--condition", type=int, default=10)
    ap.add_argument("--cand-nms", type=float, default=0.0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--max-az", type=float, default=30.0)
    ap.add_argument("--max-el", type=float, default=15.0)
    ap.add_argument("--radius-scale", type=float, default=1.5)
    ap.add_argument("--trust", type=int, default=10)
    # `perceive` reads it; this probe uses only the P-hat that comes with it.
    ap.add_argument("--bearing", default="bin")
    ap.add_argument("--save-trail", default=None)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--out", default="nvs_pilot/probe_viewdist.json")
    args = ap.parse_args(argv)

    from eval_move import perceive, truth_boxes
    from fuse_live import CORR, GATE_COS, conditioned, task_for
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
            got = perceive(rc, case, task, egtr, args, want_record=True)
            if got is None or got.get("built") is None:
                print(f"  [{index}] {case['scene']} unreadable", flush=True)
                continue
            built, rendered = got["built"], got["rendered"]
            cand = conditioned(built, egtr, task, args.condition, args.cand_nms)
            if cand is None:
                print(f"  [{index}] {case['scene']} no candidates", flush=True)
                continue
            groups = build_groups(built, cand, args.iou)
            views = build_views(built, egtr, task, cand, rendered, groups,
                                GATE_COS, CORR)
            grade_views(views, built, truth_boxes(rc, task), args.iou)
            row = {"scene": case["scene"], "instruction": case["instruction"],
                   "views": views}
            for key in SCORES:
                phat = build_vote(views, key)
                row[key] = {"phat": list(phat) if phat else None,
                            "phat_correct": grade_vote(views, phat, key),
                            "picks": {f"{sg:.0f}": None for sg in SIGMAS}}
                for sg in SIGMAS:
                    w = build_distribution(views, phat, key, sg, ANGLES)
                    row[key]["picks"][f"{sg:.0f}"] = float(
                        ANGLES[int(np.argmax(w))])
            rows.append(row)
            hits = sum(1 for v in views if v["rel"].get("hit"))
            named = sum(1 for v in views if v["rel"]["top1_group"])
            print(f"  [{index}/{len(cases)}] {case['scene']:13} "
                  f"views {len(views):>2}  named a pair {named:>2}  "
                  f"top-1 correct {hits:>2}  "
                  f"p_hat correct {row['rel']['phat_correct']}",
                  flush=True)
        except Exception as error:                              # noqa: BLE001
            import traceback
            print(f"  [{index}] ! {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
        finally:
            controller.stop()
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)),
                        exist_ok=True)
            json.dump({"sigmas": list(SIGMAS), "angles": list(ANGLES),
                       "cases": rows}, open(args.out, "w"), indent=1)

    if not rows:
        return 1
    n = len(rows)
    allv = [v for r in rows for v in r["views"]]
    print(f"\n  {n} cases, {len(allv)} views ({len(allv) / n:.1f} per case; "
          f"the sweep yields {args.views - 2})\n")
    print(f"  {'score':8} {'per-view top-1':>15} {'views naming a pair':>21} "
          f"{'vote p_hat correct':>20}")
    for key in SCORES:
        hit = sum(1 for v in allv if v[key].get("hit"))
        named = sum(1 for v in allv if v[key]["top1_group"])
        vote = sum(1 for r in rows if r[key]["phat_correct"])
        print(f"  {key:8} {hit:>8}/{len(allv):<6} {named:>13}/{len(allv):<7} "
              f"{vote:>13}/{n:<6}")
    print(f"\n  Where the distribution says to walk (azimuth, degrees):")
    for key in SCORES:
        for sg in SIGMAS:
            picks = [r[key]["picks"][f"{sg:.0f}"] for r in rows]
            print(f"    {key:6} sigma {sg:>4.0f}   "
                  + " ".join(f"{p:+.0f}" for p in picks))
    print(f"\n  Grading those headings needs walked outcomes: "
          f"probe_viewpoint.py, then eval_viewdist.py")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
