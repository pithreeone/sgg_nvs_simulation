"""
viewpick.py -- a swept record becomes a heading and a view.  The shared rule.

THIS IS THE POLICY, and the simulator and a real robot must run the same copy of
it or a number in a table cannot be attributed to a rule.

WHAT IT DOES NOT TOUCH: the robot.  In come a fused record, the poses it was
swept at, and the instruction -- no controller, no pose, no sensor.  Where the
robot then GOES is `eval_move.step_to`; whether it ARRIVED is `eval_move.look`.

Each branch below carries the counts it was chosen on.  Those are measurements;
they stay.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Bins for the azimuth vote, degrees.  Votes are often bimodal -- the target is
#: recoverable from either side -- and the mean of "25 left" and "25 right" is
#: "do not move".
BIN = 10.0


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


def nearest_view(rendered, azimuth: Optional[float]):
    """The rendered pose closest to an azimuth a rule voted for.

    A RULE THAT ONLY NAMES AN ANGLE CANNOT BE EXECUTED.  The robot walks to a
    POSE -- a sweep view's own relative transform -- so the two scalar rules
    (`bin`, `visible`) report the view their vote lands on rather than leaving
    the caller with a number and nothing to reach.
    """
    if azimuth is None or not len(rendered):
        return None
    return min(rendered,
               key=lambda r: abs(float(r["pose"]["azimuth"]) - azimuth))["pose"]



def pick_view(bearing: str, side_step: float, built: Dict[str, Any],
              rendered: Sequence[Dict[str, Any]],
              cand: Tuple[Sequence[int], Sequence[int]], egtr,
              task: Dict[str, Any], order: Sequence[Tuple[float, int, int]],
              chosen: Optional[Tuple[int, int]], ballots: Sequence[Any],
              reference: Optional[Any] = None, vlm: Optional[Any] = None
              ) -> Tuple[List[float], Optional[Dict[str, Any]],
                         List[Dict[str, Any]]]:
    """`(votes, view, detail)` -- a side, the pose the rule picked, and the
    per-view scores it picked from.

    A SWEEP VIEW IS ALREADY A FULL POSE (`nvs_lemniscate.camera_for` returns
    position, yaw and pitch), so naming a view needs nothing predicted to
    execute: the relative transform is that pose minus this one.  `votes` is the
    older, lossier channel -- a scalar azimuth that `eval_move.walk` re-derives a
    motion from about a DIFFERENT centre.  See `step_to`.
    """
    from lib.fusion import channels as ch
    from fuse_live import CORR, GATE_COS, OBJSCORE_COS
    from robot.policy.grounding import class_columns

    import torch

    view: Optional[Dict[str, Any]] = None
    votes: List[float] = []
    # THE SCORES ARE THE MECHANISM, so they come back out.  Every rule here is
    # "put a number on each view and take the argmax".
    detail: List[Dict[str, Any]] = []
    predicate = egtr["rel_names"].index(task["predicate"])

    def note(indices, values, **terms) -> None:
        for k, index in enumerate(indices):
            pose = rendered[index]["pose"]
            # Significant digits, not decimal places: `attrib`'s terms are
            # ~1e-5 and 4 dp reported every one of them as 0.0.
            def keep(v):
                return float(f"{float(v):.6g}")

            detail.append({"v": int(index),
                           "azimuth": round(float(pose["azimuth"]), 1),
                           "elevation": round(float(pose["elevation"]), 1),
                           "score": keep(values[k]),
                           **{name: keep(t[k]) for name, t in terms.items()}})

    if bearing == "vlm":
        # THE BASELINE ARM.  It reads the robot's own frame and nothing else --
        # no sweep, no fusion, no detector -- so it is the number a policy that
        # looks at the picture has to beat.  See `robot/vlm.py` for what the
        # model is and is not asked.
        #
        # ONE BIT BUYS ONE AXIS.  A side is not a pose, so the vote goes through
        # `nearest_view` exactly as `bin` and `visible` do, and the elevation is
        # whatever that view happens to carry.  This arm therefore chooses from
        # the same 20 poses as the others while using less of them.
        if vlm is None or reference is None:
            return votes, view, detail
        said = vlm.direction(reference, task["subject_class"],
                             task["object_class"])
        detail.append({"said": said, **{k: v for k, v in vlm.last.items()
                                        if k != "reply"}})
        if said is None:
            return votes, view, detail
        # POSITIVE AZIMUTH IS THE ROBOT'S LEFT; see `nvs_lemniscate.camera_for`.
        votes = [side_step if said == "LEFT" else -side_step]
        view = nearest_view(rendered, votes[0])

    elif bearing == "attrib":
        # THE RULE FALLS OUT OF B.  `ev[i,j]` is the mean over views of that
        # view's own `_pair_field(v)[i, j, predicate]`, so once the fusion has
        # settled on a pair, scoring a view by ITS OWN summand for that pair is
        # the attribution of the decision.  Go where the evidence came from.
        # A view that cannot see both endpoints scores zero: not selectable there.
        #
        # THE SCORE IS THE SUPPORT ALONE.  `rival` -- the best pair this view
        # backs whose SUBJECT is a different object -- is reported but not
        # subtracted; on the first real episode the two rank the views
        # identically.  It stays visible because a view with no support and a
        # large rival is actively misleading, not neutral.
        pursued = chosen if chosen is not None else (
            (order[0][1], order[0][2]) if order else None)
        if pursued is None:
            return votes, view, detail
        si, sj = pursued
        subjects, objects = cand
        boxes = built["boxes"]

        from robot.task.task_find import iou
        from robot.policy.evidence import pair_contributions

        # THE PURSUED PAIR FIRST, then every pair whose SUBJECT is a different
        # object -- one table, read as support and as rival.
        rivals = [(a, b) for a in subjects
                  if iou(boxes[a].tolist(), boxes[si].tolist()) < 0.5
                  for b in objects if a != b]
        taken, contrib, _ = pair_contributions(built, [pursued] + rivals,
                                               predicate)

        if taken:
            mine = contrib[:, 0]
            other = (contrib[:, 1:].max(1) if rivals
                     else np.zeros(len(taken)))
            score = mine
            note(taken, score, support=mine, rival=other)
            azimuth = np.array([float(rendered[i]["pose"]["azimuth"])
                                for i in taken], float)
            # POSITIVE AZIMUTH IS THE ROBOT'S LEFT; see `nvs_lemniscate.camera_for`.
            left_half, right_half = score[azimuth > 0], score[azimuth < 0]
            if len(left_half) and len(right_half):
                votes = [side_step
                         if left_half.mean() >= right_half.mean()
                         else -side_step]
            view = rendered[taken[int(np.argmax(score))]]["pose"]

    elif bearing == "side":
        # HOW MANY OF THE INSTRUCTION'S OWN CANDIDATES THIS VIEW STILL SEES.
        #
        # No predicate, no identity, no P-hat: a candidate that does not
        # correspond in a view can never be selected FROM that view, so this is
        # the necessary condition, and answering it needs no idea which triplet
        # is the right one.  That is why it beats the rules that do -- agreement
        # with a voted pair was right in 4 of 10, and the class-probability gain
        # is carried by the unoccluded twin, which every viewpoint sees.
        #
        # ONLY THE SUBJECT SIDE.  The landmark ranks 1st in 40 of 40, so its
        # count is near constant across azimuth: multiplying it in scored 34 of
        # 40 against 35.  `min` of the two sides scored 37 and is the next thing
        # to try -- 2 cases is not yet evidence.
        #
        # SLOT level, not deduplicated to objects: dedup scored 31 of 40 against
        # 36.  How many boxes survive is a graded measure of how clearly an
        # object is seen, and dedup throws that gradation away.
        #
        # A SIDE, THEN A SMALL STEP -- there is no angle in this rule.  The
        # smoothed curve in `viz/plot_viewdist.py` scores 35 of 40 against 36 for
        # comparing the two halves, and its argmax sat on the +-30 boundary in 35
        # of 40 cases: it was only ever answering "which side".
        #
        # The step is small and the side is re-measured every step, so an
        # overshoot flips the next reading and the robot comes back.  At 30
        # degrees it could not -- three compound to 90 and top-1 went 32 -> 9 -> 0.
        hn = torch.nn.functional.normalize(
            built["rec"]["h_ref"].float(), dim=-1)
        # POSITIVE AZIMUTH IS THE ROBOT'S LEFT; see `nvs_lemniscate.camera_for`.
        left: List[float] = []
        right: List[float] = []
        counted: List[Tuple[float, int]] = []
        for index, sweep_view in enumerate(built["rec"]["views"]):
            if int(sweep_view["v"]) in ch.SKIP_VIEWS:
                continue
            _, ok = ch.correspond(hn, sweep_view["h"], CORR, GATE_COS)
            # A view that corresponds to nothing counts as 0, never skipped:
            # that is the reading on the side the target is hidden on, and
            # dropping it is the one line that pre-sorted this measurement
            # toward the answer once already.  So both halves always have a
            # value, and the guided arm never falls back to the control.
            count = (0.0 if ok is None
                     else float(sum(1 for q in cand[0] if bool(ok[q]))))
            azimuth = float(rendered[index]["pose"]["azimuth"])
            (left if azimuth > 0 else right).append(count)
            counted.append((count, index))
        note([i for _, i in counted], [c for c, _ in counted])
        if left and right:
            votes = [side_step
                     if sum(left) / len(left) >= sum(right) / len(right)
                     else -side_step]
            # THE BEST VIEW ON THE SIDE IT CHOSE.  The side is what was measured
            # -- the two halves against each other -- but a side is not a place,
            # and `--side-step` degrees is an angle no view was rendered at.  A
            # policy that walks to a POSE has to name one, so the rule reports
            # its own strongest view on the side it just voted for.
            want = votes[0] > 0
            on_side = [(c, i) for c, i in counted
                       if (float(rendered[i]["pose"]["azimuth"]) > 0) == want]
            if on_side:
                view = rendered[max(on_side, key=lambda p: p[0])[1]]["pose"]

    elif bearing == "volatility":
        # WEIGHT EACH CANDIDATE TRIPLET BY HOW MUCH ITS RANK MOVES ACROSS THE
        # SWEEP, then score a view by how highly it ranks the unstable ones.
        #
        # A triplet visible from every angle holds the same rank everywhere, so
        # it cannot tell two viewpoints apart -- and it is what the top-1 vote
        # keeps electing.  A triplet that was occluded and then appears is the
        # one whose rank swings, and on `cases_hard` the instructed triplet sits
        # at the 99th percentile of that swing.  Ranks only, so the ~17 orders of
        # magnitude that `rel` drifts across views cancel; no predicate label, no
        # class gate, no P-hat.
        #
        # Against `side`'s count: same side accuracy (37/37 against 36/37, both
        # saturated) but it ranks VIEWS far better -- AUC 0.88 against 0.76, and
        # its best sweep view is the right one in 32 of 38 against 26.  Dropping
        # the weight scores 0.77, so the weight is what works.
        s = built["s"].float()
        nq = built["rec"]["s_ref"].shape[0]
        hn = torch.nn.functional.normalize(
            built["rec"]["h_ref"].float(), dim=-1)
        subjects, objects = cand
        # A pair this view cannot score gets the worst rank, never dropped: not
        # being findable from here is the measurement, not missing data.
        miss = float(len(subjects) * len(objects) + 1)
        # THIS SCORE CHOOSES A DIRECTION AND NOTHING ELSE.  It cannot also say
        # "stay": one sample against the max of eighteen is biased toward moving,
        # and against the median it fires on bad poses as readily as good ones
        # (83rd percentile against 72nd).  Stopping is `--stop-score`, on the
        # real frame, where the separation is 1.05 decades.
        ranks, azimuths, taken = [], [], []
        for index, sweep_view in enumerate(built["rec"]["views"]):
            if int(sweep_view["v"]) in ch.SKIP_VIEWS:
                continue
            field, ok = ch._pair_field(sweep_view, nq, hn, CORR, GATE_COS)
            scored = []
            for i in subjects:
                for j in objects:
                    if i == j:
                        continue
                    if ok is not None and not (bool(ok[i]) and bool(ok[j])):
                        continue
                    value = (float(field[i, j, predicate])
                             * float(s[i]) * float(s[j]))
                    if value > 0:
                        scored.append((value, int(i), int(j)))
            scored.sort(reverse=True)
            place = {(i, j): rank for rank, (_, i, j) in enumerate(scored, 1)}
            ranks.append([place.get((int(i), int(j)), miss)
                          for i in subjects for j in objects])
            azimuths.append(float(rendered[index]["pose"]["azimuth"]))
            taken.append(index)
        rank = np.array(ranks, float)
        weight = rank.std(0)
        weight = weight / max(weight.max(), 1e-9)
        score = (weight[None, :] / rank).sum(1)
        azimuth = np.array(azimuths, float)
        note(taken, score)
        # POSITIVE AZIMUTH IS THE ROBOT'S LEFT; see `nvs_lemniscate.camera_for`.
        left_half, right_half = score[azimuth > 0], score[azimuth < 0]
        if len(left_half) and len(right_half):
            votes = [side_step
                     if left_half.mean() >= right_half.mean()
                     else -side_step]
        # THE HIGHEST-SCORING VIEW, which is the point of the score: the halves
        # throw away the ordering WITHIN a side, the only axis this beats the
        # count on (AUC 0.88 against 0.76).  Unconditional -- unlike the halves
        # it needs no view on both sides.
        if len(score):
            view = rendered[taken[int(np.argmax(score))]]["pose"]

    elif bearing in ("reveal", "acr", "node", "edge"):
        # A VIEWPOINT HELPS IN ONE OF TWO WAYS, AND THE TWO LISTS NEED DIFFERENT
        # ONES.  On `cases_hard` the target and its distractor are the same class
        # by construction, so no appearance score can separate them and what
        # changes with the viewpoint is whether the target CORRESPONDS at all.
        # On `cases_slot` there is no distractor and nothing to disambiguate;
        # what changes is whether the object can be NAMED.  Correlated against
        # each list's own yardstick over 6 cases, correspondence scores -0.60 and
        # -0.33, appearance -0.02 and -0.40: each is blind on the list it was not
        # built for.  Standardised within the case and added, -0.50 and -0.39 --
        # the only quantity measured that speaks on both, at almost no cost where
        # a single one already worked.
        #
        # THE APPEARANCE HALF COMPARES A CANDIDATE TO ITSELF.  Best p(class)
        # over candidates is pinned by the unoccluded twin, which looks the same
        # from everywhere; dividing each candidate by its OWN spread across the
        # sweep sends every flat candidate to zero without anyone saying which is
        # which.  What survives looks more like the instructed class from HERE
        # than it usually does.
        subjects = cand[0]
        columns = class_columns(egtr, task["subject_class"])
        hn = torch.nn.functional.normalize(
            built["rec"]["h_ref"].float(), dim=-1)
        counts, appearance, azimuths, taken = [], [], [], []
        for index, sweep_view in enumerate(built["rec"]["views"]):
            if int(sweep_view["v"]) in ch.SKIP_VIEWS:
                continue
            match, ok = ch.correspond(hn, sweep_view["h"], CORR, GATE_COS)
            probs = sweep_view["probs"].float()
            counts.append(float(sum(1 for q in subjects
                                    if ok is None or bool(ok[q]))))
            # A candidate that does not correspond reads 0: not findable here.
            appearance.append([float(probs[int(match[q]), columns].sum())
                               if (ok is None or bool(ok[q])) else 0.0
                               for q in subjects])
            azimuths.append(float(rendered[index]["pose"]["azimuth"]))
            taken.append(index)
        if counts:
            def unit(values: np.ndarray) -> np.ndarray:
                return (values - values.mean()) / max(float(values.std()), 1e-9)

            appear = np.array(appearance, float)
            revealed = ((appear - appear.mean(0)[None, :])
                        / np.maximum(appear.std(0)[None, :], 1e-6)).max(1)

            # NODE AND EDGE, WHICH IS WHAT A AND R ACTUALLY ARE.  `cases_slot`'s
            # target cannot be NAMED from the start pose (node); `cases_hard`'s
            # is named perfectly but indistinguishable from a same-class twin, so
            # only the RELATION picks it out (edge).  C is neither -- it enters
            # as the gate that zeroes a candidate the view cannot see.
            #
            #   node   A's class-probability gain, each candidate against its
            #          OWN spread across the sweep, so a candidate that looks
            #          the same from everywhere contributes nothing.
            #   edge   R's ballot for the pair being pursued: did this view
            #          speak about it, and did it name the instructed predicate.
            #          BINARY on purpose -- `rel` drifts ~17 orders of magnitude
            #          across views for numerical reasons, so its magnitude is
            #          not comparable between them while its ARGMAX is.
            pursued = chosen if chosen is not None else (
                (order[0][1], order[0][2]) if order else None)
            edge = np.zeros(len(taken), float)
            if pursued is not None:
                si, sj = pursued
                spoke = {int(b["v"]): (bool(b["spoke"][si, sj])
                                       and int(b["named"][si, sj]) == predicate)
                         for b in ballots}
                edge = np.array([1.0 if spoke.get(
                    int(built["rec"]["views"][index]["v"]), False) else 0.0
                    for index in taken], float)
            node = unit(revealed)
            if bearing == "node":
                score = node
            elif bearing == "edge":
                score = unit(edge)
            elif bearing == "acr":
                score = node + unit(edge)
            else:
                score = unit(np.array(counts, float)) + unit(revealed)
            azimuth = np.array(azimuths, float)
            note(taken, score, correspond=np.array(counts, float),
                 appearance=revealed, relation=edge)
            # POSITIVE AZIMUTH IS THE ROBOT'S LEFT; see `nvs_lemniscate.camera_for`.
            left_half, right_half = score[azimuth > 0], score[azimuth < 0]
            if len(left_half) and len(right_half):
                votes = [side_step
                         if left_half.mean() >= right_half.mean()
                         else -side_step]
            view = rendered[taken[int(np.argmax(score))]]["pose"]

    elif bearing == "visible":
        # WHICH VIEW SEES THE TARGET, not which view says `behind`.  Needs no
        # predicate and only the SUBJECT endpoint, where `bin`'s lexical gate
        # left 39% of guided steps with no NVS input.  It also avoids `rel`,
        # whose cross-view variation is numerical rather than photometric (~17
        # orders for the same pair); a softmax difference between views is a
        # difference in what was VISIBLE.
        #
        # THE GAIN, NOT THE MAXIMUM: raw p(cup) picks whichever view sees the
        # DISTRACTOR best, it being unoccluded from every angle.
        index = {v: k - 1 for k, v in egtr["obj_names"].items()}.get(
            task["subject_class"])
        if index is not None:
            hn = torch.nn.functional.normalize(
                built["rec"]["h_ref"].float(), dim=-1)
            ref_p = built["probs_ref"][:, index]
            az_list = [r["pose"]["azimuth"] for r in rendered]
            best, top = None, 0.0
            for sweep_view in built["rec"]["views"]:
                if int(sweep_view["v"]) in ch.SKIP_VIEWS:
                    continue
                match, ok = ch.correspond(hn, sweep_view["h"], "mutual",
                                          OBJSCORE_COS)
                probs = sweep_view["probs"].float()
                for q in cand[0]:
                    if not bool(ok[q]):
                        continue
                    gain = float(probs[int(match[q]), index]) - float(ref_p[q])
                    if gain > top:
                        best, top = int(sweep_view["v"]), gain
            if best is not None:
                votes = [az_list[best]]
                view = rendered[best]["pose"]

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
            for sweep_view in built["rec"]["views"]:
                if int(sweep_view["v"]) in ch.SKIP_VIEWS:
                    continue
                match, ok = ch.correspond(hn, sweep_view["h"], "mutual",
                                          OBJSCORE_COS)
                if not bool(ok[si]):
                    continue
                value = float(sweep_view["probs"].float()[int(match[si])].max())
                if value > top:
                    best, top = int(sweep_view["v"]), value
            if best is not None:
                votes = [az_list[best]]
        view = nearest_view(rendered, bearing_from(votes))

    return votes, view, detail
