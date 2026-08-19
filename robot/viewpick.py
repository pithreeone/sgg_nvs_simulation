"""
viewpick.py -- a swept record becomes a heading and a view.  The shared rule.

THIS IS THE POLICY, and it is the one thing the simulator and a real robot must
run the same code for.  `eval_move.perceive` staged it inline while there was
only one caller; a robot driven from photographs is the second, and a rule that
exists twice is a rule whose two copies will disagree about which one produced a
number in a table.

WHAT IT DOES NOT TOUCH: the robot.  Nothing here reads a controller, a pose or a
simulator -- the inputs are a fused record, the poses that record was swept at,
and the instruction.  Where the robot then GOES is `eval_move.step_to`'s
arithmetic; whether it ARRIVED is `eval_move.look`'s single real frame.

The rules and the numbers behind each of them are documented at their branches
below, carried over verbatim from `perceive`: they are measurements, and
rewriting them as prose would lose the counts they turn on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def orbit_depth(depth: Optional[np.ndarray],
                fraction: float = 0.2) -> Optional[float]:
    """The sweep's radius: the median depth of the image's centre patch.

    A SCALAR OFF THE OPTICAL AXIS, WITH NO DETECTOR IN IT.  The sweep's centre
    must lie on the reference camera's own axis or az = el = 0 stops reproducing
    the reference frame, which is the identity the whole trajectory is built on
    (`nvs_lemniscate.camera_for`).  So this asks only "how far is whatever is
    straight ahead", and nothing that can name the wrong object enters.

    A PATCH, NOT THE CENTRE PIXEL: one pixel lands on a specular highlight or a
    stereo dropout often enough to matter, and the median over a fifth of the
    frame is the same quantity with that removed.

    NAN-TOLERANT, which the simulator never needed.  THOR's depth buffer is
    dense; a real sensor returns nothing for dark, glossy or too-near surfaces,
    and `real_robot.load_depth` marks those NaN rather than 0 so they cannot be
    read as "at the camera".  A plain median over a patch holding one NaN is
    NaN, which would poison the orbit silently.

    None when the patch is entirely invalid -- the radius is then undefined and
    the caller must not invent one.
    """
    if depth is None:
        return None
    height, width = depth.shape[:2]
    low, high = 0.5 - fraction / 2, 0.5 + fraction / 2
    patch = depth[int(height * low):int(height * high),
                  int(width * low):int(width * high)]
    valid = patch[np.isfinite(patch)]
    return float(np.median(valid)) if valid.size else None


def pick_view(bearing: str, side_step: float, built: Dict[str, Any],
              rendered: Sequence[Dict[str, Any]],
              cand: Tuple[Sequence[int], Sequence[int]], egtr,
              task: Dict[str, Any], order: Sequence[Tuple[float, int, int]],
              chosen: Optional[Tuple[int, int]], ballots: Sequence[Any]
              ) -> Tuple[List[float], Optional[Dict[str, Any]],
                         List[Dict[str, Any]]]:
    """`(votes, view, detail)` -- a side, the pose the rule picked, and the
    per-view scores it picked from.

    A SWEEP VIEW IS ALREADY A FULL POSE -- `nvs_lemniscate.camera_for` returns
    position, yaw and pitch -- so a policy that names a view needs nothing
    predicted to execute it: the relative transform is that pose minus this one.
    `votes` is the older, lossier channel, a scalar azimuth that `eval_move.walk`
    then re-derives a motion from about a DIFFERENT centre; see `step_to` for why
    that is worth removing.
    """
    from lib.fusion import channels as ch
    from fuse_live import (CLASS_ALIASES, CORR, GATE_COS, OBJSCORE_COS)

    import torch

    view: Optional[Dict[str, Any]] = None
    votes: List[float] = []
    # THE SCORES ARE THE MECHANISM, so they come back out.  Every rule here is
    # "put a number on each view and take the argmax"; returning only the winner
    # left the one quantity a reader has to see -- and the one a different rule
    # would replace -- trapped inside this function.
    detail: List[Dict[str, Any]] = []
    predicate = egtr["rel_names"].index(task["predicate"])

    def note(indices, values, **terms) -> None:
        for k, index in enumerate(indices):
            pose = rendered[index]["pose"]
            # SIGNIFICANT DIGITS, not decimal places.  These rules span many
            # orders of magnitude -- `attrib`'s terms are ~1e-5 -- and rounding
            # to 4 dp reported every one of them as 0.0.
            def keep(v):
                return float(f"{float(v):.6g}")

            detail.append({"v": int(index),
                           "azimuth": round(float(pose["azimuth"]), 1),
                           "elevation": round(float(pose["elevation"]), 1),
                           "score": keep(values[k]),
                           **{name: keep(t[k]) for name, t in terms.items()}})

    if bearing == "attrib":
        # THE RULE FALLS OUT OF B.  `ev[i,j]` is the mean over views of that
        # view's own `_pair_field(v)[i, j, predicate]`, so B is already a sum of
        # per-view terms.  Once the fusion has settled on a pair, scoring a view
        # by ITS OWN summand for that pair is not a new quantity -- it is the
        # attribution of the decision.  Go where the evidence came from.
        #
        # A view that cannot see both endpoints contributes zero and scores zero,
        # which is the honest reading: the pair is not selectable from there.
        #
        # THE SCORE IS THE SUPPORT ALONE.  `rival` -- the best pair this view backs
        # whose SUBJECT is a different object -- is computed and reported, but it
        # is not subtracted: on the first real episode the two rank the views
        # identically, and the simpler quantity is the one that can be described
        # in a sentence.  It stays in the output because a view with no support
        # and a large rival is not neutral but actively misleading, and that is
        # worth being able to see.
        #
        # `edge` below is this rule already, but binary and tracking `order`'s
        # predicate-free ranking -- which on a real frame is the unoccluded
        # distractor.  The pair comes from the caller here.
        pursued = chosen if chosen is not None else (
            (order[0][1], order[0][2]) if order else None)
        if pursued is None:
            return votes, view, detail
        si, sj = pursued
        subjects, objects = cand
        nq = built["rec"]["s_ref"].shape[0]
        hn = torch.nn.functional.normalize(
            built["rec"]["h_ref"].float(), dim=-1)
        boxes = built["boxes"]

        from robot.task_find import iou

        support, rival, taken = [], [], []
        for index, sweep_view in enumerate(built["rec"]["views"]):
            if int(sweep_view["v"]) in ch.SKIP_VIEWS:
                continue
            field, ok = ch._pair_field(sweep_view, nq, hn, CORR, GATE_COS)
            seen = ok is None or (bool(ok[si]) and bool(ok[sj]))
            mine = float(field[si, sj, predicate]) if seen else 0.0
            # The best pair this view backs whose SUBJECT is a different object.
            other = 0.0
            for a in subjects:
                if iou(boxes[a].tolist(), boxes[si].tolist()) >= 0.5:
                    continue
                for b in objects:
                    if a == b or (ok is not None
                                  and not (bool(ok[a]) and bool(ok[b]))):
                        continue
                    other = max(other, float(field[a, b, predicate]))
            support.append(mine)
            rival.append(other)
            taken.append(index)

        if taken:
            mine = np.array(support, float)
            other = np.array(rival, float)
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
        # THIS SCORE CHOOSES A DIRECTION AND NOTHING ELSE.  Putting the two
        # az = el = 0 views -- which re-render the pose the robot is at -- on the
        # ballot was tried, so that "stay" could win the same argmax: it never
        # did, one sample against the maximum of eighteen being biased towards
        # moving by construction.  Comparing it to the MEDIAN instead does fire,
        # and stops on a bad pose as readily as a good one, because this quantity
        # separates a correct pose from a wrong one by the 83rd percentile
        # against the 72nd.  Stopping is `--stop-score`, on the real frame, where
        # the separation is 1.05 decades.
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
        # THE HIGHEST-SCORING VIEW, and this is the whole point of the score.
        # Comparing the two halves throws away the ordering WITHIN a side, which
        # is the only axis this rule beats the count on -- AUC 0.88 against 0.76
        # -- so a side vote cannot show what it is for.  Reported unconditionally:
        # unlike the halves it needs no view on both sides to exist.
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
        # THE APPEARANCE HALF COMPARES A CANDIDATE TO ITSELF.  Taking the best
        # p(class) over candidates is pinned by the unoccluded twin, which looks
        # the same from everywhere; dividing each candidate by its OWN spread
        # across the sweep sends every flat candidate -- the twin, and the walls
        # -- to zero without anyone saying which is which.  What survives is a
        # candidate that looks more like the instructed class from here than it
        # usually does.
        subjects = cand[0]
        classes = {v: k - 1 for k, v in egtr["obj_names"].items()}
        columns = [classes[c]
                   for c in CLASS_ALIASES.get(task["subject_class"],
                                              (task["subject_class"],))
                   if c in classes]
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

            # NODE AND EDGE, WHICH IS WHAT A AND R ACTUALLY ARE.  A view helps a
            # grounding in one of two ways and the two lists are built to
            # separate them: `cases_slot`'s target cannot be NAMED from the start
            # pose (a node problem), `cases_hard`'s target is named perfectly and
            # is indistinguishable from a same-class twin, so only the RELATION
            # picks it out (an edge problem).  C is neither -- without a
            # correspondence there is no cross-view quantity at all -- so it
            # enters as the gate that zeroes a candidate the view cannot see,
            # not as a third term to be weighed against the other two.
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

    return votes, view, detail
