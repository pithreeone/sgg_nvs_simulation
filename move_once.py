"""
move_once.py -- one look at one real frame, before anything is asked to move.

The simulator's numbers say what this pipeline does when the sweep is RENDERED
by the thing it is trying to predict.  On a real robot the sweep is SEVA's, from
one photograph, and nothing in `nvs_pilot/` says what that looks like.  So this
reports the three quantities the rest of the chain is built on, and stops:

    1. THE DEPTH, which is the only place metric scale enters.  SEVA never sees
       it -- `nvs_seva` hands the model angles and its own `lookat_dist` -- so a
       depth error leaves the chosen VIEW untouched and scales the WALK.
    2. WHAT EGTR SEES in a real room.  Every candidate list downstream is
       conditioned on these classes; if the instruction's nouns are not among
       them, no viewpoint rule can recover.
    3. THE SWEEP ITSELF, as a contact sheet.  The largest unknown here: the
       whole method assumes a synthesised view can be read by a detector, and
       that has only ever been checked against THOR renders.

Nothing here walks and nothing here needs an instruction.

    python move_once.py --dir nvs_pilot/real --camera-height 0.15
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: Tile width in the contact sheets, pixels.  Each tile keeps the frame's own
#: aspect ratio and gets its caption in a band stacked above it.
TILE = 400

from robot.real_robot import FOV_V, RealRobot, aligned, load_depth, newest_pair


def depth_report(depth: np.ndarray, fraction: float = 0.2) -> float:
    """Print the depth's health and return the sweep radius it implies.

    THE CENTRE PATCH IS THE WHOLE MEASUREMENT.  `perceive` reads exactly this
    and nothing else to place the orbit, so its validity -- not the frame's
    average -- is what decides whether the step distance means anything.
    """
    valid = np.isfinite(depth)
    height, width = depth.shape[:2]
    low, high = 0.5 - fraction / 2, 0.5 + fraction / 2
    patch = depth[int(height * low):int(height * high),
                  int(width * low):int(width * high)]
    centre = patch[np.isfinite(patch)]
    print(f"  depth      {width}x{height}  "
          f"{100 * valid.mean():.1f}% valid")
    if valid.any():
        good = depth[valid]
        print(f"             p5 {np.percentile(good, 5):.2f}  "
              f"median {np.median(good):.2f}  "
              f"p95 {np.percentile(good, 95):.2f} m")
    if centre.size == 0:
        raise SystemExit("  ! the centre patch has no valid depth -- the sweep "
                         "radius is undefined and nothing below is meaningful")
    radius = float(np.median(centre))
    print(f"  centre     {100 * centre.size / patch.size:.0f}% valid, "
          f"median {radius:.3f} m   <- the sweep's radius")
    if radius < 0.3 or radius > 6.0:
        print("  ! that is outside the range these viewpoint rules were "
              "measured over (0.5-3 m); the walk will scale with it")
    return radius


def egtr_report(egtr, frame: np.ndarray, top: int, wanted: Sequence[str],
                at: Optional[Sequence[float]], out: Optional[str]) -> None:
    """What EGTR makes of a real photograph, its own ranking.

    THE PICTURE DRAWS WHAT THE LIST PRINTS, one box per named class.  An earlier
    version printed the best class per query over the top 60 and drew the top 10
    QUERIES, which are two different sets: `box` was reported at 0.21 and then
    not drawn, because eight doors and chairs outrank it query-wise.  The
    question here is "does this class have a candidate at all", so the class is
    the unit and its best query is the box.
    """
    from robot.sgg_live import predict, raw_predict

    raw = raw_predict(egtr, frame)
    scores, classes = raw["probs_softmax"].max(-1)
    order = scores.argsort(descending=True)

    # Best query per class, in score order.
    best: Dict[str, int] = {}
    for q in order.tolist():
        best.setdefault(egtr["obj_names"].get(int(classes[q]) + 1, "?"), int(q))

    print("\n  objects EGTR is most sure of:")
    shown = list(best.items())[:top]
    for name, q in shown:
        print(f"    {float(scores[q]):.3f}  {name}")

    # THE INSTRUCTION'S OWN NOUNS, ranked or not.  RANKED BY p(class) DOWN THE
    # COLUMN, never by whether the class won some query's argmax: `conditioned`
    # builds its candidate list exactly that way, so a class that is nobody's
    # argmax still has candidates and is still groundable.  An earlier version
    # looked the name up among the argmax winners and reported `bag` as absent
    # from VG150 while EGTR was putting 0.07 on it.
    columns = {v: k - 1 for k, v in egtr["obj_names"].items()}
    if wanted:
        print("\n  the classes you asked about, best query by p(class):")
        for name in wanted:
            if name not in columns:
                print(f"    --     {name}  NOT IN VG150")
                continue
            column = raw["probs_softmax"][:, columns[name]]
            q = int(column.argmax())
            shown.append((name, q))
            print(f"    {float(column[q]):.3f}  {name}  "
                  f"(argmax of that query: "
                  f"{egtr['obj_names'].get(int(classes[q]) + 1, '?')} "
                  f"{float(scores[q]):.3f})")

    if at is not None:
        shown += _at_report(raw, egtr, at)

    print("\n  triplets, EGTR's own top-5:")
    for t in predict(egtr, frame, topk=5):
        print(f"    {t['score']:.4f}  {t['subject']} "
              f"{t['predicate']} {t['object']}")

    if out:
        _draw_boxes(frame, raw, egtr, shown, len(shown) - len(wanted)
                    - (0 if at is None else AT_QUERIES),
                    os.path.join(out, "egtr.png"))


#: How many of the queries overlapping `--at` to open up.  EGTR emits several
#: boxes per object, so one is never the whole story.
AT_QUERIES = 4


def _at_report(raw, egtr, at: Sequence[float]) -> List:
    """WHAT DOES EGTR CALL THE THING HERE -- the reverse of a class query.

    `--classes` asks whether a name has a box; this asks what names a PLACE has,
    which is the question when an object is missing from every ranking and it is
    not clear whether the detector missed it or merely called it something else.
    Ranked by overlap with the box you drew, and the full head of each query's
    distribution is printed rather than its argmax: a query split evenly between
    `bag` and `pillow` reads as neither at the top-1, and that is a different
    failure from not seeing the object.
    """
    from robot.task_find import iou

    boxes, probs = raw["boxes"], raw["probs_softmax"]
    overlap = sorted(((iou(boxes[q].tolist(), list(at)), int(q))
                      for q in range(boxes.shape[0])), reverse=True)
    print(f"\n  what EGTR calls the region {[int(v) for v in at]}:")
    picked = []
    for share, q in overlap[:AT_QUERIES]:
        if share <= 0.0:
            break
        head = probs[q].topk(5)
        names = "  ".join(
            f"{egtr['obj_names'].get(int(c) + 1, '?')} {float(p):.3f}"
            for p, c in zip(head.values, head.indices))
        print(f"    IoU {share:.2f}   {names}")
        picked.append((f"IoU {share:.2f}", q))
    if not picked:
        print("    -- no query box overlaps it at all, so EGTR has no "
              "candidate here under any name")
    return picked


def _draw_boxes(frame, raw, egtr, shown, split: int, path: str) -> None:
    """MAGENTA what EGTR ranked, GREEN what you asked about."""
    import cv2

    canvas = np.ascontiguousarray(frame[:, :, ::-1]).copy()
    scores, _ = raw["probs_softmax"].max(-1)
    for index, (name, q) in enumerate(shown):
        asked = index >= split
        colour = (140, 255, 140) if asked else (230, 120, 230)
        x0, y0, x1, y1 = (int(v) for v in raw["boxes"][q].tolist())
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 3 if asked else 2)
        cv2.putText(canvas, f"{name} {float(scores[q]):.2f}", (x0 + 2, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2 if asked else 1,
                    cv2.LINE_AA)
    cv2.imwrite(path, canvas)
    print(f"    -> {path}")


def contact_sheet(reference: np.ndarray, frames: Sequence[np.ndarray],
                  poses: Sequence[Dict[str, Any]], path: str,
                  columns: int = 5) -> None:
    """The sweep as one image, each view captioned with the pose that made it.

    THE REFERENCE GOES IN FIRST, captioned as such.  A sheet of twenty
    synthesised views cannot be judged without the photograph they came from
    next to them -- the question is what the model INVENTED, which is a
    difference, not a picture.
    """
    import cv2

    tiles = []
    for frame, pose in [(reference, None), *zip(frames, poses)]:
        tile = cv2.resize(np.ascontiguousarray(np.asarray(frame)[:, :, ::-1]),
                          (TILE, int(TILE * frame.shape[0] / frame.shape[1])))
        band = np.full((20, TILE, 3), 20, dtype=tile.dtype)
        label = ("REFERENCE (the robot's own frame)" if pose is None else
                 f"az {float(pose['azimuth']):+.0f}  "
                 f"el {float(pose['elevation']):+.0f}")
        cv2.putText(band, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (140, 255, 140) if pose is None else (215, 215, 215), 1,
                    cv2.LINE_AA)
        tiles.append(np.vstack([band, tile]))
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start:start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(path, np.vstack(rows))
    print(f"    -> {path}")


def predicted_sheet(built, rendered, task, egtr, predicate, args,
                    path: str, reference=None, columns: int = 5):
    """WHAT EACH FRAME WOULD ANSWER ON ITS OWN.  MAGENTA subject, ORANGE object.

    -> (per-view report, the reference's own top-1).

    THE REFERENCE IS SCORED BY THE SAME CODE AS THE VIEWS, which it was not:
    it was handed `decide_step`'s pair, weighted by `ch.object_scores` -- a
    quantity pooled across all 20 views through the correspondence.  So the
    start pose was competing with NVS evidence already folded in while each view
    stood alone, and the panel that was supposed to be the "before" was the best
    informed tile on the sheet.

    Every frame here is treated as a photograph somebody standing there took:
    its OWN candidates by p(class), its OWN `rel[i, j, predicate]`, its OWN
    boxes, no fusion and no correspondence.  That machinery exists so the FUSION
    can pool evidence about one candidate across views, and it is measured
    elsewhere -- mixing it into this figure is what made the comparison unread-
    able.  Nothing is comparable slot-for-slot between frames and nothing needs
    to be: each answers "what would a robot standing here ground this to".
    """
    import cv2
    import torch

    from lib.fusion import channels as ch
    from fuse_live import CLASS_ALIASES
    from robot.task_find import iou

    column_of = {v: k - 1 for k, v in egtr["obj_names"].items()}

    def mass(probs, name):
        cols = [column_of[c] for c in CLASS_ALIASES.get(name, (name,))
                if c in column_of]
        return probs[:, cols].sum(-1) if cols else None

    def best_pair(probs, boxes, rel):
        """The frame's own top-1 -> (score, i, j, p_subject, p_object) or None.

        `eval_move.look`'s shape on a single frame: candidates by p(class),
        ordered by `rel[i, j, predicate]`, pairs that are one object twice
        dropped.
        """
        p_sub = mass(probs, task["subject_class"])
        p_obj = mass(probs, task["object_class"])
        if p_sub is None or p_obj is None:
            return None
        subs = torch.argsort(p_sub, descending=True)[:args.condition].tolist()
        objs = torch.argsort(p_obj, descending=True)[:args.condition].tolist()
        best = max(((float(rel[i, j, predicate]) * float(p_sub[i])
                     * float(p_obj[j]), int(i), int(j))
                    for i in subs for j in objs
                    if i != j and (not args.pair_iou
                                   or iou(boxes[i].tolist(), boxes[j].tolist())
                                   < args.pair_iou)), default=None)
        if best is None:
            return None
        return (*best, float(p_sub[best[1]]), float(p_obj[best[2]]))

    def panel(canvas, boxes, pair, line1, line2, dim=False):
        """One tile: the pair drawn, two caption lines STACKED ABOVE the image.

        Above, not over: lettering the caption onto the frame covered its top
        18%, which reads as a vertically squashed picture rather than as a
        cropped one.  The image keeps its own 16:9.
        """
        if pair is not None:
            for q, colour, thick in ((pair[0], (230, 120, 230), 3),
                                     (pair[1], (90, 200, 255), 2)):
                x0, y0, x1, y1 = (int(t) for t in boxes[q].tolist())
                cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thick)
        tile = cv2.resize(canvas, (TILE, int(TILE * canvas.shape[0]
                                             / canvas.shape[1])))
        band = np.full((34, TILE, 3), 20, dtype=tile.dtype)
        for k, (text, colour) in enumerate(((line1, (140, 255, 140) if not dim
                                             else (215, 215, 215)),
                                            (line2, (185, 185, 185)))):
            cv2.putText(band, text, (5, 14 + 15 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, colour, 1, cv2.LINE_AA)
        return np.vstack([band, tile])

    tiles, report = [], []

    # THE REFERENCE GOES FIRST, scored by `best_pair` like everything else, so
    # the "before" panel is the start pose alone.  A sheet without it cannot show
    # a change, only a state.
    ref_own = None
    if reference is not None:
        ref_own = best_pair(torch.as_tensor(built["probs_ref"]).float(),
                            built["boxes"], built["rel"])
        tiles.append(panel(
            np.ascontiguousarray(np.asarray(reference)[:, :, ::-1]).copy(),
            built["boxes"], None if ref_own is None else ref_own[1:3],
            "REFERENCE  (before moving, this frame alone)",
            "no candidate" if ref_own is None else
            f"p({task['subject_class']})={ref_own[3]:.3f}  "
            f"p({task['object_class']})={ref_own[4]:.3f}"))

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
        own = best_pair(probs, boxes, rel_v)

        pose = rendered[v]["pose"]
        report.append({"v": v,
                       "azimuth": round(float(pose["azimuth"]), 1),
                       "elevation": round(float(pose["elevation"]), 1),
                       "own_top1": None if own is None else own[1:],
                       "own_subject_box": None if own is None else
                       [round(t, 1) for t in boxes[own[1]].tolist()],
                       "own_p": None if own is None else
                       [round(own[3], 3), round(own[4], 3)]})

        skipped = v in ch.SKIP_VIEWS
        tiles.append(panel(
            np.ascontiguousarray(
                np.asarray(rendered[v]["frame"])[:, :, ::-1]).copy(),
            boxes, None if own is None else (own[1], own[2]),
            f"v{v:02d}  az {float(pose['azimuth']):+.0f}  "
            f"el {float(pose['elevation']):+.0f}"
            + ("  (duplicate of reference)" if skipped else ""),
            "no candidate" if own is None else
            f"p({task['subject_class']})={own[3]:.3f}  "
            f"p({task['object_class']})={own[4]:.3f}",
            dim=True))

    if not tiles:
        return report, ref_own

    # THE REFERENCE GETS ITS OWN ROW.  Inlined as tile 1 of 21 it read as just
    # another view, and the comparison the sheet exists for is start-pose against
    # everything else -- so the 20 swept views fill the rows beneath it.
    head, swept = (tiles[:1], tiles[1:]) if reference is not None \
        else ([], tiles)

    def band(row):
        while len(row) < columns:
            row.append(np.zeros_like(tiles[0]))
        return np.hstack(row)

    rows = [band(list(head))] if head else []
    rows += [band(swept[k:k + columns])
             for k in range(0, len(swept), columns)]
    cv2.imwrite(path, np.vstack(rows))
    print(f"    -> {path}")
    return report, ref_own


def decide_step(rc, egtr, reference, rendered, orbit, task, args) -> Dict:
    """THE DECISION AND THE STEP: `perceive`'s body without the simulator.

    Everything that reasons is shared -- `fuse_live.record` fuses the sweep,
    `fuse_live.conditioned` builds the instruction's candidates, and
    `robot.viewpick.pick_view` is the policy itself, the same call
    `eval_move.perceive` makes.  What is NOT shared is exactly what cannot be:
    the simulator's parking, its teleports and its ground-truth boxes.

    P-HAT IS NEVER COMPUTED.  `--motion arc` needs the detected landmark's 3D
    point and would put a detector in the loop that decides where to walk;
    `--motion pose` walks the chosen view's own relative transform about
    `orbit`, which is a depth reading on the optical axis.  A real robot takes
    the second, so this reports it and prints the arc's azimuth as a diagnostic
    only.
    """
    from lib.fusion import channels as ch
    from fuse_live import (CORR, GATE_COS, OBJSCORE, OBJSCORE_CLASS,
                           OBJSCORE_COS, POOL, conditioned, consensus_relabel,
                           record)
    from eval_move import step_to
    from robot import viewpick

    built = record(egtr, reference, [(i, r["frame"])
                                     for i, r in enumerate(rendered)])
    cand = conditioned(built, egtr, task, args.condition, args.cand_nms)
    if cand is None:
        print("  ! neither class is in VG150 -- no candidate list, no decision")
        return {"failed": "no candidate list"}
    which, margin, ballots = consensus_relabel(built, egtr, "argmax",
                                               per_view=True)
    s = ch.object_scores(built["rec"], mode=OBJSCORE,
                         class_mode=OBJSCORE_CLASS, cos=OBJSCORE_COS)
    from robot.task_find import iou

    rel = built["rel"]
    subjects, objects = cand
    boxes = built["boxes"]

    # A RELATION NEEDS TWO OBJECTS.  `i != j` only excludes the same query, and
    # EGTR emits several boxes per object, so a pair can be one object related to
    # itself.  THE CUT IS LOW BECAUSE THE BOXES NEST: a label inside the carton
    # it is printed on shares all of the smaller box, but IoU divides by the
    # UNION, so the pair that motivated this reads only 0.23.
    def distinct(i, j) -> bool:
        return i != j and (not args.pair_iou
                           or iou(boxes[i].tolist(),
                                  boxes[j].tolist()) < args.pair_iou)

    # TWO WEIGHTS FOR THE SAME RANKING, and on a real frame they disagree.
    #
    #   s       the query's confidence over ALL classes, which is what
    #           `fuse_live.decide` uses and what the simulator's numbers report.
    #   class   p(the instruction's own noun), the quantity `conditioned` already
    #           used to BUILD the list -- spent again on the ordering instead of
    #           discarded after selection.
    #
    # `eval_move` measured the swap as worth 2 of 40 rankings, `rel` spanning 20
    # orders of magnitude against `s`'s 1.5.  That holds where both objects are
    # detected well.  Here the target reads p(bag) 0.07 with s 0.09 while a
    # distractor reads s 0.21, so `s` elects the distractor and the difference is
    # the whole answer rather than a rounding error.  Both are computed and
    # reported; `--score` picks which one moves the robot.
    from fuse_live import CLASS_ALIASES

    columns = {v: k - 1 for k, v in egtr["obj_names"].items()}

    def mass(name: str):
        cols = [columns[c] for c in CLASS_ALIASES.get(name, (name,))
                if c in columns]
        return built["probs_ref"][:, cols].sum(-1)

    subject_p, object_p = mass(task["subject_class"]), mass(task["object_class"])

    predicate = egtr["rel_names"].index(task["predicate"])

    # CHANNEL B, THE ONE THING THAT CAN LIFT AN OCCLUDED PAIR.  A repairs a
    # QUERY's class confidence and its `gate` needs the view's argmax to match
    # the reference's -- which an occluded target's rarely does, so A lifts the
    # unoccluded distractor instead and widens the gap it exists to close.  B
    # asks a different question, per PAIR: do the other views agree that these
    # two slots stand in this relation.  Nothing about naming, so nothing for the
    # class gate to block.
    #
    # READ AT THE INSTRUCTED PREDICATE, not at `rel.argmax`.  `evidence_at` has
    # to guess what the reference proposed; an instruction says it, so the column
    # is known and a pair the reference gives no mass to can still be confirmed.
    #   mean  the average `behind` over the views that see both endpoints
    #   max   the strongest single such view -- the same shape as `rel`, which is
    #         itself one observation, and the same pooling A already uses
    #         (`OBJSCORE = "max"`, "the single best view.  Where the gain is").
    #         So beta becomes comparable to 1; the cost is that one bad view can
    #         set a pair's evidence, and SEVA does visibly break at large angles.
    ev, evs = None, {}
    if args.beta or args.w or args.pool == "both":
        for pool in (("mean", "max") if args.pool == "both" else (args.pool,)):
            agg, cnt, stats = ch.view_evidence_perpred(
                built["rec"], pool=pool, gate_cos=GATE_COS, corr=CORR)
            if agg is None:
                continue
            evs[pool] = ch.evidence_field(agg, cnt, pool)[:, :, predicate]
            print(f"  channel B ({pool})  {stats.get('n_views', 0)} views, "
                  f"{100 * stats.get('pairs_covered', 0):.0f}% of pairs covered, "
                  f"{stats.get('views_per_pair', 0):.1f} views per pair")
        ev = evs.get(args.pool if args.pool != "both" else "max")

    # ONE RANKING.  `eval_move.look`'s rule -- `rel[i, j, predicate]` weighted by
    # the object scores -- because on a robot the only thing this list is for is
    # WHICH PAIR THE SENTENCE MEANS, and that is what `look` answers.  `perceive`
    # keeps a second, predicate-free ranking (`rel.max()`) for two consumers a
    # robot has neither of: P-hat, which `--motion arc` orbits, and the pair
    # `--bearing acr` tracks across views.  `reveal` needs no pair at all.
    # THE TWO CHANNELS, EACH NORMALISED WITHIN THIS IMAGE'S CANDIDATE SET, so `w`
    # is a mixing weight and not a scale correction.  Raw `rel + beta * ev` needs
    # beta ~= 30 here purely because a single confident reference value outruns a
    # cross-view mean by a decade and a half, and no such beta can be reported as
    # a choice.  Dividing each term by its own best candidate puts both in [0, 1];
    # w = 0.5 then means the channels count equally, and it is comparable across
    # images.  w = 0 reproduces A+C exactly.
    def score_of(i, j) -> float:
        base = float(rel[i, j, predicate])
        if ev is not None:
            base = ((1.0 - args.w) * base / rel_top
                    + args.w * float(ev[i, j]) / ev_top)
        return base * float(s[i]) * float(s[j])

    cells0 = [(a, b) for a in subjects for b in objects if distinct(a, b)]
    rel_top = max((float(rel[a, b, predicate]) for a, b in cells0),
                  default=0.0) or 1.0
    ev_top = (max((float(ev[a, b]) for a, b in cells0), default=0.0) or 1.0) \
        if ev is not None else 1.0

    answer = sorted(((score_of(i, j), i, j)
                     for i in subjects for j in objects if distinct(i, j)),
                    reverse=True)
    if not answer:
        print("  ! every candidate pair is one object related to itself")
        return {"failed": "no distinct pair"}

    _, i, j = answer[0]
    if ev is not None:
        plain = max(((float(rel[a, b, predicate]) * float(s[a]) * float(s[b]),
                      a, b) for a in subjects for b in objects
                     if distinct(a, b)), default=None)
        if plain is not None:
            # BOTH MAGNITUDES, because beta is only meaningful against them:
            # `rel` on these pairs runs many orders of magnitude below 1 and `ev`
            # is a mean of the same kind of quantity, so a beta picked blind
            # either does nothing or swamps the reference entirely.
            cells = [(a, b) for a in subjects for b in objects if distinct(a, b)]
            rels = [float(rel[a, b, predicate]) for a, b in cells]
            evcol = [float(ev[a, b]) for a, b in cells]
            print(f"\n  over {len(cells)} candidate pairs:"
                  f"   rel max {max(rels):.3e} median {sorted(rels)[len(rels)//2]:.3e}"
                  f"   |   ev max {max(evcol):.3e} median "
                  f"{sorted(evcol)[len(evcol)//2]:.3e}")
            for name, (a, b) in (("w=0  (A+C)", plain[1:]),
                                 (f"w={args.w:g} (A+C+B)", (i, j))):
                print(f"  {name:18s} {[round(v, 1) for v in boxes[a].tolist()]}"
                      f"   rel={float(rel[a, b, predicate]):.3e}"
                      f"  ev={float(ev[a, b]):.3e}")
            print("  " + ("-- SAME PAIR" if plain[1:] == (i, j)
                          else "-- B CHANGED THE PAIR"))
            # WHERE B WOULD SEND US ON ITS OWN.  Deciding a beta needs to know
            # whether the evidence points at the target or merely elsewhere: a
            # beta large enough to let B speak is only worth setting if the pairs
            # it favours are the right ones.
            # A BETA SWEEP, FREE: the sweep is already synthesised and the two
            # fields are already computed, so the only thing a second job would
            # buy is the same numbers again.  `rel` and `ev` sit orders of
            # magnitude apart, so which beta lets B speak is the question, and it
            # cannot be answered by picking one.
            for pool, field in evs.items():
                col = [float(field[a, b]) for a, b in cells]
                print(f"\n  --- pool={pool} ---   ev max {max(col):.3e} "
                      f"median {sorted(col)[len(col) // 2]:.3e}")
                print("  which pair wins at each beta:")
                for b in (0, 0.3, 1, 3, 10, 30, 100, 300):
                    win = max((((float(rel[a, c, predicate])
                                 + b * float(field[a, c]))
                                * float(s[a]) * float(s[c])), a, c)
                              for a, c in cells)
                    print(f"    beta {b:>6g}   subject "
                          f"{[round(v, 1) for v in boxes[win[1]].tolist()]}")
                print("  top pairs by ev ALONE:")
                for n, (e, a, b) in enumerate(sorted(
                        ((float(field[a, b]), a, b) for a, b in cells),
                        reverse=True)[:4], 1):
                    print(f"    {n}. ev={e:.3e}  "
                          f"rel={float(rel[a, b, predicate]):.3e}"
                          f"  subject {[round(v, 1) for v in boxes[a].tolist()]}")

                # BETA HAS NO UNITS, so it cannot be reported.  Both terms are
                # normalised WITHIN this image's candidate set instead -- the
                # cross-view pooling already happened, so this is per pair -- and
                # the weight becomes a real mixing weight: w = 0.5 is "the two
                # channels count equally".
                #
                #   norm   divide by the term's best candidate.  Keeps how much a
                #          pair wins by; one outlier flattens everything else.
                #   rank   the pair's place under each term.  Immune to scale and
                #          to outliers, which is why `volatility` ranks -- `rel`
                #          drifts ~17 orders across views -- and blind to margin.
                def unit_by_max(values):
                    top = max(values) or 1.0
                    return [v / top for v in values]

                def unit_by_rank(values):
                    place = {k: n for n, (_, k) in enumerate(
                        sorted(((v, k) for k, v in enumerate(values)),
                               reverse=True))}
                    return [1.0 - place[k] / max(len(values) - 1, 1)
                            for k in range(len(values))]

                relcol = [float(rel[a, b, predicate]) for a, b in cells]
                evcol2 = [float(field[a, b]) for a, b in cells]
                for name, unit in (("norm", unit_by_max),
                                   ("rank", unit_by_rank)):
                    r, e = unit(relcol), unit(evcol2)
                    print(f"  {name}: which pair wins at each w "
                          f"(0 = rel only, 1 = ev only)")
                    for w in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
                        best = max(((1 - w) * r[k] + w * e[k], k)
                                   for k in range(len(cells)))
                        a, b = cells[best[1]]
                        print(f"    w {w:>4.1f}   subject "
                              f"{[round(v, 1) for v in boxes[a].tolist()]}")
    print(f"\n  the pair, rel[.,.,{task['predicate']}] x s x s:")
    print(f"    subject {[round(v, 1) for v in boxes[i].tolist()]}"
          f"   p({task['subject_class']})={float(subject_p[i]):.3f}"
          f"  s={float(s[i]):.3f}")
    print(f"    object  {[round(v, 1) for v in boxes[j].tolist()]}"
          f"   p({task['object_class']})={float(object_p[j]):.3f}"
          f"  s={float(s[j]):.3f}")

    record_out: Dict[str, Any] = {
        "candidates": len(answer), "subjects": len(subjects),
        "objects": len(objects),
        "top1_subject_box": [round(v, 1) for v in boxes[i].tolist()],
        "top1_object_box": [round(v, 1) for v in boxes[j].tolist()],
        "top1_p": [round(float(subject_p[i]), 3), round(float(object_p[j]), 3)]}
    print(f"  candidates  {len(subjects)} x {len(objects)} "
          f"-> {len(answer)} distinct pairs")
    if args.out:
        views, ref_own = predicted_sheet(
            built, rendered, task, egtr, predicate, args,
            os.path.join(args.out, "sweep_pred.png"), reference=reference)
        record_out["views"] = views
        # The reference judged the way every view is judged: this frame alone,
        # no fusion.  NOT the pair printed above, which is weighted by
        # `ch.object_scores` and so already carries the sweep's evidence.
        if ref_own is not None:
            record_out["ref_own_top1"] = list(ref_own[1:3])
            record_out["ref_own_subject_box"] = [
                round(v, 1) for v in boxes[ref_own[1]].tolist()]
            record_out["ref_own_p"] = [round(ref_own[3], 3),
                                       round(ref_own[4], 3)]
            same = tuple(ref_own[1:3]) == (i, j)
            print(f"  the reference ALONE (no fusion) picks "
                  f"{record_out['ref_own_subject_box']}"
                  f"  p={ref_own[3]:.3f}/{ref_own[4]:.3f}"
                  + ("  -- same pair" if same else "  -- A+C changed the pair"))
        got = [v for v in views if v["own_p"]]
        if got:
            best = max(got, key=lambda v: v["own_p"][0])
            print(f"  per-view p({task['subject_class']}) of its own top-1: "
                  f"{min(v['own_p'][0] for v in got):.3f} .. "
                  f"{best['own_p'][0]:.3f}"
                  f"   highest at az {best['azimuth']:+.0f}")

    # The first pair whose fused label names the instructed predicate.  Only
    # `--bearing acr` and `edge` read it; `reveal` needs no pair.
    chosen = None
    for a, b in ((a, b) for _, a, b in answer):
        named = int(which[a, b]) if float(margin[a, b]) >= 0.0 \
            else int(rel[a, b].argmax())
        if named == predicate:
            chosen = (a, b)
            break

    # `attrib` scores each view by its own contribution to THE PAIR THE FUSION
    # CHOSE, so it must be handed that pair rather than the lexical `chosen`,
    # which is R's gate over a different ranking.
    votes, view, scores = viewpick.pick_view(
        args.bearing, args.side_step, built, rendered, cand, egtr, task,
        answer, (i, j) if args.bearing == "attrib" else chosen, ballots)
    record_out["view_scores"] = scores
    if view is None:
        print(f"  ! `--bearing {args.bearing}` named no view")
        return {**record_out, "failed": f"{args.bearing} named no view"}
    record_out["chosen_azimuth"] = round(float(view["azimuth"]), 1)
    record_out["chosen_elevation"] = round(float(view["elevation"]), 1)

    # THE SCORES, WHICH ARE THE METHOD.  Printed in azimuth order rather than
    # score order so the shape over the sweep is readable: which SIDE the rule
    # prefers, and how sharply, is the claim -- the argmax alone hides both.
    print(f"\n  `{args.bearing}` scores each view (az, then score):")
    # WHATEVER THE RULE EMITTED, not a fixed list: `attrib` reports support and
    # rival, `reveal` reports correspond/appearance/relation, and a hardcoded
    # list silently dropped the terms of any rule added later.
    terms = [k for k in (scores[0] if scores else {})
             if k not in ("v", "azimuth", "elevation", "score")]
    head = "".join(f"{k[:9]:>11}" for k in terms)
    print(f"      {'az':>6}{'el':>5}{'score':>11}{head}")
    for row in sorted(scores, key=lambda r: r["azimuth"]):
        mark = (" <-- chosen"
                if (row["azimuth"], row["elevation"])
                == (round(float(view["azimuth"]), 1),
                    round(float(view["elevation"]), 1)) else "")
        print(f"      {row['azimuth']:>+6.1f}{row['elevation']:>+5.0f}"
              f"{row['score']:>11.3e}"
              + "".join(f"{row[k]:>11.3e}" for k in terms) + mark)

    if step_to(rc, view, orbit, args) is None:
        print("  ! no reachable pose -- see real_robot.teleport, which cannot "
              "refuse one, so this means the arithmetic failed")
        return {**record_out, "failed": "step_to returned nothing"}
    step = rc.moved_to()
    record_out["step"] = step
    ros = step["ros"]
    print("\n  THE STEP, relative to where the robot stands right now.")
    print("  REP-103 (x forward, y LEFT, theta counter-clockwise) -- ROS2 goal:")
    print(f"    x      {ros['x']:+.3f} m")
    print(f"    y      {ros['y']:+.3f} m")
    print(f"    theta  {ros['theta']:+.4f} rad  ({-step['turn']:+.1f} deg)")
    print(f"    -> in words: "
          f"{'forward' if ros['x'] >= 0 else 'back'} {abs(ros['x']):.3f} m, "
          f"{'left' if ros['y'] >= 0 else 'right'} {abs(ros['y']):.3f} m, "
          f"turn {'left' if ros['theta'] >= 0 else 'right'} "
          f"{abs(step['turn']):.1f} deg")
    print(f"    ({step['distance']:.3f} m total; translate first, then turn)")
    # The simulator's own numbers, kept visible because every other quantity
    # printed above -- the azimuths, the poses -- is in that frame, and a reader
    # comparing them needs to see where the sign flip happened.
    print(f"  Unity/THOR frame, for comparison with the azimuths above: "
          f"forward {step['forward']:+.3f}  right {step['right']:+.3f}  "
          f"yaw {step['turn']:+.1f} deg")
    return record_out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dir", default="nvs_pilot/real",
                    help="folder the robot drops its frames in; the newest "
                         "rgb/depth pair is used")
    ap.add_argument("--rgb", default=None, help="override --dir")
    ap.add_argument("--depth", default=None, help="override --dir")
    ap.add_argument("--depth-scale", type=float, default=0.001,
                    metavar="M", help="metres per unit in a 16-bit depth PNG; "
                                      "0.001 for RealSense millimetres")
    ap.add_argument("--camera-height", type=float, default=0.0, metavar="M",
                    help="camera above the floor.  Only shifts the sweep's "
                         "elevation, which the body cannot follow anyway.")
    ap.add_argument("--pitch", type=float, default=0.0, metavar="DEG",
                    help="camera tilt, positive DOWN")
    ap.add_argument("--fov", type=float, default=FOV_V, metavar="DEG",
                    help="VERTICAL field of view.  The THOR default is 60; a "
                         "RealSense D435 colour stream is 42.5, and this is "
                         "read as the vertical field everywhere downstream.")
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--max-az", type=float, default=30.0)
    ap.add_argument("--max-el", type=float, default=15.0)
    ap.add_argument("--no-egtr", action="store_true")
    ap.add_argument("--no-synth", action="store_true",
                    help="skip SEVA, for when only the depth is in question")
    ap.add_argument("--synth-steps", type=int, default=10)
    ap.add_argument("--synth-cfg", type=float, default=3.0)
    ap.add_argument("--synth-camera-scale", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--task", default=None, metavar="SUBJ,PRED,OBJ",
                    help="the instruction in VG150 words, e.g. --task "
                         "box,behind,bag.  With it, the sweep is turned into a "
                         "step; without it, this stops at the contact sheet.")
    ap.add_argument("--bearing", default="reveal",
                    choices=("attrib", "bin", "visible", "side", "volatility",
                             "reveal", "node", "edge", "acr"),
                    help="which rule picks the view; see robot/viewpick.py")
    ap.add_argument("--side-step", type=float, default=10.0)
    ap.add_argument("--condition", type=int, default=10)
    ap.add_argument("--cand-nms", type=float, default=0.0)
    ap.add_argument("--pool", choices=("mean", "max", "both"), default="both",
                    help="how channel B pools a pair's evidence ACROSS VIEWS. "
                         "`both` computes each and reports them; the ranking "
                         "then uses max.")
    ap.add_argument("--w", type=float, default=0.0, metavar="W",
                    help="mixing weight on channel B, both terms normalised "
                         "within the candidate set.  0 = A+C only, 1 = B only. "
                         "Measured to flip on the first real episode between "
                         "0.5 and 0.6.")
    ap.add_argument("--beta", type=float, default=0.0, metavar="B",
                    help="compute channel B and report the un-normalised sweep. "
                         "Does NOT drive the ranking -- `--w` does.")
    ap.add_argument("--pair-iou", type=float, default=0.15, metavar="IOU",
                    help="reject a pair whose two boxes overlap this much -- it "
                         "is one object related to itself.  Low because the "
                         "boxes NEST: a label inside its own carton reads 0.23. "
                         " 0 disables.")
    ap.add_argument("--at", default=None, metavar="X0,Y0,X1,Y1",
                    help="a pixel box round an object you care about; reports "
                         "what EGTR calls whatever it has there, distribution "
                         "and all")
    ap.add_argument("--classes", default="",
                    help="comma-separated VG150 classes the instruction names, "
                         "e.g. --classes box,bag.  Drawn GREEN and reported "
                         "whether or not they rank, because a candidate list is "
                         "built per class and not off the top of the ranking.")
    ap.add_argument("--out", default=None,
                    help="where to write egtr.png and sweep.png; defaults to "
                         "<--dir>/out")
    args = ap.parse_args(argv)

    import cv2

    rgb_path = args.rgb or newest_pair(args.dir)[0]
    depth_path = args.depth or newest_pair(args.dir)[1]
    out = args.out = args.out or os.path.join(args.dir, "out")
    os.makedirs(out, exist_ok=True)

    frame = cv2.imread(rgb_path, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    print(f"  rgb        {os.path.basename(rgb_path)}  "
          f"{frame.shape[1]}x{frame.shape[0]}")
    if not aligned(depth_path, frame.shape):
        print("  ! depth and colour are different resolutions, so they are the "
              "camera's RAW streams.\n"
              "    Resampling to match is NOT registration: the centre depth "
              "below is usable,\n"
              "    but any depth read inside a detection box is off by the "
              "stereo parallax.\n"
              "    Record with rs.align(rs.stream.color) before trusting a "
              "walk.")
    depth = load_depth(depth_path, frame.shape, args.depth_scale)
    radius = depth_report(depth)

    rc = RealRobot(frame, depth, args.camera_height, args.pitch)

    if not args.no_egtr:
        from robot.sgg_live import load_egtr

        wanted = [c.strip() for c in args.classes.split(",") if c.strip()]
        at = ([float(v) for v in args.at.split(",")] if args.at else None)
        egtr_report(load_egtr(), frame, args.top, wanted, at, out)

    if args.no_synth:
        return 0

    # THE SAME POSES `perceive` WOULD BUILD, from the same two lines, so the
    # sheet below is the sweep this pipeline actually reasons over and not a
    # look-alike.
    from robot.nvs_lemniscate import LOOKAT_DIST, camera_for, lemniscate

    from robot.nvs_lemniscate import look_at_point

    orbit = look_at_point(rc.camera_xyz, rc.agent_yaw, rc.camera_horizon,
                          radius)
    poses = [camera_for(orbit, rc.camera_xyz, az, el)
             for az, el in lemniscate(args.views, args.max_az, args.max_el)]
    span = max(math.dist([p["position"]["x"], p["position"]["z"]], [0.0, 0.0])
               for p in poses)
    print(f"\n  sweep      {len(poses)} views, +-{args.max_az:.0f} deg about a "
          f"point {radius:.2f} m ahead")
    print(f"             the furthest view stands {span:.2f} m from here")

    from robot.nvs_seva import Synthesiser

    print("\n  loading SEVA (5 GB, ~1 min) ...", flush=True)
    synth = Synthesiser(steps=args.synth_steps, cfg=args.synth_cfg,
                        camera_scale=args.synth_camera_scale,
                        lookat_dist=LOOKAT_DIST, fov=args.fov)
    frames = synth(frame, poses)
    contact_sheet(frame, frames, poses, os.path.join(out, "sweep.png"))
    for index, view in enumerate(frames):
        cv2.imwrite(os.path.join(out, f"view_{index:02d}.png"),
                    np.asarray(view)[:, :, ::-1])
    print(f"    -> {out}/view_XX.png")

    if not args.task:
        return 0

    subject, predicate, obj = (w.strip() for w in args.task.split(","))
    rendered = [{"pose": pose, "frame": np.asarray(view), "pixels": None,
                 "box": None} for pose, view in zip(poses, frames)]
    from robot.sgg_live import load_egtr

    got = decide_step(rc, load_egtr(), frame, rendered, orbit,
                      {"subject_class": subject, "predicate": predicate,
                       "object_class": obj,
                       "instruction": f"the {subject} {predicate} the {obj}"},
                      args)
    write_record(args, rgb_path, depth_path, radius, poses, got)
    return 0


def write_record(args, rgb_path: str, depth_path: str, radius: float,
                 poses, got: Dict[str, Any]) -> None:
    """`step.json` in the step folder, one appended row in the episode's TSV.

    The TSV is what a walk IS -- a sequence of steps somebody executed -- and it
    goes one level up, beside the step folders rather than inside one, so an
    episode reads as a trajectory without opening anything.
    """
    import json

    step_dir = os.path.abspath(args.dir)
    record = {"rgb": os.path.basename(rgb_path),
              "depth": os.path.basename(depth_path),
              "task": args.task, "instruction": args.task.replace(",", " "),
              "bearing": args.bearing,
              "camera_height": args.camera_height, "pitch": args.pitch,
              "fov": args.fov, "depth_scale": args.depth_scale,
              "sweep": {"views": len(poses), "max_az": args.max_az,
                        "max_el": args.max_el, "radius": round(radius, 3)},
              **got}
    with open(os.path.join(step_dir, "step.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"\n  -> {os.path.join(args.dir, 'step.json')}")

    # Beside the step folders when this IS one; inside otherwise.
    name = os.path.basename(step_dir)
    episode = os.path.dirname(step_dir) if name.startswith("step") else step_dir
    tsv = os.path.join(episode, "steps.tsv")
    columns = ("step", "x", "y", "theta_rad", "theta_deg", "chosen_az",
               "radius", "candidates", "note")
    step = got.get("step") or {}
    ros = step.get("ros") or {}
    row = (name, ros.get("x", ""), ros.get("y", ""), ros.get("theta", ""),
           "" if not step else round(-step["turn"], 1),
           got.get("chosen_azimuth", ""), round(radius, 3),
           got.get("candidates", ""), got.get("failed", ""))
    fresh = not os.path.exists(tsv)
    with open(tsv, "a") as fh:
        if fresh:
            fh.write("\t".join(columns) + "\n")
        fh.write("\t".join(str(v) for v in row) + "\n")
    print(f"  -> {tsv}")


if __name__ == "__main__":
    raise SystemExit(main())
