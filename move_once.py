"""
move_once.py -- one look at one real frame, before anything is asked to move.

The simulator's numbers say what this pipeline does when the sweep is RENDERED
by the thing it is trying to predict.  On a real robot the sweep is SEVA's, from
one photograph, and nothing in `nvs_pilot/` says what that looks like.  So this
reports one step and stops -- nothing here walks.

    THE DEPTH        the only place metric scale enters.  SEVA never sees it, so
                     a depth error leaves the chosen VIEW alone and scales the WALK.
    THE SWEEP        as a contact sheet.  The method assumes a synthesised view
                     can be read by a detector, checked so far only on THOR renders.
    THE STEP         with `--task`: the pair, the chosen view, and a ROS2 goal.

    python move_once.py --dir nvs_pilot/real --camera-height 0.15
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from robot.nvs_lemniscate import orbit_depth
from robot.real_robot import FOV_V, RealRobot, aligned, load_depth, newest_pair




def decide_step(rc, egtr, reference, rendered, orbit, task, args) -> Dict:
    """THE DECISION AND THE STEP: `eval_move.perceive`'s body without the simulator.

    Everything that reasons is shared -- `fuse_live.record` fuses the sweep,
    `grounding` builds and ranks the candidates, `viewpick.pick_view` is the
    policy.  What is not shared cannot be: the simulator's parking, its teleports
    and its ground-truth boxes.

    P-HAT IS NEVER COMPUTED.  It would need a detected landmark's 3D point, which
    puts a detector in the loop that decides where to WALK.  The step here is the
    chosen view's own relative transform about `orbit`, a depth reading on the
    optical axis.
    """
    from lib.fusion import channels as ch
    from fuse_live import (OBJSCORE, OBJSCORE_CLASS, OBJSCORE_COS, conditioned,
                           consensus_relabel, record)
    from eval_move import step_to
    from robot import evidence, viewpick
    from robot.grounding import (class_mass, pair_cells, per_view_answers,
                                 rank_pairs)

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
    rel = built["rel"]
    subjects, objects = cand
    boxes = built["boxes"]

    # A RELATION NEEDS TWO OBJECTS, and on a real photograph `i != j` is not
    # enough -- see `grounding.pair_cells` for why the overlap cut is so low.
    # THE ONE PLACE THIS RANKING DIVERGES FROM `eval_move.look`, which applies no
    # filter at all; both call the same function and say so with an argument.
    cells = pair_cells(cand, boxes=boxes, pair_iou=args.pair_iou)

    # TWO WEIGHTS FOR THE SAME RANKING, and on a real frame they disagree.  On
    # rendered frames the swap was worth 2 of 40 rankings; here the target reads
    # p(bag) 0.07 with s 0.09 while a distractor reads s 0.21, so `s` elects the
    # distractor.  Both are reported; `s` is what ranks.
    subject_p = class_mass(built["probs_ref"], egtr, task["subject_class"])
    object_p = class_mass(built["probs_ref"], egtr, task["object_class"])

    predicate = egtr["rel_names"].index(task["predicate"])

    # CHANNEL B, THE ONE THING THAT CAN LIFT AN OCCLUDED PAIR.  A repairs a
    # QUERY's class confidence behind a gate needing the view's argmax to match
    # the reference's, which an occluded target's rarely does -- so A lifts the
    # unoccluded distractor instead.  B asks per PAIR: do the other views agree
    # these two slots stand in this relation.  Nothing about naming, so nothing
    # for the class gate to block.
    #
    # READ AT THE INSTRUCTED PREDICATE, not `rel.argmax` -- the instruction names
    # the column, so a pair the reference gives no mass to can still be confirmed.
    #   mean  the average over the views that see both endpoints
    #   max   the strongest single such view.  Same pooling A uses, so beta
    #         becomes comparable to 1; the cost is that one bad view can set a
    #         pair's evidence, and SEVA does visibly break at large angles.
    ev, evs = None, {}
    if args.beta or args.w or args.pool == "both":
        # ONE PASS OVER THE VIEWS for both poolings -- and it is the same table
        # `--bearing attrib` reads down the other axis.  See `robot.evidence`.
        _, contrib, spoke = evidence.pair_contributions(built, cells, predicate)
        stats = evidence.coverage(spoke)
        print(f"  channel B  {stats['n_views']} views, "
              f"{100 * stats['pairs_covered']:.0f}% of the candidate pairs "
              f"covered, {stats['views_per_pair']:.1f} views per pair")
        for pool in (("mean", "max") if args.pool == "both" else (args.pool,)):
            evs[pool] = evidence.pooled(cells, contrib, spoke, pool)
        ev = evs.get(args.pool if args.pool != "both" else "max")

    # ONE RANKING, `eval_move.look`'s: on a robot the only thing this list is for
    # is WHICH PAIR THE SENTENCE MEANS.  `eval_move.perceive` keeps a second,
    # predicate-free one for consumers a robot has neither of -- P-hat, and the
    # pair `--bearing acr` tracks.  Both channels are normalised within this
    # image's candidate set, so `w` is a mixing weight: 0.5 counts them equally,
    # 0 reproduces A+C exactly.
    answer = rank_pairs(rel, cand, predicate, s, s, boxes=boxes,
                        pair_iou=args.pair_iou, mix=ev, weight=args.w)
    if not answer:
        print("  ! every candidate pair is one object related to itself")
        return {"failed": "no distinct pair"}

    _, i, j = answer[0]
    if ev is not None:
        # THE SAME RANKING WITH B SWITCHED OFF, so the two can be shown side by
        # side.  `mix=None` is exactly `eval_move.look`'s call.
        plain = rank_pairs(rel, cand, predicate, s, s, boxes=boxes,
                           pair_iou=args.pair_iou)
        plain = plain[0] if plain else None
        if plain is not None:
            # BOTH MAGNITUDES: `rel` runs orders of magnitude below 1 and `ev`
            # is a mean of the same quantity, so a blind beta either does nothing
            # or swamps the reference entirely.
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
            # A BETA SWEEP, AND IT IS FREE -- the sweep and both fields are
            # already computed.  A beta large enough to let B speak is only worth
            # setting if the pairs it favours are the right ones, so this shows
            # where B would send us on its own.
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
    print(f"\n  the pair the SWEEP is aimed at, "
          f"rel[.,.,{task['predicate']}] x s x s:")
    print(f"    subject {[round(v, 1) for v in boxes[i].tolist()]}"
          f"   p({task['subject_class']})={float(subject_p[i]):.3f}"
          f"  s={float(s[i]):.3f}")
    print(f"    object  {[round(v, 1) for v in boxes[j].tolist()]}"
          f"   p({task['object_class']})={float(object_p[j]):.3f}"
          f"  s={float(s[j]):.3f}")

    # `pick_*`, NOT `top1_*`.  This pair is the FUSED ranking and its only job is
    # to aim the viewpoint rule; the ANSWER is `top1_*` below -- `single_frame` on
    # the reference alone, the same call `eval_move.look` grades a pose with.
    # Both were called top1 and the record had two of them.
    record_out: Dict[str, Any] = {
        "candidates": len(answer), "subjects": len(subjects),
        "objects": len(objects),
        "pick_subject_box": [round(v, 1) for v in boxes[i].tolist()],
        "pick_object_box": [round(v, 1) for v in boxes[j].tolist()],
        "pick_p": [round(float(subject_p[i]), 3), round(float(object_p[j]), 3)]}
    print(f"  candidates  {len(subjects)} x {len(objects)} "
          f"-> {len(answer)} distinct pairs")
    if args.out:
        from viz import sweep as sweep_figures

        views, ref_row = per_view_answers(
            built, rendered, task, egtr, width=args.condition,
            nms=args.cand_nms, weight=args.weight, pair_iou=args.pair_iou)
        sweep_figures.answer_sheet(
            reference, rendered, views, ref_row, task["subject_class"],
            task["object_class"], os.path.join(args.out, "sweep_pred.png"))
        record_out["views"] = views
        # THE ANSWER: this frame alone, no fusion.  A walk is graded by running
        # this again at the pose it reached, which is the next `move_once`.
        if ref_row["own_p"]:
            record_out["top1"] = list(ref_row["own_top1"][:2])
            record_out["top1_subject_box"] = ref_row["own_subject_box"]
            record_out["top1_object_box"] = ref_row["own_object_box"]
            record_out["top1_p"] = ref_row["own_p"]
            same = tuple(ref_row["own_top1"][:2]) == (i, j)
            print(f"\n  THE ANSWER at this pose -- {task['instruction']}:")
            print(f"    subject {ref_row['own_subject_box']}"
                  f"   p={ref_row['own_p'][0]:.3f}/{ref_row['own_p'][1]:.3f}"
                  + ("   (the fused pick agrees)" if same
                     else "   (the fused pick, which aims the sweep, differs)"))
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

    # `attrib` scores a view by its contribution to THE PAIR THE FUSION CHOSE,
    # so it gets that pair, not the lexical `chosen`.
    votes, view, scores = viewpick.pick_view(
        args.bearing, args.side_step, built, rendered, cand, egtr, task,
        answer, (i, j) if args.bearing == "attrib" else chosen, ballots)
    record_out["view_scores"] = scores
    if view is None:
        print(f"  ! `--bearing {args.bearing}` named no view")
        return {**record_out, "failed": f"{args.bearing} named no view"}
    record_out["chosen_azimuth"] = round(float(view["azimuth"]), 1)
    record_out["chosen_elevation"] = round(float(view["elevation"]), 1)

    # THE SCORES, WHICH ARE THE METHOD.  In AZIMUTH order, not score order: the
    # claim is which side the rule prefers and how sharply, and the argmax alone
    # hides both.
    print(f"\n  `{args.bearing}` scores each view (az, then score):")
    # Whatever terms the rule emitted -- a hardcoded list silently dropped the
    # terms of any rule added later.
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
    # The simulator's own frame, kept visible: every azimuth printed above is in
    # it, and a reader needs to see where the sign flip happened.
    print(f"  Unity/THOR frame, for comparison with the azimuths above: "
          f"forward {step['forward']:+.3f}  right {step['right']:+.3f}  "
          f"yaw {step['turn']:+.1f} deg")
    return record_out


def main(argv: Optional[Sequence[str]] = None) -> int:
    from robot.grounding import WEIGHTS

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
    ap.add_argument("--synth-steps", type=int, default=10)
    ap.add_argument("--synth-cfg", type=float, default=3.0)
    ap.add_argument("--synth-camera-scale", type=float, default=1.0)
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
    ap.add_argument("--weight", choices=WEIGHTS, default="class",
                    help="what the METRIC's ranking weights a query by; see "
                         "eval_move.py, which shares the call and the default.")
    ap.add_argument("--pair-iou", type=float, default=0.15, metavar="IOU",
                    help="reject a pair whose two boxes overlap this much -- it "
                         "is one object related to itself.  Low because the "
                         "boxes NEST: a label inside its own carton reads 0.23. "
                         " 0 disables.")
    ap.add_argument("--out", default=None,
                    help="where to write sweep.png and sweep_pred.png; "
                         "defaults to <--dir>/out")
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
    # THE SWEEP'S RADIUS, and the only place metric scale enters the chain.
    # `nvs_lemniscate.orbit_depth` is the same call `eval_move.perceive` makes,
    # so the simulator and the robot cannot centre their sweeps differently.
    depth = load_depth(depth_path, frame.shape, args.depth_scale)
    radius = orbit_depth(depth)
    if radius is None:
        raise SystemExit("  ! the centre of the depth frame is entirely invalid "
                         "-- the sweep radius is undefined and nothing below "
                         "would be meaningful")

    rc = RealRobot(frame, depth, args.camera_height, args.pitch)

    # THE SAME POSES `perceive` WOULD BUILD, from the same two lines, so the
    # sheet below is the sweep this pipeline actually reasons over and not a
    # look-alike.
    from robot.nvs_lemniscate import (LOOKAT_DIST, camera_for, lemniscate,
                                      look_at_point)

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
    from viz.sweep import contact_sheet

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
    # `p_subject` is THE ANSWER at this pose, so the trajectory shows whether
    # walking helped -- the other columns only say where the robot went.
    columns = ("step", "p_subject", "x", "y", "theta_rad", "theta_deg",
               "chosen_az", "radius", "candidates", "note")
    step = got.get("step") or {}
    ros = step.get("ros") or {}
    row = (name, (got.get("top1_p") or [""])[0],
           ros.get("x", ""), ros.get("y", ""), ros.get("theta", ""),
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
