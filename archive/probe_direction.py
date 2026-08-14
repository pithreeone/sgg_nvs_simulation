"""
probe_direction.py -- does EGTR's predicate head represent depth ORDER at all?

`probe_behind.py` measures where `behind` ranks for a pair that really is in
that relation, and the answer on tabletop pairs is "not first".  That is
compatible with two very different diagnoses, and they call for opposite
responses:

  ABSENT     the head has no depth-order concept and emits a class-pair prior.
             Then `rel[i, j, p]` is roughly symmetric in `(i, j)`: swapping the
             two slots does not change what it says.  Nothing downstream can
             recover a quantity that was never computed, and `behind` has to be
             abandoned as the instructed predicate.

  INVERTED   the head does order the pair, with the wrong sign, or names the
             correct ordering with the converse word.  Then the information IS
             there -- `rel[i, j, behind]` discriminating the true copy from its
             twin in 28 of 40 (reports/0812.md) is the same observation -- and
             the fix is in the decision rule, not the dataset.

The two are told apart by asking the SAME question in both slot orders.  For a
staged pair where the target is genuinely behind the occluder:

    fwd_behind  = rel[target, occluder, `behind`]        should be high
    rev_behind  = rel[occluder, target, `behind`]        should be low
    fwd_front   = rel[target, occluder, `in front of`]   should be low
    rev_front   = rel[occluder, target, `in front of`]   should be high

`agree` counts the pairs where the geometrically correct orientation carries
more mass than the flipped one, per predicate.  A head that is merely absent
scores about half; one that is inverted scores well below half; one that is
right but outvoted by the class prior scores well above.

This measures ONE number per staged pair and needs no sweep, so it is cheap
next to `probe_behind.py` -- the staging is the same code path, deliberately, so
the two probes describe the same scenes.

    python probe_direction.py --assets 3
"""

from __future__ import annotations

# `python archive/<script>.py` puts archive/ on sys.path, not the repo root.
# These were moved here without it, so they could not import the generators at
# all; same shim as `gen/` and `build_robotic_task/`.
import os as _os
import sys as _sys
if __package__ in (None, ""):
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import collections
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from build_robotic_task.build_tabletop import (NEAR_EDGE_INSET, OCCLUDER_CLASSES, OCCLUDER_SIZE,
                            TABLE, TABLE_MARGIN, TARGET_CLASSES, TARGET_SIZE,
                            bisect_occluder, catalogue, on_table, stand_back)
from probe_behind import GAP, SPATIAL
from robot.proc_scene import (ROOM, aim_at, open_room, place_on, spawn,
                              surface_top, visible_box, visible_pixels)

#: The band the tabletop CASES are staged to, `(min, target, max)` -- not
#: `probe_behind`'s wide screening window.  With the lateral bisection in place
#: the probe can hit the real band, so it describes the dataset a rebuild would
#: produce rather than a looser proxy for it.
BAND = (0.25, 0.35, 0.50)

#: The two predicates that name a depth order, and which slot ORDER each one
#: should prefer given that the target is behind the occluder.  `True` means the
#: (target, occluder) row should win; `False` means (occluder, target) should.
ORDERED = {"behind": True, "in front of": False}


def stats_binom(k: int, n: int) -> str:
    """Two-sided binomial p against chance, formatted."""
    from scipy import stats as sp

    return f"{sp.binomtest(k, n, 0.5).pvalue:.2e}"


def predict_once(egtr, frame) -> Dict[str, Any]:
    """One forward pass, shared by every measurement taken on this frame.

    `directions` and `twin_scores` both need the same `rel` tensor, and running
    the model twice per view doubled the sweep's cost for nothing.
    """
    from robot.sgg_live import raw_predict

    return raw_predict(egtr, frame)


def slots_for(egtr, frame, *gt_boxes, raw=None) -> Optional[List[int]]:
    """The query index best covering each ground-truth box, or None."""
    from probe_corr import iou

    if any(gt is None for gt in gt_boxes):
        return None
    raw = raw if raw is not None else predict_once(egtr, frame)
    boxes = raw["boxes"].numpy()
    out = []
    for gt in gt_boxes:
        ious = np.array([iou(boxes[q].tolist(), gt) for q in range(len(boxes))])
        if ious.max() < 0.5:
            return None
        out.append(int(ious.argmax()))
    return out + [raw["rel"].detach().cpu().numpy()]


def twin_scores(egtr, frame, gt_target, gt_twin, gt_landmark,
                raw=None) -> Optional[Dict[str, Any]]:
    """
    Target vs its identical twin, both scored against the SAME landmark.

    This is the comparison the tabletop dataset exists for, and it is NOT the
    one `directions` makes: both pairs use the (subject, landmark) slot order,
    so the generic order skew that makes `fwd > rev` about 60% for EVERY
    predicate cancels out.

    It has its own confound, which is why every predicate is returned and not
    just `behind`.  The twin sits at the landmark's depth, nearer the camera, so
    it renders LARGER; if the relation head keys on box size or image position
    the target-vs-twin margin would look like depth discrimination while being
    appearance.  A symmetric predicate cannot carry depth order, so `near`
    scoring as well as `behind` here would settle it the same way it settled the
    slot-order question.
    """
    got = slots_for(egtr, frame, gt_target, gt_twin, gt_landmark, raw=raw)
    if got is None:
        return None
    ti, wi, oi, rel = got
    if len({ti, wi, oi}) < 3:      # one query covering two objects says nothing
        return None
    names = egtr["rel_names"]
    return {n: [float(rel[ti, oi, names.index(n)]),
                float(rel[wi, oi, names.index(n)])] for n in SPATIAL}


def directions(egtr, frame, gt_target, gt_landmark,
               raw=None) -> Optional[Dict[str, Any]]:
    """Both slot orders for every spatial predicate, on one staged frame."""
    got = slots_for(egtr, frame, gt_target, gt_landmark, raw=raw)
    if got is None:
        return None
    ti, oi, rel = got
    names = egtr["rel_names"]
    keep = [names.index(n) for n in SPATIAL]
    fwd, rev = rel[ti, oi], rel[oi, ti]
    # Cosine between the two orders over the spatial block.  1.0 means the head
    # says exactly the same thing about (i, j) and (j, i) -- no order concept.
    a, b = fwd[keep], rev[keep]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    # `probe_behind`'s own quantity, computed here so one staging pass answers
    # both questions: where `behind` ranks, AND whether the head orders the pair
    # at all.  Two probes over the same scenes would cost two THOR sessions.
    order = np.argsort(-a)
    # `behind` with the target as subject and `in front of` with the landmark as
    # subject are the SAME physical fact worded two ways, and each one is only
    # meaningful in its own slot order.  Ranking both is how the converse
    # phrasing gets a fair hearing rather than being scored in the wrong row.
    order_rev = np.argsort(-b)
    return {"fwd": {n: float(fwd[names.index(n)]) for n in SPATIAL},
            "rev": {n: float(rev[names.index(n)]) for n in SPATIAL},
            "cos": float(a @ b / denom) if denom else 0.0,
            "rank": int(np.where(order == SPATIAL.index("behind"))[0][0]) + 1,
            "rank_front_rev": int(np.where(
                order_rev == SPATIAL.index("in front of"))[0][0]) + 1,
            "fwd_argmax": SPATIAL[int(np.argmax(a))],
            "rev_argmax": SPATIAL[int(np.argmax(b))]}


def sweep_views(controller, egtr, event, views: int, lookat: float,
                twin: bool = False) -> List[Dict[str, Any]]:
    """
    The same directional measurement from every synthesised view.

    This is the question the reference frame cannot answer.  R relabels a pair
    by consensus across views, so its ceiling is not what the robot's own frame
    says -- it says `behind` 2.7% of the time -- but whether ANY viewpoint does.

    `lookat` is the orbit radius, and it is NOT `nvs_lemniscate.LOOKAT_DIST`
    here.  That constant is 0.5 m, tuned for a 0.85 m standoff; at 2.20 m it
    orbits a point a quarter of the way to the target and the camera's lateral
    travel is 0.25 m either way, which shifts the target against the landmark by
    less than one target width.  A sweep that barely moves would make "NVS does
    not help" unreadable -- it could equally be "the trajectory did not move" --
    so the radius follows the standoff and the parallax is held comparable
    across geometries.  The centre stays ON THE OPTICAL AXIS, so `camera_for`'s
    identity at az = el = 0 is preserved.
    """
    import math

    from probe_corr import SKIP_VIEWS, seg_box
    from robot.nvs_lemniscate import camera_for, lemniscate, orbit_centre
    from robot.proc_scene import View, look_from

    rc = View(event)
    rc.controller = controller
    camera = rc.camera_xyz.copy()
    centre = orbit_centre(rc, lookat)
    # The robot's own body would otherwise stand in the shot of any view that
    # orbits past it, so park it in the furthest corner.
    corner = min(((0.3, 0.3), (0.3, ROOM - 0.3), (ROOM - 0.3, 0.3),
                  (ROOM - 0.3, ROOM - 0.3)),
                 key=lambda p: -math.dist(p, (camera[0], camera[2])))
    look_from(controller, corner[0], corner[1], 0.0, 0.0, force=True)

    out, first = [], True
    for index, (az, el) in enumerate(lemniscate(views, 30.0, 15.0)):
        if index in SKIP_VIEWS:                 # duplicates of the reference
            continue
        pose = camera_for(centre, camera, az, el)
        controller.step(
            action="AddThirdPartyCamera" if first else "UpdateThirdPartyCamera",
            position=dict(pose["position"]),
            rotation={"x": pose["pitch"], "y": pose["yaw"], "z": 0.0},
            fieldOfView=60.0,
            **({} if first else {"thirdPartyCameraId": 0}))
        first = False
        got = controller.last_event
        seg = got.third_party_instance_segmentation_frames[0]
        colour = {n: c for c, n in got.color_to_object_id.items()}
        frame = np.array(got.third_party_camera_frames[0])
        box = {name: seg_box(seg, colour[name]) if name in colour else None
               for name in ("target", "occluder", "distract")}
        raw = predict_once(egtr, frame)
        measured = directions(egtr, frame, box["target"], box["occluder"],
                              raw=raw)
        if not measured:
            continue
        # The target-vs-twin comparison from THIS view.  Without it the sweep
        # can only say whether the view utters a word, which is the question the
        # ratio rule stops asking -- so pooling across views could not be tested.
        measured["twin"] = (twin_scores(egtr, frame, box["target"],
                                        box["distract"], box["occluder"],
                                        raw=raw)
                            if twin else None)
        out.append({**measured, "az": az, "el": el})
    return out


def camera_z(controller, x: float, tz: float,
             want: Optional[float]) -> Optional[Dict[str, Any]]:
    """
    Where to put the camera, and whether a ROBOT could legally stand there.

    A probe may force the pose -- it only needs a picture -- but a dataset may
    not: `build_tabletop.stand_back` accepts only a standoff THOR permits
    unforced, and that is what makes a case walkable.  A gap that improves the
    signal but only from a forced pose is therefore not a gap the generator can
    be rebuilt around, so the two cases have to be told apart rather than
    silently conflated.  `legal` is that distinction.
    """
    if want is None:
        z = stand_back(controller, x, tz)
        # Larger gaps push the target deeper, and the camera has to clear the
        # table's near edge -- past `STANDOFF`'s 1.15 m ceiling for a big gap.
        if z is not None:
            return {"z": z, "standoff": tz - z, "legal": True}
        for extra in (1.30, 1.45, 1.60, 1.75, 1.90):
            event = controller.step(
                action="Teleport", position={"x": float(x), "y": 0.9,
                                             "z": float(tz - extra)},
                rotation={"x": 0, "y": 0.0, "z": 0}, horizon=0.0,
                standing=True, forceAction=False)
            if event.metadata["lastActionSuccess"]:
                return {"z": tz - extra, "standoff": extra, "legal": True}
        return None
    event = controller.step(action="Teleport",
                            position={"x": float(x), "y": 0.9,
                                      "z": float(tz - want)},
                            rotation={"x": 0, "y": 0.0, "z": 0}, horizon=0.0,
                            standing=True, forceAction=False)
    return {"z": tz - want, "standoff": want,
            "legal": bool(event.metadata["lastActionSuccess"])}


def one(controller, egtr, target_asset: str, occ_asset: str, gap: float,
        standoff: Optional[float], views: int = 0,
        lookat: Optional[float] = None,
        twin: bool = False) -> Optional[Dict[str, Any]]:
    """Stage one pair -- same geometry as `probe_behind.one` -- and measure it."""
    controller.reset()
    centre = ROOM / 2.0
    table = spawn(controller, TABLE, "table", centre, 0.0, centre + 0.6)
    top = surface_top(table)
    box = table["axisAlignedBoundingBox"]
    edge = box["center"]["z"] - box["size"]["z"] / 2.0

    # The occluder sits `gap` in front of the target and must still be on the
    # table, so a bigger gap forces the target deeper.  This is the constraint
    # that couples gap to standoff: the camera then has to clear the near edge.
    tz = edge + max(sum(NEAR_EDGE_INSET) / 2.0, gap + TABLE_MARGIN + 0.05)
    if not on_table(table, centre, tz) or not on_table(table, centre, tz - gap):
        return None
    place_on(controller, target_asset, "target", centre, top, tz)
    where = camera_z(controller, centre, tz, standoff)
    if where is None:
        return None
    event, _ = aim_at(controller, centre, where["z"], 0.0,
                      (centre, top + 0.08, tz))
    clear = visible_pixels(event, "target")
    if clear < 300:
        return None

    # THE OCCLUDER IS BISECTED SIDEWAYS, exactly as `build_tabletop` does it.
    # Placing it dead centre cannot hit the band at a large gap: closer to the
    # camera means bigger in frame, so it buries the target past the ceiling and
    # the pair is discarded -- 2 of 13 staged at gap 0.60 before this.  Lateral
    # offset is the smooth control over COVERAGE, and it is the generator's own
    # mechanism, so using it here keeps the probe predictive of a rebuild.
    _, occ_pos = place_on(controller, occ_asset, "occluder", centre, top,
                          tz - gap)
    staged = bisect_occluder(controller, "occluder", occ_pos["y"],
                             np.array([centre, tz]),
                             np.array([centre, where["z"]]), clear, BAND,
                             along=gap / max(where["standoff"], 1e-6))
    if staged is None:
        return None
    controller.step(action="TeleportObject", objectId="occluder",
                    position=staged["position"],
                    rotation={"x": 0, "y": 0, "z": 0}, forceAction=True)
    event = controller.step(action="Pass")
    if not on_table(table, staged["position"]["x"], staged["position"]["z"]):
        return None
    hidden = 1.0 - visible_pixels(event, "target") / clear
    if not BAND[0] <= hidden <= BAND[2]:
        return None

    # The twin goes at the LANDMARK's depth, one lateral step from the target --
    # `build_tabletop`'s own rule, so that exactly one copy is behind the
    # landmark rather than both.
    pair = None
    if twin:
        staged_x = staged["position"]["x"]
        side = -1.0 if staged_x >= centre else 1.0
        wx = centre + side * 0.35
        if not on_table(table, wx, staged["position"]["z"]):
            return None
        place_on(controller, target_asset, "distract", wx, top,
                 staged["position"]["z"])
        event = controller.step(action="Pass")
        if visible_pixels(event, "distract") < 300:
            return None
        # The twin is nearer the camera than the target, so it can hide it; the
        # bisection measured the target before the twin existed.
        hidden = 1.0 - visible_pixels(event, "target") / clear
        if not BAND[0] <= hidden <= BAND[2]:
            return None
        pair = twin_scores(egtr, event.frame, visible_box(event, "target"),
                           visible_box(event, "distract"),
                           visible_box(event, "occluder"))
        if pair is None:
            return None

    got = directions(egtr, event.frame, visible_box(event, "target"),
                     visible_box(event, "occluder"))
    if got is None:
        return None
    got["twin"] = pair
    px = visible_pixels(event, "target")
    frame = event.frame.shape[0] * event.frame.shape[1]
    swept = (sweep_views(controller, egtr, event, views,
                         lookat if lookat else where["standoff"], twin)
             if views else [])
    # Depth RATIO, not just the gap.  Widening the gap at a fixed standoff
    # raises it; backing the camera off at a fixed gap lowers it.  The two
    # hypotheses about what the predicate head responds to disagree on sign,
    # so the ratio has to be reported alongside the metres.
    return {**got, "hidden": hidden, "gap": gap, "clear_px": clear,
            "target_px": px, "target_frac": px / frame, "views": swept,
            "standoff": where["standoff"], "legal": where["legal"],
            "depth_ratio": where["standoff"] / max(where["standoff"] - gap, 1e-6)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--assets", type=int, default=3,
                    help="asset combinations per class pair.  Classes with "
                         "only one asset in the size band are staged once, not "
                         "`--assets` times: the geometry is deterministic, so "
                         "repeats are identical and would inflate n.")
    ap.add_argument("--out", help="write the per-pair rows as JSON.  The "
                                  "printed table rounds to four places and the "
                                  "`behind` field is smaller than that, so the "
                                  "log cannot be re-analysed without this.")
    ap.add_argument("--gap", type=float, nargs="+", default=[GAP],
                    metavar="M",
                    help="depth separation between target and landmark, metres. "
                         "Several values sweep it.  Capped by the table: the "
                         "landmark must stay on it, so the target moves deeper "
                         "and the camera has to follow.")
    ap.add_argument("--standoff", type=float, nargs="+", default=[None],
                    metavar="M",
                    help="force the camera this far from the target instead of "
                         "taking the nearest legal `STANDOFF`.  Sweeping this "
                         "with --gap fixed isolates the camera distance from "
                         "the depth ratio, which the two rival explanations of "
                         "the predicate head disagree about.")
    ap.add_argument("--twin", action="store_true",
                    help="also stage the identical twin at the landmark's "
                         "depth and score target-vs-twin for every predicate.  "
                         "This is the comparison free of the slot-order skew.")
    ap.add_argument("--views", type=int, default=0, metavar="N",
                    help="lemniscate views to repeat the measurement from.  0 "
                         "measures the reference frame only.")
    ap.add_argument("--lookat", type=float, default=None, metavar="M",
                    help="orbit radius.  Defaults to the camera-target "
                         "distance, NOT nvs_lemniscate.LOOKAT_DIST -- see "
                         "`sweep_views`.")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    args = ap.parse_args(argv)

    from robot.sgg_live import load_egtr

    egtr = load_egtr()
    controller = open_room(args.width, args.height, args.fov)
    grid = [(g, s) for g in args.gap for s in args.standoff]
    rows: List[Dict[str, Any]] = []
    try:
        targets = catalogue(controller, TARGET_CLASSES, TARGET_SIZE)
        occluders = catalogue(controller, OCCLUDER_CLASSES, OCCLUDER_SIZE)
        for gap, standoff in grid:
            print(f"\n== gap {gap:.2f} m, standoff "
                  f"{'auto' if standoff is None else f'{standoff:.2f} m'}",
                  flush=True)
            for subject in sorted(targets):
                for landmark in sorted(occluders):
                    if subject == landmark:
                        continue
                    ta_all = sorted(targets[subject])
                    oa_all = sorted(occluders[landmark])
                    tries = min(args.assets, max(len(ta_all), len(oa_all)))
                    for k in range(tries):
                        try:
                            got = one(controller, egtr, ta_all[k % len(ta_all)],
                                      oa_all[k % len(oa_all)], gap, standoff,
                                      args.views, args.lookat, args.twin)
                        except Exception as error:              # noqa: BLE001
                            print(f"  ! {type(error).__name__}: {error}",
                                  flush=True)
                            continue
                        if got is None:
                            continue
                        rows.append({**got,
                                     "pair": f"{subject} behind {landmark}",
                                     "config": (gap, standoff)})
                        swept = got["views"]
                        print(f"  {rows[-1]['pair']:30s} rank {got['rank']}/6  "
                              f"fwd {got['fwd_argmax']:<12s} "
                              f"rev {got['rev_argmax']:<12s} "
                              f"hid {got['hidden']:.2f} "
                              f"px {got['target_frac']:.3%} "
                              f"ratio {got['depth_ratio']:.2f} "
                              f"{'legal' if got['legal'] else 'FORCED'}"
                              + (f"   views {sum(1 for v in swept if v['rank'] == 1)}"
                                 f"/{len(swept)} 1st, best rank "
                                 f"{min((v['rank'] for v in swept), default='-')}"
                                 if swept else ""), flush=True)
    finally:
        controller.stop()

    if not rows:
        print("nothing staged")
        return 1

    if args.out:
        import json
        with open(args.out, "w") as handle:
            json.dump({"band": list(BAND), "spatial": list(SPATIAL),
                       "grid": [[g, s] for g, s in grid],
                       "rows": [{**r, "config": list(r["config"])}
                                for r in rows]}, handle, indent=1)
        print(f"  wrote {args.out}")

    if len(grid) > 1:
        print(f"\n  {'gap':>5} {'stand':>6} {'n':>3} {'behind 1st':>11} "
              f"{'dir agrees':>11} {'ratio':>6} {'target px':>10} {'legal':>6}")
        for gap, standoff in grid:
            got = [r for r in rows if r["config"] == (gap, standoff)]
            if not got:
                print(f"  {gap:>5.2f} "
                      f"{'auto' if standoff is None else f'{standoff:>6.2f}'} "
                      f"  0        nothing staged")
                continue
            m = len(got)
            first = sum(1 for r in got if r["rank"] == 1)
            agree = sum(1 for r in got
                        if r["fwd"]["behind"] > r["rev"]["behind"])
            print(f"  {gap:>5.2f} "
                  f"{'  auto' if standoff is None else f'{standoff:>6.2f}'} "
                  f"{m:>3d} {first:>4d}/{m:<6d} {agree:>4d}/{m:<6d} "
                  f"{np.median([r['depth_ratio'] for r in got]):>6.2f} "
                  f"{np.median([r['target_frac'] for r in got]):>9.3%} "
                  f"{sum(1 for r in got if r['legal']):>3d}/{m:<3d}")
        print("\n  `dir agrees` is rel[target,landmark,behind] > the reversed "
              "slot order:\n  the depth-order signal, free of any need for the "
              "model to say the word.")

    n = len(rows)
    print(f"\n  {n} staged pairs, target genuinely behind occluder\n")

    # The screen `probe_behind.py` exists for, at no extra staging cost.
    print(f"  {'instruction':30s} {'n':>3s} {'behind 1st':>11s} {'median rank':>12s}")
    by_pair: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_pair[row["pair"]].append(row)
    for pair, got in sorted(by_pair.items(),
                            key=lambda kv: (-sum(1 for g in kv[1]
                                                 if g["rank"] == 1),
                                            np.median([g["rank"]
                                                       for g in kv[1]]))):
        first = sum(1 for g in got if g["rank"] == 1)
        print(f"  {pair:30s} {len(got):>3d} {first:>7d}/{len(got):<3d} "
              f"{np.median([g['rank'] for g in got]):>12.0f}")
    print(f"\n  `behind` is the top spatial predicate in "
          f"{sum(1 for r in rows if r['rank'] == 1)} of {n} reference frames\n")
    print(f"  {'predicate':14s} {'correct order wins':>18s}   what that means")
    for pred, fwd_should_win in ORDERED.items():
        agree = sum(1 for r in rows
                    if (r["fwd"][pred] > r["rev"][pred]) == fwd_should_win)
        order = "(target,landmark)" if fwd_should_win else "(landmark,target)"
        print(f"  {pred:14s} {agree:>8d}/{n:<9d}   {order} should carry more mass")

    swept = [v for r in rows for v in r["views"]]
    if swept:
        print(f"\n  ADDING NOVEL VIEWS -- {len(swept)} views over {n} scenes\n")
        print(f"  {'measurement':44s} {'reference':>12s} {'novel views':>12s}")

        def line(label, test, ref_rows=rows, view_rows=swept):
            a = sum(1 for r in ref_rows if test(r))
            b = sum(1 for v in view_rows if test(v))
            print(f"  {label:44s} {a:>4d}/{len(ref_rows):<3d}={a/len(ref_rows):>3.0%}"
                  f" {b:>5d}/{len(view_rows):<4d}={b/len(view_rows):>3.0%}")

        line("`behind` is argmax of rel[target, landmark]",
             lambda r: r["rank"] == 1)
        line("`in front of` is argmax of rel[landmark, target]",
             lambda r: r["rank_front_rev"] == 1)
        line("depth order right: behind fwd > rev",
             lambda r: r["fwd"]["behind"] > r["rev"]["behind"])
        line("depth order right: in front of rev > fwd",
             lambda r: r["rev"]["in front of"] > r["fwd"]["in front of"])

        # R's actual ceiling: it relabels by cross-view consensus, so what
        # bounds it is whether the word is reachable from ANY viewpoint at all.
        any_first = sum(1 for r in rows
                        if any(v["rank"] == 1 for v in r["views"]))
        any_front = sum(1 for r in rows
                        if any(v["rank_front_rev"] == 1 for v in r["views"]))
        agree_majority = sum(
            1 for r in rows if r["views"] and sum(
                1 for v in r["views"]
                if v["fwd"]["behind"] > v["rev"]["behind"]) > len(r["views"]) / 2)
        print(f"\n  scenes where SOME view makes `behind` the argmax:      "
              f"{any_first:>3d}/{n}  ({any_first / n:.0%})")
        print(f"  scenes where SOME view makes `in front of` the argmax: "
              f"{any_front:>3d}/{n}  ({any_front / n:.0%})")
        print(f"  scenes where a MAJORITY of views get the depth order:  "
              f"{agree_majority:>3d}/{n}  ({agree_majority / n:.0%})")
        print("\n  The first line is the ceiling on channel R: it relabels by "
              "consensus,\n  so a word no viewpoint ever utters is one no amount "
              "of agreement recovers.")

    paired = [r["twin"] for r in rows if r.get("twin")]
    if paired:
        print(f"\n  TARGET vs ITS IDENTICAL TWIN, both against the same landmark")
        print(f"  {len(paired)} scenes.  Exactly one copy is really behind it.\n")
        print(f"  {'predicate':14s} {'target wins':>13s} {'p vs 50%':>10s}   "
              f"carries depth order?")
        for pred in SPATIAL:
            k = sum(1 for t in paired if t[pred][0] > t[pred][1])
            print(f"  {pred:14s} {k:>5d}/{len(paired):<4d}="
                  f"{k / len(paired):>3.0%} "
                  f"{stats_binom(k, len(paired)):>10} "
                  f"  {'YES, if above the others' if pred == 'behind' else 'no -- control'}")
        print("\n  `near` is symmetric: it cannot distinguish the two copies by "
              "depth.\n  If it scores like `behind`, the margin is appearance, "
              "not depth order.")

    cos = np.array([r["cos"] for r in rows])
    print(f"\n  cos(fwd, rev) over the six spatial predicates: "
          f"median {np.median(cos):.3f}, min {cos.min():.3f}, max {cos.max():.3f}")
    print("  1.0 would mean the head says the same thing about both slot orders")
    same = sum(1 for r in rows if r["fwd_argmax"] == r["rev_argmax"])
    print(f"  same argmax in both orders: {same}/{n}")

    print(f"\n  argmax by slot order:")
    for key, label in (("fwd_argmax", "(target, landmark)"),
                       ("rev_argmax", "(landmark, target)")):
        count = collections.Counter(r[key] for r in rows)
        print(f"    {label:20s} " + "  ".join(
            f"{k} {v}" for k, v in count.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
