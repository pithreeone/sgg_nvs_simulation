"""
build_slot.py -- scenes where the target is only visible through a slot.

`build_tabletop.py` hides the target by a fixed fraction FROM THE REFERENCE
VIEW and says nothing about any other view.  At 35% hidden the target is partly
visible from everywhere in the sweep, so moving the camera buys a better look
rather than the only look, and "which viewpoint" has no sharp right answer.
Here it does: two occluders stand side by side with a gap between them, and the
target behind is revealed only while the line of sight threads that gap.

              [laptop]                    the target, the instruction's A
                 |  gap
   [blocker]     |     [box]              both at the target's depth minus `gap`
        \        |       /
         `--.    |   .--'                 the slot, and the only way in
             \   |  /
              [camera]

THE TWO FRONT OBJECTS ARE NOT INTERCHANGEABLE.

    occluder  THE LANDMARK -- the B of `Find the {A} behind the {B}`, and the
              only one of the three the instruction names.  Bisected sideways
              until the reference view is `--ref-occlusion` hidden, exactly as
              `build_tabletop` does it, so this list is comparable with
              `cases_easy2` at the start pose and diverges only off-axis.
    blocker   a SECOND occluder, of a different class from both the target and
              the landmark, out on the side the landmark reveals.  It is never
              named.  Its whole job is to make the view worse again further
              out, so the best view is INTERIOR to the sweep.

THE PROPERTY EVERY CASE HAS.  Not a keyhole -- that was the first draft and it
rejected nearly everything -- but this, which is weaker and is what the list is
actually for:

    the best view is more than `--interior` degrees inside the sweep's rim, and
    BOTH ends of the reachable sweep are at least `--falloff` more hidden than
    it.

In other words `walk as far as the sweep allows and look` is a losing policy on
every case here, which is the one thing `cases_easy2` cannot say: there, more
azimuth is monotonically no worse.

WHY THE ANGLES ARE SMALL, AND WHY THE WINDOW STILL FITS.  Everything downstream
is capped at 30 degrees of azimuth -- `fuse_live --max-az`, `probe_viewdist`,
and the bearings `eval_move` samples -- and a ray to the target crosses the
occluder plane at `x_target - gap * tan(azimuth)`, so the whole reachable sweep
moves it by only +-0.29 m at `gap` 0.50.  That is less than a `box` is wide, so
the target never fully clears the landmark inside the sweep (it would take 32
degrees).  It does not have to.  What is measured is silhouette OVERLAP, and the
target's own 0.38 m projects to 0.199 m at the occluder plane, so from the 50%
the landmark is bisected to:

    7 degrees off axis   ->  20% hidden, which is `--open-max`: the window opens
    the other direction  ->  100% hidden, the landmark being wider than the
                             sweep displacement plus the silhouette

A case whose window falls outside +-30 is unwinnable by any arm, so `one_case`
rejects it rather than shipping a ceiling of zero.

THE CURVE IS MEASURED, AND IT IS A COLUMN.  Every case carries `sweep`: the
target's visible pixels at each azimuth, with the occluders and without, plus
whether THOR would let the robot stand there unforced.  Nothing at evaluation
time may read it to choose views or to drop cases -- that is the mistake
`PIPELINE.md` section 10 costs out.  It is here to say what the ceiling was.

WHAT THIS LIST DOES NOT TEST.  There is no second copy of the target, so while
the instruction states a relation it does not NEED one: "the cup behind the
laptop" and "any cup" have the same answer here, which is the property
`build_tabletop`'s distractor exists to remove.  Deliberate -- the question on
this list is whether the object can be found at all, from a viewpoint that has
to be chosen -- but it means a number measured here says nothing about
relational grounding and must not be pooled with `cases_easy2`.

    python build_slot.py --n 40 --seed 1 --out nvs_pilot/cases/cases_slot.json
"""

from __future__ import annotations

# `python build_robotic_task/<script>.py` puts this directory on sys.path, not
# the repo root, so `robot.*`, `vg.*` and the sibling generators would not
# resolve.  Running as `python -m build_robotic_task.<script>` does not need
# this; it is here so both work.  Same shim as `gen/build_occlusion_dataset.py`.
import os as _os
import sys as _sys
if __package__ in (None, ""):
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from build_robotic_task.build_tabletop import (MIN_PIXELS, TABLE, TABLE_MARGIN, asset_db,
                            occluder_yaw, on_table, reject, settle)
from robot.proc_scene import (ROOM, aim_at, look_from, open_room, place_on,
                              spawn, surface_top, visible_box,
                              visible_pixels)
from vg.vg150 import THOR_TO_VG150

#: THE THREE ROLES ARE DRAWN SEPARATELY, on different criteria, because they are
#: asked for different things.  `build_tabletop` has one target pool and one
#: occluder pool and that was right when both front objects were named; here
#: only two of the three ever reach the instruction.
#:
#:   target    must be NAMEABLE and must fit behind the landmark.  SEVERAL
#:             CLASSES, and the band is wide, because nameability is no longer
#:             argued from the class name -- `--nameable-topk` measures it per
#:             case at the best view and throws the case away if the detector
#:             cannot say the word.  `laptop` alone is what the first build used,
#:             on the reasoning that it scores 1/1 on a landmark screen; the
#:             screen it actually needed was a TARGET one, and THOR laptops fail
#:             it, the lid facing the camera being an unlit black slab.
#:   landmark  must be NAMEABLE, since it is the B of `A behind B`, and wide and
#:             tall enough to hide the target.  `box` is the best-named class the
#:             repo has measured -- 8/8 where `lamp` is 0/18 and `clock` 0/5.
#:
#:             THE HEIGHT CEILING IS WHAT KEEPS IT A BOX.  VG150 `box` is fed by
#:             three THOR types -- `Box`, `TissueBox` and `GarbageCan` -- and the
#:             bins are the tall ones, 0.46 to 0.53 m, while real boxes sit at
#:             0.24 to 0.37.  A floor of 0.30, set so the landmark would out-top
#:             the target, cut 10 real boxes and kept 23 bins: 74% of the pool,
#:             and the first four cases drawn were all garbage cans rendering at
#:             18-28% OF THE FRAME.  That is `build_tabletop`'s HousePlant
#:             failure exactly -- a landmark whose own box swallows the picture
#:             makes `behind the box` satisfiable by almost anything.
#:
#:             0.24-0.40 inverts it: 17 real boxes against 6 bins.  Out-topping
#:             the target is then checked PER CASE against the asset actually
#:             drawn, which is where it belonged -- a band cannot express "taller
#:             than the other thing in this scene".
#:   blocker   NEVER NAMED, so how well EGTR reads it does not matter at all.
#:             Its requirements are purely geometric: wide enough to shut the
#:             window (a 0.10 m bottle cannot), tall enough to cover the target,
#:             and of neither named class -- a second `box` would make "behind
#:             the box" true of two objects.
#:
#: Widths carry a FLOOR as well as a ceiling, which `build_tabletop.catalogue`
#: has no way to say, and that floor is the whole reason `pool` is local.
ROLES: Dict[str, Dict[str, Any]] = {
    # THE WIDTH FLOOR IS WHAT KEEPS THE LANDMARK FROM SWALLOWING THE PICTURE.
    # Dropping it to 0.06 to admit cups and bottles put 9 of 18 attempts into
    # `landmark is 56.9x the target`: the box has to out-top and out-width the
    # target to hide it, so a small target forces a landmark that fills a quarter
    # of the frame, which is the `HousePlant` failure again.  0.18 admits the
    # larger vases, bowls and plants without that.
    "target": {"classes": ("laptop", "vase", "bowl", "plant"),
               "height": (0.14, 0.30), "width": (0.18, 0.50)},
    "landmark": {"classes": ("box",),
                 "height": (0.24, 0.40), "width": (0.30, 0.70)},
    "blocker": {"classes": ("plant", "lamp"),
                "height": (0.30, 0.55), "width": (0.25, 0.55)},
}

#: Whether the robot STANDS or CROUCHES to look.  Not a body height: an
#: unforced Teleport succeeds at y=0.9 and at no other value, so the only lever
#: on camera height is the stance -- standing puts the camera at 1.575 m, 0.74 m
#: over the table and pitched 27 degrees down; crouched puts it at 0.900 m, 0.07 m
#: over it and level.
#:
#: CROUCHED, BECAUSE A STEEP CAMERA GOES OVER THE TOP OF A TABLETOP OCCLUDER.
#: The ray to a target `gap` behind an occluder is already `gap * tan(pitch)`
#: above the table when it crosses the occluder plane: standing at `gap` 0.65
#: that is 0.33 m, so an occluder needs 0.41 m of height to intercept it, real
#: `Box` assets stop at 0.37 and `GarbageCan` runs to 0.54.  That is why the
#: first pools came out all garbage cans, and it is not a sampling problem -- the
#: bins were the only assets that could work.
#:
#: Measured on one scene, `Box_21` in front of `Laptop_2` at `gap` 0.65: the box
#: hid 3% of the laptop standing and 50% crouched.  Standing, the best view was
#: at +30 degrees -- the rim of the sweep, which is the policy this whole list
#: exists to make wrong.
#:
#: RECORDED IN THE CASE as `standing`, and `proc_scene.rebuild` reads it; `Robot`
#: infers it from the pose so the motion loop cannot stand up mid-episode.
STANDING = False

#: Separation between the target and the occluder plane, metres.  THIS IS THE
#: ONLY LEVER ON HOW FAR APART THE TWO FRONT OBJECTS CAN STAND, and it is worth
#: being explicit about why, because the instinct is that they are jammed
#: together for no reason.
#:
#: The sightline crosses the occluder plane at `x_target - gap * tan(azimuth)`,
#: so over the reachable +-30 degrees it moves +-`gap * 0.577` and no further.
#: The blocker's inner edge has to sit INSIDE that span or it never bites, and a
#: blocker that never bites is exactly the case where walking to the far edge of
#: the sweep is the right answer -- the thing this list exists to rule out.  So
#: the opening between the two is at most `gap * tan(CLOSE_AT[1])`:
#:
#:     gap    sweep spans    opening      centre to centre    target inset
#:     0.50    +-0.29 m       0.27 m         0.69 m             0.80 m
#:     0.65    +-0.38 m       0.35 m         0.77 m             0.95 m
#:     0.75    +-0.43 m       0.40 m         0.82 m             1.05 m
#:
#: 0.45, AND THE REASON IT CAME BACK DOWN FROM 0.65 IS THE CROUCH.  0.65 was set
#: when the camera stood: a standing camera pitches 27 degrees down, the sightline
#: is `gap * 0.63` above the table at the occluder plane, and only a 0.41 m
#: occluder reaches it -- so the gap had to be large enough to separate the two
#: occluders in a design where nothing but a garbage can could occlude at all.
#: Crouched the sightline is level and the height requirement collapses to half
#: the target's own, which frees the gap to be chosen on the geometry that
#: actually needs it.
#:
#: BUT IT CANNOT BE LARGE EITHER, because `gap * tan(reach)` is both the opening
#: between the two occluders AND how far the sightline sweeps across their plane
#: -- the same quantity, pulling both ways.  The landmark's inner edge is pinned
#: by the 50% bisection, so covering its own side takes a width of
#: `2 * gap * tan(reach)`, and THOR's widest box is 0.68 m:
#:
#:     gap    opening    width needed    boxes that wide
#:     0.45    0.26 m      0.52 m              3
#:     0.55    0.32 m      0.64 m              0
#:     0.65    0.38 m      0.75 m              0
#:
#: At 0.65 that showed up as leaks at both rims -- 18% and 20% hidden on the
#: landmark side, the sightline having passed its far edge -- plus three attempts
#: putting the blocker 0.58 m out where the table has 0.485 m.
#:
#: 0.55 IS A CHOSEN TRADE, NOT AN OPTIMUM.  Measured against 0.45 over six cases
#: each, both at `TARGET_TURN` 45:
#:
#:                          gap 0.45     gap 0.55
#:     opening               0.26 m       0.32 m
#:     window width, median  15.0 deg     17.5 deg
#:     best view hidden      3-13%        0-4%
#:     WORST RIM             68%          46%
#:     rims under 60%        0 of 6       3 of 6
#:     accepted              6 of 10      6 of 15
#:
#: The wider opening and the cleaner best view were worth it; the cost is real and
#: is that on half the cases the rim of the sweep still shows half the target, so
#: "walk to the end" is a weak mistake there rather than a clear one.  `--gap
#: 0.45` is the other side of it and nothing else needs changing to take it.
GAP = 0.55

#: How far the camera stands from the target, metres, nearest-first.  Longer
#: than `build_tabletop.STANDOFF` because `GAP` pushes the target onto the far
#: half of the table while the agent capsule still has to clear the near edge:
#: at `gap` 0.50 the target sits ~0.85 m in from that edge and nothing under
#: about 1.05 m is legal.
STANDOFF = (1.05, 1.15, 1.30, 1.45, 1.60)

#: Azimuths swept, degrees.  Past +-30 ON PURPOSE: the sweep has to show that
#: the window CLOSES inside the reachable range, and a curve stopping at the
#: last reachable pose cannot tell "shut" from "ran out of arc".
SWEEP = (-32.5, 32.5, 2.5)

#: How much taller than the target an occluder must be, metres.  Not zero: equal
#: heights leave the target's top edge peeping over, which reads as a partial
#: occlusion the bisection then has to chase with lateral offset alone.
TALLER_BY = 0.03

#: An extra turn on the TARGET only, degrees, on top of what `occluder_yaw`
#: gives it.  That already adds `build_tabletop.CLASS_YAW`'s 180 to bring a
#: laptop's screen round to face the camera; this puts it at three quarters
#: instead of flat on.
#:
#: Local to this file and NOT a change to `CLASS_YAW`, which the laptop shares as
#: an OCCLUDER in `cases_easy2` and `cases_hard` -- turning it there would
#: restage lists that are already measured.
TARGET_TURN = 45.0

#: The blocker is bisected to shut the view AT THE SWEEP EDGE, not at an angle
#: drawn per case.  This constant is gone; the angle is `--reach`.
#:
#: It used to be drawn from 20-28 degrees and the rim was then merely hoped to be
#: shut too.  It was not: a desk lamp covers about 8 degrees of arc, so a blocker
#: set at 22 shuts 18-26 and leaves 26-30 open.  Two of the first six cases came
#: out with a rim at 16% and 28% hidden -- 84% and 72% of the target VISIBLE from
#: the edge of the sweep, which is the policy this list exists to make wrong, and
#: they passed `--falloff` only because the best view happened to be 0%: an
#: absolute difference is a weak test when the best is already perfect.
#:
#: Bisecting at the rim makes it shut by construction.  Where the window CLOSES
#: is then measured off the blocker's own width instead of chosen, which is the
#: diversity that was worth having.


#: EGTR, loaded once and only if the nameability screen is on.  A build spawns
#: and stops a controller per attempt; reloading the detector with it would cost
#: more than the screen saves.
_EGTR: Dict[str, Any] = {}


def nameable_egtr():
    """The detector the screen judges with -- the same one every runner uses."""
    if not _EGTR:
        from robot.sgg_live import load_egtr

        _EGTR["it"] = load_egtr()
    return _EGTR["it"]


def solved_at(event, egtr, task: Dict[str, Any], condition: int,
              gate: float) -> bool:
    """`fuse_live.decide`'s verdict on one frame: is the top-1 pair the right one?

    THE SCREEN JUDGES BY THE METRIC, NOT BY A PROXY FOR IT.  The version before
    this one asked whether the detector slot covering the target argmaxed to the
    instructed class, which is a different question and a much weaker one: over
    the 40 cases that passed it, the class was right at the best view by
    construction 40/40 and the top-1 pair was right at that same angle 17/40.
    The gap is everything between naming ONE slot and winning a ranking of ~100
    PAIRS -- entering the top-10 by class mass, carrying `rel[i, j, behind]`, and
    outscoring every pair the landmark forms with the rest of the table.

    So the score, the candidate set and the IoU gate here are `eval_move.look`'s,
    imported rather than restated wherever that is possible.
    """
    from fuse_live import conditioned
    from robot.sgg_live import raw_predict
    from robot.task_find import iou

    target = visible_box(event, task["target_name"])
    landmark = visible_box(event, task["receptacle_name"])
    if not target or not landmark:
        return False
    raw = raw_predict(egtr, event.frame)
    built = {"probs_ref": raw["probs_softmax"].detach().cpu().numpy(),
             "rel": raw["rel"].detach().cpu(),
             "boxes": raw["boxes"].detach().cpu(),
             "s": raw["probs_softmax"].detach().cpu().max(-1).values}
    cand = conditioned(built, egtr, task, condition, 0.0)
    if cand is None:
        return False
    rel, s = built["rel"], built["s"].float()
    subjects, objects = cand
    predicate = egtr["rel_names"].index(task["predicate"])
    pairs = [(float(rel[i, j, predicate]) * float(s[i]) * float(s[j]), i, j)
             for i in subjects for j in objects if i != j]
    if not pairs:
        return False
    _, i, j = max(pairs)
    return (iou(built["boxes"][i].tolist(), target) >= gate
            and iou(built["boxes"][j].tolist(), landmark) >= gate)


def fan_out(args, argv: Optional[Sequence[str]]) -> int:
    """Build in `--workers` processes and merge what they write.

    THE SEED IS THE ONLY THING THAT DIFFERS, so a worker is just this script
    with one flag changed and a scratch `--out`; there is no shared state to
    race on and nothing to coordinate.  Each asks for its share of `--n` and
    the merge truncates, so a worker that runs dry does not hold the others up.

    A WORKER THAT DIES IS REPORTED AND NOT FATAL.  Unity exits non-zero on
    display trouble often enough that failing the whole build on one worker
    would make the flag useless; the merge says how many cases each produced.
    """
    import subprocess
    import tempfile
    import time

    argv = list(argv if argv is not None else _sys.argv[1:])
    share = -(-args.n // args.workers)
    stem = os.path.splitext(os.path.basename(args.out))[0]
    scratch = tempfile.mkdtemp(prefix=f"{stem}.workers.")
    running, parts = [], []
    for worker in range(args.workers):
        keep = [a for a in argv]
        for flag in ("--workers", "--stagger", "--seed", "--n", "--out"):
            while flag in keep:
                at = keep.index(flag)
                del keep[at:at + 2]
        part = os.path.join(scratch, f"{worker}.json")
        parts.append(part)
        command = [_sys.executable, os.path.abspath(__file__), *keep,
                   "--seed", str(args.seed + worker), "--n", str(share),
                   "--out", part]
        log = open(os.path.join(scratch, f"{worker}.log"), "w")
        running.append((worker, subprocess.Popen(command, stdout=log,
                                                 stderr=subprocess.STDOUT),
                        log))
        print(f"  worker {worker} seed {args.seed + worker} -> {part}",
              flush=True)
        if worker + 1 < args.workers:
            time.sleep(args.stagger)
    for worker, process, log in running:
        code = process.wait()
        log.close()
        if code:
            print(f"  ! worker {worker} exited {code}, see "
                  f"{os.path.join(scratch, f'{worker}.log')}", flush=True)

    cases, header = [], None
    for worker, part in enumerate(parts):
        if not os.path.exists(part):
            print(f"  ! worker {worker} wrote nothing", flush=True)
            continue
        got = json.load(open(part))
        header = header or {k: v for k, v in got.items() if k != "cases"}
        print(f"  worker {worker}: {len(got['cases'])} cases", flush=True)
        cases.extend(got["cases"])
    if not cases:
        print("no cases built")
        return 1
    # `scene` is `slot|<attempt index>` and two workers index independently, so
    # the merged list would carry the same name twice.  The seed is what makes
    # them different scenes, so it is what disambiguates them.
    seen: Dict[str, int] = {}
    for case in cases:
        seen[case["scene"]] = seen.get(case["scene"], 0) + 1
        if seen[case["scene"]] > 1:
            case["scene"] = f"{case['scene']}.{seen[case['scene']] - 1}"
    cases = cases[:args.n]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({**(header or {}), "cases": cases}, open(args.out, "w"), indent=1)
    print(f"\n{len(cases)} cases from {args.workers} workers -> {args.out}")
    print(f"  worker logs and parts kept in {scratch}")
    return 0


def longest_run(angles: Sequence[float], step: float
                ) -> Optional[Tuple[float, float]]:
    """The widest contiguous stretch of `angles`, which arrive sorted."""
    if not angles:
        return None
    best = run = (angles[0], angles[0])
    for az in angles[1:]:
        run = (run[0], az) if az - run[1] <= step + 1e-6 else (az, az)
        if run[1] - run[0] > best[1] - best[0]:
            best = run
    return best


def extent(entry: Dict[str, Any]) -> Tuple[float, float]:
    """An asset's (height, widest horizontal extent) from the database."""
    box = entry["boundingBox"]
    return (box["max"]["y"] - box["min"]["y"],
            max(box["max"]["x"] - box["min"]["x"],
                box["max"]["z"] - box["min"]["z"]))


def pool(controller, classes: Sequence[str], height: Tuple[float, float],
         width: Tuple[float, float],
         types: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """Asset ids per VG150 class inside a height AND width band.

    `build_tabletop.catalogue` caps width only, which is all a landmark needs.
    The blocker needs a FLOOR too: its job is to shut the window, and the
    occluder pool is full of 0.10 m bottles that never will.

    `types` filters on the THOR objectType underneath the VG150 class, for the
    case where one class is fed by several and they are not interchangeable:
    `box` is `Box`, `TissueBox` and `GarbageCan` together, and `--landmark-types
    Box` is the one-flag way to say a bin is not what the instruction means.
    """
    import collections

    out: Dict[str, List[str]] = collections.defaultdict(list)
    for asset, entry in asset_db(controller).items():
        if entry.get("primaryProperty") == "Static":
            continue
        if types and entry["objectType"] not in types:
            continue
        tall, wide = extent(entry)
        vg = THOR_TO_VG150.get(entry["objectType"])
        if (vg in classes and height[0] <= tall <= height[1]
                and width[0] <= wide <= width[1]):
            out[vg].append(asset)
    return {k: sorted(v) for k, v in out.items()}


def azimuths(spec: Sequence[float]) -> List[float]:
    """The swept angles, both ends included."""
    lo, hi, step = spec
    return [round(lo + i * step, 3)
            for i in range(int(round((hi - lo) / step)) + 1)]


def look_along(controller, target_xz, top: float, radius: float,
               azimuth: float, standing: bool = STANDING,
               horizon: Optional[float] = None,
               reach: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """Put the camera on the arc about the target at `azimuth`, aimed at it.

    `reachable` is asked BEFORE the forced placement and recorded rather than
    acted on.  A pose THOR refuses is something the robot can observe, so it
    belongs in the record; refusing to RENDER there would let the ground truth
    decide which angles exist, which is the hazard `PIPELINE.md` section 10
    prices.  `reach=False` skips the question when the answer is already known
    for this pose, which is most of them -- see `sweep_target`.

    `horizon` SKIPS `aim_at`, AND THAT IS TWO THIRDS OF THE COST.  `aim_at`
    teleports twice: once to read `cameraPosition`, then again to pitch.  But
    every pose on this arc is the same distance from the same aim point at the
    same camera height, so the pitch is IDENTICAL at every azimuth -- computing
    it once and passing it back turns three teleports per angle into one.  At
    two 25-angle sweeps per attempt and twenty attempts per accepted case that
    was the dominant cost of a build.
    """
    # AZIMUTH IS SIGNED AS `nvs_lemniscate.arc_step` SIGNS IT, which is what
    # `eval_move.walk` executes and what every sweep downstream is labelled in:
    # the camera starts behind the target, so its offset is (0, -r), and adding
    # `azimuth` to `atan2(0, -r)` puts it at `x - r sin`.  This file had the
    # opposite sign until 2026-08-15, so a case's `visible_arc` named the mirror
    # of the arc a robot walking that number would reach -- silently, since both
    # ends of a symmetric sweep exist.  Two ladder cases caught it: the rank-1
    # angles landed exactly on the negation of the recorded window.
    #
    # `close_at` and the blocker's inner bound flip with it, and those two
    # cancel: the WORLD geometry is unchanged and only the labels move.
    theta = math.radians(azimuth)
    x = float(target_xz[0] - radius * math.sin(theta))
    z = float(target_xz[1] - radius * math.cos(theta))
    yaw = float(math.degrees(math.atan2(target_xz[0] - x, target_xz[1] - z)))
    reachable = controller.step(
        action="Teleport", position={"x": x, "y": 0.9, "z": z},
        rotation={"x": 0, "y": yaw, "z": 0}, horizon=0.0,
        standing=standing,
        forceAction=False).metadata["lastActionSuccess"] if reach else None
    if horizon is None:
        event, horizon = aim_at(controller, x, z, yaw,
                                (target_xz[0], top + 0.08, target_xz[1]),
                                standing=standing)
    else:
        event = look_from(controller, x, z, yaw, horizon, force=True,
                          standing=standing)
    return event, {"azimuth": azimuth, "reachable": reachable,
                   "horizon": round(horizon, 3),
                   "start": f"{x:.4f},{z:.4f},{yaw:.2f},{horizon:.2f}"}


def sweep_target(controller, target_xz, top: float, radius: float,
                 angles: Sequence[float], standing: bool = STANDING,
                 horizon: Optional[float] = None,
                 reach: bool = True) -> List[Dict[str, Any]]:
    """The target's visible pixels at each azimuth, with the pose it was seen from.

    The first angle pays for `aim_at` and every later one reuses its pitch: the
    arc holds camera height and distance-to-target constant, so the pitch cannot
    differ.  `one_case` checks that against a fresh `aim_at` once per case
    rather than trusting it.
    """
    rows = []
    for azimuth in angles:
        event, pose = look_along(controller, target_xz, top, radius, azimuth,
                                 standing, horizon, reach)
        horizon = pose["horizon"] if horizon is None else horizon
        rows.append({**pose, "px": visible_pixels(event, "target")})
    return rows


def bisect_x(controller, name: str, rest_y: float, z_plane: float,
             x_shut: float, x_open: float, clear: int,
             band: Tuple[float, float, float], yaw: float = 0.0,
             steps: int = 12) -> Optional[Dict[str, Any]]:
    """
    Slide `name` along x at a fixed depth until the target's occlusion lands in
    `band`, and leave it at the best position found.

    Measured from WHEREVER THE CAMERA CURRENTLY IS, which is what lets one
    routine place both occluders: the landmark is bisected with the camera at
    the reference pose, the blocker with it at the angle the window should close
    at.  Occlusion is monotone between the bounds -- `x_shut` sits on that
    camera's sightline and hides the most, `x_open` is far enough out to hide
    nothing -- so the bisection converges whichever way round they lie.

    DEPTH IS FIXED, unlike `build_tabletop.bisect_occluder`, which slides along
    the camera ray.  Both occluders have to share a plane for `gap *
    tan(azimuth)` to describe the slot at all, and a search that moved them in
    depth would be changing the very quantity the window is defined in.
    """
    best = None
    for _ in range(steps):
        mid = (x_shut + x_open) / 2.0
        position = {"x": float(mid), "y": float(rest_y), "z": float(z_plane)}
        event = settle(controller, name, position, yaw)
        seen = visible_pixels(event, "target")
        occlusion = 1.0 - seen / clear if clear else 1.0
        if band[0] <= occlusion <= band[2] and (
                best is None
                or abs(occlusion - band[1]) < abs(best["occlusion"] - band[1])):
            best = {"occlusion": round(occlusion, 3), "position": position}
        # Towards `x_open` hides less.
        x_shut, x_open = (mid, x_open) if occlusion > band[1] else (x_shut, mid)
    if best is not None:
        settle(controller, name, best["position"], yaw)
    return best


def window_of(rows: Sequence[Dict[str, Any]], open_max: float,
              reach: float) -> Optional[Tuple[float, float]]:
    """The one contiguous run of open angles inside +-`reach`, or None.

    None means either nothing open at all or open in two places.  RECORDED AND
    NOT ACTED ON: a window the blocker split into two is still a scene where the
    best view is interior, which is the property the list is built for, so
    rejecting on shape would throw away cases that have it.
    """
    runs: List[List[float]] = []
    current: List[float] = []
    for row in rows:
        if abs(row["azimuth"]) <= reach and row["hidden"] <= open_max:
            current.append(row["azimuth"])
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return (runs[0][0], runs[0][-1]) if len(runs) == 1 else None


def one_case(controller, rng: random.Random, index: int, args,
             pools) -> Optional[Dict[str, Any]]:
    """Build one slot scene and return its case record, or None if it will not stage."""
    targets, landmarks, blockers = pools
    target_class = sorted(targets)[index % len(targets)]
    # The landmark is not the target's class, or "the box behind the box" names
    # two objects and nothing disambiguates.
    named = [c for c in landmarks if c != target_class]
    if not named:
        return None
    occ_class = rng.choice(named)
    occ_asset = rng.choice(landmarks[occ_class])
    # The blocker is neither named class, for the same reason one step further
    # out: a second `box` in frame would make `behind the box` true of two.
    free = [c for c in blockers if c not in (target_class, occ_class)]
    if not free:
        return None
    blk_class = rng.choice(free)
    blk_asset = rng.choice(blockers[blk_class])

    # THE OCCLUDERS ARE DRAWN FIRST AND THE TARGET IS DRAWN UNDER THEM.  Both
    # have to out-top the target or they cannot hide it, and the three pools
    # overlap in height -- laptops run to 0.38 m, the landmark band stops at
    # 0.40 -- so drawing all three independently and checking afterwards threw
    # away 14 of 24 attempts on that check alone.  Sampling the target from what
    # is SHORT ENOUGH turns the constraint into a draw and costs no attempts.
    #
    # It does condition the target on the landmark: a short box can only ever
    # appear with a short laptop.  That is a property of this list worth knowing
    # -- target height and landmark height are not independent -- and it is the
    # honest trade for the pool no longer being 74% garbage cans.
    db = asset_db(controller)
    ceiling = min(extent(db[occ_asset])[0],
                  extent(db[blk_asset])[0]) - TALLER_BY
    short_enough = [a for a in targets[target_class]
                    if extent(db[a])[0] <= ceiling]
    if not short_enough:
        return reject(f"no {target_class} fits under {ceiling + TALLER_BY:.2f} m "
                      f"({occ_asset}, {blk_asset})")
    asset = rng.choice(short_enough)

    controller.reset()
    centre = ROOM / 2.0
    table = spawn(controller, TABLE, "table", centre, 0.0, centre + 0.6)
    top = surface_top(table)
    box = table["axisAlignedBoundingBox"]
    near = box["center"]["z"] - box["size"]["z"] / 2.0

    # THE TARGET FIRST, ALONE, AND DEEP.  The occluders sit `gap` in front of it
    # and have to clear the near margin themselves, so the target's inset is
    # derived from the gap rather than drawn and rejected afterwards.  Unlike
    # `build_tabletop` the FAR edge binds too -- at `gap` 0.50 the inset is
    # 0.80 m of the 0.97 m the margins leave -- so the target is checked onto the
    # table like everything else.
    inset = args.gap + TABLE_MARGIN + 0.05
    tx = centre + rng.uniform(-0.12, 0.12)
    tz = near + rng.uniform(inset, inset + 0.12)
    if not on_table(table, tx, tz):
        return reject(f"target at inset {tz - near:.2f} m is off the table")
    # TURN THE TARGET TO FACE THE CAMERA TOO.  `occluder_yaw` was applied to
    # both occluders and not to this, so the target was spawned at whatever face
    # the artist modelled towards +z -- for every `Laptop_*` asset that is the
    # BACK of the screen, a featureless dark slab.  `build_tabletop.CLASS_YAW`
    # exists for exactly this and says why: the screen is both the surface that
    # gets occluded and the only part that makes the thing recognisably a laptop,
    # so which way it faces changes the occlusion AND the naming.
    #
    # The wider-side half of `occluder_yaw` comes along, and for a target that is
    # a small bonus rather than the point -- more pixels to detect -- and for a
    # laptop it is a no-op, the asset already being wider in x than deep in z.
    target_yaw = (occluder_yaw(controller, asset, target_class)
                  + args.target_turn) % 360.0
    _, target_pos = place_on(controller, asset, "target", tx, top, tz,
                             target_yaw)
    target_xz = np.array([tx, tz])

    radius = next((s for s in STANDOFF if controller.step(
        action="Teleport",
        position={"x": float(tx), "y": 0.9, "z": float(tz - s)},
        rotation={"x": 0, "y": 0.0, "z": 0}, horizon=0.0,
        standing=args.standing,
        forceAction=False).metadata["lastActionSuccess"]), None)
    if radius is None:
        return reject("nowhere legal to stand and see it")

    # THE CLEAR CURVE, measured with nothing else on the table.  The denominator
    # has to be PER ANGLE: the target's silhouette changes with the view, so
    # dividing every angle by the reference count would read a target that
    # merely turns a wider face to the camera as one the occluders reveal.
    angles = azimuths(args.sweep)
    # CAN THE LANDMARK REACH THE SIGHTLINE AT ALL?  Asked before any occluder is
    # spawned, because the answer is geometry and the alternative is finding out
    # from a 12-probe bisection that renders 12 frames to conclude nothing.
    #
    # The camera looks DOWN -- it sits ~0.66 m above the point it aims at, which
    # at radius 1.05 is a 32 degree pitch -- so the ray to the target is already
    # `gap * slope` above the table by the time it crosses the occluder plane.
    # An occluder shorter than that passes UNDER the sightline and hides
    # nothing, however wide it is and however carefully it is slid sideways.
    #
    # This is what made the first pools all garbage cans.  At `gap` 0.65 the ray
    # is 0.33 m up, so hiding half of a 0.16 m target needs 0.41 m of occluder;
    # real `Box` assets stop at 0.37 m and `GarbageCan` runs to 0.54, so the
    # bins were not a sampling artefact -- they were the only assets that could
    # work at that gap.  Shrinking `gap` is what lets a box back in, and the
    # separation it costs is bought back with more azimuth, not more depth.
    _, ref_pose = look_along(controller, target_xz, top, radius, 0.0, args.standing)
    slope = math.tan(math.radians(ref_pose["horizon"]))
    reach_up = extent(db[asset])[0] / 2.0 + args.gap * slope
    if extent(db[occ_asset])[0] < reach_up:
        return reject(f"{occ_asset} is {extent(db[occ_asset])[0]:.2f} m against "
                      f"a sightline {reach_up:.2f} m up at the occluder plane: "
                      f"it passes under the target")

    # ONE aim_at for the whole build, and a check that reusing it is legitimate:
    # the arc is a circle at fixed camera height, so the pitch at the far edge
    # must equal the pitch at the reference.  If it does not, the geometry is not
    # what the saving assumes and the sweep is wrong, not merely slow.
    pitch = ref_pose["horizon"]
    _, edge_pose = look_along(controller, target_xz, top, radius, angles[-1],
                              args.standing)
    if abs(edge_pose["horizon"] - pitch) > 0.05:
        raise RuntimeError(f"pitch varies along the arc: {pitch:.3f} at 0 deg "
                           f"against {edge_pose['horizon']:.3f} at "
                           f"{angles[-1]:+.1f} -- the reuse in `sweep_target` "
                           f"is not valid here")
    clear = {r["azimuth"]: r["px"]
             for r in sweep_target(controller, target_xz, top, radius, angles,
                                   args.standing, pitch, reach=False)}
    if clear[0.0] < MIN_PIXELS:
        return reject(f"target only {clear[0.0]} px unoccluded")

    # THE LANDMARK, bisected at the REFERENCE pose to the requested occlusion.
    # `sign` is the side it is pushed towards, and therefore the side the window
    # is NOT on: moving the camera that way walks the sightline further into it.
    sign = rng.choice((-1.0, 1.0))
    z_plane = tz - args.gap
    occ_yaw = occluder_yaw(controller, occ_asset, occ_class)
    _, occ_pos = place_on(controller, occ_asset, "occluder", tx, top, z_plane,
                          occ_yaw)
    look_along(controller, target_xz, top, radius, 0.0, args.standing, pitch,
               reach=False)
    band = (args.min_occlusion, args.ref_occlusion, args.max_occlusion)
    staged = bisect_x(controller, "occluder", occ_pos["y"], z_plane,
                      tx, tx + sign * 0.45, clear[0.0], band, occ_yaw)
    if staged is None:
        return reject(f"{occ_asset} never reached {args.ref_occlusion:.0%} at "
                      f"the reference view")
    if not on_table(table, staged["position"]["x"], z_plane):
        return reject("landmark ended up over the table's edge")

    # THE BLOCKER, bisected at the angle the window is meant to close at, on the
    # far side.  Its inner bound is where that angle's sightline crosses the
    # occluder plane -- any further in and it starts eating the reference view
    # the landmark was just bisected for -- and its outer bound is far enough to
    # miss entirely.
    # `sign` is the WORLD direction the landmark was pushed, so the window opens
    # the other way.  Under `arc_step`'s sign the sightline crosses the occluder
    # plane at `tx - gap tan(azimuth)`, so reaching the `-sign` side takes an
    # azimuth of `+sign` -- the opposite of what this line said before the
    # convention was fixed, and it has to move with it or the blocker is bisected
    # at an angle on the side the window is not on.
    close_at = sign * args.reach
    blk_yaw = occluder_yaw(controller, blk_asset, blk_class)
    _, blk_pos = place_on(controller, blk_asset, "blocker", tx, top, z_plane,
                          blk_yaw)
    look_along(controller, target_xz, top, radius, close_at, args.standing,
               pitch, reach=False)
    blocked = bisect_x(controller, "blocker", blk_pos["y"], z_plane,
                       tx - args.gap * math.tan(math.radians(close_at)),
                       tx - sign * 0.55, clear[close_at],
                       (args.shut_frac, 1.0, 1.0), blk_yaw)
    if blocked is None:
        return reject(f"{blk_asset} never shut the window at "
                      f"{close_at:+.1f} deg")
    if not on_table(table, blocked["position"]["x"], z_plane):
        return reject("blocker ended up over the table's edge")

    # THE WHOLE CURVE, now that both are placed.  Everything above was measured
    # one angle at a time with one occluder in the room, so none of it binds:
    # the blocker can reach back into the reference view, and the landmark can
    # spill past the angle the blocker was set at.
    rows = sweep_target(controller, target_xz, top, radius, angles,
                        args.standing, pitch)
    for row in rows:
        row["clear_px"] = clear[row["azimuth"]]
        row["hidden"] = round(1.0 - row["px"] / max(row["clear_px"], 1), 3)
    by_angle = {r["azimuth"]: r for r in rows}

    reference = by_angle[0.0]
    if not band[0] <= reference["hidden"] <= band[2]:
        return reject(f"the blocker moved the reference view to "
                      f"{reference['hidden']:.0%} hidden, outside the band")
    # THE PROPERTY THIS LIST IS FOR, and it is weaker than a narrow slot.  What
    # has to be false is "walk as far as you can to one side and look": the best
    # view must be INTERIOR to the reachable sweep, and both ends must be
    # meaningfully worse than it.  An earlier version demanded the target be
    # 80% hidden at both ends -- a true keyhole -- and nearly every attempt died
    # on `never shut the window`, rejecting scenes that had the property asked
    # for.  Being 30 points worse at the edges is enough to make walking to an
    # edge a mistake; being fully hidden there is a different, harder dataset.
    inside = [r for r in rows if abs(r["azimuth"]) <= args.reach]
    best = min(inside, key=lambda r: r["hidden"])
    if best["hidden"] > args.open_max:
        return reject(f"never opens: best is {best['hidden']:.0%} hidden at "
                      f"{best['azimuth']:+.1f} deg")
    if best["px"] < MIN_PIXELS:
        return reject(f"opens to only {best['px']} px")
    if abs(best["azimuth"]) > args.interior:
        return reject(f"the best view is at {best['azimuth']:+.1f} deg, on the "
                      f"rim of the sweep: walking to the end would do")
    ends = [by_angle[a] for a in (-args.reach, args.reach) if a in by_angle]
    weak = [e for e in ends if e["hidden"] - best["hidden"] < args.falloff]
    if weak:
        return reject(
            "an end of the sweep is as good as the best view: "
            + ", ".join(f"{e['azimuth']:+.0f} deg {e['hidden']:.0%} vs "
                        f"{best['hidden']:.0%}" for e in weak))
    # Recorded, not required.  A window in two pieces is a legitimate scene --
    # the blocker split it -- and rejecting on shape would drop cases that have
    # the property above.
    window = window_of(rows, args.open_max, args.reach)

    # THE LANDMARK IS BOUNDED ON BOTH SIDES, and only the lower bound is
    # `build_tabletop`'s.  Below the target's own size, "the laptop behind the
    # box" describes a box smaller than the laptop and the premise is gone.
    # Above `--max-landmark-ratio`, the landmark is the picture: the first four
    # cases built here had landmarks 5.3 to 10.5 times the target, filling
    # 18-28% of the frame, which is the state `build_tabletop`'s HousePlant note
    # describes as making `behind the box` satisfiable by almost anything.  A
    # band on the asset cannot catch either end -- the landmark is `gap` nearer
    # the camera, so it renders larger than its metres suggest -- which is why
    # both are measured in PIXELS, at the OPEN angle where the target is all
    # there.
    open_event, _ = look_along(controller, target_xz, top, radius,
                               best["azimuth"], args.standing, pitch,
                               reach=False)
    occluder_px = visible_pixels(open_event, "occluder")
    if best["px"] >= occluder_px:
        return reject(f"target is {best['px']} px against the landmark's "
                      f"{occluder_px}: the target is the bigger object")
    if occluder_px > args.max_landmark_ratio * best["px"]:
        return reject(f"landmark is {occluder_px / best['px']:.1f}x the target "
                      f"({occluder_px} px, {occluder_px / (args.width * args.height):.0%} "
                      f"of the frame): it swallows the picture")

    # IS IT NAMEABLE FROM THE BEST VIEW?  Every test above is geometry -- pixels
    # of silhouette, degrees of arc -- and a target can pass all of them and
    # still be unrecognisable.  A THOR laptop turned to the camera is a black
    # slab: the first build of this list read 97% of its pixels as visible at the
    # window and EGTR still called it `screen`, `pot` or `door`, so `walk into
    # the window` was a losing move for a reason that has nothing to do with
    # viewpoint.
    #
    # THE POINT IS TO CONTROL A VARIABLE, not to flatter the detector.  A list
    # meant to ask "what does moving buy?" has to hold "can it be named at all?"
    # fixed, or the two are confounded and the ceiling measures naming.  The
    # screen runs at the BEST view only -- that is the one pose where the answer
    # must be yes -- and the achieved rank is recorded so the cut can be
    # tightened offline instead of by rebuilding.
    #
    # It is a detector-in-the-loop filter and the paper has to say so: cases are
    # screened so the target is nameable from its best viewpoint, which makes the
    # ceiling a property of VIEWPOINT rather than of recognition.
    solved_arc = None
    if args.solved_width:
        task = {"instruction": f"Find the {target_class} behind the {occ_class}",
                "predicate": "behind", "subject_class": target_class,
                "object_class": occ_class, "target_name": "target",
                "receptacle_name": "occluder", "distractors": []}
        egtr = nameable_egtr()
        # THE BEST VIEW FIRST, and reject there.  Most candidates that reach
        # this line fail, and failing on one frame instead of 25 is the
        # difference between a screen that costs a fifth of the build and one
        # that costs the build.
        if not solved_at(open_event, egtr, task, args.condition,
                         args.solved_iou):
            return reject("the top-1 pair is wrong even at the best view")
        step = float(args.sweep[2])
        solved = [best["azimuth"]]
        for row in inside:
            az = row["azimuth"]
            if az == best["azimuth"]:
                continue
            event = look_along(controller, target_xz, top, radius, az,
                               args.standing, pitch, reach=False)[0]
            if solved_at(event, egtr, task, args.condition, args.solved_iou):
                solved.append(az)
        run = longest_run(sorted(solved), step)
        if run is None or run[1] - run[0] < args.solved_width:
            width = 0.0 if run is None else run[1] - run[0]
            return reject(f"the solved arc is only {width:.1f} deg wide, under "
                          f"{args.solved_width:.1f}")
        if abs((run[0] + run[1]) / 2) > args.interior:
            return reject(f"the solved arc is centred at "
                          f"{(run[0] + run[1]) / 2:+.1f} deg, on the rim of the "
                          f"sweep: walking to the end would do")
        solved_arc = [run[0], run[1]]

    event, ref_pose = look_along(controller, target_xz, top, radius, 0.0,
                                args.standing, pitch, reach=False)

    # RE-CHECK THE START POSE, NOW THAT EVERYTHING IS ON THE TABLE.  `radius` was
    # chosen right after the target was placed and before either occluder
    # existed, so the collision check it passed was against a different scene.
    # The occluders sit `gap` nearer the camera, and on a case at the near end of
    # STANDOFF that is enough to make the pose illegal: `slot|46` shipped with a
    # start pose THOR accepts with the target alone and refuses with all three,
    # 0.206 m from a table edge whose collider reaches 0.23 m out.
    #
    # It has to be asked UNFORCED and here, because `rebuild` asks it unforced
    # too -- that is the whole contract, a case that cannot be replayed is a case
    # whose geometry is wrong -- and a case that fails there fails for every
    # runner, after the GPU time.  This is what caught it: 1 of 40.
    x, z, yaw, horizon = (float(v) for v in ref_pose["start"].split(","))
    if not controller.step(
            action="Teleport", position={"x": x, "y": 0.9, "z": z},
            rotation={"x": 0, "y": yaw, "z": 0}, horizon=horizon,
            standing=args.standing,
            forceAction=False).metadata["lastActionSuccess"]:
        return reject(f"the start pose is legal with the target alone but not "
                      f"with the occluders in place (radius {radius})")
    event = look_along(controller, target_xz, top, radius, 0.0, args.standing,
                       pitch, reach=False)[0]
    return {
        "scene": f"slot|{index}",
        "instruction": f"Find the {target_class} behind the {occ_class}",
        "predicate": "behind",
        "subject_class": target_class, "object_class": occ_class,
        "target_name": "target", "landmark_name": "occluder",
        "occluder_name": "occluder",
        # The blocker is named so a probe can measure it, NOT so an instruction
        # can refer to it.  `blocker_class` is recorded because it has to differ
        # from `object_class` for the instruction to name one object, and that
        # is worth being able to check after the fact.
        "blocker_name": "blocker", "blocker_class": blk_class,
        # There is no same-class competitor on this list.  Recorded as a flag
        # rather than a null `distractor_name`, so a runner that needs one fails
        # loudly instead of grading against a name that is None.
        "has_distractor": False,
        "occluder_position": staged["position"],
        "blocker_position": blocked["position"],
        "target_yaw": target_yaw,
        "occluder_yaw": occ_yaw, "blocker_yaw": blk_yaw,
        # THE ARC THE POLICY IS ACTUALLY ASKED TO FIND: the widest contiguous
        # stretch of azimuths where `fuse_live.decide`'s top-1 pair is the
        # instructed one.  `visible_arc` below is the GEOMETRIC window and the
        # two are not the same window -- on the list built before this screen
        # they overlapped 58% -- so a policy graded against the metric has to be
        # aimed at this one.
        "solved_arc": solved_arc,
        "staged_occlusion": reference["hidden"],
        "clear_px": clear[0.0], "target_px": reference["px"],
        "occluder_px": occluder_px,
        # Recorded so `--max-landmark-ratio` can be tightened on the finished
        # list instead of by rebuilding it.
        "landmark_ratio": round(occluder_px / max(best["px"], 1), 2),
        # THE WINDOW, and the ceiling it sets.  `visible_arc` is the contiguous
        # run of angles at or under `open_max` hidden; `best_azimuth` is where
        # the target is most visible; `reachable_arc` is the part of the window
        # THOR would let the robot stand in unforced.  Columns, never filters.
        "visible_arc": list(window) if window else None,
        "best_azimuth": best["azimuth"], "best_hidden": best["hidden"],
        "best_px": best["px"],
        "reachable_arc": [r["azimuth"] for r in inside
                          if r["reachable"] and r["hidden"] <= args.open_max],
        # `rebuild` reads this; without it the replay stands up and sees a
        # scene 0.68 m of camera height away from the one staged.  See `STANDING`.
        "standing": args.standing,
        "gap": args.gap, "radius": radius, "close_at": close_at,
        "sweep": rows,
        "table": {"asset": TABLE, "x": centre, "z": centre + 0.6, "top": top},
        # WITHOUT `yaw` HERE THE REPLAY IS A DIFFERENT SCENE -- `spawn` defaults
        # the rotation to 0, so a target turned to face the camera would come
        # back showing its featureless side, and every occlusion in `sweep` would
        # be one no run ever saw.  `build_tabletop` records the occluder's yaw
        # for the same reason.
        "objects": [{"name": "target", "asset": asset,
                     "position": target_pos, "yaw": target_yaw},
                    {"name": "occluder", "asset": occ_asset,
                     "position": staged["position"], "yaw": occ_yaw},
                    {"name": "blocker", "asset": blk_asset,
                     "position": blocked["position"], "yaw": blk_yaw}],
        "target_box": visible_box(event, "target"),
        "start": ref_pose["start"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-turn", type=float, default=TARGET_TURN,
                    metavar="DEG", help="extra yaw on the target, on top of the "
                                        "180 that brings a laptop screen round "
                                        "to the camera.  Changes its silhouette, "
                                        "so it changes every occlusion measured")
    ap.add_argument("--standing", action="store_true", default=STANDING,
                    help="stage from a STANDING camera (1.575 m, pitched 27 deg "
                         "down) instead of a crouched one (0.900 m, level).  "
                         "Standing, a tabletop occluder is passed over rather "
                         "than looked through -- see `STANDING`")
    ap.add_argument("--gap", type=float, default=GAP, metavar="M",
                    help="target-to-occluder separation.  The slot lives in "
                         "gap * tan(azimuth), so this sets how many metres the "
                         "reachable sweep spans across the occluder plane")
    ap.add_argument("--ref-occlusion", type=float, default=0.50, metavar="F",
                    help="how much of the target the LANDMARK hides at the "
                         "reference view")
    ap.add_argument("--min-occlusion", type=float, default=0.35)
    ap.add_argument("--max-occlusion", type=float, default=0.65)
    ap.add_argument("--open-max", type=float, default=0.30, metavar="F",
                    help="hidden fraction at or below which a view counts OPEN."
                         "  A floor on 'there is a good view at all'; --falloff "
                         "is what makes it the RIGHT view")
    # ALL THREE OF THESE WERE THE KEYHOLE'S NUMBERS, and they outlived it.  On
    # the first 18 attempts after the acceptance test was relaxed to --falloff,
    # 9 died on the blocker failing to reach 80%, 3 on a best view of 26-28%
    # against an --open-max of 0.20, and 2 on a landmark that could not land
    # inside a 0.40-0.60 reference band.  None of those three scenes lacked the
    # property the list is for; they failed thresholds set for a stricter one.
    #
    # The blocker's own target moved the furthest, and it is the clearest case:
    # with a best view at 13% hidden, --falloff 0.30 needs the ends at 43%, so
    # bisecting the blocker to 80% was asking for nearly twice what any test
    # downstream reads.  It is also the hardest to deliver -- lamp and plant
    # assets have bounding boxes far wider than their opaque silhouette, so a
    # blocker centred on the sightline covers much less than its width suggests.
    ap.add_argument("--shut-frac", type=float, default=0.55, metavar="F",
                    help="how far the BLOCKER is bisected to shut the view at "
                         "its own angle.  Not an acceptance test -- see "
                         "--falloff for the property cases are kept on")
    ap.add_argument("--interior", type=float, default=25.0, metavar="DEG",
                    help="the best view must be within this of the reference. "
                         "Beyond it the optimum sits on the rim of the sweep "
                         "and 'walk to the end' is the answer, which is the "
                         "case this list exists to exclude")
    # 0.15, FROM THE OBSERVED DISTRIBUTION rather than a guess.  Over 30
    # attempts the achieved falloff was 28, 23, 22, 20, 11, 8 and 6 points, and
    # the scene that was eyeballed and accepted sat at 18 (an end at 19% hidden
    # against a best view of 1%).  0.30 rejected all seven; 0.15 keeps the four
    # largest plus that one and still excludes the 6-to-11-point cases, where
    # the end of the sweep really is as good a view as the best.
    ap.add_argument("--falloff", type=float, default=0.15, metavar="F",
                    help="how much MORE hidden each end of the reachable sweep "
                         "must be than the best view.  This, not a keyhole, is "
                         "what makes walking to an end a mistake")
    ap.add_argument("--reach", type=float, default=30.0, metavar="DEG",
                    help="the azimuth everything downstream is capped at "
                         "(fuse_live --max-az, eval_move's bearings).  A window "
                         "outside this is unwinnable by any arm, so cases are "
                         "rejected on it rather than shipped with a ceiling "
                         "of zero")
    ap.add_argument("--sweep", type=float, nargs=3, default=list(SWEEP),
                    metavar=("LO", "HI", "STEP"))
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    # Classes as flags, size bands as `ROLES`, on the split `build_tabletop`
    # uses: which classes a list is about is the thing a run changes, and the
    # bands are measurements that a change would invalidate the notes on.
    for role in ROLES:
        ap.add_argument(f"--{role}-classes", nargs="+", metavar="C",
                        default=list(ROLES[role]["classes"]),
                        help=f"VG150 classes the {role} may be drawn from; see "
                             f"`ROLES` for the size band and the reason")
        ap.add_argument(f"--{role}-types", nargs="+", metavar="T", default=None,
                        help=f"restrict the {role} to these THOR objectTypes "
                             f"within those classes.  `--landmark-types Box` "
                             f"drops the garbage cans VG150 also calls `box`; "
                             f"the height band already leaves them a minority")
    # LOOSE ON PURPOSE, AND THE REASON MATTERS.  This was 3.0, set against the
    # STANDING camera where a landmark filling 18-28% of the frame meant a
    # garbage can had been drawn.  It does not transfer: crouched, a box that
    # hides half the target necessarily presents its whole front face, and real
    # `Box` assets measure 4.1 to 7.5 times the target at 17-26% of the frame --
    # the same range the bins were rejected for.  The same number is a symptom
    # under one camera and a consequence under the other, so at 3.0 this guard
    # rejected the entire crouched design (11 of 30 attempts).
    #
    # Kept as a backstop against the absurd rather than removed, and `one_case`
    # records the achieved ratio either way, so a stricter cut can be made on
    # the case list afterwards without rebuilding it.
    # THE SCREEN, AND WHAT IT IS FOR.  Turned off (0) the list ships cases no
    # viewpoint solves: the first build was screened on geometry alone and shipped
    # targets 97%% visible at the window that EGTR still read as `screen` or
    # `door`, which confounds `what does moving buy` with `can this be named at
    # all`.  Screened on the class name instead, 40 of 40 cases named the target
    # at their best view and only 17 of 40 had the top-1 PAIR right there -- the
    # proxy bought 43%%.  So the screen asks the metric itself, at every azimuth,
    # and keeps the case only if a contiguous stretch of them answers.
    #
    # It is a detector-in-the-loop filter and the paper has to say so: the arc is
    # defined RELATIVE TO THIS DETECTOR, which makes the ceiling a property of
    # viewpoint rather than of recognition, and makes this list unfit for
    # comparing one SGG model against another.  What it is fit for is the
    # comparison it was built for -- multi-view against single-view, same model.
    ap.add_argument("--solved-width", type=float, default=10.0, metavar="DEG",
                    help="reject a case unless `fuse_live.decide`'s top-1 pair "
                         "is correct across a contiguous arc at least this "
                         "wide.  0 turns the screen off")
    ap.add_argument("--solved-iou", type=float, default=0.5, metavar="IOU",
                    help="IoU both endpoints of the top-1 pair must reach for "
                         "the angle to count as solved; `eval_move`'s `--iou`")
    ap.add_argument("--condition", type=int, default=10, metavar="K",
                    help="candidates per side the screen scores, "
                         "`eval_move`'s `--condition`")
    ap.add_argument("--max-landmark-ratio", type=float, default=10.0,
                    metavar="X", help="reject a case whose landmark renders "
                                      "more than X times the target's pixels")
    # 6 was right when geometry was the only screen and roughly one attempt in
    # five survived.  Asking the metric costs about three times that, so the cap
    # became the binding constraint rather than the backstop it is meant to be.
    ap.add_argument("--tries", type=int, default=6, metavar="X",
                    help="give up after X attempts per case asked for")
    # ONE THOR PROCESS PER WORKER, NOT ONE THOR PER THREAD.  A build is a long
    # sequence of renders with a detector pass on some of them, so it scales by
    # process and by nothing else -- but each worker carries its own Unity AND
    # its own EGTR, which on an 8 GB card measured 2.9 GB apiece.  Two fit; the
    # third is what took the machine down before, the GPU also being the one
    # driving the display.  Workers are SUBPROCESSES rather than forks because a
    # CUDA context does not survive `fork`, and they are staggered because two
    # Unity instances claiming the display at the same instant is its own hazard.
    ap.add_argument("--workers", type=int, default=1, metavar="N",
                    help="build in N processes, seeds `--seed` .. `--seed`+N-1, "
                         "and merge.  Each costs about 2.9 GB of VRAM")
    ap.add_argument("--stagger", type=float, default=45.0, metavar="S",
                    help="seconds between worker launches")
    ap.add_argument("--out", default="nvs_pilot/cases/cases_slot.json")
    args = ap.parse_args(argv)

    if args.workers > 1:
        return fan_out(args, argv)

    rng = random.Random(args.seed)
    controller = open_room(args.width, args.height, args.fov)
    cases: List[Dict[str, Any]] = []
    try:
        pools = tuple(pool(controller, tuple(getattr(args, f"{role}_classes")),
                           ROLES[role]["height"], ROLES[role]["width"],
                           getattr(args, f"{role}_types"))
                      for role in ("target", "landmark", "blocker"))
        for role, got in zip(("target:  ", "landmark:", "blocker: "), pools):
            print("  " + role + " " + (", ".join(f"{k}x{len(v)}"
                                                 for k, v in got.items())
                                       or "NOTHING IN THE BAND"))
        if not all(pools):
            print("  ! a slot needs all three roles")
            return 1
        attempt = 0
        while len(cases) < args.n and attempt < args.tries * args.n:
            case = None
            try:
                case = one_case(controller, rng, attempt, args, pools)
            except Exception as error:                       # noqa: BLE001
                print(f"  ! {type(error).__name__}: {error}", flush=True)
            attempt += 1
            if case:
                cases.append(case)
                arc = case["visible_arc"]
                ends = [r["hidden"] for r in case["sweep"]
                        if abs(r["azimuth"]) == args.reach]
                print(f"[{len(cases)}/{args.n}] {case['instruction']:34s} "
                      f"ref {case['staged_occlusion']:.0%}  best "
                      f"{case['best_hidden']:.0%} at "
                      f"{case['best_azimuth']:+5.1f}  ends "
                      + "/".join(f"{h:.0%}" for h in ends)
                      + (f"  window {arc[0]:+.1f}..{arc[1]:+.1f}" if arc
                         else "  window split"), flush=True)
    finally:
        controller.stop()

    if not cases:
        print("no cases built")
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"predicate_families": ["behind"],
                   "occlusion_band": [args.min_occlusion, args.ref_occlusion,
                                      args.max_occlusion],
                   "procedural": True, "slot": True, "reach": args.reach,
                   "open_max": args.open_max, "shut_frac": args.shut_frac,
                   "cases": cases}, fh, indent=1)
    walked = sorted(abs(c["best_azimuth"]) for c in cases)
    widths = sorted(c["visible_arc"][1] - c["visible_arc"][0]
                    for c in cases if c["visible_arc"])
    print(f"\n{len(cases)}/{attempt} attempts -> {args.out}")
    print(f"  best view {walked[0]:.1f}-{walked[-1]:.1f} deg off axis, median "
          f"{walked[len(walked) // 2]:.1f}")
    if widths:
        print(f"  window width {widths[0]:.1f}-{widths[-1]:.1f} deg, median "
              f"{widths[len(widths) // 2]:.1f} ({len(widths)}/{len(cases)} "
              f"in one piece)")
    print(f"  window reachable unforced: "
          f"{sum(1 for c in cases if c['reachable_arc'])}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
