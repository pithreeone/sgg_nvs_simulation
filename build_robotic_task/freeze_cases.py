"""
freeze_cases.py -- turn a discovered case list into a VERIFIED one.

`find_cases.py` emits cases it believes are stageable: it checks, in arithmetic,
that some occluder type has room to stand between the camera and the target.
That is a weaker claim than it sounds.  "There is room to stand there" and "it
actually hides the target" are different questions, and only the second needs a
render -- so 9 of 36 cases in the first big list failed at run time, seven of
them reporting a best legal placement that hid 0-10%.

This file closes the gap by DOING the staging.  For each case it walks the
ranked occluder types -- tallest first, `find_cases.rank_occluders` -- and keeps
the first one that measurably hides the target.  The type it settles on is
written back, so a frozen case names an occluder that has been seen to work
rather than one that was inferred to fit.

What a frozen case gains over a discovered one:

  * zero attrition at run time -- every case in the file has been staged once
  * the occluder is the VERIFIED one, which need not be the first choice
  * the exact placement is recorded -- instance name and world position -- so
    every later run REPLAYS it rather than re-running the grid search, which is
    not reproducible across runs and moved the achieved occlusion by several
    points between repeats
  * `staged_occlusion` records what was achieved, so the list can be filtered by
    difficulty without opening THOR
  * `predicate` is explicit.  It is `on` for every case today because that is
    the only family EGTR reaches (44% recall against ~0 on the rest, TASKS.md),
    but the field exists so a later list can mix predicates without a format
    change.

The frozen file stores the QUESTION and the SCENE, never an answer: nothing
about EGTR, ranks or viewpoints is in it, and the runners re-derive every box
and every occlusion at the pose they are standing in.  What changed on
2026-08-10 is that the scene is now pinned exactly -- `occluder_position` is
replayed, so `staged_occlusion` is a guarantee about the next run rather than a
report about this one.

    python freeze_cases.py --cases nvs_pilot/cases/cases_big.json \\
        --out nvs_pilot/cases/cases_frozen.json     # deleted; see nvs_pilot/README.md
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
import os
from typing import Any, Dict, List, Optional, Sequence

from robot import drive
from robot.drive_triplet_scene import measure
from build_robotic_task.find_cases import rank_occluders
from gen.occlusion import pose_snapshot
from robot.task_find import (behind_task, build_tasks,
                             in_front_task, place_beside,
                             put_in_front)
from vg.vg150 import THOR_TO_VG150


def verify(rc, case: Dict[str, Any], band: Sequence[float],
           predicate: str = "on", max_ratio: float = 4.0,
           max_box: float = 0.0,
           distract: bool = False) -> Optional[Dict[str, Any]]:
    """Stage this case for real.  Returns the frozen record, or None."""
    nameable = sorted(o["name"] for o in rc.event.metadata["objects"]
                      if o["objectType"] in THOR_TO_VG150)
    state = measure(rc, nameable, 60.0)
    entries = {o["name"]: o for o in rc.event.metadata["objects"]}
    task = next((t for t in build_tasks(state["objects"], entries)
                 if t["target_name"] == case["target_name"]), None)
    if task is None:
        print("    ! no longer an unambiguous instruction here")
        return None

    # The type the discovery picked first, then the rest in the same order it
    # would have considered them.  Trying the recorded one first keeps a case
    # that already works byte-identical to what the runs used.
    ranked = rank_occluders(rc.event, entries[task["target_name"]],
                            entries[task["receptacle_name"]],
                            rc.camera_xyz[[0, 2]], max_ratio)
    order = ([case["occluder_type"]] if case.get("occluder_type") else []) + \
            [t for t in ranked if t != case.get("occluder_type")]
    if predicate != "on":
        # The occluder is now an ENDPOINT of the instruction, so a type VG150
        # cannot name (CoffeeMachine, Statue, ...) is not a candidate at all.
        # Filtering here rather than after staging saves a render per attempt.
        order = [t for t in order if t in THOR_TO_VG150]

    for occluder_type in order:
        staged = put_in_front(rc, task["target_name"], occluder_type,
                              task["receptacle_name"],
                              min_occlusion=band[0], target_occlusion=band[1],
                              max_occlusion=band[2])
        if staged is not None:
            # THE OCCLUDER'S SHARE OF THE FRAME, measured after staging.
            # `rank_occluders` caps the physical AABB, which is the wrong
            # quantity: what made "behind the plant" vacuous is the box a
            # DETECTOR draws, and that grows with the sprawl of the leaves and
            # with perspective once the plant has been slid toward the camera.
            # Capping the AABB at 4x changed nothing -- HousePlant survived 12
            # times out of 12.  This is the same idea measured where it counts.
            from gen.occlusion import visible_pixels
            frame_px = rc.event.frame.shape[0] * rc.event.frame.shape[1]
            share = visible_pixels(rc.event, staged["occluder"]) / frame_px
            if max_box and share > max_box:
                print(f"    ! {occluder_type} covers {share:.1%} of the frame, "
                      f"over the {max_box:.0%} cap")
                continue
            beside = None
            if distract:
                # A second object of the TARGET's VG150 class, so the relation
                # has something to disambiguate.  iTHOR has no two instances of
                # one objectType, so this is another type mapping to the same
                # class -- see `place_beside`.
                want = THOR_TO_VG150.get(
                    entries[task["target_name"]]["objectType"])
                twin = next((o["name"] for o in rc.event.metadata["objects"]
                             if o["name"] != task["target_name"]
                             and (o.get("moveable") or o.get("pickupable"))
                             and THOR_TO_VG150.get(o["objectType"]) == want),
                            None)
                if twin is None:
                    print(f"    ! no second `{want}` in this scene")
                    continue
                beside = place_beside(rc, task["target_name"], twin,
                                      task["receptacle_name"])
                if beside is None:
                    print(f"    ! {twin} cannot stand clear of the sightline")
                    continue
                print(f"    + distractor {twin} at "
                      f"{beside['lateral']:.2f} m off the ray, "
                      f"{beside['visible_px']} px visible")
            if predicate in ("behind", "in front of"):
                # The occluder is now ON the sightline, so the relation exists.
                # Rewriting AFTER staging rather than before is what makes it
                # true by construction instead of asserted.  Into a NEW name --
                # assigning over `task` and then `continue`ing left the next
                # occluder in the loop dereferencing None.
                rewrite = (behind_task if predicate == "behind"
                           else in_front_task)
                relation = rewrite(task, staged["occluder"],
                                   state["objects"], entries)
                if relation is None:
                    print(f"    ! {occluder_type} cannot name a "
                          f"`{predicate}` instruction here")
                    continue
                task = relation
            return {
                "scene": case["scene"],
                "start": case["start"],
                "instruction": task["instruction"],
                "predicate": task["predicate"],
                "subject_class": task["subject_class"],
                "object_class": task["object_class"],
                "target_name": task["target_name"],
                "landmark_name": task.get("landmark_name"),
                "receptacle_name": task["receptacle_name"],
                "occluder_type": occluder_type,
                "instruction_on": task.get("on_instruction"),
                # The INSTANCE and its world position, so a run replays this
                # placement instead of searching for its own -- see
                # `task_find.stage_at` for why the search is not reproducible.
                "occluder_name": staged["occluder"],
                "occluder_share": round(share, 4),
                "distractor_name": None if not beside else beside["distractor"],
                "distractor_position": None if not beside else beside["position"],
                "occluder_position": staged["position"],
                # Every moveable object's pose AFTER staging.  The occluder
                # alone is not enough -- THOR's load settle moves the target
                # too, and that changed its visible box between runs.
                "scene_poses": pose_snapshot(rc.event),
                "staged_occlusion": staged["occlusion"],
                "occluder_alternatives": [t for t in order
                                          if t != occluder_type],
            }
    print(f"    ! none of {order or '[]'} hides {task['target_name']}")
    return None



def _one(payload):
    """
    Stage one case in this process.  Module level, so `spawn` can pickle it.

    Each case already opened and closed its own THOR instance in the serial
    version, so cases are independent and a worker pool needs no shared state.
    What it does need is for the CHILD to import `drive` lazily -- importing
    ai2thor at module scope in every worker costs more than the staging.
    """
    index, total, case, args = payload
    from robot import drive

    print(f"[{index}/{total}] {case['scene']}  {case['instruction']}", flush=True)
    rc = drive.open_scene(case["scene"], args.width, args.height, args.fov,
                          case["start"])
    try:
        record = verify(rc, case, (args.min_occlusion, args.target_occlusion,
                                   args.max_occlusion), args.predicate,
                        args.max_occluder_ratio, args.max_occluder_box,
                        args.distract)
    except Exception as error:                                   # noqa: BLE001
        print(f"    ! {type(error).__name__}: {error}", flush=True)
        record = None
    finally:
        rc.stop()
    if record:
        swapped = ("" if record["occluder_type"] == case.get("occluder_type")
                   else f"  (swapped from {case.get('occluder_type')})")
        print(f"    ok  {record['occluder_type']} hides "
              f"{record['staged_occlusion']:.0%}{swapped}", flush=True)
    return index, record


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", nargs="+", required=True,
                    help="one or more discovered case files; duplicates by "
                         "(scene, target) are dropped")
    ap.add_argument("--min-occlusion", type=float, default=0.15,
                    help="a staging that hides less than this is not a hard "
                         "question and the case is dropped")
    ap.add_argument("--target-occlusion", type=float, default=0.50,
                    help="the placement whose measured occlusion is NEAREST "
                         "this is the one kept")
    ap.add_argument("--max-occlusion", type=float, default=0.90,
                    help="placements that hide more than this are rejected; a "
                         "target with no pixels left is a separate regime")
    ap.add_argument("--max-occluder-ratio", type=float, default=4.0,
                    help="cap on the occluder's silhouette relative to the "
                         "target's.  It is an ENDPOINT of a `behind` "
                         "instruction, and a box spanning a whole house plant "
                         "makes the relation vacuous.")
    ap.add_argument("--max-occluder-box", type=float, default=0.0,
                    help="cap on the fraction of the FRAME the staged occluder "
                         "covers, from its instance mask.  0 = no cap, and the "
                         "share is recorded either way so a threshold can be "
                         "chosen from the distribution rather than guessed -- "
                         "the AABB cap was guessed twice and bit neither time.")
    ap.add_argument("--distract", action="store_true",
                    help="also place a second object of the TARGET's class, off "
                         "the sightline.  Without one the relation in `the "
                         "bottle behind the plant` is decoration: all 71 frozen "
                         "cases had zero same-class distractors, so `behind` "
                         "never had to disambiguate anything.")
    ap.add_argument("--predicate", default="on",
                    choices=("on", "behind", "in front of"),
                    help="`behind` rewrites each task as the relation staging "
                         "creates -- the target behind its occluder -- which is "
                         "the viewpoint-DEPENDENT family a novel view can "
                         "actually inform.  See task_find.behind_task.")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=1,
                    help="stage this many cases at once.  Each worker runs its "
                         "own THOR instance, so this is bounded by GPU memory "
                         "and by how many Unity processes the machine tolerates, "
                         "not by CPU cores.")
    ap.add_argument("--out", default="nvs_pilot/cases/cases_frozen.json")
    args = ap.parse_args(argv)

    raw: List[Dict[str, Any]] = []
    seen = set()
    for path in args.cases:
        for case in json.load(open(path))["cases"]:
            key = (case["scene"], case["target_name"])
            if key in seen:
                continue
            seen.add(key)
            raw.append(case)
    print(f"{len(raw)} distinct discovered cases from {len(args.cases)} file(s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def save(records: List[Dict[str, Any]]) -> None:
        """Rewrite the output file.  Called after EVERY case, not at the end.

        Staging a hundred cases is twenty minutes of THOR, and a machine that
        dies at case 6 with an end-of-run write leaves nothing at all -- which
        happened twice.  The file is small and the rewrite is far cheaper than
        one staging, so there is no reason to batch it.
        """
        with open(args.out, "w") as fh:
            json.dump({"predicate_families": sorted({c["predicate"]
                                                     for c in records}),
                       "occlusion_band": [args.min_occlusion,
                                          args.target_occlusion,
                                          args.max_occlusion],
                       "cases": records}, fh, indent=1)

    payloads = [(i, len(raw), case, args) for i, case in enumerate(raw, 1)]
    if args.workers > 1:
        import concurrent.futures as cf
        import multiprocessing as mp

        # `spawn`: THOR holds a Unity process and a socket, and forking a parent
        # that has already opened one hands the child a descriptor it must not
        # touch.  A fresh interpreter per worker costs a few seconds once.
        with cf.ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=mp.get_context("spawn")) as pool:
            done = []
            for result in pool.map(_one, payloads):
                done.append(result)
                save([r for _, r in sorted(done, key=lambda x: x[0]) if r])
    else:
        done = []
        for payload in payloads:
            done.append(_one(payload))
            save([r for _, r in done if r])

    # Input order, not completion order, so the frozen list is reproducible.
    frozen = [record for _, record in sorted(done, key=lambda r: r[0]) if record]

    save(frozen)

    import collections
    print(f"\n{len(frozen)}/{len(raw)} verified -> {args.out}")
    print(f"  floor plans      {len({c['scene'] for c in frozen})}")
    print(f"  start poses      {len({(c['scene'], c['start']) for c in frozen})}")
    bands = collections.Counter(
        "0.75+" if c["staged_occlusion"] >= 0.75 else
        "0.50-0.75" if c["staged_occlusion"] >= 0.50 else
        "0.25-0.50" if c["staged_occlusion"] >= 0.25 else "<0.25"
        for c in frozen)
    for band in ("<0.25", "0.25-0.50", "0.50-0.75", "0.75+"):
        print(f"  occlusion {band:<10} {bands[band]}")
    print("  instructions     "
          + ", ".join(f"{k} x{v}" for k, v in
                      collections.Counter(c["instruction"][9:]
                                          for c in frozen).most_common(6)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
