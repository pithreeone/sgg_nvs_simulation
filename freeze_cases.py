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
  * `staged_occlusion` records what was achieved, so the list can be filtered by
    difficulty without opening THOR
  * `predicate` is explicit.  It is `on` for every case today because that is
    the only family EGTR reaches (44% recall against ~0 on the rest, TASKS.md),
    but the field exists so a later list can mix predicates without a format
    change.

The frozen file is still a POINTER to a question, not a record of an answer:
nothing about EGTR, ranks or viewpoints is stored, and the runners re-derive
every box and every occlusion at the pose they are standing in.  THOR's physics
settle is not bit-reproducible, so `staged_occlusion` is what one staging
achieved, not a guarantee about the next.

    python freeze_cases.py --cases nvs_pilot/cases_big.json \\
        --out nvs_pilot/cases_frozen.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from robot import drive
from robot.drive_triplet_scene import measure
from find_cases import rank_occluders
from robot.task_find import build_tasks, put_in_front
from vg.vg150 import THOR_TO_VG150


def verify(rc, case: Dict[str, Any], band: Sequence[float]
           ) -> Optional[Dict[str, Any]]:
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
                            rc.camera_xyz[[0, 2]])
    order = ([case["occluder_type"]] if case.get("occluder_type") else []) + \
            [t for t in ranked if t != case.get("occluder_type")]

    for occluder_type in order:
        staged = put_in_front(rc, task["target_name"], occluder_type,
                              task["receptacle_name"],
                              min_occlusion=band[0], target_occlusion=band[1],
                              max_occlusion=band[2])
        if staged is not None:
            return {
                "scene": case["scene"],
                "start": case["start"],
                "instruction": task["instruction"],
                "predicate": task["predicate"],
                "subject_class": task["subject_class"],
                "object_class": task["object_class"],
                "target_name": task["target_name"],
                "receptacle_name": task["receptacle_name"],
                "occluder_type": occluder_type,
                "staged_occlusion": staged["occlusion"],
                "occluder_alternatives": [t for t in order
                                          if t != occluder_type],
            }
    print(f"    ! none of {order or '[]'} hides {task['target_name']}")
    return None


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
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--out", default="nvs_pilot/cases_frozen.json")
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

    frozen: List[Dict[str, Any]] = []
    for index, case in enumerate(raw, 1):
        print(f"[{index}/{len(raw)}] {case['scene']}  {case['instruction']}")
        rc = drive.open_scene(case["scene"], args.width, args.height, args.fov,
                              case["start"])
        try:
            record = verify(rc, case, (args.min_occlusion,
                                       args.target_occlusion,
                                       args.max_occlusion))
        except Exception as error:                       # noqa: BLE001
            print(f"    ! {type(error).__name__}: {error}")
            record = None
        finally:
            rc.stop()
        if record:
            frozen.append(record)
            swapped = ("" if record["occluder_type"] == case.get("occluder_type")
                       else f"  (swapped from {case.get('occluder_type')})")
            print(f"    ok  {record['occluder_type']} hides "
                  f"{record['staged_occlusion']:.0%}{swapped}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"predicate_families": sorted({c["predicate"]
                                                 for c in frozen}),
                   "occlusion_band": [args.min_occlusion,
                                      args.target_occlusion,
                                      args.max_occlusion],
                   "cases": frozen}, fh, indent=1)

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
