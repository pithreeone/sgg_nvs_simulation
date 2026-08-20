"""
find_cases.py -- the task instances the NVS-pointer experiment runs on.

One case is everything a runner needs to reproduce a question from
scratch: a scene, a pose to stand in, which instruction to follow, and which
object type to slide in front of the target to make the question hard.  Nothing
measured is stored -- the eval re-derives the task, the boxes and the occlusion
at the pose, because `SetObjectPoses` renumbers objectIds and THOR's physics
settle is not bit-reproducible.  A case is a POINTER TO A QUESTION, not a
record of an answer.

How a case is found:

  * A host is a VG150-nameable receptacle that currently holds something
    VG150-nameable.  Anything else cannot support "the X on the Y" in words the
    model can predict.
  * The robot stands 1.2-2.5 m off it (`drive_triplet_scene.standoff_spots`,
    the same band that file argues for) and aims at it, then `measure` +
    `build_tasks` produce every unambiguous instruction that view supports.
  * The occluder is the largest moveable object already near the target.  It is
    named by TYPE because `task_find.put_in_front` resolves the type to its
    nearest instance at stage time, after the ids have been renumbered.

The occlusion is STAGED rather than found.  `gen/find_walkaround.py` finds
poses where stock furniture already half-hides a target and is the honest
source, but it yields whatever the room happens to offer; staging gives a fixed
number of cases at a controlled occlusion, which is what a pilot needs.  Say so
in anything the pilot's numbers appear in.

How little the rooms offer, measured, because it is the whole justification for
staging: sweeping 30 stock scenes for a nameable object at 35-65% occlusion
with real depth separation behind it returned THREE cases, all of them the same
relation -- a chair behind a dining table (FloorPlan201/223/227, occlusion
0.55/0.62/0.53, recovering 2.2-2.9x the pixels at +-10 deg of orbit).  A case
list built that way would be one relation type repeated, which supports a claim
about chairs and tables and nothing else.

Cases are NOT filtered on whether EGTR then fails at the reference.  Filtering
there would select for exactly the outcome the experiment measures, and the
cases where it succeeds are the stratum that shows whether NVS breaks what
already worked.

    python find_cases.py --n 20 --out datasets/robot/cases.json
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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from robot import drive
from robot.drive_triplet_scene import measure, standoff_spots
from robot.robot_controller import _xz, horizon_towards, yaw_towards
from robot.task_find import build_tasks
from vg.vg150 import THOR_TO_VG150

#: Living rooms and bedrooms.  TASKS.md measures VG150 coverage at 77% of
#: objectTypes in FloorPlan201 against 63% in FloorPlan1 -- a kitchen's contents
#: are mostly unnameable (`microwave`, `fridge`, `toaster`, `spoon`, `knife` are
#: all absent from VG150), so a kitchen yields few instructions per view.
SCENES = ("FloorPlan201", "FloorPlan203", "FloorPlan209", "FloorPlan215",
          "FloorPlan219", "FloorPlan223", "FloorPlan227", "FloorPlan230",
          "FloorPlan301", "FloorPlan303")

#: How far from the target an occluder may already be standing.
OCCLUDER_RADIUS = 1.5

#: How far the occluder's underside may sit from the target's before the two
#: stop counting as resting on the same surface.  `task_find.put_in_front`
#: slides the occluder in x/z and KEEPS ITS OWN y -- which leaves it standing
#: only if it started on the same surface.  Without this test the largest
#: nearby moveable is a Chair or a SideTable at floor level, and sliding one of
#: those in front of a book lying on a dining table parks it UNDER the table,
#: hiding nothing.  Measured on FloorPlan203, that is exactly what the first
#: version picked.
SAME_SURFACE = 0.15


def size_of(entry: Dict[str, Any]) -> Dict[str, float]:
    return ((entry.get("axisAlignedBoundingBox") or {}).get("size") or {})


def underside(entry: Dict[str, Any]) -> float:
    box = entry.get("axisAlignedBoundingBox") or {}
    centre = box.get("center") or entry["position"]
    return float(centre.get("y", 0.0)) - float(size_of(entry).get("y", 0.0)) / 2.0


def horizontal_extent(entry: Dict[str, Any]) -> float:
    size = size_of(entry)
    return max(float(size.get("x", 0.0)), float(size.get("z", 0.0)))


def frontal_area(entry: Dict[str, Any]) -> float:
    """Rough silhouette a camera sees: how wide it is times how tall it stands."""
    return horizontal_extent(entry) * float(size_of(entry).get("y", 0.0))


#: How many legal box-centres an occluder needs before the case is worth
#: emitting.  One is not enough: a single legal placement means the geometry has
#: exactly one answer and no room to trade position against how much it covers,
#: and measured over the first list those cases came back hiding 0-8%.
MIN_LEGAL_PLACEMENTS = 8


def height_of(entry: Dict[str, Any]) -> float:
    return float(size_of(entry).get("y", 0.0))


#: How much bigger than the target an occluder's silhouette may be.  Only a
#: lower bound existed, which is the wrong half to leave open once the occluder
#: became an ENDPOINT of the instruction: a HousePlant's detection box spans the
#: sprawl of its leaves, so "behind the plant" is satisfiable by most of the
#: frame, and the rendered stops show exactly that -- a faucet paired with a
#: plant box covering the left third of the picture, called `behind`, accepted.
#: A cap keeps the landmark a landmark.
MAX_OCCLUDER_RATIO = 4.0


def rank_occluders(event, target: Dict[str, Any], receptacle: Dict[str, Any],
                   camera_xz: np.ndarray,
                   max_ratio: float = MAX_OCCLUDER_RATIO) -> List[str]:
    """
    Every workable occluder type, tallest first.

    A LIST rather than a winner because "fits on the surface with room to move"
    and "actually hides the target" are different questions, and only the second
    needs a render.  `find_cases` takes the first; `freeze_cases` walks the list
    until one measurably occludes, which is what turned 7 of the 9 attrition
    cases into runnable ones.

    The tallest moveable object on the target's surface that can actually be
    STAGED there -- verified with the same geometry the staging will use.

    Ranking by height, not by silhouette area, and the measurement that changed
    it: over the first case list a Vase succeeded three times out of three with
    108-205 legal placements and 87-89% occlusion, while a Laptop failed most
    attempts.  A vase is tall and narrow, which is exactly the shape that both
    fits on a table and covers what is behind it; a laptop is wide and low, so
    the places it fits are not the places that hide anything.  The old
    `frontal_area` key (width x height) preferred the laptop.

    The real filter is `legal_placements`, which is pure arithmetic and is the
    identical function `put_in_front` will run.  Checking only that the occluder
    FITS on the surface is a weaker question and it let through six cases that
    later turned out to have nowhere legal between the camera and the target.
    """
    from robot.task_find import legal_placements

    floor = underside(target)
    here = np.array([target["position"]["x"], target["position"]["z"]])
    entries = {o["name"]: o for o in event.metadata["objects"]}

    candidates = []
    for entry in event.metadata["objects"]:
        if entry["name"] in (target["name"], receptacle["name"]):
            continue
        if not (entry.get("moveable") or entry.get("pickupable")):
            continue
        if abs(underside(entry) - floor) > SAME_SURFACE:
            continue
        if math.dist(here, (entry["position"]["x"],
                            entry["position"]["z"])) > OCCLUDER_RADIUS:
            continue
        if not (frontal_area(target) <= frontal_area(entry)
                <= max_ratio * frontal_area(target)):
            continue
        # Tallest first; among equals the smaller footprint, which has more
        # places to stand and more room to be slid into the line of sight.
        candidates.append((-height_of(entry), horizontal_extent(entry), entry))

    ranked = []
    for _, _, entry in sorted(candidates, key=lambda c: (c[0], c[1])):
        room = legal_placements(entries, target, entry, receptacle, camera_xz)
        if room is not None and len(room["candidates"]) >= MIN_LEGAL_PLACEMENTS:
            ranked.append(entry["objectType"])
    # De-duplicated, order preserved: two instances of a type are one choice.
    return list(dict.fromkeys(ranked))


def pick_occluder(event, target: Dict[str, Any], receptacle: Dict[str, Any],
                  camera_xz: np.ndarray) -> Optional[str]:
    """The first workable occluder type, or None."""
    ranked = rank_occluders(event, target, receptacle, camera_xz)
    return ranked[0] if ranked else None


def hosts_of(event) -> List[Dict[str, Any]]:
    """Nameable receptacles that currently hold something nameable."""
    by_id = {o["objectId"]: o for o in event.metadata["objects"]}
    out = []
    for entry in event.metadata["objects"]:
        if entry["objectType"] not in THOR_TO_VG150 or not entry.get("receptacle"):
            continue
        held = [by_id[i] for i in (entry.get("receptacleObjectIds") or [])
                if i in by_id]
        if any(h["objectType"] in THOR_TO_VG150 for h in held):
            out.append(entry)
    # Biggest first: a dining table carries more instructions than a shelf, and
    # a case list that opens with the crowded surfaces reaches `--n` in fewer
    # `measure` sweeps -- each of which is an amodal Disable/Enable pass.
    return sorted(out, key=horizontal_extent, reverse=True)


def cases_in(rc, scene: str, fov: float, want: int, hosts_max: int,
             spots_max: int) -> List[Dict[str, Any]]:
    targets = sorted(o["name"] for o in rc.event.metadata["objects"]
                     if o["objectType"] in THOR_TO_VG150)
    found: List[Dict[str, Any]] = []
    seen_targets = set()

    for host in hosts_of(rc.event)[:hosts_max]:
        aim_xz = _xz(host["position"])
        for spot in standoff_spots(rc, aim_xz, aim_xz, (1.2, 2.5), spots_max):
            yaw = yaw_towards(spot, aim_xz)
            rc.teleport(position={"x": float(spot[0]),
                                  "y": float(rc.agent_position["y"]),
                                  "z": float(spot[1])},
                        yaw=yaw, horizon=0.0)
            horizon = horizon_towards(rc.camera_xyz,
                                      np.array([host["position"]["x"],
                                                host["position"]["y"],
                                                host["position"]["z"]]))
            rc.teleport(yaw=yaw, horizon=horizon)

            state = measure(rc, targets, fov)
            entries = {o["name"]: o for o in rc.event.metadata["objects"]}
            for task in build_tasks(state["objects"], entries):
                if task["target_name"] in seen_targets:
                    continue
                occluder = pick_occluder(rc.event, entries[task["target_name"]],
                                         entries[task["receptacle_name"]],
                                         rc.camera_xyz[[0, 2]])
                if occluder is None:
                    continue
                seen_targets.add(task["target_name"])
                found.append({
                    "scene": scene,
                    "start": f"{spot[0]:.4f},{spot[1]:.4f},{yaw:.2f},{horizon:.2f}",
                    "instruction": task["instruction"],
                    "subject_class": task["subject_class"],
                    "object_class": task["object_class"],
                    "target_name": task["target_name"],
                    "receptacle_name": task["receptacle_name"],
                    "occluder_type": occluder,
                    "reference_occlusion": task["target_occlusion"],
                })
                print(f"  + {task['instruction']:36s} target "
                      f"{task['target_name']:22s} occluder {occluder}")
                if len(found) >= want:
                    return found
    return found


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--per-scene", type=int, default=3,
                    help="cap per floor plan, so 20 cases are not 20 views of "
                         "one dining table")
    ap.add_argument("--scenes", nargs="*", default=list(SCENES))
    ap.add_argument("--hosts", type=int, default=3, help="receptacles tried per scene")
    ap.add_argument("--spots", type=int, default=2, help="standoff poses tried per host")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--out", default="datasets/robot/cases.json")
    args = ap.parse_args(argv)

    cases: List[Dict[str, Any]] = []
    for scene in args.scenes:
        if len(cases) >= args.n:
            break
        print(f"\n{scene}")
        rc = drive.open_scene(scene, args.width, args.height, args.fov)
        try:
            want = min(args.per_scene, args.n - len(cases))
            cases.extend(cases_in(rc, scene, args.fov, want, args.hosts,
                                  args.spots))
        finally:
            rc.stop()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"cases": cases}, fh, indent=1)
    scenes = sorted({c["scene"] for c in cases})
    print(f"\n{len(cases)} cases over {len(scenes)} scenes -> {args.out}")
    if len(cases) < args.n:
        print(f"  ! wanted {args.n}; raise --hosts/--spots or add scenes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
