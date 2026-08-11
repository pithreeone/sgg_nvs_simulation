"""
task_find.py -- L1 of the referring-expression task: "Find the book on the table".

One frame, one instruction, one question: does EGTR's top-K scene graph contain
that triplet AND is its subject box on the object the instruction actually meant?

The grounding is the whole point.  Asked only whether `book on table` appears
among the predictions, EGTR says yes almost always -- in a dining room nine of
its top ten triplets share one `table` query -- while saying nothing about WHICH
book.  So a hit here needs both: the class triple must match, and the predicted
subject must land on the instance THOR says the instruction refers to.

How a task is made (THOR is the examiner, not the model):

  * The target must be nameable in VG150 and so must its receptacle, or the
    instruction cannot be written in words the model can predict.
  * `parentReceptacles` is MULTIPLE -- one book reports Chair, Chair and
    DiningTable at once -- so the receptacle is the SMALLEST of them, the most
    direct support.  Without that rule `the book on the table` and `the book on
    the chair` are both true and the task has no answer.
  * The (target class, receptacle class) pair must be unique in view.  Two books
    on two tables make the instruction ambiguous, and grading a model against an
    ambiguous instruction measures nothing.  That ambiguity is deliberate in L2,
    where the distractor is the point; here it disqualifies the task.
  * Only `on`.  Measured in TASKS.md, EGTR reaches 44% recall on `on` and ~0 on
    the other families, so an instruction built on `behind` fails for reasons
    that have nothing to do with the robot or the viewpoint.

Scoring:

  * The class triple must match exactly (`--any-predicate` relaxes it, and is
    worth looking at as a diagnostic: it separates "wrong relation" from
    "did not find the object").
  * The subject box must reach `--iou` against the target, scored as the better
    of its amodal and its visible box.  EGTR draws what it can see, the
    annotation knows the whole object, and which is fairer depends on how hidden
    the target is -- so take the max, as `build_occlusion_dataset.py` argues.
  * The object end is matched by class only.  A dining table fills a third of
    the frame; requiring IoU on it would pass on geometry alone.
  * Rank, never score.  The top triplet scores ~0.01 here because the ranking is
    `rel x s_subject x s_object`, and that product is not comparable between
    frames.  So the result is hit@K, K in {10, 20, 100}.

    python task_find.py --scene FloorPlan203
    python task_find.py --resume driveable/fp203/scenario.json --out tasks/fp203
    python task_find.py --scene FloorPlan203 --negatives   # false-positive control
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from robot import drive
from robot.drive_triplet_scene import measure
from vg.vg150 import THOR_TO_VG150

#: The K's reported, and therefore how many triplets EGTR is asked for.
KS = (10, 20, 100)


def iou(a: Sequence[float], b: Optional[Sequence[float]]) -> float:
    if b is None:
        return 0.0
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    overlap = (x1 - x0) * (y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def best_iou(box: Sequence[float], target: Dict[str, Any]) -> float:
    """IoU against whichever of the target's two boxes is kinder to the model."""
    return max(iou(box, target.get("bbox_amodal")),
               iou(box, target.get("bbox_visible")))


#: Support tolerances, metres.  `SUPPORT_GAP` is how far a resting object's
#: underside may sit from the surface it rests on -- measured on FloorPlan203 the
#: agreement is exact (book, vase and plate all bottom out at 0.756, the
#: DiningTable's top), so this only absorbs the odd loose axis-aligned box.
SUPPORT_GAP = 0.10
#: How far outside the host's footprint the resting object's centre may sit.
SUPPORT_MARGIN = 0.05


def extent(entry: Dict[str, Any]) -> Optional[Tuple[Dict[str, float],
                                                    Dict[str, float]]]:
    box = entry.get("axisAlignedBoundingBox")
    if not box or not box.get("size"):
        return None
    return box["center"], box["size"]


def rests_on(target: Dict[str, Any], host: Dict[str, Any]) -> bool:
    """
    Does `target` actually sit on `host`?

    `parentReceptacles` cannot answer this.  It is a containment test against the
    receptacle's volume, so a chair tucked under a dining table reports the
    TABLE as its child and the table reports the chair as its parent -- which is
    why the repo's own `support_relations` emits `table on chair`, and why the
    first version of this file generated "Find the book on the chair" for a book
    lying on the table.  A chair's bounding box tops out at its BACK (1.033 m on
    FloorPlan203), well above the tabletop the book is on (0.756 m), so
    comparing the underside to the surface separates them cleanly.
    """
    t, h = extent(target), extent(host)
    if t is None or h is None:
        return False
    (t_centre, t_size), (h_centre, h_size) = t, h
    bottom = t_centre["y"] - t_size["y"] / 2.0
    top = h_centre["y"] + h_size["y"] / 2.0
    if abs(bottom - top) > SUPPORT_GAP:
        return False
    for axis in ("x", "z"):
        half = h_size[axis] / 2.0 + SUPPORT_MARGIN
        if abs(t_centre[axis] - h_centre[axis]) > half:
            return False
    return True


def surface_height(entry: Dict[str, Any]) -> float:
    box = extent(entry)
    return -float("inf") if box is None else box[0]["y"] + box[1]["y"] / 2.0


def add_distractor(rc, object_type: str, host_type: str, seed: int = 1
                   ) -> Optional[str]:
    """
    L2: put a second `object_type` on a `host_type`, so the relation disambiguates.

    `InitialRandomSpawn` duplicates -- it cannot create from nothing, so the
    scene must already hold one.  It also relocates ~30 unrelated objects as a
    side effect, which is why the originals are snapshotted and restored around
    it; that is `occlusion_pipeline.arrange`'s own protocol.
    """
    import gen.occlusion_pipeline as P

    before = P.snapshot(rc.event)
    rc.controller.step(action="InitialRandomSpawn", randomSeed=seed,
                       forceVisible=True, placeStationary=True,
                       numPlacementAttempts=25,
                       numDuplicatesOfType=[{"objectType": object_type,
                                             "count": 2}])
    P.restore(rc.controller, before)
    clones = [o["name"] for o in rc.controller.last_event.metadata["objects"]
              if o["name"] not in before and o["objectType"] == object_type]
    if not clones:
        print(f"  ! could not duplicate {object_type}")
        return None

    clone = clones[0]
    if host_type.lower() == "floor":
        return place_on_floor(rc, clone)

    hosts = [o for o in rc.controller.last_event.metadata["objects"]
             if o["objectType"] == host_type and o.get("receptacle")]
    # Nearest hosts first, and within a host the LOWEST candidate points first:
    # a chair's spawn coordinates span its whole box, and the seat is the low
    # ones.  Both lists are capped because each attempt is a THOR step and an
    # exhaustive search over six chairs ran past two minutes.
    for host in sorted(hosts, key=lambda o: o["distance"])[:3]:
        points = sorted(P.spawn_points(rc.controller, host["objectId"]),
                        key=lambda p: p["y"])[:12]
        for point in points:
            entry = P.by_name(rc.controller.last_event, clone)
            event = rc.controller.step(action="PlaceObjectAtPoint",
                                       objectId=entry["objectId"],
                                       position=point)
            if not event.metadata["lastActionSuccess"]:
                continue
            # `PlaceObjectAtPoint` reporting success is not enough.  A chair's
            # spawn coordinates come off its whole bounding box, which tops out
            # at the BACK, so the duplicate was released above seat height and
            # settled on the dining table instead -- and the run then produced
            # two `book on table` tasks and dropped both as ambiguous.  So the
            # placement is accepted only if the same support test the tasks are
            # built from agrees the book is on the host.
            rc.event = event
            placed = P.by_name(event, clone)
            if placed and rests_on(placed, P.by_name(event, host["name"])):
                print(f"  added {clone} on {host['name']}")
                return clone
    print(f"  ! nothing on a {host_type} held the duplicate "
          f"(placements landed elsewhere)")
    return None


def place_on_floor(rc, clone: str, near: float = 2.0) -> Optional[str]:
    """
    Put the duplicate on the floor in front of the robot, and check it renders.

    The floor is not a VG150 class -- neither `vg150.THOR_TO_VG150` nor EGTR's
    150 names contain `floor` -- so this object can never be the OBJECT end of a
    predicted triplet, and it gets no instruction of its own.  That is fine and
    is the point of L2: it exists to make "the book" ambiguous, so that "on the
    table" is doing the referring.  Success here is only that it lands in view.
    """
    import math

    import gen.occlusion_pipeline as P

    from robot.drive_triplet_scene import agent_masks

    position = rc.agent_position
    yaw = math.radians(rc.agent_yaw)
    reachable = rc.get_reachable_positions()
    names_by_id = {o["objectId"]: o["name"]
                   for o in rc.controller.last_event.metadata["objects"]}

    # Sweep the distance ahead rather than trusting one.  The camera is pitched
    # down when it is looking at a tabletop, so how far along the floor is still
    # in frame depends on that pitch: at horizon 45 deg, 1.0-1.5 m renders and
    # 2.0 m falls outside the view entirely.
    for distance in (near * f for f in (0.5, 0.625, 0.75, 0.875, 1.0)):
        ahead = np.array([position["x"] + math.sin(yaw) * distance,
                          position["z"] + math.cos(yaw) * distance])
        order = np.argsort(np.linalg.norm(reachable - ahead.reshape(1, 2),
                                         axis=1))
        for index in order[:3]:
            spot = reachable[index]
            entry = P.by_name(rc.controller.last_event, clone)
            event = rc.controller.step(
                action="PlaceObjectAtPoint", objectId=entry["objectId"],
                position={"x": float(spot[0]), "y": 0.1, "z": float(spot[1])})
            if not event.metadata["lastActionSuccess"]:
                continue
            rc.event = event
            if clone in agent_masks(event, names_by_id):
                print(f"  added {clone} on the floor {distance:.2f} m ahead "
                      f"-- distractor, no instruction of its own")
                return clone
    print("  ! no floor spot put the duplicate in view")
    return None


def aabb(entry: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The axis-aligned box as (low corner, high corner)."""
    box = extent(entry)
    if box is None:
        return None
    centre, size = box
    half = np.array([size["x"], size["y"], size["z"]]) / 2.0
    middle = np.array([centre["x"], centre["y"], centre["z"]])
    return middle - half, middle + half


def boxes_overlap(a: Tuple[np.ndarray, np.ndarray],
                  b: Tuple[np.ndarray, np.ndarray], slack: float = 0.005
                  ) -> bool:
    """Do two boxes share volume?  `slack` lets them touch without counting."""
    return bool(np.all(np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]) > slack))


#: The surface is searched as a GRID, not along the camera-to-target ray.  The
#: ray was the first attempt and it failed for a reason worth keeping: a target
#: worth instructing about tends to sit near the camera-side EDGE of its table --
#: that is what makes it visible -- so the strip of ray between the camera and
#: the target is mostly off the table.  Measured over the ten pilot tasks, 67 of
#: 70 ray positions were rejected as off-surface in four different scenes, and on
#: FloorPlan303 all 70 were, because a 0.4 m Laptop cannot sit anywhere on a
#: 0.5 m SideTable with its footprint fully supported.  The tabletop has legal
#: places the ray does not pass through.
STAGE_GRID = 0.04
#: How far in front of the target a candidate may sit, along the line of sight,
#: and how far off that line.  Beyond these it is not between the camera and the
#: target in any useful sense.
STAGE_GAP = (0.02, 0.70)
STAGE_LATERAL = 0.30
#: Legal placements are cheap to enumerate and expensive to measure, so only a
#: sample is rendered.  The sample is SPREAD across the ordering rather than
#: taken from the front: the candidate list runs from most-occluding to least,
#: so the first 30 are all near-total occlusions and a target in the middle of
#: the range would never be measured at all.
STAGE_RENDERS = 30
#: The occlusion band a staged case aims for.  `put_in_front` keeps the legal
#: placement whose measured occlusion is NEAREST `TARGET`, not the largest --
#: taking the largest drove 36 of 60 cases past 0.75 and 17 of them to a
#: complete disappearance, which piles the whole list into one extreme and
#: leaves no middle band to compare against.  A target that cannot be seen at
#: all is a legitimate regime (a single frame provably cannot answer it) but it
#: should be a stratum, not the default.
MIN_STAGED_OCCLUSION = 0.10
TARGET_STAGED_OCCLUSION = 0.50
MAX_STAGED_OCCLUSION = 0.90


def host_of(entries: Dict[str, Any], target: Dict[str, Any]) -> Optional[str]:
    """The surface `target` really rests on -- `build_tasks`' own rule."""
    by_object_id = {e["objectId"]: n for n, e in entries.items()}
    hosts = [by_object_id[i] for i in (target.get("parentReceptacles") or [])
             if i in by_object_id]
    hosts = [h for h in hosts if rests_on(target, entries[h])]
    if not hosts:
        return None
    return max(hosts, key=lambda h: surface_height(entries[h]))


def legal_placements(entries: Dict[str, Any], target: Dict[str, Any],
                     occluder: Dict[str, Any], receptacle: Dict[str, Any],
                     camera_xz: np.ndarray, grid: float = STAGE_GRID
                     ) -> Optional[Dict[str, Any]]:
    """
    Where `occluder` may legally stand on `receptacle` to hide `target`.

    Pure arithmetic -- no THOR, no render -- so `find_cases` can ask the SAME
    question at discovery time that `put_in_front` will ask at staging time.
    Keeping one definition of "legal" is the point: the first version checked
    only that the occluder FIT on the surface, which is a different question
    from whether it fits SOMEWHERE BETWEEN THE CAMERA AND THE TARGET, and cases
    that passed the first test failed the second in the run.

    Returns None if the occluder cannot fit on the surface at all, otherwise the
    candidate box-centres ordered by decreasing expected occlusion, plus what
    `put_in_front` needs to turn one into a pose.
    """
    surface, target_box, occluder_box = (aabb(receptacle), aabb(target),
                                         aabb(occluder))
    if surface is None or target_box is None or occluder_box is None:
        return None
    surface_top = float(surface[1][1])
    half = (occluder_box[1] - occluder_box[0])[[0, 2]] / 2.0
    low, high = surface[0][[0, 2]] + half, surface[1][[0, 2]] - half
    if np.any(high < low):
        return None                       # does not fit on this surface at all

    centre_xz = ((occluder_box[0] + occluder_box[1]) / 2.0)[[0, 2]]
    target_xz = np.array([target["position"]["x"], target["position"]["z"]])
    forward = target_xz - camera_xz
    forward = forward / max(float(np.linalg.norm(forward)), 1e-6)
    left_of = np.array([-forward[1], forward[0]])

    # Only things resting on THIS surface can be hit; a chair's box spans floor
    # to backrest and encloses tabletop airspace it does not occupy.
    neighbours = [(n, b) for n, b in
                  ((n, aabb(e)) for n, e in entries.items()) if b is not None
                  and n not in (target["name"], receptacle["name"],
                                occluder["name"])
                  and abs(float(b[0][1]) - surface_top) <= SUPPORT_GAP]

    candidates = []
    for cx in np.arange(low[0], high[0] + 1e-6, grid):
        for cz in np.arange(low[1], high[1] + 1e-6, grid):
            delta = target_xz - np.array([cx, cz])
            along = float(delta @ forward)
            lateral = abs(float(delta @ left_of))
            if not STAGE_GAP[0] <= along <= STAGE_GAP[1]:
                continue
            if lateral > STAGE_LATERAL:
                continue
            shift = np.array([cx - centre_xz[0], surface_top
                              - float(occluder_box[0][1]), cz - centre_xz[1]])
            here = (occluder_box[0] + shift, occluder_box[1] + shift)
            if boxes_overlap(here, target_box):
                continue
            if any(boxes_overlap(here, b) for _, b in neighbours):
                continue
            candidates.append((lateral, along, float(cx), float(cz)))

    # Nearest the line of sight first, then nearest the target: the order of
    # decreasing expected occlusion, so a render budget is spent where it pays.
    candidates.sort()
    return {"candidates": candidates,
            "pivot": np.array([occluder["position"]["x"],
                               occluder["position"]["z"]]) - centre_xz,
            "lift": surface_top - float(occluder_box[0][1]),
            "footprint": (float(2 * half[0]), float(2 * half[1]))}


def spread(items: Sequence[Any], count: int) -> List[Any]:
    """`count` items spanning the whole list, both ends included."""
    if len(items) <= count:
        return list(items)
    step = (len(items) - 1) / (count - 1)
    return [items[i] for i in
            dict.fromkeys(round(k * step) for k in range(count))]


def behind_task(task: Dict[str, Any], occluder_name: str,
                objects: Sequence[Dict[str, Any]], entries: Dict[str, Any]
                ) -> Optional[Dict[str, Any]]:
    """
    Rewrite an `on` task as the `behind` relation the staging just created.

    No new geometry.  `put_in_front` slides the occluder along the CAMERA-TO-
    TARGET ray, so once it is placed the scene contains, by construction and
    from this camera, "target behind occluder".  The subject, its box, its
    distractors and the grading are all unchanged; only the object endpoint and
    the predicate move.

    WHY BOTHER.  `on` is a viewpoint-INVARIANT fact -- a book on a table is on
    it from everywhere -- and `channels.py` argues that only viewpoint-dependent
    facts can be informed by a second viewpoint, which is why its predCond set
    ships as `behind / in front of / above / under / near` with `on` excluded.
    Measuring a multi-view mechanism on `on` asks it to help where it cannot.

    THE COST, stated because it decides which experiment may use this: `behind`
    holds FROM A POSE.  The fusion experiment grades at the reference pose and
    is fine.  A walking experiment is not -- move far enough and the target
    stops being behind the occluder, so the instruction stops being true of the
    scene it was written about.

    Returns None when the rewrite would be ambiguous: the occluder has to be
    VG150-nameable, has to be a different class from the target, and has to be
    the only instance of its class in view, or "the vase" names two things.
    """
    from vg.vg150 import THOR_TO_VG150

    occluder = entries.get(occluder_name)
    if occluder is None:
        return None
    object_class = THOR_TO_VG150.get(occluder["objectType"])
    if object_class is None:
        return None
    if object_class == task["subject_class"]:
        return None
    # `objects` is `measure`'s list, the same one `build_tasks` consumes.
    twins = [o for o in objects
             if o.get("vg150_class") == object_class
             and o["name"] != occluder_name]
    if twins:
        return None
    return {**task,
            "instruction": f"Find the {task['subject_class']} behind the "
                           f"{object_class}",
            "predicate": "behind",
            "object_class": object_class,
            "receptacle_name": occluder_name,
            "on_instruction": task["instruction"]}


def in_front_task(task: Dict[str, Any], occluder_name: str,
                  objects: Sequence[Dict[str, Any]], entries: Dict[str, Any]
                  ) -> Optional[Dict[str, Any]]:
    """
    The same staged geometry as `behind_task`, said the other way round.

    `put_in_front` leaves the occluder between the camera and the target, so
    both "target behind occluder" and "occluder in front of target" are true of
    the scene.  Which one to ask is not a matter of taste: measured on ds4,
    EGTR reaches R@100 1.5% on `behind` against 10.2% on `in front of`, seven
    times better, and 38.4% against 20.8% under the human convention.  It
    barely predicts one and predicts the other about as well as `near`.

    THE INSTANCE UNDER TEST MOVES WITH THE SUBJECT.  "Find the vase in front of
    the paper" asks for the vase, so the vase is what has to be grounded, and
    `target_name` becomes the occluder.  Grading the paper here -- the thing the
    sentence uses only as a landmark -- would be scoring a question nobody
    asked.  The consequence is worth stating in anything these numbers appear
    in: the sought object is NOT occluded, so this measures whether a novel view
    helps establish a RELATION whose landmark is hidden, not whether it reveals
    a hidden object.  The occlusion still makes the relation hard to see; it no
    longer makes the subject hard to see.

    Returns None when the rewrite would be ambiguous -- the occluder must be
    VG150-nameable, a different class from the landmark, and the only instance
    of its class in view, or "the vase" names two things.
    """
    from vg.vg150 import THOR_TO_VG150

    occluder = entries.get(occluder_name)
    if occluder is None:
        return None
    subject_class = THOR_TO_VG150.get(occluder["objectType"])
    if subject_class is None or subject_class == task["subject_class"]:
        return None
    if any(o.get("vg150_class") == subject_class and o["name"] != occluder_name
           for o in objects):
        return None
    return {**task,
            "instruction": f"Find the {subject_class} in front of the "
                           f"{task['subject_class']}",
            "predicate": "in front of",
            "subject_class": subject_class,
            "object_class": task["subject_class"],
            # The graded instance is the SUBJECT, so both the box the grader
            # measures and the distractor set follow it to the occluder.
            "target_name": occluder_name,
            "landmark_name": task["target_name"],
            "receptacle_name": task["target_name"],
            "distractors": [],
            "on_instruction": task["instruction"]}


def stage_at(rc, target_name: str, occluder_name: str,
             position: Dict[str, float],
             snapshot: Optional[Sequence[Dict[str, Any]]] = None
             ) -> Optional[Dict[str, Any]]:
    """
    Replay a staging `put_in_front` already found, instead of searching again.

    WHY THIS EXISTS.  A frozen case used to record the occluder's TYPE and the
    occlusion band, and every run re-ran the grid search to find a placement
    inside that band.  The search is not deterministic across runs -- it starts
    from whatever the physics settle left and keeps the candidate NEAREST the
    target occlusion, so two runs of the same case land the occluder a few
    centimetres apart and the target is hidden by, say, 43% instead of 39%.
    Measured: repeats of one case returned grounded ranks of 19 / 18 / 17, and
    the single-view total over 59 cases moved between 6 and 7.  That is the same
    size as the effect the experiment is trying to measure.

    Replay is exact only if the WHOLE moveable scene is restored, which is why
    `snapshot` exists and why passing just the occluder is not enough.  THOR
    settles physics on load, that settle is not reproducible, and every other
    moveable object -- including the TARGET -- lands slightly differently each
    time.  Measured on FloorPlan211 across three runs that replayed only the
    occluder: the occlusion held at 39% but the target's own visible box moved,
    IoU 0.846 / 0.810 / 0.810, and the single-view rank went 11 / 15 / 13.

    With the snapshot, `SetObjectPoses` applies no gravity and no collision
    resolution, so restoring it reproduces the scene rather than approximating
    it.  Poses are world coordinates keyed by NAME, so the objectId renumbering
    `SetObjectPoses` causes does not matter.

    What is NOT replayed is any measurement.  The occlusion, the boxes and the
    grading are re-derived here exactly as before; only the search is skipped.
    """
    from gen.occlusion import by_name, set_pose, set_poses, visible_pixels

    if by_name(rc.event, target_name) is None:
        print(f"  ! {target_name} is not in this scene")
        return None
    if not any(o["name"] == occluder_name for o in rc.event.metadata["objects"]):
        print(f"  ! occluder {occluder_name} is not in this scene")
        return None

    if snapshot:
        # Baseline first, with the occluder still wherever the scene put it --
        # the number only means "how much of the target this occluder hides" if
        # it is measured before the occluder moves into the sightline.
        home = [p for p in snapshot if p["objectName"] != occluder_name]
        here = next(p for p in rc_snapshot(rc) if p["objectName"] == occluder_name)
        if not set_poses(rc.controller, home + [here]):
            return None
        rc.controller.step(action="Done")
        rc.event = rc.controller.last_event
        baseline = visible_pixels(rc.event, target_name)
        if not set_poses(rc.controller, list(snapshot)):
            return None
    else:
        baseline = visible_pixels(rc.event, target_name)
        set_pose(rc.controller, occluder_name,
                 (float(position["x"]), float(position["y"]),
                  float(position["z"])))
    # The RGB of a `SetObjectPoses` event lags its own masks; see `put_in_front`.
    rc.controller.step(action="Done")
    rc.event = rc.controller.last_event

    remaining = visible_pixels(rc.event, target_name)
    occlusion = round(max(0.0, 1.0 - remaining / baseline), 3) if baseline else 1.0
    print(f"  {occluder_name} replayed in front of {target_name}: "
          f"occlusion {baseline} -> {remaining} px ({occlusion:.0%} hidden)")
    return {"occluder": occluder_name, "position": dict(position),
            "baseline_px": baseline, "remaining_px": remaining,
            "occlusion": occlusion, "replayed": True,
            "pinned": bool(snapshot)}


def rc_snapshot(rc):
    """This scene's moveable poses right now."""
    from gen.occlusion import pose_snapshot

    return pose_snapshot(rc.event)


def put_in_front(rc, target_name: str, occluder_type: str,
                 receptacle_name: Optional[str] = None,
                 grid: float = STAGE_GRID,
                 min_occlusion: float = MIN_STAGED_OCCLUSION,
                 target_occlusion: float = TARGET_STAGED_OCCLUSION,
                 max_occlusion: float = MAX_STAGED_OCCLUSION
                 ) -> Optional[Dict[str, Any]]:
    """
    Slide an object along the camera-to-target ray until it hides the target,
    WITHOUT putting it through the target or over the edge of the table.

    `gen.occlusion.occlude` does this by bisection but places the occluder at
    `depth_fraction` ALONG THE RAY -- in mid-air for a target lying on a table --
    and `SetObjectPoses` applies no gravity, so it would float.  Keeping the
    occluder's own `y` and moving only x/z was the first fix and it is not
    enough, because nothing then stopped the occluder leaving the surface
    sideways or ending up inside the target.  Measured over the ten pilot tasks,
    the naive placement was defective in 8 of 10: five put 5-11% of the
    occluder's volume inside the target, six left a third to two thirds of its
    footprint hanging off the table, and on FloorPlan303 a Laptop finished 9.2 cm
    above the desk with nothing under it.  A render nobody believes cannot
    support a claim about what a detector failed to see -- the same argument
    `gen/find_walkaround.py` makes about placement in general.

    So geometry is a REJECTION test, not a term to trade off against occlusion:

      * the occluder's underside is set ON the surface (its pivot is not its
        box centre, so the drop is measured from its own box)
      * its footprint must lie entirely over the receptacle
      * its box must not intersect the target's
      * its box must not intersect anything else RESTING ON THE SAME SURFACE

    That last restriction is deliberate.  An axis-aligned box is a bad proxy for
    a dining chair -- it spans floor to backrest and encloses the tabletop
    airspace the chair does not occupy -- so testing against every object in the
    room reports 96% "overlap" between a vase on the table and a chair beside
    it.  Restricting the test to objects whose underside is near the surface
    keeps it to things the occluder could really hit.

    Candidates are filtered geometrically FIRST, in arithmetic, and only the
    survivors are rendered: a rejected pose costs nothing but a subtraction.

    Reports the visible pixels before and after at the SAME pose, which is the
    causal test: it says the occluder is what hides the target rather than the
    target happening to be invisible.
    """
    from gen.occlusion import by_name, set_pose, visible_pixels

    target = by_name(rc.event, target_name)
    occluders = [o for o in rc.event.metadata["objects"]
                 if o["objectType"] == occluder_type and o["name"] != target_name
                 and (o.get("moveable") or o.get("pickupable"))]
    if target is None or not occluders:
        print(f"  ! no moveable {occluder_type} to hide {target_name}")
        return None

    entries = {o["name"]: o for o in rc.event.metadata["objects"]}
    if receptacle_name is None:
        receptacle_name = host_of(entries, target)
        if receptacle_name is None:
            print(f"  ! {target_name} rests on nothing nameable; cannot check "
                  f"that an occluder would stay on the surface")
            return None

    baseline = visible_pixels(rc.event, target_name)
    camera = rc.event.metadata["cameraPosition"]
    camera_xz = np.array([camera["x"], camera["z"]])
    target_xz = np.array([target["position"]["x"], target["position"]["z"]])

    # Every instance of the type, nearest first: an instance that cannot fit on
    # this surface says nothing about the next one, and `find_cases` chose the
    # TYPE from whichever instance it happened to look at.
    for occluder in sorted(occluders, key=lambda o: math.dist(
            (o["position"]["x"], o["position"]["z"]), tuple(target_xz))):
        home = dict(occluder["position"])
        room = legal_placements(entries, target, occluder,
                                entries[receptacle_name], camera_xz, grid)
        if room is None:
            print(f"  ! {occluder['name']} does not fit on {receptacle_name}")
            continue
        candidates, pivot, lift = (room["candidates"], room["pivot"],
                                   room["lift"])
        if not candidates:
            continue

        best, seen_band = None, []
        for lateral, along, cx, cz in spread(candidates, STAGE_RENDERS):
            x, z = np.array([cx, cz]) + pivot
            if not set_pose(rc.controller, occluder["name"],
                            (x, home["y"] + lift, z)):
                continue
            rc.event = rc.controller.last_event
            left = visible_pixels(rc.event, target_name)
            occlusion = 1.0 - left / baseline if baseline else 1.0
            seen_band.append(occlusion)
            if not min_occlusion <= occlusion <= max_occlusion:
                continue
            # Nearest the target band, not the most hidden.
            miss = abs(occlusion - target_occlusion)
            if best is None or miss < best["miss"]:
                best = {"occluder": occluder["name"], "gap": round(along, 3),
                        "sideways": round(lateral, 3),
                        "occlusion": round(occlusion, 3), "miss": miss,
                        "baseline_px": baseline, "remaining_px": left,
                        "receptacle": receptacle_name, "home": home,
                        "legal": len(candidates),
                        "position": {"x": float(x), "y": home["y"] + lift,
                                     "z": float(z)}}
        set_pose(rc.controller, occluder["name"],
                 (home["x"], home["y"], home["z"]))
        rc.event = rc.controller.last_event
        if best is None:
            span = (f"{min(seen_band):.0%}-{max(seen_band):.0%}"
                    if seen_band else "nothing")
            print(f"  ! {occluder['name']} hides {span} of {target_name}, "
                  f"never inside [{min_occlusion:.0%}, {max_occlusion:.0%}]")
            continue
        best.pop("miss")

        set_pose(rc.controller, best["occluder"],
                 (best["position"]["x"], best["position"]["y"],
                  best["position"]["z"]))
        # The RGB of a `SetObjectPoses` event LAGS ITS OWN MASKS.  Measured on
        # FloorPlan203: the instance mask already reports the book at 2819
        # visible pixels while the rendered frame still shows the vase at its
        # old position -- MAE 2.59 against the next frame, 20.4% of pixels
        # different.  One more step reconciles them.  Anything that hands this
        # frame to a detector without it is grading the scene BEFORE staging;
        # that is worth a rank of 3 instead of 67 on this very case.
        rc.controller.step(action="Done")
        rc.event = rc.controller.last_event
        best["remaining_px"] = visible_pixels(rc.event, target_name)
        print(f"  {best['occluder']} placed {best['gap']:.2f} m in front of "
              f"{target_name} ({best['sideways']:.2f} m off the sightline, on "
              f"{receptacle_name}): occlusion {best['baseline_px']} -> "
              f"{best['remaining_px']} px ({best['occlusion']:.0%} hidden), "
              f"{best['legal']} legal placements")
        return best

    print(f"  ! nothing of type {occluder_type} can be staged in front of "
          f"{target_name} on {receptacle_name}")
    return None


def build_tasks(objects: Sequence[Dict[str, Any]], entries: Dict[str, Any]
                ) -> List[Dict[str, Any]]:
    """
    Every unambiguous "the X on the Y" the view supports.

    `objects` is what `measure` reports, so it is already restricted to what is
    in view and big enough to be worth naming (its `min_reference_px` floor) and
    to what is not hidden past `MAX_OCCLUSION`.
    """
    in_view = {o["name"]: o for o in objects if o["vg150_class"]}
    by_object_id = {entries[n]["objectId"]: n for n in in_view if n in entries}

    proposals = []
    for name, target in in_view.items():
        parents = [by_object_id[i] for i in (target.get("parent_receptacles") or [])
                   if i in by_object_id]
        parents = [p for p in parents if in_view[p]["vg150_class"]
                   and in_view[p]["vg150_class"] != target["vg150_class"]]
        parents = [p for p in parents if rests_on(entries[name], entries[p])]
        if not parents:
            continue
        # Highest surface wins: a plate on a book on a table rests on the book.
        parent = max(parents, key=lambda p: surface_height(entries[p]))
        proposals.append({
            "instruction": f"Find the {target['vg150_class']} on the "
                           f"{in_view[parent]['vg150_class']}",
            "subject_class": target["vg150_class"],
            "predicate": "on",
            "object_class": in_view[parent]["vg150_class"],
            "target_name": name,
            "receptacle_name": parent,
            "target_occlusion": target["occlusion"],
            "receptacle_occlusion": in_view[parent]["occlusion"],
            "target_box_amodal": target["bbox_amodal"],
            "target_box_visible": target["bbox_visible"],
            # Other instances of the SAME class in view.  L2 lives here: with a
            # second book on the floor, `book on table` grounded on that book is
            # the failure the instruction exists to catch, and it is invisible
            # to a class-level check.
            "distractors": [
                {"name": other["name"], "occlusion": other["occlusion"],
                 "bbox_amodal": other["bbox_amodal"],
                 "bbox_visible": other["bbox_visible"]}
                for other in in_view.values()
                if other["name"] != name
                and other["vg150_class"] == target["vg150_class"]],
        })

    # Uniqueness at CLASS level, which is the level the instruction speaks at.
    counts: Dict[Tuple[str, str], int] = {}
    for task in proposals:
        key = (task["subject_class"], task["object_class"])
        counts[key] = counts.get(key, 0) + 1
    unique, ambiguous = [], []
    for task in proposals:
        key = (task["subject_class"], task["object_class"])
        (unique if counts[key] == 1 else ambiguous).append(task)
    for task in ambiguous:
        print(f"  (dropped, ambiguous: {task['instruction']} -- another "
              f"instance shares the class pair)")
    return unique


def grade(task: Dict[str, Any], triplets: Sequence[Dict[str, Any]],
          objects: Sequence[Dict[str, Any]], threshold: float,
          any_predicate: bool) -> Dict[str, Any]:
    """
    Where in the ranking the instruction is answered, and whether it is grounded.

    Two ranks are reported because they fail differently:

      `class_rank`     the first triplet whose class triple matches -- the model
                       said the right words.
      `grounded_rank`  the first that ALSO puts its subject on the right
                       instance -- the model meant the right thing.

    A `class_rank` with no `grounded_rank` is the interesting failure: the graph
    contains `book on table` and it is a different book.
    """
    target = {"bbox_amodal": task["target_box_amodal"],
              "bbox_visible": task["target_box_visible"]}
    class_rank = grounded_rank = distractor_rank = None
    best = best_distractor = 0.0
    matches = []
    for rank, rel in enumerate(triplets, 1):
        if rel["subject"] != task["subject_class"]:
            continue
        if rel["object"] != task["object_class"]:
            continue
        if not any_predicate and rel["predicate"] != task["predicate"]:
            continue
        overlap = best_iou(rel["subject_box"], target)
        best = max(best, overlap)
        # Highest IoU against any same-class distractor, so "it found A book"
        # can be told apart from "it found THE book".
        other = max((best_iou(rel["subject_box"], d)
                     for d in task.get("distractors") or []), default=0.0)
        best_distractor = max(best_distractor, other)
        if other >= threshold and other > overlap and distractor_rank is None:
            distractor_rank = rank
        matches.append({"rank": rank, "predicate": rel["predicate"],
                        "score": rel["score"], "subject_iou": round(overlap, 3)})
        if class_rank is None:
            class_rank = rank
        if overlap >= threshold and grounded_rank is None:
            grounded_rank = rank
    return {
        "class_rank": class_rank,
        "grounded_rank": grounded_rank,
        "distractor_rank": distractor_rank,
        "n_distractors": len(task.get("distractors") or []),
        "best_subject_iou": round(best, 3),
        "best_distractor_iou": round(best_distractor, 3),
        "hit": {f"hit@{k}": bool(grounded_rank and grounded_rank <= k)
                for k in KS},
        "class_hit": {f"class@{k}": bool(class_rank and class_rank <= k)
                      for k in KS},
        "matches": matches[:5],
    }


def negatives(tasks: Sequence[Dict[str, Any]], objects: Sequence[Dict[str, Any]]
              ) -> List[Dict[str, Any]]:
    """
    One false instruction per task: same target class, a receptacle it is NOT on.

    The control the task needs.  A model that answers "yes, top-3" to every
    instruction scores well on the real tasks and is useless, and only a
    deliberately false instruction shows it.
    """
    classes = {o["vg150_class"] for o in objects if o["vg150_class"]}
    out = []
    for task in tasks:
        taken = {task["object_class"], task["subject_class"]}
        wrong = sorted(classes - taken)
        if not wrong:
            continue
        out.append({**task,
                    "instruction": f"Find the {task['subject_class']} on the "
                                   f"{wrong[0]}  [FALSE]",
                    "object_class": wrong[0],
                    "false": True})
    return out


def draw(frame: np.ndarray, task: Dict[str, Any], result: Dict[str, Any],
         triplets: Sequence[Dict[str, Any]]) -> np.ndarray:
    """
    The instruction, the instance THOR meant, and the TRIPLET EGTR answered with.

    White  the target instance, as THOR knows it.
    Green  a grounded match -- right words, right instance.
    Blue   the model said the right words about the WRONG instance, which is the
           failure this whole file exists to make visible.

    Both ends of the predicted relation are drawn and joined, because the answer
    is a triplet: a `book on table` whose `table` box is the floor is wrong in a
    way that a subject box alone cannot show.  The object end is dashed -- it is
    matched by class only, never by IoU (a dining table fills a third of the
    frame; requiring overlap on it would pass on geometry alone).
    """
    import cv2

    canvas = np.ascontiguousarray(frame[:, :, ::-1])
    font = cv2.FONT_HERSHEY_SIMPLEX

    def tag(text, x, y, colour, scale=0.46):
        (w, h), _ = cv2.getTextSize(text, font, scale, 1)
        x = int(np.clip(x, 0, canvas.shape[1] - w - 8))
        y = int(np.clip(y, h + 8, canvas.shape[0] - 2))
        cv2.rectangle(canvas, (x, y - h - 7), (x + w + 7, y + 2), (25, 25, 25), -1)
        cv2.putText(canvas, text, (x + 4, y - 3), font, scale, colour, 1,
                    cv2.LINE_AA)

    def dashed(box, colour, dash=10):
        x0, y0, x1, y1 = box
        for x in range(x0, x1, dash * 2):
            cv2.line(canvas, (x, y0), (min(x + dash, x1), y0), colour, 2)
            cv2.line(canvas, (x, y1), (min(x + dash, x1), y1), colour, 2)
        for y in range(y0, y1, dash * 2):
            cv2.line(canvas, (x0, y), (x0, min(y + dash, y1)), colour, 2)
            cv2.line(canvas, (x1, y), (x1, min(y + dash, y1)), colour, 2)

    # The target instance.  `bbox_amodal` is absent when the run did not pay for
    # an amodal reference, so the visible box is the one that is always there.
    truth = task.get("target_box_amodal") or task.get("target_box_visible")
    if truth:
        box = [int(v) for v in truth]
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]),
                      (255, 255, 255), 2)
        occlusion = task.get("target_occlusion")
        tag(f"target: {task['subject_class']}"
            + ("" if occlusion is None else f"  occ {occlusion:.0%}"),
            box[0], box[1] - 4, (255, 255, 255))

    rank = result["grounded_rank"] or result["class_rank"]
    if rank:
        rel = triplets[rank - 1]
        colour = (150, 255, 150) if result["grounded_rank"] else (255, 170, 110)
        subject = [int(v) for v in rel["subject_box"]]
        obj = [int(v) for v in rel["object_box"]]
        cv2.rectangle(canvas, (subject[0], subject[1]), (subject[2], subject[3]),
                      colour, 2)
        dashed(obj, colour)
        centre = lambda b: ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
        cv2.line(canvas, centre(subject), centre(obj), colour, 1, cv2.LINE_AA)
        tag(f"#{rank} {rel['subject']} {rel['predicate']} {rel['object']}"
            f"  IoU {result['best_subject_iou']:.2f}",
            subject[0], subject[3] + 20, colour)
        tag(rel["object"], obj[0], obj[1] - 4, colour, 0.42)

    verdict = ("GROUNDED HIT" if result["grounded_rank"] else
               "wrong instance" if result["class_rank"] else "not found")
    colour = ((150, 255, 150) if result["grounded_rank"] else
              (255, 170, 110) if result["class_rank"] else (160, 160, 160))
    caption = (f"{task['instruction']}   ->   {verdict}"
               + (f" at rank {result['grounded_rank'] or result['class_rank']}"
                  if rank else ""))
    (w, h), _ = cv2.getTextSize(caption, font, 0.55, 1)
    y = canvas.shape[0] - 12
    cv2.rectangle(canvas, (8, y - h - 10), (18 + w, y + 6), (25, 25, 25), -1)
    cv2.putText(canvas, caption, (14, y), font, 0.55, colour, 1, cv2.LINE_AA)
    return canvas


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scene", default="FloorPlan203")
    ap.add_argument("--start", metavar="X,Z,YAW,HORIZON",
                    help="write it as --start=-4.75,-1.75,20,45")
    ap.add_argument("--resume", metavar="SCENARIO",
                    help="take scene, clutter and start pose from a "
                         "drive_triplet_scene.py scenario.json")
    ap.add_argument("--topk", type=int, default=max(KS))
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--any-predicate", action="store_true",
                    help="accept any predicate between the right classes")
    ap.add_argument("--occlude", metavar="CLASS:TYPE",
                    help="L3 probe: hide the CLASS target of a task behind a "
                         "moveable TYPE, then ask the same question again, "
                         "e.g. --occlude book:Vase")
    ap.add_argument("--distractor", metavar="TYPE:HOST",
                    help="L2: duplicate TYPE and put it on a HOST, so the "
                         "relation has to disambiguate, e.g. --distractor Book:Chair")
    ap.add_argument("--negatives", action="store_true",
                    help="also ask one deliberately false instruction per task")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--sgg-root", default=None)
    ap.add_argument("--out", default="tasks")
    args = ap.parse_args(argv)

    import cv2

    from robot.sgg_live import SGG_ROOT, load_egtr, predict

    if args.resume:
        from robot.drive_triplet_scene import resume

        rc, plan, _ = resume(args.resume, args.width, args.height, args.fov)
        targets = plan["targets"]
        scene = json.load(open(args.resume))["scene"]
    else:
        rc = drive.open_scene(args.scene, args.width, args.height, args.fov,
                              args.start)
        scene = args.scene
        # The stock scene has no clutter plan, so the target list is simply
        # everything VG150 can name.
        targets = sorted(o["name"] for o in rc.event.metadata["objects"]
                         if o["objectType"] in THOR_TO_VG150)

    try:
        print(f"{scene}   {drive.pose_line(rc)}")
        if args.distractor:
            object_type, host_type = args.distractor.split(":")
            clone = add_distractor(rc, object_type, host_type)
            if clone:
                targets = sorted(set(targets) | {clone})
        state = measure(rc, targets, args.fov)
        entries = {o["name"]: o for o in rc.event.metadata["objects"]}
        tasks = build_tasks(state["objects"], entries)
        if args.negatives:
            tasks = tasks + negatives(tasks, state["objects"])
        if not tasks:
            print("no unambiguous 'X on Y' in this view -- move and try again")
            return 1
        print(f"{len(state['objects'])} objects in view, {len(tasks)} tasks\n")

        egtr = load_egtr(args.sgg_root or SGG_ROOT)
        triplets = predict(egtr, rc.event.frame, args.topk)

        os.makedirs(args.out, exist_ok=True)
        rows = []
        for index, task in enumerate(tasks):
            result = grade(task, triplets, state["objects"], args.iou,
                           args.any_predicate)
            rows.append({**task, "result": result})
            verdict = ("hit" if result["grounded_rank"] else
                       "WRONG INSTANCE" if result["distractor_rank"] else
                       "wrong box" if result["class_rank"] else "miss")
            if result["n_distractors"]:
                verdict += f"  [{result['n_distractors']} distractor(s)]"
            print(f"  {task['instruction']:<44} occ "
                  f"{task['target_occlusion']:.2f}  "
                  f"class_rank {str(result['class_rank']):>4}  "
                  f"grounded {str(result['grounded_rank']):>4}  "
                  f"IoU {result['best_subject_iou']:.2f}  {verdict}")
            name = (f"task_{index:02d}_{task['subject_class']}_on_"
                    f"{task['object_class']}.png")
            cv2.imwrite(os.path.join(args.out, name),
                        draw(rc.event.frame, task, result, triplets))

        real = [r for r in rows if not r.get("false")]
        false = [r for r in rows if r.get("false")]
        print()
        for k in KS:
            hits = sum(r["result"]["hit"][f"hit@{k}"] for r in real)
            classes = sum(r["result"]["class_hit"][f"class@{k}"] for r in real)
            print(f"  hit@{k:<4} {hits}/{len(real)}"
                  f"     class-only@{k:<4} {classes}/{len(real)}")
        if false:
            fp = sum(r["result"]["class_hit"][f"class@{max(KS)}"] for r in false)
            print(f"\n  false instructions answered at all: {fp}/{len(false)}"
                  f"  (lower is better)")

        if args.occlude:
            target_class, occluder_type = args.occlude.split(":")
            chosen = next((r for r in rows if not r.get("false")
                           and r["subject_class"] == target_class), None)
            if chosen is None:
                print(f"\nno task about a {target_class} to occlude")
            else:
                print(f"\n--- hiding {chosen['target_name']} behind a "
                      f"{occluder_type}, same pose")
                staged = put_in_front(rc, chosen["target_name"], occluder_type,
                                         chosen["receptacle_name"])
                if staged:
                    # Re-measure: the occluder moved, so every box and every
                    # occlusion figure in the view is stale.  Re-deriving the
                    # tasks too would be wrong -- the question must stay the
                    # one that was asked before -- so only the boxes are
                    # refreshed, matched back by name.
                    after = measure(rc, targets, args.fov)
                    fresh = {o["name"]: o for o in after["objects"]}
                    task = dict(chosen)
                    del task["result"]
                    moved = fresh.get(task["target_name"])
                    if moved is None:
                        print("  target no longer measurable in this view")
                    else:
                        task.update({
                            "target_occlusion": moved["occlusion"],
                            "target_box_amodal": moved["bbox_amodal"],
                            "target_box_visible": moved["bbox_visible"]})
                        triplets_after = predict(egtr, rc.event.frame, args.topk)
                        result = grade(task, triplets_after, after["objects"],
                                       args.iou, args.any_predicate)
                        before = chosen["result"]
                        print(f"  {task['instruction']}")
                        print(f"    before: occ {chosen['target_occlusion']:.2f}"
                              f"  class_rank {before['class_rank']}"
                              f"  grounded {before['grounded_rank']}")
                        print(f"    after:  occ {task['target_occlusion']:.2f}"
                              f"  class_rank {result['class_rank']}"
                              f"  grounded {result['grounded_rank']}")
                        cv2.imwrite(os.path.join(args.out, "occluded.png"),
                                    draw(rc.event.frame, task, result,
                                         triplets_after))
                        rows.append({**task, "result": result,
                                     "occluded": staged})

        with open(os.path.join(args.out, "tasks.json"), "w") as fh:
            json.dump({"scene": scene, "pose": state["pose"], "iou": args.iou,
                       "topk": args.topk, "any_predicate": args.any_predicate,
                       "tasks": rows, "triplets": triplets}, fh, indent=1)
        print(f"\nwrote {os.path.join(args.out, 'tasks.json')} and "
              f"{len(rows)} pngs")
        return 0
    finally:
        rc.stop()


if __name__ == "__main__":
    raise SystemExit(main())
