"""
measure.py -- what the robot's own camera can actually see from where it stands.

GROUND TRUTH, NOT A DECISION.  Everything here reads THOR's instance masks, so
nothing in it may reach a policy; `eval_move.truth_boxes` calls `geometry` to
decide whether a top-1 answer was right, and `measure` re-reads occlusion at a
pose.  A rule that consulted either would be grading itself.

    measure()   occlusion and relations at the current pose, using the DATASET's
                own protocol (`build.sgg.build_occlusion_dataset.annotate_relations`,
                same amodal references, same GT definition) so a number measured
                here is comparable with one measured at build time
    geometry()  amodal and visible boxes of named objects

The measurement is through the ACTING AGENT's camera, whose eye height is fixed
(1.576 standing, 0.95 crouched), not through the third-party camera at 1.0-1.5 m
that `occlusion_ds4` was rendered with.  That is why `resume` re-measures rather
than trusting the recorded numbers: the dataset supplies candidate poses, and
what gets reported is what the robot sees.

This was `drive_triplet_scene.py`, 1219 lines, most of them an interactive WASD
window for exploring those scenes by hand.  Six callers imported it for the
functions below and never for the window; the window is gone.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import build.sgg.occlusion_pipeline as P
from build.sgg.build_occlusion_dataset import (DEFAULT_FOV, MAX_OCCLUSION,
                                         annotate_relations)
from build.sgg.geometry import bbox_of, intrinsics_record, pick_focus
from robot.world.robot_controller import RobotController
from robot.task.find_partial_triplets import qualifying
from vg.vg150 import THOR_TO_VG150


DS4_ARRANGE = dict(n_surfaces=3, spacing=0.18, small_extra=16, large_extra=6)


def agent_masks(event, names_by_id: Dict[str, str]
                ) -> Dict[str, Tuple[int, List[int]]]:
    """Visible pixel count and box per object name, from the agent camera."""
    out: Dict[str, Tuple[int, List[int]]] = {}
    for object_id, mask in (event.instance_masks or {}).items():
        name = names_by_id.get(object_id)
        if name is None:            # structural geometry THOR does not name
            continue
        box = bbox_of(mask)
        if box is not None:
            out[name] = (int(mask.sum()), box)
    return out


def visible_only(rc, wanted: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """
    Just the boxes the camera actually draws.  No clearing, no extra renders.

    This is the fast path and, measured, it is also the whole of the grading
    signal.  Over twelve staged frames EGTR's box agreed with the target's
    VISIBLE extent 6 times and with its AMODAL extent 0 times (6 ties, all at
    low occlusion), and the agreement gap widens with occlusion -- 0.41 amodal
    vs 0.89 visible on a half-hidden lamp.  So `max(amodal, visible)` was
    always just `visible`: dropping the amodal branch changed the hit count
    from 10 to 10.  `0806_progress.md` reports the same thing independently
    ("bbox_visible is the evaluator's box ... amodal checks are systematically
    pessimistic for occluded objects").

    Clearing the scene is also what the pipeline was actually spending its time
    on: `DisableObject`/`EnableObject` are one THOR step EACH, and a living room
    holds 30-60 nameable or moveable objects, so an amodal reference cost ~2N
    steps against the 20 renders of the sweep itself.
    """
    names_by_id = {o["objectId"]: o["name"]
                   for o in rc.event.metadata["objects"]}
    seen = agent_masks(rc.event, names_by_id)
    out = {}
    for name in wanted:
        seen_px, seen_box = seen.get(name, (0, None))
        out[name] = {"bbox_amodal": None, "bbox_visible": seen_box,
                     "reference_px": None, "visible_px": seen_px,
                     "occlusion": None}
    return out


def pose_of(rc: RobotController) -> Dict[str, Any]:
    camera = rc.camera_xyz
    return {"position": dict(rc.agent_position),
            "camera_position": {"x": float(camera[0]), "y": float(camera[1]),
                                "z": float(camera[2])},
            "yaw": rc.agent_yaw, "horizon": rc.camera_horizon,
            "body": rc.height_level}


def measure(rc: RobotController, targets: Sequence[str], fov: float,
            max_occlusion: float = MAX_OCCLUSION) -> Dict[str, Any]:
    """
    Occlusion and relations at the pose the robot is standing in right now.

    The amodal reference is the dataset's: disable everything nameable, enable
    one object, and the pixels it draws alone are what it would occupy if
    nothing were in front of it.  Two differences from the build, both
    deliberate:

      * The camera does not move between renders here, so the mask is read
        straight off the Enable/Disable event instead of re-aiming a camera.
        That shortcut is why the sweep MUST pause physics auto-sim first.
        Measured on stock FloorPlan203: without the pause, 12 of 15 objects come
        back with no instance mask at all on the `EnableObject` event -- the
        DiningTable, 83340 visible pixels, among them -- so they were silently
        dropped from the annotation.  The re-spawned object simply is not in the
        segmentation that same frame.  The build gets away with reading it
        because it calls `look()` in between, which costs it a frame.  Pausing
        also freezes the scene: unpaused, re-enabling objects under a disabled
        support let a Vase drop 0.216 m mid-sweep, so its "amodal reference" was
        measured somewhere the object never was.
      * Only objects with at least one VISIBLE pixel are swept.  The build has
        to sweep all of them because it annotates ten viewpoints at once; here
        the sweep is the per-keystroke cost, and an object with zero visible
        pixels is at occlusion 1.0 -- past `max_occlusion`, so it would be
        dropped anyway and cannot land in the band.
    """
    entries = {o["name"]: o for o in rc.event.metadata["objects"]}
    names_by_id = {o["objectId"]: o["name"] for o in rc.event.metadata["objects"]}
    observed = agent_masks(rc.event, names_by_id)

    swept = [n for n in targets if n in observed and n in entries]
    ids = {n: entries[n]["objectId"] for n in swept}
    rc.controller.step(action="PausePhysicsAutoSim")
    try:
        for object_id in ids.values():
            rc.controller.step(action="DisableObject", objectId=object_id)

        amodal: Dict[str, Optional[Tuple[int, List[int]]]] = {}
        for name in swept:
            event = rc.controller.step(action="EnableObject", objectId=ids[name])
            amodal[name] = agent_masks(event, names_by_id).get(name)
            rc.controller.step(action="DisableObject", objectId=ids[name])
        for object_id in ids.values():
            rc.controller.step(action="EnableObject", objectId=object_id)
    finally:
        rc.controller.step(action="UnpausePhysicsAutoSim")
    rc.event = rc.controller.last_event

    camera = rc.camera_xyz
    min_px = int(P.MIN_EXTENT_FRACTION * rc.width * rc.height)

    objects = []
    for name in swept:
        entry, reference = entries[name], amodal.get(name)
        if reference is None:
            continue
        ref_px, ref_box = reference
        if ref_px < min_px:
            continue
        seen_px, seen_box = observed.get(name, (0, None))
        occlusion = max(0.0, 1.0 - seen_px / ref_px)
        if occlusion > max_occlusion:
            continue
        objects.append({
            "name": name,
            "thor_type": entry["objectType"],
            "vg150_class": THOR_TO_VG150.get(entry["objectType"]),
            "object_id": entry["objectId"],
            "position": dict(entry["position"]),
            "distance": math.dist(tuple(camera),
                                  (entry["position"]["x"], entry["position"]["y"],
                                   entry["position"]["z"])),
            "parent_receptacles": entry.get("parentReceptacles") or [],
            "pickupable": bool(entry["pickupable"]),
            "moveable": bool(entry.get("moveable")),
            "bbox_amodal": ref_box,
            "bbox_visible": seen_box,
            "reference_px": ref_px,
            "visible_px": seen_px,
            "occlusion": round(occlusion, 4),
        })

    # Relations come from the dataset's own annotator, so a GT redefinition in
    # `vg_gt` lands here too and the triplets printed in the window are the same
    # ones `eval_occlusion.py` would score.
    record = {"intrinsics": intrinsics_record(rc.width, rc.height, fov),
              "views": [{"index": 0, "image": "", "camera": pose_of(rc),
                         "objects": objects}]}
    annotate_relations(record, entries, rate=ANNOTATION_RATE)
    return {"pose": pose_of(rc), "objects": objects,
            "relations": record["views"][0]["relations"]}


def geometry(rc, wanted: Sequence[str], names: Sequence[str],
             amodal: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Amodal and visible boxes of `wanted`, at the pose the robot is standing in.

    With `amodal=False` this is `visible_only` -- see there for why that is the
    default in the runners.

    The same protocol as `measure` above -- pause physics, disable
    everything nameable, enable one object at a time -- but for a named few
    rather than for the whole scene, and WITHOUT that function's
    `MAX_OCCLUSION` filter.  The filter is why this exists: a staged target sits
    at 0.7-0.8 occlusion, past the ceiling, so `measure` drops it and there is
    nothing left to grade.  A dropped target is the outcome under test, not a
    reason to lose the row.

    Pausing is not optional; `measure` documents the failure it prevents (12 of
    15 objects come back with no instance mask on the `EnableObject` event).
    Ids are read fresh from the current event because staging goes through
    `SetObjectPoses`, which renumbers every objectId.
    """
    if not amodal:
        return visible_only(rc, wanted)

    entries = {o["name"]: o for o in rc.event.metadata["objects"]}
    names_by_id = {o["objectId"]: o["name"] for o in rc.event.metadata["objects"]}
    seen = agent_masks(rc.event, names_by_id)

    ids = {n: entries[n]["objectId"] for n in names if n in entries}
    alone: Dict[str, Any] = {}
    rc.controller.step(action="PausePhysicsAutoSim")
    try:
        for object_id in ids.values():
            rc.controller.step(action="DisableObject", objectId=object_id)
        for name in wanted:
            if name not in ids:
                continue
            event = rc.controller.step(action="EnableObject",
                                       objectId=ids[name])
            alone[name] = agent_masks(event, names_by_id).get(name)
            rc.controller.step(action="DisableObject", objectId=ids[name])
        for object_id in ids.values():
            rc.controller.step(action="EnableObject", objectId=object_id)
    finally:
        rc.controller.step(action="UnpausePhysicsAutoSim")
    # A no-op step to get a frame rendered with every object back in the scene.
    # `measure` documents that a just-re-enabled object is missing from the
    # segmentation of that same event; the frame EGTR is asked about must not be
    # one where half the room has yet to reappear.
    rc.controller.step(action="Done")
    rc.event = rc.controller.last_event

    out = {}
    for name in wanted:
        reference = alone.get(name)
        ref_px, ref_box = reference if reference else (0, None)
        seen_px, seen_box = seen.get(name, (0, None))
        out[name] = {
            "bbox_amodal": ref_box,
            "bbox_visible": seen_box,
            "reference_px": ref_px,
            "visible_px": seen_px,
            "occlusion": round(max(0.0, 1.0 - seen_px / ref_px), 4)
                         if ref_px else 1.0,
        }
    return out


def at_pose(rc: RobotController, pose: Dict[str, Any], targets: Sequence[str],
            fov: float, lo: float, hi: float, both: bool) -> Dict[str, Any]:
    """
    Put the robot back on a pose and measure it there.

    The `Teleport` is checked.  It fails when the clutter has settled into the
    spot asked for -- the rebuild is not bit-reproducible -- and an unchecked
    failure leaves the agent at THOR's default spawn while everything downstream
    reports the pose it was ASKED for: the frame, the boxes and the relations
    would all be of some other corner of the room.
    """
    rc.set_height_level(pose["body"])
    ok = rc.teleport(position=pose["position"], yaw=pose["yaw"],
                     horizon=pose["horizon"], standing=(pose["body"] == "stand"))
    result = measure(rc, targets, fov)
    result["teleport_ok"] = bool(ok)
    if not ok:
        print(f"  ! Teleport to "
              f"({pose['position']['x']:+.2f}, {pose['position']['z']:+.2f}) "
              f"yaw {pose['yaw']:.1f} refused -- the robot is at "
              f"({result['pose']['position']['x']:+.2f}, "
              f"{result['pose']['position']['z']:+.2f}) instead")
    result["hits"] = [r for r in result["relations"]
                      if qualifying(r, lo, hi, both)]
    return result


def remove_objects(rc: RobotController, plan: Dict[str, Any],
                   patterns: Sequence[str]) -> List[str]:
    """
    Take objects out of the scene before anything is measured.

    Matches a `name`, an `objectType` or a VG150 class, case-insensitively, so
    `--remove book` takes both books and `--remove Book_e173324d` takes one.
    `DisableObject` is the same mechanism the amodal pass uses, so a removed
    object is gone from the render, from `targets`, and from the relations --
    which is the point: removing what was hiding something CHANGES the occlusion
    of what it hid, and that gets re-measured rather than carried over.
    """
    wanted = {p.strip().lower() for p in patterns if p.strip()}
    if not wanted:
        return []
    removed = []
    for entry in list(rc.event.metadata["objects"]):
        keys = {entry["name"].lower(), entry["objectType"].lower(),
                str(THOR_TO_VG150.get(entry["objectType"])).lower()}
        if not (keys & wanted):
            continue
        event = rc.controller.step(action="DisableObject",
                                   objectId=entry["objectId"])
        if event.metadata["lastActionSuccess"]:
            removed.append(entry["name"])
        else:
            print(f"  ! THOR refused to remove {entry['name']}")
    rc.event = rc.controller.last_event
    plan["targets"] = [t for t in plan["targets"] if t not in set(removed)]
    if removed:
        print(f"  removed {len(removed)}: {', '.join(removed)}")
    return removed


def rebuild(rc: RobotController, seed: int,
            expected_added: Optional[List[str]] = None,
            arrange: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Re-run the dataset's clutter arrangement, and check it came back the same."""
    focus = pick_focus(rc.controller, seed)
    plan = P.arrange(rc.controller, seed=seed, focus=focus,
                     **(arrange or DS4_ARRANGE))
    rc.event = rc.controller.last_event
    rc.get_reachable_positions(refresh=True)
    if expected_added is not None and sorted(plan["placed"]) != sorted(expected_added):
        # Not fatal: THOR's physics settle is not bit-reproducible across
        # versions, so a differing clutter list means the viewpoint is a hint
        # rather than a replay.  The measurement below is still the truth.
        print(f"  ! clutter differs from the dataset record "
              f"({len(plan['placed'])} placed vs {len(expected_added)})")
    return plan


def resume(path: str, width: int = 800, height: int = 600,
           fov: float = DEFAULT_FOV, headless: bool = False,
           lo: float = 0.4, hi: float = 0.8, both: bool = False,
           remove: Sequence[str] = ()
           ) -> Tuple[RobotController, Dict[str, Any], Dict[str, Any]]:
    """
    Reopen a saved `scenario.json` with the robot standing on its start pose.

    This is the entry point for driving the scenario from your own code -- the
    search is not repeated, so it costs one scene load plus the clutter rebuild:

        rc, plan, here = resume("driveable/fp203/scenario.json")
        rc.move_ahead(); rc.rotate(30)                  # your policy here
        here = at_pose(rc, pose_of(rc), plan["targets"], 60.0, 0.4, 0.8, False)
        print(len(here["hits"]), "triplets still in band")

    `rc.controller` is the raw THOR controller if you need an action this class
    does not wrap.  Nothing else in the process may open a second controller on
    the same scene -- THOR is one Unity process per controller.
    """
    with open(path) as fh:
        scenario = json.load(fh)
    rc = RobotController(scene=scenario["scene"], width=width, height=height,
                         field_of_view=fov, visibility_distance=15.0,
                         headless=headless, verbose=False)
    plan = rebuild(rc, scenario["seed"], arrange=scenario.get("arrange"))
    # Whatever the scenario was saved without stays out, plus anything asked for
    # now -- otherwise resuming would silently put the books back.
    here = {"removed": remove_objects(
        rc, plan, list(scenario.get("removed") or []) + list(remove))}
    here.update(at_pose(rc, scenario["start"], plan["targets"], fov, lo, hi,
                        both))
    here["candidate"] = scenario.get("source_view", {})
    here["plan"] = plan
    here["scene"], here["seed"] = scenario["scene"], scenario["seed"]
    return rc, plan, here
