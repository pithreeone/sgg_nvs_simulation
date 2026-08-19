"""
drive_triplet_scene.py -- open a THOR scene at a viewpoint where a triplet is
HALF hidden, and hand the robot to you from there.

The band is the point.  A triplet whose endpoints are both in plain sight is
already solved from one frame, and one hidden past ~0.8 has so few pixels left
that no per-frame model can recover it -- Regime B in TASKS.md, which needs a
prior or memory rather than a better scene graph.  In between, the object is
"detectable but not usable": EGTR sees something and may mis-name it or attach
the wrong relation, and the robot has enough evidence to know it should go and
look again.  That is the regime worth driving around in.

The start pose is measured, not assumed.  `occlusion_ds4` already records which
scene/seed/view carries such a triplet, but its views come from a third-party
camera at 1.0-1.5 m, and the acting agent's eye height is fixed (1.576 stand /
0.95 crouch -- see TASKS.md).  So the dataset only supplies CANDIDATES: each
qualifying view's camera pose is snapped to a reachable agent pose, and the
occlusion is then re-measured through the robot's own camera with the dataset's
own protocol (`gen.build_occlusion_dataset.annotate_relations`, same amodal
references, same GT definition).  What gets reported is what the robot sees.

    # find a scene, drive it in a window (WASD; press SPACE to re-measure)
    python drive_triplet_scene.py

    # a specific scene, and write the scenario out for your own code to load
    python drive_triplet_scene.py --scene FloorPlan203 --out driveable/fp203

    # come back to that start pose later without repeating the search
    python drive_triplet_scene.py --resume driveable/fp203/scenario.json

    # no display available: report the pose and the triplets, then exit
    python drive_triplet_scene.py --driver none

    # widen the band, or demand both endpoints be partly hidden
    python drive_triplet_scene.py --lo 0.3 --hi 0.9 --both-endpoints

Keys in the window
    w / s      drive forward / back           a / d   turn left / right
    q / e      strafe left / right            r / f   look up / down
    c          toggle crouch/stand            SPACE   re-measure occlusion here
    p          save frame + pose to --out     ESC/x   quit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import gen.occlusion_pipeline as P
from gen.build_occlusion_dataset import (
    DEFAULT_FOV,
    annotate_relations,
    bbox_of,
    intrinsics,
    pick_focus,
)
from gen.occlusion import MAX_OCCLUSION
from robot.find_partial_triplets import qualifying
from robot.robot_controller import RobotController, horizon_towards, yaw_towards
from vg.vg150 import THOR_TO_VG150
from vg.vg_gt import VG_ANNOTATION_RATE

#: The clutter parameters `occlusion_ds4` was built with.  A scene rebuilt with
#: anything else is a different scene, and the dataset's candidate viewpoints
#: stop meaning anything -- see *How a scene is built* in TASKS.md.
DS4_ARRANGE = dict(n_surfaces=3, spacing=0.18, small_extra=16, large_extra=6)

#: Where the winning pose of a search is remembered, so the grid runs once per
#: scene build rather than once per session.  Only the POSE is cached, never the
#: occlusion figures: `P.arrange` is not bit-reproducible (physics settle), so a
#: rebuilt scene is near-identical but not identical, and a cached measurement
#: would drift away from the render it claims to describe.  Re-measuring one
#: pose costs a single amodal sweep against the grid's dozen-plus.
CACHE_DIR = ".drive_poses"

#: Share of object pairs the ground truth annotates, session-wide -- a GT
#: DEFINITION, like `MAX_OCCLUSION`, not a per-call argument, so it lives here
#: rather than being threaded through five signatures.  `--rate all` sets it to
#: None, which keeps every geometric proposal: that is a diagnostic mode for
#: "why is my triplet missing", and its relation counts are not comparable with
#: `eval_occlusion.py` or with anything in TASKS.md.
ANNOTATION_RATE: Optional[float] = VG_ANNOTATION_RATE


def cache_path(root: str, scene: str, seed: int, view: Optional[int],
               lo: float, hi: float, both: bool) -> str:
    # The band and the both-endpoints rule are part of the key: they change
    # which pose wins, so a pose found for (0.4, 0.8) must not be handed back
    # for a --lo 0.3 run.
    band = f"{lo:g}-{hi:g}" + ("-both" if both else "")
    name = f"{scene}_s{seed}" + (f"_v{view}" if view is not None else "")
    return os.path.join(root, f"{name}_{band}.json")


# --------------------------------------------------------------------------
# candidate start poses, from the dataset's own annotation
# --------------------------------------------------------------------------

def dataset_candidates(dataset: str, lo: float, hi: float, both: bool,
                       scene: Optional[str] = None,
                       seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Views of `dataset` whose relations qualify, richest first.

    Ranked by how many qualifying triplets the view holds, because a viewpoint
    with six half-hidden relations is a better place to start driving than one
    with a single borderline one -- and because the re-measurement through the
    agent's camera loses some of them to the eye-height change, so starting from
    the richest view leaves the most margin.
    """
    import glob

    by_view: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(dataset, "*", "scene.json"))):
        with open(path) as fh:
            record = json.load(fh)
        directory = os.path.dirname(path)
        if scene and record["scene"] != scene:
            continue
        if seed is not None and record["seed"] != seed:
            continue
        for view in record["views"]:
            hits = [r for r in view["relations"] if qualifying(r, lo, hi, both)]
            if not hits:
                continue
            by_view[(directory, view["index"])] = {
                "scene": record["scene"],
                "seed": record["seed"],
                "scene_dir": directory,
                "view": view["index"],
                "camera": view["camera"],
                "aim": aim_point(view, hits),
                "added": record["added"],
                "n_dataset_hits": len(hits),
                "dataset_hits": [triplet_str(r) for r in hits],
            }
    return sorted(by_view.values(), key=lambda c: -c["n_dataset_hits"])


def aim_point(view: Dict[str, Any], hits: Sequence[Dict[str, Any]]
              ) -> Dict[str, float]:
    """
    Where the robot should look: the centroid of the half-hidden triplets.

    Reusing the recorded camera *pitch* does not work.  The dataset aimed each
    camera at the focus surface from a height in 1.0-1.5 m; the agent's eye sits
    at 1.576 m or 0.95 m, and re-using the pitch from a different height points
    the camera at the floor -- the first version of this script opened on a
    picture of floorboards with one chair leg in it.  Aiming at a POINT survives
    the height change, and the point that matters is the objects the band is
    about.
    """
    wanted = {name for rel in hits
              for name in (rel["subject_name"], rel["object_name"])}
    points = [o["position"] for o in view["objects"] if o["name"] in wanted]
    if not points:
        return dict(view["camera"]["position"])
    return {axis: float(np.mean([p[axis] for p in points])) for axis in "xyz"}


def selection(result: Dict[str, Any], show: str) -> List[Dict[str, Any]]:
    """
    Which relations the overlay can box: the in-band ones, or all of them.

    `all` exists because "why is there no box on the vase" is usually not a
    rendering question -- the vase is annotated at occlusion 0.00, so no relation
    it takes part in has an endpoint in the band, and the band is what the
    overlay cycles.
    """
    return result["relations"] if show == "all" else result["hits"]


def pick(relations: Sequence[Dict[str, Any]], phrase: Optional[str]) -> int:
    """Index of the first relation whose `subject predicate object` matches."""
    if not phrase:
        return 0
    wanted = " ".join(phrase.lower().split())
    for index, rel in enumerate(relations):
        text = f"{rel['subject']} {rel['predicate']} {rel['object']}".lower()
        if wanted in text:
            return index
    print(f"  ! nothing matching '{phrase}' -- showing the first instead")
    return 0


def triplet_str(rel: Dict[str, Any]) -> str:
    return (f"{rel['subject']} {rel['predicate']} {rel['object']}"
            f"  [{rel['subject_occlusion']:.2f}/{rel['object_occlusion']:.2f}]")


# --------------------------------------------------------------------------
# measurement through the robot's own camera
# --------------------------------------------------------------------------

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
    record = {"intrinsics": intrinsics(rc.width, rc.height, fov),
              "views": [{"index": 0, "image": "", "camera": pose_of(rc),
                         "objects": objects}]}
    annotate_relations(record, entries, rate=ANNOTATION_RATE)
    return {"pose": pose_of(rc), "objects": objects,
            "relations": record["views"][0]["relations"]}


def pose_of(rc: RobotController) -> Dict[str, Any]:
    camera = rc.camera_xyz
    return {"position": dict(rc.agent_position),
            "camera_position": {"x": float(camera[0]), "y": float(camera[1]),
                                "z": float(camera[2])},
            "yaw": rc.agent_yaw, "horizon": rc.camera_horizon,
            "body": rc.height_level}


# --------------------------------------------------------------------------
# choosing where to stand
# --------------------------------------------------------------------------

def probe(rc: RobotController, targets: Sequence[str], candidate: Dict[str, Any],
          fov: float, lo: float, hi: float, both: bool, bodies: Sequence[str],
          yaw_jitter: Sequence[float], horizon_jitter: Sequence[float],
          radius: Tuple[float, float] = (1.2, 2.5), positions: int = 2,
          verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Re-measure the neighbourhood of one dataset viewpoint, agent-side.

    The grid exists because the eye-height change is not a small perturbation:
    looking at the same clutter from 1.576 m instead of 1.2 m looks down on it,
    and an object the dataset had half hidden can come clear.  Each cell aims at
    the triplet centroid and then offsets, so the offsets are jitter around a
    sensible view rather than around the dataset's un-transferable pitch.  One
    amodal sweep per cell.
    """
    camera, aim = candidate["camera"], candidate["aim"]
    aim_xz = np.array([aim["x"], aim["z"]])
    spots = standoff_spots(rc, aim_xz,
                           np.array([camera["position"]["x"],
                                     camera["position"]["z"]]),
                           radius, positions)
    if not spots:
        return []

    out = []
    for spot in spots:
        position = {"x": float(spot[0]), "y": float(rc.agent_position["y"]),
                    "z": float(spot[1])}
        yaw = yaw_towards(spot, aim_xz)
        for body in bodies:
            eye = np.array([position["x"],
                            position["y"] + RobotController.HEIGHT_LEVELS[body],
                            position["z"]])
            horizon = horizon_towards(eye, np.array([aim["x"], aim["y"],
                                                     aim["z"]]))
            for d_yaw in yaw_jitter:
                for d_horizon in horizon_jitter:
                    rc.set_height_level(body)
                    if not rc.teleport(position=position, yaw=yaw + d_yaw,
                                       horizon=horizon + d_horizon,
                                       standing=(body == "stand")):
                        continue
                    result = measure(rc, targets, fov)
                    hits = [r for r in result["relations"]
                            if qualifying(r, lo, hi, both)]
                    result["hits"] = hits
                    out.append(result)
                    if verbose:
                        print(f"    {np.linalg.norm(spot - aim_xz):.1f}m "
                              f"{body:<6} yaw {d_yaw:+5.1f} horizon "
                              f"{d_horizon:+5.1f}  "
                              f"{len(result['relations']):3d} relations, "
                              f"{len(hits):2d} in band")
    return out


def standoff_spots(rc: RobotController, aim_xz: np.ndarray,
                   camera_xz: np.ndarray, radius: Tuple[float, float],
                   count: int) -> List[np.ndarray]:
    """
    Reachable positions that stand OFF the triplet, nearest the dataset's camera.

    Snapping straight to the dataset camera's footprint is what the first
    version did, and it put the robot 0.5 m from a dining chair looking 45 deg
    down at the floor: the objects were all still in frame -- the count of
    in-band triplets was the highest of the grid -- and the frame was useless to
    look at and to drive from.  `radius` is the dataset's own camera band
    (1.2-2.0 m, see `camera_poses`), widened a little because the agent stands
    taller and needs the extra distance to keep the pitch shallow.
    """
    reachable = rc.get_reachable_positions()
    if len(reachable) == 0:
        return []
    distance = np.linalg.norm(reachable - aim_xz.reshape(1, 2), axis=1)
    inside = np.flatnonzero((distance >= radius[0]) & (distance <= radius[1]))
    if inside.size == 0:            # tight room: fall back to the nearest ring
        inside = np.argsort(np.abs(distance - float(np.mean(radius))))[:count]
    order = inside[np.argsort(
        np.linalg.norm(reachable[inside] - camera_xz.reshape(1, 2), axis=1))]
    return [reachable[i] for i in order[:count]]


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


def choose(rc: RobotController, candidates: Sequence[Dict[str, Any]], fov: float,
           lo: float, hi: float, both: bool, max_probes: int,
           bodies: Sequence[str], yaw_jitter: Sequence[float],
           horizon_jitter: Sequence[float], radius: Tuple[float, float],
           positions: int, remove: Sequence[str] = ()
           ) -> Optional[Dict[str, Any]]:
    """First candidate whose re-measurement still qualifies, best pose of its grid."""
    plan: Optional[Dict[str, Any]] = None
    removed: List[str] = []
    for index, candidate in enumerate(candidates[:max_probes]):
        print(f"  probe {index + 1}/{min(max_probes, len(candidates))}: "
              f"{candidate['scene']} s{candidate['seed']} view {candidate['view']:02d} "
              f"({candidate['n_dataset_hits']} dataset hits)")
        if plan is None:
            plan = rebuild(rc, candidate["seed"],
                           expected_added=candidate.get("added"))
            removed = remove_objects(rc, plan, remove)
        results = probe(rc, plan["targets"], candidate, fov, lo, hi, both,
                        bodies, yaw_jitter, horizon_jitter, radius, positions)
        results = [r for r in results if r["hits"]]
        if results:
            # Most triplets in the band wins, and ties go to the pose that sees
            # the most relations at all -- a start view with more of the scene
            # in it leaves more for the robot to do something about.
            best = max(results, key=lambda r: (len(r["hits"]),
                                               len(r["relations"])))
            # The grid left the robot standing in its LAST cell, not the winning
            # one.  Driving on from there, or drawing the winner's boxes over
            # that frame, silently mixes two poses: the first version of this
            # put a `plate` box on a brick wall three metres from the plate.
            # Re-measuring rather than trusting the cached numbers keeps the
            # frame, the boxes and the relations from one single event.
            restored = at_pose(rc, best["pose"], targets=plan["targets"],
                               fov=fov, lo=lo, hi=hi, both=both)
            if len(restored["hits"]) != len(best["hits"]):
                print(f"  ! re-measure at the winning pose gives "
                      f"{len(restored['hits'])} in band, not "
                      f"{len(best['hits'])}")
            restored["candidate"] = candidate
            restored["plan"] = plan
            restored["removed"] = removed
            return restored
    return None


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


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def report(result: Dict[str, Any], lo: float, hi: float,
           show: str = "band") -> None:
    pose = result["pose"]
    print(f"\nstart pose  x {pose['position']['x']:+.2f}  "
          f"z {pose['position']['z']:+.2f}  yaw {pose['yaw']:.1f}  "
          f"horizon {pose['horizon']:.1f}  body {pose['body']} "
          f"(eye {pose['camera_position']['y']:.2f} m)")
    shown = selection(result, show)
    print(f"{len(result['relations'])} relations in view, "
          f"{len(result['hits'])} with an endpoint in ({lo}, {hi})"
          f"{'; listing all' if show == 'all' else ''}:\n")
    # Numbered, because the number is what `--triplet N` takes.
    for index, rel in enumerate(shown, 1):
        marks = "".join("*" if lo < o < hi else " " for o in
                        (rel["subject_occlusion"], rel["object_occlusion"]))
        print(f"  {index:3d}. {rel['subject']:>12} {rel['predicate']:^11} "
              f"{rel['object']:<12}"
              f"  occ {rel['subject_occlusion']:.2f}/{rel['object_occlusion']:.2f} "
              f"{marks}  ({rel['annotation']})")
    print("\n  * = the endpoint inside the band")


def save(result: Dict[str, Any], outdir: str, scene: str, seed: int,
         frame: Optional[np.ndarray], lo: float, hi: float, both: bool,
         selected: int = 0, show: str = "band") -> None:
    os.makedirs(outdir, exist_ok=True)
    scenario = {
        "scene": scene,
        "seed": seed,
        "arrange": DS4_ARRANGE,
        "band": [lo, hi],
        "both_endpoints": both,
        # Replayed by `--resume`, so a scene the books were taken out of stays
        # that way.
        "removed": result.get("removed") or [],
        # Whatever the start came from -- a dataset view, `--start`, or a pose
        # driven to by hand.  Not a fixed set of keys: the three paths carry
        # different provenance, and assuming the dataset's keys crashed
        # `--start`.  `added` is dropped only because it is long.
        "source_view": {k: v for k, v in (result.get("candidate") or {}).items()
                        if k != "added"},
        "start": result["pose"],
        "relations": result["relations"],
        "in_band": result["hits"],
        "objects": result["objects"],
    }
    path = os.path.join(outdir, "scenario.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scenario, fh, indent=1)
    print(f"\nwrote {path}")
    if frame is not None:
        from PIL import Image
        image = os.path.join(outdir, "start.png")
        Image.fromarray(frame).save(image)
        print(f"wrote {image}")
    if frame is not None and selection(result, show):
        # One triplet per image, on purpose: a living-room view carries a dozen
        # in-band relations and drawing them together is the unreadable overlay
        # `visualize_occlusion.py` had to split into two files.
        import cv2

        shown = selection(result, show)
        rel = shown[selected % len(shown)]
        canvas = np.ascontiguousarray(frame[:, :, ::-1])
        draw_triplet(canvas, rel, {o["name"]: o for o in result["objects"]},
                     f"[{selected % len(shown) + 1}/{len(shown)}]")
        path = os.path.join(outdir, "triplet.png")
        cv2.imwrite(path, canvas)
        print(f"wrote {path}   {triplet_str(rel)}")


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------

HELP = ["w/s drive  a/d turn  q/e strafe", "r/f look  c crouch",
        "n/b next/prev triplet  t overlay", "SPACE measure  p save  ESC quit"]

#: One triplet is drawn at a time, and the two endpoints have to be told apart at
#: a glance, so the colour carries the ROLE (subject / object) rather than the
#: occlusion band `visualize_occlusion.py` colours by -- the occlusion is in the
#: label instead.  BGR, because the overlay is composed in OpenCV.
SUBJECT_COLOUR = (90, 190, 255)     # amber
OBJECT_COLOUR = (150, 255, 150)     # green


def draw_triplet(canvas: np.ndarray, rel: Dict[str, Any],
                 boxes: Dict[str, Dict[str, Any]], position: str = "") -> None:
    """
    Box both endpoints of ONE relation, in place, on a BGR image.

    Outer box amodal, inner box visible -- `visualize_occlusion.py`'s convention,
    so the gap between the two rectangles IS the occlusion the label states.
    """
    import cv2

    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.46
    centres = {}

    def tag(text: str, x: int, y: int, colour: Tuple[int, int, int]) -> None:
        (w, h), _ = cv2.getTextSize(text, font, scale, 1)
        x = int(np.clip(x, 0, canvas.shape[1] - w - 8))
        y = int(np.clip(y, h + 8, canvas.shape[0] - 2))
        cv2.rectangle(canvas, (x, y - h - 7), (x + w + 7, y + 2), (25, 25, 25), -1)
        cv2.putText(canvas, text, (x + 4, y - 3), font, scale, colour, 1,
                    cv2.LINE_AA)

    for role, key, colour in (("S", "subject", SUBJECT_COLOUR),
                              ("O", "object", OBJECT_COLOUR)):
        entry = boxes.get(rel[f"{key}_name"])
        if entry is None:
            continue
        ax0, ay0, ax1, ay1 = (int(v) for v in entry["bbox_amodal"])
        cv2.rectangle(canvas, (ax0, ay0), (ax1, ay1), colour, 2)
        if entry.get("bbox_visible"):
            vx0, vy0, vx1, vy1 = (int(v) for v in entry["bbox_visible"])
            cv2.rectangle(canvas, (vx0, vy0), (vx1, vy1), colour, 1)
        centres[role] = ((ax0 + ax1) // 2, (ay0 + ay1) // 2)
        tag(f"{role}: {rel[key]}  occ {entry['occlusion']:.0%}", ax0, ay0 - 4,
            colour)

    if "S" in centres and "O" in centres:
        cv2.arrowedLine(canvas, centres["S"], centres["O"], (245, 245, 245), 1,
                        cv2.LINE_AA, tipLength=0.04)

    caption = (f"{rel['subject']} {rel['predicate']} {rel['object']}"
               f"   occ S {rel['subject_occlusion']:.2f} / "
               f"O {rel['object_occlusion']:.2f}   {position}")
    (w, h), _ = cv2.getTextSize(caption, font, 0.55, 1)
    y = canvas.shape[0] - 12
    cv2.rectangle(canvas, (8, y - h - 10), (18 + w, y + 6), (25, 25, 25), -1)
    cv2.putText(canvas, caption, (14, y), font, 0.55, (245, 245, 245), 1,
                cv2.LINE_AA)


def hud(frame: np.ndarray, rc: RobotController, hits: List[Dict[str, Any]],
        stale: bool, lo: float, hi: float,
        boxes: Optional[Dict[str, Dict[str, Any]]] = None,
        selected: int = 0, overlay: bool = True) -> np.ndarray:
    import cv2

    canvas = np.ascontiguousarray(frame[:, :, ::-1])       # RGB -> BGR
    if overlay and hits and boxes and not stale:
        # Not while stale: the boxes were measured at the previous pose, and
        # drawing them over a frame the robot has since driven away from puts
        # rectangles on the wrong objects.
        draw_triplet(canvas, hits[selected % len(hits)], boxes,
                     f"[{selected % len(hits) + 1}/{len(hits)}]")
    panel = np.zeros((canvas.shape[0], 420, 3), dtype=np.uint8)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.42

    def line(image, text, y, colour=(230, 230, 230)):
        cv2.putText(image, text, (10, y), font, scale, colour, 1, cv2.LINE_AA)

    pose = pose_of(rc)
    line(panel, f"x {pose['position']['x']:+.2f}  z {pose['position']['z']:+.2f}"
                f"  yaw {pose['yaw']:6.1f}", 24)
    line(panel, f"horizon {pose['horizon']:+.1f}   body {pose['body']}"
                f"   steps {rc.step_count}", 44)
    head = (f"{len(hits)} triplets in ({lo}, {hi})"
            + ("  [STALE - press SPACE]" if stale else ""))
    line(panel, head, 74, (120, 170, 255) if stale else (120, 255, 170))

    # The list is truncated to what fits above the key help, and says so -- an
    # overflowing list drew the last few triplets on top of the help text.
    top, row, floor = 98, 34, canvas.shape[0] - 60
    room = max(1, (floor - top) // row)
    # Scrolled so the selected triplet is always on the panel, since only that
    # one is boxed in the image and a selection you cannot see is confusing.
    first = 0 if not hits else max(0, min(selected % len(hits) - room // 2,
                                          len(hits) - room))
    shown = hits[first:first + room]
    y = top
    for offset, rel in enumerate(shown):
        current = hits and first + offset == selected % len(hits)
        mark = ">" if current else " "
        colour = (255, 255, 255) if current else (200, 200, 200)
        line(panel, f"{mark} {rel['subject']} {rel['predicate']} {rel['object']}",
             y, colour)
        line(panel, f"   occ {rel['subject_occlusion']:.2f} / "
                    f"{rel['object_occlusion']:.2f}", y + 15, (150, 150, 150))
        y += row
    if len(shown) < len(hits):
        line(panel, f"  ({first + 1}-{first + len(shown)} of {len(hits)})", y,
             (150, 150, 150))
    for index, text in enumerate(HELP):
        line(panel, text, canvas.shape[0] - 46 + index * 16, (140, 140, 140))
    return np.hstack([canvas, panel])


def drive(rc: RobotController, targets: Sequence[str], result: Dict[str, Any],
          fov: float, lo: float, hi: float, both: bool,
          outdir: Optional[str], scene: str, seed: int,
          selected: int = 0, show: str = "band") -> None:
    import cv2

    hits, stale, saves = selection(result, show), False, 0
    boxes = {o["name"]: o for o in result["objects"]}
    overlay = True
    print("\ndriving -- click the window first, then use the keys "
          "(n/b picks the triplet, SPACE re-measures, ESC quits)")
    while True:
        try:
            cv2.imshow("drive_triplet_scene",
                       hud(rc.event.frame, rc, hits, stale, lo, hi, boxes,
                           selected, overlay))
        except cv2.error as error:      # OpenCV built without GUI support
            print(f"  cannot open a window ({error}); "
                  f"re-run with --driver none")
            return
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("x")):
            break
        moved = True
        if key == ord("w"):
            rc.move_ahead()
        elif key == ord("s"):
            rc.move_back()
        elif key == ord("a"):
            rc.rotate(-rc.rotate_step)
        elif key == ord("d"):
            rc.rotate(rc.rotate_step)
        elif key == ord("q"):
            rc._step(action="MoveLeft", moveMagnitude=rc.move_magnitude)
        elif key == ord("e"):
            rc._step(action="MoveRight", moveMagnitude=rc.move_magnitude)
        elif key == ord("r"):
            rc.look(-15.0)
        elif key == ord("f"):
            rc.look(15.0)
        elif key == ord("c"):
            rc.set_height_level("crouch" if rc.height_level == "stand" else "stand")
        elif key == ord(" "):
            here = measure(rc, targets, fov)
            here["hits"] = [r for r in here["relations"]
                            if qualifying(r, lo, hi, both)]
            here["candidate"] = result["candidate"]
            here["removed"] = result.get("removed")
            report(here, lo, hi)
            hits = selection(here, show)
            result, boxes, selected = here, {o["name"]: o for o in here["objects"]}, 0
            moved, stale = False, False
        elif key in (ord("n"), ord("b")):
            if hits:
                selected = (selected + (1 if key == ord("n") else -1)) % len(hits)
                print(f"  [{selected + 1}/{len(hits)}] "
                      f"{triplet_str(hits[selected])}")
            moved = False
        elif key == ord("t"):
            overlay, moved = not overlay, False
        elif key == ord("p"):
            if outdir:
                sub = os.path.join(outdir, f"pose_{saves:02d}")
                save(result, sub, scene, seed, rc.event.frame, lo, hi, both,
                     selected, show)
                saves += 1
            else:
                print("  (no --out given, nothing saved)")
            moved = False
        else:
            moved = False
        if moved:
            stale = True
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------

def chosen_index(result: Dict[str, Any], args) -> int:
    """`--pick` beats `--triplet`, since naming the triplet is the stronger ask."""
    return (pick(selection(result, args.show), args.pick) if args.pick
            else args.triplet - 1)


def start_here(args) -> int:
    """
    Stand on a pose given on the command line, with no search at all.

    The pose is snapped to the navmesh, because THOR will happily `Teleport` the
    agent into a wall and every subsequent `MoveAhead` then fails.  No claim is
    made that anything is in the band here -- whatever is measured is printed,
    including nothing.
    """
    if not args.scene:
        print("--start needs --scene")
        return 1
    parts = args.start.split(",")
    if len(parts) not in (4, 5):
        print("--start wants X,Z,YAW,HORIZON[,BODY]")
        return 1
    x, z, yaw, horizon = (float(v) for v in parts[:4])
    body = parts[4].strip().lower() if len(parts) == 5 else "stand"
    if body not in RobotController.HEIGHT_LEVELS:
        print(f"BODY must be one of {list(RobotController.HEIGHT_LEVELS)}")
        return 1

    seed = 1 if args.seed is None else args.seed
    rc = RobotController(scene=args.scene, width=args.width, height=args.height,
                         field_of_view=args.fov, visibility_distance=15.0,
                         headless=args.headless, verbose=False)
    try:
        plan = rebuild(rc, seed)
        removed = remove_objects(rc, plan, args.remove)
        reachable = rc.get_reachable_positions()
        if len(reachable) == 0:
            print(f"{args.scene} has no reachable position")
            return 1
        # Nearest first, and keep trying: the navmesh is computed for the empty
        # scene, so a position can be reachable and still refuse a Teleport once
        # the clutter is standing on it.
        order = np.argsort(np.linalg.norm(
            reachable - np.array([[x, z]]), axis=1))[:8]
        result = None
        for index in order:
            spot = reachable[index]
            moved = math.dist((x, z), (float(spot[0]), float(spot[1])))
            pose = {"position": {"x": float(spot[0]),
                                 "y": float(rc.agent_position["y"]),
                                 "z": float(spot[1])},
                    "yaw": yaw, "horizon": horizon, "body": body}
            result = at_pose(rc, pose, plan["targets"], args.fov, args.lo,
                             args.hi, args.both_endpoints)
            if result["teleport_ok"]:
                if moved > 0.2:
                    print(f"  snapped to the navmesh, {moved:.2f} m from the "
                          f"pose asked for")
                break
        if result is None or not result["teleport_ok"]:
            print(f"  nothing near ({x:+.2f}, {z:+.2f}) accepts the agent")
            return 1
        result["candidate"] = {"start": "--start", "argument": args.start}
        result["removed"] = removed
        report(result, args.lo, args.hi, args.show)
        index = chosen_index(result, args)
        if args.out:
            save(result, args.out, args.scene, seed, rc.event.frame,
                 args.lo, args.hi, args.both_endpoints, index, args.show)
        if args.driver == "window":
            drive(rc, plan["targets"], result, args.fov, args.lo, args.hi,
                  args.both_endpoints, args.out, args.scene, seed,
                  index, args.show)
        return 0
    finally:
        rc.stop()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--resume", metavar="SCENARIO",
                    help="reopen a scenario.json written by an earlier run and "
                         "drive from its start pose; skips the search")
    ap.add_argument("--dataset", default="datasets/sgg/occlusion_ds4",
                    help="annotated build the candidate viewpoints come from")
    ap.add_argument("--scene", default=None, help="e.g. FloorPlan203")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--view", type=int, default=None, metavar="N",
                    help="probe only this dataset view index, instead of the "
                         "richest ones in order; see --list-views")
    ap.add_argument("--list-views", action="store_true",
                    help="print the qualifying dataset views and exit; opens "
                         "no THOR session")
    # Write it as --start=... : a leading minus in X makes argparse read the
    # value as another option name.
    ap.add_argument("--start", metavar="X,Z,YAW,HORIZON[,BODY]",
                    help="skip the search and stand here -- needs --scene, and "
                         "must be written --start=-3.5,-1,45,20 because of the "
                         "leading minus; BODY is stand (default) or crouch")
    ap.add_argument("--lo", type=float, default=0.4)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--both-endpoints", action="store_true",
                    help="require both endpoints inside the band")
    ap.add_argument("--probes", type=int, default=6,
                    help="dataset viewpoints to re-measure before giving up")
    ap.add_argument("--remove", nargs="*", default=[], metavar="WHAT",
                    help="take objects out of the scene before measuring; a "
                         "name, a THOR type or a VG150 class, e.g. --remove book")
    ap.add_argument("--cache", default=CACHE_DIR,
                    help="where winning poses are remembered, so the grid "
                         "search runs once per scene build")
    ap.add_argument("--rescan", action="store_true",
                    help="ignore the remembered pose and search again")
    ap.add_argument("--bodies", nargs="*", default=["stand", "crouch"],
                    choices=sorted(RobotController.HEIGHT_LEVELS),
                    help="body poses to try; the only two eye heights the "
                         "acting agent has")
    ap.add_argument("--yaw-jitter", type=float, nargs="*",
                    default=[0.0, -20.0, 20.0],
                    help="degrees off the aim at the triplet centroid")
    ap.add_argument("--horizon-jitter", type=float, nargs="*",
                    default=[0.0, -10.0])
    ap.add_argument("--radius", type=float, nargs=2, default=[1.2, 2.5],
                    metavar=("MIN", "MAX"),
                    help="how far the robot stands off the triplet, in metres")
    ap.add_argument("--positions", type=int, default=2,
                    help="standoff positions to try per candidate view")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=DEFAULT_FOV,
                    help="VERTICAL fov; the dataset's 60 by default, so the "
                         "occlusion figures stay comparable to it")
    ap.add_argument("--driver", choices=["window", "none"], default="window",
                    help="'none' measures the start pose, saves, and exits")
    ap.add_argument("--headless", action="store_true",
                    help="CloudRendering; implies --driver none")
    ap.add_argument("--out", default=None,
                    help="write scenario.json / start.png / triplet.png here")
    ap.add_argument("--triplet", type=int, default=1, metavar="N",
                    help="which triplet to box in triplet.png, 1-based in the "
                         "order --driver none prints them")
    ap.add_argument("--pick", metavar="PHRASE",
                    help="box the first triplet matching this text instead, "
                         "e.g. --pick 'plate behind vase'")
    ap.add_argument("--show", choices=["band", "all"], default="band",
                    help="which relations the overlay may box: only the in-band "
                         "ones (default) or every relation in view")
    ap.add_argument("--rate", default=str(VG_ANNOTATION_RATE), metavar="R",
                    help="share of object pairs the GT annotates (default "
                         f"{VG_ANNOTATION_RATE}); 'all' keeps every geometric "
                         "proposal -- a diagnostic, not comparable with "
                         "eval_occlusion.py")
    args = ap.parse_args(argv)

    global ANNOTATION_RATE
    ANNOTATION_RATE = (None if args.rate.strip().lower() in ("all", "none")
                       else float(args.rate))
    if ANNOTATION_RATE is None:
        print("--rate all: no annotation-rate cut, every geometric proposal "
              "kept -- not comparable with eval_occlusion.py")

    if args.headless:
        args.driver = "none"

    if args.resume:
        rc, plan, result = resume(args.resume, args.width, args.height, args.fov,
                                  args.headless, args.lo, args.hi,
                                  args.both_endpoints, args.remove)
        try:
            print(f"resumed {args.resume}")
            report(result, args.lo, args.hi, args.show)
            index = chosen_index(result, args)
            if args.out:
                # Writing the resumed state out is how an edit to it -- a
                # `--remove`, or a pose driven to by hand -- becomes a scenario
                # of its own instead of being lost on exit.
                save(result, args.out, result["scene"], result["seed"],
                     rc.event.frame, args.lo, args.hi, args.both_endpoints,
                     index, args.show)
            if args.driver == "window":
                drive(rc, plan["targets"], result, args.fov, args.lo, args.hi,
                      args.both_endpoints,
                      args.out or os.path.dirname(args.resume),
                      result["scene"], result["seed"], index, args.show)
            return 0
        finally:
            rc.stop()

    if args.start:
        return start_here(args)

    print(f"candidates from {args.dataset}, band ({args.lo}, {args.hi})"
          f"{', both endpoints' if args.both_endpoints else ''}")
    candidates = dataset_candidates(args.dataset, args.lo, args.hi,
                                    args.both_endpoints, args.scene, args.seed)
    if args.view is not None:
        candidates = [c for c in candidates if c["view"] == args.view]
    if not candidates:
        print("no qualifying view in the dataset for that filter")
        return 1
    scenes = {(c["scene"], c["seed"]) for c in candidates}
    print(f"{len(candidates)} qualifying views over {len(scenes)} scene builds")

    if args.list_views:
        for candidate in candidates:
            print(f"\n  {candidate['scene']} seed {candidate['seed']} "
                  f"view {candidate['view']:02d}  "
                  f"{candidate['n_dataset_hits']} in band")
            for text in candidate["dataset_hits"][:6]:
                print(f"      {text}")
            if len(candidate["dataset_hits"]) > 6:
                print(f"      ... {len(candidate['dataset_hits']) - 6} more")
        print(f"\nre-run with --scene <name> --seed <n> --view <n> to probe one")
        return 0

    # One THOR session per scene build, because rebuilding the clutter is the
    # expensive part; probing sticks to the first candidate's build.
    scene, seed = candidates[0]["scene"], candidates[0]["seed"]
    candidates = [c for c in candidates if (c["scene"], c["seed"]) == (scene, seed)]
    print(f"\nopening {scene} seed {seed} ({len(candidates)} candidate views)")

    remembered = cache_path(args.cache, scene, seed, args.view, args.lo,
                            args.hi, args.both_endpoints)
    if os.path.exists(remembered) and not args.rescan:
        print(f"reusing the pose in {remembered} (--rescan to search again)")

    rc = RobotController(scene=scene, width=args.width, height=args.height,
                         field_of_view=args.fov, visibility_distance=15.0,
                         headless=args.headless, verbose=False)
    try:
        started = time.time()
        if os.path.exists(remembered) and not args.rescan:
            with open(remembered) as fh:
                cached = json.load(fh)
            plan = rebuild(rc, seed)
            removed = remove_objects(rc, plan, args.remove)
            result = at_pose(rc, cached["start"], plan["targets"], args.fov,
                             args.lo, args.hi, args.both_endpoints)
            result["candidate"] = cached.get("source_view", {})
            result["plan"] = plan
            result["removed"] = removed
            if not result["hits"]:
                print("  ! the cached pose has nothing in band now -- "
                      "re-run with --rescan")
        else:
            result = choose(rc, candidates, args.fov, args.lo, args.hi,
                            args.both_endpoints, args.probes, args.bodies,
                            args.yaw_jitter, args.horizon_jitter,
                            tuple(args.radius), args.positions, args.remove)
            if result is None:
                print("\nno probed agent pose kept a triplet in the band -- "
                      "widen --yaw-jitter/--horizon-jitter or raise --probes")
                return 1
            os.makedirs(args.cache, exist_ok=True)
            with open(remembered, "w", encoding="utf-8") as fh:
                json.dump({"scene": scene, "seed": seed,
                           "arrange": DS4_ARRANGE,
                           "band": [args.lo, args.hi],
                           "both_endpoints": args.both_endpoints,
                           "start": result["pose"],
                           "source_view": {
                               k: v for k, v
                               in (result.get("candidate") or {}).items()
                               if k != "added"}}, fh, indent=1)
            print(f"  remembered the pose in {remembered}")
        print(f"\nready in {time.time() - started:.0f}s")
        report(result, args.lo, args.hi, args.show)
        index = chosen_index(result, args)
        if args.out:
            save(result, args.out, scene, seed, rc.event.frame,
                 args.lo, args.hi, args.both_endpoints, index, args.show)
        if args.driver == "window":
            drive(rc, result["plan"]["targets"], result, args.fov,
                  args.lo, args.hi, args.both_endpoints, args.out, scene, seed,
                  index, args.show)
        return 0
    finally:
        rc.stop()


if __name__ == "__main__":
    raise SystemExit(main())
