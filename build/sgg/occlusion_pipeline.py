"""
occlusion_pipeline.py -- generate cluttered scenes and measure how much each
object is occluded from each viewpoint.

    arrange     scatter objects over the real surfaces, physically settled
    sweep       render a set of viewpoints
    measure     per (object, viewpoint) occlusion, from instance masks
    select      keep the pairs in whatever band you want

The whole thing rests on `DisableObject`.  An object's occlusion is

    1 - visible_px / visible_px_with_every_other_moveable_object_disabled

so the denominator needs the object rendered alone against the static room.
`DisableObject` does exactly that -- it removes the object from the render, not
just from the metadata, `EnableObject` restores the frame to within one pixel,
and each call costs 21 ms against 38 ms for `SetObjectPoses`.

An earlier version believed `DisableObject` was broken and worked around it by
teleporting objects to (50, 50, 50) with `SetObjectPoses`.  That action rebuilds
the pose list for every moveable object and invalidates every objectId, so the
cost forced a compromise: objects were split in half into "targets" and
"occluders" and only the targets were ever measured.  The split threw away half
the data and could not see an object hidden behind another target at all.  It
was unnecessary -- the original test had passed a stale objectId captured before
an `InitialRandomSpawn` renumbered everything, and THOR reports that failure with
an empty error message.

Cost is (1 + candidates) sweeps, which is affordable because the disabled state
persists across viewpoints: disable everything once, then enable one object at a
time and sweep, rather than re-parking per viewpoint.
"""

from __future__ import annotations

# `python build/sgg/<script>.py` puts build/sgg/ on sys.path, not the repo root, so the
# root-level modules (vg150, vg_gt, ...) would not resolve.  Running as
# `python -m build.sgg.<script>` does not need this; it is here so both work.
import os as _os
import sys as _sys
if __package__ in (None, ""):
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import collections
import json
import math
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from build.sgg.occlusion import moveable, by_name

#: Counter-top clutter.  Counts are TOTALS for the scene, not additions, so a 4
#: means three new ones where the scene already had one.
SMALL_SPAWN = [
    ("Apple", 4), ("Bowl", 3), ("Mug", 4), ("Bottle", 3),
    ("Tomato", 4), ("Pot", 3), ("Cup", 3), ("Plate", 3),
]

#: Floor-standing furniture, which is what actually hides things -- the clutter
#: above spans 0.12-0.39 m and an apple cannot occlude much of anything.
#:
#: These need `allowMoveable=True` on `InitialRandomSpawn`.  All of them are
#: `moveable` but not `pickupable`, and the duplicator silently ignores that
#: class by default: asking for 3 ShelvingUnits returns the 1 already in the
#: scene, with no error.
LARGE_SPAWN = [
    ("HousePlant", 3), ("ShelvingUnit", 2), ("GarbageCan", 2),
]

#: Horizontal extent above which an object belongs on the floor, not a worktop.
LARGE_EXTENT_M = 0.50


def auto_spawn(event, small_extra: int = 16, large_extra: int = 6
               ) -> List[Tuple[str, int]]:
    """
    Duplicate counts chosen from what the scene actually contains.

    The hand-written lists above are kitchen furniture and are simply inert
    anywhere else: `InitialRandomSpawn` duplicates existing objects rather than
    creating them from nothing, so asking a living room for four Mugs returns
    nothing at all and reports no error.  Living rooms hold a better set anyway --
    Sofa 2.76 m, DiningTable 2.04 m, ArmChair 1.97 m against the kitchen's largest
    moveable object at 1.05 m.

    Types with no VG150 mapping are skipped.  A living room's most numerous small
    objects are Statue, CreditCard, KeyChain, RemoteControl and RoomDecor, none of
    which VG150 can name, so they could occlude but never be scored -- and
    spending the placement budget on them crowds out the objects that can be.

    Counts returned are TOTALS, as `numDuplicatesOfType` expects, so an existing
    population of two asking for four more comes back as six.
    """
    from vg.vg150 import THOR_TO_VG150

    present: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for entry in moveable(event):
        if entry["objectType"] in THOR_TO_VG150:
            present[entry["objectType"]].append(entry)

    big = {t: v for t, v in present.items()
           if max(extent(e) for e in v) >= LARGE_EXTENT_M}
    small = {t: v for t, v in present.items() if t not in big}

    out: List[Tuple[str, int]] = []
    for group, extra in ((big, large_extra), (small, small_extra)):
        if not group:
            continue
        # Biggest first among the furniture, since occluding power goes with
        # footprint; the small ones are shared out evenly.
        order = sorted(group, key=lambda t: -max(extent(e) for e in group[t]))
        share = max(1, extra // len(order))
        for kind in order:
            out.append((kind, len(group[kind]) + share))
    return out

#: Slack when testing whether a footprint corner still lands on the surface.  The
#: spawn clouds are dense -- median nearest-neighbour spacing 0.031 m on the big
#: counter, 0.055 m on the small one -- so a corner genuinely over the surface
#: always has a sample within this radius and one hanging off the edge has none.
FOOTPRINT_TOL_M = 0.08

#: Unoccluded size below which "50% occluded" is meaningless and the ratio is
#: dominated by mask aliasing.  Derived, not chosen: `vg_gt.MIN_LINEAR_EXTENT` is
#: 0.06 of the frame's linear size, measured from what VG's annotators were
#: willing to draw a box around, so the area floor is 0.06^2 of the frame.  A
#: flat 200 px was used at first and filled the "usable" band with apples 20
#: pixels across, whose 93% occlusion was really just distance.
MIN_EXTENT_FRACTION = 0.06 ** 2

BANDS = [(0.0, 0.05), (0.05, 0.25), (0.25, 0.50),
         (0.50, 0.75), (0.75, 0.95), (0.95, 1.01)]


# --------------------------------------------------------------------------
# arrange
# --------------------------------------------------------------------------

def extent(entry: Dict[str, Any]) -> float:
    size = entry["axisAlignedBoundingBox"]["size"]
    return math.hypot(size["x"], size["z"])


def surfaces_near(event, origin: Dict[str, float],
                  types=("CounterTop", "DiningTable", "CoffeeTable", "SideTable",
                         "Desk", "Shelf")) -> List[Dict]:
    """Placement surfaces ordered by distance from a point."""
    found = [o for o in event.metadata["objects"] if o["objectType"] in types]
    return sorted(found, key=lambda o: math.dist(
        (o["position"]["x"], o["position"]["z"]), (origin["x"], origin["z"])))


def spawn_points(controller, receptacle_id: str) -> List[Dict[str, float]]:
    event = controller.step(action="GetSpawnCoordinatesAboveReceptacle",
                            objectId=receptacle_id, anywhere=True)
    if not event.metadata["lastActionSuccess"]:
        return []
    return event.metadata["actionReturn"] or []


def interior_slots(points: Sequence[Dict[str, Any]],
                   half_extent: float) -> List[Dict[str, Any]]:
    """
    Keep the points where an object of this size sits FULLY on the surface.

    `GetSpawnCoordinatesAboveReceptacle` gives positions for an object's centre
    and says nothing about its footprint, and the cloud runs right to the edge --
    it spans the counter's bounding box to within 1-2 cm.  A bowl on an edge
    point overhangs by its own radius, which is what made objects look about to
    fall off.

    Two tests, because neither is sufficient alone.  The bounding-box inset is
    exact on a rectangular counter and is the only test with any resolution for
    small objects: the cloud test cannot see an overhang narrower than the
    sampling spacing, so at a 0.08 m tolerance it passed all 2205 points for a
    0.15 m apple.  The cloud test in turn catches the L-shaped counters, whose
    axis-aligned boxes take in a lot of empty floor that the inset would happily
    place a bowl on.
    """
    if not points:
        return []
    cell = FOOTPRINT_TOL_M
    occupied = {(int(math.floor(p["x"] / cell)), int(math.floor(p["z"] / cell)))
                for p in points}

    def covered(x: float, z: float) -> bool:
        cx, cz = int(math.floor(x / cell)), int(math.floor(z / cell))
        return any((cx + dx, cz + dz) in occupied
                   for dx in (-1, 0, 1) for dz in (-1, 0, 1))

    h = half_extent
    probes = ((h, h), (h, -h), (-h, h), (-h, -h),
              (h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h))
    kept = []
    for point in points:
        box = point.get("surface")
        if box is not None:
            centre, size = box["center"], box["size"]
            if (abs(point["x"] - centre["x"]) + h > size["x"] / 2.0
                    or abs(point["z"] - centre["z"]) + h > size["z"] / 2.0):
                continue
        if all(covered(point["x"] + dx, point["z"] + dz) for dx, dz in probes):
            kept.append(point)
    return kept


def park_agent(controller, camera: Dict[str, float], yaw_deg: float,
               reachable: Optional[Sequence[Dict[str, float]]] = None) -> bool:
    """
    Move the agent out of shot, behind the camera.

    Third-party cameras render the agent's body -- it is the robot-looking figure
    that turned up in the living-room frames, and it cost 4.3% of the pixels in
    one 800x800 view.  It is worse than cosmetic: the agent occludes real objects,
    so some of the "occlusion" being measured was the agent's own body, and since
    it has no entry in `metadata["objects"]` it appears in no instance mask and
    `DisableObject` cannot touch it.

    Anything strictly behind the camera plane is outside the frustum for any field
    of view under 180 degrees, so the most negative projection onto the camera's
    forward axis is the safest spot.  `export_multiview` used "farthest reachable
    position" instead, which is only a heuristic -- it left a residual difference
    of 0.37-0.89 grey levels because in a small room the farthest point can still
    be in frame.
    """
    if reachable is None:
        reachable = controller.step(
            action="GetReachablePositions").metadata["actionReturn"] or []
    if not reachable:
        return False
    rad = math.radians(yaw_deg)
    fx, fz = math.sin(rad), math.cos(rad)

    def behindness(point: Dict[str, float]) -> float:
        dx, dz = point["x"] - camera["x"], point["z"] - camera["z"]
        norm = math.hypot(dx, dz) or 1e-6
        return -(dx * fx + dz * fz) / norm

    spot = max(reachable, key=behindness)
    if behindness(spot) <= 0.0:
        # Camera backed into a corner with no reachable floor behind it; fall
        # back to whatever is farthest and accept the risk.
        spot = max(reachable, key=lambda p: math.dist((p["x"], p["z"]),
                                                      (camera["x"], camera["z"])))
    return controller.step(action="Teleport", position=dict(spot),
                           rotation={"x": 0.0, "y": yaw_deg, "z": 0.0},
                           horizon=0.0).metadata["lastActionSuccess"]


def footprint_xz(entry: Dict[str, Any]) -> Tuple[float, float]:
    size = entry["axisAlignedBoundingBox"]["size"]
    return size["x"] / 2.0, size["z"] / 2.0


def unoccupied_slots(points: Sequence[Dict[str, Any]],
                     occupants: Sequence[Dict[str, Any]],
                     half_extent: float,
                     same_level_m: float = 0.35) -> List[Dict[str, Any]]:
    """
    Drop the slots something is already sitting in.

    `interior_slots` only asks whether an object fits on the SURFACE; it does not
    look at what is standing there.  `PlaceObjectAtPoint` does no collision test
    either -- asked for an occupied spot it reports success and resolves the
    interpenetration by shoving the newcomer upward, where it seats with
    `parentReceptacles` pointing at whatever was underneath.  The ground truth
    then asserts `bowl on cup`, which is true of the simulator and absurd as a
    description of a kitchen.

    Ordering the placements largest-first is not sufficient, and measuring showed
    why: only the DUPLICATES are placed, while the scene's own objects are
    restored to their stock positions beforehand.  A mug that came with the room
    is in its slot before any sort can matter.  This filter is what actually
    prevents the stack, because it treats stock objects and already-placed
    duplicates alike.

    `same_level_m` keeps a bowl on the floor from blocking a slot on the counter
    above it -- only occupants at roughly the slot's own height count.
    """
    if not occupants:
        return list(points)
    out = []
    for point in points:
        clear = True
        for entry in occupants:
            centre = entry["position"]
            if abs(centre["y"] - point["y"]) > same_level_m:
                continue
            reach = half_extent + extent(entry) / 2.0
            if math.dist((point["x"], point["z"]), (centre["x"], centre["z"])) < reach:
                clear = False
                break
        if clear:
            out.append(point)
    return out


def free_floor_slots(reachable: Sequence[Dict[str, float]],
                     obstacles: Sequence[Dict[str, Any]],
                     half_x: float, half_z: float,
                     grid: float = 0.30) -> List[Dict[str, float]]:
    """
    Floor positions where a piece of furniture of this size actually fits.

    `GetReachablePositions` says the AGENT can stand somewhere, which is a much
    weaker claim than a sofa fitting there.  Placing 2 m furniture straight onto
    those points put lamps in the middle of the floor and pushed chairs through
    walls and pillars, because `PlaceObjectAtPoint` does no collision test of its
    own -- it will happily seat an object intersecting the geometry.

    Two conditions, mirroring `interior_slots`: the footprint's corners must all
    lie over reachable floor (so nothing pokes through a wall), and the footprint
    must not overlap anything already standing there.
    """
    if not reachable:
        return []
    cell = grid
    occupied = {(int(math.floor(p["x"] / cell)), int(math.floor(p["z"] / cell)))
                for p in reachable}

    def on_floor(x: float, z: float) -> bool:
        cx, cz = int(math.floor(x / cell)), int(math.floor(z / cell))
        return any((cx + dx, cz + dz) in occupied
                   for dx in (-1, 0, 1) for dz in (-1, 0, 1))

    boxes = []
    for entry in obstacles:
        box = entry["axisAlignedBoundingBox"]
        centre, size = box["center"], box["size"]
        boxes.append((centre["x"], centre["z"], size["x"] / 2.0, size["z"] / 2.0))

    probes = ((half_x, half_z), (half_x, -half_z), (-half_x, half_z),
              (-half_x, -half_z), (half_x, 0.0), (-half_x, 0.0),
              (0.0, half_z), (0.0, -half_z))
    out = []
    for point in reachable:
        if not all(on_floor(point["x"] + dx, point["z"] + dz) for dx, dz in probes):
            continue
        if any(abs(point["x"] - bx) < half_x + hx and abs(point["z"] - bz) < half_z + hz
               for bx, bz, hx, hz in boxes):
            continue
        out.append(point)
    return out


def snapshot(event) -> Dict[str, Dict[str, Dict[str, float]]]:
    return {o["name"]: {"position": dict(o["position"]),
                        "rotation": dict(o["rotation"])}
            for o in moveable(event)}


def restore(controller, poses: Dict[str, Dict[str, Dict[str, float]]]) -> bool:
    """
    Put the named objects back where they were.

    Kinematic restoration through `SetObjectPoses` is safe here, and only here,
    because these coordinates were read from a settled scene: an object cannot
    fall out of a pose it was already resting in.  The action takes the complete
    set and deletes anything omitted, so every moveable object is listed.
    """
    entries = []
    for entry in moveable(controller.last_event):
        home = poses.get(entry["name"])
        entries.append({"objectName": entry["name"],
                        "position": dict(home["position"]) if home
                        else dict(entry["position"]),
                        "rotation": dict(home["rotation"]) if home
                        else dict(entry["rotation"])})
    return controller.step(action="SetObjectPoses",
                           objectPoses=entries).metadata["lastActionSuccess"]


def arrange(controller, seed: int = 0, focus: Optional[Dict[str, float]] = None,
            n_surfaces: int = 2, spacing: float = 0.18, max_objects: int = 40,
            use_floor: bool = True, spawn_scale: int = 1,
            floor_radius: float = 3.0,
            small_extra: Optional[int] = None,
            large_extra: Optional[int] = None) -> Dict[str, Any]:
    """
    Add clutter to the scene as it stands, leaving everything physically settled.

    The scene's own objects are put back exactly where they were and only the
    freshly spawned duplicates are positioned.  An earlier version randomised
    everything, on the theory that the layout should be a property of the seed
    rather than of the stock scene.  That is true of a kitchen worktop and false
    of a living room: rearranging FloorPlan205 scattered its lamps and cushions
    across the middle of the floor and pushed chairs through the walls, because a
    sofa's sensible position is a fact about the room that the generator has no
    way to rediscover.  The stock layout is a good initialisation; clutter goes on
    top of it.

    `InitialRandomSpawn` relocates about 30 of the scene's 77 objects as a side
    effect of duplicating anything, so "leave the originals alone" means
    snapshotting them beforehand and restoring them after.

    Placement goes through `PlaceObjectAtPoint`, never `SetObjectPoses`.
    `SetObjectPoses` is purely kinematic and cannot be talked out of it: it
    silently ignores a `placeStationary` argument, and an object it lifts 0.25 m
    above a counter is still 0.24 m up after two simulated seconds, with or
    without `MakeAllObjectsMoveable`.  `PlaceObjectAtPoint` resolves the request
    against the surface -- ask for 0.30 m above a counter and the object lands ON
    it with `parentReceptacles` set -- and takes a LIST of candidates, walking it
    until one works.

    Its `lastActionSuccess` is not trustworthy on its own.  Asked to place into an
    occupied spot it reports success and resolves the interpenetration by shoving
    the object upwards; a Mug came to rest at y=2.11 over a 1.12 m counter with
    nothing beneath it, and ejected objects come out kinematic so they never fall
    back -- ten accumulated seconds of physics moved none of them.  The only
    reliable check is a non-empty `parentReceptacles`, so placement is verified,
    retried once, and finally disabled if it still will not seat.
    """
    rng = random.Random(seed)
    focus = focus or controller.last_event.metadata["agent"]["position"]

    before = snapshot(controller.last_event)
    spawn = auto_spawn(
        controller.last_event,
        small_extra=(16 * spawn_scale if small_extra is None else small_extra),
        large_extra=0 if not use_floor
        else (6 * spawn_scale if large_extra is None else large_extra))
    controller.step(action="InitialRandomSpawn", randomSeed=seed, forceVisible=True,
                    placeStationary=False, numPlacementAttempts=25,
                    allowFloor=use_floor, allowMoveable=use_floor,
                    numDuplicatesOfType=[{"objectType": t, "count": n}
                                         for t, n in spawn])
    restore(controller, before)
    duplicates = [o["name"] for o in moveable(controller.last_event)
                  if o["name"] not in before]

    slots: List[Dict[str, Any]] = []
    for surface in surfaces_near(controller.last_event, focus)[:n_surfaces]:
        box = surface["axisAlignedBoundingBox"]
        centre = box["center"]
        points = spawn_points(controller, surface["objectId"])
        points.sort(key=lambda p: math.dist((p["x"], p["z"]),
                                            (centre["x"], centre["z"])))
        kept: List[Dict[str, Any]] = []
        for point in points:
            if all(math.dist((point["x"], point["z"]), (k["x"], k["z"])) >= spacing
                   for k in kept):
                kept.append({**point, "surface": box})
        slots.extend(kept)
    rng.shuffle(slots)

    reachable: List[Dict[str, float]] = []
    if use_floor:
        found = controller.step(
            action="GetReachablePositions").metadata["actionReturn"] or []
        reachable = [p for p in found
                     if math.dist((p["x"], p["z"]), (focus["x"], focus["z"]))
                     <= floor_radius]
        rng.shuffle(reachable)

    placed: List[str] = []
    entries = {o["name"]: o for o in moveable(controller.last_event)}
    movers = [n for n in duplicates if n in entries]
    rng.shuffle(movers)
    movers = movers[:max_objects]

    big = [n for n in movers if extent(entries[n]) >= LARGE_EXTENT_M]
    if not use_floor:
        big = []
        movers = [n for n in movers if extent(entries[n]) < LARGE_EXTENT_M]
    small = [n for n in movers if n not in big]

    # Largest first WITHIN each group, not just floor-before-counter.
    #
    # `PlaceObjectAtPoint` resolves an occupied slot by shoving the newcomer
    # upward, and it seats there: `parentReceptacles` comes back pointing at
    # whatever was already in the spot.  With `small` in shuffled order that
    # means a Plate can land on an Apple, and the ground truth then asserts
    # `plate on apple` -- true of the simulator, absurd as a scene description.
    #
    # Measured on `occlusion_ds3` (shuffled): 3 of 396 support relations had a
    # host smaller than its subject -- `vase on plate` x2, `bowl on cup` x1.
    # Rare, but each one is a guaranteed-wrong annotation on the single predicate
    # the benchmark scores best, so it is worth the sort.
    #
    # Descending order makes the stacking physical instead: the Plate is placed
    # while the slot is clear, and an Apple arriving later settles ON it, which
    # is a relation VG annotates constantly (`apple on plate`).  Sorting rather
    # than shuffling does not bias WHERE things go -- `slots` is already
    # shuffled, so the per-object window is a random region either way.
    big.sort(key=lambda n: -extent(entries[n]))
    small.sort(key=lambda n: -extent(entries[n]))

    # Obstacles for the floor test: everything already standing in the room,
    # minus the duplicates, which are still wherever the spawner dropped them.
    standing = [o for o in controller.last_event.metadata["objects"]
                if o["name"] not in duplicates]

    def occupants(exclude: str) -> List[Dict[str, Any]]:
        """Small objects currently resting on something, minus the one being placed."""
        return [o for o in controller.last_event.metadata["objects"]
                if o["name"] != exclude
                and o["parentReceptacles"]
                and extent(o) < LARGE_EXTENT_M
                and (o["name"] not in duplicates or o["name"] in placed)]

    def candidates(name: str, offset: int) -> List[Dict[str, float]]:
        if name in big:
            half_x, half_z = footprint_xz(entries[name])
            usable = free_floor_slots(reachable, standing, half_x, half_z)
            if not usable:
                return []
            span = max(1, len(usable) // max(1, len(big)))
            window = usable[offset * span:(offset + 1) * span] or usable
            return [{"x": p["x"], "y": entries[name]["position"]["y"], "z": p["z"]}
                    for p in window]
        # Filtered per object: the safe region shrinks with the footprint, so a
        # Pot (0.35 m) has far less of the counter available than an Apple.
        half = extent(entries[name]) / 2.0
        usable = interior_slots(slots, half) or slots
        # ... then drop whatever is already sitting there.  Read fresh from the
        # simulator so it sees both the scene's stock objects and the duplicates
        # placed earlier in this same loop.
        usable = unoccupied_slots(usable, occupants(name), half) or usable
        span = max(1, len(usable) // max(1, len(small)))
        window = usable[offset * span:(offset + 1) * span] or usable
        return [{"x": s["x"], "y": s["y"], "z": s["z"]} for s in window]

    def seated(name: str) -> bool:
        entry = by_name(controller.last_event, name)
        return bool(entry and entry["parentReceptacles"])

    failed: List[str] = []
    # Furniture first: it needs the most room, and a floor slot taken by a stool
    # is one a bowl was never going to want.
    for index, name in enumerate(big + small):
        offset = index if name in big else index - len(big)
        event = controller.step(action="PlaceObjectAtPoint",
                                objectId=entries[name]["objectId"],
                                positions=candidates(name, offset),
                                forceKinematic=False)
        (placed if event.metadata["lastActionSuccess"] else failed).append(name)
    controller.step(action="AdvancePhysicsStep", simSeconds=1.0)

    failed += [n for n in placed if not seated(n)]
    placed = [n for n in placed if seated(n)]
    for offset, name in enumerate(failed):
        event = controller.step(action="PlaceObjectAtPoint",
                                objectId=entries[name]["objectId"],
                                positions=candidates(name, offset + 3),
                                forceKinematic=False)
        if event.metadata["lastActionSuccess"]:
            placed.append(name)
    controller.step(action="AdvancePhysicsStep", simSeconds=1.0)

    dropped = [n for n in placed if not seated(n)] + [n for n in failed
                                                      if n not in placed]
    placed = [n for n in placed if seated(n)]
    for name in set(dropped):
        # Disabled rather than left in mid-air, so nothing floats in the render
        # and nothing occludes from a position no real object could occupy.
        entry = by_name(controller.last_event, name)
        if entry:
            controller.step(action="DisableObject", objectId=entry["objectId"])

    # Targets are every object VG150 can name, not just the ones added.  The
    # scene's own sofa hidden behind an added chair is exactly the sample wanted,
    # and per-object references make it measurable at no extra cost.
    #
    # This used to be restricted to `moveable()`, on the assumption that the
    # amodal pass -- which disables everything and re-enables one object at a
    # time -- could only toggle moveable objects.  It cannot: `DisableObject`
    # succeeds on `Cabinet`, `CounterTop`, `Drawer`, `Fridge`, `Sink` and
    # `Window`, all of which report `moveable=False, pickupable=False`.
    #
    # The restriction cost the benchmark most of its vocabulary.  `occlusion_ds2`
    # carries 18 VG150 classes against the 38 the same scenes contain, and none
    # of them is a fixed fitting, so no relation could ever mention a counter, a
    # cabinet or a window.  That is also why the vertical family was empty:
    # `drawer under counter` and `cabinet above counter` are the configurations
    # VG annotates most (`drawer under counter` is 80% of 84 human labels), and
    # both endpoints are static.  Relabelling could not have recovered them --
    # the objects were not in the file.
    from vg.vg150 import THOR_TO_VG150
    disabled = set(dropped)
    targets = sorted(o["name"] for o in controller.last_event.metadata["objects"]
                     if o["name"] not in disabled
                     and o["objectType"] in THOR_TO_VG150)

    return {"placed": sorted(placed), "dropped": sorted(disabled),
            "targets": targets, "slots": len(slots)}


# --------------------------------------------------------------------------
# sweep and measure
# --------------------------------------------------------------------------

def viewpoints(controller, yaws: Sequence[float] = (0, 45, 90, 135, 180, 225, 270, 315),
               horizons: Sequence[float] = (0, 20),
               near: Optional[Dict[str, float]] = None, radius: float = 2.5,
               limit: Optional[int] = None, seed: int = 0) -> List[Dict[str, Any]]:
    """
    Reachable floor positions crossed with yaw and pitch, near the arrangement.

    Restricted by radius because a pose on the far side of the room sees none of
    the clutter and costs a render to learn that.
    """
    positions = controller.step(
        action="GetReachablePositions").metadata["actionReturn"] or []
    if near is not None:
        positions = [p for p in positions
                     if math.dist((p["x"], p["z"]), (near["x"], near["z"])) <= radius]
    poses = [{"position": p, "yaw": float(y), "horizon": float(h)}
             for p in positions for y in yaws for h in horizons]
    if limit is not None and len(poses) > limit:
        poses = random.Random(seed).sample(poses, limit)
    return poses


def sweep(controller, poses: Sequence[Dict[str, Any]]) -> List[Dict[str, int]]:
    """Render every pose; return {object name: visible pixels} for each."""
    frames = []
    for pose in poses:
        controller.step(action="Teleport", position=pose["position"],
                        rotation={"x": 0.0, "y": pose["yaw"], "z": 0.0},
                        horizon=pose["horizon"])
        event = controller.last_event
        names = {o["objectId"]: o["name"] for o in event.metadata["objects"]}
        frames.append({names[k]: int(v.sum())
                       for k, v in event.instance_masks.items() if k in names})
    return frames


def measure(controller, poses: Sequence[Dict[str, Any]], targets: Sequence[str],
            min_px: int) -> List[Dict[str, Any]]:
    """
    Occlusion of every target at every viewpoint.

    The disabled state survives a `Teleport`, so the whole viewpoint list can be
    swept once per target rather than toggling per viewpoint: disable every
    moveable object, then bring them back one at a time.  Each of those sweeps
    shows one target alone against the static room, which is its unoccluded
    reference.
    """
    observed = sweep(controller, poses)

    ids = {o["name"]: o["objectId"] for o in moveable(controller.last_event)}
    for name in ids:
        controller.step(action="DisableObject", objectId=ids[name])

    rows = []
    for name in targets:
        if name not in ids:
            continue
        controller.step(action="EnableObject", objectId=ids[name])
        alone = sweep(controller, poses)
        controller.step(action="DisableObject", objectId=ids[name])
        for index, frame in enumerate(alone):
            reference = frame.get(name, 0)
            # A zero reference means the target is not in this view at all, so
            # there is nothing to be occluded and no ratio to take.
            if reference < max(min_px, 1):
                continue
            visible = observed[index].get(name, 0)
            rows.append({"view": index, "name": name,
                         "reference_px": reference, "visible_px": visible,
                         "occlusion": round(max(0.0, 1.0 - visible / reference), 4)})

    for name in ids:
        controller.step(action="EnableObject", objectId=ids[name])
    return rows


# --------------------------------------------------------------------------
# select and report
# --------------------------------------------------------------------------

def histogram(rows: Sequence[Dict[str, Any]]) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for row in rows:
        for lo, hi in BANDS:
            if lo <= row["occlusion"] < hi:
                counts[(lo, hi)] += 1
                break
    return counts


def select(rows: Sequence[Dict[str, Any]], low: float, high: float
           ) -> List[Dict[str, Any]]:
    return [r for r in rows if low <= r["occlusion"] < high]


def run(scene: str, seed: int, n_views: int, size: int, headless: bool,
        **options) -> Dict[str, Any]:
    from ai2thor.controller import Controller

    kwargs = dict(scene=scene, width=size, height=size,
                  renderInstanceSegmentation=True, visibilityDistance=15.0)
    if headless:
        kwargs["platform"] = "CloudRendering"
    controller = Controller(**kwargs)
    try:
        focus = surfaces_near(controller.last_event,
                              controller.last_event.metadata["agent"]["position"]
                              )[0]["position"]
        poses = viewpoints(controller, near=focus, limit=n_views, seed=seed,
                           radius=options.pop("radius", 2.5))
        if not poses:
            return {"error": "no reachable viewpoints near the arrangement"}
        plan = arrange(controller, seed=seed, focus=focus, **options)
        rows = measure(controller, poses, plan["targets"],
                       int(MIN_EXTENT_FRACTION * size * size))
        return {"scene": scene, "seed": seed, "poses": poses, "plan": plan,
                "rows": rows}
    finally:
        controller.stop()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--views", type=int, default=40)
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--surfaces", type=int, default=2)
    parser.add_argument("--spacing", type=float, default=0.18)
    parser.add_argument("--spawn-scale", type=int, default=1)
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--no-floor", action="store_true",
                        help="counter-top clutter only")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    result = run(args.scene, args.seed, args.views, args.size, args.headless,
                 n_surfaces=args.surfaces, spacing=args.spacing,
                 spawn_scale=args.spawn_scale, radius=args.radius,
                 use_floor=not args.no_floor)
    if "error" in result:
        print(result["error"])
        return 1

    rows, plan, poses = result["rows"], result["plan"], result["poses"]
    print(f"{args.scene}  seed={args.seed}  {len(poses)} viewpoints  "
          f"{len(plan['placed'])} added, {len(plan['dropped'])} dropped, "
          f"{len(plan['targets'])} targets")
    print(f"{len(rows)} (object, viewpoint) samples above the size floor "
          f"({int(MIN_EXTENT_FRACTION * args.size * args.size)} px)\n")

    counts = histogram(rows)
    total = sum(counts.values()) or 1
    print(f"{'occlusion band':<20}{'samples':>9}{'share':>8}")
    for lo, hi in BANDS:
        n = counts[(lo, hi)]
        print(f"  {lo:.2f} - {hi:.2f}      {n:>9}{n / total * 100:>7.1f}%")

    usable = select(rows, 0.25, 0.95)
    print(f"\nusable band 0.25-0.95: {len(usable)} samples, "
          f"{len({r['name'] for r in usable})} objects, "
          f"{len({r['view'] for r in usable})} viewpoints")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
