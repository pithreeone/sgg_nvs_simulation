"""
proc_scene.py -- a bare room, a table, and exactly the objects we put on it.

The iTHOR floor plans were fought for a day and lost on three counts.  A scene
holds no two instances of one objectType, so "the bottle behind the plant" never
had a second bottle to disambiguate from and the relation in the instruction did
no work.  An L-shaped counter's bounding box spans the empty inner corner, so
staged objects hovered in mid-air.  And an occluder was whatever the room
happened to offer, which was a house plant covering a fifth of the frame often
enough to make `behind` vacuous.

A procedural scene fixes all three by construction: the room is a rectangle, the
table is a rectangle on it, and every object is one we chose and placed.  The
price is that the scenes are bare -- one room, one table, three objects -- so
anything measured here is about perception under controlled geometry, not about
a robot in a house.  Say so wherever these numbers appear.

THREE THINGS THE PROCEDURAL BACKEND DOES DIFFERENTLY, all found the hard way:

  `snapToGrid` must be False.  A procedural house has no navmesh --
  `GetReachablePositions` returns zero points -- and Teleport with snapping
  indexes that empty array, failing with `ArgumentOutOfRangeException` and
  leaving the agent at its spawn point of y = -38.86, outside the world.  Every
  frame then renders the skybox, which looks exactly like a scene that did not
  build.

  `event.instance_masks` is incomplete.  It lists the structural pieces --
  walls, floor -- and omits everything spawned, so it reports zero pixels for
  objects that are plainly in the picture.  The underlying data is fine:
  `instance_segmentation_frame` paints each spawned object its own colour and
  `color_to_object_id` maps it back.  `visible_pixels` here reads those instead,
  and gives per-INSTANCE counts even for two copies of one asset.

  `SpawnAsset` positions the PIVOT, not the base, and nothing settles it: a
  procedural spawn takes no gravity pass, so an asset whose pivot sits at the
  centre of its bounding box ends up half inside the table and stays there.
  `place_on` measures the spawned box and lifts.  See its docstring.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The room, metres.  Big enough to stand 1-2 m off a table without backing into
#: a wall, small enough that the walls give the sweep something to parallax
#: against -- an object against an infinite void has no depth cues at all.
ROOM = 4.0
WALL_HEIGHT = 2.5

#: Materials must exist in `GetMaterials`; `CeilingDrywall` does not, and a
#: missing name fails house creation with an unhelpful "key not present".
FLOOR_MATERIAL = "DarkWoodFloors"
WALL_MATERIAL = "bathroomTilesGrey"


def house(room: float = ROOM, height: float = WALL_HEIGHT) -> Dict[str, Any]:
    """An empty square room.  Objects go in afterwards, with `spawn`."""
    corners = [(0, 0), (0, room), (room, room), (room, 0)]
    walls = [{"id": f"wall|{i}", "roomId": "room|0",
              "material": {"name": WALL_MATERIAL},
              "polygon": [{"x": a[0], "y": 0, "z": a[1]},
                          {"x": b[0], "y": 0, "z": b[1]},
                          {"x": b[0], "y": height, "z": b[1]},
                          {"x": a[0], "y": height, "z": a[1]}]}
             for i, (a, b) in enumerate(zip(corners, corners[1:] + corners[:1]))]
    return {
        "id": "tabletop", "doors": [], "windows": [], "objects": [],
        "walls": walls,
        "rooms": [{"id": "room|0", "roomType": "LivingRoom",
                   "floorMaterial": {"name": FLOOR_MATERIAL},
                   "floorPolygon": [{"x": x, "y": 0, "z": z} for x, z in corners],
                   "ceilings": [], "children": []}],
        "proceduralParameters": {"ceilingMaterial": {"name": FLOOR_MATERIAL},
                                 "floorColliderThickness": 1.0, "lights": [],
                                 "reflections": [], "skyboxId": "Sky1"},
        "metadata": {"schema": "1.0.0"},
    }


def open_room(width: int = 800, height: int = 600, fov: float = 60.0,
              room: float = ROOM):
    """A controller sitting in an empty room, ready to be furnished.

    `THOR_HEADLESS=1` SELECTS CloudRendering, which draws through Vulkan and
    needs no X server -- the one change a machine without a display requires.
    Left unset the platform is chosen by ai2thor as before, so a desktop run is
    untouched.  The GPU must have Vulkan drivers; CloudRendering fails at
    startup rather than falling back if it does not.
    """
    from ai2thor.controller import Controller

    extra = {}
    if os.environ.get("THOR_HEADLESS") == "1":
        from ai2thor.platform import CloudRendering

        extra["platform"] = CloudRendering

    return Controller(scene=house(room), width=width, height=height,
                      fieldOfView=fov, renderDepthImage=True,
                      renderInstanceSegmentation=True,
                      # Not optional; see the module docstring.
                      snapToGrid=False, **extra)


def spawn(controller, asset: str, name: str, x: float, y: float, z: float,
          rotation: float = 0.0) -> Dict[str, Any]:
    """Place one asset and return its metadata entry."""
    event = controller.step(action="SpawnAsset", assetId=asset,
                            generatedId=name,
                            position={"x": float(x), "y": float(y),
                                      "z": float(z)},
                            rotation={"x": 0, "y": float(rotation), "z": 0})
    if not event.metadata["lastActionSuccess"]:
        raise RuntimeError(f"SpawnAsset {asset} as {name}: "
                           f"{event.metadata.get('errorMessage')}")
    return next(o for o in event.metadata["objects"] if o["name"] == name)


def surface_top(entry: Dict[str, Any]) -> float:
    """The y a thing standing ON this object rests at."""
    box = entry["axisAlignedBoundingBox"]
    return box["center"]["y"] + box["size"]["y"] / 2.0


#: Air between a placed object's base and the surface, metres.  Not zero: a base
#: measured flush occasionally interpenetrates the tabletop by a hair and the
#: object is rendered clipped.
CLEARANCE = 0.005


def place_on(controller, asset: str, name: str, x: float, top: float, z: float,
             rotation: float = 0.0) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Put `asset` on a surface whose top is at `top`, standing on its base.

    `SpawnAsset` puts the asset's PIVOT at the given point, and the pivot is
    wherever the artist left it -- at the base for some assets, at the centre of
    the bounding box for others.  Spawning everything at `top + 0.02`, as this
    module did at first, therefore sinks every centre-pivoted asset half its own
    height into the table: a box sat with its lid at tabletop level and its body
    inside the wood.

    There is no field that says where the pivot is, so measure instead of
    predicting: spawn, read the resulting axis-aligned box, and lift by whatever
    it takes to bring the base to the surface.  Returns the object's metadata
    entry and the corrected position -- record THAT position in a case, or the
    rebuild sinks the object again.
    """
    spawn(controller, asset, name, x, top, z, rotation)
    event = controller.step(action="Pass")
    entry = next(o for o in event.metadata["objects"] if o["name"] == name)
    box = entry["axisAlignedBoundingBox"]
    base = box["center"]["y"] - box["size"]["y"] / 2.0
    position = {"x": float(x), "y": float(top + (top + CLEARANCE - base)),
                "z": float(z)}
    event = controller.step(action="TeleportObject", objectId=name,
                            position=position,
                            rotation={"x": 0, "y": float(rotation), "z": 0},
                            forceAction=True)
    if not event.metadata["lastActionSuccess"]:
        raise RuntimeError(f"place_on {asset}: "
                           f"{event.metadata.get('errorMessage')}")
    entry = next(o for o in event.metadata["objects"] if o["name"] == name)
    return entry, position


def visible_pixels(event, name: str) -> int:
    """
    How many pixels of `name` the camera sees.  Same contract as
    `gen.occlusion.visible_pixels`, so occlusion and grading carry over.

    Reads the segmentation FRAME rather than `event.instance_masks`, which omits
    spawned objects entirely -- see the module docstring.  Two copies of one
    asset get different colours, so this is per-instance, which is the whole
    reason a procedural scene is worth the trouble.
    """
    colour = next((c for c, oid in event.color_to_object_id.items()
                   if oid == name), None)
    if colour is None:
        return 0
    frame = event.instance_segmentation_frame
    return int((frame == np.array(colour, dtype=frame.dtype)).all(-1).sum())


def visible_box(event, name: str) -> Optional[List[float]]:
    """xyxy pixel box of `name`, or None if it draws nothing."""
    colour = next((c for c, oid in event.color_to_object_id.items()
                   if oid == name), None)
    if colour is None:
        return None
    frame = event.instance_segmentation_frame
    ys, xs = np.where((frame == np.array(colour, dtype=frame.dtype)).all(-1))
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()),
            float(xs.max() + 1), float(ys.max() + 1)]


def rebuild(controller, case: Dict[str, Any]):
    """
    Put a frozen tabletop case back exactly as it was generated.

    The whole scene is three spawns and a table, all at coordinates the case
    records, so a rebuild is EXACT -- there is no physics settle to diverge and
    no placement search to redo.  That is the main practical gain over the iTHOR
    lists, where a case could only pin the occluder and the rest of the room
    drifted between runs.

    Returns the event at the case's own camera pose.
    """
    controller.reset()
    # THE TABLE IS OPTIONAL.  Both tabletop lists put every object on one, and it
    # is recorded apart from `objects` because it is the room rather than a prop.
    # `cases_on` has no table at all -- its landmark is a chair on the floor, and
    # it is in `objects` like everything else -- so a missing key means "this
    # scene is its objects", not a broken case.
    table = case.get("table")
    if table:
        spawn(controller, table["asset"], "table", table["x"], 0.0, table["z"])
    for entry in case["objects"]:
        # `yaw` defaults to 0 so the older case files, written before the
        # occluder was turned, replay exactly as they always did.
        spawn(controller, entry["asset"], entry["name"],
              entry["position"]["x"], entry["position"]["y"],
              entry["position"]["z"], entry.get("yaw", 0.0))
    # NOT forced.  `build_tabletop.stand_back` only accepts a start pose the
    # collision check passes, so a case that cannot be replayed unforced is a
    # case whose geometry is wrong, and failing here says so at once.
    x, z, yaw, horizon = (float(v) for v in case["start"].split(","))
    # `standing` defaults to True so every case written before the stance was
    # recorded replays exactly as it always did.  It is read from its own field
    # and not from `start`, which is a four-field contract several callers parse.
    #
    # WITHOUT THIS A CROUCHED CASE IS A DIFFERENT SCENE, by 0.68 m of camera
    # height and about 27 degrees of pitch.  See `look_from` for what that does
    # to a tabletop occlusion; `build_slot.py` stages crouched for exactly that
    # reason, and replaying it standing would measure occlusions never staged.
    return look_from(controller, x, z, yaw, horizon,
                     standing=case.get("standing", True))


class View:
    """
    Enough of a `RobotController` for the geometry helpers to work off a bare
    procedural event.

    `unproject` and `point_in_box` want `camera_xyz`, `agent_yaw`,
    `camera_horizon` and `event`, and nothing else -- they were written against
    the iTHOR wrapper, but the maths is the camera's, not the scene's.  Wrapping
    the event is cheaper and less error-prone than teaching every caller two
    ways to ask where the camera is.
    """

    def __init__(self, event):
        self.event = event

    @property
    def camera_xyz(self) -> np.ndarray:
        position = self.event.metadata["cameraPosition"]
        return np.array([position["x"], position["y"], position["z"]])

    @property
    def agent_yaw(self) -> float:
        return float(self.event.metadata["agent"]["rotation"]["y"])

    @property
    def camera_horizon(self) -> float:
        return float(self.event.metadata["agent"]["cameraHorizon"])

    @property
    def agent_position(self) -> Dict[str, float]:
        return dict(self.event.metadata["agent"]["position"])


class Robot(View):
    """
    A `RobotController` stand-in for the procedural room, so the motion loop
    runs unchanged.

    `eval_move` drives an iTHOR wrapper: it teleports, asks where the camera is,
    and falls back to the navmesh when a pose is refused.  None of that is about
    iTHOR except the navmesh, which this room does not have -- a procedural house
    returns an empty `GetReachablePositions` -- so `nearest_reachable` has
    nothing to offer and says so.

    MOVES ARE COLLISION-CHECKED AND A BLOCKED ONE IS REFUSED.  The path between
    two poses is not simulated -- `walk` teleports -- but the DESTINATION has to
    be somewhere the robot could stand, or the episode is measuring a camera
    that passes through furniture.  An earlier version forced every teleport and
    merely counted the collisions; on the first twelve cases 100% of the steps
    collided in five of them, because the START pose itself was illegal.  That
    is fixed in the generator (`build_tabletop.stand_back`), and this refuses
    what is left.
    """

    def __init__(self, controller, event, standing: Optional[bool] = None):
        super().__init__(event)
        self.controller = controller
        self.refused = 0
        self.steps = 0
        # THE STANCE HAS TO BE REMEMBERED, not re-defaulted per move.  `rebuild`
        # sets it from the case; every later pose goes through `teleport`, and a
        # hardcoded `standing=True` there would stand the robot up on its FIRST
        # STEP -- silently, with no refused action and no error, so every pose
        # after step 0 would see a scene the case never staged.
        #
        # READ OFF THE POSE, not taken on trust.  Every caller builds this as
        # `Robot(controller, rebuild(controller, case))` and would have to be
        # told the stance separately -- `eval_move`, `probe_viewdist`, and
        # whatever is written next, which is the one that will forget.  The event
        # already knows: THOR puts the camera ~0.68 m above a standing agent and
        # at its own height when crouched, so the gap says which.
        self.standing = self.stance_of(event) if standing is None \
            else bool(standing)

    @staticmethod
    def stance_of(event) -> bool:
        """Is the agent in this event standing?  From the camera-to-body gap."""
        return (event.metadata["cameraPosition"]["y"]
                - event.metadata["agent"]["position"]["y"]) > 0.3

    def teleport(self, position: Optional[Dict[str, float]] = None,
                 yaw: Optional[float] = None,
                 horizon: Optional[float] = None) -> bool:
        """Move and/or aim.  False, and nothing moves, if the pose is blocked."""
        where = position or self.agent_position
        yaw = self.agent_yaw if yaw is None else yaw
        horizon = self.camera_horizon if horizon is None else horizon
        self.steps += 1
        event = self.controller.step(
            action="Teleport", position={"x": float(where["x"]),
                                         "y": float(where["y"]),
                                         "z": float(where["z"])},
            rotation={"x": 0, "y": float(yaw), "z": 0},
            horizon=float(horizon), standing=self.standing, forceAction=False)
        if not event.metadata["lastActionSuccess"]:
            self.refused += 1
            return False
        self.event = event
        return True

    def nearest_reachable(self, xz) -> None:
        """No navmesh in a procedural house; `teleport` never needs the fallback."""
        return None

    def set_height_level(self, level: str) -> None:
        """Ignored: the stance comes from the CASE, not from the control loop.

        `eval_move` calls this on an iTHOR wrapper to crouch for a low shelf.
        Here the stance is a property of the scene the case was staged in, and
        letting the loop change it mid-episode would move the camera by 0.68 m
        between poses -- which is the whole occlusion the case is built on.
        """


def aim_at(controller, x: float, z: float, yaw: float, target_xyz,
           y: float = 0.9, standing: bool = True):
    """
    Stand at (x, z) and pitch so `target_xyz` is centred.  Returns the event.

    The pitch cannot be computed before standing: THOR's camera sits about
    0.68 m ABOVE the agent position it is teleported to, so computing the angle
    from the body height gives roughly zero -- a level camera, a table seen
    edge-on, and most of the frame spent on the far wall.  Teleport first, read
    `cameraPosition`, then pitch.

    `standing` reaches both teleports, and it has to: the first is what
    `cameraPosition` is read from, so passing it to only the second would pitch
    for a camera 0.68 m above the one that ends up taking the picture.
    """
    event = look_from(controller, x, z, yaw, 0.0, y, force=True,
                      standing=standing)
    camera = event.metadata["cameraPosition"]
    flat = math.hypot(target_xyz[0] - camera["x"], target_xyz[2] - camera["z"])
    horizon = math.degrees(math.atan2(camera["y"] - target_xyz[1],
                                      max(flat, 1e-6)))
    return look_from(controller, x, z, yaw, horizon, y, force=True,
                     standing=standing), horizon


def look_from(controller, x: float, z: float, yaw: float, horizon: float,
              y: float = 0.9, force: bool = False, standing: bool = True):
    """
    Stand somewhere and aim.  Returns the event.

    `standing` IS THE ONLY LEVER ON CAMERA HEIGHT, and `y` is not one: an
    unforced Teleport succeeds at y=0.9 and at no other value, because anything
    else puts the capsule into the floor.  What varies is the stance --

        standing   camera 1.575 m, 0.74 m over a 0.834 m tabletop, pitched about
                   27 degrees DOWN to aim at it
        crouched   camera 0.900 m, 0.07 m over it, essentially level

    -- and the difference decides whether a tabletop occluder occludes at all.
    The ray to a target `gap` behind an occluder clears the occluder plane by
    `gap * tan(pitch)`, so standing it is 0.33 m up at `gap` 0.65 and a 0.37 m
    box passes UNDER the sightline: measured, a `Box_21` in front of a
    `Laptop_2` hid 3% of it standing and 50% crouched, same scene.  Occlusion on
    a table is a horizontal fact and a steep camera goes over the top of it.

    `force` skips THOR's capsule-collision check, and the two callers want
    opposite answers.  PLACING A CAMERA wants it forced: standing 0.9 m from a
    tabletop object puts the agent's capsule inside the table even though the
    camera, 0.68 m higher, is well clear of it, and half the attempts at that
    distance were rejected -- silently, since the generator catches the error and
    moves on, so the surviving cases were quietly the ones far enough away to be
    the small targets we were trying to get rid of.  A ROBOT MOVING wants it
    checked: walking through the table is a failure worth seeing, and this room
    has no navmesh -- `GetReachablePositions` is empty -- so the collision test
    is the only thing that will ever say so.  Hence: off by default, on where the
    agent is a tripod.
    """
    event = controller.step(action="Teleport",
                            position={"x": float(x), "y": float(y),
                                      "z": float(z)},
                            rotation={"x": 0, "y": float(yaw), "z": 0},
                            horizon=float(horizon), standing=bool(standing),
                            forceAction=bool(force))
    if not event.metadata["lastActionSuccess"]:
        raise RuntimeError(f"Teleport: {event.metadata.get('errorMessage')}")
    return event
