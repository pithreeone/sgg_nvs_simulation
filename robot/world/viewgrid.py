"""
viewgrid.py -- the poses a robot could stand at and still see both objects.

THE CONTROL ARM'S ACTION SPACE, and only that.  `nvs_lemniscate.lemniscate`
answers "where would I synthesise a view", which is a question about the method;
this answers "where could the robot go at all", which is a question about the
room.  The two are deliberately different sets -- see `eval_move`'s `--control`.

A SQUARE LATTICE laid out along the pair's own axis, after
`../ros2/tools/dataset/viewpoints.py`, whose parameterisation this borrows.  The
lattice rather than the (r, theta) rings that tool also offers, for the reason
its own docstring gives: rings crowd points together near the pair and thin them
out far away, so an arm drawing UNIFORMLY from a ring list is really drawing
mostly from the near ring, and "random" would then mean "random, mostly close
up".  A lattice samples the floor evenly.  Every point still carries its own r
and azimuth, so reading accuracy against distance or against angle is a matter
of binning afterwards -- which is not true the other way round.

The lattice is laid out along the line joining the two objects, not along the
room's axes: azimuth 0 stands on the NEAR object's side looking along the line,
180 is the same line from the far side.  Angles from the room would mean a
different thing in every scene; angles from the pair mean the same thing in all
of them.

NEAR AND FAR, not subject and object.  `viewpoints.py` calls the first argument
the subject and means "the one that should be nearer the camera", which for
`Find the {A} behind the {B}` is B -- the landmark, the sentence's OBJECT.
Naming the arguments by the grammar and passing them in that order puts the
whole sector behind the pair, which is where the instruction stops being true;
naming them by geometry cannot make that mistake.

THE AXIS IS BUILT FROM SPAWN POSITIONS, which are PIVOTS and not geometric
centres -- `SpawnAsset` places the pivot, as `proc_scene`'s own docstring says.
So the lattice's azimuth 0 is a few degrees off the line the silhouettes
actually occlude along, and the sector is therefore not exactly symmetric about
the robot's start pose: one side has more room to walk than the other.  Measured
on `slot|16`, whose three pivots are collinear to 3 decimal places, the
silhouettes still only overlap 43%.

Left uncorrected on purpose.  The 2D visible box is not the fix -- it is the
silhouette AFTER occlusion, so defining the axis with it would define the input
from the result.  The honest fix is the 3D bounding box in THOR's metadata,
which is independent of the view; until that is worth the work, the asymmetry is
recorded here rather than mistaken for a bug later.

`--control-span` narrows the lattice to a sector about azimuth 0.  A full circle
is the honest action space for "where could the robot go", but for a `behind`
task the far half is where the sentence stops being TRUE -- from there the
landmark is behind the target -- and an arm that lands there scores zero for a
reason that is not about choosing viewpoints.

Where the robot stands and where it looks are separate variables, so each
position is crossed with a few yaw offsets.

WHAT IS NOT FILTERED: occlusion.  The real tool drops a viewpoint whose line to
the target is blocked, which is right for a robot navigating a room and wrong
here -- "find the laptop behind the box" would have its own premise filtered
out.  A pose qualifies if the pair is IN FRAME and big enough to detect; whether
one object hides the other is the task, not a constraint on the task.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Lattice spacing in metres, and the annulus of distances from the pair's
#: midpoint it is cut to.  The cases stand the robot at 1.5-2.4 m; `R_MIN` keeps
#: it off the table and `R_MAX` inside a 4 m room.
SPACING = 0.3
R_MIN = 0.8
R_MAX = 2.4

#: Degrees of azimuth the lattice is cut to, about the pair's own axis; 360
#: keeps the whole floor.  150 is `viewpoints.py`'s own default, arrived at on
#: the real robot: it keeps the sector where the subject is genuinely the nearer
#: object and drops the far side, where `behind` reads as `in front of` and the
#: instruction is no longer true of what the camera sees.
SPAN = 150.0

#: Where it looks, relative to aiming at the midpoint.  `viewpoints.py`'s own
#: default, from the real robot: aiming dead centre is one choice among several
#: and a pose that only frames the pair when aimed perfectly is not a pose the
#: robot can actually use.
YAW_OFFSETS = (-20.0, 0.0, 20.0)

#: Nominal object width, metres, for the "is it big enough to detect" test, and
#: the fraction of the frame's width it must span.  A laptop at 3 m spans about
#: 6% of an 800x600 60-degree frame and EGTR does not find it.
OBJECT_SIZE = 0.30
MIN_FRAME_FRACTION = 0.08

#: Keep the body off the walls of the `ROOM`-metre square.
WALL_MARGIN = 0.4


def _hfov(fov_vertical: float, width: int, height: int) -> float:
    """THOR's `fieldOfView` is VERTICAL; the framing test is horizontal."""
    return math.degrees(2.0 * math.atan(
        math.tan(math.radians(fov_vertical) / 2.0) * width / height))


def _bearing_error(camera: np.ndarray, yaw_deg: float,
                   point: np.ndarray) -> float:
    """Degrees between the camera's axis and `point`, in the ground plane."""
    axis = np.array([math.sin(math.radians(yaw_deg)),
                     math.cos(math.radians(yaw_deg))])
    to = point - camera
    norm = float(np.linalg.norm(to))
    if norm < 1e-6:
        return 180.0
    cos = float(np.clip(axis @ (to / norm), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def lattice(centre: np.ndarray, axis_deg: float, spacing: float,
            r_min: float, r_max: float, span: float
            ) -> List[Tuple[np.ndarray, float, float]]:
    """-> [(xz, r, azimuth)] on a square lattice in the pair's own frame.

    `range(-n, n)`, not `(-n, n + 1)`: with the half-cell offset below, `n + 1`
    puts an extra column and row on the positive side and leaves the lattice off
    centre.  Half a cell off centre in both directions, because a point exactly
    on the midpoint would be a viewpoint at zero range.
    """
    half = span / 2.0 if span < 360.0 else 180.0
    theta = math.radians(axis_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    out = []
    n = int(math.ceil(r_max / spacing))
    for i in range(-n, n):
        for j in range(-n, n):
            u, v = (i + 0.5) * spacing, (j + 0.5) * spacing
            r = math.hypot(u, v)
            if not r_min <= r <= r_max:
                continue
            azimuth = math.degrees(math.atan2(v, u))
            if abs(math.degrees(math.atan2(math.sin(math.radians(azimuth)),
                                           math.cos(math.radians(azimuth))))) \
                    > half:
                continue
            # Rotate the lattice into the room's frame.
            out.append((centre + np.array([u * sin + v * cos,
                                           u * cos - v * sin]),
                        r, azimuth))
    return out


def feasible(rc, near_xz: Sequence[float], far_xz: Sequence[float],
             fov: float, width: int, height: int, room: float,
             spacing: float = SPACING, r_min: float = R_MIN,
             r_max: float = R_MAX, span: float = SPAN,
             yaw_offsets: Sequence[float] = YAW_OFFSETS,
             object_size: float = OBJECT_SIZE,
             min_fraction: float = MIN_FRAME_FRACTION
             ) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, int]]:
    """-> (poses, the midpoint they aim at, why each rejection happened).

    The counts come back because a candidate set that shrinks silently is how an
    experiment ends up measuring something other than what it says.

    Poses carry the same keys `nvs_lemniscate.camera_for` returns, so `step_to`
    executes one without knowing which family it came from, plus the `r` and
    `azimuth` each was drawn at so a result can be binned by either.
    """
    near = np.asarray(near_xz, float)
    far = np.asarray(far_xz, float)
    centre = (near + far) / 2.0
    span_vec = near - far
    if float(np.linalg.norm(span_vec)) < 1e-6:
        span_vec = np.array([0.0, 1.0])
    # Azimuth 0 lies on the NEAR object's side of the midpoint.
    axis_deg = math.degrees(math.atan2(span_vec[0], span_vec[1]))

    half_h = _hfov(fov, width, height) / 2.0
    y = float(rc.agent_position["y"])
    start = {"position": dict(rc.agent_position), "yaw": rc.agent_yaw,
             "horizon": rc.camera_horizon}

    poses: List[Dict[str, Any]] = []
    why = {"wall": 0, "frame": 0, "size": 0, "stand": 0}
    try:
        for spot, radius, azimuth in lattice(centre, axis_deg, spacing,
                                             r_min, r_max, span):
            if not (WALL_MARGIN <= spot[0] <= room - WALL_MARGIN
                    and WALL_MARGIN <= spot[1] <= room - WALL_MARGIN):
                why["wall"] += 1
                continue
            # THE FAR OBJECT IS THE ONE THAT FAILS: it spans least.
            reach = max(float(np.linalg.norm(spot - near)),
                        float(np.linalg.norm(spot - far)))
            fraction = math.degrees(
                2.0 * math.atan(object_size / 2.0 / max(reach, 1e-3))) \
                / (2.0 * half_h)
            if fraction < min_fraction:
                why["size"] += 1
                continue

            aimed = math.degrees(math.atan2(centre[0] - spot[0],
                                            centre[1] - spot[1])) % 360.0
            framed = False
            for offset in yaw_offsets:
                yaw = (aimed + offset) % 360.0
                if max(_bearing_error(spot, yaw, near),
                       _bearing_error(spot, yaw, far)) > half_h:
                    continue
                framed = True
                # THE ROOM HAS NO NAVMESH, so "can the robot stand here" is
                # answered by asking THOR to put it there.  Cheap, and it is the
                # same test `step_to` already relies on.
                if not rc.teleport(position={"x": float(spot[0]), "y": y,
                                             "z": float(spot[1])},
                                   yaw=yaw, horizon=0.0):
                    why["stand"] += 1
                    continue
                poses.append({
                    "position": {"x": float(spot[0]), "y": y,
                                 "z": float(spot[1])},
                    "yaw": yaw, "pitch": 0.0,
                    "azimuth": float(azimuth), "elevation": 0.0,
                    "radius": float(radius), "yaw_offset": float(offset)})
            if not framed:
                why["frame"] += 1
    finally:
        rc.teleport(position=start["position"], yaw=start["yaw"],
                    horizon=start["horizon"])
    return poses, centre, why
