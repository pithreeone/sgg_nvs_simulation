"""
real_robot.py -- frames from a real camera, wearing the simulator's interface.

Everything from `eval_move.perceive` down reads the robot through six things:
an RGB frame, a metric depth frame, and four numbers saying where the camera
is.  None of that is iTHOR's -- `proc_scene.View` already made the same point
for a procedural room -- so a real robot needs no simulator, only these six.

THE POSE IS PINNED AT THE ORIGIN, and that is not a shortcut.  `perceive` builds
the sweep's centre from the camera's OWN optical axis and `step_to` returns
`spot - here`, so every quantity in the chain is relative to the pose the frame
was taken at.  Calling that pose (0, h, 0) with yaw 0 makes the output a
displacement in the robot's own frame, and no global localisation is ever
needed.  The height `h` matters only because the sweep's elevation lifts the
virtual camera, which a body cannot follow anyway; the yaw is arbitrary.

`teleport` MOVES NOTHING -- it records the pose that was asked for.  A real
robot is on the other end of a person reading numbers off a terminal, so a step
is a quantity to be reported, not an action to be taken.  `moved_to` turns the
last recorded pose into the (forward, left, turn) a driver can execute.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


#: Metres per unit in a 16-bit depth PNG.  RealSense writes millimetres.
DEPTH_SCALE = 0.001

#: Vertical FOV of the colour stream, degrees.  THE DEFAULT IS A REALSENSE D435
#: (69.4 x 42.5), NOT the 60 the THOR experiments use, because `unproject` reads
#: this as the vertical field and takes its focal length from the image HEIGHT.
#: Wrong here, every depth-to-3D point is wrong by the ratio of the tangents.
FOV_V = 42.5


def load_depth(path: str, shape: Tuple[int, int],
               scale: float = DEPTH_SCALE) -> np.ndarray:
    """A depth image as metres, resampled to the colour frame's shape.

    NEAREST, never linear: interpolating across a depth discontinuity invents a
    surface halfway between the foreground and the wall behind it, and the
    median inside a detection box is exactly where that would land.

    THE RESAMPLE IS NOT A REGISTRATION.  A stereo camera's depth and colour
    streams have different centres and different fields of view, so matching the
    shapes leaves a parallax error no resize can remove.  Good enough for the
    scalar at the image centre, which is all `--motion pose` reads; wrong for
    `point_in_box`.  Record with `rs.align(rs.stream.color)` and this becomes a
    no-op -- `aligned` below says which case you are in.

    ZERO IS "NO READING", not "at the camera", so it comes back NaN.  Callers
    must use `nanmedian`; a plain median over a box touching the sky returns
    NaN and poisons the whole chain silently.
    """
    import cv2

    if path.endswith(".npy"):
        raw = np.load(path)
        scale = 1.0                     # a float array is already metres
    else:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(path)
    if raw.ndim == 3:
        raw = raw[..., 0]
    metres = raw.astype(np.float32) * float(scale)
    metres[metres <= 0] = np.nan
    height, width = shape[:2]
    if metres.shape[:2] != (height, width):
        metres = cv2.resize(metres, (width, height),
                            interpolation=cv2.INTER_NEAREST)
    return metres


def aligned(depth_path: str, rgb_shape: Tuple[int, int]) -> bool:
    """Did the two streams arrive at the same resolution?  See `load_depth`."""
    import cv2

    if depth_path.endswith(".npy"):
        shape = np.load(depth_path, mmap_mode="r").shape[:2]
    else:
        shape = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).shape[:2]
    return tuple(shape) == tuple(rgb_shape[:2])


def newest_pair(folder: str) -> Tuple[str, str]:
    """The most recent (rgb, depth) the robot dropped in `folder`.

    Matched on the TIMESTAMP PREFIX rather than on mtime: the two files are
    written moments apart, and pairing them by time alone crosses the streams
    the one time a write is slow.
    """
    def image(name: str) -> bool:
        # `*_view.png` IS NOT DEPTH.  A depth camera's viewer writes a colourised
        # copy beside the real thing; it is 8-bit RGB, so reading it as
        # millimetres yields a scene 0-0.255 m deep and every step comes out
        # tiny and wrong rather than failing.  Excluded by name, and the
        # SHORTEST candidate wins, so `depth_image.png` beats any decoration.
        low = name.lower()
        return (low.endswith((".png", ".jpg", ".jpeg", ".npy"))
                and "view" not in low and "colour" not in low
                and "color" not in low)

    names = [n for n in os.listdir(folder) if image(n)]
    rgbs = sorted(n for n in names if "rgb" in n.lower())
    if not rgbs:
        raise FileNotFoundError(f"no *rgb* image in {folder}")
    rgb = rgbs[-1]
    stem = rgb.lower().split("rgb")[0]
    depths = sorted((n for n in names if n.lower().startswith(stem)
                     and "depth" in n.lower()), key=len)
    if not depths:
        raise FileNotFoundError(f"no depth image matching {rgb} in {folder}")
    return os.path.join(folder, rgb), os.path.join(folder, depths[0])


class Event:
    """The two arrays `perceive`, `look` and `point_in_box` read."""

    def __init__(self, frame: np.ndarray, depth: np.ndarray) -> None:
        self.frame = frame
        self.depth_frame = depth
        # Ground truth the simulator would carry and a real room does not.
        # Empty rather than absent, so anything asking gets an honest nothing.
        self.metadata: Dict[str, Any] = {"objects": []}


class RealRobot:
    """A `RobotController` stand-in for frames a real robot sent.

    Same six-property surface as `proc_scene.View`, plus a `teleport` that
    records instead of moving.  `controller` is None: the only caller that wants
    one is `perceive`'s `GetReachablePositions`, which exists to park the
    simulated robot out of its own shot, and a real robot is never in its own
    shot.
    """

    controller = None

    def __init__(self, frame: np.ndarray, depth: np.ndarray,
                 camera_height: float = 0.0, pitch: float = 0.0) -> None:
        self.event = Event(frame, depth)
        self.camera_height = float(camera_height)
        self.pitch = float(pitch)
        self.requested: List[Dict[str, Any]] = []
        self.refused = 0
        self.steps = 0

    # -- where the camera is.  The origin, by construction; see the module note.
    @property
    def camera_xyz(self) -> np.ndarray:
        return np.array([0.0, self.camera_height, 0.0])

    @property
    def agent_yaw(self) -> float:
        return 0.0

    @property
    def camera_horizon(self) -> float:
        return self.pitch

    @property
    def agent_position(self) -> Dict[str, float]:
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    def teleport(self, position: Optional[Dict[str, float]] = None,
                 yaw: Optional[float] = None,
                 horizon: Optional[float] = None) -> bool:
        """Record the pose that was asked for.  Always accepted.

        ALWAYS TRUE, deliberately.  The `BACKOFF` and `RADIUS_LADDER` fallbacks
        in `eval_move` escalate until the SIMULATOR accepts a pose, and there is
        nothing here that can refuse one -- a real robot's navigation stack is
        the thing that knows, and it is downstream of this file.  So the first
        rung is always taken and the ladders are inert, which is the honest
        behaviour: this module does not know what is reachable and must not
        pretend to.
        """
        self.steps += 1
        self.requested.append(
            {"position": dict(position) if position else self.agent_position,
             "yaw": None if yaw is None else float(yaw),
             "horizon": None if horizon is None else float(horizon)})
        return True

    def nearest_reachable(self, xz) -> None:
        """No navmesh here; see `teleport`."""
        return None

    def stop(self) -> None:
        return None

    def moved_to(self) -> Optional[Dict[str, float]]:
        """The recorded step as something a driver can execute.

        THE POSITION AND THE YAW COME FROM DIFFERENT CALLS.  `step_to` teleports
        to the spot, then teleports again to set yaw and pitch from where it
        landed -- so the last recorded pose carries the aim and the last one
        naming a position carries the place.  Reading only the final entry gives
        a turn with no translation.

        TWO FRAMES, AND THEY DISAGREE ABOUT WHICH WAY IS LEFT.

        `forward`/`right`/`turn` are the SIMULATOR's, read off its own basis
        vectors rather than guessed, because a wrong guess sends the robot the
        wrong way and the geometry still looks self-consistent.
        `robot_controller.unproject` builds `right = (cos yaw, 0, -sin yaw)`,
        which at yaw 0 is +x, so +x is the robot's RIGHT; and
        `forward = (sin yaw, 0, cos yaw)` reaches +x at yaw +90, so a POSITIVE
        yaw turns RIGHT.  Both were labelled the opposite way here once.

        `ros` is the same step under REP-103 -- x forward, y LEFT, theta
        counter-clockwise, in radians -- which is a HANDEDNESS CHANGE and not a
        renaming: y and theta both flip sign.  It is the pair a ROS2 goal takes,
        and the reason both are returned is that either one alone is a trap for
        whoever reads it next.

        `turn` re-aims at the sweep's hub AFTER the translation, so a sidestep
        right normally comes back with a left turn -- the two having opposite
        signs is the expected shape, not a contradiction.
        """
        if not self.requested:
            return None
        placed = next((r for r in reversed(self.requested)
                       if r["position"] != self.agent_position), None)
        aimed = next((r for r in reversed(self.requested)
                      if r["yaw"] is not None), None)
        if placed is None:
            return None
        x, z = placed["position"]["x"], placed["position"]["z"]
        turn = float(_wrap(aimed["yaw"])) if aimed else 0.0
        return {"forward": round(float(z), 3), "right": round(float(x), 3),
                "turn": round(turn, 1),
                # THE SAME STEP IN REP-103, which is what a ROS2 goal wants and
                # is NOT a relabelling: the two frames have opposite handedness,
                # so `y` and `theta` both flip sign against the fields above.
                # Feeding the Unity numbers to a ROS stack mirrors every step.
                "ros": {"x": round(float(z), 3), "y": round(-float(x), 3),
                        "theta": round(math.radians(-turn), 4)},
                "distance": round(float(math.hypot(x, z)), 3),
                "pitch": round(float(aimed["horizon"]), 1)
                         if aimed and aimed["horizon"] is not None else 0.0}


def _wrap(degrees: float) -> float:
    """To (-180, 180], so a small right turn reads -8 rather than 352."""
    return (float(degrees) + 180.0) % 360.0 - 180.0
