"""
drive.py -- a plain THOR scene you can drive around.  Nothing else.

No clutter generation, no occlusion measurement, no ground truth: the stock
scene, the robot standing in it, and its camera.  `drive_triplet_scene.py` is
the one that does all of that; this is the one to open when you only want frames
to feed a model.

    python drive.py                             # FloorPlan203
    python drive.py --scene FloorPlan1
    python drive.py --start=-4.75,-1.75,20,45   # stand somewhere specific

Keys
    w / s   forward / back        a / d   turn      q / e   strafe
    r / f   look up / down        c       crouch/stand
    p       save frame.png        ESC/x   quit

From your own code:

    from robot.drive import open_scene
    rc = open_scene("FloorPlan203")             # robot is standing in it
    frame = rc.event.frame                       # HxWx3 uint8 RGB
    rc.move_ahead(); rc.rotate(30)
    rc.stop()
"""

from __future__ import annotations

# `python robot/<script>.py` puts robot/ on sys.path, not the repo root, so the
# top-level packages would not import.  Same bootstrap as gen/.
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)


import argparse
from typing import Optional, Sequence

import numpy as np

from robot.robot_controller import RobotController


#: Default step sizes, deliberately smaller than THOR's usual 0.25 m / 30 deg.
#: `snapToGrid` is off, so the base takes any magnitude -- the 0.25 m navmesh
#: grid only constrains where `nearest_reachable` may put you, not how far a
#: MoveAhead goes.  Every keypress is one THOR action either way, so a smaller
#: step costs proportionally more keypresses to cross a room.
STEP_M = 0.10
TURN_DEG = 10.0
LOOK_DEG = 5.0


def open_scene(scene: str = "FloorPlan203", width: int = 800, height: int = 600,
               fov: float = 60.0, start: Optional[str] = None,
               headless: bool = False, step: float = STEP_M,
               turn: float = TURN_DEG) -> RobotController:
    """
    Load `scene` and leave the robot standing in it.

    `start` is "X,Z,YAW,HORIZON" and is snapped to the navmesh -- THOR will
    teleport the agent into a wall if asked, and every later MoveAhead then
    fails.  Without it the robot keeps the scene's default spawn.
    """
    rc = RobotController(scene=scene, width=width, height=height,
                         field_of_view=fov, visibility_distance=15.0,
                         move_magnitude=step, rotate_step=turn,
                         headless=headless, verbose=False)
    if start:
        x, z, yaw, horizon = (float(v) for v in start.split(",")[:4])
        spot = rc.nearest_reachable(np.array([x, z]))
        if spot is not None:
            rc.teleport(position={"x": float(spot[0]),
                                  "y": float(rc.agent_position["y"]),
                                  "z": float(spot[1])},
                        yaw=yaw, horizon=horizon)
    return rc


def pose_line(rc: RobotController) -> str:
    position = rc.agent_position
    return (f"x {position['x']:+.2f}  z {position['z']:+.2f}  "
            f"yaw {rc.agent_yaw:6.1f}  horizon {rc.camera_horizon:+5.1f}  "
            f"{rc.height_level}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scene", default="FloorPlan203")
    ap.add_argument("--start", metavar="X,Z,YAW,HORIZON",
                    help="write it as --start=-4.75,-1.75,20,45")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0, help="VERTICAL fov")
    ap.add_argument("--out", default="frame.png", help="where p writes")
    ap.add_argument("--step", type=float, default=STEP_M, metavar="M",
                    help=f"metres per drive/strafe keypress (default {STEP_M})")
    ap.add_argument("--turn", type=float, default=TURN_DEG, metavar="DEG",
                    help=f"degrees per turn keypress (default {TURN_DEG})")
    ap.add_argument("--look", type=float, default=LOOK_DEG, metavar="DEG",
                    help=f"degrees per look keypress (default {LOOK_DEG})")
    args = ap.parse_args(argv)

    import cv2

    rc = open_scene(args.scene, args.width, args.height, args.fov, args.start,
                    step=args.step, turn=args.turn)
    print(f"{args.scene}   {pose_line(rc)}")
    print(f"step {args.step} m   turn {args.turn} deg   look {args.look} deg")
    print("w/s drive  a/d turn  q/e strafe  r/f look  c crouch  p save  ESC quit")
    try:
        saves = 0
        while True:
            cv2.imshow("drive", rc.event.frame[:, :, ::-1])   # RGB -> BGR
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("x")):
                break
            elif key == ord("w"):
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
                rc.look(-args.look)
            elif key == ord("f"):
                rc.look(args.look)
            elif key == ord("c"):
                rc.set_height_level(
                    "crouch" if rc.height_level == "stand" else "stand")
            elif key == ord("p"):
                path = args.out if saves == 0 else args.out.replace(
                    ".png", f"_{saves:02d}.png")
                cv2.imwrite(path, rc.event.frame[:, :, ::-1])
                print(f"wrote {path}   {pose_line(rc)}")
                saves += 1
                continue
            else:
                continue
            print(pose_line(rc))
        cv2.destroyAllWindows()
        return 0
    finally:
        rc.stop()


if __name__ == "__main__":
    raise SystemExit(main())
