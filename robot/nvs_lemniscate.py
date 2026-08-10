"""
nvs_lemniscate.py -- the NVS views a robot would ask for, rendered by the simulator.

The end goal is a robot that resolves an occlusion by SYNTHESISING views rather
than driving to them.  Before trusting a generative model with that, this asks
the easier question: if the novel views were perfect, would they recover the
relation at all?  So the lemniscate is rendered by THOR instead of predicted --
an oracle NVS, and an upper bound on what `Stable-Virtual-Camera` can buy here.

This closes a gap an earlier survey left open.  That survey rendered strips of
stock scenes where a few degrees of orbit recovers 2.2-2.9x the target's pixels
(see `find_cases.py`), but never ran EGTR on them -- so it showed that PIXELS
come back, which is not the claim.  Whether the RELATION comes back is what the
sweep here measures.

Trajectory.  Matched to the one the NVS pipeline actually uses --
`sgg_nvs/script/run_nvs_occlusion.sh` runs Stable Virtual Camera with
`--trajectory lemniscate --num_views 20 --max_az 10 --max_el 10`, and
`sgg_nvs/0806_progress.md` records the pose list as `t = 2*pi*(v+1)/20`.  That
submodule is not checked out here, so the figure-eight itself is reconstructed as
the Gerono form:

    azimuth(t)   = max_az * sin(t)
    elevation(t) = max_el * sin(2t)

which is checkable rather than assumed: the same note observes that v=9 (the
centre crossing, t=pi) and v=19 (t=2pi) both reproduce the input pose at pixel
MAE ~5 against 30-61 for the rest.  This script measures that MAE and prints it.
If those two views are NOT the near-duplicates, the reconstruction is wrong and
nothing below should be believed.

The reference view is the ROBOT's view, and that is not a detail.  The sweep
orbits a point on the robot's own optical axis, so az = el = 0 reproduces its
frame -- measured MAE 2.3 against it, the same level as the two duplicate views,
where the rest of the sweep sits at 17-45.  An NVS model is handed the robot's
frame as input; a sweep whose reference is some other viewpoint is not the
question.  The first version orbited the TARGET and aimed every view at it,
which put the "reference" at the robot's position but looking 23 deg off its
heading -- a frame the robot never saw.

One honest gap remains.  Stable Virtual Camera works in normalised scene units
with `--lookat_dist 10.0` and there is no established mapping from that to THOR
metres, so the orbit distance here defaults to the target's own depth along that
axis.  `--radius` overrides it; the family of viewpoints is the same shape, the
scale is a choice.

    python nvs_lemniscate.py --scene FloorPlan203 --start=-4.75,-1.75,20,45 \
        --occlude book:Vase --out nvs/fp203_book
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from robot import drive
import gen.occlusion_pipeline as P
from robot.drive_triplet_scene import measure
from gen.build_occlusion_dataset import bbox_of
from robot.task_find import build_tasks, grade, put_in_front
from vg.vg150 import THOR_TO_VG150


def lemniscate(views: int, max_az: float, max_el: float
               ) -> List[Tuple[float, float]]:
    """
    The (azimuth, elevation) pairs, in the NVS pipeline's own order.

    `t = 2*pi*(v+1)/views` is kept verbatim even though it wastes two frames --
    v = views//2 - 1 lands on the centre crossing and the last wraps to t=0 --
    because the point of this file is to mirror the trajectory the generative
    model is given, not to improve it.  `0806_progress.md` already lists fixing
    the parameterisation as future work.
    """
    poses = []
    for v in range(views):
        t = 2.0 * math.pi * (v + 1) / views
        poses.append((max_az * math.sin(t), max_el * math.sin(2.0 * t)))
    return poses


def look_at_point(camera: np.ndarray, yaw: float, pitch: float,
                  distance: float) -> np.ndarray:
    """The point `distance` along the camera's optical axis (Unity: +pitch is down)."""
    y = math.radians(yaw)
    p = math.radians(pitch)
    forward = np.array([math.sin(y) * math.cos(p), -math.sin(p),
                        math.cos(y) * math.cos(p)])
    return camera + forward * distance


def camera_for(centre: np.ndarray, reference: np.ndarray,
               azimuth: float, elevation: float) -> Dict[str, Any]:
    """
    Where the camera goes for one (azimuth, elevation), orbiting `centre`.

    `centre` is a point on the REFERENCE camera's own optical axis, which is
    what makes az = el = 0 reproduce the reference view exactly -- same
    position, same aim, MAE 0 -- and that identity is the whole point.  An
    earlier version orbited the TARGET and aimed each view at it, so the az=0
    view sat at the robot's position but looked 23 deg to the right of where the
    robot was looking: a "reference" the robot never saw, and a baseline no NVS
    model would ever be handed, since its input IS the robot's frame.

    This is also Stable Virtual Camera's own parameterisation -- an orbit about
    a lookat point at `--lookat_dist` along the input axis -- so the two sweeps
    describe the same family of viewpoints.
    """
    v = reference - centre
    radius = float(np.linalg.norm(v))
    theta = math.atan2(v[0], v[2]) + math.radians(azimuth)
    phi = math.asin(np.clip(v[1] / max(radius, 1e-6), -1.0, 1.0)) \
        + math.radians(elevation)
    position = centre + radius * np.array([
        math.sin(theta) * math.cos(phi), math.sin(phi),
        math.cos(theta) * math.cos(phi)])
    flat = math.hypot(centre[0] - position[0], centre[2] - position[2])
    yaw = math.degrees(math.atan2(centre[0] - position[0],
                                  centre[2] - position[2])) % 360.0
    pitch = math.degrees(math.atan2(position[1] - centre[1], max(flat, 1e-3)))
    return {"position": {"x": float(position[0]), "y": float(position[1]),
                         "z": float(position[2])},
            "yaw": yaw, "pitch": pitch,
            "azimuth": azimuth, "elevation": elevation}


def arc_step(centre: np.ndarray, here: np.ndarray, azimuth: float
             ) -> np.ndarray:
    """
    Swing `here` about `centre` by `azimuth` degrees, staying on the floor.

    The ground-plane half of `camera_for`, for the case where a sweep has
    chosen a DIRECTION and the robot has to execute it.  Elevation is not
    dropped so much as never present: the action space is the floor, so a
    view's elevation can only ever have contributed a vote.  Because the step
    is an arc on the navmesh's own plane it is always a legal shape of move,
    which "go stand where that virtual camera was" never was.

    Both points are XZ, not XYZ.

    NO CALLER YET, and that is the one thing to check before trusting this file
    is tidy: the move-point half of the fusion pointer is not built.  `fuse_live`
    fuses and scores, it does not walk.  If that step is abandoned, delete this.
    """
    offset = here - centre
    radius = float(np.linalg.norm(offset))
    theta = math.atan2(offset[0], offset[1]) + math.radians(azimuth)
    return centre + radius * np.array([math.sin(theta), math.cos(theta)])


def park_once(rc, away_from: Dict[str, float],
              reachable: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """
    Send the robot to the far side of the room, once, for the whole sweep.

    The dataset build re-parks per camera move (`occlusion_pipeline.park_agent`,
    "behind the camera" moves with the camera).  That is wrong for a sweep whose
    frames are compared to each other: the robot CASTS A SHADOW, so the same
    camera pose renders differently depending on where the robot is standing --
    measured MAE 2.38 between a near and a far parking of the same pose, which is
    the same order as the difference between viewpoints one lemniscate step
    apart.  Re-parking per view therefore injects a moving shadow into the very
    signal being measured.  Parking once puts the same shadow in all 21 frames.

    A perfect pixel match against the robot's OWN frame is not achievable at any
    parking: the robot is standing in that shot and its shadow is in it (~2.8
    MAE).  That floor is a property of THOR, not of this sweep.
    """
    spot = max(reachable, key=lambda p: math.dist(
        (p["x"], p["z"]), (away_from["x"], away_from["z"])))
    rc.controller.step(action="Teleport",
                       position={"x": spot["x"], "y": spot["y"], "z": spot["z"]},
                       rotation={"x": 0.0, "y": 0.0, "z": 0.0}, horizon=0.0,
                       standing=True)
    return spot


def aim(rc, pose: Dict[str, Any], fov: float, first: bool,
        reachable: Sequence[Dict[str, float]]) -> Any:
    """Point the third-party camera.  The robot is already parked; see `park_once`."""
    action = "AddThirdPartyCamera" if first else "UpdateThirdPartyCamera"
    extra = {} if first else {"thirdPartyCameraId": 0}
    rc.controller.step(action=action, position=dict(pose["position"]),
                       rotation={"x": pose["pitch"], "y": pose["yaw"], "z": 0.0},
                       fieldOfView=fov, **extra)
    return rc.controller.last_event


def target_mask(event, name: str) -> Tuple[int, Optional[List[int]]]:
    """Visible pixels and box of one object in the third-party frame."""
    names = {o["objectId"]: o["name"] for o in event.metadata["objects"]}
    for object_id, mask in event.third_party_instance_masks[0].items():
        if names.get(object_id) == name:
            return int(mask.sum()), bbox_of(mask)
    return 0, None


def sweep(rc, poses: Sequence[Dict[str, Any]], target: str, fov: float,
          reachable: Sequence[Dict[str, float]], keep_frames: bool
          ) -> List[Dict[str, Any]]:
    out = []
    for pose in poses:
        event = aim(rc, pose, fov, first=_first_flag(rc), reachable=reachable)
        pixels, box = target_mask(event, target)
        row = {"pose": pose, "pixels": pixels, "box": box}
        if keep_frames:
            row["frame"] = np.array(event.third_party_camera_frames[0])
        out.append(row)
    return out


def _first_flag(rc) -> bool:
    """True until a third-party camera exists on this controller."""
    if getattr(rc, "_nvs_camera_added", False):
        return False
    rc._nvs_camera_added = True
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scene", default="FloorPlan203")
    ap.add_argument("--start", metavar="X,Z,YAW,HORIZON",
                    help="reference pose; write it as --start=-4.75,-1.75,20,45")
    ap.add_argument("--occlude", metavar="CLASS:TYPE", default=None,
                    help="stage the occlusion first, e.g. --occlude book:Vase")
    ap.add_argument("--target-class", default=None,
                    help="which task to follow through the sweep; defaults to "
                         "the --occlude class, else the first task")
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--max-az", type=float, default=30.0)
    ap.add_argument("--max-el", type=float, default=10.0)
    ap.add_argument("--radius", type=float, default=None,
                    help="orbit standoff in metres; default is the reference "
                         "camera's own distance to the target")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--out", default="nvs_lemniscate")
    args = ap.parse_args(argv)

    import cv2

    from robot.sgg_live import SGG_ROOT, load_egtr, predict

    rc = drive.open_scene(args.scene, args.width, args.height, args.fov,
                          args.start)
    try:
        print(f"{args.scene}   reference {drive.pose_line(rc)}")
        targets = sorted(o["name"] for o in rc.event.metadata["objects"]
                         if o["objectType"] in THOR_TO_VG150)

        # The instruction is built at the REFERENCE pose, before staging, so the
        # question the sweep answers is the question the robot was given.
        state = measure(rc, targets, args.fov)
        entries = {o["name"]: o for o in rc.event.metadata["objects"]}
        tasks = build_tasks(state["objects"], entries)
        wanted = args.target_class or (args.occlude.split(":")[0]
                                       if args.occlude else None)
        task = next((t for t in tasks
                     if wanted is None or t["subject_class"] == wanted), None)
        if task is None:
            print(f"no task about a {wanted} in this view")
            return 1
        print(f"instruction: {task['instruction']}   "
              f"target {task['target_name']}")

        staged = None
        if args.occlude:
            staged = put_in_front(rc, task["target_name"],
                                  args.occlude.split(":")[1])
            if staged is None:
                return 1

        target_entry = P.by_name(rc.event, task["target_name"])
        target_position = dict(target_entry["position"])
        reference_camera = rc.camera_xyz
        # Default the orbit centre to the target's DEPTH along the robot's own
        # optical axis, not to the target itself: that keeps the parallax about
        # the thing being asked for while leaving the reference view untouched.
        distance = args.radius or float(np.linalg.norm(
            reference_camera - np.array([target_position["x"],
                                         target_position["y"],
                                         target_position["z"]])))
        centre = look_at_point(reference_camera, rc.agent_yaw,
                               rc.camera_horizon, distance)
        print(f"orbit centre {distance:.2f} m along the reference axis, "
              f"lemniscate {args.views} views, "
              f"max_az {args.max_az:g} max_el {args.max_el:g}")

        poses = [camera_for(centre, reference_camera, az, el)
                 for az, el in lemniscate(args.views, args.max_az, args.max_el)]
        # Index 0 is az = el = 0, which IS the reference view: same position,
        # same aim as the robot's own frame, so every MAE below is measured
        # against the image an NVS model would have been given as input.
        poses.insert(0, camera_for(centre, reference_camera, 0.0, 0.0))

        # The MAE baseline is the ROBOT's frame -- the image an NVS model would
        # be handed -- captured before any third-party camera exists.  Comparing
        # the sweep against its own az=0 view instead makes the identity check
        # vacuous: it cannot tell you the reference is the reference.
        agent_frame = np.array(rc.event.frame, dtype=np.int32)
        reachable = rc.controller.step(
            action="GetReachablePositions").metadata["actionReturn"] or []
        parked = park_once(rc, poses[0]["position"], reachable)
        print(f"robot parked at ({parked['x']:+.2f}, {parked['z']:+.2f}) for the "
              f"whole sweep, {math.dist((parked['x'], parked['z']), (poses[0]['position']['x'], poses[0]['position']['z'])):.1f} m away")

        # Pass 1: everything present.  Frames for EGTR, visible pixels for the
        # occlusion.  Pass 2: the target alone, for its amodal reference in each
        # view -- the disabled state survives a camera move, which is why the
        # sweep is once per pass and not once per view.
        present = sweep(rc, poses, task["target_name"], args.fov, reachable,
                        keep_frames=True)

        # Re-read the ids HERE, not from `entries`.  Staging goes through
        # `SetObjectPoses`, which rebuilds the pose list and renumbers every
        # objectId, and `DisableObject` on a stale id fails with an EMPTY error
        # message -- the trap `gen/occlusion_pipeline.py` documents.  With the
        # pre-staging ids nothing was disabled, so the "target alone" pass still
        # had the vase in front of the book and reported its occlusion as 0.29
        # instead of 0.75.  The failures are counted below for the same reason.
        current = {o["name"]: o for o in rc.controller.last_event.metadata["objects"]}
        ids = {n: current[n]["objectId"] for n in targets if n in current}
        rc.controller.step(action="PausePhysicsAutoSim")
        try:
            refused = 0
            for object_id in ids.values():
                event = rc.controller.step(action="DisableObject",
                                           objectId=object_id)
                refused += not event.metadata["lastActionSuccess"]
            if refused:
                print(f"  ! {refused}/{len(ids)} objects refused DisableObject; "
                      f"the amodal reference is not amodal")
            rc.controller.step(action="EnableObject",
                               objectId=ids[task["target_name"]])
            alone = sweep(rc, poses, task["target_name"], args.fov, reachable,
                          keep_frames=False)
            for object_id in ids.values():
                rc.controller.step(action="EnableObject", objectId=object_id)
        finally:
            rc.controller.step(action="UnpausePhysicsAutoSim")

        egtr = load_egtr(SGG_ROOT)
        os.makedirs(args.out, exist_ok=True)

        rows = []
        print(f"\n{'view':>5} {'az':>6} {'el':>6} {'occ':>5} {'px':>7} "
              f"{'MAE':>5}  class  grounded")
        for index, (seen, ref) in enumerate(zip(present, alone)):
            pose = seen["pose"]
            occlusion = (1.0 - seen["pixels"] / ref["pixels"]
                         if ref["pixels"] else 1.0)
            here = dict(task)
            here["target_occlusion"] = round(max(0.0, occlusion), 3)
            here["target_box_amodal"] = ref["box"]
            here["target_box_visible"] = seen["box"]
            triplets = predict(egtr, seen["frame"], args.topk)
            result = grade(here, triplets, [], args.iou, False)
            mae = float(np.abs(seen["frame"].astype(np.int32)
                               - agent_frame).mean())
            label = "REF" if index == 0 else f"{index - 1:02d}"
            print(f"{label:>5} {pose['azimuth']:+6.1f} {pose['elevation']:+6.1f} "
                  f"{here['target_occlusion']:5.2f} {seen['pixels']:7d} "
                  f"{mae:5.1f}  {str(result['class_rank']):>5}  "
                  f"{str(result['grounded_rank']):>8}")
            cv2.imwrite(os.path.join(args.out, f"view_{label}.png"),
                        seen["frame"][:, :, ::-1])
            rows.append({"view": label, "pose": pose,
                         "occlusion": here["target_occlusion"],
                         "visible_px": seen["pixels"],
                         "reference_px": ref["pixels"], "mae_vs_ref": round(mae, 2),
                         "result": result})

        recovered = [r for r in rows[1:] if r["result"]["grounded_rank"]]
        print(f"\nreference: class_rank {rows[0]['result']['class_rank']}, "
              f"grounded {rows[0]['result']['grounded_rank']}, "
              f"occlusion {rows[0]['occlusion']:.2f}")
        print(f"recovered in {len(recovered)}/{len(rows) - 1} novel views"
              + (f"; best rank {min(r['result']['grounded_rank'] for r in recovered)}"
                 if recovered else ""))

        # The reconstruction check: these two views should be near-duplicates of
        # the reference if the parameterisation is right.
        duplicates = [r for r in rows[1:] if r["mae_vs_ref"] < 10]
        print(f"MAE vs the robot's own frame: REF {rows[0]['mae_vs_ref']:.1f}, "
              f"near-duplicates {[r['view'] for r in duplicates] or 'none'} "
              f"(expected {args.views // 2 - 1:02d} and {args.views - 1:02d}), "
              f"others {min((r['mae_vs_ref'] for r in rows[1:] if r not in duplicates), default=0):.0f}"
              f"-{max((r['mae_vs_ref'] for r in rows[1:] if r not in duplicates), default=0):.0f}")

        with open(os.path.join(args.out, "sweep.json"), "w") as fh:
            json.dump({"scene": args.scene, "instruction": task["instruction"],
                       "target": task["target_name"], "staged": staged,
                       "orbit_centre_distance": distance,
                       "orbit_centre": [float(v) for v in centre],
                       "views": args.views,
                       "max_az": args.max_az, "max_el": args.max_el,
                       "rows": rows}, fh, indent=1)
        print(f"\nwrote {os.path.join(args.out, 'sweep.json')} and "
              f"{len(rows)} frames")
        return 0
    finally:
        rc.stop()


if __name__ == "__main__":
    raise SystemExit(main())
