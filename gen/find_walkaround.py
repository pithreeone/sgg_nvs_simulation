"""
find_walkaround.py -- FIND poses where stock furniture already half-hides a target.

Nothing is moved.  Two earlier generators DID move things, and both were deleted;
their failures are the reason this one only searches.

`build_hidden_target.py` slid whatever object happened to be nearby along the
camera-to-target ray until the target's pixels went away.  It worked as a pixel
exercise and failed as a scene: a Television ended up hovering beside a side
table.  A render nobody believes cannot support a claim about what a detector
"failed" to see.  It also hid things like a desk lamp, where the fix is a small
sidestep rather than a route.

`build_walkaround.py` then CONSTRUCTED the arrangement -- occluder broadside to
the sightline, target directly behind -- and was wrong for a subtler reason: its
only placement guard was that the agent could stand near the chosen point, which
says nothing about furniture geometry.  A 2.71 m Sofa passed that test while
intersecting a dining table.  `GetReachablePositions` is a navigation mesh, not a
collision test, and no amount of clearance radius turns one into the other.

(`task_find.put_in_front` stages an occluder too, but only among objects already
resting on the target's own surface, and it rejects any pose whose box leaves
that surface or intersects anything on it.  That is the collision test these two
lacked.)

The stock scenes are already physically valid, so the search here can only
produce valid scenes.  What it gives up is control: whatever occlusion the room
offers is what you get, and a scene with no half-occluded view yields nothing.

For a (target, occluder) pair the method is:

  1. sweep reachable poses aimed at the target, recording its visible pixels
  2. the pose where it is MOST visible defines the target's unoccluded size
  3. a start pose is one where the target sits in the occlusion band -- partly
     visible, not gone.  Driving occlusion to 100% makes a scenario nothing can
     solve: with no pixels there is no evidence, so no per-frame model can
     recover the object and a better scene graph cannot help -- that regime
     needs memory or a prior, not a detector.  Half-hidden is "detectable but
     not usable", which is the regime a better scene graph can actually fix.
  4. the occluder is confirmed CAUSALLY, by disabling it and re-rendering from
     the same pose.  Geometric overlap alone would credit the wrong object when
     several stand in the way.
  5. the reference must be RECOVERABLE BY A SMALL VIEW CHANGE.  The camera is
     orbited a few degrees about the target and the target's pixels re-measured;
     the scenario is kept only if some view within the sweep gains at least
     `--min-gain`.  This replaced an earlier criterion -- "a pose exists 60+
     degrees around the occluder" -- which selected for walking somewhere else
     entirely.  A small-baseline NVS model synthesises about 10 degrees, so a
     scenario only tests it if 10 degrees is enough.

     Measuring the sweep rather than proxying it also catches the case nothing
     else does: on FloorPlan230 a laptop sat 55% occluded by the coffee table it
     rested on, so occluder and target were at the SAME depth, parallax was zero
     and the sweep was flat at 0.93-1.00x.  No filter on occlusion, size or
     geometry rejects that; only measuring the gain does.

`DisableObject` is used only as a measuring instrument and the object is always
re-enabled; the scene is never left modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os as _os
import sys as _sys
if __package__ in (None, ""):
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from gen.occlusion import _size, moveable, visible_pixels
from vg.vg150 import THOR_TO_VG150

#: Occluder must be this wide -- furniture scale, something you route around.
MIN_OCCLUDER_M = 1.0

#: Target band: large enough to detect unoccluded, small enough to be hidden.
TARGET_RANGE_M = (0.30, 1.60)

#: Unoccluded pixels a target needs before its occlusion means anything.  Below
#: this the object is unresolvable and "50% hidden" is measuring mask aliasing.
MIN_CLEAR_PX = 1500

#: Camera offsets, in degrees about the target, standing in for what a
#: small-baseline NVS model synthesises.
#:
#: 0 MUST be in this list.  It was not, and the gain was then computed against
#: the pose's agent-camera pixel count from `sweep_all` -- which uses a fixed
#: 60-degree yaw grid and does not aim at anything.  So the ratio divided an
#: AIMED third-party view by an UNAIMED agent view and reported 3-7x, almost all
#: of which was the effect of pointing the camera at the target rather than of
#: moving it ten degrees.  Measuring 0 with the same camera, aimed the same way,
#: is what makes the ratio mean parallax.
SWEEP_DEG = (-10, -5, 0, 5, 10)

#: Required ratio of best-in-sweep pixels to reference pixels.  1.25 is a
#: quarter more of the target visible from ten degrees away.
MIN_GAIN = 1.25

#: Standoff band for a usable view, in metres.
VIEW_RANGE_M = (1.2, 3.5)


def arc_deg(pivot, a, b) -> float:
    v1 = (a[0] - pivot[0], a[1] - pivot[1])
    v2 = (b[0] - pivot[0], b[1] - pivot[1])
    n1 = math.hypot(*v1) or 1e-6
    n2 = math.hypot(*v2) or 1e-6
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def look_at(controller, spot, point, horizon: float = 10.0):
    yaw = math.degrees(math.atan2(point["x"] - spot["x"],
                                  point["z"] - spot["z"])) % 360.0
    event = controller.step(action="Teleport", position=spot,
                            rotation={"x": 0.0, "y": yaw, "z": 0.0},
                            horizon=horizon)
    return event, yaw


def nameable(event) -> List[Dict[str, Any]]:
    out = []
    for entry in event.metadata["objects"]:
        vg = THOR_TO_VG150.get(entry["objectType"])
        if not vg:
            continue
        out.append({"entry": entry, "vg": vg,
                    "width": max(_size(entry)[0], _size(entry)[2])})
    return out


def sweep_all(controller, spots, yaws, horizon: float = 10.0
              ) -> List[Dict[str, Any]]:
    """
    Every object's visible pixels from every pose, in one pass.

    This used to aim the camera at one nominated target and sweep the poses, so
    the cost multiplied by the number of targets and a living room's 15-20
    nameable objects made a single scene take thousands of renders.  Capping the
    target list fixed the cost and broke the search: sorting by size and keeping
    the largest six discarded the 0.51 m chair that was the best example in
    FloorPlan227.

    One render already contains every object's mask, so recording all of them
    costs nothing extra and decouples the sweep from the target count entirely.
    Fixed yaws replace aiming; a target merely has to be IN the frame, not
    centred, for its pixel count to be meaningful.
    """
    names_by_id = {o["objectId"]: o["name"]
                   for o in controller.last_event.metadata["objects"]}
    out = []
    for spot in spots:
        for yaw in yaws:
            event = controller.step(action="Teleport", position=spot,
                                    rotation={"x": 0.0, "y": float(yaw), "z": 0.0},
                                    horizon=horizon)
            if not event.metadata["lastActionSuccess"]:
                continue
            pixels = {}
            for object_id, mask in event.instance_masks.items():
                name = names_by_id.get(object_id)
                if name is not None:
                    pixels[name] = int(mask.sum())
            out.append({"spot": spot, "yaw": float(yaw), "px": pixels})
    return out


def parallax(controller, spot, target_xz, target_name: str, reference_px: int,
             angles=SWEEP_DEG, horizon: float = 10.0) -> Dict[str, Any]:
    """
    Re-measure the target from a few degrees either side, orbiting the target.

    A third-party camera is used rather than teleporting the agent: an NVS view
    is synthesised, so it need not lie on the navigation mesh, and requiring it
    to would throw away valid scenarios for an irrelevant reason.
    """
    radius = math.dist((spot["x"], spot["z"]), (target_xz["x"], target_xz["z"]))
    base = math.atan2(spot["x"] - target_xz["x"], spot["z"] - target_xz["z"])
    out = {}
    for index, degrees in enumerate(angles):
        angle = base + math.radians(degrees)
        cx = target_xz["x"] + math.sin(angle) * radius
        cz = target_xz["z"] + math.cos(angle) * radius
        yaw = math.degrees(math.atan2(target_xz["x"] - cx,
                                      target_xz["z"] - cz)) % 360.0
        # Add ONCE per controller, then only update.  `AddThirdPartyCamera`
        # costs 260 ms against 77 ms for an update AND appends a new camera
        # every call -- after eleven candidate evaluations the scene carried
        # eleven cameras, every one of them rendered on every subsequent step.
        # That is what made the search appear to hang: the cost grew with the
        # number of candidates already examined, not with the work remaining.
        if not controller.last_event.third_party_camera_frames:
            action, extra = "AddThirdPartyCamera", {}
        else:
            action, extra = "UpdateThirdPartyCamera", {"thirdPartyCameraId": 0}
        controller.step(action=action, position={"x": cx, "y": spot["y"], "z": cz},
                        rotation={"x": horizon, "y": yaw, "z": 0.0},
                        fieldOfView=60.0, **extra)
        event = controller.last_event
        names = {o["objectId"]: o["name"] for o in event.metadata["objects"]}
        masks = event.third_party_instance_masks[0]
        pixels = 0
        for object_id, mask in masks.items():
            if names.get(object_id) == target_name:
                pixels = int(mask.sum())
                break
        out[degrees] = pixels
    # Reference is this sweep's own 0-degree frame, never the caller's count.
    reference = out.get(0) or reference_px or 1
    best_deg = max(out, key=out.get)
    return {"by_angle": out, "best_deg": best_deg,
            "best_px": out[best_deg], "reference_px": reference,
            "gain": round(out[best_deg] / max(reference, 1), 3)}


def blocker_at(controller, spot, target_xz, target_name: str,
               candidates, blocked_px: int) -> Optional[Dict[str, Any]]:
    """
    Which object is hiding the target from here -- established by removing it.

    Overlap in the image is not enough: with a sofa, a table and a wall all in
    the way, several objects overlap the target and only one is responsible.
    """
    best = None
    for candidate in candidates:
        object_id = candidate["entry"]["objectId"]
        controller.step(action="DisableObject", objectId=object_id)
        event, _ = look_at(controller, spot, target_xz)
        freed = visible_pixels(event, target_name) if \
            event.metadata["lastActionSuccess"] else blocked_px
        controller.step(action="EnableObject", objectId=object_id)
        gain = freed - blocked_px
        if gain > 0 and (best is None or gain > best["gain"]):
            best = {"candidate": candidate, "gain": gain, "clear_px": freed}
    return best


def search(controller, scene: str, reachable, band, min_gain, rescues,
           allow_same_class: bool, max_poses: int, seed: int,
           max_targets: int = 6) -> Dict[str, Any]:
    import random
    rng = random.Random(seed)

    event = controller.last_event
    objects = nameable(event)
    occluders = sorted([o for o in objects if o["width"] >= MIN_OCCLUDER_M],
                       key=lambda o: -o["width"])
    targets = [o for o in objects
               if TARGET_RANGE_M[0] <= o["width"] <= TARGET_RANGE_M[1]
               and (o["entry"].get("pickupable") or o["entry"].get("moveable"))]
    # No size cap: `sweep_all` costs the same whatever the target count, so
    # every candidate is considered.  Capping by size previously discarded the
    # best example in FloorPlan227, a 0.51 m chair.
    targets = sorted(targets, key=lambda o: -o["width"])[:max_targets]
    if not occluders or not targets:
        return {"error": f"{scene}: no furniture-scale occluder or target"}

    spots = [reachable[i] for i in rng.sample(range(len(reachable)),
                                              min(len(reachable), max_poses))]
    candidates_out: List[Dict[str, Any]] = []
    yaws = [y for y in range(0, 360, 60)]
    frames = sweep_all(controller, spots, yaws)
    if not frames:
        return {"error": f"{scene}: no reachable pose rendered"}

    for target in targets:
        name = target["entry"]["name"]
        target_xz = {"x": target["entry"]["position"]["x"],
                     "z": target["entry"]["position"]["z"]}
        seen = {i: {"px": f["px"].get(name, 0), "yaw": f["yaw"], "spot": f["spot"]}
                for i, f in enumerate(frames)}
        best_px = max(v["px"] for v in seen.values())
        if best_px < MIN_CLEAR_PX:
            continue
        partial = [(i, v) for i, v in seen.items()
                   if band[0] <= 1.0 - v["px"] / best_px <= band[1]
                   and v["px"] >= 200]
        partial.sort(key=lambda kv: -kv[1]["px"])

        for index, view in partial[:3]:
            spot = view["spot"]
            distance = math.dist((spot["x"], spot["z"]),
                                 (target_xz["x"], target_xz["z"]))
            if not (VIEW_RANGE_M[0] <= distance <= VIEW_RANGE_M[1]):
                continue

            # Cheapest decisive test first.  The sweep is 4 renders; confirming
            # the occluder costs up to 12, so gain is checked before blame.
            sweep_result = parallax(controller, spot, target_xz, name, view["px"])
            if sweep_result["gain"] < min_gain:
                continue

            pool = [o for o in occluders
                    if o["entry"]["name"] != name
                    and (allow_same_class or o["vg"] != target["vg"])]
            blocker = blocker_at(controller, spot, target_xz, name, pool[:6],
                                 view["px"])
            if blocker is None:
                continue
            clear_px = blocker["clear_px"]
            occlusion = 1.0 - view["px"] / clear_px
            if not (band[0] <= occlusion <= band[1]) or clear_px < MIN_CLEAR_PX:
                continue

            occ_entry = blocker["candidate"]
            # Depth separation is what makes parallax possible at all: an
            # occluder at the target's own depth cannot be moved off it by ten
            # degrees.  Reported so a flat sweep is explicable rather than
            # mysterious.
            cam_to_target = distance
            cam_to_occ = math.dist(
                (spot["x"], spot["z"]),
                (occ_entry["entry"]["position"]["x"],
                 occ_entry["entry"]["position"]["z"]))
            candidates_out.append({
                "scene": scene, "moved_objects": 0,
                "target": {"name": name, "vg150_class": target["vg"],
                           "thor_type": target["entry"]["objectType"],
                           "width_m": round(target["width"], 2),
                           "position": target["entry"]["position"]},
                "occluder": {"name": occ_entry["entry"]["name"],
                             "vg150_class": occ_entry["vg"],
                             "thor_type": occ_entry["entry"]["objectType"],
                             "width_m": round(occ_entry["width"], 2),
                             "position": occ_entry["entry"]["position"]},
                "start": {"position": spot, "yaw": view["yaw"], "horizon": 10,
                          "occlusion": round(occlusion, 3),
                          "visible_px": view["px"], "unoccluded_px": clear_px,
                          "distance_m": round(distance, 2)},
                "nvs": {**sweep_result,
                        "depth_ratio": round(cam_to_occ / max(cam_to_target, 1e-6), 3)},
            })

    if candidates_out:
        # Best in the scene, not the first found: the first qualifying pose was
        # rarely the most legible one.
        return max(candidates_out, key=lambda r: r["nvs"]["gain"])
    return {"error": f"{scene}: no view with occlusion in "
                     f"{band[0]}-{band[1]} and NVS gain >= {min_gain}"}


def build(scene: str, seed: int, out: str, band, min_gain, rescues,
          width, height, allow_same_class, max_poses,
          max_targets: int = 6) -> Dict[str, Any]:
    from ai2thor.controller import Controller
    from PIL import Image

    controller = Controller(scene=scene, width=width, height=height,
                            renderInstanceSegmentation=True,
                            visibilityDistance=25.0)
    try:
        reachable = controller.step(
            action="GetReachablePositions").metadata["actionReturn"] or []
        record = search(controller, scene, reachable, band, min_gain, rescues,
                        allow_same_class, max_poses, seed, max_targets)
        if "error" in record:
            return record
        os.makedirs(out, exist_ok=True)
        target_xz = {"x": record["target"]["position"]["x"],
                     "z": record["target"]["position"]["z"]}
        event, _ = look_at(controller, record["start"]["position"], target_xz)
        Image.fromarray(event.frame).save(os.path.join(out, "start.png"))
        record["start"]["image"] = "start.png"
        with open(os.path.join(out, "scenario.json"), "w", encoding="utf-8") as h:
            json.dump(record, h, indent=1)
        return record
    finally:
        controller.stop()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--scenes", nargs="*",
                        default=["FloorPlan201", "FloorPlan203", "FloorPlan215",
                                 "FloorPlan227", "FloorPlan230"])
    parser.add_argument("--seeds", type=int, nargs="*", default=[1])
    parser.add_argument("--out", default="datasets/walkaround_found")
    parser.add_argument("--hide", type=float, nargs=2, default=[0.35, 0.65],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--min-gain", type=float, default=MIN_GAIN,
                        help="best-in-sweep pixels / reference pixels")
    parser.add_argument("--rescues", type=int, default=3)
    parser.add_argument("--max-targets", type=int, default=6,
                        help="objects tried per scene, largest first")
    parser.add_argument("--max-poses", type=int, default=45,
                        help="reachable poses sampled per target")
    parser.add_argument("--allow-same-class", action="store_true")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args(argv)

    made = 0
    for scene in args.scenes:
        for seed in args.seeds:
            record = build(scene, seed,
                           os.path.join(args.out, f"{scene}_s{seed}"),
                           tuple(args.hide), args.min_gain, args.rescues,
                           args.width, args.height, args.allow_same_class,
                           args.max_poses, args.max_targets)
            if "error" in record:
                print(f"  {record['error']}")
                continue
            made += 1
            t, o, s, n = (record["target"], record["occluder"],
                          record["start"], record["nvs"])
            print(f"  {scene} s{seed}: {t['thor_type']}({t['vg150_class']}) "
                  f"{s['occlusion']*100:.0f}% behind {o['thor_type']}"
                  f"({o['vg150_class']}) at {s['distance_m']}m  ->  "
                  f"NVS {n['best_deg']:+g} deg = {n['gain']:.2f}x  "
                  f"(depth ratio {n['depth_ratio']})")
    print(f"\n{made} scenarios under {args.out}/  (nothing was moved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
