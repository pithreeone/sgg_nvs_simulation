"""
build_sideview_sweep.py -- one staged occlusion, photographed from an arc of viewpoints.

The qualitative figure the paper needs: an object the camera cannot see properly from where it
stands, and the same object from a few degrees to the side.  Mining that out of `occlusion_ds4`
was possible but never clean -- its ten views per scene are ten independent poses, so between the
occluded frame and the clear one the distance, the framing and half the room change too, and a
reader cannot tell which of those moved the detector's score.

Here exactly one thing varies.  The scene is staged once, the camera flies an arc of constant
radius ABOUT THE TARGET, and every frame is aimed back at it -- so apparent size, image position
and lighting hold while the sightline past the occluder opens up.  What changes across the strip
is occlusion and nothing else, which is the only reason the object-score curve in
`../sgg_nvs/scratch/sideview_figure.py` means what its caption will say it means.

WHY THE OCCLUDER IS OFFSET SIDEWAYS rather than parked on the camera ray.  On the ray was tried
first, for symmetry: an occluder covering the target's middle clears the same way whichever
direction the camera goes.  It does not work, for two reasons measured on the first render.  To
hide 65% of the target while centred on it, the occluder must appear SMALLER than the target --
a 2 cm object in front of a bottle, which is not a picture of anything being hidden.  And the
parallax is too small to undo: at a 0.16 m gap and a 1.0 m radius, swinging 50 degrees moved the
occluder only 0.12 m across a 0.25 m laptop, so occlusion went UP with azimuth (0.64 -> 0.78)
instead of down.  Offset sideways, as `build_tabletop.bisect_occluder` puts it, both problems go:
a large occluder hides a fraction of the target from one side, and clearing it needs only the
target's own width of parallax, which a 0.25 m gap supplies by 20 degrees.

The arc is therefore ASYMMETRIC, and that is kept rather than hidden.  Swinging away from the
occluder reveals the target; swinging towards it drags the silhouette across and hides more before
it passes.  That is what walking around a real occluder does, and a figure showing both is worth
more than one showing only the good direction.

THE CAMERA IS CROUCHED, and this is not a detail.  `robot.world.proc_scene.look_from` measures a
standing camera at 1.575 m against a 0.834 m tabletop -- 27 degrees down -- and the sightline to
anything behind a tabletop occluder then passes OVER it: a Box_21 in front of a Laptop_2 hid 3%
standing and 50% crouched, same scene.  A standing sweep photographs a table, not an occlusion.

    python build/sgg/build_sideview_sweep.py --n 8 --occlusion 0.65
    python build/sgg/build_sideview_sweep.py --n 1 --occlusion 0.80 --span 60 --frames 25

Output, per case, under --out/<case_id>/:
    frame_00.png .. frame_NN.png     the arc, index 0 at --span degrees, centre frame at azimuth 0
    sweep.json                       per frame: azimuth, camera pose, GT visible px / box / occlusion
"""

from __future__ import annotations

# `python build/sgg/<script>.py` puts build/sgg/ on sys.path, not the repo root, so `robot.*` and the sibling
# generators would not resolve.  Same shim as build_occlusion_dataset.py.
import os as _os
import sys as _sys
if __package__ in (None, ""):
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from build.robot.build_tabletop import (MIN_PIXELS, OCCLUDER_CLASSES,
                                               OCCLUDER_SIZE, TABLE,
                                               TABLE_MARGIN, TARGET_CLASSES,
                                               TARGET_SIZE, catalogue, on_table,
                                               occluder_yaw, stage_occluder)
from robot.world.proc_scene import (ROOM, aim_at, open_room, place_on, spawn,
                              surface_top, visible_box, visible_pixels)

#: Where the target sits, metres in from the table's near edge.  Deeper than
#: `build_tabletop.NEAR_EDGE_INSET`, and it has to be: the occluder goes `GAP` in front of the
#: target and must stand on the table too, so the inset has to cover the gap plus `on_table`'s
#: margin.
INSET = 0.70

#: Separation between target and occluder along the camera ray, metres.  This is the lever on
#: PARALLAX, which is the whole mechanism of the figure: the occluder's apparent shift as the
#: camera swings by `a` is about `gap * sin(a) / (radius - gap)`, and it has to exceed the target's
#: own apparent width for the sweep to clear it.  At 0.25 m and radius 0.85 that is 0.12 rad by 20
#: degrees against a target 0.10-0.22 m wide subtending at most 0.26 -- so the far end of the arc
#: clears comfortably while the middle of it is still partly blocked.
GAP = 0.25

#: Camera height, metres, and it is NOT the default.  See the docstring: a standing camera is
#: 1.575 m over a 0.834 m tabletop and looks 27 degrees DOWN, and its sightline to anything behind
#: a tabletop occluder passes over the top of that occluder.  `look_from(standing=False)` puts the
#: camera at 0.900 m, essentially level with the tabletop, which is the height at which a thing
#: standing on a table can hide a thing behind it.
STANDING = False


def arc_pose(target_xz, radius: float, azimuth_deg: float) -> Tuple[float, float, float]:
    """Camera (x, z, yaw) on a circle about the target.  Azimuth 0 is the staged viewpoint.

    THOR yaw is clockwise from +z with forward `(sin yaw, ., cos yaw)`, so a camera placed at
    `target + R(sin a, -cos a)` looks back at the target with yaw `-a` exactly.
    """
    a = math.radians(azimuth_deg)
    x = float(target_xz[0] + radius * math.sin(a))
    z = float(target_xz[1] - radius * math.cos(a))
    return x, z, -float(azimuth_deg)


def one_case(controller, rng: random.Random, index: int, catalogues,
             want: float, radius: float, azimuths: Sequence[float],
             out_dir: str) -> Optional[Dict[str, Any]]:
    """Stage one scene, fly the arc, write the frames.  None if it will not stage."""
    targets, occluders = catalogues
    target_class = sorted(targets)[index % len(targets)]
    asset = rng.choice(targets[target_class])
    choices = [c for c in occluders if c != target_class]
    if not choices:
        return None
    occ_class = rng.choice(choices)
    occ_asset = rng.choice(occluders[occ_class])
    case_id = f"{index:02d}_{target_class}_behind_{occ_class}"
    print(f"[{case_id}] target={asset} occluder={occ_asset}", flush=True)

    controller.reset()
    centre = ROOM / 2.0
    table = spawn(controller, TABLE, "table", centre, 0.0, centre + 0.6)
    top = surface_top(table)
    box = table["axisAlignedBoundingBox"]
    edge = box["center"]["z"] - box["size"]["z"] / 2.0

    # THE TARGET ALONE FIRST: its unoccluded pixel count is the denominator every occlusion below
    # is measured against, so nothing else may be in the picture yet.
    tx = centre
    tz = edge + INSET
    target, target_pos = place_on(controller, asset, "target", tx, top, tz)
    ty = target["axisAlignedBoundingBox"]["center"]["y"]
    target_xz = np.array([tx, tz])

    cx, cz, yaw = arc_pose(target_xz, radius, 0.0)
    event, _ = aim_at(controller, cx, cz, yaw, (tx, ty, tz), standing=STANDING)
    clear = visible_pixels(event, "target")
    if clear < MIN_PIXELS:
        print(f"    - target only {clear} px unoccluded")
        return None

    # WHICH WAY THE OCCLUDER SLIDES IS FIXED, not drawn: it decides which half of the arc reveals
    # the target, and a figure whose reveal direction flips between cases cannot be read as a
    # strip.  `sense=+1` puts the occluder on the +x side, so NEGATIVE azimuth is the direction
    # that clears it -- see the asymmetry note in the module docstring.
    occ_yaw = occluder_yaw(controller, occ_asset, occ_class)
    _, occ_pos = place_on(controller, occ_asset, "occluder", tx, top, tz - GAP, occ_yaw)
    staged = stage_occluder(controller, "occluder", occ_pos["y"], target_xz,
                            np.array([cx, cz]), clear,
                            (max(0.0, want - 0.12), want, min(0.98, want + 0.12)),
                            sense=1.0, yaw=occ_yaw, gap=GAP)
    if staged is None:
        print(f"    - {occ_asset} never landed in the band")
        return None
    if not on_table(table, staged["position"]["x"], staged["position"]["z"]):
        print("    - occluder ended up over the table's edge")
        return None
    print(f"    staged {staged['occlusion']:.0%} hidden at offset {staged['offset']:.2f} m "
          f"(asked {want:.0%})", flush=True)

    frames: List[Dict[str, Any]] = []
    os.makedirs(os.path.join(out_dir, case_id), exist_ok=True)
    for i, azimuth in enumerate(azimuths):
        cx, cz, yaw = arc_pose(target_xz, radius, azimuth)
        event, horizon = aim_at(controller, cx, cz, yaw, (tx, ty, tz), standing=STANDING)
        name = f"frame_{i:02d}.png"
        Image.fromarray(event.frame).save(os.path.join(out_dir, case_id, name))
        seen = visible_pixels(event, "target")
        frames.append({
            "index": i, "image": name, "azimuth_deg": round(float(azimuth), 2),
            "camera": {"x": round(cx, 4), "z": round(cz, 4),
                       "yaw": round(yaw, 2), "horizon": round(horizon, 2)},
            "visible_px": seen,
            # Occlusion against the STAGED-VIEW clear count, so 0 means "as visible as this object
            # ever gets".  The arc holds the radius, so apparent size is constant and the ratio is
            # a statement about the occluder rather than about how near the camera got.
            "occlusion": round(max(0.0, 1.0 - seen / clear), 3),
            "gt_box": visible_box(event, "target"),
            "occluder_px": visible_pixels(event, "occluder"),
            # Needed downstream, not here: once the target is mostly hidden its visible box is a
            # sliver beside the occluder, and a detector query that found the OCCLUDER can overlap
            # that sliver well enough to be matched to the target.  `sideview_figure.py` compares
            # the two boxes to catch it, and cannot without this one.
            "occluder_box": visible_box(event, "occluder"),
        })

    record = {
        "case_id": case_id,
        "target": {"class": target_class, "asset": asset, "position": target_pos},
        "occluder": {"class": occ_class, "asset": occ_asset, "yaw": occ_yaw,
                     "position": staged["position"], "offset_m": staged["offset"],
                     "gap_m": GAP},
        "table": {"asset": TABLE, "x": centre, "z": centre + 0.6, "top": top},
        "radius_m": radius,
        "standing": STANDING,
        "clear_px": clear,
        "requested_occlusion": want,
        "staged_occlusion": staged["occlusion"],
        "frames": frames,
    }
    with open(os.path.join(out_dir, case_id, "sweep.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=8, help="how many cases to stage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--occlusion", type=float, default=0.65,
                    help="how much of the target the occluder hides at azimuth 0")
    ap.add_argument("--span", type=float, default=50.0,
                    help="the arc runs from -span to +span degrees about the target")
    ap.add_argument("--frames", type=int, default=21, help="frames on the arc, odd so one sits at 0")
    ap.add_argument("--radius", type=float, default=0.85, help="camera distance from the target, m")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--target-classes", nargs="+", default=list(TARGET_CLASSES))
    ap.add_argument("--occluder-classes", nargs="+", default=list(OCCLUDER_CLASSES))
    ap.add_argument("--out", default="datasets/sgg/sideview_demo")
    args = ap.parse_args(argv)

    if args.frames % 2 == 0:
        raise SystemExit("--frames must be odd so one frame lands on azimuth 0")
    azimuths = np.linspace(-args.span, args.span, args.frames)

    rng = random.Random(args.seed)
    controller = open_room(args.width, args.height, args.fov)
    catalogues = (catalogue(controller, args.target_classes, TARGET_SIZE),
                  catalogue(controller, args.occluder_classes, OCCLUDER_SIZE))
    print(f"targets {[f'{k}:{len(v)}' for k, v in sorted(catalogues[0].items())]}")
    print(f"occluders {[f'{k}:{len(v)}' for k, v in sorted(catalogues[1].items())]}")

    os.makedirs(args.out, exist_ok=True)
    built = []
    for index in range(args.n):
        record = one_case(controller, rng, index, catalogues, args.occlusion,
                          args.radius, azimuths, args.out)
        if record is not None:
            built.append({"case_id": record["case_id"],
                          "staged_occlusion": record["staged_occlusion"],
                          "clear_px": record["clear_px"]})
    controller.stop()

    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump({"span_deg": args.span, "frames": args.frames,
                   "radius_m": args.radius, "requested_occlusion": args.occlusion,
                   "cases": built}, fh, indent=1)
    print(f"\n{len(built)}/{args.n} staged -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
