"""
build_tabletop.py -- generate the cases, instead of hunting for them.

`find_cases.py` searched thirty stock scenes for a nameable object something
else could be slid in front of, so every constraint it satisfied was one the
room happened to permit.  Here the room is ours -- an empty rectangle, a table,
three objects -- and every property the experiment needs is set rather than
found.

  target      what the instruction asks for
  distractor  A SECOND COPY OF THE SAME ASSET, which iTHOR cannot hold: a scene
              there has no two instances of one objectType, and all 71 frozen
              iTHOR cases had zero same-class distractors, which is why the
              predicate never did any work.  With one bottle in the room, "the
              bottle behind the vase" and "any bottle" are the same instruction.
              It stands BESIDE THE OCCLUDER, at the occluder's own depth, so
              exactly one copy is behind the landmark -- see `one_case`.
  occluder    a tall narrow asset bisected along the camera-to-target ray until
              the measured occlusion lands in the band

The instruction is `Find the {target} behind the {occluder}`, true by
construction from the camera pose stored with the case.  Bought: two identical
instances, exact occlusion, geometry comparable across cases.  Paid: bare
scenes, so these numbers are about perception under controlled geometry and not
about a robot in a house.  Both belong in any write-up.

    python build_tabletop.py --n 12 --out datasets/robot/cases_hard.json
"""

from __future__ import annotations

# `python build/robot/<script>.py` puts this directory on sys.path, not
# the repo root, so `robot.*`, `vg.*` and the sibling generators would not
# resolve.  Running as `python -m build.robot.<script>` does not need
# this; it is here so both work.  Same shim as `build/sgg/build_occlusion_dataset.py`.
import os as _os
import sys as _sys
if __package__ in (None, ""):
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import collections
import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from robot.world.proc_scene import (ROOM, aim_at, look_from, open_room, place_on,
                              spawn, surface_top, visible_box, visible_pixels)
from vg.vg150 import THOR_TO_VG150

#: Classes a target may belong to.  CHOSEN BY RANK, NOT BY ARGMAX: conditioning
#: sorts all 200 queries by p(class) and takes the top N, so what matters is the
#: rank of the object's own query.  Every class here made the top 10 on 4 clean
#: renders out of 4, three of them after scoring 0/4 on argmax -- EGTR calls a
#: laptop "box" and an aubergine "banana" while still ranking them first for the
#: true class.  An argmax screen would have left `bottle` and `bowl`.
TARGET_CLASSES = ("bottle", "cup", "bowl", "fruit", "food", "vegetable",
                  "box", "book", "laptop")

#: Landmark classes: tall and narrow hides a tabletop object without its own box
#: swallowing the frame, as an iTHOR HousePlant's did over a fifth of the picture
#: (making `behind the plant` satisfiable by almost anything).  `vase` is gone --
#: landmark of a third of the old cases, top 10 on 1 render out of 4, the worst
#: of any class screened.
OCCLUDER_CLASSES = ("bottle", "box", "lamp", "plant", "clock", "laptop")

#: Size bands, metres: big enough to detect, small enough to hide, and the
#: occluder has to out-top the target.  The target floor was 0.07, which is ~30
#: pixels tall -- a smudge.
TARGET_SIZE = (0.10, 0.22)
OCCLUDER_SIZE = (0.20, 0.35)

#: Below this the question stops being about occlusion and becomes one about the
#: detector's resolution.
MIN_PIXELS = 300

#: How much of the DISTRACTOR the occluder may hide.  A guard, not the mechanism
#: (the occluder slides away from it), but a wide one can still reach across, and
#: a case where both copies are behind the landmark tests nothing.
MAX_DISTRACTOR_OCCLUSION = 0.10

TABLE = "Dining_Table_7_1"

#: How far inside the table's footprint an object must sit, metres.  The table is
#: round and its axis-aligned box is square, so that box's corners are thin air.
TABLE_MARGIN = 0.25

#: Camera distance from the target, metres -- TRIED NEAREST FIRST, nearest
#: collision-free one wins.  Distance is the only lever on apparent size (a 10 cm
#: object is 43 px tall at 1.2 m and 80 px at 0.65 m; the 60 degree field is
#: assumed by `unproject` and the lemniscate alike), but the pose must be one the
#: robot can STAND at: this table's collider reaches 0.23 m past its visible edge
#: and half the poses in a 0.85-1.05 band were illegal.  Forcing the teleport, as
#: this once did, put every START pose somewhere the robot could not be, and the
#: motion experiment then measured a walk through furniture.  The fix is a target
#: on the table's NEAR HALF, not a camera further back, which shrinks it again.
STANDOFF = (0.55, 0.70, 0.85, 1.00, 1.15)

#: Where the target sits, metres in from the table's near edge.  Bounded below by
#: the OCCLUDER, which goes some fraction of the way towards the camera and has
#: to land on the table too.
NEAR_EDGE_INSET = (0.45, 0.60)

#: Where on the camera-to-target ray the occluder may sit, as a fraction of the
#: way to the camera.  A second axis is needed because offset alone fails for
#: most assets -- with `along` pinned at 0.45, 21 of 24 attempts never landed in
#: the band, a narrow bottle never covering a quarter of the target however it is
#: slid and a wide alarm clock jumping from nothing to everything between
#: adjacent probes.  Distance sets apparent WIDTH, offset sets how much of the
#: target that width COVERS.
ALONG = (0.25, 0.40, 0.55, 0.70)

#: A half turn for classes whose modelled front faces away from a camera looking
#: down +z.  A laptop seen from behind is a featureless wedge -- the screen both
#: does the occluding and is what makes it recognisably a laptop -- so facing
#: changes occlusion and naming together.  180 degrees leaves the axis-aligned
#: box identical, so it does not interact with `occluder_yaw`'s quarter turn.
CLASS_YAW = {"laptop": 180.0}

#: GetAssetDatabase is a round trip returning the whole catalogue and is static
#: for a build, so it is fetched once per controller.
_ASSET_DB: Dict[int, Dict[str, Any]] = {}


def asset_db(controller) -> Dict[str, Any]:
    """The build's asset database, fetched once."""
    if id(controller) not in _ASSET_DB:
        _ASSET_DB[id(controller)] = controller.step(
            action="GetAssetDatabase").metadata["actionReturn"]
    return _ASSET_DB[id(controller)]


def reject(why: str) -> None:
    """Say why this attempt was thrown away.  Returns None, so `return reject(…)`."""
    print(f"    - {why}", flush=True)


def settle(controller, name: str, position: Dict[str, float],
           yaw: float = 0.0):
    """Teleport `name` there at `yaw`, and return the frame that follows.

    `yaw` is passed and not zeroed: the occluder is spawned turned so its wider
    side faces the camera, and a hardcoded zero in the bisection's inner teleport
    silently undid that before a single occlusion was measured.
    """
    controller.step(action="TeleportObject", objectId=name, position=position,
                    rotation={"x": 0, "y": float(yaw), "z": 0},
                    forceAction=True)
    return controller.step(action="Pass")


def catalogue(controller, classes: Sequence[str], size: Tuple[float, float],
              width: float = 0.30) -> Dict[str, List[str]]:
    """Asset ids per VG150 class, from the build's own database.

    Looked up rather than written down: a hand-typed list had `Mug_5`, `Bowl_6`
    and `Apple_1` in it, two of which do not exist, and it failed at spawn with a
    bare KeyNotFoundException.

    `width` caps horizontal extent, keeping a landmark's own detection box off
    the frame.  A flag and not a constant because 0.30 admits only TWO `box`
    assets in the occluder height band against 36 bottles -- 40 cases on 2
    objects, worse concentration than the `lamp` it replaced (18 of 40 cases on
    one asset).  At 0.40 it is 11 boxes.
    """
    out: Dict[str, List[str]] = collections.defaultdict(list)
    for asset, entry in asset_db(controller).items():
        if entry.get("primaryProperty") == "Static":
            continue
        box = entry["boundingBox"]
        height = box["max"]["y"] - box["min"]["y"]
        extent = max(box["max"]["x"] - box["min"]["x"],
                     box["max"]["z"] - box["min"]["z"])
        if not (size[0] <= height <= size[1] and extent <= width):
            continue
        vg = THOR_TO_VG150.get(entry["objectType"])
        if vg in classes:
            out[vg].append(asset)
    return {k: sorted(v) for k, v in out.items()}


def occluder_yaw(controller, asset: str, vg_class: str = "") -> float:
    """The turn putting this asset's wider side, and its front, towards the view.

    `dz > dx` means it is modelled deeper than wide and a quarter turn about Y
    swaps the two; `CLASS_YAW` adds any half turn the class needs.
    """
    box = asset_db(controller)[asset]["boundingBox"]
    dx = box["max"]["x"] - box["min"]["x"]
    dz = box["max"]["z"] - box["min"]["z"]
    return (90.0 if dz > dx else 0.0) + CLASS_YAW.get(vg_class, 0.0)


def stage_occluder(controller, name: str, rest_y: float, target_xz, camera_xz,
                   clear: int, band: Tuple[float, float, float],
                   sense: float = 1.0, yaw: float = 0.0,
                   gap: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Try each distance, bisecting sideways at each, and keep the closest hit.

    `gap` pins the separation in METRES instead of sweeping `ALONG`'s fractions,
    which were the wrong unit once the occluders got wide: the floor of 0.25 puts
    a 0.40 m box just 0.21 m from the target at the near standoff and the two
    then intersect.  Metres are also what `on_table` and `NEAR_EDGE_INSET` use,
    so a pinned gap makes the on-table constraint checkable in advance rather
    than discovered as a rejection.
    """
    best = None
    distance = float(np.linalg.norm(camera_xz - target_xz))
    alongs = ((gap / max(distance, 1e-6),) if gap is not None else ALONG)
    for along in alongs:
        found = bisect_occluder(controller, name, rest_y, target_xz, camera_xz,
                                clear, band, along, sense=sense, yaw=yaw)
        if found is None:
            continue
        if best is None or abs(found["occlusion"] - band[1]) < abs(
                best["occlusion"] - band[1]):
            best = found
        if abs(best["occlusion"] - band[1]) < 0.02:
            break
    if best is not None:
        settle(controller, name, best["position"], yaw)
    return best


def bisect_occluder(controller, name: str, rest_y: float, target_xz, camera_xz,
                    clear: int, band: Tuple[float, float, float],
                    along: float = 0.45, steps: int = 12, sense: float = 1.0,
                    yaw: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Slide the occluder SIDEWAYS across the target until the band is hit.

    Moving it along the ray changes only apparent SIZE, a coarse control: the
    first version stepped 27% -> 43% -> 71% -> 91% between adjacent probes and
    landed inside a 25-50% band on 2 attempts out of 24.  Lateral offset changes
    how much of the target the silhouette COVERS, which is smooth and monotone,
    so a bisection converges on any band width.

    `along` is fixed here because the two axes trade off and one suffices.
    `rest_y` is `place_on`'s height, held fixed -- sliding is horizontal, and
    recomputing it per probe would let the occluder drift vertically while the
    bisection thinks it is only moving sideways.  `sense` is WHICH WAY it slides,
    set opposite the distractor: sliding always one way let it wander into the
    second copy and hide part of that too, making both copies "behind the box"
    and emptying the instruction of content.
    """
    ray = camera_xz - target_xz
    distance = float(np.linalg.norm(ray))
    forward = ray / max(distance, 1e-6)
    sideways = np.array([-forward[1], forward[0]])
    base = target_xz + forward * (distance * along)

    def occlusion_at(offset: float) -> Tuple[float, Dict[str, float]]:
        where = base + sideways * offset * sense
        position = {"x": float(where[0]), "y": float(rest_y),
                    "z": float(where[1])}
        seen = visible_pixels(settle(controller, name, position, yaw), "target")
        return (1.0 - seen / clear if clear else 1.0), position

    # Offset 0 covers the most, far enough sideways covers nothing, and more
    # offset hides less -- so bisect between them for the target occlusion.
    low, high = 0.0, 0.35
    best = None
    for _ in range(steps):
        mid = (low + high) / 2.0
        occlusion, position = occlusion_at(mid)
        if band[0] <= occlusion <= band[2] and (
                best is None
                or abs(occlusion - band[1]) < abs(best["occlusion"] - band[1])):
            best = {"offset": round(mid, 4), "along": along,
                    "occlusion": round(occlusion, 3), "position": position}
        low, high = (mid, high) if occlusion > band[1] else (low, mid)

    # PUT IT BACK: the search leaves the occluder where the LAST probe went, not
    # where the best one was.  An early run recorded a case at 47% occlusion
    # whose rendered target had zero pixels, the final probe having hidden it
    # completely.
    if best is not None:
        settle(controller, name, best["position"], yaw)
    return best


def stand_back(controller, x: float, tz: float) -> Optional[float]:
    """
    The nearest z the robot can legally stand at while looking down +z at `tz`.

    `STANDOFF` nearest-first, keeping the first pose THOR accepts WITHOUT
    `forceAction`.  This is the whole guarantee that a case can be replayed and
    walked: `rebuild` needs no force, and `walk` can refuse an illegal
    destination honestly instead of teleporting through the table and counting it
    afterwards.
    """
    for standoff in STANDOFF:
        z = tz - standoff
        event = controller.step(action="Teleport",
                                position={"x": float(x), "y": 0.9,
                                          "z": float(z)},
                                rotation={"x": 0, "y": 0.0, "z": 0},
                                horizon=0.0, standing=True, forceAction=False)
        if event.metadata["lastActionSuccess"]:
            return z
    return None


def on_table(table: Dict[str, Any], x: float, z: float) -> bool:
    """Is (x, z) somewhere an object can stand on this table?"""
    box = table["axisAlignedBoundingBox"]
    half_x = box["size"]["x"] / 2.0 - TABLE_MARGIN
    half_z = box["size"]["z"] / 2.0 - TABLE_MARGIN
    return (abs(x - box["center"]["x"]) <= half_x
            and abs(z - box["center"]["z"]) <= half_z)


def clear_of(controller, cam, yaw: float, horizon: float, name: str,
             blocker: str, restore: Dict[str, float]) -> int:
    """
    How many pixels of `name` would show if `blocker` were not there.

    Needed because the distractor is placed AFTER the occluder is staged, so no
    earlier frame has it unobstructed and there is no denominator for the
    fraction the occluder hides.  Lifting the blocker out of the room and putting
    it back is exact: the procedural scene runs no physics, so nothing settles
    differently for having been moved.
    """
    controller.step(action="TeleportObject", objectId=blocker,
                    position={"x": 0.0, "y": -5.0, "z": 0.0},
                    rotation={"x": 0, "y": 0, "z": 0}, forceAction=True)
    seen = visible_pixels(
        look_from(controller, cam[0], cam[1], yaw, horizon, force=True), name)
    settle(controller, blocker, restore)
    return seen


def one_case(controller, rng: random.Random, index: int,
             band: Tuple[float, float, float], catalogues,
             gap: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Build one scene and return its case record, or None if it will not stage."""
    targets, occluders = catalogues
    target_class = sorted(targets)[index % len(targets)]
    asset = rng.choice(targets[target_class])
    # The landmark must not be the target's own class, or "the bottle behind the
    # bottle" names three bottles and nothing disambiguates.
    choices = [c for c in occluders if c != target_class]
    if not choices:
        return None
    occ_class = rng.choice(choices)
    occ_asset = rng.choice(occluders[occ_class])

    controller.reset()
    centre = ROOM / 2.0
    table = spawn(controller, TABLE, "table", centre, 0.0, centre + 0.6)
    top = surface_top(table)

    # THE TARGET FIRST, ALONE: its unoccluded pixel count is what the bisection
    # is measured against, so nothing else may be in the picture yet.
    #
    # THE INSET FOLLOWS THE GAP.  The occluder sits `gap` in front of the target
    # and has to be on the table too, so the target's distance in from the near
    # edge cannot be less than the gap plus `on_table`'s margin.  Deriving that
    # rather than rejecting afterwards stops a larger gap from silently throwing
    # away most attempts.
    edge = (table["axisAlignedBoundingBox"]["center"]["z"]
            - table["axisAlignedBoundingBox"]["size"]["z"] / 2.0)
    tx = centre + rng.uniform(-0.15, 0.15)
    floor = (NEAR_EDGE_INSET[0] if gap is None
             else max(NEAR_EDGE_INSET[0], gap + TABLE_MARGIN + 0.05))
    tz = edge + rng.uniform(floor, max(floor + 0.15, NEAR_EDGE_INSET[1]))
    sign = rng.choice((-1.0, 1.0))
    _, target_pos = place_on(controller, asset, "target", tx, top, tz)

    z = stand_back(controller, tx, tz)
    if z is None:
        return reject("nowhere legal to stand and see it")
    cam = np.array([tx, z])
    yaw = 0.0
    event, horizon = aim_at(controller, cam[0], cam[1], yaw,
                            (tx, top + 0.08, tz))
    clear = visible_pixels(event, "target")
    if clear < MIN_PIXELS:
        return reject(f"target only {clear} px unoccluded")

    # TURN THE OCCLUDER'S WIDER SIDE TOWARDS THE CAMERA.  The camera looks down
    # +z, so what covers the target is the occluder's X extent -- and these
    # assets are not axisymmetric: a soap bottle 0.16 across one way is 0.08 the
    # other.  At yaw 0 whichever face the artist modelled facing +z is presented,
    # and several bottles never reached the band for want of width they already
    # had.  A quarter turn about the STANDING axis is physically free (still
    # upright on its base, so `place_on`'s measurement holds) and is the only
    # rotation `spawn` exposes.
    occ_yaw = occluder_yaw(controller, occ_asset, occ_class)
    _, occ_pos = place_on(controller, occ_asset, "occluder", tx, top,
                          tz - (gap if gap is not None else 0.2), occ_yaw)
    staged = stage_occluder(controller, "occluder", occ_pos["y"],
                            np.array([tx, tz]), cam, clear, band, sense=sign,
                            yaw=occ_yaw, gap=gap)
    if staged is None:
        return reject(f"{occ_asset} never landed in the band")
    if not on_table(table, staged["position"]["x"], staged["position"]["z"]):
        return reject("occluder ended up over the table's edge")

    # THE DISTRACTOR GOES BESIDE THE OCCLUDER, NOT BESIDE THE TARGET, because
    # `behind` is a statement about DEPTH.  It used to share the target's depth
    # one lateral step away, and with the occluder in front of both copies both
    # were behind it -- "the vegetable behind the box" picked out two vegetables.
    # Sliding the occluder away from the distractor, the previous fix, only
    # stopped their SILHOUETTES overlapping, a weaker property and not what the
    # word means.  At the occluder's own depth the distractor is level with it,
    # hence beside the landmark rather than behind it, and exactly one copy
    # satisfies the relation.  The cost is that the near copy renders larger; it
    # is the same asset, so the instruction still turns on the relation and not
    # on appearance.
    dz = staged["position"]["z"]
    dx = tx - sign * rng.uniform(0.28, 0.42)
    if not on_table(table, dx, dz):
        return reject(f"distractor at ({dx:.2f}, {dz:.2f}) is off the table")
    _, distract_pos = place_on(controller, asset, "distract", dx, top, dz)
    event = look_from(controller, cam[0], cam[1], yaw, horizon, force=True)

    # Now VERIFY what the placement was meant to guarantee: a wide occluder can
    # still reach across, and a near distractor can cover the thing it competes
    # with.
    staged_distract = visible_pixels(event, "distract")
    if staged_distract < MIN_PIXELS:
        return reject(f"distractor only {staged_distract} px")
    hidden_distract = 1.0 - staged_distract / max(clear_of(
        controller, cam, yaw, horizon, "distract", "occluder",
        staged["position"]), 1)
    if hidden_distract > MAX_DISTRACTOR_OCCLUSION:
        return reject(f"occluder still hides {hidden_distract:.0%} of the "
                      f"distractor, so both copies are 'behind' it")
    # The distractor is nearer the camera than the target, so for the first time
    # it can hide it.  The bisection measured the target before it existed, so
    # that number is no longer the truth.
    target_px = visible_pixels(event, "target")
    hidden_target = 1.0 - target_px / max(clear, 1)
    if not band[0] <= hidden_target <= band[2]:
        return reject(f"distractor moved the target to {hidden_target:.0%} "
                      f"hidden, outside the band")

    # THE TARGET MUST BE THE SMALLER OBJECT, or "the cup behind the laptop" is a
    # sentence about a cup bigger than the laptop and the premise is gone.  The
    # band does not catch it -- a wide occluder can hide 35% of something much
    # larger than itself.
    #
    # Measured, not assumed: `probe_class.py --width-cap 0.30` staged this
    # generator's own target pool and found the `box` assets at 15928 px against
    # cup's 2540, realising a median target rank of 14.  The reason is the
    # DISTRACTOR, the same asset unoccluded and so larger still: an object that
    # size is covered by more than a dozen queries which, being the instructed
    # class, fill the top of the p(class) ranking ahead of the target.  Capping
    # the target against the landmark caps the distractor with it.  The
    # comparison is lenient by construction -- the occluder is `gap` nearer the
    # camera, so at equal physical size it renders larger -- which makes it a
    # guard against the absurd rather than a tuning knob.
    occluder_px = visible_pixels(event, "occluder")
    if clear >= occluder_px:
        return reject(f"target is {clear} px unoccluded against the landmark's "
                      f"{occluder_px}: the target is the bigger object")

    return {
        "scene": f"tabletop|{index}",
        "instruction": f"Find the {target_class} behind the {occ_class}",
        "predicate": "behind",
        "subject_class": target_class, "object_class": occ_class,
        "target_name": "target", "landmark_name": "occluder",
        "distractor_name": "distract", "occluder_name": "occluder",
        "occluder_position": staged["position"],
        # WITHOUT THIS THE REPLAY IS A DIFFERENT SCENE.  `rebuild` respawns from
        # the record and `spawn`'s rotation defaults to 0, so a case built with a
        # turned occluder would be replayed with an unturned one, and the
        # occlusion measured at build time is not the occlusion any probe, fusion
        # run or motion run ever saw.
        "occluder_yaw": occ_yaw,
        # Measured on the FINAL scene; the bisection's was taken before the
        # distractor existed.
        "staged_occlusion": round(hidden_target, 3),
        "bisected_occlusion": staged["occlusion"],
        "clear_px": clear, "target_px": target_px,
        "distractor_px": staged_distract, "occluder_px": occluder_px,
        "distractor_occlusion": round(hidden_distract, 3),
        "target_box": visible_box(event, "target"),
        "table": {"asset": TABLE, "x": centre, "z": centre + 0.6, "top": top},
        "objects": [{"name": "target", "asset": asset,
                     "position": target_pos},
                    {"name": "distract", "asset": asset,
                     "position": distract_pos},
                    {"name": "occluder", "asset": occ_asset,
                     "position": staged["position"], "yaw": occ_yaw}],
        "start": f"{cam[0]:.4f},{cam[1]:.4f},{yaw:.2f},{horizon:.2f}",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-occlusion", type=float, default=0.25)
    ap.add_argument("--target-occlusion", type=float, default=0.35)
    ap.add_argument("--max-occlusion", type=float, default=0.50)
    # SQUARE, to match the experiment; see `build_slot`.
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    # WHICH CLASSES, as flags rather than by editing the constants, because those
    # constants are what `reports/0812.md` reproduces and a silent change would
    # make that line describe a different dataset.
    #
    # Worth narrowing, on a measurement: asked whether ANY query covering the
    # object argmaxes to the instructed class, the landmark classes score `box`
    # 8/8, `bottle` 5/8, `laptop` 1/1 -- and `lamp` 0/18, `clock` 0/5.  EGTR
    # never names `Desk_Lamp_1` a lamp, and `lamp` was 18 of 40 landmarks off a
    # SINGLE asset, so half the set shared one object.  Effective n is the reason
    # to narrow, not nameability: selection ranks by p(class), not argmax, and
    # under p(class) the landmark is found in 39 of 40.
    ap.add_argument("--target-classes", nargs="+", default=list(TARGET_CLASSES),
                    metavar="C", help="VG150 classes the target may be drawn "
                                      "from; see the note in the source")
    ap.add_argument("--occluder-classes", nargs="+",
                    default=list(OCCLUDER_CLASSES), metavar="C",
                    help="VG150 classes the landmark may be drawn from")
    ap.add_argument("--gap", type=float, default=None, metavar="M",
                    help="separation between target and landmark in METRES.  "
                         "Omitted, `ALONG`'s fractions are swept as before, "
                         "whose floor of 0.25 puts a wide occluder close enough "
                         "to intersect the target.  Setting this also pushes the "
                         "target deeper so the landmark stays on the table.")
    ap.add_argument("--occluder-width", type=float, default=0.30, metavar="M",
                    help="cap on the landmark's horizontal extent.  0.30 (the "
                         "old hardcoded value) admits only 2 `box` assets; 0.40 "
                         "admits 11.  See `catalogue`.")
    ap.add_argument("--out", default="datasets/robot/cases_hard.json")
    args = ap.parse_args(argv)

    band = (args.min_occlusion, args.target_occlusion, args.max_occlusion)
    rng = random.Random(args.seed)
    controller = open_room(args.width, args.height, args.fov)
    cases: List[Dict[str, Any]] = []
    try:
        catalogues = (catalogue(controller, tuple(args.target_classes),
                                TARGET_SIZE),
                      catalogue(controller, tuple(args.occluder_classes),
                                OCCLUDER_SIZE, args.occluder_width))
        for label, pool in zip(("targets:  ", "occluders:"), catalogues):
            print("  " + label + ", ".join(f"{k}x{len(v)}"
                                           for k, v in pool.items()))
        attempt = 0
        while len(cases) < args.n and attempt < args.n * 4:
            case = None
            try:
                case = one_case(controller, rng, attempt, band, catalogues,
                                args.gap)
            except Exception as error:                       # noqa: BLE001
                print(f"  ! {type(error).__name__}: {error}", flush=True)
            attempt += 1
            if case:
                cases.append(case)
                print(f"[{len(cases)}/{args.n}] {case['instruction']:38s} "
                      f"{case['staged_occlusion']:.0%} hidden, target "
                      f"{case['target_px']} px, distractor "
                      f"{case['distractor_px']} px", flush=True)
    finally:
        controller.stop()

    if not cases:
        print("no cases built")
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"predicate_families": ["behind"],
                   "occlusion_band": list(band),
                   "procedural": True, "cases": cases}, fh, indent=1)
    occ = sorted(c["staged_occlusion"] for c in cases)
    print(f"\n{len(cases)}/{attempt} attempts -> {args.out}")
    print(f"  occlusion {occ[0]:.0%}-{occ[-1]:.0%}, median "
          f"{occ[len(occ) // 2]:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
