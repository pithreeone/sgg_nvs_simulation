"""
plot_paths.py -- the two arms' routes, from above.

A table saying "evidence walked 0.41 m and random 0.13 m to the same result"
does not show WHAT the difference in bearing looked like.  This draws the floor
plane -- AI2-THOR's x-z, metres -- with both arms starting from the same pose
and arcing about the same anchor, so the shapes can be compared directly.

    square      where both arms START -- they share it, so the two routes
                always leave from the same point
    solid       the evidence arm, marker per step
    dashed      the random control
    cross       P-hat, the point both arcs turn about
    filled dot  the instructed instance; hollow, the landmark
    ring        where an arm stopped -- green if it stopped on the right
                instance, red if it stopped on the wrong one

An arm that stops early has a short path by definition, so read the markers
before the length: a long confident-looking route that ends in a red ring is
worse than a short one that ends in green.

    python viz/plot_paths.py --results nvs_pilot/move_paths.json --n 6
"""

from __future__ import annotations

# `python viz/<script>.py` puts viz/ on sys.path, not the repo root, so the
# top-level modules would not import.  Same bootstrap as analysis/ and gen/.
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)


import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence


def window(ax, staged: Dict[str, Any]) -> None:
    """The arc of camera positions the GENERATOR measured the target visible from.

    Only `build_slot` lists have one.  Drawn as the actual floor positions rather
    than a wedge, at the radius the case was staged at, so "did the robot end up
    in the window" is read off the picture instead of inferred from an azimuth.

    The azimuth convention is `nvs_lemniscate.arc_step`'s -- camera at azimuth
    `a` sits at `(tx - r sin a, tz - r cos a)` -- which is what the generator was
    corrected to on 2026-08-15.  A file built before that names the mirror arc.
    """
    import numpy as np

    arc = staged.get("visible_arc")
    if not arc:
        return
    at = {o["name"]: o["position"] for o in staged["objects"]}
    tx, tz = at["target"]["x"], at["target"]["z"]
    radius = float(staged["radius"])
    a = np.radians(np.linspace(arc[0], arc[1], 40))
    ax.plot(tx - radius * np.sin(a), tz - radius * np.cos(a),
            color="#2ca02c", linewidth=5, alpha=0.30, solid_capstyle="round",
            zorder=1, label="window")


def draw(ax, case: Dict[str, Any], staged: Optional[Dict[str, Any]] = None
         ) -> None:
    centre = case.get("centre_xz")
    if staged:
        window(ax, staged)
    for name, style, colour in (("evidence", "-", "#1f77b4"),
                                ("random", "--", "#888888")):
        arm = case[name]
        xs = [s["xz"][0] for s in arm["trail"] if "xz" in s]
        zs = [s["xz"][1] for s in arm["trail"] if "xz" in s]
        if not xs:
            continue
        ax.plot(xs, zs, style, color=colour, marker="o", markersize=3,
                linewidth=1.4, label=name, zorder=3)
        # WHERE THE ARM ENDED, always ringed.  The episode is fixed length
        # unless `--stop-score` fired, so "stopped" is not a state an arm is
        # either in or not -- it ends somewhere either way, and that pose is what
        # a deployed robot would live with.  Ringing only early stops left most
        # panels with no verdict on them at all.
        ax.plot(xs[-1], zs[-1], "o", markersize=11, markerfacecolor="none",
                markeredgewidth=2,
                markeredgecolor="#2ca02c" if arm["correct"] else "#d62728",
                zorder=4)
    # The shared start.  Drawn last of the path elements and larger than a step
    # marker, because "which end is the beginning" is the first thing anyone
    # asks of a route and the step markers do not say.
    first = next((s["xz"] for s in case["evidence"]["trail"] if "xz" in s), None)
    if first:
        ax.plot(first[0], first[1], "s", color="#ff7f0e", markersize=9,
                zorder=6, label="start")
    if centre:
        ax.plot(centre[0], centre[1], "x", color="#444444", markersize=9,
                markeredgewidth=2, zorder=2)
    if case.get("target_xz"):
        ax.plot(*case["target_xz"], "o", color="#111111", markersize=7, zorder=5)
    if case.get("landmark_xz"):
        ax.plot(*case["landmark_xz"], "o", markerfacecolor="none",
                markeredgecolor="#111111", markersize=8, zorder=5)
    ax.set_title(f"{case['scene']}  {case['instruction'][5:]}", fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, linewidth=0.5)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results", default="nvs_pilot/move_paths.json")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--walked-only", action="store_true", default=True,
                    help="skip cases answered without moving; they plot as a dot")
    ap.add_argument("--out", default="nvs_pilot/move_paths.png")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = json.load(open(args.results))["cases"]
    if args.walked_only:
        cases = [c for c in cases if not c["start_stops"]]
    cases = cases[:args.n]
    if not cases:
        print("nothing to plot")
        return 1

    columns = min(3, len(cases))
    rows = (len(cases) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.0 * rows),
                             squeeze=False)
    for index, case in enumerate(cases):
        draw(axes[index // columns][index % columns], case)
    for index in range(len(cases), rows * columns):
        axes[index // columns][index % columns].axis("off")
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle("evidence (solid) vs random (dashed) -- x-z floor plane, metres."
                 "  orange square = shared start,  x = anchor,"
                 "  filled = instructed instance,  hollow = landmark,"
                 "  ring = stopped (green right / red wrong)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"{len(cases)} cases -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
