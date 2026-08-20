"""
plot_viewdist.py -- the viewpoint distribution per case, and where it says to go.

Offline: reads a `probe_sideview.py` sweep and a `probe_viewpoint.py` ladder,
runs no model, and grades the heading the distribution picks against what
actually happened when the robot walked to each angle.

Per panel: the sweep's own samples, the smoothed curve, the argmax the policy
would walk to, and -- as ticks along the bottom -- the measured outcome at every
angle the ladder tried.

POSITIVE AZIMUTH MOVES THE CAMERA LEFT.  Verified against
`nvs_lemniscate.camera_for`: at az +30 the camera lands at -x, and with the robot
facing +z that is its left hand.  So the x axis is drawn INVERTED, and the left
of every panel is the robot's left.  This costs nothing and it is the difference
between reading the figure and mis-reading it.

    python viz/plot_viewdist.py --sweep nvs_pilot/probe_sideview4.json
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
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from probe_viewdist import ANGLES

#: Azimuth smoothing width, degrees.  Flat between 6 and 20 on `cases_hard`.
SIGMA = 20.0


def weight(view) -> float:
    """The rule under test: how many of the ten subject candidates it sees."""
    return float(view["subj_ok"])


def smooth(views, at, sigma: float = SIGMA) -> np.ndarray:
    az = np.array([v["azimuth"] for v in views], float)
    w = np.array([weight(v) for v in views], float)
    k = np.exp(-0.5 * ((np.asarray(at, float)[:, None] - az[None, :])
                       / sigma) ** 2)
    return (k @ w) / np.maximum(k.sum(1), 1e-9)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--sweep", default="nvs_pilot/probe_sideview4.json")
    ap.add_argument("--ladder", default="nvs_pilot/probe_viewpoint_ladder.json")
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--out", default="nvs_pilot/viewgrid/viewdist.png")
    args = ap.parse_args(argv)

    sv = {r["scene"]: r for r in json.load(open(args.sweep))["cases"]}
    lad = json.load(open(args.ladder))
    walked = {r["scene"]: {float(a): v for a, v in r["ranks"].items()}
              for r in lad["cases"]}
    # THE LADDER SETS THE CASE LIST, not the sweep: a case the sweep could not
    # read is a failure of the method, and taking the sweep's keys would drop it.
    scenes = [r["scene"] for r in lad["cases"]]

    def oracle_side(scene: str) -> Optional[str]:
        left = any(v == 1 for a, v in walked[scene].items() if a > 0)
        right = any(v == 1 for a, v in walked[scene].items() if a < 0)
        return "left" if left and not right else \
               "right" if right and not left else None

    grid = np.linspace(min(ANGLES), max(ANGLES), 121)
    cols = 5
    rows = int(np.ceil(len(scenes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.2 * rows),
                             squeeze=False, sharex=True)
    good = 0
    for index, scene in enumerate(scenes):
        ax = axes[index // cols][index % cols]
        side = oracle_side(scene)
        if side:
            ax.axvspan(0 if side == "left" else -32,
                       32 if side == "left" else 0,
                       color="#8fd18f", alpha=0.16, zorder=0)
        row = sv.get(scene)
        if row is None:
            ax.text(0.5, 0.5, "no sweep", ha="center", transform=ax.transAxes)
            continue
        ax.plot(grid, smooth(row["views"], grid, args.sigma), color="#1f4e9c",
                lw=1.8, zorder=3)
        ax.scatter([v["azimuth"] for v in row["views"]],
                   [weight(v) for v in row["views"]], s=14, color="#555555",
                   zorder=4)
        pick = ANGLES[int(np.argmax(smooth(row["views"], ANGLES, args.sigma)))]
        hit = walked[scene].get(pick) == 1
        good += hit
        ax.axvline(pick, color="#e08a00", lw=1.8, ls="--", zorder=5)
        for angle in ANGLES:
            got = walked[scene].get(angle)
            ax.plot(angle, -0.9,
                    marker="o" if got == 1 else ("x" if isinstance(got, int)
                                                 else "s" if got == "unreachable"
                                                 else "."),
                    ms=6 if got == 1 else 4.5, mew=1.5, clip_on=False, zorder=6,
                    color="#2e8b3d" if got == 1
                    else "#c23b22" if isinstance(got, int)
                    else "#7a5cc0" if got == "unreachable" else "#aaaaaa")
        ax.set_title(f"{scene}  oracle {side}  pick {pick:+.0f} "
                     f"{'HIT' if hit else 'miss'}", fontsize=8,
                     color="#2e8b3d" if hit else "#c23b22")
        ax.set_ylim(-1.6, 10.4)
        ax.set_yticks([0, 5, 10])
        ax.set_xlim(32, -32)                       # inverted: see the docstring
        ax.set_xticks([30, 20, 10, 0, -10, -20, -30])
        ax.tick_params(labelsize=6)
    for k in range(len(scenes), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#555555",
               label="one sweep view: how many of its 10 candidates it sees"),
        Line2D([], [], color="#1f4e9c", lw=2,
               label=f"smoothed over azimuth (sigma {args.sigma:.0f})"),
        Line2D([], [], color="#e08a00", lw=2, ls="--",
               label="argmax = the heading it would walk"),
        Line2D([], [], color="#8fd18f", lw=8, alpha=0.4,
               label="oracle side (where walking works)"),
        Line2D([], [], marker="o", ls="", color="#2e8b3d",
               label="walked there: top-1 correct"),
        Line2D([], [], marker="x", ls="", color="#c23b22",
               label="walked there: wrong rank"),
        Line2D([], [], marker=".", ls="", color="#aaaaaa",
               label="walked there: no rank at all"),
    ], loc="upper center", ncol=3, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, 1.005))
    fig.supxlabel("sweep azimuth (deg)      <-- camera moves LEFT        "
                  "camera moves RIGHT -->", fontsize=10)
    fig.suptitle(f"Viewpoint distribution from the correspondence count  "
                 f"-- picks a working heading in {good}/{len(scenes)} cases",
                 fontsize=13, y=1.021)
    fig.tight_layout(rect=(0, 0.01, 1, 0.985))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=95, bbox_inches="tight")
    print(f"  {good}/{len(scenes)}   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
