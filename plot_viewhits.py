"""
plot_viewhits.py -- which sweep views got the triplet right, case by case.

One row per case, one marker per view, placed at the view's azimuth.  FILLED
GREEN means that view's own top-1 pair is the instructed triplet; HOLLOW RED
means it is not.  The band behind each row is the side the ladder says a walk
works from.

This is the per-view picture behind the aggregate: 37% of views are right, but
60% on the working side against 16% on the other.  The point of drawing it is to
see whether that split is a clean edge, a gradient, or a few cases carrying it.

The marker uses GROUND TRUTH and is a diagnostic, never an input to a policy.

POSITIVE AZIMUTH MOVES THE CAMERA LEFT, so the x axis is inverted -- see
`plot_viewdist.py`.

    python plot_viewhits.py --sweep nvs_pilot/probe_sideview4.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--sweep", default="nvs_pilot/probe_sideview4.json")
    ap.add_argument("--ladder", default="nvs_pilot/probe_viewpoint_ladder.json")
    ap.add_argument("--out", default="nvs_pilot/viewgrid/viewhits.png")
    args = ap.parse_args(argv)

    sweep = json.load(open(args.sweep))["cases"]
    walked = {r["scene"]: {float(a): v for a, v in r["ranks"].items()}
              for r in json.load(open(args.ladder))["cases"]}

    def oracle_side(scene: str) -> Optional[str]:
        got = walked.get(scene, {})
        left = any(v == 1 for a, v in got.items() if a > 0)
        right = any(v == 1 for a, v in got.items() if a < 0)
        return "left" if left and not right else \
               "right" if right and not left else None

    fig, ax = plt.subplots(figsize=(11, 0.34 * len(sweep) + 2.4))
    for y, row in enumerate(sweep):
        side = oracle_side(row["scene"])
        if side:
            ax.barh(y, 32 if side == "left" else -32, left=0, height=0.82,
                    color="#8fd18f", alpha=0.20, zorder=0)
        hits = 0
        for view in row["views"]:
            got = view.get("hit")
            hits += bool(got)
            # The two elevations share an azimuth; nudge them apart vertically
            # so one does not hide the other.
            ax.scatter(view["azimuth"], y + (0.16 if view["v"] % 2 else -0.16),
                       s=34, zorder=3,
                       marker="o" if got else "o",
                       facecolor="#2e8b3d" if got else "none",
                       edgecolor="#2e8b3d" if got else "#c23b22",
                       linewidths=1.2)
        ax.text(34, y, f"{hits}/{len(row['views'])}", fontsize=7,
                va="center", ha="left",
                color="#2e8b3d" if hits else "#c23b22")
    ax.set_yticks(range(len(sweep)))
    ax.set_yticklabels([r["scene"] for r in sweep], fontsize=7)
    ax.set_ylim(-0.8, len(sweep) - 0.2)
    ax.invert_yaxis()
    ax.set_xlim(38, -34)                      # inverted: +az is the robot's left
    ax.set_xticks([30, 20, 10, 0, -10, -20, -30])
    ax.set_xlabel("sweep azimuth (deg)      <-- camera moves LEFT        "
                  "camera moves RIGHT -->")
    ax.axvline(0, color="#666666", lw=0.8, ls=":")
    ax.grid(axis="x", alpha=0.2)
    allv = [v for r in sweep for v in r["views"]]
    hit = sum(1 for v in allv if v.get("hit"))
    ax.set_title(f"Which views got the triplet right  --  "
                 f"{hit}/{len(allv)} views ({hit / len(allv):.0%}) over "
                 f"{len(sweep)} cases", fontsize=12)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor="#2e8b3d",
               markeredgecolor="#2e8b3d", label="this view's top-1 IS the triplet"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor="#c23b22", label="it is not"),
        Line2D([], [], color="#8fd18f", lw=8, alpha=0.5,
               label="side the ladder says a walk works from"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.02 - 2.0 / len(sweep)),
        ncol=3, fontsize=9, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"  {hit}/{len(allv)} views correct   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
