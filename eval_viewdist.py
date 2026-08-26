"""
eval_viewdist.py -- grade the viewpoint distribution against walked outcomes.

`probe_viewdist.py` builds the distribution and records where it points.  This
file reads that JSON plus `probe_viewpoint.py`'s measured "walk to azimuth A,
was top-1 correct" table and answers the only question that matters: is the
heading it chooses better than the alternatives?

    stand still        the reference frame alone.  What motion has to beat.
    fixed angle        always turn the same way.  Beats chance whenever the
                       case list is lopsided, so it is the real bar, not chance.
    random angle       chance, averaged over the candidate headings.
    distribution       argmax of the smoothed agreement.
    ceiling            some candidate heading works.  No policy exceeds this.

THREE RULES THIS FILE OBEYS, each because breaking one produced a wrong result
that survived several rounds of analysis:

 1. THE DENOMINATOR IS EVERY CASE.  A case the method cannot speak about is a
    failure, not an exclusion.  Reporting 27/38 instead of 27/40 quietly turned
    two systematic failures -- the two scenes where no view corresponds at all
    -- into missing data.
 2. NO OUTCOME-DEPENDENT FALLBACK.  "If that angle gives no rank, try the next"
    reads the answer: the robot cannot know its own top-1 is wrong.  A refused
    POSE is different and may be retried, because THOR refusing a teleport is
    something the robot observes -- that retry lives in `eval_move.walk`.
 3. `unreachable` AND `no rank` ARE NOT THE SAME VALUE.  Conflating them made a
    policy that skipped both look like it was avoiding bad geometry when it was
    reading the answer, and made every "unreachable" count uninterpretable.
    Older probe_viewpoint JSONs wrote both as null; those files cannot separate
    the two and are read here as plain failures.

    python eval_viewdist.py --dist results/probe_viewdist.json \
                            --walked results/probe_viewpoint_ladder.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def walked_table(path: str) -> Dict[str, Dict[float, Any]]:
    """scene -> {azimuth: 1 | other rank | None | "unreachable"}."""
    got = json.load(open(path))
    return {r["scene"]: {float(a): v for a, v in r["ranks"].items()}
            for r in got["cases"]}


def hit(walked: Dict[float, Any], azimuth: float) -> bool:
    """Did walking to `azimuth` put the instructed triplet at rank 1?

    Anything else -- a worse rank, no rank at all, a refused pose -- is False.
    Rule 2: this function never says "try somewhere else".
    """
    return walked.get(azimuth) == 1


def oracle_side(walked: Dict[float, Any]) -> Optional[str]:
    """`left` / `right` when exactly one side reaches rank 1, else None."""
    left = any(v == 1 for a, v in walked.items() if a < 0)
    right = any(v == 1 for a, v in walked.items() if a > 0)
    return "left" if left and not right else "right" if right and not left \
        else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dist", default="results/probe_viewdist.json")
    ap.add_argument("--walked", default="results/probe_viewpoint_ladder.json")
    ap.add_argument("--cases", default="datasets/robot/cases_hard.json",
                    help="the FULL list -- rule 1, the denominator is every "
                         "case, including ones the probe could not read")
    ap.add_argument("--score", default="rel", choices=("rel", "rel_s"))
    ap.add_argument("--plot", default=None, metavar="DIR",
                    help="also draw the per-case and summary figures")
    args = ap.parse_args(argv)

    dist = json.load(open(args.dist))
    rows = {r["scene"]: r for r in dist["cases"]}
    angles = [float(a) for a in dist["angles"]]
    sigmas = [float(s) for s in dist["sigmas"]]
    walked = walked_table(args.walked)
    scenes = [c["scene"] for c in json.load(open(args.cases))["cases"]
              if c["scene"] in walked]
    n = len(scenes)
    missing = [s for s in scenes if s not in rows]

    print(f"  {n} cases ({args.cases}), score = {args.score}")
    if missing:
        print(f"  {len(missing)} the probe could not read, counted as failures: "
              f"{', '.join(missing)}")
    print(f"\n  {'policy':34} {'top-1 correct':>14}")
    print(f"  {'stand still (do not move)':34} "
          f"{sum(1 for s in scenes if hit(walked[s], 0.0)):>8}/{n}")
    for a in angles:
        print(f"  {f'fixed heading {a:+.0f}':34} "
              f"{sum(1 for s in scenes if hit(walked[s], a)):>8}/{n}")
    rng = np.random.default_rng(0)
    rand = float(np.mean([[hit(walked[s], rng.choice(angles)) for s in scenes]
                          for _ in range(400)])) * n
    print(f"  {'random heading (chance)':34} {rand:>8.1f}/{n}")
    for sg in sigmas:
        got = 0
        for s in scenes:
            row = rows.get(s)
            if row is None:
                continue                      # rule 1: counts as a failure
            pick = row[args.score]["picks"].get(f"{sg:.0f}")
            got += pick is not None and hit(walked[s], pick)
        print(f"  {f'distribution, sigma {sg:.0f}':34} {got:>8}/{n}")
    print(f"  {'ceiling (some heading works)':34} "
          f"{sum(1 for s in scenes if any(hit(walked[s], a) for a in angles)):>8}/{n}")

    # Does the chosen heading at least land on the right SIDE?
    print(f"\n  {'sigma':>6} {'side agrees with oracle':>24}")
    for sg in sigmas:
        ok = said = 0
        for s in scenes:
            row, side = rows.get(s), oracle_side(walked[s])
            if row is None or side is None:
                continue
            pick = row[args.score]["picks"].get(f"{sg:.0f}")
            if pick is None:
                continue
            said += 1
            ok += ("left" if pick < 0 else "right") == side
        print(f"  {sg:>6.0f} {ok:>17}/{said:<4}")
    sides = [oracle_side(walked[s]) for s in scenes]
    print(f"  (oracle is one-sided in {sum(1 for x in sides if x)}/{n} cases: "
          f"{sides.count('left')} left, {sides.count('right')} right, so "
          f"always guessing the commoner side scores "
          f"{max(sides.count('left'), sides.count('right'))}/{n})")

    # The aligned mean: the cleanest test of whether ANY side signal exists.
    grid = np.linspace(min(angles), max(angles), 121)
    from probe_viewdist import build_distribution, build_vote
    stack = []
    for s in scenes:
        row, side = rows.get(s), oracle_side(walked[s])
        if row is None or side is None:
            continue
        phat = row[args.score]["phat"]
        w = build_distribution(row["views"], phat, args.score, 14.0, grid)
        stack.append(w[::-1] if side == "left" else w)
    if stack:
        A = np.array(stack)
        m = A.mean(0)
        print(f"\n  aligned mean over {len(A)} cases (sigma 14, oracle side "
              f"flipped positive)")
        print(f"    oracle side {m[grid > 0].mean():.3f}   "
              f"other side {m[grid < 0].mean():.3f}   "
              f"difference {m[grid > 0].mean() - m[grid < 0].mean():+.3f}")
        print(f"    A difference near zero means the distribution carries no "
              f"side information at all.")

    if args.plot:
        plot(rows, walked, scenes, angles, args, grid)
    return 0


def plot(rows, walked, scenes, angles, args, grid) -> None:
    """Per-case panels plus the aligned mean and the policy bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from probe_viewdist import build_distribution

    os.makedirs(args.plot, exist_ok=True)
    cols = 5
    rows_n = int(np.ceil(len(scenes) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.3 * rows_n),
                             squeeze=False, sharex=True)
    lo, hi = min(angles) - 3, max(angles) + 3
    stack = []
    for index, scene in enumerate(scenes):
        ax = axes[index // cols][index % cols]
        side = oracle_side(walked[scene])
        if side:
            ax.axvspan(lo if side == "left" else 0, 0 if side == "left" else hi,
                       color="#8fd18f", alpha=0.16, zorder=0)
        row = rows.get(scene)
        if row is None:
            ax.text(0.5, 0.55, "probe could not read\n(counts as a failure)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#b03030")
            ax.set_title(f"{scene}   oracle {side}", fontsize=9,
                         color="#b03030")
        else:
            views, phat = row["views"], row[args.score]["phat"]
            w = build_distribution(views, phat, args.score, 14.0, grid)
            ax.plot(grid, w, color="#1f4e9c", lw=2.0, zorder=3)
            ax.fill_between(grid, 0, w, color="#1f4e9c", alpha=0.12, zorder=2)
            for v in views:
                g = v[args.score]["top1_group"]
                agree = (phat is not None and g is not None
                         and tuple(g) == tuple(phat))
                ax.plot([v["azimuth"], v["azimuth"]], [0, 1.0 if agree else 0],
                        color="#bbbbbb", lw=0.8, zorder=1)
                ax.plot(v["azimuth"], 1.0 if agree else 0.0, marker="o",
                        ms=5.5, zorder=4,
                        color="#2e8b3d" if v[args.score].get("hit")
                        else "#c23b22")
            pick = row[args.score]["picks"].get("14")
            if pick is not None:
                ax.axvline(pick, color="#e08a00", lw=1.8, ls="--", zorder=5)
            good = pick is not None and hit(walked[scene], pick)
            ax.set_title(f"{scene}   oracle {side}   pick "
                         f"{pick:+.0f}   -> {'HIT' if good else 'miss'}",
                         fontsize=9, color="#2e8b3d" if good else "#c23b22")
            if side:
                stack.append(w[::-1] if side == "left" else w)
        for a in list(angles) + [0.0]:
            v = walked[scene].get(a)
            ax.plot(a, -0.13,
                    marker="o" if v == 1 else ("x" if isinstance(v, int)
                                               else "s" if v == "unreachable"
                                               else "."),
                    ms=7 if v == 1 else 5, mew=1.6, clip_on=False, zorder=6,
                    color="#2e8b3d" if v == 1
                    else "#c23b22" if isinstance(v, int)
                    else "#7a5cc0" if v == "unreachable" else "#999999")
        ax.set_ylim(-0.2, 1.12)
        ax.set_xlim(lo, hi)
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="#dddddd", lw=0.8)
    for k in range(len(scenes), rows_n * cols):
        axes[k // cols][k % cols].axis("off")
    handles = [
        Line2D([], [], marker="o", ls="", color="#2e8b3d",
               label="view whose own top-1 is truly correct"),
        Line2D([], [], marker="o", ls="", color="#c23b22",
               label="view whose top-1 is wrong (or named nothing)"),
        Line2D([], [], color="#1f4e9c", lw=2, label="smoothed agreement, sigma 14"),
        Line2D([], [], color="#e08a00", lw=2, ls="--", label="argmax = chosen heading"),
        Line2D([], [], color="#8fd18f", lw=8, alpha=0.4, label="oracle side"),
        Line2D([], [], marker="o", ls="", color="#2e8b3d",
               label="ticks: walked there, top-1 correct"),
        Line2D([], [], marker="x", ls="", color="#c23b22",
               label="ticks: walked there, wrong rank"),
        Line2D([], [], marker="s", ls="", color="#7a5cc0",
               label="ticks: pose refused"),
        Line2D([], [], marker=".", ls="", color="#999999",
               label="ticks: no rank at all"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 1.004))
    fig.suptitle(f"Viewpoint distribution per case  ({args.score})",
                 fontsize=15, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = os.path.join(args.plot, f"viewdist_cases_{args.score}.png")
    fig.savefig(out, dpi=95, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    if stack:
        A = np.array(stack)
        m, se = A.mean(0), A.std(0) / np.sqrt(len(A))
        a1.axvspan(0, grid.max(), color="#8fd18f", alpha=0.18,
                   label="oracle side")
        a1.plot(grid, m, color="#1f4e9c", lw=2.4)
        a1.fill_between(grid, m - se, m + se, color="#1f4e9c", alpha=0.22)
        a1.axvline(0, color="#888888", lw=1)
        a1.set_title(f"Aligned mean over {len(A)} cases\noracle side "
                     f"{m[grid > 0].mean():.3f}  vs  other side "
                     f"{m[grid < 0].mean():.3f}", fontsize=10)
        a1.legend(fontsize=9, frameon=False)
    a1.set_xlabel("azimuth, flipped so the oracle side is positive (deg)")
    a1.set_ylabel("mean agreement with p_hat")

    n = len(scenes)
    names = ["stand\nstill"]
    vals = [sum(1 for s in scenes if hit(walked[s], 0.0))]
    for a in angles:
        names.append(f"fixed\n{a:+.0f}")
        vals.append(sum(1 for s in scenes if hit(walked[s], a)))
    rng = np.random.default_rng(0)
    names.append("random")
    vals.append(float(np.mean([[hit(walked[s], rng.choice(angles))
                                for s in scenes] for _ in range(400)])) * n)
    for sg in (6.0, 14.0, 20.0):
        got = 0
        for s in scenes:
            row = rows.get(s)
            if row is None:
                continue
            pick = row[args.score]["picks"].get(f"{sg:.0f}")
            got += pick is not None and hit(walked[s], pick)
        names.append(f"dist\nsigma {sg:.0f}")
        vals.append(got)
    names.append("ceiling")
    vals.append(sum(1 for s in scenes if any(hit(walked[s], a) for a in angles)))
    colour = (["#999999"] * (len(angles) + 2) + ["#1f4e9c"] * 3 + ["#2e8b3d"])
    a2.bar(range(len(vals)), vals, color=colour)
    for i, v in enumerate(vals):
        a2.text(i, v + 0.4, f"{v:.0f}", ha="center", fontsize=8)
    a2.set_xticks(range(len(names)))
    a2.set_xticklabels(names, fontsize=7)
    a2.set_ylabel(f"cases with top-1 correct (out of {n})")
    a2.set_title("Walking there: what each policy gets\n"
                 "no outcome-dependent fallback; every case in the denominator",
                 fontsize=9)
    fig.tight_layout()
    out = os.path.join(args.plot, f"viewdist_summary_{args.score}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
