"""
eval_grounding.py -- how well `grounding.single_frame` grounds an instruction,
on 4500 relations instead of 40.

THE ROBOT'S GRADER, ON THE SGG DATASET.  `eval_move` reports this metric on 40
staged cases per list, which is enough to compare two viewpoint rules and not
enough to say where the GROUNDING itself fails.  `datasets/sgg/occlusion_ds4`
holds 900 rendered views and 7235 annotated relations, every endpoint carrying
its own occlusion fraction -- so the same metric, run here, is stratified by the
one variable the method exists to address.

NEITHER `scoring/` NOR THE ROBOT PIPELINE, which is why it is in here.
`scoring/eval_occlusion.py` is a cross-repo library measuring FREE-FORM R@K --
`../sgg_nvs` imports it by bare name and its module names are a contract.  This
asks a different question of the same images and the same occlusion bands:

    eval_occlusion    of every relation the model predicted, how many GT
                      relations are in the top K?
    eval_grounding    given the instruction "bowl on counter", is the model's
                      FIRST pair the annotated bowl and the annotated counter?

The bands and the box matcher are imported from there rather than restated, so
the two tables can be read against each other.

THE CHAIN IS SPLIT, because `top1` alone cannot say what to fix:

    listed     a box with IoU >= `--iou` on the annotated subject is in the
               SUBJECT shortlist, and likewise the object.  Below this line no
               ranking can recover the case.
    pairable   both, so the correct pair EXISTS to be ranked.
    rank       where it lands once scored.
    top1       rank == 1, which is what `eval_move` reports.

ONLY UNAMBIGUOUS INSTRUCTIONS.  2707 of the 7235 relations share their
(class, predicate, class) triple with another instance in the same view -- two
bowls on two counters -- and "the bowl on the counter" does not say which.
Grading an instance against an ambiguous instruction measures nothing; this is
`robot/task_find.py`'s rule and it is applied here too.

    python analysis/eval_grounding.py
    python analysis/eval_grounding.py --grid --out nvs_pilot/grounding_ds4.json
"""

from __future__ import annotations

# `python analysis/<script>.py` puts analysis/ on sys.path, not the repo root, so
# the top-level modules would not import.  Same bootstrap as gen/.
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)


import argparse
import collections
import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from robot.grounding import WEIGHTS
from scoring.eval_occlusion import BANDS, band_of, best_iou

#: The settings `--grid` sweeps -- 24 of them, and they cost one re-ranking
#: each because the detector runs once per VIEW and every setting reads the same
#: output.
#:
#:   weight      what the ranking multiplies in.  `s` and `class` are what
#:               `eval_move` and `move_once` respectively run today; `both` is
#:               their product and has never been measured.
#:   pair_iou    reject a pair whose boxes overlap this much -- one object
#:               related to itself.
#:   condition   how deep the shortlist goes, which SETS THE CEILING: a
#:               relation whose box is not listed cannot be ranked first by any
#:               scoring rule.
#:   connectivity  whether the score keeps EGTR's p(these two are related at
#:                  all).  It is the same number for all 50 predicates, so it
#:                  cannot answer an instruction, and it is the larger factor by
#:                  orders of magnitude.
GRID = [{"weight": w, "pair_iou": p, "condition": c, "connectivity": k}
        for w in ("s", "class", "both")
        for p in (0.0, 0.15)
        for c in (5, 10, 20, 40)
        for k in ("on", "off")]


def instructions(scene: Dict[str, Any], view: Dict[str, Any], image: str
                 ) -> List[Dict[str, Any]]:
    """The relations in one view that an instruction can name UNAMBIGUOUSLY.

    A relation is dropped when another instance pair in the same view has the
    same three words.  See the module docstring.
    """
    entry = {o["name"]: o for o in view["objects"]}
    triple = collections.Counter()
    for r in view["relations"]:
        s, o = entry.get(r["subject_name"]), entry.get(r["object_name"])
        if s and o:
            triple[(s["vg150_class"], r["predicate"], o["vg150_class"])] += 1

    out = []
    for r in view["relations"]:
        s, o = entry.get(r["subject_name"]), entry.get(r["object_name"])
        if not s or not o:
            continue
        words = (s["vg150_class"], r["predicate"], o["vg150_class"])
        if triple[words] != 1:
            continue
        out.append({"task": {"subject_class": words[0], "predicate": words[1],
                             "object_class": words[2]},
                    "subject": s, "object": o, "image": image,
                    "occlusion": max(r["subject_occlusion"],
                                     r["object_occlusion"]),
                    "scene": scene, "view": view["index"]})
    return out


def links(shot: Dict[str, Any], ask: Dict[str, Any], egtr,
          setting: Dict[str, Any], iou_hit: float) -> Dict[str, Any]:
    """The chain, for one instruction on one frame."""
    from robot.grounding import single_frame

    rel = shot["rel" if setting.get("connectivity", "on") == "on"
              else "rel_predicate"]
    read = single_frame(shot["probs"], shot["boxes"], rel, egtr,
                        ask["task"], width=setting["condition"],
                        weight=setting["weight"],
                        pair_iou=setting["pair_iou"])
    task = ask["task"]
    base = {"instruction": f"{task['subject_class']} {task['predicate']} "
                           f"{task['object_class']}",
            "image": ask["image"],
            "gt_subject": ask["subject"]["bbox_visible"]
                          or ask["subject"]["bbox_amodal"],
            "gt_object": ask["object"]["bbox_visible"]
                         or ask["object"]["bbox_amodal"],
            "top_subject": None, "top_object": None,
            "predicate": ask["task"]["predicate"], "occlusion": ask["occlusion"],
            "listed_subject": False, "listed_object": False, "pairable": False,
            "rank": None, "top1": False, "winner_on_subject": False,
            "p_subject": 0.0, "p_object": 0.0}
    if read is None or not read["order"]:
        # VG150 cannot spell one of the classes, or no pair survived the filter.
        return {**base, "unspellable": read is None}
    boxes, (subs, objs) = shot["boxes"], read["candidates"]

    def on(q, entry) -> bool:
        return best_iou(boxes[int(q)].tolist(), entry) >= iou_hit

    good_s = {int(q) for q in subs if on(q, ask["subject"])}
    good_o = {int(q) for q in objs if on(q, ask["object"])}
    rank = next((r for r, (_, i, j) in enumerate(read["order"], 1)
                 if int(i) in good_s and int(j) in good_o), None)
    # THE SHORTLIST'S OWN CONFIDENCE, which costs nothing and the robot can read
    # at run time: the best p(the instruction's word) any listed query reaches.
    subject_p, object_p = read["p"]
    base["p_subject"] = max(float(subject_p[int(q)]) for q in subs)
    base["p_object"] = max(float(object_p[int(q)]) for q in objs)
    top = read["order"][0]
    return {**base, "listed_subject": bool(good_s),
            "listed_object": bool(good_o), "pairable": bool(good_s and good_o),
            "rank": rank, "top1": rank == 1, "unspellable": False,
            "top_subject": [round(v, 1) for v in boxes[top[1]].tolist()],
            "top_object": [round(v, 1) for v in boxes[top[2]].tolist()],
            # WHICH END WAS WRONG.  A top-1 sitting on the right subject and the
            # wrong landmark is a different failure from one on neither.
            "winner_on_subject": on(read["order"][0][1], ask["subject"])}


def table(rows: Sequence[Dict[str, Any]], key, order: Sequence,
          title: str) -> None:
    """One block: the chain, grouped by `key`."""
    print(f"\n  {title}")
    print(f"    {'':14}{'n':>6}{'listed':>9}{'pairable':>10}{'top1':>8}"
          f"{'top3':>7}{'top10':>7}{'med.rank':>10}")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    for name in list(order) + [k for k in groups if k not in order]:
        got = groups.get(name)
        if not got:
            continue
        n = len(got)
        ranked = [r["rank"] for r in got if r["rank"]]
        pct = lambda c: f"{100 * c / n:5.1f}%"                    # noqa: E731
        print(f"    {str(name):14}{n:>6}"
              f"{pct(sum(1 for r in got if r['listed_subject'] and r['listed_object'])):>9}"
              f"{pct(sum(1 for r in got if r['pairable'])):>10}"
              f"{pct(sum(1 for r in got if r['top1'])):>8}"
              f"{pct(sum(1 for r in ranked if r <= 3)):>7}"
              f"{pct(sum(1 for r in ranked if r <= 10)):>7}"
              f"{(int(np.median(ranked)) if ranked else 0):>10}")


#: p(class) bins for `confidence`.  Log-ish, because the interesting range is
#: the bottom: a shortlist whose best candidate reads 0.01 is a different object
#: from one reading 0.5, and the gap between 0.5 and 0.9 is not.
P_BINS = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.10), (0.10, 0.30), (0.30, 1.01)]


def confidence(rows: Sequence[Dict[str, Any]]) -> None:
    """IS THE SHORTLIST'S OWN p(class) A SIGNAL THAT IT HOLDS THE RIGHT BOX?

    A robot cannot read `pairable` -- that needs the answer.  It CAN read the
    best p(the instruction's word) among the queries it shortlisted, before
    ranking anything.  If that number separates a list holding the target from
    one that does not, it is both an adaptive cut on K and the run-time signal
    for "I cannot ground this from here, move".
    """
    print("\n  by the shortlist's own best p(class) -- the weaker of the two "
          "ends")
    print(f"    {'p(class)':14}{'n':>6}{'listed':>9}{'pairable':>10}"
          f"{'top1':>8}{'top1|pairable':>15}")
    worst = [(min(r["p_subject"], r["p_object"]), r) for r in rows]
    for low, high in P_BINS:
        got = [r for v, r in worst if low <= v < high]
        if not got:
            continue
        n = len(got)
        pairable = sum(1 for r in got if r["pairable"])
        top1 = sum(1 for r in got if r["top1"])
        print(f"    {low:.2f}-{high:.2f}   {n:>6}"
              f"{100 * sum(1 for r in got if r['listed_subject'] and r['listed_object']) / n:>8.1f}%"
              f"{100 * pairable / n:>9.1f}%{100 * top1 / n:>7.1f}%"
              f"{(100 * top1 / pairable if pairable else 0):>14.1f}%")
    # WHAT AN ABSTAIN RULE WOULD BUY.  Refusing below a cut keeps the cases the
    # shortlist is confident about; the question is how much of the ANSWER goes
    # with the refusals.
    print("\n  refusing to answer below a cut on that number:")
    print(f"    {'cut':>6}{'answered':>11}{'top1 of those':>15}"
          f"{'correct kept':>14}")
    total_top1 = sum(1 for r in rows if r["top1"])
    for cut in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20):
        kept = [r for v, r in worst if v >= cut]
        if not kept:
            continue
        hit = sum(1 for r in kept if r["top1"])
        print(f"    {cut:>6.2f}{100 * len(kept) / len(rows):>10.1f}%"
              f"{100 * hit / len(kept):>14.1f}%"
              f"{100 * hit / max(total_top1, 1):>13.1f}%")


def draw(rows: Sequence[Dict[str, Any]], args) -> None:
    """The three sheets.  Evenly spaced through each group, not the first N,
    which would all come from one scene."""
    import numpy as _np
    from PIL import Image

    from viz import grounding as figures

    os.makedirs(args.figures, exist_ok=True)
    groups = (("correct", [r for r in rows if r["top1"]]),
              ("ranked", [r for r in rows if r["pairable"] and not r["top1"]]),
              ("missed", [r for r in rows if not r["pairable"]]))
    for name, got in groups:
        if not got:
            continue
        step = max(1, len(got) // args.examples)
        picked = got[::step][:args.examples]
        for r in picked:
            r["frame"] = _np.asarray(Image.open(r["image"]).convert("RGB"))
        figures.sheet(picked, os.path.join(args.figures, f"{name}.png"),
                      f"{name}  ({len(got)} of {len(rows)})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", default="datasets/sgg/occlusion_ds4")
    ap.add_argument("--n", type=int, default=0,
                    help="0 = every view; otherwise stop after N VIEWS")
    ap.add_argument("--grid", action="store_true",
                    help="sweep --weight, --pair-iou and --condition")
    ap.add_argument("--weight", choices=WEIGHTS, default="class")
    ap.add_argument("--connectivity", choices=("on", "off"), default="on",
                    help="EGTR predicts p(this predicate) and p(these two are "
                         "related at all) separately, and its own ranking "
                         "multiplies them.  Only the first answers an "
                         "INSTRUCTION -- the second is the same number for all "
                         "50 predicates, and it is the larger of the two by "
                         "orders of magnitude.  `on` is what every number in "
                         "this repo was measured with; `off` drops it, so the "
                         "score is one probability per word of the sentence.")
    ap.add_argument("--pair-iou", type=float, default=0.0)
    ap.add_argument("--condition", type=int, default=10)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--figures", metavar="DIR", default=None,
                    help="write correct.png / ranked.png / missed.png here -- "
                         "`--examples` cases of each.  Single setting only.")
    ap.add_argument("--examples", type=int, default=12, metavar="N")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from PIL import Image

    from robot.sgg_live import load_egtr, raw_predict

    scenes = sorted(glob.glob(os.path.join(args.root, "*", "scene.json")))
    if not scenes:
        raise SystemExit(f"no scene.json under {args.root}")
    egtr = load_egtr()

    settings = GRID if args.grid else [{"weight": args.weight,
                                        "pair_iou": args.pair_iou,
                                        "condition": args.condition,
                                        "connectivity": args.connectivity}]

    # ONE VIEW AT A TIME, EVERY SETTING.  A view's relation field is
    # 200 x 200 x 50 float32 = 8 MB, so holding 900 of them to loop the settings
    # outside would want 7 GB.  The detector still runs once per view.
    scored: Dict[int, List[Dict[str, Any]]] = {k: [] for k in
                                               range(len(settings))}
    views, total = 0, 0
    for path in scenes:
        scene = json.load(open(path))
        for view in scene["views"]:
            if args.n and views >= args.n:
                break
            image = os.path.join(os.path.dirname(path), view["image"])
            asks = instructions(scene["scene"], view, image)
            views += 1
            if not asks:
                continue
            if not os.path.exists(image):
                continue
            frame = np.asarray(Image.open(image).convert("RGB"))
            raw = raw_predict(egtr, frame)
            # BOTH RELATION TENSORS, so `connectivity` is a grid axis rather
            # than a second run of the detector.  16 MB, and one view is held
            # at a time.
            shot = {"probs": raw["probs_softmax"].detach().cpu().float(),
                    "rel": raw["rel"].detach().cpu(),
                    "rel_predicate": raw["rel_predicate"].detach().cpu(),
                    "boxes": raw["boxes"].detach().cpu()}
            total += len(asks)
            for k, setting in enumerate(settings):
                for ask in asks:
                    scored[k].append(links(shot, ask, egtr, setting, args.iou))
            if views % 50 == 0:
                print(f"  {views} views, {total} instructions", flush=True)
        if args.n and views >= args.n:
            break

    print(f"\n{views} views, {total} unambiguous instructions.\n"
          f"IoU >= {args.iou} on BOTH endpoints, against the better of the "
          f"amodal and visible box.\n"
          f"connectivity {args.connectivity}.")

    results = []
    for k, setting in enumerate(settings):
        rows = scored[k]
        label = (f"weight={setting['weight']}  pair-iou={setting['pair_iou']:g}"
                 f"  K={setting['condition']}  "
                 f"connectivity={setting['connectivity']}")
        n = len(rows)
        # THE FULL BREAKDOWN ONLY WHEN THERE IS ONE SETTING TO READ.  24 blocks
        # of two tables is not a result, it is a wall; `--grid` gets one line
        # each and the summary at the end.
        if len(settings) == 1:
            print(f"\n{'=' * 78}\n  {label}")
            table(rows, lambda r: band_of(r["occlusion"]) or "0.75+",
                  [b[2] for b in BANDS], "by occlusion of the worse endpoint")
            table(rows, lambda r: r["predicate"],
                  ["on", "behind", "near", "above", "in front of", "under"],
                  "by predicate")
            confidence(rows)
            if args.figures:
                draw(rows, args)
        results.append({**setting, "n": n,
                        "top1": sum(1 for r in rows if r["top1"]),
                        "pairable": sum(1 for r in rows if r["pairable"]),
                        "listed_subject": sum(1 for r in rows
                                              if r["listed_subject"]),
                        "listed_object": sum(1 for r in rows
                                             if r["listed_object"]),
                        "unspellable": sum(1 for r in rows
                                           if r.get("unspellable"))})

    if len(results) > 1:
        print(f"\n{'=' * 78}\n  every setting, ranked by top-1:\n")
        print(f"    {'weight':>7}{'pair-iou':>10}{'K':>4}{'conn':>6}"
              f"{'top1':>9}{'ceiling':>10}{'lost to rank':>14}")
        for r in sorted(results, key=lambda r: -r["top1"]):
            print(f"    {r['weight']:>7}{r['pair_iou']:>10g}{r['condition']:>4}"
                  f"{r['connectivity']:>6}"
                  f"{100 * r['top1'] / r['n']:>8.1f}%"
                  f"{100 * r['pairable'] / r['n']:>9.1f}%"
                  f"{100 * (r['pairable'] - r['top1']) / r['n']:>13.1f}%")

    best = max(results, key=lambda r: r["top1"])
    n = best["n"]
    print(f"\n{'=' * 78}")
    print(f"  Best of {len(results)}: weight={best['weight']} "
          f"pair-iou={best['pair_iou']:g} K={best['condition']} "
          f"connectivity={best['connectivity']}")
    print(f"    top-1          {best['top1']:>6}/{n}  "
          f"({100 * best['top1'] / n:.1f}%)")
    print(f"    ceiling        {best['pairable']:>6}/{n}  "
          f"(the pair was in the shortlist at all)")
    print(f"    lost to rank   {best['pairable'] - best['top1']:>6}  "
          f"-- present, not first")
    print(f"    lost to list   {n - best['pairable']:>6}  "
          f"-- subject listed {100 * best['listed_subject'] / n:.0f}%, "
          f"object {100 * best['listed_object'] / n:.0f}%")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"root": args.root, "iou": args.iou, "views": views,
                   "settings": results}, open(args.out, "w"), indent=1)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
