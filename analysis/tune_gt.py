"""
tune_gt.py -- re-derive the geometric relations under candidate constraints and
score each variant against the predictions already on disk.

Measured on `occlusion_ds2`, three predicates carry 63% of the ground truth and
between them matched one relation out of 369:

    on      181 GT   58 matched   horizontal median 0.38 m    0% beyond 1.5 m
    above   172        0                            0.32 m   13%
    behind  164        1                            1.54 m   52%
    under    33        0                            0.43 m   15%

`on` is not a better predicate than the others; it is the one whose definition
already agreed with the annotation convention, because it comes from
`parentReceptacles` and a supported object is necessarily touching and aligned
with its support.  The geometric predicates were derived with a 3D truth check
and a 2D likelihood and nothing that requires the two objects to be spatially
RELATED, so a plant on a table qualifies as `above` a chair standing beside it.

Two constraints are trialled, one per failure mode:

    horizontal    cap the 3D horizontal separation.  Matched relations sit at a
                  median of 0.27 m and unmatched at 0.64 m, and only 3% of
                  matches exceed 1.5 m against 22% of misses, so distance alone
                  separates them.
    cover         require the subject's box to overlap the object's.  VG's own
                  `above` has a median `cover` of 0.484 -- annotators say "above"
                  about things standing over one another, not merely higher.

Nothing is re-rendered: the stored per-view boxes, positions and occlusion are
enough to rebuild the relation set, so a variant costs a second.

SUPERSEDED as a diagnosis, kept as a sweep tool.  Both constraints were measured
and neither helps: capping horizontal separation at 0.5 m raises `above`'s
pair-localisation rate only from 38% to 41% while discarding two thirds of the
relations, because EGTR was never proposing those pairs in the first place.  The
constraints treat the symptom.  The cause was that the 2D likelihood deciding
WHICH predicate a pair carries classifies VG's own labels at 55.3% against a
68.8% majority baseline -- see `vg_pair_prior.py` and the TASKS.md section "The
2D feature model cannot name a relation".  The stacking gate survives as
`vg_gt.VERTICAL_MAX_HORIZONTAL_M`, on correctness grounds rather than recall.
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
import copy
import glob
import itertools
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scoring.analyze_multiview import build_synonyms
from scoring.eval_occlusion import band_of, load, matched, ranked
from vg.vg_gt import features_2d, support_pairs, vg_calibrated_relations

BANDS = ["clear", "light", "moderate", "heavy"]

#: Families the constraints apply to.  `on`/`in` come from `parentReceptacles`
#: and are left alone -- they are asserted by the simulator, not inferred, and
#: they are the only ones currently working.
VERTICAL = {"above", "under"}
DEPTH = {"behind", "in front of"}


def horizontal(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.hypot(a["x"] - b["x"], a["z"] - b["z"])


def rebuild(view: Dict[str, Any], width: int, height: int,
            max_horizontal: Optional[float],
            min_cover: Optional[float]) -> List[Dict[str, Any]]:
    """
    The view's relations with the geometric ones re-derived under a constraint.

    Support relations are copied through untouched; only the `vg_gt` output is
    regenerated, then filtered.  Filtering after the fact rather than inside
    `vg_gt` keeps the trial honest about what it changes -- the family choice and
    the 10% annotation-rate cut still happen exactly as they did when the dataset
    was built.
    """
    support = [r for r in view["relations"] if r["annotation"] == "parentReceptacles"]
    entries = {o["name"]: o for o in view["objects"]}
    occlusion = {name: o["occlusion"] for name, o in entries.items()}

    shim = [{"object_id": o["name"], "vg150_class": o["vg150_class"],
             "bbox_xyxy": o["bbox_amodal"], "position": o["position"],
             "distance": o["distance"], "loose": False}
            for o in view["objects"] if o["vg150_class"]]
    exclude = support_pairs([{**r, "subject_id": r["subject_name"],
                              "object_id": r["object_name"]} for r in support])
    geometric = vg_calibrated_relations(shim, width, height, exclude)

    out = list(support)
    for rel in geometric:
        subject = entries.get(rel["subject_id"])
        obj = entries.get(rel["object_id"])
        if subject is None or obj is None:
            continue
        predicate = rel["predicate"]

        if max_horizontal is not None and predicate in (VERTICAL | DEPTH):
            if horizontal(subject["position"], obj["position"]) > max_horizontal:
                continue
        if min_cover is not None and predicate in VERTICAL:
            f = features_2d(subject["bbox_amodal"], obj["bbox_amodal"],
                            float(width), float(height))
            if f["cover"] < min_cover:
                continue

        out.append({
            "subject": rel["subject"], "predicate": predicate,
            "object": rel["object"],
            "subject_name": rel["subject_id"], "object_name": rel["object_id"],
            "subject_occlusion": occlusion.get(rel["subject_id"], 0.0),
            "object_occlusion": occlusion.get(rel["object_id"], 0.0),
            "annotation": rel["annotation"],
        })
    return out


def score(pairs, syn, iou_min: float, max_horizontal: Optional[float],
          min_cover: Optional[float]) -> Dict[str, Any]:
    per_pred_n: collections.Counter = collections.Counter()
    per_pred_hit: collections.Counter = collections.Counter()
    band_n: collections.Counter = collections.Counter()
    band_hit: collections.Counter = collections.Counter()

    for record, view, pred in pairs:
        width = record["intrinsics"]["width"]
        height = record["intrinsics"]["height"]
        rels = ranked(pred)[:100]
        boxes = {o["name"]: o for o in view["objects"]}
        for gt_rel in rebuild(view, width, height, max_horizontal, min_cover):
            if gt_rel["subject_name"] not in boxes or gt_rel["object_name"] not in boxes:
                continue
            name = band_of(max(gt_rel["subject_occlusion"],
                               gt_rel["object_occlusion"]))
            if name is None:
                continue
            hit = matched(gt_rel, boxes, rels, syn, iou_min, True)
            per_pred_n[gt_rel["predicate"]] += 1
            per_pred_hit[gt_rel["predicate"]] += hit
            band_n[name] += 1
            band_hit[name] += hit
    return {"pred_n": per_pred_n, "pred_hit": per_pred_hit,
            "band_n": band_n, "band_hit": band_hit,
            "total": sum(per_pred_n.values()),
            "hits": sum(per_pred_hit.values())}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--gt", default="datasets/occlusion_ds2")
    parser.add_argument("--preds",
                        default="/home/pithreeone/Ben/japan_intern/sgg_nvs/"
                                "results/occlusion_ds2_baseline")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--synonyms", default="a")
    parser.add_argument("--horizontal", type=float, nargs="*",
                        default=[None, 1.5, 1.0, 0.7])
    parser.add_argument("--cover", type=float, nargs="*",
                        default=[None, 0.1, 0.3])
    args = parser.parse_args(argv)

    pairs = load(args.gt, args.preds)
    if not pairs:
        print(f"no predictions matched under {args.preds}/")
        return 1
    syn = build_synonyms(args.synonyms)
    print(f"{len(pairs)} views, IoU>={args.iou}\n")

    horizontals = [None if h in (None, 0) else h for h in args.horizontal]
    covers = [None if c in (None, 0) else c for c in args.cover]

    print("=" * 78)
    print("VARIANTS  (horizontal cap on above/under/behind, cover floor on above/under)")
    print("=" * 78)
    header = (f"{'horiz':>7}{'cover':>7}{'GT':>7}{'hits':>6}{'R@100':>8}"
              + "".join(f"{b:>10}" for b in BANDS))
    print(header)
    results = {}
    for h, c in itertools.product(horizontals, covers):
        out = score(pairs, syn, args.iou, h, c)
        results[(h, c)] = out
        row = (f"{('-' if h is None else f'{h:.1f}'):>7}"
               f"{('-' if c is None else f'{c:.2f}'):>7}"
               f"{out['total']:>7}{out['hits']:>6}"
               f"{out['hits'] / max(out['total'], 1) * 100:>7.1f}%")
        for b in BANDS:
            n = out["band_n"][b]
            row += (f"{out['band_hit'][b] / n * 100:>9.1f}%" if n else f"{'-':>10}")
        print(row)

    print("\n" + "=" * 78)
    print("PER PREDICATE: GT count and R@100, baseline vs best variant")
    print("=" * 78)
    base = results[(None, None)]
    best = max(results.items(),
               key=lambda kv: kv[1]["hits"] / max(kv[1]["total"], 1))
    print(f"best variant: horizontal={best[0][0]}  cover={best[0][1]}\n")
    print(f"{'predicate':<14}{'GT base':>9}{'R base':>9}{'GT new':>9}{'R new':>9}"
          f"{'kept':>8}")
    for predicate in sorted(base["pred_n"], key=lambda p: -base["pred_n"][p]):
        bn, bh = base["pred_n"][predicate], base["pred_hit"][predicate]
        nn, nh = best[1]["pred_n"][predicate], best[1]["pred_hit"][predicate]
        print(f"{predicate:<14}{bn:>9}{bh / max(bn, 1) * 100:>8.1f}%"
              f"{nn:>9}{nh / max(nn, 1) * 100:>8.1f}%{nn / max(bn, 1) * 100:>7.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
