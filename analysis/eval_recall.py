"""
eval_recall.py -- R@K and mR@K on the multiview reference views.

Reports both ground truths side by side:

  metric    the 3D-threshold definitions in `export_samples.py`
  vg        the human-convention definitions in `vg_gt.py`

mR@K is the mean of the per-predicate recalls, unweighted.  It matters more than
R@K here than it usually does: `on` is 24% of the metric ground truth and the only
predicate whose original definition agreed with VG's convention, so R@K is close
to being a measurement of `on` alone.  mR@K gives the seven predicates equal say
and therefore reports the head/tail split honestly.

Matching follows the usual SGG rule -- both boxes at IoU >= 0.5 against the same
GT pair, and the predicate string equal.  Two options depart from it, each for a
reason measured rather than assumed:

  --synonyms   VG150 does not separate some classes that THOR does, so scoring
               `seat` against the model's `chair` as a miss measures the
               vocabulary and not the model.  Tier `a` holds the groups where
               VG150 itself overlaps.
  --no-inverse turns off accepting `under(B,A)` for `above(A,B)`.  The VG ground
               truth emits ONE direction per pair, because human annotators label
               both only 2-7% of the time, so without inverse acceptance a model
               is penalised for a semantically identical restatement.  The metric
               ground truth emitted both directions and so never needed this.
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
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from scoring.analyze_multiview import build_synonyms, iou, load_pairs, ranked, same_class
from vg.vg_gt import support_pairs, vg_calibrated_relations

#: EGTR has no passive predicates, so a relation stated in the other direction
#: comes back as its inverse rather than as a subject/object swap.
INVERSE = {
    "above": "under", "under": "above",
    "behind": "in front of", "in front of": "behind",
    "near": "near",
}

PREDICATES = ["on", "in", "near", "above", "under", "behind", "in front of"]
KS = (20, 50, 100)


def metric_gt(scene: Dict[str, Any], view: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The original 3D-threshold ground truth, as stored."""
    return view["relations"]


def vg_gt_relations(scene: Dict[str, Any], view: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Support edges as stored, geometric edges re-derived in VG's convention.

    `on`/`in` are kept untouched: they come from `parentReceptacles`, which the
    simulator asserts directly, and their 2D footprint already matches VG (subject
    box inside object box, cover median 1.000).  Only the predicates whose metric
    definitions disagreed with the convention are recomputed.
    """
    support = [r for r in view["relations"] if r["predicate"] in ("on", "in")]
    geometric = vg_calibrated_relations(
        view["objects"], scene["width"], scene["height"], support_pairs(support))
    return support + geometric


def matched(gt_rel: Dict[str, Any], rels: List[Dict[str, Any]],
            boxes: Dict[str, Sequence[float]], syn: Dict[str, set],
            iou_min: float, inverse: bool) -> bool:
    sbox, obox = boxes.get(gt_rel["subject_id"]), boxes.get(gt_rel["object_id"])
    if sbox is None or obox is None:
        return False
    predicate = gt_rel["predicate"]
    reverse = INVERSE.get(predicate) if inverse else None

    for r in rels:
        if (r["predicate"] == predicate
                and same_class(gt_rel["subject"], r["s_label"], syn)
                and same_class(gt_rel["object"], r["o_label"], syn)
                and iou(r["s_bbox"], sbox) >= iou_min
                and iou(r["o_bbox"], obox) >= iou_min):
            return True
        if (reverse and r["predicate"] == reverse
                and same_class(gt_rel["object"], r["s_label"], syn)
                and same_class(gt_rel["subject"], r["o_label"], syn)
                and iou(r["s_bbox"], obox) >= iou_min
                and iou(r["o_bbox"], sbox) >= iou_min):
            return True
    return False


def score(pairs, make_gt: Callable, syn, iou_min: float, inverse: bool,
          strict: bool) -> Tuple[Dict[int, Dict[str, int]], Dict[str, int]]:
    """Per-K, per-predicate hit counts plus the shared GT totals."""
    hits = {k: collections.Counter() for k in KS}
    totals: collections.Counter = collections.Counter()

    for scene, pred, _ in pairs:
        view = scene["reference"]
        boxes = {o["object_id"]: o["bbox_xyxy"] for o in view["objects"]}
        all_rels = ranked(pred)
        relations = [r for r in make_gt(scene, view)
                     if not isinstance(r.get("object"), list)
                     and not (strict and r.get("loose"))]
        for gt_rel in relations:
            if gt_rel["subject_id"] not in boxes or gt_rel["object_id"] not in boxes:
                continue
            totals[gt_rel["predicate"]] += 1
            for k in KS:
                if matched(gt_rel, all_rels[:k], boxes, syn, iou_min, inverse):
                    hits[k][gt_rel["predicate"]] += 1
    return hits, totals


def metrics(hits, totals, present: List[str]) -> Dict[str, float]:
    """R@K (micro, over all relations) and mR@K (macro, mean of per-predicate)."""
    n_all = sum(totals.values())
    out = {}
    for k in KS:
        out[f"R@{k}"] = sum(hits[k].values()) / n_all * 100
        out[f"mR@{k}"] = (sum(hits[k][p] / totals[p] for p in present)
                          / len(present) * 100)
    return out


COLUMNS = [f"R@{k}" for k in KS] + [f"mR@{k}" for k in KS]


def summary_row(label: str, values: Dict[str, float], width: int = 30) -> str:
    return f"{label:<{width}}" + "".join(f"{values[c]:>9.1f}" for c in COLUMNS)


def summary_header(width: int = 30) -> str:
    head = f"{'':<{width}}" + "".join(f"{c:>9}" for c in COLUMNS)
    return head + "\n" + "-" * len(head)


def table(name: str, hits, totals, present: List[str]) -> None:
    print(f"\n{name}")
    print("-" * 74)
    header = f"{'predicate':<14}{'GT':>6}" + "".join(f"{'R@'+str(k):>10}" for k in KS)
    print(header + f"{'hits@100':>11}")
    for predicate in present:
        n = totals[predicate]
        row = f"{predicate:<14}{n:>6}"
        for k in KS:
            row += f"{hits[k][predicate] / n * 100:>9.1f}%"
        print(row + f"{hits[100][predicate]:>11}")

    n_all = sum(totals.values())
    print("-" * 74)
    row = f"{'R@K (micro)':<14}{n_all:>6}"
    for k in KS:
        row += f"{sum(hits[k].values()) / n_all * 100:>9.1f}%"
    print(row + f"{sum(hits[100].values()):>11}")

    row = f"{'mR@K (macro)':<14}{len(present):>6}"
    for k in KS:
        row += f"{sum(hits[k][p] / totals[p] for p in present) / len(present) * 100:>9.1f}%"
    print(row)
    print(f"{'relations/view':<14}{n_all / 200:>6.1f}")


#: Configurations swept by `--summary`.  Each varies exactly one thing against
#: the recommended setting (synonym tier `a`, IoU 0.5, all relations), so a row's
#: difference from the `a` row is attributable.
SWEEP = [
    ("exact match",            dict(synonyms="none", strict=False)),
    ("synonyms A",             dict(synonyms="a", strict=False)),
    ("synonyms A+B",           dict(synonyms="ab", strict=False)),
    ("synonyms A, strict",     dict(synonyms="a", strict=True)),
]


def summary(pairs, args) -> int:
    """One row per (ground truth, configuration), R@K and mR@K as columns."""
    print(f"{len(pairs)} reference views   IoU>={args.iou}\n")
    variants = [("GT A  3D metric thresholds", metric_gt, False),
                ("GT B  VG human convention", vg_gt_relations, not args.no_inverse)]
    if args.which == "metric":
        variants = variants[:1]
    elif args.which == "vg":
        variants = variants[1:]

    for name, make_gt, inverse in variants:
        print(name)
        print(summary_header())
        for label, config in SWEEP:
            syn = build_synonyms(config["synonyms"])
            hits, totals = score(pairs, make_gt, syn, args.iou, inverse,
                                 config["strict"])
            present = [p for p in PREDICATES if totals[p]]
            values = metrics(hits, totals, present)
            n = sum(totals.values())
            print(summary_row(f"{label}  (n={n})", values))
        print()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--gt", default="datasets/multiview")
    parser.add_argument("--preds", default="datasets/multiview_ref")
    parser.add_argument("--synonyms", default="a", choices=("none", "a", "ab"))
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--no-inverse", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="exclude relations involving a loose class mapping")
    parser.add_argument("--which", default="both", choices=("both", "metric", "vg"))
    parser.add_argument("--summary", action="store_true",
                        help="one row per configuration, R@K and mR@K as columns")
    args = parser.parse_args(argv)

    pairs = load_pairs(args.gt, args.preds)
    if not pairs:
        print(f"no matching predictions in {args.preds}/")
        return 1

    if args.summary:
        return summary(pairs, args)

    syn = build_synonyms(args.synonyms)

    print(f"{len(pairs)} reference views   synonyms={args.synonyms}   IoU>={args.iou}"
          f"   inverse={'off' if args.no_inverse else 'on'}"
          f"{'   strict' if args.strict else ''}")

    variants = []
    if args.which in ("both", "metric"):
        # The metric GT emits both directions of every inverse pair, so inverse
        # acceptance would double-count rather than rescue anything.
        variants.append(("GT A -- 3D metric thresholds (export_samples.py)",
                         metric_gt, False))
    if args.which in ("both", "vg"):
        variants.append(("GT B -- VG human convention (vg_gt.py)",
                         vg_gt_relations, not args.no_inverse))

    for name, make_gt, inverse in variants:
        hits, totals = score(pairs, make_gt, syn, args.iou, inverse, args.strict)
        present = [p for p in PREDICATES if totals[p]]
        table(name, hits, totals, present)
    return 0


if __name__ == "__main__":
    sys.exit(main())
