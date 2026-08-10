"""
eval_occlusion.py -- recall stratified by how occluded the relation's endpoints are.

This is the measurement the dataset exists to make.  VG annotates what its
annotators could see, so a model trained on it has never been supervised on a
half-hidden object; the claim is that recall therefore collapses as occlusion
rises, in a way no VG-derived benchmark can show.  The output is one table:

    occlusion band     GT      R@20   R@50   R@100

A box match is scored against BOTH the amodal and the visible box, taking
whichever gives the higher IoU.  Neither alone is fair.  A detector shown an
80%-occluded object predicts roughly its visible extent, so amodal-only fails it
at any sensible threshold; but the visible box SHRINKS as occlusion grows, so
visible-only makes IoU easier the more hidden the object is, rewarding exactly
what the benchmark is meant to penalise.

Inverse predicates are accepted (`under(B,A)` for `above(A,B)`) because the
ground truth emits one direction per pair -- measured over VG train, human
annotators labelled both directions only 2-7% of the time -- and EGTR has no
passive predicates, so a relation stated the other way round comes back as its
inverse rather than as a subject/object swap.

`--accept` chooses how strictly a wording has to match.

  exact    the predicate string, plus the inverse table above.  Report this.
  human    also accept any wording VG's annotators measurably use for THOSE TWO
           CLASSES at a comparable rate -- `near` for a `chair`/`table` pair our
           ground truth calls `behind`, which 48% of VG's 805 annotations of that
           class pair also call `near`.  See `vg_pair_prior.py`.

`human` is not free.  Widening the accept-set raises recall on its own: a RANDOM
accept-set of the same per-relation size scores 22.4% where the VG-licensed one
scores 27.2%, so about half of that particular gain is permissiveness rather than
convention.  Quote `exact` as the headline and `human` as the upper bound on what
a wording disagreement can be costing.
"""

from __future__ import annotations

# Importable BOTH ways, and that is a requirement rather than a nicety:
# `../sgg_nvs/lib/occ_eval.py` puts this directory on sys.path and imports this
# module by BARE NAME, while everything in the simulation repo imports it as
# `scoring.eval_occlusion`.  The bare-name route leaves the repo root off the
# path, so the sibling import below would fail without this.
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
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scoring.analyze_multiview import build_synonyms, iou, same_class
from gen.occlusion import MAX_OCCLUSION

INVERSE = {
    "above": "under", "under": "above",
    "behind": "in front of", "in front of": "behind",
    "near": "near",
}

#: The top edge tracks the build ceiling: `band_of` returns None outside the
#: bands, so a hardcoded edge below `MAX_OCCLUSION` silently drops every
#: relation between the two from all metrics.
#: Uniform quarters, so one band's recall is comparable to its neighbour's
#: without first discounting for width.  The old edges (0.05/0.25/0.50) put a
#: 0.05-wide band beside a 0.40-wide one, which made the slope unreadable.
BANDS = [(0.00, 0.25, "0.00-0.25"), (0.25, 0.50, "0.25-0.50"),
         (0.50, 0.75, "0.50-0.75"), (0.75, MAX_OCCLUSION + 0.01, "0.75+")]

KS = (20, 50, 100)


def ranked(pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    objects = pred["objects"]
    rels = []
    for r in pred["relationships"]:
        s, o = objects[r["sub_idx"]], objects[r["obj_idx"]]
        rels.append({"s_label": s["label"], "o_label": o["label"],
                     "s_bbox": s["bbox"], "o_bbox": o["bbox"],
                     "predicate": r["label"],
                     "score": r.get("pure_rel_score", r.get("score", 0.0))})
    rels.sort(key=lambda r: -r["score"])
    return rels


def best_iou(pred_box: Sequence[float], entry: Dict[str, Any]) -> float:
    """Higher of the amodal and visible IoU -- see the module docstring."""
    value = iou(pred_box, entry["bbox_amodal"])
    visible = entry.get("bbox_visible")
    if visible:
        value = max(value, iou(pred_box, visible))
    return value


def wordings(gt_rel: Dict[str, Any], accept: str) -> Tuple[set, set]:
    """
    The predicate strings that count as stating this relation, and its inverse.

    Under `exact` this is the predicate and the one-entry inverse table.  Under
    `human` it also includes wordings VG's annotators use for the same class
    pair, and the inverse table is replaced by the measured one in
    `vg_pair_prior`, which is where `on(A,B) <- under(B,A)` and `<- has(B,A)`
    come from.  Those two alone recover 129 of our 391 `on` relations on
    `occlusion_ds2`: they are the same fact with the arguments swapped, and VG
    humans write `has(o,s)` for `on(s,o)` 34% of the time -- more often than they
    repeat `on`.
    """
    predicate = gt_rel["predicate"]
    if accept == "exact":
        reverse = INVERSE.get(predicate)
        return {predicate}, ({reverse} if reverse else set())
    from vg.vg_pair_prior import accept_set, inverse_accept_set
    subject, obj = gt_rel["subject"], gt_rel["object"]
    return (accept_set(predicate, subject, obj),
            inverse_accept_set(predicate, subject, obj))


def matched(gt_rel: Dict[str, Any], boxes: Dict[str, Dict[str, Any]],
            rels: Sequence[Dict[str, Any]], syn, iou_min: float,
            inverse: bool, class_agnostic: bool = False,
            accept: str = "exact") -> bool:
    """
    Does any predicted triplet cover this relation?

    `class_agnostic` drops the object-label requirement and keeps the boxes and
    the predicate.  That is not a way of forgiving mistakes: 56% of localised
    objects get the wrong class here, mostly genuine model error rather than
    vocabulary mismatch -- `chair` comes back as `bed` 20 times -- and folding
    that into one number leaves the relation head unmeasurable.  Reporting both
    separates "found the pair and named the relation" from "named the objects",
    which is what the field's PredCls / SGCls / SGDet split exists to do.
    """
    subject = boxes.get(gt_rel["subject_name"])
    obj = boxes.get(gt_rel["object_name"])
    if subject is None or obj is None:
        return False
    forward, reverse = wordings(gt_rel, accept)
    if not inverse:
        reverse = set()

    def names(gt_a: str, gt_b: str, pred_a: str, pred_b: str) -> bool:
        if class_agnostic:
            return True
        return (same_class(gt_a, pred_a, syn) and same_class(gt_b, pred_b, syn))

    for r in rels:
        if (r["predicate"] in forward
                and names(gt_rel["subject"], gt_rel["object"],
                          r["s_label"], r["o_label"])
                and best_iou(r["s_bbox"], subject) >= iou_min
                and best_iou(r["o_bbox"], obj) >= iou_min):
            return True
        if (r["predicate"] in reverse
                and names(gt_rel["object"], gt_rel["subject"],
                          r["s_label"], r["o_label"])
                and best_iou(r["s_bbox"], obj) >= iou_min
                and best_iou(r["o_bbox"], subject) >= iou_min):
            return True
    return False


def detected(entry: Dict[str, Any], objects: Sequence[Dict[str, Any]], syn,
             iou_min: float) -> bool:
    """Was this object found at all, as an object rather than inside a triplet?"""
    return any(same_class(entry["vg150_class"], o["label"], syn)
               and best_iou(o["bbox"], entry) >= iou_min for o in objects)


def load(gt_root: str, pred_root: str) -> List[Tuple[Dict, Dict, Dict]]:
    out = []
    for path in sorted(glob.glob(os.path.join(gt_root, "*", "scene.json"))):
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        stem = f"{record['scene']}_s{record['seed']}"
        for view in record["views"]:
            name = f"{stem}_view_{view['index']:02d}_results.json"
            pred_path = os.path.join(pred_root, name)
            if not os.path.exists(pred_path):
                continue
            with open(pred_path, encoding="utf-8") as handle:
                out.append((record, view, json.load(handle)))
    return out


def band_of(value: float) -> Optional[str]:
    for low, high, name in BANDS:
        if low <= value < high:
            return name
    return None


#: Configurations swept by `--summary`.  Each varies exactly ONE thing against
#: the row above it, so a difference is attributable:
#:
#:   exact -> human           what wording disagreement costs
#:   human -> class-agnostic  what object NAMING costs, with wording already
#:                            forgiven; isolates the relation head
#:
#: Same shape as `eval_recall.SWEEP`, deliberately -- the two evaluators should
#: be readable side by side.
SWEEP = [
    ("exact match", dict(accept="exact", class_agnostic=False)),
    ("human wordings", dict(accept="human", class_agnostic=False)),
    ("human + class-agnostic", dict(accept="human", class_agnostic=True)),
]

SUMMARY_COLUMNS = [f"R@{k}" for k in KS] + [f"mR@{k}" for k in KS]


def tally(pairs, syn, iou_min: float, inverse: bool, accept: str,
          class_agnostic: bool):
    """Per-K hits by band and by predicate, plus the shared GT totals."""
    hits = {k: collections.Counter() for k in KS}
    totals: collections.Counter = collections.Counter()
    per_pred = collections.defaultdict(collections.Counter)
    for _, view, pred in pairs:
        rels = ranked(pred)
        boxes = {o["name"]: o for o in view["objects"]}
        for gt_rel in view["relations"]:
            name = band_of(max(gt_rel["subject_occlusion"],
                               gt_rel["object_occlusion"]))
            if name is None or gt_rel["subject_name"] not in boxes \
                    or gt_rel["object_name"] not in boxes:
                continue
            totals[name] += 1
            per_pred[gt_rel["predicate"]][("n", name)] += 1
            for k in KS:
                if matched(gt_rel, boxes, rels[:k], syn, iou_min, inverse,
                           class_agnostic, accept):
                    hits[k][name] += 1
                    per_pred[gt_rel["predicate"]][(k, name)] += 1
    return hits, totals, per_pred


def macro(per_pred, k: int) -> float:
    """mR@k -- mean of the per-predicate recalls, unweighted.

    It matters more here than R@k does: `on` is about 43% of the ground truth and
    the only predicate whose derivation already agreed with VG's convention, so
    R@k is close to a measurement of `on` alone.
    """
    present = [p for p, c in per_pred.items()
               if sum(v for (t, _), v in c.items() if t == "n")]
    if not present:
        return 0.0
    total = 0.0
    for p in present:
        n = sum(v for (t, _), v in per_pred[p].items() if t == "n")
        total += sum(v for (t, _), v in per_pred[p].items() if t == k) / n
    return total / len(present) * 100


def summary(pairs, args, syn, inverse: bool) -> int:
    """One row per configuration; R@K and mR@K as columns."""
    print(f"{len(pairs)} views   IoU>={args.iou} (best of amodal/visible)   "
          f"synonyms={args.synonyms}   inverse={'on' if inverse else 'off'}\n")
    width = 26
    head = f"{'':<{width}}{'GT':>7}" + "".join(f"{c:>9}" for c in SUMMARY_COLUMNS)
    print(head)
    print("-" * len(head))
    per_band = {}
    for label, config in SWEEP:
        hits, totals, per_pred = tally(pairs, syn, args.iou, inverse, **config)
        n = sum(totals.values())
        values = {f"R@{k}": sum(hits[k].values()) / n * 100 for k in KS}
        values.update({f"mR@{k}": macro(per_pred, k) for k in KS})
        print(f"{label:<{width}}{n:>7}"
              + "".join(f"{values[c]:>9.1f}" for c in SUMMARY_COLUMNS))
        per_band[label] = (hits, totals)

    print(f"\nR@100 BY OCCLUSION BAND")
    names = [b[2] for b in BANDS]
    head = f"{'':<{width}}" + "".join(f"{n:>13}" for n in names)
    print(head)
    print("-" * len(head))
    for label, _ in SWEEP:
        hits, totals = per_band[label]
        row = f"{label:<{width}}"
        for n in names:
            row += (f"{hits[100][n] / totals[n] * 100:>12.1f}%" if totals[n]
                    else f"{'-':>13}")
        print(row)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--gt", default="datasets/occlusion_ds2")
    parser.add_argument("--preds",
                        default="/home/pithreeone/Ben/japan_intern/sgg_nvs/"
                                "results/occlusion_ds2_baseline")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--synonyms", default="a", choices=("none", "a", "ab"))
    parser.add_argument("--no-inverse", action="store_true")
    parser.add_argument("--class-agnostic", action="store_true",
                        help="match on boxes and predicate only, ignoring object "
                             "labels; isolates the relation head from naming")
    parser.add_argument("--accept", default="exact", choices=("exact", "human"),
                        help="`human` also accepts wordings VG annotators use for "
                             "the same class pair; see the module docstring for "
                             "why it must be reported next to `exact`")
    parser.add_argument("--summary", action="store_true",
                        help="one row per accept mode, R@K and mR@K as columns, "
                             "so exact and human are read side by side")
    args = parser.parse_args(argv)

    pairs = load(args.gt, args.preds)
    if not pairs:
        print(f"no predictions matched under {args.preds}/")
        return 1
    syn = build_synonyms(args.synonyms)
    inverse = not args.no_inverse

    if args.summary:
        return summary(pairs, args, syn, inverse)

    hits = {k: collections.Counter() for k in KS}
    totals: collections.Counter = collections.Counter()
    per_pred = collections.defaultdict(collections.Counter)
    obj_found: collections.Counter = collections.Counter()
    obj_total: collections.Counter = collections.Counter()

    for _, view, pred in pairs:
        rels = ranked(pred)
        boxes = {o["name"]: o for o in view["objects"]}

        for entry in view["objects"]:
            name = band_of(entry["occlusion"])
            if name is None:
                continue
            obj_total[name] += 1
            obj_found[name] += detected(entry, pred["objects"], syn, args.iou)

        for gt_rel in view["relations"]:
            worst = max(gt_rel["subject_occlusion"], gt_rel["object_occlusion"])
            name = band_of(worst)
            if name is None or gt_rel["subject_name"] not in boxes \
                    or gt_rel["object_name"] not in boxes:
                continue
            totals[name] += 1
            per_pred[gt_rel["predicate"]][("n", name)] += 1
            for k in KS:
                if matched(gt_rel, boxes, rels[:k], syn, args.iou, inverse,
                           args.class_agnostic, args.accept):
                    hits[k][name] += 1
                    per_pred[gt_rel["predicate"]][(k, name)] += 1

    print(f"{len(pairs)} views   IoU>={args.iou} (best of amodal/visible)   "
          f"synonyms={args.synonyms}   inverse={'on' if inverse else 'off'}"
          f"   accept={args.accept}"
          f"{'   CLASS-AGNOSTIC' if args.class_agnostic else ''}\n")

    print("=" * 74)
    print("RELATION RECALL BY ENDPOINT OCCLUSION")
    print("=" * 74)
    print(f"{'occlusion':<22}{'GT':>7}" + "".join(f"{'R@'+str(k):>11}" for k in KS))
    for _, _, name in BANDS:
        n = totals[name]
        if not n:
            continue
        row = f"{name:<22}{n:>7}"
        for k in KS:
            row += f"{hits[k][name] / n * 100:>10.1f}%"
        print(row)
    grand = sum(totals.values())
    row = f"{'all':<22}{grand:>7}"
    for k in KS:
        row += f"{sum(hits[k].values()) / grand * 100:>10.1f}%"
    print(row)

    # mR matters more than R here than it usually does: `on` is half the ground
    # truth and the only predicate whose derivation already agreed with VG, so
    # R@K is close to a measurement of `on` alone.
    present = [p for p, c in per_pred.items()
               if sum(v for (t, _), v in c.items() if t == "n")]
    row = f"{'mR@K (macro, ' + str(len(present)) + ' preds)':<22}{'':>7}"
    for k in KS:
        total = 0.0
        for p in present:
            n = sum(v for (t, _), v in per_pred[p].items() if t == "n")
            got = sum(v for (t, _), v in per_pred[p].items() if t == k)
            total += got / n
        row += f"{total / len(present) * 100:>10.1f}%"
    print(row)

    print("\n" + "=" * 74)
    print("OBJECT DETECTION BY OCCLUSION  (upstream of the relation head)")
    print("=" * 74)
    print(f"{'occlusion':<22}{'objects':>9}{'detected':>10}{'rate':>9}")
    for _, _, name in BANDS:
        n = obj_total[name]
        if n:
            print(f"{name:<22}{n:>9}{obj_found[name]:>10}"
                  f"{obj_found[name] / n * 100:>8.1f}%")

    print("\n" + "=" * 74)
    print("PER PREDICATE, R@100")
    print("=" * 74)
    names = [b[2] for b in BANDS]
    print(f"{'predicate':<14}{'GT':>6}" + "".join(f"{n:>12}" for n in names))
    for predicate, counter in sorted(
            per_pred.items(),
            key=lambda kv: -sum(v for (t, _), v in kv[1].items() if t == "n")):
        total = sum(v for (t, _), v in counter.items() if t == "n")
        row = f"{predicate:<14}{total:>6}"
        for name in names:
            n = counter[("n", name)]
            row += (f"{counter[(100, name)] / n * 100:>11.0f}%" if n else f"{'-':>12}")
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
