"""
analyze_multiview.py -- diagnose WHERE recall is lost on the multiview reference views.

Reads GT from   multiview/<scene>/<ref>/scene.json
and predictions from <preds>/<scene>_<ref>_results.json

The headline recall number says how much is missed; this says what is missing it.
Every GT relation is assigned exactly one failure mode, in order:

  ok                  matched at rank < K
  rank_too_deep       matched, but below the top-K cut
  predicate_wrong     both boxes located, predicate mismatched
  family_wrong        ... and it is not even the right action family
  box_iou_failed      both classes predicted somewhere, but no box pair clears IoU
  subject_missing     the subject's VG150 class never appears in the prediction
  object_missing      the object's class never appears

Also reports, for every GT object, which class EGTR actually assigned to the box
that best overlaps it.  That separates a genuine detection failure from a naming
disagreement: the model may be looking right at the object and calling it
something else, which no amount of fusion will fix but a mapping change would.
"""

from __future__ import annotations


import argparse
import collections
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vg.vg150 import PREDICATE_FAMILY, RelationFamily

IOU_MATCH = 0.5

#: Classes VG150 does not cleanly separate, so scoring one against the other as a
#: miss measures the vocabulary rather than the model.  Chosen from the MEASURED
#: naming table, not from intuition.
#:
#: This list used to hold five groups.  Three of them turned out to be covering
#: for a wrong entry in `THOR_TO_VG150` rather than for a real vocabulary overlap:
#: once `Sofa`/`Stool`/`Footstool` were remapped from `seat` to `chair` and
#: `GarbageCan` from `basket` to `box`, the contribution of {chair, seat, bench},
#: {lamp, light} and {bag, basket, box} fell to +0.1, 0.0 and 0.0 points of R@100.
#: A synonym allowance that buys nothing should not be in the headline
#: configuration, so they are gone; trimming cost 0.1 points (13.3 -> 13.2) and
#: removed three arbitrary concessions.
#:
#: Tier B stays split out because it is arguable rather than merely unhelpful.
SYNONYMS_A = [                     # defensible: VG150 itself overlaps here
    {"table", "desk"},             # +1.2 R@100 -- the only substantial group
    {"cabinet", "drawer"},         # +0.2; in THOR both are fronts of one unit
]
SYNONYMS_B = [                     # arguable: report separately
    {"cabinet", "door", "drawer"},
    {"curtain", "window"},
    {"vase", "bowl", "cup", "pot"},
    {"laptop", "screen"},
    {"counter", "sink", "table"},
]


def build_synonyms(tier: str):
    groups = []
    if tier in ("a", "ab"):
        groups += SYNONYMS_A
    if tier == "ab":
        groups += SYNONYMS_B
    mapping = {}
    for group in groups:
        for member in group:
            mapping.setdefault(member, set()).update(group)
    return mapping


def same_class(a: str, b: str, syn) -> bool:
    return a == b or (a in syn and b in syn[a])


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def family_of(predicate: str) -> RelationFamily:
    spec = PREDICATE_FAMILY.get(predicate)
    return spec[0] if spec else RelationFamily.UNKNOWN


def load_pairs(gt_dir: str, pred_dir: str) -> List[Tuple[Dict, Dict, str]]:
    out = []
    for path in sorted(glob.glob(os.path.join(gt_dir, "*", "r*", "scene.json"))):
        with open(path, encoding="utf-8") as handle:
            gt = json.load(handle)
        name = f"{gt['scene']}_{gt['ref_label']}"
        pred_path = os.path.join(pred_dir, f"{name}_results.json")
        if not os.path.exists(pred_path):
            continue
        with open(pred_path, encoding="utf-8") as handle:
            pred = json.load(handle)
        out.append((gt, pred, name))
    return out


def ranked(pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    objects = pred["objects"]
    rels = []
    for r in pred["relationships"]:
        s, o = objects[r["sub_idx"]], objects[r["obj_idx"]]
        rels.append({
            "s_label": s["label"], "o_label": o["label"],
            "s_bbox": s["bbox"], "o_bbox": o["bbox"],
            "predicate": r["label"],
            "score": r.get("pure_rel_score", r.get("score", 0.0)),
            "all_scores": r.get("all_scores") or {},
        })
    rels.sort(key=lambda r: -r["score"])
    return rels


def classify(gt_rel, gt_boxes, rels, predicted_labels, k: int, objects=None,
             syn=None):
    """
    Assign one failure mode to a GT relation.

    `box_iou_failed` and `pair_not_proposed` must be kept apart.  EGTR emits ~200
    object queries but only ~100 relationship triplets, so both endpoints can be
    detected perfectly while no triplet happens to connect that particular pair.
    Lumping the two together reads as a detection problem when it is really a
    pair-proposal problem -- and they call for completely different fixes.
    """
    syn = syn or {}
    subj, obj = gt_rel["subject"], gt_rel["object"]
    if isinstance(obj, list):
        return "unsupported", None
    sbox, obox = gt_boxes.get(gt_rel["subject_id"]), gt_boxes.get(gt_rel["object_id"])
    if sbox is None or obox is None:
        return "unsupported", None

    if not any(same_class(subj, p, syn) for p in predicted_labels):
        return "subject_missing", None
    if not any(same_class(obj, p, syn) for p in predicted_labels):
        return "object_missing", None

    best_pair_rank = None
    best_pred_at_pair = None
    for index, r in enumerate(rels):
        if not (same_class(subj, r["s_label"], syn)
                and same_class(obj, r["o_label"], syn)):
            continue
        if iou(r["s_bbox"], sbox) < IOU_MATCH or iou(r["o_bbox"], obox) < IOU_MATCH:
            continue
        if best_pair_rank is None:
            best_pair_rank, best_pred_at_pair = index, r["predicate"]
        if r["predicate"] == gt_rel["predicate"]:
            return ("ok" if index < k else "rank_too_deep"), index

    if best_pair_rank is None:
        # Are both endpoints detected at all, as OBJECTS rather than within a
        # triplet?  If so the detection is fine and the pair was simply never
        # proposed.
        if objects is not None:
            s_ok = any(same_class(subj, o["label"], syn)
                       and iou(o["bbox"], sbox) >= IOU_MATCH for o in objects)
            o_ok = any(same_class(obj, o["label"], syn)
                       and iou(o["bbox"], obox) >= IOU_MATCH for o in objects)
            if s_ok and o_ok:
                return "pair_not_proposed", None
            if not s_ok:
                return "subject_box_failed", None
            return "object_box_failed", None
        return "box_iou_failed", None
    if family_of(best_pred_at_pair) is family_of(gt_rel["predicate"]):
        return "predicate_wrong", best_pair_rank
    return "family_wrong", best_pair_rank


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--gt", default="datasets/sgg/multiview")
    parser.add_argument("--preds", default="datasets/sgg/multiview_ref")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--synonyms", default="none", choices=("none","a","ab"),
                        help="treat VG150 classes the vocabulary does not separate "
                             "as equivalent; 'a' = defensible groups only")
    parser.add_argument("--strict", action="store_true",
                        help="exclude relations flagged `loose`")
    args = parser.parse_args(argv)

    syn = build_synonyms(args.synonyms)
    pairs = load_pairs(args.gt, args.preds)
    if not pairs:
        print(f"no matching predictions in {args.preds}/")
        return 1

    modes = collections.Counter()
    per_pred = collections.defaultdict(collections.Counter)
    naming = collections.Counter()       # (gt class -> predicted class at best box)
    gt_class_found = collections.Counter()
    gt_class_total = collections.Counter()
    n_rels = 0

    for gt, pred, name in pairs:
        rels = ranked(pred)
        gt_boxes = {o["object_id"]: o["bbox_xyxy"] for o in gt["reference"]["objects"]}
        predicted_labels = {o["label"] for o in pred["objects"]}

        # --- naming: what does EGTR call each GT object? --------------------
        for entry in gt["reference"]["objects"]:
            if args.strict and entry.get("loose"):
                continue
            gt_class_total[entry["vg150_class"]] += 1
            best, best_iou = None, 0.0
            for candidate in pred["objects"]:
                value = iou(candidate["bbox"], entry["bbox_xyxy"])
                if value > best_iou:
                    best, best_iou = candidate, value
            if best is not None and best_iou >= IOU_MATCH:
                naming[(entry["vg150_class"], best["label"])] += 1
                if best["label"] == entry["vg150_class"]:
                    gt_class_found[entry["vg150_class"]] += 1

        for gt_rel in gt["reference"]["relations"]:
            if args.strict and gt_rel.get("loose"):
                continue
            mode, _ = classify(gt_rel, gt_boxes, rels, predicted_labels, args.k,
                               objects=pred["objects"], syn=syn)
            if mode == "unsupported":
                continue
            n_rels += 1
            modes[mode] += 1
            per_pred[gt_rel["predicate"]][mode] += 1

    print(f"{len(pairs)} views, {n_rels} GT relations"
          f"{'  (strict only)' if args.strict else ''}\n")

    print("=" * 72)
    print("WHERE RECALL IS LOST")
    print("=" * 72)
    order = ["ok", "rank_too_deep", "predicate_wrong", "family_wrong",
             "pair_not_proposed", "subject_box_failed", "object_box_failed",
             "box_iou_failed", "subject_missing", "object_missing"]
    for mode in order:
        if not modes[mode]:
            continue
        print(f"  {mode:<18}{modes[mode]:>6}{modes[mode]/n_rels*100:>7.1f}%")
    print(f"  {'RECALL@'+str(args.k):<18}{modes['ok']:>6}"
          f"{modes['ok']/n_rels*100:>7.1f}%")

    print("\n" + "=" * 72)
    print("PER PREDICATE")
    print("=" * 72)
    cols = ["ok", "family_wrong", "pair_not_proposed", "subject_box_failed",
            "object_box_failed", "subject_missing", "object_missing"]
    print(f"{'predicate':<14}{'n':>5}{'recall':>8}" + "".join(f"{c[:9]:>11}" for c in cols[1:]))
    for predicate, counter in sorted(per_pred.items(),
                                     key=lambda kv: -sum(kv[1].values())):
        total = sum(counter.values())
        print(f"{predicate:<14}{total:>5}{counter['ok']/total*100:>7.1f}%"
              + "".join(f"{counter[c]:>11}" for c in cols[1:]))

    print("\n" + "=" * 72)
    print("NAMING: what EGTR calls each GT object (box IoU >= 0.5)")
    print("=" * 72)
    print(f"{'GT class':<12}{'in GT':>7}{'agreed':>8}{'agree%':>8}   most common EGTR label")
    for cls, total in gt_class_total.most_common(18):
        agreed = gt_class_found[cls]
        others = [(p, c) for (g, p), c in naming.items() if g == cls and p != cls]
        others.sort(key=lambda t: -t[1])
        alt = ", ".join(f"{p}({c})" for p, c in others[:3]) or "-"
        print(f"{cls:<12}{total:>7}{agreed:>8}{agreed/total*100:>7.0f}%   {alt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
