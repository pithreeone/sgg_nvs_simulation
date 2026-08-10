"""
vg_conventions.py -- measure what VG's HUMAN annotators mean by each predicate.

Our THOR ground truth was defined with 3D metric thresholds (`near` <= 0.5 m,
`above`/`under` vertical gap 0.15-1.0 m, `behind` by true depth order).  VG
annotators never saw depth: they labelled from a single photograph.  So a metric
definition can be geometrically correct and still disagree with every VG label,
which is exactly what the recall breakdown showed -- 0% on `above`/`under`, 2% on
`near`, while `on` (whose definition happens to be 2D-compatible) reached 44%.

This script recovers the annotation convention from the annotations themselves,
NOT from any model's predictions.  Calibrating against EGTR's output would make
the benchmark measure agreement-with-EGTR, so improvements to EGTR could no
longer be detected by it.  VG labels are human, so they are a legitimate target.

Reads   train.json  (COCO: images / annotations / categories)
        rel.json    ({split: {image_id: [[sub_idx, obj_idx, pred_id], ...]}})
where sub_idx/obj_idx index that image's annotations in file order.

Emits, per predicate, the distribution of seven scale-invariant 2D features so
THOR GT can be redefined in the same terms.  Percentiles, not just medians:
a threshold needs to know the spread it has to cover.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

#: The predicates our THOR export can produce, so those are the ones whose
#: convention we need.  `in front of` is VG's spelling.
TARGET_PREDICATES = ["on", "in", "near", "above", "under", "behind", "in front of"]

#: Reported alongside the targets because they are the same spatial idea under a
#: different word; if VG uses them interchangeably our GT should accept either.
NEIGHBOUR_PREDICATES = ["over", "on back of", "sitting on", "standing on",
                        "lying on", "against", "beside", "next to", "at",
                        "attached to", "mounted on", "hanging from", "in between",
                        "between", "part of", "belonging to", "covering", "covered in"]

PERCENTILES = (10, 25, 50, 75, 90)


def features(sub, obj, width: float, height: float) -> Dict[str, float]:
    """
    Seven scale-invariant descriptors of a subject/object box pair.

    Everything is normalised by image size so a 1024x768 photo and a 512x512
    render are comparable.  `dy` is positive when the subject sits LOWER in the
    image, matching screen coordinates, because that is the axis the monocular
    depth cue lives on -- higher in frame reads as further away, which is how a
    human annotator decides `behind` without any depth information.
    """
    sx1, sy1, sw, sh = sub
    ox1, oy1, ow, oh = obj
    sx2, sy2 = sx1 + sw, sy1 + sh
    ox2, oy2 = ox1 + ow, oy1 + oh
    scx, scy = sx1 + sw / 2.0, sy1 + sh / 2.0
    ocx, ocy = ox1 + ow / 2.0, oy1 + oh / 2.0

    ix = max(0.0, min(sx2, ox2) - max(sx1, ox1))
    iy = max(0.0, min(sy2, oy2) - max(sy1, oy1))
    inter = ix * iy
    area_s = max(sw * sh, 1e-6)
    area_o = max(ow * oh, 1e-6)

    return {
        "dy": (scy - ocy) / height,
        "dx": abs(scx - ocx) / width,
        "d2": math.hypot((scx - ocx) / width, (scy - ocy) / height),
        "iou": inter / max(area_s + area_o - inter, 1e-6),
        "cover": inter / area_s,            # fraction of SUBJECT inside object
        "area_ratio": area_s / area_o,
        "below": 1.0 if sy2 > oy2 else 0.0,  # subject's bottom edge is lower
    }


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def collect(coco_path: str, rel_path: str, split: str, limit: Optional[int]):
    with open(rel_path, encoding="utf-8") as handle:
        rel_data = json.load(handle)
    categories = rel_data["rel_categories"]
    per_image = rel_data[split]

    with open(coco_path, encoding="utf-8") as handle:
        coco = json.load(handle)
    sizes = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
    names = {c["id"]: c["name"] for c in coco["categories"]}

    # Annotations grouped by image, preserving file order -- that order is what
    # rel.json's indices refer to, so it must not be sorted.
    by_image: Dict[int, List[dict]] = collections.defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)

    samples = collections.defaultdict(list)
    class_pairs = collections.defaultdict(collections.Counter)
    skipped = 0
    seen = 0
    for image_id_str, triplets in per_image.items():
        image_id = int(image_id_str)
        anns = by_image.get(image_id)
        size = sizes.get(image_id)
        if not anns or not size:
            skipped += 1
            continue
        width, height = size
        for sub_idx, obj_idx, pred_id in triplets:
            if sub_idx >= len(anns) or obj_idx >= len(anns):
                skipped += 1
                continue
            predicate = categories[pred_id]
            samples[predicate].append(
                features(anns[sub_idx]["bbox"], anns[obj_idx]["bbox"], width, height))
            class_pairs[predicate][
                (names.get(anns[sub_idx]["category_id"], "?"),
                 names.get(anns[obj_idx]["category_id"], "?"))] += 1
        seen += 1
        if limit and seen >= limit:
            break
    return samples, class_pairs, skipped


def report(samples, class_pairs, predicates, show_pairs: int):
    keys = ["dy", "dx", "d2", "iou", "cover", "area_ratio", "below"]
    total = sum(len(v) for v in samples.values())
    print(f"{total} human-annotated relations, {len(samples)} predicates\n")

    print("=" * 96)
    print("MEDIAN 2D FEATURES PER PREDICATE  (VG human annotations)")
    print("=" * 96)
    print(f"{'predicate':<16}{'n':>7}" + "".join(f"{k:>11}" for k in keys))
    for predicate in predicates:
        rows = samples.get(predicate)
        if not rows:
            continue
        print(f"{predicate:<16}{len(rows):>7}"
              + "".join(f"{percentile([r[k] for r in rows], 50):>11.3f}" for k in keys))

    print("\n" + "=" * 96)
    print("SPREAD OF THE DISCRIMINATIVE FEATURES  (p10 / p25 / p50 / p75 / p90)")
    print("=" * 96)
    for feature in ("dy", "cover", "d2", "area_ratio"):
        print(f"\n  [{feature}]")
        for predicate in predicates:
            rows = samples.get(predicate)
            if not rows:
                continue
            values = [r[feature] for r in rows]
            spread = "  ".join(f"{percentile(values, q):>7.3f}" for q in PERCENTILES)
            print(f"    {predicate:<16}{len(values):>7}   {spread}")

    if show_pairs:
        print("\n" + "=" * 96)
        print("MOST COMMON CLASS PAIRS  (what kind of thing gets this label)")
        print("=" * 96)
        for predicate in predicates:
            counter = class_pairs.get(predicate)
            if not counter:
                continue
            top = ", ".join(f"{s}-{o}({c})" for (s, o), c in counter.most_common(show_pairs))
            print(f"  {predicate:<16}{top}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = "/home/pithreeone/dataset/visual_genome"
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--root", default=root)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N images (for a quick look)")
    parser.add_argument("--pairs", type=int, default=0,
                        help="show this many top class pairs per predicate")
    parser.add_argument("--all", action="store_true",
                        help="include the neighbouring spatial predicates too")
    parser.add_argument("--dump", default=None,
                        help="write the raw per-predicate feature stats to JSON")
    args = parser.parse_args(argv)

    coco = os.path.join(args.root, f"{args.split}.json")
    rel = os.path.join(args.root, "rel.json")
    for path in (coco, rel):
        if not os.path.exists(path):
            print(f"missing {path}")
            return 1

    samples, class_pairs, skipped = collect(coco, rel, args.split, args.limit)
    predicates = list(TARGET_PREDICATES)
    if args.all:
        predicates += [p for p in NEIGHBOUR_PREDICATES if p in samples]
    report(samples, class_pairs, predicates, args.pairs)
    if skipped:
        print(f"\n({skipped} triplets/images skipped: index out of range or no size)")

    if args.dump:
        keys = ["dy", "dx", "d2", "iou", "cover", "area_ratio", "below"]
        out = {}
        for predicate, rows in samples.items():
            out[predicate] = {"n": len(rows)}
            for key in keys:
                values = [r[key] for r in rows]
                out[predicate][key] = {f"p{q}": percentile(values, q) for q in PERCENTILES}
        with open(args.dump, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=1)
        print(f"\nwrote {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
