"""
find_partial_triplets.py -- pull the triplets whose endpoints are *partly* hidden.

A triplet qualifies when at least one of its two endpoints has an occlusion rate
strictly inside a window (0.4, 0.8) by default.  That window is the interesting
one: below it the object is essentially in plain sight, and above it -- see the
`0.75+` band in TASKS.md -- so little of the object survives that a miss says
more about the detector's confidence floor than about relation reasoning.

Everything needed is already in `scene.json`, so this never opens THOR.  The
occlusion figures are the build's own: 1 - visible_px / reference_px, where the
reference is the object rendered alone, so they are amodal-referenced and
comparable across views.

    python find_partial_triplets.py                       # ds4, table + per-scene counts
    python find_partial_triplets.py --lo 0.4 --hi 0.8
    python find_partial_triplets.py --dataset datasets/sgg/occlusion_ds3
    python find_partial_triplets.py --both-endpoints      # require BOTH in the window
    python find_partial_triplets.py --json hits.json      # full listing for downstream use
    python find_partial_triplets.py --list 40             # print the first 40 hits
"""

from __future__ import annotations


import argparse
import collections
import glob
import json
import os
from typing import Any, Dict, List


def qualifying(rel: Dict[str, Any], lo: float, hi: float,
               both: bool) -> bool:
    """True when the relation's endpoint occlusions satisfy the window."""
    occs = [rel["subject_occlusion"], rel["object_occlusion"]]
    inside = [lo < o < hi for o in occs]
    return all(inside) if both else any(inside)


def collect(dataset: str, lo: float, hi: float, both: bool) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    scenes = sorted(glob.glob(os.path.join(dataset, "*", "scene.json")))
    if not scenes:
        raise SystemExit(f"no scene.json under {dataset}/")
    for path in scenes:
        with open(path) as fh:
            scene = json.load(fh)
        scene_dir = os.path.dirname(path)
        for view in scene["views"]:
            for rel in view["relations"]:
                if not qualifying(rel, lo, hi, both):
                    continue
                hits.append({
                    "scene": scene["scene"],
                    "scene_dir": scene_dir,
                    "view": view["index"],
                    "image": os.path.join(scene_dir, view["image"]),
                    "subject": rel["subject"],
                    "predicate": rel["predicate"],
                    "object": rel["object"],
                    "subject_name": rel["subject_name"],
                    "object_name": rel["object_name"],
                    "subject_occlusion": rel["subject_occlusion"],
                    "object_occlusion": rel["object_occlusion"],
                    "annotation": rel["annotation"],
                })
    return hits


def totals(dataset: str) -> tuple[int, int]:
    n_rel = n_view = 0
    for path in sorted(glob.glob(os.path.join(dataset, "*", "scene.json"))):
        with open(path) as fh:
            scene = json.load(fh)
        for view in scene["views"]:
            n_view += 1
            n_rel += len(view["relations"])
    return n_view, n_rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/sgg/occlusion_ds4")
    ap.add_argument("--lo", type=float, default=0.4)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--both-endpoints", action="store_true",
                    help="require both endpoints inside the window, not just one")
    ap.add_argument("--json", metavar="PATH", help="write the full listing here")
    ap.add_argument("--list", type=int, default=0, metavar="N",
                    help="print the first N hits")
    ap.add_argument("--per-scene", action="store_true",
                    help="print the per-scene breakdown")
    args = ap.parse_args()

    hits = collect(args.dataset, args.lo, args.hi, args.both_endpoints)
    n_view, n_rel = totals(args.dataset)

    which = "both endpoints" if args.both_endpoints else "at least one endpoint"
    print(f"{args.dataset}: {n_rel} relations over {n_view} views")
    print(f"{which} with {args.lo} < occlusion < {args.hi}: "
          f"{len(hits)} triplets ({100.0 * len(hits) / max(n_rel, 1):.1f}%)")

    by_pred = collections.Counter(h["predicate"] for h in hits)
    print("\nby predicate")
    for pred, n in by_pred.most_common():
        print(f"  {pred:<8} {n:>5}")

    by_scene = collections.Counter(h["scene"] for h in hits)
    print(f"\nspread over {len(by_scene)} floor plans, "
          f"{len({(h['scene_dir'], h['view']) for h in hits})} distinct views")
    if args.per_scene:
        for scene, n in sorted(by_scene.items(), key=lambda kv: -kv[1]):
            print(f"  {scene:<16} {n:>5}")

    if args.list:
        print(f"\nfirst {min(args.list, len(hits))} triplets")
        for h in hits[:args.list]:
            print(f"  {h['scene']:<14} v{h['view']:02d}  "
                  f"{h['subject']:>10} {h['predicate']:^7} {h['object']:<10}  "
                  f"occ {h['subject_occlusion']:.2f}/{h['object_occlusion']:.2f}  "
                  f"{h['image']}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "dataset": args.dataset,
                "window": [args.lo, args.hi],
                "both_endpoints": args.both_endpoints,
                "n_relations_total": n_rel,
                "n_views_total": n_view,
                "triplets": hits,
            }, fh, indent=1)
        print(f"\nwrote {len(hits)} triplets to {args.json}")


if __name__ == "__main__":
    main()
