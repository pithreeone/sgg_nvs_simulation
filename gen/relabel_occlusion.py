"""
relabel_occlusion.py -- re-derive an occlusion dataset's relations, in place.

A ground-truth definition change does not need the scene rebuilt.  Every input
`vg_gt` reads is already stored per view: the amodal box, the 3D position, the
camera distance, the VG150 class, and the endpoint occlusion.  So a redefinition
costs a second per scene instead of a THOR session per scene, and -- importantly
for comparing definitions -- it runs against the *same* renders and the same
predictions, so nothing but the annotation changes.

What is NOT re-derived: `on`/`in`.  Those come from `parentReceptacles`, which
the simulator asserts and which is not recoverable from the stored JSON, so they
are carried through untouched.  They are also the one family whose original
definition already agreed with VG's convention.

Run with --dry-run first; it prints the same diff without writing.
"""

from __future__ import annotations

# `python gen/<script>.py` puts gen/ on sys.path, not the repo root, so the
# root-level modules (vg150, vg_gt, ...) would not resolve.  Running as
# `python -m gen.<script>` does not need this; it is here so both work.
import os as _os
import sys as _sys
if __package__ in (None, ""):
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import collections
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from vg.vg_gt import VERTICAL_MAX_HORIZONTAL_M, support_pairs, vg_calibrated_relations

SUPPORT = "parentReceptacles"


def rederive(record: Dict[str, Any], rank_by: str, name_by: str,
             vertical_max_horizontal: Optional[float]) -> List[Dict[str, Any]]:
    """The record's views with their geometric relations replaced."""
    width = record["intrinsics"]["width"]
    height = record["intrinsics"]["height"]
    out = []
    for view in record["views"]:
        entries = {o["name"]: o for o in view["objects"]}
        occlusion = {name: o["occlusion"] for name, o in entries.items()}
        support = [r for r in view["relations"] if r["annotation"] == SUPPORT]

        # `vg_gt` keys on `object_id` and reads `bbox_xyxy`; the stable id here is
        # `name` and the box is the amodal one, same shim the builder uses.
        shim = [{"object_id": o["name"], "vg150_class": o["vg150_class"],
                 "bbox_xyxy": o["bbox_amodal"], "position": o["position"],
                 "distance": o["distance"], "loose": False}
                for o in view["objects"] if o["vg150_class"]]
        exclude = support_pairs([{**r, "subject_id": r["subject_name"],
                                  "object_id": r["object_name"]} for r in support])
        geometric = vg_calibrated_relations(
            shim, width, height, exclude, rank_by=rank_by, name_by=name_by,
            vertical_max_horizontal=vertical_max_horizontal)

        relations = list(support)
        for rel in geometric:
            subject, obj = rel["subject_id"], rel["object_id"]
            if subject not in occlusion or obj not in occlusion:
                continue
            relations.append({
                "subject": rel["subject"], "predicate": rel["predicate"],
                "object": rel["object"],
                "subject_name": subject, "object_name": obj,
                "subject_occlusion": occlusion[subject],
                "object_occlusion": occlusion[obj],
                "annotation": rel["annotation"],
            })
        out.append(relations)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--root", default="datasets/sgg/occlusion_ds2")
    parser.add_argument("--rank-by", default="human", choices=("human", "geometry"))
    parser.add_argument("--name-by", default="geometry", choices=("human", "geometry"))
    parser.add_argument("--vertical-max-horizontal", type=float,
                        default=VERTICAL_MAX_HORIZONTAL_M,
                        help="0 disables the above/under stacking gate")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    gate = args.vertical_max_horizontal or None
    before: collections.Counter = collections.Counter()
    after: collections.Counter = collections.Counter()
    paths = sorted(glob.glob(os.path.join(args.root, "*", "scene.json")))
    if not paths:
        print(f"no scenes under {args.root}/")
        return 1

    for path in paths:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        for view in record["views"]:
            for rel in view["relations"]:
                before[rel["predicate"]] += 1
        rebuilt = rederive(record, args.rank_by, args.name_by, gate)
        for view, relations in zip(record["views"], rebuilt):
            view["relations"] = relations
            for rel in relations:
                after[rel["predicate"]] += 1
        if not args.dry_run:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(record, handle)

    n_before, n_after = sum(before.values()), sum(after.values())
    print(f"{len(paths)} scenes   rank_by={args.rank_by}   name_by={args.name_by}"
          f"   vertical gate={gate}{'   DRY RUN' if args.dry_run else ''}\n")
    print(f"{'predicate':<14}{'before':>8}{'after':>8}{'delta':>8}")
    for predicate in sorted(set(before) | set(after), key=lambda p: -after[p]):
        print(f"{predicate:<14}{before[predicate]:>8}{after[predicate]:>8}"
              f"{after[predicate] - before[predicate]:>+8}")
    print(f"{'TOTAL':<14}{n_before:>8}{n_after:>8}{n_after - n_before:>+8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
