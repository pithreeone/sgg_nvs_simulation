"""
relabel_multiview.py -- reapply THOR_TO_VG150 to an existing export, in place.

A class mapping change only affects which VG150 word names each object, and that
is a pure function of `thor_type`.  So there is no need for the heavier
`refresh_multiview_gt()` path, which starts a THOR instance and teleports to every
stored pose: nothing geometric moves, no PNG is touched, and the relation set is
unchanged apart from the two endpoint strings.

Run with --dry-run first; it prints the same diff without writing.
"""

from __future__ import annotations

# `python build/sgg/<script>.py` puts build/sgg/ on sys.path, not the repo root, so the
# root-level modules (vg150, vg_gt, ...) would not resolve.  Running as
# `python -m build.sgg.<script>` does not need this; it is here so both work.
import os as _os
import sys as _sys
if __package__ in (None, ""):
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import collections
import glob
import json
import os
import sys
from typing import Optional, Sequence

from build.sgg.export_samples import LOOSE_MAPPINGS
from vg.vg150 import THOR_TO_VG150


def relabel(record: dict) -> collections.Counter:
    """Rewrite every class name in one scene record; return the changes made."""
    changes: collections.Counter = collections.Counter()
    # object_id -> new class, so relation endpoints stay consistent with the
    # objects they point at even if a view lists them in a different order.
    new_class = {}
    loose = {}

    views = [record["reference"]] + record.get("views", [])
    for view in views:
        for entry in view.get("objects", []):
            thor_type = entry.get("thor_type")
            target = THOR_TO_VG150.get(thor_type)
            if target is None:
                continue
            if entry.get("vg150_class") != target:
                changes[(entry["vg150_class"], target)] += 1
            entry["vg150_class"] = target
            entry["loose"] = thor_type in LOOSE_MAPPINGS
            new_class[entry["object_id"]] = target
            loose[entry["object_id"]] = entry["loose"]

    relation_sets = [view.get("relations") for view in views]
    relation_sets.append(record.get("scene_relations"))
    for relations in relation_sets:
        for r in relations or []:
            for role in ("subject", "object"):
                key = new_class.get(r.get(f"{role}_id"))
                if key is not None:
                    r[role] = key
            r["loose"] = bool(loose.get(r.get("subject_id"))
                              or loose.get(r.get("object_id")))
    return changes


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dir", default="datasets/sgg/multiview")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.dir, "*", "r*", "scene.json")))
    if not paths:
        print(f"nothing under {args.dir}/")
        return 1

    total: collections.Counter = collections.Counter()
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        changes = relabel(record)
        total.update(changes)
        if not args.dry_run:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=1)

    print(f"{len(paths)} scene records{'  (dry run)' if args.dry_run else ''}\n")
    if not total:
        print("no class names changed")
        return 0
    print(f"{'was':<14}{'now':<14}{'objects':>9}")
    for (old, new), count in total.most_common():
        print(f"{old:<14}{new:<14}{count:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
