"""
vg_pair_prior.py -- P(predicate | subject class, object class), measured on VG.

`vg_conventions.py` measured what a box PAIR looks like for each predicate, and
`vg_gt.py` used those fitted distributions to decide which predicate a THOR pair
carries.  That step does not work, and the test is direct: run the same fitted
model back over VG's own held-out annotations and ask it to recover the human's
label.

    predicate      n      argmax acc   always-say-`on`
    on          11911         73.9%
    in           1739          5.9%
    near         1893          8.2%
    above         522         36.0%
    under         231         67.1%
    behind        517         28.2%
    in front of   498          6.4%
    OVERALL     17311         55.3%             68.8%

55.3% against a 68.8% majority baseline: the 2D box features are *below* a
constant.  That is not a tuning problem.  The fitted Gaussians are nearly
identical across predicates -- `above` has dy mu -0.175 sd 0.196, `behind` -0.121
sd 0.203, `near` -0.021 sd 0.220, and `cover` sits at 0.37-0.51 for all of them --
so the family competition in `vg_gt._proposals` was decided by an overlap of
about 90% between the candidate distributions, i.e. very close to a coin flip.

The signal it was missing is the class pair.  The SAME test, using only
P(predicate | subject class, object class) estimated on train and applied to val:

    class-pair argmax   80.8%          geometry argmax   55.3%

What a human calls a pair is mostly a fact about what the two things ARE, not
about where their boxes sit.  `chair`/`table` is annotated 805 times in VG and
`near` 48% of the time; `bowl`/`table` 458 times and `on` 78%.  Geometry decides
whether a relation is TRUE -- it stays in `vg_gt` as the 3D gate -- but it should
never have been deciding which true relation a human would have written down.

This is measured from human annotations, exactly like `vg_prior.json`, so it does
not reintroduce the circularity `vg_gt`'s docstring warns about.  It was found by
looking at where EGTR disagreed with our ground truth, but every substitution it
licenses is then checked against VG's counts and kept only if humans make it too.
The two agree: EGTR calls our `behind chair-table` pairs `near`, and so do 48% of
VG's annotators; it calls our `above bowl-table` pairs `on`, and so do 78%.

Two uses, kept separate because they are different claims:

  relatedness()   how often humans relate these two classes AT ALL.  `vg_gt`
                  ranks by this to decide which 10% of available pairs to
                  annotate.  TASKS.md concluded that which pairs an annotator
                  remarks on is "attention -- not recoverable from geometry",
                  which is right; it is recoverable from class co-occurrence.
  accept_set()    which wordings a human describing THESE two classes would
                  plausibly have used, so the matcher does not score a model
                  wrong for writing `near` where 48% of humans also write `near`.
                  Widening the accept-set raises recall on its own -- a random
                  accept-set of the same size scores 22.4% against this one's
                  27.2% -- so the honest reading is that roughly half of that
                  particular gain is permissiveness and half is convention.
                  Report exact match alongside it.

Build with `python vg_pair_prior.py --build`; the result is committed as
`vg_pair_prior.json` so nothing downstream needs the VG files present.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Set

PRIOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "vg_pair_prior.json")

#: Class pairs seen fewer times than this are dropped when building.  They cannot
#: support an estimate and they are most of the file.
MIN_PAIR_COUNT = 2

#: A substitution is licensed only from REAL observations of that class pair, not
#: from the smoothed backoff -- otherwise an unseen pair falls back to the
#: marginal, where `on` is 28.5%, and `on` becomes acceptable for everything.
MIN_EVIDENCE = 10

#: Accept wording `q` for ground-truth predicate `p` when humans use `q` for this
#: class pair at least this often relative to how often they use `p`.  At 1.0 the
#: rule reads: "a human describing these two classes was at least as likely to
#: write q as to write p".  Measured to be insensitive between 0.5 and 4.0.
DEFAULT_TAU = 1.0

#: Wordings that mean the same spatial fact.  These are predicate-level and
#: class-independent, unlike the class-conditioned substitutions above: `sitting
#: on` and `on` are the same claim about any pair of objects.
SYNONYMS: Dict[str, Set[str]] = {
    "on": {"on", "sitting on", "standing on", "lying on", "laying on",
           "attached to", "mounted on", "growing on", "painted on", "parked on"},
    "in": {"in", "inside"},
    "above": {"above", "over"},
    "under": {"under", "below", "beneath", "underneath"},
    "near": {"near", "next to", "beside", "by", "at", "along"},
    "behind": {"behind"},
    "in front of": {"in front of"},
}

#: Logical inverses: the same fact stated with the arguments swapped.  EGTR has no
#: passive voice, so a relation read the other way round comes back as its inverse
#: rather than as a subject/object swap.
#:
#: `on` -> `under` was missing.  It is the same kind of inversion as
#: `above` -> `under`, and it costs real matches: over 150 views the model states
#: `under(host, thing)` for 79 of our 391 `on` relations and was scored wrong for
#: all of them.  `has` is included because VG humans do the same thing -- measured
#: over pairs labelled `on(s,o)`, the reverse ordering carries `has` 34% of the
#: time and `on` only 14%, so "table has plate" is the MORE common human wording
#: for "plate on table".
INVERSES: Dict[str, Set[str]] = {
    "above": {"under"},
    "under": {"above"},
    "behind": {"in front of"},
    "in front of": {"behind"},
    "near": {"near"},
    "on": {"under", "has"},
    "in": {"has"},
}

_DATA: Optional[Dict] = None


def data() -> Dict:
    global _DATA
    if _DATA is None:
        with open(PRIOR_PATH, encoding="utf-8") as handle:
            _DATA = json.load(handle)
    return _DATA


def counts(subject: str, obj: str) -> Dict[str, int]:
    """Human predicate counts for this ORDERED class pair."""
    return data()["pair"].get(f"{subject}|{obj}", {})


def relatedness(subject: str, obj: str) -> int:
    """
    How often humans relate these two classes at all, either ordering.

    This is the annotation-rate signal.  A `chair`/`table` pair is related 1031
    times in VG train and a `pillow`/`box` pair never, so when only 10% of the
    available pairs in a view may be annotated, the first is the one a human
    would have picked.
    """
    return sum(counts(subject, obj).values()) + sum(counts(obj, subject).values())


def p_predicate(predicate: str, subject: str, obj: str, alpha: float = 5.0) -> float:
    """P(predicate | classes), smoothed toward VG's overall predicate marginal."""
    d = counts(subject, obj)
    n = sum(d.values())
    marginal = data()["marginal"]
    base = marginal.get(predicate, 1) / max(sum(marginal.values()), 1)
    return (d.get(predicate, 0) + alpha * base) / (n + alpha)


def expand(predicate: str) -> Set[str]:
    """`predicate` plus the wordings that mean the same thing for any pair."""
    return set(SYNONYMS.get(predicate, {predicate}))


def accept_set(predicate: str, subject: str, obj: str,
               tau: float = DEFAULT_TAU,
               min_evidence: int = MIN_EVIDENCE,
               targets: Sequence[str] = tuple(SYNONYMS)) -> Set[str]:
    """
    Wordings a human describing THESE two classes might have used instead.

    Returns `expand(predicate)` unchanged when VG has too little evidence for the
    pair, so the rule can only ever widen on measured grounds.
    """
    out = expand(predicate)
    d = counts(subject, obj)
    n = sum(d.values())
    if n < min_evidence:
        return out
    own = max(d.get(predicate, 0) / n, 1.0 / n)
    for other in targets:
        if other != predicate and d.get(other, 0) / n >= tau * own:
            out |= expand(other)
    return out


def inverse_accept_set(predicate: str, subject: str, obj: str,
                       **kwargs) -> Set[str]:
    """Wordings that state the same fact with the arguments swapped."""
    out: Set[str] = set()
    for other in INVERSES.get(predicate, set()):
        out |= accept_set(other, obj, subject, **kwargs) if other in SYNONYMS else {other}
    return out


# --------------------------------------------------------------------------
# Building


def build(root: str, split: str = "train") -> Dict:
    with open(os.path.join(root, "rel.json"), encoding="utf-8") as handle:
        rel = json.load(handle)
    categories = rel["rel_categories"]
    with open(os.path.join(root, f"{split}.json"), encoding="utf-8") as handle:
        coco = json.load(handle)
    names = {c["id"]: c["name"] for c in coco["categories"]}

    # File order is what rel.json's indices refer to, so it must not be sorted.
    by_image = collections.defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)

    pair = collections.defaultdict(collections.Counter)
    marginal: collections.Counter = collections.Counter()
    for image_id, triplets in rel[split].items():
        anns = by_image.get(int(image_id))
        if not anns:
            continue
        for sub_idx, obj_idx, pred_id in triplets:
            if sub_idx >= len(anns) or obj_idx >= len(anns):
                continue
            subject = names.get(anns[sub_idx]["category_id"], "?")
            obj = names.get(anns[obj_idx]["category_id"], "?")
            pair[f"{subject}|{obj}"][categories[pred_id]] += 1
            marginal[categories[pred_id]] += 1

    kept = {k: dict(v) for k, v in pair.items() if sum(v.values()) >= MIN_PAIR_COUNT}
    return {"split": split, "pair": kept, "marginal": dict(marginal)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--root", default="/home/pithreeone/dataset/visual_genome")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--show", nargs=2, metavar=("SUBJECT", "OBJECT"),
                        help="print the human predicate distribution for a pair")
    args = parser.parse_args(argv)

    if args.build:
        out = build(args.root)
        with open(PRIOR_PATH, "w", encoding="utf-8") as handle:
            json.dump(out, handle)
        print(f"wrote {PRIOR_PATH}: {len(out['pair'])} class pairs, "
              f"{sum(out['marginal'].values())} relations")
        return 0

    if args.show:
        subject, obj = args.show
        for a, b in ((subject, obj), (obj, subject)):
            d = counts(a, b)
            total = sum(d.values())
            top = ", ".join(f"{k}:{v/total*100:.0f}%"
                            for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:6])
            print(f"  {a} ? {b}   n={total:<6} {top or '-'}")
        print(f"  relatedness={relatedness(subject, obj)}")
        print(f"  accept_set(on) = {sorted(accept_set('on', subject, obj))}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
