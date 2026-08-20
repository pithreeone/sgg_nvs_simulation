"""
evidence.py -- channel B, computed once and read two ways.

`contrib[v][i, j]` is view v's own relation field, mapped back into the
REFERENCE slot space and read at the instructed predicate.  Everything B does is
a projection of that table:

    pool over v            -> ev[i, j]   which PAIR the sentence means
    hold (i, j) fixed      -> a score per VIEW, which is `viewpick`'s `attrib`

It was computed separately at each of those two places -- twice per pool in
`move_once`, once more inside `attrib` -- so the relationship between them was
invisible and any change to the gate had three homes.

THE GATE.  A view may speak about a pair only if BOTH endpoints have a mutual-NN
correspondence in it at cosine >= GATE_COS.  Ungated, a view seeing neither
endpoint still votes at full weight, and its `rel[i, j]` is then a fact about a
different pair of objects entirely.  A view that cannot speak reads 0 and is not
counted, which is what makes `mean` a mean over the QUALIFYING views.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

Cell = Tuple[int, int]


def pair_contributions(built, cells: Sequence[Cell], predicate: int,
                       corr: str = None, gate_cos: float = None
                       ) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """-> (view indices, contrib [V, cells], spoke [V, cells] bool).

    `cells` is the caller's own candidate set: `attrib` compares every pair with
    a distinct subject, the pair ranking applies `--pair-iou` first, and the two
    are not the same set.  The gate and the field are.
    """
    import torch

    from lib.fusion import channels as ch
    from fuse_live import CORR, GATE_COS

    corr = CORR if corr is None else corr
    gate_cos = GATE_COS if gate_cos is None else gate_cos

    nq = built["rec"]["s_ref"].shape[0]
    hn = torch.nn.functional.normalize(built["rec"]["h_ref"].float(), dim=-1)
    rows, spoke, taken = [], [], []
    for index, view in enumerate(built["rec"]["views"]):
        if int(view["v"]) in ch.SKIP_VIEWS:
            continue
        field, ok = ch._pair_field(view, nq, hn, corr, gate_cos)
        seen = [ok is None or (bool(ok[i]) and bool(ok[j])) for i, j in cells]
        rows.append([float(field[i, j, predicate]) if s else 0.0
                     for s, (i, j) in zip(seen, cells)])
        spoke.append(seen)
        taken.append(index)
    return (taken, np.array(rows, float).reshape(len(taken), len(cells)),
            np.array(spoke, bool).reshape(len(taken), len(cells)))


def pooled(cells: Sequence[Cell], contrib: np.ndarray, spoke: np.ndarray,
           how: str = "mean") -> Dict[Cell, float]:
    """`ev[i, j]`, as a dict the ranking can subscript with `ev[i, j]`.

      mean  over the views that see both endpoints.
      max   the strongest single such view -- the same shape as `rel`, which is
            itself one observation, and the pooling A already uses.  Comparable
            to 1, at the cost that one bad view can set a pair's evidence.

    A pair no view could speak about reads 0.
    """
    if not len(contrib):
        return {cell: 0.0 for cell in cells}
    if how == "max":
        value = contrib.max(0)
    else:
        count = spoke.sum(0)
        value = np.where(count > 0, contrib.sum(0) / np.maximum(count, 1), 0.0)
    return {cell: float(v) for cell, v in zip(cells, value)}


def coverage(spoke: np.ndarray) -> Dict[str, float]:
    """How much of the candidate set the sweep actually spoke about."""
    if not len(spoke):
        return {"n_views": 0, "pairs_covered": 0.0, "views_per_pair": 0.0}
    return {"n_views": int(spoke.shape[0]),
            "pairs_covered": float(spoke.any(0).mean()),
            "views_per_pair": float(spoke.sum(0).mean())}
