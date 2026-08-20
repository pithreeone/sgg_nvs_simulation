"""
grounding.py -- three sheets: the cases the metric gets right, and the two ways
it gets them wrong.

    correct.png   top-1 is the annotated pair
    ranked.png    the pair IS in the shortlist, ranked below first
    missed.png    no shortlisted box is on the annotated one

GREEN the annotation, MAGENTA the top-1's subject, ORANGE its object.  The two
failure sheets need different fixes, which is the whole reason for splitting
them -- see `analysis/eval_grounding.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from viz import sheets


def sheet(rows: Sequence[Dict[str, Any]], path: str, title: str,
          columns: int = 4) -> None:
    """One sheet.  Each row carries its frame, the boxes and the numbers."""
    if not rows:
        return
    tiles = [sheets.tile(
        r["frame"],
        [f"{r['instruction']}",
         f"rank {r['rank'] or '-'}   p {r['p_subject']:.2f}/{r['p_object']:.2f}"],
        colours=[sheets.TRUTH, sheets.FAINT],
        boxes=((r["gt_subject"], sheets.TRUTH, 2),
               (r["gt_object"], sheets.TRUTH, 2),
               (r.get("top_subject"), sheets.SUBJECT, 2),
               (r.get("top_object"), sheets.OBJECT, 1)))
        for r in rows]
    sheets.sheet(tiles, path, columns, banner=title)
