"""
sweep.py -- the two figures a synthesised sweep gets.

    sweep.png       the frames as they came out of the synthesiser
    sweep_pred.png  the same frames, each with the pair IT would ground the
                    instruction to

No detector and no fusion here: `answer_sheet` draws exactly what the rows say
(`move_once.per_view_answers` computes them), so the picture cannot show
anything the record does not.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from viz import sheets


def contact_sheet(reference, frames: Sequence, poses: Sequence[Dict[str, Any]],
                  path: str, columns: int = 5) -> None:
    """The sweep as one image, each view captioned with the pose that made it.

    THE REFERENCE GOES IN FIRST.  The question is what the model INVENTED, which
    is a difference -- twenty synthesised views cannot be judged without the
    photograph they came from beside them.
    """
    tiles = [sheets.tile(frame, [label], colours=[colour])
             for frame, label, colour in
             [(reference, "REFERENCE (the robot's own frame)", sheets.TRUTH)]
             + [(frame, f"az {float(pose['azimuth']):+.0f}  "
                        f"el {float(pose['elevation']):+.0f}", sheets.DIM)
                for frame, pose in zip(frames, poses)]]
    sheets.sheet(tiles, path, columns)


def answer_sheet(reference, rendered: Sequence[Dict[str, Any]],
                 rows: Sequence[Dict[str, Any]],
                 reference_row: Optional[Dict[str, Any]],
                 subject: str, obj: str, path: str,
                 columns: int = 5) -> None:
    """MAGENTA the subject each frame picked, ORANGE its object endpoint.

    THE REFERENCE GETS ITS OWN ROW: the comparison is the start pose -- what the
    robot can answer WITHOUT moving -- against everything else.
    """
    def panel(frame, row, line1, dim):
        return sheets.tile(
            frame, [line1,
                    "no candidate" if not row["own_p"] else
                    f"p({subject})={row['own_p'][0]:.3f}  "
                    f"p({obj})={row['own_p'][1]:.3f}"],
            colours=[sheets.DIM if dim else sheets.TRUTH, sheets.FAINT],
            boxes=((row["own_subject_box"], sheets.SUBJECT, 3),
                   (row["own_object_box"], sheets.OBJECT, 2)))

    head = [] if reference is None or reference_row is None else [
        panel(reference, reference_row,
              "REFERENCE  (before moving, this frame alone)", False)]
    tiles = [panel(rendered[row["v"]]["frame"], row,
                   f"v{row['v']:02d}  az {row['azimuth']:+.0f}  "
                   f"el {row['elevation']:+.0f}"
                   + ("  (duplicate of reference)" if row.get("duplicate")
                      else ""), True)
             for row in rows]
    if not tiles and not head:
        return
    sheets.sheet(tiles, path, columns, head=head)
