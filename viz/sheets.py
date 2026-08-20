"""
sheets.py -- a contact sheet: frames, captioned, tiled into one image.

`move_once` draws the sweep twice and `eval_move` draws the walk once per arm.
Same picture, different questions, so the tiling lives here once.

THE CAPTION IS DRAWN AFTER THE RESIZE.  Lettering at 800x600 and then scaling to
400 wide puts the strokes through the same interpolation as the picture, and a
0.44-scale font does not survive it.

Frames, boxes and strings only -- no controller, no task, no model, so the
simulator and the real robot draw the same figure.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

#: Default tile width.  Tiles keep their frame's aspect ratio unless `size` says
#: otherwise.
TILE = 400

#: One meaning per colour across every sheet in the repo, BGR.
TRUTH = (140, 255, 140)         # the instructed instance, or a heading line
SUBJECT = (230, 120, 230)       # the pair's subject endpoint
OBJECT = (90, 200, 255)         # its object endpoint
DIM = (215, 215, 215)           # a caption that is not the point of the tile
FAINT = (185, 185, 185)         # its second line

LINE = 15                       # caption line pitch
BAND = 20                       # caption height for one line
FONT_SCALE = 0.44
BANNER_HEIGHT = 30
BANNER_SCALE = 0.55


def draw_boxes(canvas: np.ndarray,
               boxes: Sequence[Tuple[Sequence[float], Tuple[int, int, int],
                                     int]]) -> np.ndarray:
    """`(box, colour, thickness)` rectangles onto a BGR canvas, in place."""
    import cv2

    for box, colour, thick in boxes:
        if box is None:
            continue
        x0, y0, x1, y1 = (int(v) for v in box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thick)
    return canvas


def as_bgr(frame) -> np.ndarray:
    """An RGB frame as a contiguous, writable BGR canvas."""
    return np.ascontiguousarray(np.asarray(frame)[:, :, ::-1]).copy()


def tile(frame, lines: Sequence[str], *,
         colours: Optional[Sequence[Tuple[int, int, int]]] = None,
         width: int = TILE, size: Optional[Tuple[int, int]] = None,
         boxes: Sequence[Tuple[Sequence[float], Tuple[int, int, int],
                               int]] = (),
         over: bool = False, bgr: bool = False) -> np.ndarray:
    """One captioned panel.  `frame` is RGB unless `bgr`, and is not modified.

    `size` forces (w, h); without it the tile is `width` wide at the frame's own
    aspect ratio.

    `over` letters the caption ONTO the image instead of into a band above it.
    Above is better -- a band over the frame covers its top 18%, which reads as a
    squashed picture rather than a cropped one -- but tiles of a fixed size have
    nowhere to put the extra rows.
    """
    import cv2

    canvas = np.asarray(frame).copy() if bgr else as_bgr(frame)
    draw_boxes(canvas, boxes)
    if size is not None:
        panel = cv2.resize(canvas, tuple(size))
    else:
        panel = cv2.resize(
            canvas, (width, int(width * canvas.shape[0] / canvas.shape[1])))

    lines = list(lines)
    colours = list(colours) if colours else (
        [TRUTH] + [FAINT] * (len(lines) - 1))
    height = BAND + (LINE - 1) * (len(lines) - 1)
    if over:
        cv2.rectangle(panel, (0, 0), (panel.shape[1], height), (20, 20, 20), -1)
        band, origin = panel, 4
    else:
        band = np.full((height, panel.shape[1], 3), 20, dtype=panel.dtype)
        origin = 5
    for k, text in enumerate(lines):
        cv2.putText(band, text, (origin, 14 + LINE * k),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, colours[k], 1,
                    cv2.LINE_AA)
    return panel if over else np.vstack([band, panel])


def row(tiles: Sequence[np.ndarray],
        columns: Optional[int] = None) -> np.ndarray:
    """Tiles side by side, padded out to `columns` cells with blank ones.

    Without `columns` the row is however long it is -- what a per-arm walk needs,
    the arms taking different numbers of steps -- and `stack` pads them instead.
    """
    cells = list(tiles)
    while columns and len(cells) < columns:
        cells.append(np.zeros_like(cells[0]))
    return np.hstack(cells)


def stack(rows: Sequence[np.ndarray], banner: Optional[str] = None,
          colour: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Rows into one image, padded to the widest, with an optional title.

    The title goes on once, across the top: every panel below answers the same
    question, and without it a sheet of frames is just a sheet of rooms.
    """
    import cv2

    rows = list(rows)
    width = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == width else
            np.hstack([r, np.zeros((r.shape[0], width - r.shape[1], 3),
                                   dtype=r.dtype)]) for r in rows]
    if banner is None:
        return np.vstack(rows)
    bar = np.full((BANNER_HEIGHT, width, 3), 32, dtype=rows[0].dtype)
    cv2.putText(bar, banner, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, BANNER_SCALE,
                colour, 1, cv2.LINE_AA)
    return np.vstack([bar, *rows])


def sheet(tiles: Sequence[np.ndarray], path: str, columns: int = 5, *,
          head: Sequence[np.ndarray] = (), banner: Optional[str] = None,
          quiet: bool = False) -> None:
    """`tiles` wrapped into `columns` and written out.

    `head` gets a row of its own above them -- where the reference frame belongs,
    since the comparison every sheet exists for is the start pose against the rest.
    """
    import cv2

    rows = [row(list(head), columns)] if len(head) else []
    rows += [row(tiles[k:k + columns], columns)
             for k in range(0, len(tiles), columns)]
    cv2.imwrite(path, stack(rows, banner))
    if not quiet:
        print(f"    -> {path}")
