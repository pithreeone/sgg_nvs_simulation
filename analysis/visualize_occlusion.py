"""
visualize_occlusion.py -- draw the annotation over the rendered views.

Two overlays per view, written side by side rather than stacked on one image,
because a living room view carries around 18 relations and drawing all of them
over the boxes is unreadable:

    *_boxes.png       one box per object, tinted by how occluded it is, with the
                      visible extent drawn inside the amodal extent
    *_relations.png   subject -> object arrows with the predicate, plus a legend

The amodal box is the outer one and the visible box the inner one, so the gap
between them IS the occlusion.  That is the annotation VG does not have: its
boxes are drawn around what the annotator could see, so a half-hidden object is
either boxed at its visible extent or skipped, and either way the relation it
takes part in goes unlabelled.
"""

from __future__ import annotations

# `python analysis/<script>.py` puts analysis/ on sys.path, not the repo root, so
# the top-level modules would not import.  Same bootstrap as gen/.
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)


import argparse
import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

#: Occlusion bands and their colours, dark to bright as the object disappears.
BANDS = [
    (0.00, (110, 200, 110), "clear"),
    (0.25, (230, 200, 70), "25%+"),
    (0.50, (240, 140, 50), "50%+"),
    (0.75, (230, 70, 70), "75%+"),
    (0.95, (170, 60, 200), "95%+"),
]


def band(occlusion: float) -> Tuple[int, int, int]:
    colour = BANDS[0][1]
    for threshold, rgb, _ in BANDS:
        if occlusion >= threshold:
            colour = rgb
    return colour


def font(size: int = 13):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def label(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str,
          colour: Tuple[int, int, int], fnt, anchor: str = "la") -> None:
    """Text on a filled plate, so it stays readable over any render."""
    x, y = xy
    box = draw.textbbox((x, y), text, font=fnt, anchor=anchor)
    draw.rectangle([box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1], fill=(0, 0, 0))
    draw.text((x, y), text, fill=colour, font=fnt, anchor=anchor)


def centre(box: Sequence[float]) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def draw_boxes(image: Image.Image, objects: Sequence[Dict[str, Any]],
               min_occlusion: float, max_occlusion: float = 1.01) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    fnt = font(13)
    for entry in sorted(objects, key=lambda o: o["occlusion"]):
        if not min_occlusion <= entry["occlusion"] <= max_occlusion:
            continue
        colour = band(entry["occlusion"])
        amodal = entry["bbox_amodal"]
        draw.rectangle(amodal, outline=colour, width=2)
        visible = entry.get("bbox_visible")
        if visible and entry["occlusion"] > 0.02:
            # Dashed, to keep the two readable where they nearly coincide.
            x1, y1, x2, y2 = visible
            for start, end, horizontal in (((x1, y1), (x2, y1), True),
                                           ((x1, y2), (x2, y2), True),
                                           ((x1, y1), (x1, y2), False),
                                           ((x2, y1), (x2, y2), False)):
                span = (end[0] - start[0]) if horizontal else (end[1] - start[1])
                steps = max(1, int(abs(span) / 6))
                for i in range(0, steps, 2):
                    a = (start[0] + (span * i / steps if horizontal else 0),
                         start[1] + (0 if horizontal else span * i / steps))
                    b = (start[0] + (span * (i + 1) / steps if horizontal else 0),
                         start[1] + (0 if horizontal else span * (i + 1) / steps))
                    draw.line([a, b], fill=colour, width=1)
        label(draw, (amodal[0] + 2, amodal[1] + 2),
              f"{entry['vg150_class']} {entry['occlusion']:.0%}", colour, fnt)
    return out


def draw_relations(image: Image.Image, view: Dict[str, Any],
                   min_occlusion: float, limit: int,
                   max_occlusion: float = 1.01) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    fnt = font(12)
    boxes = {o["name"]: o["bbox_amodal"] for o in view["objects"]
             if o["occlusion"] <= max_occlusion}

    rows = [r for r in view["relations"]
            if min_occlusion <= max(r["subject_occlusion"], r["object_occlusion"])
            <= max_occlusion
            and r["subject_name"] in boxes and r["object_name"] in boxes]
    # Most-occluded first, so the ones the dataset exists for survive the cut.
    rows.sort(key=lambda r: -max(r["subject_occlusion"], r["object_occlusion"]))
    rows = rows[:limit]

    for rel in rows:
        worst = max(rel["subject_occlusion"], rel["object_occlusion"])
        colour = band(worst)
        start = centre(boxes[rel["subject_name"]])
        end = centre(boxes[rel["object_name"]])
        draw.line([start, end], fill=colour, width=2)
        # Arrowhead at the object end.
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        for side in (-0.4, 0.4):
            draw.line([end, (end[0] - 11 * math.cos(angle + side),
                             end[1] - 11 * math.sin(angle + side))],
                      fill=colour, width=2)
        draw.ellipse([start[0] - 3, start[1] - 3, start[0] + 3, start[1] + 3],
                     fill=colour)
        mid = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        label(draw, mid,
              f"{rel['subject']} {rel['predicate']} {rel['object']}", colour, fnt,
              anchor="mm")

    header = (f"view {view['index']}  {len(view['relations'])} relations, "
              f"{len(rows)} drawn (endpoint occlusion >= {min_occlusion:.0%})")
    label(draw, (6, 6), header, (255, 255, 255), font(13))
    return out


def legend(width: int) -> Image.Image:
    strip = Image.new("RGB", (width, 26), (0, 0, 0))
    draw = ImageDraw.Draw(strip)
    fnt = font(12)
    x = 8
    draw.text((x, 6), "solid = amodal   dashed = visible   occlusion:",
              fill=(220, 220, 220), font=fnt)
    x += 250
    for _, rgb, name in BANDS:
        draw.rectangle([x, 7, x + 14, 19], fill=rgb)
        draw.text((x + 19, 6), name, fill=rgb, font=fnt)
        x += 70
    return strip


def render_record(path: str, outdir: str, min_occlusion: float,
                  limit: int, max_occlusion: float = 1.01,
                  only_views: Optional[Sequence[int]] = None) -> Tuple[int, int]:
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    src = os.path.dirname(path)
    os.makedirs(outdir, exist_ok=True)

    drawn = 0
    for view in record["views"]:
        if only_views is not None and view["index"] not in only_views:
            continue
        image = Image.open(os.path.join(src, view["image"])).convert("RGB")
        strip = legend(image.width)
        for kind, canvas in (("boxes", draw_boxes(image, view["objects"], 0.0,
                                                  max_occlusion)),
                             ("relations",
                              draw_relations(image, view, min_occlusion, limit,
                                             max_occlusion))):
            sheet = Image.new("RGB", (canvas.width, canvas.height + strip.height))
            sheet.paste(canvas, (0, 0))
            sheet.paste(strip, (0, canvas.height))
            name = f"{record['scene']}_s{record['seed']}_v{view['index']:02d}_{kind}.png"
            sheet.save(os.path.join(outdir, name))
        drawn += 1
    relations = sum(len(v["relations"]) for v in record["views"])
    return drawn, relations


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--root", default="datasets/sgg/occlusion_ds4")
    parser.add_argument("--out", default="datasets/sgg/occlusion_ds4/_viz")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="directory names to render; default all")
    parser.add_argument("--min-occlusion", type=float, default=0.25,
                        help="only draw relations with an endpoint at least this "
                             "hidden; 0 draws everything")
    parser.add_argument("--max-occlusion", type=float, default=0.80,
                        help="hide targets more hidden than this; matches the "
                             "dataset's own MAX_OCCLUSION")
    parser.add_argument("--views", type=int, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=10,
                        help="most-occluded N relations per view, to stay legible")
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.root, "*", "scene.json")))
    if args.scenes:
        wanted = set(args.scenes)
        paths = [p for p in paths if os.path.basename(os.path.dirname(p)) in wanted]
    if not paths:
        print(f"no scene.json under {args.root}/")
        return 1

    total_views = total_relations = 0
    for path in paths:
        views, relations = render_record(path, args.out, args.min_occlusion,
                                         args.limit, args.max_occlusion,
                                         args.views)
        name = os.path.basename(os.path.dirname(path))
        print(f"  {name:<22}{views:>3} views  {relations:>5} relations  "
              f"{relations / max(views, 1):>5.1f} per view")
        total_views += views
        total_relations += relations
    print(f"\n{total_views} views, {total_relations} relations, "
          f"{total_relations / max(total_views, 1):.1f} per view -> {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
