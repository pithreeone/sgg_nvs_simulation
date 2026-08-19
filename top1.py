"""
top1.py -- one image, one instruction, the pair EGTR would ground it to.

SELF-CONTAINED.  Copy this file anywhere next to an `sgg_nvs` checkout and it
runs: the model loading, the forward pass, the box conversion and the ranking are
all inlined, so it imports nothing from `sgg_nvs_simulation`.  The only external
requirement is the EGTR checkout itself, since that is where the network lives.

The ranking is `eval_move.look`'s and nothing else -- candidates by p(class),
ordered by `rel[i, j, predicate] * s_i * s_j`, single frame, no fusion and no
sweep.  That is why it can be pointed at a synthesised view: it answers "what
would a robot standing here ground the sentence to" for any photograph.

    python top1.py rgb_image.jpg --task box,behind,chair
    python top1.py out/view_*.png --task box,behind,chair --top 5
    SGG_ROOT=~/sgg_nvs python top1.py img.png --task box,behind,chair
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from typing import Any, Dict, Optional, Sequence

import numpy as np

#: The EGTR checkout.  Its modules import as `model.egtr`, `util.box_ops`, so its
#: root goes on sys.path.  Default assumes it sits beside this file's parent.
SGG_ROOT = os.environ.get(
    "SGG_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sgg_nvs"))

#: The finetuned EGTR, and the DETR whose feature extractor and architecture it
#: was built on.  Local checkouts: passing the HF name instead would fetch a
#: config that is already on disk.
ARTIFACT = ("ckpt/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32"
            "__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune"
            "__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002"
            "__visual_genome__finetune/version_0")
ARCHITECTURE = ("ckpt/pretrained_detr__SenseTime__deformable-detr/batch__32"
                "__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune"
                "/version_0")

#: Words VG150 splits that an instruction does not.  p(class) sums over the set.
CLASS_ALIASES: Dict[str, Sequence[str]] = {
    "laptop": ("laptop", "screen"),
}


def load_egtr(sgg_root: str = SGG_ROOT, num_queries: int = 200) -> Dict[str, Any]:
    """The model, its feature extractor and its label vocabularies.

    `assign=True` on `load_state_dict` is not optional on torch 2.6: without it
    the weights land on the meta device and every prediction is noise.
    """
    import torch

    root = os.path.abspath(os.path.expanduser(sgg_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    from model.deformable_detr import (DeformableDetrConfig,
                                       DeformableDetrFeatureExtractor)
    from model.egtr import DetrForSceneGraphGeneration

    artifact = os.path.join(root, ARTIFACT)
    with open(os.path.join(root, "obj_categories.json")) as fh:
        obj_names = {int(k): v for k, v in json.load(fh).items()}
    with open(os.path.join(root, "rel_categories.json")) as fh:
        raw = json.load(fh)
    # Stored 1-based as a dict; the model's relation index is 0-based over the
    # same order, so it is rebuilt as a list rather than indexed by the integer.
    rel_names = [raw[str(i)] for i in range(1, len(raw) + 1)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DeformableDetrFeatureExtractor.from_pretrained(
        os.path.join(root, ARCHITECTURE), size=800, max_size=1333)
    config = DeformableDetrConfig.from_pretrained(artifact)
    config.logit_adjustment, config.logit_adj_tau = False, 0.3
    config.num_rel_labels = len(rel_names)
    model = DetrForSceneGraphGeneration(config)

    checkpoints = sorted(
        glob(os.path.join(artifact, "checkpoints", "epoch=*.ckpt")),
        key=lambda p: int(p.split("epoch=")[1].split("-")[0]))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint under {artifact}")
    state = torch.load(checkpoints[-1], map_location="cpu")["state_dict"]
    model.load_state_dict({k[6:]: v for k, v in state.items()}, assign=True)
    model.to(device).eval()
    print(f"EGTR on {device}  {os.path.basename(checkpoints[-1])}")
    return {"model": model, "fe": extractor, "device": device, "torch": torch,
            "obj_names": obj_names, "rel_names": rel_names,
            "num_labels": len(obj_names), "num_queries": num_queries}


def predict(egtr, frame: np.ndarray):
    """-> (probs [Q, C] softmax, boxes [Q, 4] xyxy in pixels, rel [Q, Q, R])."""
    from PIL import Image

    torch = egtr["torch"]
    image = Image.fromarray(frame)
    inputs = egtr["fe"](images=image, return_tensors="pt").to(egtr["device"])
    with torch.no_grad():
        out = egtr["model"](**inputs, output_attentions=False,
                            output_attention_states=True,
                            output_hidden_states=True)

    rel = torch.clamp(out.pred_rel[0], 0.0, 1.0)
    if out.pred_connectivity is not None:
        rel = torch.mul(rel, torch.clamp(out.pred_connectivity[0], 0.0, 1.0))

    # cxcywh in [0, 1] -> xyxy in pixels; `util.box_ops.rescale_bboxes` inlined.
    cx, cy, w, h = out.pred_boxes[0].cpu().unbind(-1)
    boxes = torch.stack([cx - 0.5 * w, cy - 0.5 * h,
                         cx + 0.5 * w, cy + 0.5 * h], dim=-1)
    width, height = image.size
    boxes = boxes * torch.tensor([width, height, width, height],
                                 dtype=torch.float32)
    probs = out.logits[0].softmax(-1)[:, :egtr["num_labels"]].cpu()
    return probs, boxes, rel.cpu()


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    overlap = (x1 - x0) * (y1 - y0)
    areas = [max(0.0, q[2] - q[0]) * max(0.0, q[3] - q[1]) for q in (a, b)]
    union = areas[0] + areas[1] - overlap
    return overlap / union if union > 0 else 0.0


def rank(egtr, frame, subject: str, predicate: str, obj: str,
         width: int = 10, pair_iou: float = 0.15):
    """-> ([(score, i, j)] best first, boxes, p_subject [Q], p_object [Q])."""
    probs, boxes, rel = predict(egtr, frame)
    s = probs.max(-1).values
    column = {v: k - 1 for k, v in egtr["obj_names"].items()}

    def mass(name):
        cols = [column[c] for c in CLASS_ALIASES.get(name, (name,))
                if c in column]
        if not cols:
            raise SystemExit(f"'{name}' is not a VG150 class")
        return probs[:, cols].sum(-1)

    if predicate not in egtr["rel_names"]:
        raise SystemExit(f"'{predicate}' is not a VG150 predicate")
    pred = egtr["rel_names"].index(predicate)
    p_sub, p_obj = mass(subject), mass(obj)
    subs = p_sub.argsort(descending=True)[:width].tolist()
    objs = p_obj.argsort(descending=True)[:width].tolist()

    # A RELATION NEEDS TWO OBJECTS.  `i != j` only excludes the same query, and
    # EGTR emits several boxes per object, so a pair can be one object related to
    # itself.  The cut is low because the boxes NEST -- a label inside the carton
    # it is printed on shares all of the smaller box, but IoU divides by the
    # union, so such a pair reads only ~0.23.
    order = sorted(((float(rel[i, j, pred]) * float(s[i]) * float(s[j]), i, j)
                    for i in subs for j in objs
                    if i != j and (not pair_iou
                                   or iou(boxes[i].tolist(), boxes[j].tolist())
                                   < pair_iou)), reverse=True)
    return order, boxes, p_sub, p_obj


def draw(frame, boxes, pair, path: str, label: str) -> None:
    """MAGENTA the subject, ORANGE the object."""
    import cv2

    canvas = np.ascontiguousarray(frame[:, :, ::-1]).copy()
    for q, colour, thick in ((pair[0], (230, 120, 230), 3),
                             (pair[1], (90, 200, 255), 2)):
        x0, y0, x1, y1 = (int(v) for v in boxes[q].tolist())
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thick)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(canvas, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, canvas)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--task", required=True, metavar="SUBJ,PRED,OBJ",
                    help="VG150 words, e.g. box,behind,chair")
    ap.add_argument("--sgg-root", default=SGG_ROOT)
    ap.add_argument("--width", type=int, default=10,
                    help="candidates per side, by p(class)")
    ap.add_argument("--pair-iou", type=float, default=0.15,
                    help="reject a pair that is one object twice.  0 disables.")
    ap.add_argument("--top", type=int, default=1,
                    help="print this many pairs; the first is the one drawn")
    ap.add_argument("--suffix", default="_top1",
                    help="written beside each input as <name><suffix>.png")
    args = ap.parse_args(argv)

    import cv2

    subject, predicate, obj = (w.strip() for w in args.task.split(","))
    egtr = load_egtr(args.sgg_root)

    for path in args.images:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"\n{path}\n  ! cannot read")
            continue
        frame = bgr[:, :, ::-1].copy()
        order, boxes, p_sub, p_obj = rank(egtr, frame, subject, predicate, obj,
                                          args.width, args.pair_iou)
        print(f"\n{os.path.basename(path)}")
        if not order:
            print("  no distinct pair")
            continue
        for n, (_, i, j) in enumerate(order[:max(args.top, 1)], 1):
            print(f"  {n}. {subject} {[round(v, 1) for v in boxes[i].tolist()]}"
                  f" p={float(p_sub[i]):.3f}"
                  f"   {obj} {[round(v, 1) for v in boxes[j].tolist()]}"
                  f" p={float(p_obj[j]):.3f}")
        _, i, j = order[0]
        out = os.path.splitext(path)[0] + args.suffix + ".png"
        draw(frame, boxes, (i, j), out,
             f"{subject} {predicate} {obj}    "
             f"p({subject})={float(p_sub[i]):.3f}  "
             f"p({obj})={float(p_obj[j]):.3f}")
        print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
