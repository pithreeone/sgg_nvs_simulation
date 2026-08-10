"""
sgg_live.py -- drive the robot and run EGTR on what it sees.

The scene comes from `drive.py` (stock THOR, no clutter, no ground truth) and the
scene graph comes from `../sgg_nvs`, loaded once and kept on the GPU.  Press
SPACE and the current frame goes through EGTR; the top-K triplets are drawn on
it.  Nothing here is scored against anything -- what you see is what the network
said.

The ranking is EGTR's own, copied from `inference_egtr.py` rather than
reinvented: `pred_rel.max(-1) * s_subject * s_object`, self-pairs zeroed, sorted
descending.  That product, not the bare relation logit, is what the SGDet
evaluator ranks by, so the top-10 shown here are the top-10 that would be
scored.

    python sgg_live.py                                  # FloorPlan203
    python sgg_live.py --scene FloorPlan1 --topk 10
    python sgg_live.py --auto                           # infer after every move
    python sgg_live.py --image driveable/fp203/start.png   # one image, no THOR

Keys
    w / s   forward / back      a / d   turn        q / e   strafe
    r / f   look up / down      c       crouch/stand
    SPACE   run EGTR here       p       save png + json      ESC/x   quit
"""

from __future__ import annotations

# `python robot/<script>.py` puts robot/ on sys.path, not the repo root, so the
# top-level packages would not import.  Same bootstrap as gen/.
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)


import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: The SGG project is a sibling checkout of the REPO, not of this package -- its
#: modules import as `model.egtr`, `util.box_ops`, `lib.pytorch_misc`, so its
#: root has to be on sys.path.  No name collides with this repo (`gen`, `vg`,
#: `robot`, ...).  Two levels up because this file lives in `robot/`; it moved
#: there on 2026-08-10 and the missing level was the only breakage.
SGG_ROOT = os.environ.get(
    "SGG_ROOT", os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "..", "sgg_nvs"))

#: The finetuned EGTR, and the DETR whose feature extractor and architecture it
#: was built on.  Both are local checkouts: passing the HF name
#: "SenseTime/deformable-detr" instead would go to the network for a config that
#: is already on disk.
ARTIFACT = ("ckpt/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32"
            "__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune"
            "__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002"
            "__visual_genome__finetune/version_0")
ARCHITECTURE = ("ckpt/pretrained_detr__SenseTime__deformable-detr/batch__32"
                "__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune"
                "/version_0")


def load_egtr(sgg_root: str = SGG_ROOT, artifact: str = ARTIFACT,
              architecture: str = ARCHITECTURE, num_queries: int = 200
              ) -> Dict[str, Any]:
    """
    Load the model once.  Returns everything `predict` needs.

    `assign=True` on `load_state_dict` is not optional on torch 2.6 -- see the
    comment in `inference_egtr.py`; without it the weights land on the meta
    device and every prediction is noise.
    """
    from glob import glob

    import torch

    sys.path.insert(0, os.path.abspath(sgg_root))
    from model.deformable_detr import (DeformableDetrConfig,
                                       DeformableDetrFeatureExtractor)
    from model.egtr import DetrForSceneGraphGeneration

    root = os.path.abspath(sgg_root)
    artifact_path = os.path.join(root, artifact)
    architecture_path = os.path.join(root, architecture)

    with open(os.path.join(root, "obj_categories.json")) as fh:
        obj_names = {int(k): v for k, v in json.load(fh).items()}
    with open(os.path.join(root, "rel_categories.json")) as fh:
        raw = json.load(fh)
    # Stored 1-based as a dict; the model's relation index is 0-based over the
    # same order (VG's `rel_categories` minus `__background__`), so it has to be
    # rebuilt as a list rather than indexed by the integer.
    rel_names = [raw[str(i)] for i in range(1, len(raw) + 1)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        architecture_path, size=800, max_size=1333)
    config = DeformableDetrConfig.from_pretrained(artifact_path)
    config.logit_adjustment, config.logit_adj_tau = False, 0.3
    config.num_rel_labels = len(rel_names)
    model = DetrForSceneGraphGeneration(config)

    checkpoints = sorted(
        glob(os.path.join(artifact_path, "checkpoints", "epoch=*.ckpt")),
        key=lambda p: int(p.split("epoch=")[1].split("-")[0]))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint under {artifact_path}")
    state = torch.load(checkpoints[-1], map_location="cpu")["state_dict"]
    model.load_state_dict({k[6:]: v for k, v in state.items()}, assign=True)
    model.to(device).eval()
    print(f"EGTR on {device}"
          f"{' (' + __import__('torch').cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}"
          f"  {os.path.basename(checkpoints[-1])}")
    return {"model": model, "fe": feature_extractor, "device": device,
            "obj_names": obj_names, "rel_names": rel_names,
            "num_queries": num_queries, "num_labels": len(obj_names),
            "torch": torch}


def raw_predict(state: Dict[str, Any], frame: np.ndarray) -> Dict[str, Any]:
    """
    Everything EGTR produced for one frame, before any ranking is imposed.

    `predict` is one view of this -- the top-K triplets -- and it throws away
    what multi-view fusion needs: the full `num_queries x num_queries x 50`
    relation tensor, and the per-query class distribution rather than its
    argmax.  Both live behind one forward pass here so the two callers cannot
    drift apart.

    Both probability normalisations are returned because the reference fusion
    (`../sgg_nvs/inference_semantic_anchor_fused_egtr.py`) uses them for
    different jobs -- sigmoid for object confidence, softmax for the SGG
    ranking -- and `predict`'s own ranking has always been the softmax one.
    """
    from PIL import Image

    from util.box_ops import rescale_bboxes

    torch = state["torch"]
    image = Image.fromarray(frame)
    inputs = state["fe"](images=image, return_tensors="pt").to(state["device"])
    with torch.no_grad():
        outputs = state["model"](**inputs, output_attentions=False,
                                 output_attention_states=True,
                                 output_hidden_states=True)

    logits = outputs.logits[0]
    rel = torch.clamp(outputs.pred_rel[0], 0.0, 1.0)
    if outputs.pred_connectivity is not None:
        rel = torch.mul(rel, torch.clamp(outputs.pred_connectivity[0], 0.0, 1.0))
    return {
        "probs_softmax": logits.softmax(-1)[:, :state["num_labels"]],
        "probs_sigmoid": logits.sigmoid()[:, :state["num_labels"]],
        # The decoder's own object representation, one vector per query.  The
        # cross-view correspondence in `fuse_live.py` is cosine similarity of
        # these, which is what `../sgg_nvs/scratch/mapback_cache.py` matches on.
        "query": outputs.last_hidden_state[0],
        "boxes": rescale_bboxes(outputs.pred_boxes[0].cpu(),
                                torch.tensor(image.size)),
        "rel": rel,
        "size": image.size,
    }


def predict(state: Dict[str, Any], frame: np.ndarray, topk: int = 10,
            obj_threshold: float = 0.1) -> List[Dict[str, Any]]:
    """Top-`topk` triplets for one RGB frame, EGTR's own ranking."""
    from lib.pytorch_misc import argsort_desc

    torch = state["torch"]
    raw = raw_predict(state, frame)
    obj_scores, classes = torch.max(raw["probs_softmax"], -1)
    boxes, rel = raw["boxes"], raw["rel"]

    n = state["num_queries"]
    pair_scores = torch.outer(obj_scores, obj_scores)
    pair_scores[torch.arange(n), torch.arange(n)] = 0.0
    triplet_scores = torch.mul(rel.max(-1)[0], pair_scores)
    order = argsort_desc(triplet_scores.cpu().numpy())[:topk, :]

    out = []
    for sub_q, obj_q in order:
        sub_q, obj_q = int(sub_q), int(obj_q)
        predicate = int(rel[sub_q, obj_q].argmax().item())
        out.append({
            "subject": state["obj_names"].get(int(classes[sub_q]) + 1,
                                              f"obj_{int(classes[sub_q])}"),
            "predicate": state["rel_names"][predicate],
            "object": state["obj_names"].get(int(classes[obj_q]) + 1,
                                             f"obj_{int(classes[obj_q])}"),
            "score": float(triplet_scores[sub_q, obj_q]),
            "subject_score": float(obj_scores[sub_q]),
            "object_score": float(obj_scores[obj_q]),
            "subject_box": [float(v) for v in boxes[sub_q].tolist()],
            "object_box": [float(v) for v in boxes[obj_q].tolist()],
            "subject_query": sub_q, "object_query": obj_q,
        })
    return out


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

#: One colour per triplet rank, cycled.  BGR.
COLOURS = [(90, 190, 255), (150, 255, 150), (255, 170, 120), (170, 150, 255),
           (120, 235, 235), (235, 150, 235), (150, 200, 100), (100, 160, 255),
           (200, 200, 120), (160, 255, 200)]


def overlay(frame: np.ndarray, triplets: Sequence[Dict[str, Any]],
            stale: bool = False, elapsed: Optional[float] = None) -> np.ndarray:
    """Frame with the triplets drawn, plus a panel listing them in rank order."""
    import cv2

    canvas = np.ascontiguousarray(frame[:, :, ::-1])        # RGB -> BGR
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.44

    def tag(text: str, x: int, y: int, colour) -> None:
        (w, h), _ = cv2.getTextSize(text, font, scale, 1)
        x = int(np.clip(x, 0, canvas.shape[1] - w - 8))
        y = int(np.clip(y, h + 8, canvas.shape[0] - 2))
        cv2.rectangle(canvas, (x, y - h - 7), (x + w + 7, y + 2), (25, 25, 25), -1)
        cv2.putText(canvas, text, (x + 4, y - 3), font, scale, colour, 1,
                    cv2.LINE_AA)

    if not stale:
        # One box per QUERY, not per triplet.  EGTR reuses the same detection
        # across its top-10 -- nine of ten here name the same `table` -- so
        # drawing per triplet stacked ten rectangles on one object and buried the
        # frame.  The colour is that of the query's best-ranked triplet.
        seen: Dict[int, Dict[str, Any]] = {}
        for rank, rel in enumerate(triplets):
            for role in ("subject", "object"):
                query = rel[f"{role}_query"]
                if query not in seen:
                    seen[query] = {"box": rel[f"{role}_box"],
                                   "label": rel[role], "rank": rank}
        for query, entry in seen.items():
            colour = COLOURS[entry["rank"] % len(COLOURS)]
            x0, y0, x1, y1 = (int(v) for v in entry["box"])
            cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)

        for rank, rel in enumerate(triplets):
            colour = COLOURS[rank % len(COLOURS)]
            sx = (rel["subject_box"][0] + rel["subject_box"][2]) / 2
            sy = (rel["subject_box"][1] + rel["subject_box"][3]) / 2
            ox = (rel["object_box"][0] + rel["object_box"][2]) / 2
            oy = (rel["object_box"][1] + rel["object_box"][3]) / 2
            cv2.arrowedLine(canvas, (int(sx), int(sy)), (int(ox), int(oy)),
                            colour, 1, cv2.LINE_AA, tipLength=0.03)
            # Labels stagger along the arrow by rank so two triplets sharing a
            # pair of objects do not write on top of each other.
            t = 0.35 + 0.3 * ((rank % 3) / 2.0)
            tag(f"{rank + 1} {rel['predicate']}", int(sx + (ox - sx) * t),
                int(sy + (oy - sy) * t), colour)

        for query, entry in seen.items():
            colour = COLOURS[entry["rank"] % len(COLOURS)]
            tag(entry["label"], int(entry["box"][0]), int(entry["box"][1]) - 4,
                colour)

    panel = np.zeros((canvas.shape[0], 430, 3), dtype=np.uint8)
    header = (f"EGTR top-{len(triplets)}"
              + (f"  {elapsed * 1000:.0f} ms" if elapsed else ""))
    if stale:
        header = "moved -- press SPACE to run EGTR"
    cv2.putText(panel, header, (10, 26), font, 0.46,
                (120, 170, 255) if stale else (120, 255, 170), 1, cv2.LINE_AA)
    y = 56
    for rank, rel in enumerate(triplets if not stale else []):
        colour = COLOURS[rank % len(COLOURS)]
        cv2.putText(panel, f"{rank + 1:2d}. {rel['subject']} "
                           f"{rel['predicate']} {rel['object']}",
                    (10, y), font, scale, colour, 1, cv2.LINE_AA)
        cv2.putText(panel, f"     {rel['score']:.3f}  "
                           f"(obj {rel['subject_score']:.2f}/"
                           f"{rel['object_score']:.2f})",
                    (10, y + 15), font, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
        y += 34
    for index, text in enumerate(
            ["w/s drive  a/d turn  q/e strafe", "r/f look  c crouch",
             "SPACE run EGTR  p save  ESC quit"]):
        cv2.putText(panel, text, (10, canvas.shape[0] - 46 + index * 16), font,
                    0.4, (140, 140, 140), 1, cv2.LINE_AA)
    return np.hstack([canvas, panel])


# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scene", default="FloorPlan203")
    ap.add_argument("--start", metavar="X,Z,YAW,HORIZON",
                    help="write it as --start=-4.75,-1.75,20,45")
    ap.add_argument("--image", help="run on this image and exit, no THOR")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--auto", action="store_true",
                    help="run EGTR after every move, not only on SPACE")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=None, metavar="M")
    ap.add_argument("--turn", type=float, default=None, metavar="DEG")
    ap.add_argument("--look", type=float, default=None, metavar="DEG")
    ap.add_argument("--sgg-root", default=SGG_ROOT)
    ap.add_argument("--out", default="sgg_live", help="where p writes")
    args = ap.parse_args(argv)

    import cv2

    from robot import drive

    state = load_egtr(args.sgg_root)

    def run(frame: np.ndarray):
        started = time.time()
        triplets = predict(state, frame, args.topk)
        elapsed = time.time() - started
        for rank, rel in enumerate(triplets, 1):
            print(f"  {rank:2d}. {rel['subject']:>12} {rel['predicate']:^12} "
                  f"{rel['object']:<12} {rel['score']:.3f}")
        return triplets, elapsed

    def store(frame: np.ndarray, triplets, elapsed, index: int) -> None:
        os.makedirs(args.out, exist_ok=True)
        png = os.path.join(args.out, f"sgg_{index:02d}.png")
        cv2.imwrite(png, overlay(frame, triplets, False, elapsed))
        with open(os.path.join(args.out, f"sgg_{index:02d}.json"), "w") as fh:
            json.dump({"scene": args.scene, "topk": args.topk,
                       "triplets": triplets}, fh, indent=1)
        print(f"wrote {png}")

    if args.image:
        from PIL import Image

        frame = np.array(Image.open(args.image).convert("RGB"))
        triplets, elapsed = run(frame)
        store(frame, triplets, elapsed, 0)
        return 0

    rc = drive.open_scene(args.scene, args.width, args.height, args.fov,
                          args.start,
                          step=args.step or drive.STEP_M,
                          turn=args.turn or drive.TURN_DEG)
    look = args.look or drive.LOOK_DEG
    try:
        triplets, elapsed = run(rc.event.frame)
        stale, saves = False, 0
        while True:
            cv2.imshow("sgg_live", overlay(rc.event.frame, triplets, stale,
                                           elapsed))
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("x")):
                break
            moved = True
            if key == ord("w"):
                rc.move_ahead()
            elif key == ord("s"):
                rc.move_back()
            elif key == ord("a"):
                rc.rotate(-rc.rotate_step)
            elif key == ord("d"):
                rc.rotate(rc.rotate_step)
            elif key == ord("q"):
                rc._step(action="MoveLeft", moveMagnitude=rc.move_magnitude)
            elif key == ord("e"):
                rc._step(action="MoveRight", moveMagnitude=rc.move_magnitude)
            elif key == ord("r"):
                rc.look(-look)
            elif key == ord("f"):
                rc.look(look)
            elif key == ord("c"):
                rc.set_height_level(
                    "crouch" if rc.height_level == "stand" else "stand")
            elif key == ord(" "):
                print(drive.pose_line(rc))
                triplets, elapsed = run(rc.event.frame)
                moved, stale = False, False
            elif key == ord("p"):
                store(rc.event.frame, triplets, elapsed, saves)
                saves += 1
                moved = False
            else:
                moved = False
            if moved:
                print(drive.pose_line(rc))
                if args.auto:
                    triplets, elapsed = run(rc.event.frame)
                    stale = False
                else:
                    stale = True
        cv2.destroyAllWindows()
        return 0
    finally:
        rc.stop()


if __name__ == "__main__":
    raise SystemExit(main())
