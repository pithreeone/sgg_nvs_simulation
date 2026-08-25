"""
top1.py -- one image, one instruction, the pair EGTR would ground it to.

SELF-CONTAINED.  Copy this file anywhere next to an `sgg_nvs` checkout and it
runs: the model loading, the forward pass, the box conversion and the ranking are
all inlined, so it imports nothing from `sgg_nvs_simulation`.  The only external
requirement is the EGTR checkout itself, since that is where the network lives.

The ranking is `robot/grounding.py:single_frame`'s and nothing else --
candidates by p(class), ordered by `rel[i,j,predicate] * p_sub_i * p_obj_j`,
single frame, no fusion and no sweep.  That is why it can be pointed at a
synthesised view: it answers "what would a robot standing here ground the
sentence to" for any photograph.

KEEP THIS IN STEP BY HAND.  Being self-contained is the point and the cost: the
shared version cannot be imported, so a change there has to be repeated here.
Last synced when the ranking weight became p(class).

    # files, as before
    python top1.py rgb_image.jpg --task box,behind,chair
    python top1.py out/view_*.png --task box,behind,chair --top 5
    SGG_ROOT=~/sgg_nvs python top1.py img.png --task box,behind,chair

    # LIVE: no file arguments -> subscribe to the robot camera and open a window
    python3 top1.py --task box,behind,chair
    python3 top1.py --topic /oakd/rgb/image_raw/compressed --task box,behind,chair
    python3 top1.py                       # no --task: just the stream, no model
    python3 top1.py --task box,behind,chair --record ~/captures/run.mp4

In the ROS 2 container the two halves live apart -- rclpy comes from /opt/ros,
torch from the venv built by `requirements-egtr.txt` -- so run it as:

    source /opt/ros/jazzy/setup.bash
    SGG_ROOT=~/sgg_nvs ~/venv/bin/python top1.py --task box,behind,chair

In the window: `q` or ESC quits, `s` writes the frame on screen to --snapshot-dir.

Why the topic must be a /compressed one: the raw sensor_msgs/Image topics do not
survive the zenoh bridge (1280x720 is 2.76 MB a frame -- measured zero frames in
30 s, while the JPEG topic ran at 20 Hz), so this subscribes to CompressedImage
and hands the bytes to `cv2.imdecode`, exactly like `tools/grab_image.py`.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from glob import glob
from threading import Thread
from typing import Any, Dict, Optional, Sequence

import numpy as np

#: The EGTR checkout.  Its modules import as `model.egtr`, `util.box_ops`, so its
#: root goes on sys.path.  Default assumes it sits beside this file's parent.
SGG_ROOT = os.environ.get(
    "SGG_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sgg_nvs"))

#: The robot's colour camera, same default as `tools/grab_rgbd.py`.  The OAK-D on
#: the TurtleBot 4 publishes `/oakd/rgb/image_raw/compressed` instead.
DEFAULT_TOPIC = "/camera/camera/color/image_raw/compressed"

#: Where `s` puts a snapshot.  A bind mount (host: ./captures), so it survives a
#: container rebuild -- unlike the working directory, which is not mounted.
DEFAULT_SNAPSHOT_DIR = os.path.expanduser("~/captures")

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

    # WEIGHTED BY p(THE INSTRUCTION'S OWN NOUN), not by the query's confidence
    # over all classes.  The latter is high for a box the detector is merely sure
    # is a chair; measured over 4528 instructions on datasets/sgg/occlusion_ds4
    # (analysis/eval_grounding.py) it scores 29.9% against 30.6%, and degrades
    # far faster as `width` grows -- 23.1% against 28.2% at 40.
    #
    # A RELATION NEEDS TWO OBJECTS.  `i != j` only excludes the same query, and
    # EGTR emits several boxes per object, so a pair can be one object related to
    # itself.  The cut is low because the boxes NEST -- a label inside the carton
    # it is printed on shares all of the smaller box, but IoU divides by the
    # union, so such a pair reads only ~0.23.
    order = sorted(((float(rel[i, j, pred]) * float(p_sub[i]) * float(p_obj[j]),
                     i, j)
                    for i in subs for j in objs
                    if i != j and (not pair_iou
                                   or iou(boxes[i].tolist(), boxes[j].tolist())
                                   < pair_iou)), reverse=True)
    return order, boxes, p_sub, p_obj


def banner(canvas, text: str):
    """The dark strip along the top, with `text` in it.  In place."""
    import cv2

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(canvas, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def ground(egtr, frame, task, width: int = 10, pair_iou: float = 0.15,
           top: int = 1):
    """One RGB frame -> (BGR canvas, banner label, report lines).

    MAGENTA the subject, ORANGE the object.  The banner is left to the caller, so
    it can append its own timing.  `task` None returns the frame untouched, which
    is what lets the live view run without EGTR.  `lines` is empty when there is
    nothing to ground: no model, or no pair survived `pair_iou`.
    """
    import cv2

    canvas = np.ascontiguousarray(frame[:, :, ::-1]).copy()
    if task is None:
        return canvas, "no --task: raw stream", []

    subject, predicate, obj = task
    order, boxes, p_sub, p_obj = rank(egtr, frame, subject, predicate, obj,
                                      width, pair_iou)
    sentence = f"{subject} {predicate} {obj}"
    if not order:
        return canvas, f"{sentence}    no distinct pair", []

    lines = [f"{n}. {subject} {[round(v, 1) for v in boxes[i].tolist()]}"
             f" p={float(p_sub[i]):.3f}"
             f"   {obj} {[round(v, 1) for v in boxes[j].tolist()]}"
             f" p={float(p_obj[j]):.3f}"
             for n, (_, i, j) in enumerate(order[:max(top, 1)], 1)]

    _, i, j = order[0]
    for q, colour, thick in ((i, (230, 120, 230), 3), (j, (90, 200, 255), 2)):
        x0, y0, x1, y1 = (int(v) for v in boxes[q].tolist())
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thick)
    label = (f"{sentence}    p({subject})={float(p_sub[i]):.3f}  "
             f"p({obj})={float(p_obj[j]):.3f}")
    return canvas, label, lines


def live(egtr, task, args) -> int:
    """Subscribe to a CompressedImage topic, ground every frame, show a window."""
    import cv2
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    rclpy.init()
    node = rclpy.create_node("top1_live")
    # ONE SLOT, NOT A QUEUE: at ~20 Hz in and 4 fps out, anything buffered is
    # already stale by its turn.  Keeping the newest frame keeps the view live.
    latest: Dict[str, Any] = {}
    node.create_subscription(CompressedImage, args.topic,
                             lambda msg: latest.__setitem__("msg", msg),
                             qos_profile_sensor_data)

    # THE CALLBACKS NEED THEIR OWN THREAD.  Spinning between frames instead
    # leaves the queue unread for 0.25 s at a time, and Cyclone DDS reassembles
    # these fragmented 0.4 MB frames in only 4 slots (DefragUnreliableMaxSamples)
    # -- once half-arrived frames hold them all, nothing completes again and the
    # subscription is wedged for good (measured: dead at 188 frames).
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    Thread(target=executor.spin, daemon=True).start()

    print(f"subscribed {args.topic}\n"
          f"  q / ESC quits   s saves to {args.snapshot_dir}")
    shown = warned = headless = False
    frames = 0
    last_lines: Sequence[str] = ()
    said: set = set()
    t_start = time.time()
    # Opened on the first frame, when the size is known.  It records the
    # annotated canvas, not the raw stream -- use `ros2 bag record` for that.
    writer = None
    t_first = None

    def wait_keys(delay_ms: int) -> str:
        """-> "quit" / "save" / "".  Also pumps the window's event loop, so it
        has to be called while idling too or the window stops repainting.
        """
        if not shown:
            time.sleep(delay_ms / 1000.0)
            return ""
        return {ord("q"): "quit", 27: "quit",
                ord("s"): "save"}.get(cv2.waitKey(delay_ms) & 0xFF, "")
    try:
        while rclpy.ok():
            msg = latest.pop("msg", None)
            if msg is None:
                if not frames and not warned and \
                        time.time() - t_start > args.timeout:
                    warned = True
                    print(f"  no frame in {args.timeout:.0f}s.  check the topic:"
                          f"\n    ros2 topic list | grep compressed"
                          f"\n    ros2 topic bw {args.topic}", file=sys.stderr)
                if wait_keys(10) == "quit":
                    break
                continue

            bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8),
                               cv2.IMREAD_COLOR)
            if bgr is None:          # truncated JPEG: wait for the next one
                continue
            frame = bgr[:, :, ::-1].copy()

            # EGTR prints from inside its forward pass -- the CUDA-kernel
            # fallback fires a dozen times a frame.  Each message once is enough.
            noise = io.StringIO()
            t0 = time.time()
            with redirect_stdout(noise):
                canvas, label, lines = ground(egtr, frame, task, args.width,
                                              args.pair_iou, args.top)
            dt = time.time() - t0
            frames += 1
            for said_line in noise.getvalue().splitlines():
                if said_line.strip() and said_line not in said:
                    said.add(said_line)
                    print(f"  [egtr] {said_line}")
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            banner(canvas, f"{label}    {dt * 1e3:.0f} ms"
                           f"  lat {time.time() - stamp:.2f}s")

            if args.record:
                if writer is None:
                    height, width = canvas.shape[:2]
                    writer = cv2.VideoWriter(
                        args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                        args.record_fps, (width, height))
                    if not writer.isOpened():
                        raise SystemExit(f"cannot write {args.record}")
                    t_first = time.time()
                    print(f"  recording {width}x{height} @ "
                          f"{args.record_fps} fps -> {args.record}")
                writer.write(canvas)

            # The terminal only hears about a frame that grounds somewhere new,
            # otherwise 20 Hz of identical boxes buries everything else.
            if lines and tuple(lines) != tuple(last_lines):
                last_lines = lines
                print(f"\n#{frames}")
                for line in lines:
                    print(f"  {line}")

            if not headless:
                try:
                    cv2.imshow(args.window, canvas)
                    shown = True
                except cv2.error as exc:
                    if shown:
                        raise
                    reason = getattr(exc, "err", str(exc)).strip()
                    if writer is None:
                        print(f"  cannot open a window ({reason}).\n"
                              f"  X11 forwarding: run ./run.sh (it does the "
                              f"xauth dance) and check DISPLAY inside the "
                              f"container.", file=sys.stderr)
                        return 1
                    # A missing display should not stop a recording run.
                    # Ctrl-C ends it, since there are no keys without a window.
                    headless = True
                    print(f"  no window ({reason}) -- recording only, Ctrl-C to "
                          f"stop", file=sys.stderr)

            action = wait_keys(1)
            if action == "quit":
                break
            if action == "save":
                os.makedirs(args.snapshot_dir, exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out = os.path.join(args.snapshot_dir, f"top1_{ts}.png")
                cv2.imwrite(out, canvas)
                print(f"  -> {out}")
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        # A signal (SIGINT/SIGTERM) already shut the context down inside rclpy's
        # own handler, and shutting it down twice raises RCLError.
        if rclpy.ok():
            rclpy.shutdown()
    if writer is not None:
        writer.release()
        span = time.time() - (t_first or time.time())
        real = frames / span if span > 0 else 0.0
        print(f"  -> {args.record}  ({frames} frames, {span:.0f}s, "
              f"written at {args.record_fps:g} fps)")
        if real and abs(real - args.record_fps) > 0.5:
            # The header fps decides playback speed, so retime, don't re-encode.
            print(f"     grounding actually ran at {real:.1f} fps.  To play at "
                  f"real speed:\n"
                  f"     ffmpeg -r {real:.2f} -i {args.record} -c copy "
                  f"{os.path.splitext(args.record)[0]}_realtime.mp4")
    print(f"\n{frames} frames")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("images", nargs="*",
                    help="image files; omit them to subscribe to --topic")
    ap.add_argument("--task", metavar="SUBJ,PRED,OBJ",
                    help="VG150 words, e.g. box,behind,chair.  Required for "
                         "files; omit it live to view the stream without EGTR")
    ap.add_argument("--topic", default=DEFAULT_TOPIC,
                    help=f"live CompressedImage topic (default {DEFAULT_TOPIC})")
    ap.add_argument("--window", default="top1", help="live window title")
    ap.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR,
                    help="where `s` writes a frame")
    ap.add_argument("--record", metavar="OUT.mp4",
                    help="also write what the window shows to an mp4")
    ap.add_argument("--record-fps", type=float, default=4.0,
                    help="playback rate stamped into the mp4 (default 4, which "
                         "is about what EGTR sustains).  Only the header.")
    ap.add_argument("--timeout", type=float, default=20.0,
                    help="seconds before complaining that no frame arrived")
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

    task = None
    if args.task:
        words = tuple(w.strip() for w in args.task.split(","))
        if len(words) != 3:
            ap.error("--task takes exactly SUBJ,PRED,OBJ")
        task = words
    elif args.images:
        ap.error("--task is required when grounding image files")

    egtr = load_egtr(args.sgg_root) if task else None

    if not args.images:
        return live(egtr, task, args)

    import cv2

    for path in args.images:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"\n{path}\n  ! cannot read")
            continue
        frame = bgr[:, :, ::-1].copy()
        canvas, label, lines = ground(egtr, frame, task, args.width,
                                      args.pair_iou, args.top)
        print(f"\n{os.path.basename(path)}")
        if not lines:
            print("  no distinct pair")
            continue
        for line in lines:
            print(f"  {line}")
        out = os.path.splitext(path)[0] + args.suffix + ".png"
        cv2.imwrite(out, banner(canvas, label))
        print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
