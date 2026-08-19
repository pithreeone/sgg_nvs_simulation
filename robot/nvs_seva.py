"""
nvs_seva.py -- Stable Virtual Camera in the seam `nvs_lemniscate.sweep` opened.

`sweep(..., synth=f)` wants `(poses) -> [HxWx3 uint8]`; this is that `f`.  The
model gets ONE frame -- the robot's own view -- and the same (azimuth,
elevation) list `camera_for` was given, which is already SEVA's `lookat_dist`
parameterisation, so the poses are replayed rather than converted.  Only the two
signs differ, and `--calibrate` settles them against THOR's frames.

    python robot/nvs_seva.py --calibrate --cases datasets/robot/cases_slot.json
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# `python robot/<script>.py` puts robot/ on sys.path, not the repo root, so the
# top-level packages would not import.  Same bootstrap as gen/.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


#: The checkout, not the installed `seva` package: `resolve_poses` lives in the
#: repo root's `simple_inference.py`, which pip does not install.
SEVA_ROOT = os.environ.get(
    "SEVA_ROOT", os.path.abspath(os.path.join(
        _ROOT, "..", "sgg_nvs", "third_party",
        "Stable-Virtual-Camera-Modified")))

#: Read off THOR's own sweep of the same poses: its positive azimuth walks the
#: camera LEFT and SEVA's walks it right, and its positive elevation looks DOWN
#: at the table where SEVA's ends up underneath it.  A wrong sign mirrors every
#: view silently, so both are measured, not argued.
AZ_SIGN = -1.0
EL_SIGN = -1.0


class Synthesiser:
    """SEVA, loaded once (5 GB of weights) and called once per sweep."""

    def __init__(self, steps: int = 10, cfg: float = 3.0,
                 camera_scale: float = 1.0, seed: int = 23,
                 two_pass: bool = False, T: int = 21,
                 lookat_dist: float = 0.5, fov: float = 60.0,
                 az_sign: float = AZ_SIGN, el_sign: float = EL_SIGN,
                 dump: Optional[str] = None, device: str = "cuda:0") -> None:
        if SEVA_ROOT not in sys.path:
            sys.path.insert(0, SEVA_ROOT)

        import torch

        from seva.eval import IS_TORCH_NIGHTLY
        from seva.model import SGMWrapper
        from seva.modules.autoencoder import AutoEncoder
        from seva.modules.conditioner import CLIPConditioner
        from seva.sampling import DiscreteDenoiser
        from seva.utils import load_model

        self.steps, self.cfg, self.camera_scale = steps, cfg, camera_scale
        self.seed, self.two_pass, self.T = seed, two_pass, T
        self.lookat_dist, self.fov = lookat_dist, fov
        self.az_sign, self.el_sign = az_sign, el_sign
        self.dump, self.device = dump, device
        self.tag = ""          # the caller's name for the current sweep
        self._sweeps = 0

        self.ae = AutoEncoder(chunk_size=1).to(device)
        self.conditioner = CLIPConditioner().to(device)
        self.denoiser = DiscreteDenoiser(num_idx=1000, device=device)
        self.model = SGMWrapper(
            load_model(model_version=1.1, device="cpu").eval()).to(device)
        if IS_TORCH_NIGHTLY:
            self.ae, self.conditioner, self.model = [
                torch.compile(m, dynamic=False)
                for m in (self.ae, self.conditioner, self.model)]

    def for_reference(self, reference: np.ndarray):
        """The callable `sweep` takes: this synthesiser bound to one input frame."""
        return lambda poses: self(reference, poses)

    def __call__(self, reference: np.ndarray,
                 poses: Sequence[Dict[str, Any]]) -> List[np.ndarray]:
        """One diffusion pass over the whole trajectory, frames in pose order."""
        import torch
        from PIL import Image

        from seva.eval import infer_prior_stats, run_one_scene
        from seva.geometry import get_default_intrinsics
        from simple_inference import resolve_poses

        spec = ";".join(f"{self.az_sign * float(p['azimuth'])},"
                        f"{self.el_sign * float(p['elevation'])}"
                        for p in poses)
        w2cs = resolve_poses(poses=spec, lookat_dist=self.lookat_dist)
        if len(w2cs) != len(poses):
            raise ValueError(f"resolve_poses returned {len(w2cs)} for "
                             f"{len(poses)} poses")

        height, width = reference.shape[:2]
        # SEVA works at a /64 grid on a 576 long side; the frames go back out at
        # the robot's own resolution because EGTR is what reads them and both
        # arms must be read at the same size.
        L = 576
        ratio = width / height
        W, H = (L, int(L / ratio)) if width < height else (int(L * ratio), L)
        W, H = (W // 64) * 64, (H // 64) * 64

        with tempfile.TemporaryDirectory(
                dir=os.environ.get("TMPDIR")) as work:
            path = os.path.join(work, "input.png")
            Image.fromarray(np.asarray(reference, dtype=np.uint8)).save(path)

            ref_w2c = torch.eye(4)
            c2ws = torch.linalg.inv(
                torch.cat([ref_w2c[None], torch.stack(list(w2cs))]))[:, :3, :]
            # THOR's `fieldOfView` is VERTICAL; `get_default_intrinsics` reads
            # the horizontal one when W >= H.  Passing 60 straight through would
            # hand the model a 60-degree-wide frame that is really 75.6 wide.
            hfov = 2.0 * math.atan(math.tan(math.radians(self.fov) / 2.0)
                                   * W / H)
            Ks = get_default_intrinsics(np.full((len(c2ws),), hfov),
                                        aspect_ratio=W / H).float()
            Ks[:, :2] *= torch.tensor([W, H]).reshape(1, -1, 1).repeat(
                Ks.shape[0], 1, 1).float()

            version_dict = {"H": H, "W": W, "T": self.T, "C": 4, "f": 8,
                            "options": {
                                "chunk_strategy_first_pass": "nearest-gt",
                                "chunk_strategy":
                                    "interp" if self.two_pass else "nearest",
                                "guider_types": [1, 2] if self.two_pass else [1],
                                "cfg": [self.cfg, 2.0] if self.two_pass
                                       else [self.cfg],
                                "camera_scale": self.camera_scale,
                                "num_steps": self.steps, "cfg_min": 1.2,
                                "encoding_t": 1, "decoding_t": 1,
                                "save_input": False,
                                "beta_linear_start": 5e-6, "log_snr_shift": 2.4,
                                "video_save_fps": 30.0,
                                "num_targets": len(w2cs)}}

            anchors = infer_prior_stats(self.T, 1, num_total_frames=len(w2cs),
                                        version_dict=version_dict)
            prior = np.linspace(1, len(w2cs), anchors).tolist()
            rounded = [round(i) for i in prior]
            for _ in run_one_scene(
                    "img2trajvid" if self.two_pass else "img2trajvid_s-prob",
                    version_dict, self.model, self.ae, self.conditioner,
                    self.denoiser,
                    {"img": [path] + [None] * len(w2cs), "input_indices": [0],
                     "prior_indices": prior},
                    {"c2w": c2ws, "K": Ks,
                     "input_indices": list(range(len(c2ws)))},
                    work, use_traj_prior=self.two_pass,
                    traj_prior_Ks=Ks[rounded] if self.two_pass else None,
                    traj_prior_c2ws=c2ws[rounded] if self.two_pass else None,
                    seed=self.seed):
                pass

            frames = self._read(os.path.join(work, "samples-rgb"),
                                len(w2cs), width, height)

        if self.dump:
            self._dump(reference, frames)
        self._sweeps += 1
        return frames

    @staticmethod
    def _read(directory: str, want: int, width: int, height: int
              ) -> List[np.ndarray]:
        """`save_output` writes `000.png ...` already sorted into pose order."""
        from PIL import Image

        names = sorted(n for n in os.listdir(directory) if n.endswith(".png"))
        if len(names) != want:
            raise RuntimeError(f"SEVA wrote {len(names)} frames for {want} poses")
        return [np.asarray(Image.open(os.path.join(directory, n))
                           .convert("RGB").resize((width, height),
                                                  Image.BILINEAR))
                for n in names]

    def _dump(self, reference: np.ndarray, frames: Sequence[np.ndarray]) -> None:
        """Every synthesised sweep on disk, for the eyeball that the metric is not."""
        from PIL import Image

        out = os.path.join(self.dump, self.tag, f"sweep{self._sweeps:04d}")
        os.makedirs(out, exist_ok=True)
        Image.fromarray(np.asarray(reference, dtype=np.uint8)).save(
            os.path.join(out, "input.png"))
        for i, frame in enumerate(frames):
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                os.path.join(out, f"{i:03d}.png"))


def _sharpness_match(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of the two gradient magnitudes, in [-1, 1].

    NOT mean absolute error: a wrong elevation puts the camera under the table,
    whose smooth beige underside scores a BETTER MAE against a detailed frame
    than a sharp view that is merely rotated wrong.  MAE rewards blur; gradients
    do not.
    """
    def grad(img):
        g = np.asarray(img, np.float32).mean(-1)
        return np.hypot(np.diff(g, axis=1)[:-1], np.diff(g, axis=0)[:, :-1])

    x, y = grad(a).ravel(), grad(b).ravel()
    x, y = x - x.mean(), y - y.mean()
    scale = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / scale) if scale else 0.0


def calibrate(argv_args) -> int:
    """WHICH WAY IS LEFT.  Four sign conventions, scored against THOR's sweep.

    A mirrored azimuth looks like a plausible novel view and turns every bearing
    the fusion reports into its opposite, which is what `eval_move` measures.
    """
    import json

    from robot.nvs_lemniscate import (LOOKAT_DIST, camera_for, lemniscate,
                                      look_at_point, park_once, sweep)
    from robot.proc_scene import ROOM, Robot, look_from, open_room, rebuild

    case = json.load(open(argv_args.cases))["cases"][argv_args.case]
    print(f"calibrating on {case['scene']}  {case['instruction']}", flush=True)

    controller = open_room(argv_args.width, argv_args.height, argv_args.fov)
    try:
        rc = Robot(controller, rebuild(controller, case))
        camera = rc.camera_xyz.copy()
        reference = rc.event.frame.copy()
        orbit = look_at_point(camera, rc.agent_yaw, rc.camera_horizon,
                              LOOKAT_DIST)
        poses = [camera_for(orbit, camera, az, el)
                 for az, el in lemniscate(argv_args.views, argv_args.max_az,
                                          argv_args.max_el)]
        corner = min(((0.3, 0.3), (0.3, ROOM - 0.3), (ROOM - 0.3, 0.3),
                      (ROOM - 0.3, ROOM - 0.3)),
                     key=lambda p: -math.dist(p, (camera[0], camera[2])))
        look_from(controller, corner[0], corner[1], 0.0, 0.0, force=True)
        truth = [r["frame"] for r in
                 sweep(rc, poses, case["target_name"], argv_args.fov, [],
                       keep_frames=True)]
    finally:
        controller.stop()

    if argv_args.dump:
        # The truth goes on disk too; a synthesised sweep with nothing to hold
        # it against is unreadable.
        from PIL import Image

        out = os.path.join(argv_args.dump, "truth")
        os.makedirs(out, exist_ok=True)
        Image.fromarray(np.asarray(reference, dtype=np.uint8)).save(
            os.path.join(out, "input.png"))
        for i, frame in enumerate(truth):
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                os.path.join(out, f"{i:03d}.png"))

    # One model, four conventions: the signs are read per call.
    model = Synthesiser(steps=argv_args.synth_steps,
                        camera_scale=argv_args.synth_camera_scale,
                        two_pass=argv_args.synth_two_pass,
                        lookat_dist=LOOKAT_DIST, fov=argv_args.fov,
                        dump=argv_args.dump)
    rows = []
    for az_sign in (1.0, -1.0):
        for el_sign in (1.0, -1.0):
            model.az_sign, model.el_sign = az_sign, el_sign
            frames = model(reference, poses)
            match = float(np.mean([_sharpness_match(a, b)
                                   for a, b in zip(frames, truth)]))
            mae = float(np.mean([np.mean(np.abs(a.astype(np.float32)
                                                - b.astype(np.float32)))
                                 for a, b in zip(frames, truth)]))
            rows.append((match, az_sign, el_sign, mae))
            print(f"  az {az_sign:+.0f}  el {el_sign:+.0f}   "
                  f"match {match:6.3f}   (MAE {mae:6.2f})", flush=True)

    rows.sort(reverse=True)
    print(f"\n  best: AZ_SIGN = {rows[0][1]:+.0f}, EL_SIGN = {rows[0][2]:+.0f} "
          f"(match {rows[0][0]:.3f} against {rows[-1][0]:.3f}).\n"
          f"  Current: {AZ_SIGN:+.0f} / {EL_SIGN:+.0f}.  Look at --dump too.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--calibrate", action="store_true",
                    help="score the four azimuth/elevation sign conventions "
                         "against THOR's own sweep and print the winner")
    ap.add_argument("--cases", default="datasets/robot/cases_slot.json")
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--max-az", type=float, default=30.0)
    ap.add_argument("--max-el", type=float, default=15.0)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--synth-steps", type=int, default=10)
    ap.add_argument("--synth-camera-scale", type=float, default=1.0)
    ap.add_argument("--synth-two-pass", action="store_true")
    ap.add_argument("--dump", default=None, metavar="DIR",
                    help="write every synthesised sweep here")
    args = ap.parse_args(argv)

    if not args.calibrate:
        ap.error("nothing to do; --calibrate is the only mode this module runs "
                 "on its own.  `eval_move.py --synth seva` is the caller.")
    return calibrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
