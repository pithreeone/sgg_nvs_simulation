"""
fuse_live.py -- run the paper's map-back fusion on one live THOR frame + its sweep.

The offline study feeds `lib/fusion/channels.py` out of a cache that
`scratch/mapback_cache.py` wrote from PNGs on disk.  Nothing about the fusion
needs that: the cache is a schema, not a mechanism.  This file assembles the
SAME record in memory from a frame the robot just took plus the novel views
synthesised around it, and calls the very same functions.  So what runs here is
the paper's fusion, not a reimplementation of it -- which is the only way a
robotics result can be evidence about the paper's method.

The channel set is the one `logs/occlusion_ds4_refpred_assign.log` was run at:

    B (predCond)   beta, pool=mean, endpoint gate cos>=0.80, corr=assign,
                   predcond_mode=refpred -- each view is read at the predicate
                   the REFERENCE proposes, so `on` needs no hand-set predicate
                   list.  Our instructions are all `on`, and the shipped
                   5-predicate list excludes it.
    A (objscore)   OFF, as in that run.  channels.py measures it at +0.0 R@100
                   on top of B.
    C (dedup)      OFF here, deliberately: with C on, a single-view-vs-fused
                   delta bundles deduplication into a number reported as
                   multi-view.  C is a fair thing to add later, to BOTH arms.
    D (predicate calibration)  OFF: it is a prior fitted on labels, and it
                   moves both arms identically.

Views 9 and 19 are dropped by `channels.SKIP_VIEWS`, and that is not a detail:
the lemniscate returns to zero offset at t=pi and t=2pi, so those two frames
reproduce the reference pose.  Counting them manufactures cross-view agreement
out of a copy of the input.

With `--beta 0` the fused ranking is `max_p(rel) * s_i*s_j`, which is exactly
what `sgg_live.predict` computes -- so that setting is a self-test of the
assembly, not an experiment, and the runner checks it.

    python fuse_live.py --case 0 1 2 --beta 1.0
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Sparse width of a view's own relation field.  `assign` correspondence reads
#: the UN-mapped field, which the cache stores at 500 view pairs
#: (`az30el15_v20_um500`); measured equivalent to 2000 for B's recall.
VIEW_TOPK = 500

#: `logs/occlusion_ds4_refpred_assign.log`.
GATE_COS = 0.80
CORR = "assign"
POOL = "mean"


def sparse_pairs(torch, rel, keep: int = VIEW_TOPK):
    """`mapback_cache.topk_sparse`: keep the strongest `keep` pairs of a field."""
    n = rel.shape[0]
    strength = rel.max(-1)[0].clone()
    strength.fill_diagonal_(0.0)
    top = torch.topk(strength.flatten(), min(keep, n * n)).indices
    ii, jj = top // n, top % n
    return torch.stack([ii, jj], 1).short().cpu(), rel[ii, jj].half().cpu()


def record(egtr, reference: np.ndarray,
           views: Sequence[Tuple[int, np.ndarray]]) -> Dict[str, Any]:
    """
    One in-memory `mapback_cache` record: the reference, then every view.

    Field for field what `--layout dirs --unmapped_topk` writes, because
    `channels.view_evidence_perpred` reads it by name.  `probs` is the class
    SOFTMAX and `s_ref`/`c_ref` its max, matching `mapback_cache.py`; the
    sigmoid EGTR also emits belongs to channel A, which is off here.
    """
    from robot.sgg_live import raw_predict

    torch = egtr["torch"]

    def one(frame):
        raw = raw_predict(egtr, frame)
        return {"probs": raw["probs_softmax"].detach().cpu().float(),
                "h": raw["query"].detach().cpu().float(),
                "rel": raw["rel"].detach().cpu().float(),
                "boxes": raw["boxes"].detach().cpu().float()}

    ref = one(reference)
    s_ref, c_ref = ref["probs"].max(-1)
    entries = []
    for index, frame in views:
        view = one(frame)
        vidx, vval = sparse_pairs(torch, view["rel"])
        entries.append({"v": int(index), "h": view["h"].half(),
                        "probs": view["probs"].half(),
                        "vidx": vidx, "vval": vval,
                        "boxes": view["boxes"]})
    return {"rec": {"s_ref": s_ref.half(), "c_ref": c_ref.short(),
                    "h_ref": ref["h"].half(), "views": entries},
            "rel": ref["rel"], "s": s_ref, "c": c_ref, "boxes": ref["boxes"]}


def fuse(built: Dict[str, Any], beta: float, pool: str = POOL,
         gate_cos: float = GATE_COS, corr: str = CORR):
    """
    -> (rank [NQ,NQ], coverage stats).  `beta = 0` gives the single-view rank.

    Straight through `lib.fusion.channels`: per-predicate view evidence, read
    at the reference's proposed predicate, added to the reference's own ranking
    with weight `beta`.
    """
    from lib.fusion import channels as ch

    rel, s = built["rel"], built["s"].float()
    if beta <= 0:
        return ch.pair_rank(rel, s, None, 0.0)[1], {}
    agg, cnt, stats = ch.view_evidence_perpred(built["rec"], pool=pool,
                                               gate_cos=gate_cos, corr=corr)
    if agg is None:
        return ch.pair_rank(rel, s, None, 0.0)[1], {}
    evidence = ch.evidence_at(agg.float(), cnt.float(), rel, pool)
    return ch.pair_rank(rel, s, evidence, beta)[1], stats


def triplets_of(built: Dict[str, Any], rank, egtr,
                topk: int = 10) -> List[Dict[str, Any]]:
    """
    A ranking matrix as the triplet list the rest of this repo speaks.

    Boxes and classes come from the REFERENCE slots for every arm, which is
    what makes single-view and fused directly comparable and what lets
    `task_find.grade` score both against the robot's own frame -- a novel
    view's boxes are in a frame the grader has no ground truth for.
    """
    from lib.pytorch_misc import argsort_desc

    rel, c_ref, boxes = built["rel"], built["c"], built["boxes"]
    s = built["s"].float()
    out = []
    for sub_q, obj_q in argsort_desc(rank.numpy())[:topk, :]:
        sub_q, obj_q = int(sub_q), int(obj_q)
        predicate = int(rel[sub_q, obj_q].argmax())
        out.append({
            "subject": egtr["obj_names"].get(int(c_ref[sub_q]) + 1,
                                             f"obj_{int(c_ref[sub_q])}"),
            "predicate": egtr["rel_names"][predicate],
            "object": egtr["obj_names"].get(int(c_ref[obj_q]) + 1,
                                            f"obj_{int(c_ref[obj_q])}"),
            "score": float(rank[sub_q, obj_q]),
            "subject_score": float(s[sub_q]), "object_score": float(s[obj_q]),
            "subject_box": [float(v) for v in boxes[sub_q].tolist()],
            "object_box": [float(v) for v in boxes[obj_q].tolist()],
            "subject_query": sub_q, "object_query": obj_q,
        })
    return out


def run_case(case: Dict[str, Any], args, egtr) -> Optional[Dict[str, Any]]:
    """Stage the case, sweep it, fuse it, and score both arms on the SAME frame."""
    from robot import drive
    from robot.drive_triplet_scene import measure
    from eval_nvs_pointer import geometry, regrade
    from robot.nvs_lemniscate import camera_for, lemniscate, park_once, sweep
    from robot.robot_controller import point_in_box
    from robot.sgg_live import predict
    from robot.task_find import build_tasks, put_in_front
    from vg.vg150 import THOR_TO_VG150

    rc = drive.open_scene(case["scene"], args.width, args.height, args.fov,
                          case["start"])
    try:
        nameable = sorted(o["name"] for o in rc.event.metadata["objects"]
                          if o["objectType"] in THOR_TO_VG150)
        names = sorted({o["name"] for o in rc.event.metadata["objects"]
                        if o["objectType"] in THOR_TO_VG150
                        or o.get("moveable") or o.get("pickupable")})
        state = measure(rc, nameable, args.fov)
        entries = {o["name"]: o for o in rc.event.metadata["objects"]}
        task = next((t for t in build_tasks(state["objects"], entries)
                     if t["target_name"] == case["target_name"]), None)
        if task is None:
            print("  ! the instruction is no longer unambiguous here")
            return None
        print(f"  {task['instruction']}   target {task['target_name']}")
        staged = put_in_front(rc, task["target_name"], case["occluder_type"],
                              task["receptacle_name"])
        if staged is None:
            print("  ! staging failed")
            return None

        reference = rc.event.frame.copy()
        camera = rc.camera_xyz.copy()

        # The orbit centre, from the OBJECT anchor's box and the real depth
        # frame.  No ground truth -- see `robot_controller.point_in_box`.
        from robot.sgg_live import raw_predict
        object_id = {v: k - 1 for k, v in egtr["obj_names"].items()}[
            task["object_class"]]
        ref_raw = raw_predict(egtr, reference)
        anchor = int(ref_raw["probs_softmax"][:, object_id].argmax())
        centre = point_in_box(rc, ref_raw["boxes"][anchor].tolist(), args.fov)
        if centre is None:
            print("  ! no depth at the object anchor")
            return None

        poses = [camera_for(centre, camera, az, el)
                 for az, el in lemniscate(args.views, args.max_az, args.max_el)]
        reachable = rc.controller.step(
            action="GetReachablePositions").metadata["actionReturn"] or []
        park_once(rc, {"x": float(camera[0]), "y": float(camera[1]),
                       "z": float(camera[2])}, reachable)
        rendered = sweep(rc, poses, task["target_name"], args.fov, reachable,
                         keep_frames=True)

        built = record(egtr, reference,
                       [(i, r["frame"]) for i, r in enumerate(rendered)])

        # Grade on the robot's own frame, which is the only frame the ground
        # truth is defined in.  Both arms share it.
        rc.teleport(position={"x": float(camera[0]),
                              "y": float(rc.agent_position["y"]),
                              "z": float(camera[2])})
        geo = geometry(rc, [task["target_name"]]
                       + [d["name"] for d in task.get("distractors") or []],
                       names, False)

        rows = {}
        for label, beta in (("single", 0.0), ("fused", args.beta)):
            rank, stats = fuse(built, beta)
            triplets = triplets_of(built, rank, egtr, args.topk)
            _, result = regrade(task, geo, triplets, args.iou)
            rows[label] = {"result": result, "stats": stats,
                           "top": [f"{t['subject']} {t['predicate']} "
                                   f"{t['object']}" for t in triplets[:3]]}
            print(f"    {label:7s} class {str(result['class_rank']):>5}  "
                  f"grounded {str(result['grounded_rank']):>5}  "
                  f"best IoU {result['best_subject_iou']:.3f}   "
                  f"{rows[label]['top'][0] if rows[label]['top'] else '-'}")
            if stats:
                print(f"            gate: {stats['views_per_pair']:.2f} of "
                      f"{stats['n_views']} views speak per pair, "
                      f"{stats['pairs_covered']:.1%} of pairs covered")

        # Self-test: beta=0 must reproduce `sgg_live.predict` exactly, or the
        # record assembly is wrong somewhere upstream of the fusion.
        direct = predict(egtr, reference, args.topk)
        rank0, _ = fuse(built, 0.0)
        mine = triplets_of(built, rank0, egtr, args.topk)
        agree = sum(a["subject_query"] == b["subject_query"]
                    and a["object_query"] == b["object_query"]
                    for a, b in zip(direct, mine))
        print(f"    [self-test] beta=0 reproduces predict(): "
              f"{agree}/{len(direct)} of the top-{args.topk} pairs")

        return {"scene": case["scene"], "instruction": task["instruction"],
                "staged_occlusion": staged["occlusion"],
                "self_test": f"{agree}/{len(direct)}",
                **{k: {"class_rank": v["result"]["class_rank"],
                       "grounded_rank": v["result"]["grounded_rank"],
                       "best_subject_iou": v["result"]["best_subject_iou"],
                       "top3": v["top"][:3]} for k, v in rows.items()}}
    finally:
        rc.stop()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="nvs_pilot/cases_frozen.json")
    ap.add_argument("--case", type=int, nargs="*", default=[0])
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--max-az", type=float, default=30.0)
    ap.add_argument("--max-el", type=float, default=15.0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--out", default="nvs_pilot/fuse_live.json")
    args = ap.parse_args(argv)

    from robot.sgg_live import load_egtr

    cases = json.load(open(args.cases))["cases"]
    egtr = load_egtr()

    results = []
    for index in args.case:
        print(f"\n[{index}] {cases[index]['scene']}")
        try:
            outcome = run_case(cases[index], args, egtr)
        except Exception as error:                              # noqa: BLE001
            print(f"  ! {type(error).__name__}: {error}")
            outcome = None
        if outcome:
            results.append({"case": index, **outcome})

    if args.out and results:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"beta": args.beta, "gate": GATE_COS, "corr": CORR,
                       "pool": POOL, "cases": results}, fh, indent=1)
        found = [r for r in results if r["fused"]["grounded_rank"]]
        print(f"\n{len(results)} cases; fused grounded in {len(found)}, "
              f"single-view grounded in "
              f"{sum(1 for r in results if r['single']['grounded_rank'])}"
              f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
