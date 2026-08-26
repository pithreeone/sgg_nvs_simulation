"""Verify, independently, what an ARGMAX filter costs the candidate pool.

reports/0812.md quotes 12/40 for argmax-based detection, from `probe_tabletop.py`.
That probe asked a slightly different question than `conditioned` would -- it
asked whether ANY query covering the object argmaxes to the instructed class,
with no top-K at all -- so this re-measures both selection rules in the SAME
shape as the pipeline actually uses them:

    prob    (current)  rank ALL queries by p(instructed class), keep top K
    argmax  (proposed) keep only queries whose argmax class IS the instructed
                       class, then rank those by p(instructed class), keep top K

and reports, per endpoint and jointly, how often a query COVERING the real
object (IoU >= 0.5) survives.  Joint is the ceiling: `conditioned` needs both
ends, so a rule that keeps the landmark but loses the target scores zero.

    python probe_argmax.py results/cases/cases_easy2.json

The joint column is what motivated rebuilding the case list on 08-12.  On
`cases_easy` it was 25/40 even at K=40, so 15 cases could not be won by any
decision rule; on `cases_easy2` it is 40/40 at K=10.  See results/README.md.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/pithreeone/Ben/japan_intern/simulation")

from robot.world.proc_scene import open_room, rebuild, visible_box       # noqa: E402
from robot.sgg_live import load_egtr, raw_predict                  # noqa: E402
from robot.task.task_find import iou                                    # noqa: E402

IOU_HIT = 0.5
KS = (1, 2, 3, 5, 10, 20, 40)
CASES = sys.argv[1] if len(sys.argv) > 1 else "results/cases/cases_tabletop.json"
cases = json.load(open(CASES))["cases"]
print(f"  cases: {CASES}")
egtr = load_egtr()
controller = open_room(800, 600, 60.0)
rows = []
try:
    for case in cases:
        event = rebuild(controller, case)
        raw = raw_predict(egtr, event.frame)
        probs = raw["probs_softmax"].cpu().numpy()
        boxes = raw["boxes"].numpy()
        classes = {v: k - 1 for k, v in egtr["obj_names"].items()}
        argmax = probs.argmax(-1)
        row = {"scene": case["scene"]}
        for role, want in (("target", case["subject_class"]),
                           ("occluder", case["object_class"])):
            truth = visible_box(event, role)
            index = classes.get(want)
            if truth is None or index is None:
                row[role] = None
                continue
            covering = {q for q in range(len(boxes))
                        if iou(boxes[q].tolist(), truth) >= IOU_HIT}
            # rule `prob`: all queries, ranked by p(class)
            order = list(np.argsort(-probs[:, index]))
            prob_rank = next((r for r, q in enumerate(order, 1)
                              if q in covering), None)
            # rule `argmax`: only queries whose argmax IS this class
            keep = [q for q in order if argmax[q] == index]
            arg_rank = next((r for r, q in enumerate(keep, 1)
                             if q in covering), None)
            row[role] = {"prob_rank": prob_rank, "argmax_rank": arg_rank,
                         "argmax_pool": len(keep),
                         "covering": len(covering)}
        rows.append(row)
        t, o = row.get("target"), row.get("occluder")
        print(f"  {case['scene']:14s} target prob#{t and t['prob_rank']} "
              f"argmax#{t and t['argmax_rank']} (pool {t and t['argmax_pool']})"
              f"   landmark prob#{o and o['prob_rank']} "
              f"argmax#{o and o['argmax_rank']} (pool {o and o['argmax_pool']})",
              flush=True)
finally:
    controller.stop()

n = len(rows)


def within(role, key, k):
    return sum(1 for r in rows if r.get(role) and r[role][key]
               and r[role][key] <= k)


def joint(key, k):
    return sum(1 for r in rows
               if r.get("target") and r.get("occluder")
               and r["target"][key] and r["occluder"][key]
               and r["target"][key] <= k and r["occluder"][key] <= k)


print(f"\n  {n} cases.  A query COVERING the real object at IoU >= 0.5 survives:\n")
print(f"  {'K':>4} | {'target prob':>12} {'target argmax':>14} | "
      f"{'lmk prob':>10} {'lmk argmax':>12} | {'JOINT prob':>11} "
      f"{'JOINT argmax':>13}")
for k in KS:
    print(f"  {k:>4} | {within('target','prob_rank',k):>9}/{n:<2} "
          f"{within('target','argmax_rank',k):>11}/{n:<2} | "
          f"{within('occluder','prob_rank',k):>7}/{n:<2} "
          f"{within('occluder','argmax_rank',k):>9}/{n:<2} | "
          f"{joint('prob_rank',k):>8}/{n:<2} {joint('argmax_rank',k):>10}/{n:<2}")

print(f"\n  no covering query survives the argmax filter AT ALL:")
for role, lab in (("target", "target"), ("occluder", "landmark")):
    dead = sum(1 for r in rows if r.get(role) and r[role]["argmax_rank"] is None)
    print(f"    {lab:9s} {dead}/{n}")
both = sum(1 for r in rows if r.get("target") and r.get("occluder")
           and r["target"]["argmax_rank"] and r["occluder"]["argmax_rank"])
print(f"    both ends survive the filter at any K: {both}/{n}")
out = "results/probe_argmax_" + CASES.split("/")[-1]
json.dump(rows, open(out, "w"), indent=1)
print(f"\n  wrote {out}")
