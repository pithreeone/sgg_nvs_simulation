"""Is the target's naming failure caused by OCCLUSION, or by the class itself?

The tabletop scenes contain a control that answers this exactly: the distractor
is a BYTE-IDENTICAL COPY of the target, same class, same asset, placed
unoccluded.  So for every case there is a matched pair differing in one
variable.

    target     35% hidden by the landmark
    distract   the same asset, 0% hidden

If the copy is named and the target is not, occlusion is the cause and pooling
class evidence across views is the right fix -- `conditioned` reads
`probs_ref`, the reference frame alone, so nothing in A+C+R can currently do it.
If both fail equally, the class is the cause and no viewpoint helps.

Reported for both selection rules, since they can disagree:
    prob    rank among ALL queries by p(class)
    argmax  survives only if some covering query argmaxes to the class

Measured on `cases_easy` (08-12): both copies named 17, only the unoccluded twin
7, only the target 4, NEITHER 12.  `neither` being the largest failure bucket is
why the case list was rebuilt on class grounds rather than on geometry -- see
`probe_class.py` and nvs_pilot/README.md.
"""
import collections
import json
import sys

import numpy as np

sys.path.insert(0, "/home/pithreeone/Ben/japan_intern/simulation")

from robot.proc_scene import open_room, rebuild, visible_box       # noqa: E402
from robot.sgg_live import load_egtr, raw_predict                  # noqa: E402
from robot.task_find import iou                                    # noqa: E402

IOU_HIT = 0.5
CASES = sys.argv[1] if len(sys.argv) > 1 else "nvs_pilot/cases/cases_easy.json"
cases = json.load(open(CASES))["cases"]
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
        index = classes.get(case["subject_class"])
        row = {"scene": case["scene"], "cls": case["subject_class"],
               "occlusion": case["staged_occlusion"]}
        for role in ("target", "distract"):
            truth = visible_box(event, role)
            if truth is None or index is None:
                row[role] = None
                continue
            covering = {q for q in range(len(boxes))
                        if iou(boxes[q].tolist(), truth) >= IOU_HIT}
            order = list(np.argsort(-probs[:, index]))
            row[role] = {
                "prob_rank": next((r for r, q in enumerate(order, 1)
                                   if q in covering), None),
                "argmax_ok": any(argmax[q] == index for q in covering),
                "best_p": max((float(probs[q][index]) for q in covering),
                              default=0.0),
                "px": int(abs(truth[2] - truth[0]) * abs(truth[3] - truth[1])),
            }
        rows.append(row)
        t, d = row["target"], row["distract"]

        def brief(e):
            if e is None:
                return "not localised"
            return (f"argmax {'Y' if e['argmax_ok'] else '.'} "
                    f"rank {str(e['prob_rank']):<4} p {e['best_p']:.3f}")

        print(f"  {case['scene']:14s} {row['cls']:6s} target {brief(t)}  |  "
              f"twin {brief(d)}", flush=True)
finally:
    controller.stop()

ok = [r for r in rows if r["target"] and r["distract"]]
n = len(ok)
print(f"\n  {n} cases with both copies localised\n")
print(f"  {'':10} {'argmax 通過':>12} {'prob rank 中位數':>18} "
      f"{'p(class) 中位數':>16} {'像素中位數':>12}")
for role, lab in (("target", "target 35%遮"), ("distract", "twin  0%遮 ")):
    a = sum(1 for r in ok if r[role]["argmax_ok"])
    ranks = [r[role]["prob_rank"] for r in ok if r[role]["prob_rank"]]
    print(f"  {lab:10} {a:>8}/{n:<3} {np.median(ranks):>17.0f} "
          f"{np.median([r[role]['best_p'] for r in ok]):>15.3f} "
          f"{np.median([r[role]['px'] for r in ok]):>11.0f}")

# The paired test: same asset, same class, one variable.
gain = sum(1 for r in ok if r["distract"]["argmax_ok"]
           and not r["target"]["argmax_ok"])
loss = sum(1 for r in ok if r["target"]["argmax_ok"]
           and not r["distract"]["argmax_ok"])
from scipy import stats                                            # noqa: E402
print(f"\n  配對: 只有 twin 通過 {gain}, 只有 target 通過 {loss}, "
      f"p={stats.binomtest(gain, max(gain + loss, 1), 0.5).pvalue:.4f}")
both = sum(1 for r in ok if r["target"]["argmax_ok"] and r["distract"]["argmax_ok"])
neither = sum(1 for r in ok if not r["target"]["argmax_ok"]
              and not r["distract"]["argmax_ok"])
print(f"       兩者都通過 {both}, 兩者都不通過 {neither}")
print(f"\n  `兩者都不通過` 是 multi-view 救不到的那部分: {neither}/{n} "
      f"({neither / n:.0%}) -- 那是 class 的問題, 不是遮擋的問題")
print(f"  `只有 twin 通過` 是遮擋造成的, 也就是 multi-view 的頭寸: "
      f"{gain}/{n} ({gain / n:.0%})")
byc = collections.defaultdict(lambda: [0, 0, 0])
for r in ok:
    b = byc[r["cls"]]
    b[2] += 1
    b[0] += r["target"]["argmax_ok"]
    b[1] += r["distract"]["argmax_ok"]
print(f"\n  {'class':8} {'target':>8} {'twin':>8} {'n':>4}")
for cls, (t, d, m) in sorted(byc.items(), key=lambda kv: -kv[1][2]):
    print(f"  {cls:8} {t:>6}/{m:<2} {d:>6}/{m:<2} {m:>4}")
json.dump(rows, open("nvs_pilot/probe_occl.json", "w"), indent=1)
print("\n  wrote nvs_pilot/probe_occl.json")
