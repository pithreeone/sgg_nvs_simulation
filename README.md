# SGG evaluation in AI2-THOR

**An occlusion benchmark.** Render cluttered THOR scenes, annotate every relation
with how hidden its endpoints are, and measure how EGTR's recall falls as
occlusion rises — against a ground-truth definition calibrated to what VG's human
annotators actually wrote down rather than to metric geometry.

**[TASKS.md](TASKS.md) is the record.** It carries every result, the reasoning
behind each design decision, and — deliberately — the findings that were later
withdrawn. Read it before changing anything; several obvious-looking
"improvements" have already been tried and measured to fail.

## Where things are

**[PIPELINE.md](PIPELINE.md) is the one to read first** if the question is how
the method works — what happens between "Find the cup behind the laptop" and
one box the robot walks to, and specifically where each fusion channel enters
and what it is able to change. It is kept current, unlike `reports/`, which are
dated snapshots.

The repo does two things. The top level shows exactly those two and nothing
else; everything a run needs but no one reads first is one level down.

```
build/sgg/                1. build the occlusion dataset  (needs AI2-THOR)
build/robot/ 2. build a case list -- an instruction, the objects that
                       make it true, and a camera pose it is true from.  Reads
                       no model.  `build_tabletop.py` (target + same-asset
                       distractor: the relation has to say WHICH), `build_slot.py`
                       (target behind two occluders: the viewpoint has to be
                       CHOSEN).
fuse_live.py        3. the robot experiments -- run a case list against EGTR.
                       The current one: the paper's A+C+R over a live frame and
                       its synthesised sweep
eval_move.py           the same perception, repeated, plus a stop rule and a
                       bearing policy -- see PIPELINE.md section 7

robot/              the robot, the sweep, EGTR, staging and grading -- and the
                    shared rule: grounding.py is the metric, viewpick.py the
                    viewpoint policy, evidence.py channel B
viz/                every figure: the tiling primitives, and the scripts that
                    draw a case list, a sweep or a walk
scoring/            R@K against the dataset -- a CROSS-REPO interface, read by
                    ../sgg_nvs; module names there are part of the contract
vg/                 VG150 vocabulary and the priors fitted on it
analysis/           report scripts nothing imports, incl. eval_grounding.py --
                    the robot's metric over datasets/sgg/, 4528 instructions

datasets/           everything a THOR run produced; see datasets/README.md
results/          robot-experiment results and the frozen task list
experiments/
  build_logs/       provenance for each dataset
reports/            one file per weekly write-up; reports/README.md indexes them
```

Each package's `__init__.py` says what belongs in it and what deliberately does
not — read those before moving a file. Two have been moved back out already
because the import graph, not the file name, decides where something lives.

`datasets/sgg/occlusion_ds4` is the dataset to use. It changes nothing about the
headline numbers — see the ds4 section of [TASKS.md](TASKS.md) — but it doubles
n and is the first version whose heavily-occluded band (`0.75+`, n=462) is large
enough to read.

Predictions for the occlusion sets live outside this tree, under
`../sgg_nvs/results/occlusion_ds*_baseline/`.

**Where the work is right now:** the "Where this is now" section of the newest
report, [reports/0813.md](reports/0813.md).

## The code, by job

### Ground truth — what counts as a relation

| file | job |
|---|---|
| `vg150.py` | THOR objectType → VG150 class, and the predicate families |
| `vg_conventions.py` | measures what VG's *human* annotators mean by each predicate |
| `vg_gt.py` | derives THOR relations in that convention (3D truth gate + 2D bands) |
| `vg_pair_prior.py` | `P(predicate \| subject class, object class)` over 315642 VG relations |
| `vg_prior.json`, `vg_pair_prior.json` | the fitted priors, read at import |

The one thing to understand before touching these: the 2D box features cannot
name a relation. Run back over VG's own labels they score 55.3% against a 68.8%
always-say-`on` baseline. The class pair scores 80.8%. `vg_gt` therefore uses
geometry to decide whether a relation is *true* and the class-pair prior to
decide which pairs are worth annotating.

### Dataset generation

The end-to-end procedure — the six steps, the two build-time filters, and why
relation geometry uses amodal boxes — is written up as *How a scene is built* in
[TASKS.md](TASKS.md). Read that before this table; the table only says which file
holds which part.

| file | job |
|---|---|
| `occlusion.py` | occlusion primitives, and the shared `MAX_OCCLUSION` ceiling |
| `occlusion_pipeline.py` | scatter clutter over real surfaces, physically settled |
| `build_occlusion_dataset.py` | **entry point** — render views, annotate, write `scene.json` |
| `survey_scenes.py` | ranks all 120 floor plans; how the scene lists were chosen |
| `export_samples.py`, `export_multiview.py`, `thor_camera.py` | the `multiview/` export |
| `relabel_occlusion.py`, `relabel_multiview.py` | re-derive annotations in place, no THOR |

`relabel_occlusion.py` is the one to reach for when a ground-truth *definition*
changes: everything `vg_gt` needs is already in `scene.json`, so a redefinition
costs a second per scene instead of a THOR session, and runs against identical
renders and identical predictions.

### Evaluation

| file | job |
|---|---|
| `eval_occlusion.py` | **main** — R@K / mR@K by occlusion band; `--summary` for the compact table |
| `eval_recall.py` | the same for `multiview/`, sweeping GT definitions |
| `analyze_multiview.py` | *where* recall is lost — one failure mode per relation |
| `tune_gt.py` | sweep GT constraints against predictions already on disk |

Two flags on `eval_occlusion.py` that are easy to misread:

- `--accept exact|human` — `human` also accepts wordings VG annotators use for
  the same class pair. It is **not free**: a random accept-set of the same size
  recovers about half the same gain, so quote `exact` and treat `human` as an
  upper bound on what wording disagreement costs.
- `--class-agnostic` — drops object labels, keeping boxes and predicate. Isolates
  the relation head from naming, the way PredCls/SGCls/SGDet does.

### Picking a scene to drive

| file | job |
|---|---|
| `find_partial_triplets.py` | query the annotation on disk: triplets with an endpoint half hidden, no THOR |
| `drive_triplet_scene.py` | open such a scene live and hand the robot to you at that viewpoint |

`find_partial_triplets.py` is a read of `scene.json`; the band defaults to
`0.4 < occlusion < 0.8`, which is 2069 of `occlusion_ds4`'s 7235 relations
(28.6%), or 206 with *both* endpoints inside it.

`drive_triplet_scene.py` treats those views as candidates only. The dataset's
cameras sit at 1.0–1.5 m and the acting agent's eye height is fixed, so it
rebuilds the clutter, aims the agent at the centroid of the half-hidden
triplets from a 1.2–2.5 m standoff, and **re-measures** occlusion through the
robot's own camera with `annotate_relations` before reporting anything. Reusing
the recorded camera *pitch* instead of re-aiming does not work — it points the
agent at the floorboards.

Four ways to say where to start: let it pick (default), `--list-views` then
`--view N` for a particular dataset viewpoint, `--start=X,Z,YAW,HORIZON[,BODY]`
for a pose of your own, or drive somewhere in the window and press `p` — that
writes a scenario at the pose you are standing in.

The grid search runs **once** per scene build: the winning pose is remembered in
`.drive_poses/` and later runs reuse it (58 s → 9 s on FloorPlan203 s3 v6, same
pose, same 30 relations / 14 in band). `--rescan` forces a new search. Only the
pose is cached, never the occlusion figures — `P.arrange` is not bit-reproducible,
so every start is re-measured against the render actually in front of the robot.

Driving is the default; `--driver none` measures the start pose and exits.
`--resume driveable/fp203/scenario.json` reopens a saved start pose without
repeating the search, and `resume()` is the same thing as a function for a
policy of your own — it hands back a `RobotController` already standing there.

It writes `triplet.png` alongside `start.png`: **one** relation, its two
endpoints boxed amodal-outside/visible-inside as in `visualize_occlusion.py`,
colour by role rather than by band, with the predicate and both occlusions in
the caption. `--triplet N` picks which one; `n`/`b` cycle in the window. One at a
time is deliberate — these views carry a dozen in-band relations each.

### Left over from the removed embodied experiment

An embodied experiment (Task 1 — height-conditioned object search) lived here and
was deleted along with its results. Its interpreter outlived it as unreachable
code and has now gone too: `decision_tree.py` (the graph→`ActionStruct` policy)
and the 844 lines of `robot_controller.py` that executed those ActionStructs —
`execute`, `check_success`, the nine behaviour generators, and the helpers only
they called. Nothing outside that island ever called into it.

What remains of `robot_controller.py` is the base API every current caller
actually uses — `RobotController`, `_xz`, `yaw_towards`, `horizon_towards`,
`unproject`, `point_in_box` — read by `export_samples.py`,
`export_multiview.py`, `thor_camera.py`, `drive.py` and the NVS runners. 1383
lines down to 464.

### Diagnostics

`visualize_occlusion.py` — draw the ground truth over a rendered view, to eyeball
whether an annotation is one a human would have written. Defaults to
`occlusion_ds4`.

## Running things

```bash
# build a dataset (needs the ai2thor env)
python build_occlusion_dataset.py --rooms all --out occlusion_ds5

# re-derive annotations after a GT change, no THOR needed
python relabel_occlusion.py --root occlusion_ds4 --dry-run

# score
python eval_occlusion.py --gt occlusion_ds4 \
    --preds ../sgg_nvs/results/occlusion_ds4_baseline --summary
```

Dataset generation needs `ai2thor` and `cv2`; evaluation needs neither, so it
runs in the plain environment.
