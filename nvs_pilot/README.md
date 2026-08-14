# nvs_pilot

Inputs and results for the robot experiments. Nothing here is code; the runners
are at the repo root. Case lists are tracked in git, renders are not, and
`fuse_live*.json` is tracked because it is the result, not build output.

## Case lists — a case is a POINTER TO A QUESTION plus a pinned scene

Every list here is PROCEDURAL — built by `build_robotic_task/`, replayed exactly
by `robot.proc_scene.rebuild`. The iTHOR lists are gone; see the end of this
section for what that costs.

| file | n | predicate | band | what it measures |
|---|---|---|---|---|
| `cases_easy2.json` | 40 | `behind` | 0.25 / 0.35 / 0.50 | **relational grounding** — two identical copies of the target, so only the relation says which |
| `cases_hard.json` | 40 | `behind` | 0.50 / 0.60 / 0.75 | the same at 60% occlusion — the difficulty is OCCLUSION, the one thing a novel view can repair |
| `cases_slot.json` | 40 | `behind` | 0.35 / 0.50 / 0.65 | **viewpoint choice** — one target behind two occluders, visible only over an interval of azimuth.  No same-class copy, so the relation is stated and not needed |
| `cases_tabletop.json` | 40 | `behind` | 0.25 / 0.35 / 0.50 | superseded by `cases_easy2` — see below |
| `cases_easy.json` | 40 | `behind` | 0.25 / 0.35 / 0.50 | superseded by `cases_easy2` — see below |

The three lists in play are `cases_easy2`, `cases_hard` and `cases_slot`, and
they are not interchangeable: the first two ask whether the right INSTANCE is
picked, the third whether the right VIEWPOINT is. Pooling them would average two
different questions.

`cases_easy2.json` replaced `cases_easy.json` on 08-12 and the reason is
measured, not stylistic. Candidate selection ranks all 200 queries by
p(instructed class) and keeps the top K on each side; on `cases_easy` the
correct pair was outside K=10 in 15 of 40 cases, so those 15 could not be won by
ANY decision rule. Two faults caused it, both found by `probe_class.py`:

  * `box` as the subject. Its assets render at 15928 px against cup's 2540 --
    larger than the laptop they are said to stand behind -- and the distractor
    is the same asset unoccluded, so the dozen-odd queries covering that giant
    copy fill the p(class) ranking ahead of the target. Median target rank 14.
  * `--occluder-width 0.30` admits ONE laptop asset, so 23 of 40 cases shared a
    single laptop. At 0.40 there are 23 assets and the rebuild used 18 of them.

`cases_easy2` is `cup`/`bottle` behind `laptop` at width 0.40, plus a new
generator constraint that the target's unoccluded pixel count must be below the
landmark's. Coverage went 25/40 to 40/40 at K=10 and single-view top-1 from
8/40 to 31/40. `cases_hard` is the same recipe at a 50-75% occlusion band,
built to put the difficulty back into OCCLUSION rather than into naming --
occlusion is the only thing a novel viewpoint can repair.

`cases_tabletop.json` was the first procedural list, and `cases_easy2` is it with
the two faults above fixed. It is kept because `reports/0812.md` reports on it,
and it differs in two ways worth knowing before comparing: the gap between target
and landmark is 0.175-0.21 m rather than a pinned 0.35, and it carries no
`occluder_yaw`, so its occluders replay untorned.

`cases_slot.json` is the newest and the only one built for a question about
VIEWPOINTS rather than instances. It is also the only one staged from a CROUCHED
camera (`standing: false`, camera 0.900 m against 1.575 m), because a standing
camera pitches 27 degrees down and a tabletop occluder is then passed over rather
than looked through — measured, a box hid 3% of a laptop standing and 50%
crouched, same scene. Its start poses therefore only replay correctly through the
`standing` field, and its occlusion numbers are not comparable with the lists
above. Each case additionally carries `sweep`, the target's visible pixels at
every azimuth with the occluders and without: a COLUMN, never a filter.

Its asymmetry needs stating in any write-up. The blocker is bisected at the rim of
the sweep so its side is 98-100% hidden there, but the landmark is only ~0.5 m
wide against a ±0.32 m sightline sweep and its inner edge is pinned by the 50%
bisection, so on 22 of 40 cases the LANDMARK side of the rim is under 60% hidden.
"Walking to the end loses" is true on the blocker side and weak on the other.

### What the deleted iTHOR lists cost

`cases_wide.json` (102, raw discovery), `cases_behind.json` (71, the `behind`
list) and `cases_frozen.json` (59, `on`) were removed on 08-14. They came from
`find_cases.py` searching stock floor plans, so they carry no `objects` field and
replay through `drive.open_scene` rather than `proc_scene.rebuild` — a second
backend for a line of work that had been superseded on measurement: all 71 had
ZERO same-class distractors, which is why the predicate never did any work there.

The cost is that `reports/0812.md`'s headline table, and the `fuse_live_*.json`
results below it, are no longer re-runnable — the numbers stand as a dated
snapshot only. Recover with `git checkout e06cd9a -- nvs_pilot/cases_behind.json`
(and `f2aea53` for `cases_frozen.json`).

## Results

| file | what | cited by |
|---|---|---|
| `fuse_live_cond.json` | **current** — A+C+R on 71 `behind` cases, conditioned on the instruction's class pair, robot criterion | `reports/0812.md`, headline table |
| `fuse_live_behind.json` | the same cases WITHOUT conditioning: free-form SGDet, 1 of 71 | `reports/0812.md` |
| `fuse_live_acr.json` | A+C+R on the 59 `on` cases: 6 → 6, the measurement that retired `on` | `reports/0812.md` |
| `probe_size_behind.json` | `archive/probe_size.py` on `cases_behind` — is the target too small to detect? 31/71 detected, and detection does not fall with size | `reports/0812.md` |
| `loop_k10/results.json` | the K=10 walking run.  The 1.4 GB of renders beside it were removed 08-14 | `reports/0812.md` |
| `fuse_tabletop_rb.json` | **current static** — 40 tabletop cases, single 3/40 vs A+C+R 6/40 | `reports/0812.md` |
| `fuse_tabletop_k100.json` | the same, ordered by each pair's best predicate instead of `behind`: 5/40 | `reports/0812.md` |
| `move_tabletop.json` | **current motion** — evidence 3 correct vs random 0, of 34 walked | `reports/0812.md` |
| `move_stops_rerun/`, `move_tabletop_rerun.log` | the 08-11 22:42 rerun. Its result JSON was byte-identical to `move_tabletop.json` and was deleted; the renders are NOT identical (22 frames against 24, produced after `render_stops` changed) so they are kept as the newer pictures of the same decisions | — |


### 08-12 — the rebuilt list, and four mechanisms that did not work

Every row is `--condition 10`, top-1 decision (rank the ~100 conditioned pairs,
take rank 1, grade both endpoints).

| file | what | single | A+C+R | reading |
|---|---|---|---|---|
| `fuse_easy.json` | old `cases_easy` | 8/40 | 9/40 | paired 1-0. The +1 is A's; C+R adds nothing |
| `fuse_easy2.json` | **rebuilt list** | **31/40** | **31/40** | A+C+R chose the IDENTICAL pair in 40/40 |
| `fuse_easy2_gap.json` | the same, recording the rank-1/rank-2 margin | 31/40 | 31/40 | see below |
| `fuse_easy2_converse.json` | ranked by `log rel[behind] − log rel[in front of]` | 18/40 | 18/40 | fixed 2, broke 15 |
| `fuse_hard_max.json` | `cases_hard`, + channel M pooling the predicate over views with `max` | 13/40 | 14/40 | M: fixed 5, broke 5 |
| `fuse_hard_mean.json` | the same with `mean`, M's control | 13/40 | 14/40 | M: fixed 5, broke 4 |

`probe_class_w030.json` is the class measurement the rebuild rests on;
`probe_argmax_cases_{easy,easy2,hard}.json` are the coverage tables;
`probe_occl.json` is the paired target-vs-twin control that showed the old
list's losses were naming (12 of 40 with both copies unnamed) rather than
occlusion (7 of 40 with only the unoccluded copy named).

`probe_class.json` -- the first pass, at `--width-cap 0.22 --assets 5` -- was
deleted rather than kept as a wider sweep. Taking five assets per class means
taking the five ALPHABETICALLY FIRST, which is not a sample: it scored `cup`
4/5 with p 0.042 where the full 34-asset pool gives 33/34 with p 0.342, and a
file that says cup is mediocre is worse than no file. `probe_class_w030.json`
runs the generator's own pool and is the one to quote.

Four negative results, each with its own control, and one shared cause:

  * **converse ratio** — 31/40 to 18/40. All 40 winners have
    `rel[behind]/rel[in front of]` at 10^13: `sgg_live.py` clamps `pred_rel` at
    0, so the denominator sits on a numerical floor and the ratio ranks by
    which pair's denominator collapsed hardest, not by depth.
  * **channel M, `max`** — net 0, and its `mean` control did as well (net +1).
    If a max were picking up the one view that clears the landmark, the mean --
    which cannot pick up a single good view -- should not match it.
  * **variance across views** — the occluded target is more view-variable than
    its unoccluded twin in 59% of scenes for `behind`, against 51% for the
    symmetric control `near`. Eight points is not a mechanism.
  * **rank-1/rank-2 margin as a confidence signal** — AUC 0.61. Best threshold
    buys 78% to 83% answering accuracy while handing 5 of 40 cases to the
    motion arm, 2 of which were already right.

The shared cause is measurable: on `probe_twinviews.json` the same pair's `rel`
spans ~17 orders of magnitude ACROSS VIEWS, and it does so equally for the
occluded target (16.93), its unoccluded twin (16.60), and the symmetric `near`
(17.21). The across-view variation is numerical, not photometric, so any
decision statistic computed from the cross-view `rel` field is reading noise.

Coverage on `cases_hard` closes the remaining door: the target still has a
covering query inside the top 10 in 38 of 40 cases at 60% occlusion. The
reference frame can still SEE the target, so "carry a detection back from a
view that sees it" -- the one job no existing channel does -- has almost nothing
to carry.

**These tabletop files describe a case list that no longer exists, and 38 of its
40 scenes are unrecoverable.** `cases_tabletop.json` was rebuilt at 08-11 16:56,
AFTER `fuse_tabletop_k100` (16:26) and `fuse_tabletop_rb` (16:39) had run.

Comparing scene IDs understates this: `tabletop|N` is the build loop's index,
not an identity, so 22 of 40 IDs appear in both files while naming different
questions — `tabletop|10` was `cup behind clock` and is now `cup behind box`.
Matching on instruction AND staged occlusion, **2 of 40 scenes survive**.

They cannot be restored. `cases_tabletop.json` was never tracked in git, so
there is no earlier revision (`cases_behind`, `cases_frozen` and `cases_wide`
are tracked — this one was the omission), and the result files record only
`scene`, `instruction` and `staged_occlusion`, not the asset ids and positions a
rebuild would need.

So `reports/0812.md`'s 3/40 → 6/40 is an unverifiable number: re-running on the
current list yields a different measurement on different questions, not a
reproduction. Quote it only with that caveat, or replace it with a rerun. The
files are kept because they are its only record, not because they are checkable.

Both tabletop results were produced with

```
--condition 10 --topk 100 --beta 0 --objscore mean --rank-by instructed
```

and graded at BOTH ENDPOINTS: the subject box on the instructed copy and the
object box on the landmark, IoU >= 0.5 against the VISIBLE box. There is no
amodal fallback here — `visible_box` reads the segmentation frame, so a 35%
occluded target is graded on the third of it that shows, which is stricter than
the iTHOR path's `max(amodal, visible)`.

## What was deleted, and why it is not a loss

Runs superseded by a later one on the same question: the topk=10 pass (window
too narrow to see a 1-point effect), the B-channel passes (B is not in the
method), and the mild-band `on` run (produced before `fuse_live` honoured the
case file's occlusion band, so it silently re-staged at the default 0.50 and was
a duplicate of the hard run). Logs went with them — every number a report
quotes lives in the JSON, and the logs were narration of runs that no longer
have a claim attached.

`loop_k10/results.json` stays because it is the only record of the walking
experiment.  Its 1.4 GB of renders went on 08-14: the experiment is retired
(ground truth was inside its control loop), the JSON carries every number
`reports/0812.md` quotes, and 93% of this directory was pictures of a run
nothing cites.  `trail_hard/` and `trail_easy2_r12/` went with them -- pose-by-
pose renders from bearing rules that `reports/0821.md` replaced.

Removed 08-12, all verified byte-identical or incomplete before removal:
`move_tabletop_0811.{json,log}` and `move_stops_0811/` (md5-identical to the
undated versions they snapshot), `fuse_tabletop_0811.json` (md5-identical to
`fuse_tabletop_rb.json`, so the 17:18 timestamp is a copy and not a second
run), `move_tabletop_rerun.json` (md5-identical to `move_tabletop.json`),
`move_easy_k10_PARTIAL` and `move_easy_side_PARTIAL` (interrupted at 32/40 and
25/40, so neither has a denominator), and `fuse_live.json` (three exploratory
cases, no claim attached).

Probe outputs removed the same day, each superseded by a strictly better
measurement of the same question rather than merely by age:

  * `probe_class.{json,log}` — the alphabetical-sample bug above.
  * `probe_twin_gap020` and `probe_twin_gap060` — both hold ZERO view sweeps,
    so the target-vs-twin comparison in them is the reference frame's alone.
    `probe_twinviews.json` is the same comparison with 1134 views, and it is
    the file the 17-orders-of-magnitude finding comes from.
  * `probe_gap_sweep` — the gap is no longer a free variable; `--gap 0.35` is
    fixed in every case list built since.
  * `probe_k.json` — a K sweep with one arm, replaced by the full K table in
    `probe_argmax_cases_*.json`, which reports both endpoints and both
    selection rules.
  * `probe_argmax.json` — ran against `cases_tabletop.json`, the list that no
    longer matches its own results.

`probe_bearing.py` went with them: nothing imports it, no report cites it, and
the bearing question it asked is answered by `eval_move.py --bearing side`.
`probe_argmax.py` and `probe_occl.py` were RESTORED to the repo root the same
day — they had only ever existed in a scratch directory, which is not a place
a file the README cites can live.
