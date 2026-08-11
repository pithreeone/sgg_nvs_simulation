# nvs_pilot

Inputs and results for the robot experiments. Nothing here is code; the runners
are at the repo root. Case lists are tracked in git, renders are not, and
`fuse_live*.json` is tracked because it is the result, not build output.

## Case lists — a case is a POINTER TO A QUESTION plus a pinned scene

| file | n | predicate | band | scene pinned |
|---|---|---|---|---|
| `cases_wide.json` | 102 | — | — | no |
| `cases_behind.json` | 71 | `behind` | 0.25 / 0.35 / 0.50 | **yes** |
| `cases_frozen.json` | 59 | `on` | 0.15 / 0.50 / 0.90 | no |

`cases_wide.json` is raw discovery (`find_cases.py`) over 34 floor plans,
interleaved kitchen / living room / bedroom. It is the INPUT to freezing, not
runnable itself.

`cases_behind.json` is the current list. "Scene pinned" means each case stores
`occluder_position` and `scene_poses` — a snapshot of every moveable object —
so a rerun restores the exact scene instead of re-running the placement search.
That matters: the search is not reproducible, and repeats of one case moved the
achieved occlusion by several points and the resulting rank by 4. With the
snapshot, four repeats gave identical occlusion and identical A+C+R ranks;
residual noise is ±1 rank from GPU non-determinism.

`cases_frozen.json` has no snapshot because it predates the mechanism. It is
kept because `reports/0813.md` reports on it.

## Results

| file | what | cited by |
|---|---|---|
| `fuse_live_cond.json` | **current** — A+C+R on 71 `behind` cases, conditioned on the instruction's class pair, robot criterion | `reports/0816.md`, headline table |
| `fuse_live_behind.json` | the same cases WITHOUT conditioning: free-form SGDet, 1 of 71 | `reports/0816.md` |
| `fuse_live_acr.json` | A+C+R on the 59 `on` cases: 6 → 6, the measurement that retired `on` | `reports/0816.md` |
| `cases_behind_size.json` | `probe_size.py` — is the target too small to detect? 31/71 detected, and detection does not fall with size | `reports/0816.md` |
| `loop_k10/` | the K=10 walking run, 1.4 GB of renders plus `results.json` | `reports/0813.md` |
| `tasks_behind.png` | `show_tasks.py` — 12 cases drawn, graded instance solid, landmark dashed | — |

## What was deleted, and why it is not a loss

Runs superseded by a later one on the same question: the topk=10 pass (window
too narrow to see a 1-point effect), the B-channel passes (B is not in the
method), and the mild-band `on` run (produced before `fuse_live` honoured the
case file's occlusion band, so it silently re-staged at the default 0.50 and was
a duplicate of the hard run). Logs went with them — every number a report
quotes lives in the JSON, and the logs were narration of runs that no longer
have a claim attached.

`loop_k10/` stays despite its size because it is the only record of the
walking experiment and re-running is 16 minutes plus a superseded pointer.
