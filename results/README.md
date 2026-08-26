# results

Experiment output.  Nothing here is code; the runners are at the repo root and
the case lists are in `datasets/robot/`.

Renders are not tracked (`results/**/*.png`) and neither are build logs.  What
stays is what a report cites, and only for as long as it does: a result whose
experiment has been superseded is deleted rather than kept as a second answer to
the same question.  Git history holds them if a number ever has to be checked.

## robot_movement/

The current work: does choosing the viewpoint matter, and by how much.  Six
policies over two case lists, three fixed steps each.  See its own README --
that is where the policies and the two lists are explained.

## The rest

| file | what it is | cited by |
|---|---|---|
| `fuse_*.json` | STATIC fusion: one reference frame plus a full sweep, no motion, top-1 measured.  `fuse_live.py`, not `eval_move.py` | `reports/0812.md` |
| `move_tabletop.json` | the 08-12 motion run on `cases_tabletop` | `reports/0812.md` |
| `move_hard_side.json`, `move_hard_stop.json` | the `side` rule, and the stopping-rule run | `reports/0821.md` |
| `grounding_ds4.json` | the instruction-conditioned top-1 over 4528 instructions of `datasets/sgg/occlusion_ds4`, 48 settings.  Why `--weight` defaults to `class` | — |
| `figs_viewgrid/` | the control arm's action space, drawn | — |

## Deleted on 2026-08-26

The `move_*` runs of 08-14 to 08-21 and every `probe_*.json`: all of them were
measured on `cases_slot` and `cases_hard` at 800x600, before the start-pose
screen and before `--policy`, so none of them is comparable with anything in
`robot_movement/`.  Two reports still quote probe numbers and now have no file
behind them.

`loop_k10/` went too.  It was the only record of the K=10 walking experiment,
which is retired for a reason that makes the record worth nothing to re-read:
ground truth was inside its control loop.

The probe SCRIPTS stay -- `probe_viewpoint.py` asks whether a solvable viewpoint
exists at all, which is the ceiling no policy can exceed, and a new case list
needs that answer before it needs a policy.  Only their old answers are gone.
