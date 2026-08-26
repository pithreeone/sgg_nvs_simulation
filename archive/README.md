# Finished probes

These answered a question and are not on the current path. They are kept because
`reports/0812.md`, `PIPELINE.md` and `results/README.md` cite them as the
evidence for claims still in force — a citation to a deleted file cannot be
checked. Nothing in the live path imports anything here, so they can move back
or be dropped without breaking a run.

One line each: what it asked, and what it found.

| script | question | answer |
|---|---|---|
| `probe_argmax.py` | do the conditioned candidates even contain the target? | yes — the ceiling any reweighting can reach |
| `probe_behind.py` | where does `behind` rank for a pair really in that relation? | not first, on tabletop pairs |
| `probe_class.py` | which target/landmark classes does EGTR read at all? | picked the class pairs the generators use |
| `probe_corr.py` | does slot correspondence across views work? | yes; `GATE_COS`, `SKIP_VIEWS` and `seg_box` came from here |
| `probe_direction.py` | is the predicate head absent, or inverted? | it does order the pair; the word is outvoted by a class prior |
| `probe_occl.py` | how much occlusion do the staged cases actually have? | fixed the band the generator targets |
| `probe_size.py` | does the target's pixel size explain the failures? | partly; drove the size bands in `build/robot/build_tabletop.py` |
| `probe_tabletop.py` | is the procedural room readable by EGTR at all? | yes, and which assets |
| `show_candidates.py` | viewer: what the conditioned candidate list looks like | — |
| `show_views.py` | viewer: the sweep, with correspondence drawn | — |
| `eval_nvs_loop.py` | experiment 1: motion with an ORACLE pointer | superseded — ground truth was inside the control loop |

`probe_corr.py` is the one with dependents: `probe_behind.py`,
`probe_direction.py` and `show_views.py` import it, and they moved together.

## The live path, for contrast

`probe_viewpoint.py` (which angles work — the ceiling) -> `probe_sideview.py`
(per-view sweep data) -> `probe_viewdist.py` (the distribution) ->
`eval_move.py` (walk it).  `eval_viewdist.py`, which graded the
distribution's heading against the ladder, went on 2026-08-26 with the P-hat
it graded.
See `reports/0821.md`.
