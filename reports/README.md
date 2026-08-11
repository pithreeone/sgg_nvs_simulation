# reports

One file per weekly write-up, newest at the top of the table. A report is a
snapshot: what was run, what it measured, and what was believed at the time. It
is NOT kept up to date afterwards — when a later week overturns something, the
later report says so and the earlier one is left as written, because a record
that is silently edited is not a record.

| report | covers | headline |
|---|---|---|
| [0816.md](0816.md) | both robot experiments to date: the oracle pointer with motion, and A+C+R on an instruction without | pointer — no effect on success (p = 0.61), decisive on speed (12–0, p = 0.0002), but it reads ground truth. Fusion — 23% → 27% (+5 −2, p ≈ 0.23); the binding constraint is instance disambiguation, not the relation |

## Which one describes where we are

`0816.md` is the only report and covers everything run so far. It opens with a
table of the two experiments, which are not interchangeable: the oracle pointer
(`eval_nvs_pointer.py` / `eval_nvs_loop.py`, with motion, ground truth in the
loop) and the instruction-conditioned fusion (`fuse_live.py --condition 10`, no
motion, no ground truth in the loop).

**Nothing has yet combined the current fusion with robot motion.** That is the
next step.

## Naming

`MMDD.md`, the date the report was written.  `0813.md` was folded into `0816.md`
rather than kept beside it: both described the same investigation a few days
apart, and two files invited reading the superseded one as current.  Reports are
not edited after the fact, but merging two live drafts of the same week is not
that.

## Not reports

`../README.md` is the repo entry point and describes what each file does.
`../TASKS.md` is the running engineering log — measurements, conventions and
the reasons behind constants, accumulated rather than dated. Both stay at the
root because they are current, not historical.
