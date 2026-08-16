# reports

One file per weekly write-up, newest at the top of the table. A report is a
snapshot: what was run, what it measured, and what was believed at the time. It
is NOT kept up to date afterwards — when a later week overturns something, the
later report says so and the earlier one is left as written, because a record
that is silently edited is not a record.

| report | covers | headline |
|---|---|---|
| [0821.md](0821.md) | choosing WHERE TO WALK from the sweep alone, on the tabletop cases | comparing the two halves of the sweep by how many of the instruction's candidates still correspond gets top-1 correct in 32 of 40 after one step, against 16 for a random heading, 20 for the best fixed angle and 13 for standing still — and only the first step earns anything |
| [0812.md](0812.md) | both robot experiments to date: the oracle pointer with motion, and A+C+R on an instruction without | pointer — no effect on success (p = 0.61), decisive on speed (12–0, p = 0.0002), but it reads ground truth. Fusion — 23% → 27% (+5 −2, p ≈ 0.23); the binding constraint is instance disambiguation, not the relation |

## Which one describes where we are

`0821.md`.  It covers the live question -- which way to walk -- on the tabletop
cases, and its criterion is `fuse_live.decide`'s: is the top-1 pair at the pose
the robot reached the instructed instance.

`0812.md` is the record of the iTHOR line and of the tabletop generator that
replaced it.  Its static fusion numbers still stand; its motion section does not,
because both the bearing rule and the stop rule it used have been replaced.

## Naming

`MMDD.md`, the date the report was written.  `0813.md` was folded into `0812.md`
rather than kept beside it: both described the same investigation a few days
apart, and two files invited reading the superseded one as current.

Reports are not normally edited after the fact.  `0812.md` was cut down on 14
August anyway -- the oracle-pointer detail, the candidate-list mechanics, an
occlusion-band split and a section of forward planning all went -- because it had
grown to 484 lines of a line of work that has since been retired, and length was
stopping it from being read at all.  What was removed was superseded method and
plans, never a measurement.

## Not reports

`../README.md` is the repo entry point and describes what each file does.
`../TASKS.md` is the running engineering log — measurements, conventions and
the reasons behind constants, accumulated rather than dated. Both stay at the
root because they are current, not historical.
