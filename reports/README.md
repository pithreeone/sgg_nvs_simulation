# reports

One file per weekly write-up, newest at the top of the table. A report is a
snapshot: what was run, what it measured, and what was believed at the time. It
is NOT kept up to date afterwards — when a later week overturns something, the
later report says so and the earlier one is left as written, because a record
that is silently edited is not a record.

| report | covers | headline |
|---|---|---|
| [0813.md](0813.md) | NVS as a viewpoint pointer, 59 tasks, K=10 | success rate is a coin flip (p = 0.613), speed is decisive (12–0, p = 0.00024) — but the pointer reads ground truth, so it is an upper bound, not a deployable result |

## Which one describes where we are

The newest report's own "Where this is now" section, and nothing else. Two
experiments run in this repo and they are not interchangeable — the oracle
pointer (`eval_nvs_pointer.py`, `eval_nvs_loop.py`) and the fusion pointer
(`fuse_live.py`). Every table in `0813.md` is the first one.

## Naming

`MMDD.md`, the date the report was written. Future weeks may want `YYYY-MM-DD`
once two years overlap; not worth renaming the existing one for.

## Not reports

`../README.md` is the repo entry point and describes what each file does.
`../TASKS.md` is the running engineering log — measurements, conventions and
the reasons behind constants, accumulated rather than dated. Both stay at the
root because they are current, not historical.
