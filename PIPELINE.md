# How the instruction-conditioned pipeline actually works

What happens between "Find the cup behind the laptop" and one box the robot
walks to. Written 08-12 because the same question kept being asked in a form
the code could answer but no document did: **where does each fusion channel
enter, and what does it get to change?**

This is a description of the MECHANISM, not a snapshot of results, so unlike
`reports/*.md` it is kept current. Every claim below carries the file and line
that implements it; if a line moves, fix the anchor rather than the claim.

---

## 1. The scene

`build/robot/build_tabletop.py` stages three objects on a table:

```
            [target]         the thing the instruction names, 35% hidden
               |  0.35 m
          [landmark]  [distractor]
               |  1.0 m
            [camera]
```

`distractor` is the **same asset and same class** as `target`. It sits beside
the landmark rather than behind it, so exactly one copy satisfies the relation.

This is the whole reason the dataset exists. Recognising a cup is not enough to
answer "the cup behind the laptop" — there are two identical cups, and only the
spatial relation separates them. Any result that could be obtained by object
detection alone is not measuring what this project is about.

The instruction is always `Find the {A} behind the {B}`.

---

## 2. What EGTR produces from one image

One forward pass, 200 decoder queries. Per query:

| | |
|---|---|
| `boxes[q]` | a candidate box |
| `probs[q]` | 150 class probabilities — p(this box is a cup), p(laptop), … |
| `s[q]` | object score: confidence in whatever class this query likes best |

Plus one relation tensor:

```
rel[i, j, p]      i = subject query, j = object query, p = one of 50 predicates
rel[i, j, behind] "how much query i is behind query j"
```

Two properties of `rel` drive nearly every result in this project:

* it spans about **14 orders of magnitude**, and the instructed pair sits low
  in that range (median 2.3e-10);
* `robot/sgg_live.py:155` clamps it at zero (`torch.clamp(pred_rel, 0.0, 1.0)`),
  so small values are truncated rather than merely small.

---

## 3. The NVS sweep

`fuse_live.py --views 20` orbits the reference camera on a Gerono lemniscate
(azimuth ±30°, elevation ±15°), running EGTR on each view. Two views duplicate
the reference and are skipped, leaving 18.

Each view's queries are matched back to the REFERENCE frame's queries by cosine
similarity of decoder hidden states, gated at 0.8 (`lib/fusion/channels.py`,
`_pair_field`). That correspondence is what lets a view's opinion be written
into the reference frame's coordinates — the paper's map-back. All three
channels are built on it:

| channel | what it computes |
|---|---|
| **A** | pools each query's object score `s` over the views (`max` / `mean`) |
| **C** | merges queries with high box IoU and the same class into one slot |
| **R** | each view names one predicate for a pair; the majority relabels it |

The `single` arm uses none of this — reference frame only. That is the
comparison "does NVS help".

---

## 4. Step one — which pairs are even eligible

`conditioned()`, [fuse_live.py:886](fuse_live.py#L886):

```
subject side   the 10 queries with the highest p(cup)
object side    the 10 queries with the highest p(laptop)
               -> 10 x 10 = 100 candidate pairs
```

Only the two class names the instruction supplies are used. No ground truth.
The predicate is deliberately NOT restricted: whether the model comes to say
`behind` about a pair is the quantity under test, and fixing it would hand over
the answer.

**No fusion channel participates in this step.** Two facts in the code:

* `conditioned` reads `built["probs_ref"]` — the reference frame's own class
  probabilities, not a pooled version;
* `cand` is computed at line 886, **outside** the arm loop that begins at
  [line 906](fuse_live.py#L906), so `single`, `A` and `A+C+R` are handed the
  identical candidate list.

The consequence is the single most important structural fact about this
pipeline: **if the right cup is not in the top 10 by p(cup), no channel can
recover it.** A, C and R reorder what is already on the list; none of them adds
to it. `reports/0812.md` states the same thing for A alone.

This is why changing the OBJECT CLASSES moved the numbers so much more than any
method change. Candidate coverage at K=10 went 25/40 on `cases_easy` to 40/40
on `cases_easy2`, and single-view top-1 followed it from 8/40 to 31/40 — while
converse ranking, cross-view pooling and every other decision rule tried on the
same data moved it by 0 or made it worse.

---

## 5. Step two — which pair the robot acts on

`decide()`, [fuse_live.py:410](fuse_live.py#L410). The score is one line:

```python
score(i, j) = rel[i, j, behind] * s_i * s_j
```

argmax over the ~100 candidates. What each channel does here:

| channel | how it enters | what it can change |
|---|---|---|
| **A** | replaces `built_row["s"]` ([line 911](fuse_live.py#L911)) | a MULTIPLIER on a quantity spanning 14 orders of magnitude |
| **C** | passed as `groups` ([line 927](fuse_live.py#L927)) | DROPS pairs whose endpoints are the same merged object; changes no score |
| **R** | **not passed to `decide` at all** | nothing |

`decide`'s signature is
`decide(built, egtr, task, rank, candidates, geo, iou_hit, groups, score_mode)`
— there is no `relabel` parameter. R cannot reach the decision.

Measured consequence on `cases_easy2`: **A+C+R chose the identical pair as
`single` in 40 of 40 cases.** That is not a bug; it is what this table
predicts.

---

## 6. Step three — the triplet list, and why there are two metrics

`triplets_of()` ([line 936](fuse_live.py#L936)) builds the ranked triplet list
the robot reads. **This is the only place R acts** — it relabels entries.

The two metrics read different objects:

| metric | reads | asks | requires the word `behind`? |
|---|---|---|---|
| `top-1 decision` | `decide()` | are the top pair's two boxes on the real target and the real landmark (IoU ≥ 0.5)? | **no** |
| `robot succeeds` | `triplets_of()` | the robot scans for the first triplet matching the instruction and walks to it — is that one right? | **yes** |

`robot succeeds` is the robot's own criterion (the code says so at
[fuse_live.py:948](fuse_live.py#L948)). `top-1 decision` is a diagnostic added
later; the robot never uses it.

The split explains the shape of every result in this project. On
`cases_tabletop`, A+C+R moved `robot succeeds` 3/40 → 6/40 while `top-1
decision` moved 8/40 → 9/40. R doubled how often `behind` was uttered at all
(13/40 → 26/40), which loosens the LEXICAL GATE; it did not make the ranking
find the right object more often.

---

## 7. Static and motion — how the two experiments relate

**motion = static, repeated, plus a stop rule and a bearing policy.**

```
static (fuse_live.py)   look once -> answer

motion (eval_move.py)   look -> stop rule says "sure enough"  -> answer
                                stop rule says "not yet"      -> pick a bearing,
                                                                 move, look again
                        up to 10 poses, each one a static experiment
```

Step 0 of a motion episode IS the static experiment, and the numbers confirm it:
`fuse_easy` scores top-1 8/40, and `move_easy`'s step 0 scores 8/40.

The two added components:

| | decides | current implementation |
|---|---|---|
| stop rule | WHICH pose's answer counts | the first triplet in the list labelled `behind` |
| bearing policy | where to step next | `evidence` (view votes) or `random` |

### They ablate different things

Both motion arms use identical perception — A+C+R in each, and `look` is the
same code for both ([eval_move.py:139](eval_move.py#L139)). Only the bearing
differs. So:

```
static   does NVS make ONE LOOK better?      single  vs  A+C+R
motion   does NVS make the WALK smarter?     random  vs  evidence
```

Motion cannot measure perception; static cannot measure navigation. Their
numbers must not be merged into one claim.

### The same two metrics carry over

`eval_move` records a per-pose `recall`, and `recall == 1` is that pose's top-1
decision. So a motion episode has a ceiling and a realisation:

```
move_easy (40 cases)      step 0     best pose     stopped AND correct
  evidence                 8/40        17/40             2/40
  random                   8/40        12/40             2/40
```

* **8 -> 17** is what MOVING buys, and it is where `evidence` beats `random`
  (17 vs 12) — a comparison that never passes through the lexical gate, unlike
  the 4-vs-1 usually quoted.
* **17 -> 2** is what the STOP RULE costs. The robot stopped 16 times and was
  right twice; 15 episodes had already seen the right answer and did not cash
  it in.

Both experiments end at 2/40 for the same reason, and it is not perception.

---

## 8. Where the pipeline currently loses

Measured on `cases_easy2` (40 cases, cup/bottle behind laptop, 35% occluded):

| stage | question | passes |
|---|---|---|
| 1. candidates | is the correct pair among the 100? | **40/40** |
| 2. ranking | does `rel[behind]·s·s` put it first? | **31/40** |
| 3. lexical gate | is that pair's argmax predicate `behind`? | **0/40** |

`robot succeeds` is stage 3, and stage 3 is where 78% becomes 5%. Both metrics
on the same 40 cases, so the two effects can be read off one table:

| | single | A+C+R | |
|---|---|---|---|
| `top-1 decision` | **31/40** | **31/40** | relation is given; only "which one" is asked |
| `robot succeeds` | 1/40 | 2/40 | additionally demands the model VOLUNTEER `behind` |

The 2 that `robot succeeds` gets are a SUBSET of the 31 — it finds nothing the
ranking missed, it only discards 29 of what the ranking found. And A+C+R moves
the second row and not the first, which is section 5 and 6 restated: R acts on
labels, and only the second row reads labels.

Stage 3 does **not** fail because the model cannot tell which cup is behind the
laptop. Against an identical distractor, ranking by the `behind` slice picks the
right copy 31/40 times, and the slot-order test
(`rel[target,landmark,behind] > rel[landmark,target,behind]`) holds in 32/47
staged pairs — 68% against a 50% baseline. The depth information is present.

It fails because the model's preferred WORD for such a pair is something else.
Over 47 staged pairs where the target genuinely is behind the landmark, the top
spatial predicate was `in front of` 14, `near` 13, `on` 11, and `behind` only 3.
On the 40 winning pairs of `cases_easy2`: `near` 30, `in front of` 7, `under` 3,
**`behind` 0**.

And `near` is not wrong. A cup 0.35 m from a laptop is near it. The stop rule
asks a question with several true answers and accepts one of them.

---

## 9. What follows for anyone changing this

* A change that only reweights existing candidates has a ceiling set by step
  one. Check candidate coverage (`probe_argmax.py`) before attributing a result
  to a decision rule.
* A change that should affect the RANKING must modify `rel[i, j, predicate]`
  itself. A, C and R structurally cannot; `--pool-predicate` (channel M) is the
  only path that does, and it measured net 0 with its own `mean` control
  scoring the same — see `results/README.md`.
* Anything built on cross-view differences in `rel` should be treated as
  suspect until shown otherwise. The same pair's `rel` varies by ~17 orders of
  magnitude across views, and it does so equally for the occluded target
  (16.93), its unoccluded twin (16.60), and the symmetric predicate `near`
  (17.21). That variation is numerical, not photometric.
* A change aimed at `robot succeeds` is a change to the lexical gate, and the
  fair version applies to every arm. Giving only the fused arm a better stop
  rule measures the stop rule, not the fusion.

## 10. Measurement hazards, with the cost each one already had

Every entry here is a mistake that produced a plausible number, survived
several rounds of analysis, and was only caught by a control. They are written
down because vigilance did not catch any of them — structure did.

### The truth may add a column. It may never remove a row.

A ground-truth test that decides WHICH views, candidates, or cases enter a
computation pre-sorts the sample toward the answer. Concretely, this line, added
to a per-view scoring loop derived from `probe_sideview.py`:

```python
if not parts or not any(p["ok"] for p in parts):   # p["ok"] IS the truth
    continue
```

deleted every view where the instructed pair failed to correspond — exactly the
views looking from the side the target is hidden on. On 10 `cases_hard` scenes,
removing it moved:

| | with the filter | without |
|---|---|---|
| views entering the distribution, per case | 12.9 | **18.0** (all of them) |
| per-view top-1 correct | 47% | **34%** |
| vote for the object pair correct | 6/10 | **4/10** |
| aligned oracle-side margin | +0.060 | **+0.017** |

The distribution went from looking like a policy to being indistinguishable
from a random heading (4/10 against 4.2/10). `probe_viewdist.py` enforces the
rule structurally: `build_*` takes no `truth` argument, a view that corresponds
to nothing is recorded rather than dropped, and `build_views` asserts that every
non-skipped sweep view produced a row.

### The denominator is every case

A case the method cannot speak about is a failure, not missing data. Reporting
27/38 rather than 27/40 turned two systematic failures — the scenes where no
view corresponds at all, `tabletop|33` and `tabletop|41` — into an exclusion,
and those two are winnable (some heading reaches rank 1 for both).

### A refused pose and a wrong answer are different failures

`probe_viewpoint.py` wrote both as `null`, which made two things unreadable:
its own `unreachable` column, and any policy that skipped nulls. Skipping "no
rank" reads the answer — the robot cannot know its top-1 is wrong. Skipping a
refused POSE is legitimate, because THOR refusing a teleport is something the
robot observes; that retry belongs in `eval_move.walk`, which now escalates the
arc radius through `RADIUS_LADDER` until one is accepted. The two values are now
distinct (`"unreachable"` vs `null`).

Worth knowing: the ladder changed **0 of 280** case-by-angle outcomes at radius
1.5, so reachability was never the constraint on these scenes. Every `null` in
`results/probe_viewpoint_ladder.json` is a perception failure.

### Small samples decided direction twice, wrongly

Both of these were reported as findings and then reversed by the full list:

* `--assets 5` sampled the asset pool ALPHABETICALLY, not at random, giving cup
  4/5 (p 0.042) and a recommendation of `bottle`. The full 34-asset pool gives
  cup 33/34 (p 0.342).
* On 5 cases, dropping `s` from the score took the concentration rule from
  2/5 to 4/5 and side agreement to 5/5. On 38 cases `rel` and `rel * s` are a
  wash (26 vs 28, 25 vs 23, 31 vs 32).

### Nothing important lives outside the repository

`/tmp` and the session scratchpad were wiped three times, each time destroying
working probes and their outputs. A measurement that took forty minutes of GPU
time belongs in a tracked file before it is analysed, not after.
