# SGG ground truth and the occlusion benchmark

**This file is the record.** It carries every result, the reasoning behind each
design decision, and — deliberately — the findings that were later withdrawn.
Read it before changing anything; several obvious-looking "improvements" have
already been tried and measured to fail.

Read in order:

1. **What counts as a relation.** THOR gives metric geometry; VG150 recall is
   scored against what *human* annotators chose to write down. The two disagree,
   and calibrating to the humans rather than to the geometry — or to the model —
   is what makes any recall number mean something.
2. **How the current dataset is built.** The complete build specification for
   `occlusion_ds4` — one command, every constant it resolves to, the six steps,
   what counts as a relation, and the two build-time filters. **If you only read
   one section, read that one.** It is self-contained by design.
3. **The datasets, in version order.** `occlusion_ds2` → `ds3` → `ds4`: what each
   build changed, what it bought, and why the headline four-band occlusion table
   is confounded and must not be quoted as it stands.

An embodied experiment (Task 1 — height-conditioned object search) also lived
here and has been removed along with its results.

## Known infrastructure constraints

- **VG150 cannot name most of a kitchen** — no `microwave`, `fridge`, `toaster`,
  `spoon`, `knife`, `mirror`, and no `key`/`keychain`. Coverage is 63% of
  objectTypes in FloorPlan1 and 77% in FloorPlan201, and both figures include
  loose mappings (`Sofa`→`seat`, `Television`→`screen`, `Pan`→`pot`).
- **VG150 classes are coarse, causing instance ambiguity.** `lamp` covers both
  `DeskLamp` and `FloorLamp` (3.6 m apart), `table` covers three THOR types,
  `cup` covers `Mug` and `Cup`.
- **Camera height is not free on the acting agent.** The default iTHOR agent
  exposes only `Stand` / `Crouch` (~1.576 m / ~0.95 m eye height). Third-party
  cameras allow any height but do not affect `visible` metadata. This is why the
  dataset sweeps use a third-party camera — see `HEIGHT_RANGE`.

---

## Ground truth must follow VG's annotation convention, not metric geometry

Measured on 200 reference views of `multiview/` against EGTR's predictions
(`multiview_ref/`), R@100 was 8.6% overall but **44.1%** on `on` alone — higher
than EGTR's own `on` recall on VG150 (~0.37). So there is **no domain gap**: the
model transfers to THOR renders fine. `above`, `under` and `in` scored exactly
0.0% and `near` 2.1%, and that was a fault in our ground truth, not the model.

The cause: our GT was defined with 3D metric thresholds (`near` ≤ 0.5 m,
`above`/`under` a 0.15–1.0 m gap with the host's footprint over the target,
`behind` by true depth with 2D overlap in [0.4, 0.9]). **A VG annotator had no
depth information at all** — they worked from one photograph. Every threshold was
geometrically defensible and disagreed with the human convention. `on` was the
accident that worked, because "subject's box inside the object's box" happens to
be what `parentReceptacles` produces.

### Calibrate to human annotations, never to the model

`vg_conventions.py` measures seven scale-invariant 2D box features over all
315642 VG train relations; `vg_gt.py` re-derives THOR GT to match, keeping the 3D
derivation so every emitted relation stays *true* of the scene (3D contradiction
rate: 0.0–0.1%). Calibrating against EGTR's *predictions* would have made the
benchmark measure agreement-with-EGTR, so an improvement to EGTR could no longer
be detected by it. Human labels are a legitimate target; model output is not.

EGTR reproduces the human convention closely on THOR imagery — medians agree to
within 0.02 on `dy` and exactly on `below` — which is why calibrating to humans
raises recall without the circularity.

### What the human data said, including where intuition was wrong

| measurement | finding | design consequence |
|---|---|---|
| `cover` median for `on` | 1.000 | why `on` already worked |
| `below` for `behind`/`in front of` | exactly 0 / exactly 1 | convention is the **monocular ground-plane cue**, not true depth |
| frame-height vs true depth in THOR | agree 78% (Δdepth > 0.5 m) | keep true depth — correct, and mostly convention-compatible |
| both directions of a pair labelled | only 2–7% | old GT emitted **both**, doubling the denominator (38/38, 368/368) |
| subject is the smaller box | 70–74%, all four predicates | direction rule; reproduces VG's `above`:`under` 1.6:1 as a *consequence* |
| same-class pairs | 2.4% of labels vs 9.5% available (0.25×) | exclude, except `near` (7.9% ≈ baseline) |
| both objects large | 21.8% of labels vs 7.8% available (**2.79×**) | **do not filter.** `drawer under cabinet` looks like part-whole, but humans prefer large-large pairs — the filter intuition suggested would have moved *away* from the convention |
| annotated fraction of available pairs | 10% (5.5 relations / 55 pairs) | the selection criterion (see below) |

### Selectivity is annotation rate, not a definition

Per-feature bands over-generated 6×: 33 relations/view against VG's 5.5. Passing
five one-dimensional tests separately is much easier than sitting inside the
five-dimensional cloud, so selection uses the **joint** likelihood under the
fitted human distributions (`vg_prior.json`).

Even that does not close the gap — a cut rejecting half of VG's *own* annotations
still admits 89% of THOR pairs, because an indoor scene is structured enough that
almost every pair genuinely looks typical. Our views have *fewer* candidates than
VG images (median 9 objects / 36 pairs vs 11 / 55); the entire difference is that
VG labels 10% of pairs and exhaustive derivation labels 89%. **What VG leaves out
is not untrue, it is unremarked**, and which 10% an annotator remarks on is
attention — not recoverable from geometry.

So VG's measured *rate* becomes the cut-off: rank by human typicality, keep the
top 10%. Both the ordering and the cut come from human statistics.

### Result

`eval_recall.py --summary`, 200 reference views, IoU 0.5. **Report mR as well as
R** — `on` is 24% of the ground truth and the only predicate whose original
definition already agreed with VG, so R@K alone is close to a measurement of `on`.

| GT | config | n | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 |
|---|---|---|---|---|---|---|---|---|
| A | exact match | 1377 | 6.5 | 8.4 | 9.1 | 3.9 | 5.2 | 5.6 |
| A | synonyms A | 1377 | 7.8 | 9.7 | 10.2 | 4.7 | 5.9 | 6.3 |
| A | synonyms A, strict | 907 | 9.0 | 10.8 | 11.2 | 5.7 | 6.8 | 7.1 |
| B | exact match | 1360 | 7.4 | 9.9 | 11.7 | 4.7 | 8.4 | 9.7 |
| **B** | **synonyms A** | **1360** | **8.8** | **11.5** | **13.2** | **5.9** | **9.6** | **10.9** |
| B | synonyms A+B | 1360 | 9.5 | 12.9 | 14.9 | 6.6 | 10.8 | 12.6 |
| B | synonyms A, strict | 948 | 9.9 | 12.9 | 14.9 | 7.0 | 13.8 | 15.3 |

Per predicate on the recommended row (GT B, synonyms A):

| predicate | GT | R@20 | R@50 | R@100 | was (GT A) |
|---|---|---|---|---|---|
| `on` | 322 | 32.6% | 39.4% | 40.7% | 40.7% |
| `under` | 410 | 2.2% | 4.6% | 6.8% | **0.0%** |
| `above` | 270 | 1.1% | 1.5% | 4.1% | **0.0%** |
| `near` | 39 | 5.1% | 10.3% | 12.8% | 2.1% |
| `in front of` | 9 | 0.0% | 11.1% | 11.1% | 1.4% |
| `behind` | 304 | 0.0% | 0.3% | 1.0% | 0.0% |
| `in` | 6 | 0.0% | 0.0% | 0.0% | 0.0% |

Density is unchanged (6.9 → 6.8 relations/view) and the predicates that were
structurally unreachable now register. **mR gains far more than R** — +73%
against +29% — because the fix is entirely in the tail.

Two things follow for the fusion experiment. Track **mR@100**, not R@100: R is
dominated by `on`, which saturates by rank 20 and will not move. And do not use
**R@20 at all** — tail hits land between ranks 20 and 100 (mR 4.7 → 9.7 across
that range), so R@20 is blind to exactly the effect being tested.

Set expectations honestly: this is +2 pp. **Definitional mismatch was never the
main loss.** 59% sits in detection/localisation and 20% in pair proposal — 79%
upstream of the relation head, which is where multi-view fusion acts.

`behind` stays near zero for two compounding reasons: EGTR barely predicts it
(1.4% of its output) and we keep true depth where the convention uses frame
height. That disagreement covers 22% of pairs and is the one place our GT is
deliberately *more correct* than VG.

### The 2D feature model cannot name a relation — the class pair can

The convention fix above raised `above`/`under` off the floor on `multiview` but
they stayed at 0.0% on `occlusion_ds2`. Diagnosing that turned up a fault one
level deeper: **the fitted 2D distributions in `vg_prior.json` do not carry
enough signal to choose a predicate at all**, and `vg_gt._proposals` was using
them for exactly that.

The test is direct — run the fitted model back over VG's own held-out
annotations and ask it to recover the human's label:

| predicate | n | argmax accuracy |
|---|---|---|
| `on` | 11911 | 73.9% |
| `in` | 1739 | 5.9% |
| `near` | 1893 | 8.2% |
| `above` | 522 | 36.0% |
| `under` | 231 | 67.1% |
| `behind` | 517 | 28.2% |
| `in front of` | 498 | 6.4% |
| **overall** | **17311** | **55.3%** |

Against a 68.8% always-say-`on` baseline. **It is worse than a constant.** The
cause is visible in the parameters: `above` has `dy` mu −0.175 sd 0.196,
`behind` −0.121 sd 0.203, `near` −0.021 sd 0.220, and `cover` sits in
0.37–0.51 for all seven. The distributions overlap ~90%, so both the family
competition and the top-10% ranking were close to arbitrary.

**The signal it was missing is the class pair.** The same test using only
`P(predicate | subject class, object class)`, fitted on train and applied to
val, scores **80.8%**. What a human calls a pair is mostly a fact about what the
two things *are*:

| class pair | n in VG train | human labels |
|---|---|---|
| `chair`–`table` | 805 | `near` 48%, `at` 31%, `under` 6%, `behind` **4%** |
| `bowl`–`table` | 458 | `on` 78%, `above` 12%, `sitting on` 7% |
| `vase`–`table` | 295 | `on` 85%, `above` 14% |
| `lamp`–`chair` | 29 | `near` 52%, `behind` 38% |
| `chair`–`plant` | 6 | `near` 100% |

Our ground truth called the `chair`–`table` pairs `behind` **58 times**.

Measured against VG's marginal over *the class pairs our own scenes contain*,
the old derivation was badly miscalibrated:

| predicate | VG, our class pairs | our GT (before) | our GT (after) |
|---|---|---|---|
| `on` | 53.4% | 49.1% | 49.1% |
| `near` | 10.3% | **4.0%** | 13.6% |
| `above` | 8.0% | 13.4% | 3.8% |
| `under` | 3.8% | 5.5% | 0.8% |
| `behind` | **1.7%** | **25.1%** | 25.0% |
| `in front of` | 1.3% | 2.9% | 7.9% |

`behind` was over-emitted **15×**.

#### Why aligning to EGTR was safe here

`vg_gt.py` warns against calibrating to model output, and that warning stands.
But the disagreements EGTR flagged turned out to be the *same* ones VG's human
counts flag, independently:

| our GT said | EGTR said | VG humans say, same class pair |
|---|---|---|
| `behind chair-table` | `near` (132×) | `near` 48% of 805 |
| `above bowl-table` | `on` | `on` 78% of 458 |
| `above vase-table` | `on` | `on` 76% of 327 |
| `on plate-table` | `under(table, plate)` (79×) | reverse ordering carries `on` 49% |
| `on X-table` | `has(table, X)` (50×) | reverse ordering carries `has` **34%** |

So EGTR was used to find *where to look*, and every substitution was then kept
or rejected on VG's counts. That ordering is what keeps it non-circular, and it
is the same procedure that earlier fixed `Sofa`→`seat` and the
`pickupable`/`moveable` support bug.

#### Two walls, measured separately

Recall is lost in two independent places, and only the second is a definition
problem. `occlusion_ds2`, 150 views, baseline EGTR:

| GT predicate | n | pair localised at all | R@100 given localised |
|---|---|---|---|
| `on` | 391 | 71.6% | 49% |
| `behind` | 200 | 40.5% | ~0% |
| `above` | 107 | 38.3% | 0% |
| `in front of` | 23 | 43.5% | ~0% |
| `near` | 32 | 43.8% | 43% |
| `under` | 44 | **4.5%** | — |

`under` is not a naming problem: EGTR never proposes those pairs, because our
`under` had a median 2D `cover` of **exactly 0.000** — the two boxes did not
overlap at all, against VG's `under` mean cover of 0.426.

#### What changed

1. `vg_pair_prior.py` — `P(predicate | classes)` and pair relatedness over all
   315642 VG train relations, committed as `vg_pair_prior.json`.
2. `vg_gt.py` `rank_by="human"` (new default) — the 10% annotation-rate cut now
   ranks by how often humans relate these two *classes*, not by the geometry
   score. TASKS.md previously concluded that which pairs an annotator remarks on
   is "attention — not recoverable from geometry". Correct, and it *is*
   recoverable from class co-occurrence.
3. `vg_gt.py` `VERTICAL_MAX_HORIZONTAL_M = 0.8` — `above`/`under` now require the
   pair to be stacked. Nothing previously did: a plant on a table qualified as
   `above` a chair standing beside it (median horizontal separation 0.74 m).
4. `eval_occlusion.py --accept human` — accepts wordings VG uses for the same
   class pair, and adds the inverses the matcher was missing (`on(A,B)` ←
   `under(B,A)`, ← `has(B,A)`).
5. `relabel_occlusion.py` — re-derives relations in place from stored JSON, no
   THOR session and no re-render, so definitions can be compared against
   identical images and identical predictions. Reversible.

#### Result, and an honest discount

`occlusion_ds2`, 150 views, IoU 0.5, synonyms A. These isolate the *definition*
change; the current headline numbers are on `occlusion_ds3` further down.

| | R@20 | R@50 | R@100 | mR@100 |
|---|---|---|---|---|
| before | 11.8 | 14.1 | 15.9 | 9.0 |
| after, `--accept exact` | 11.9 | 14.8 | **17.3** | 7.9 |
| after, `--accept human` | 18.8 | 24.2 | **29.7** | 27.0 |

`multiview` GT B, exact match, moves 11.7 → 13.4 R@100 and 9.7 → 10.8 mR@100
from the selection change alone.

**Discount the second row deliberately.** Widening an accept-set raises recall
whatever it accepts. Control: a *random* accept-set of the same per-relation
size scores 25.1% ± 0.7 (10 draws). So of the +12.4 points, **7.8 is
permissiveness and 4.6 is convention**. `--accept exact` stays the default and
the headline; `human` is the upper bound on what wording disagreement costs.

So the definition fix is worth **+1.4 points**, not +14. Consistent with the
earlier finding that 79% of the loss sits upstream in detection and pair
proposal — which the localisation table above confirms directly.

#### Root cause found: the benchmark never annotated fixed fittings

The section below correctly identified that `occlusion_ds2` contains almost no
genuine `above`/`under` configurations, and attributed it to clutter being
scattered flat over surfaces. That was only half of it. The other half:

**`occlusion_pipeline.arrange()` restricted its target list to `moveable()`
objects**, so no cabinet, counter, drawer, window, sink or toilet was ever
annotated in any view. `occlusion_ds2` carries **18 VG150 classes against the 38
its scenes contain**, and not one of them is a fixed fitting.

That is why relabelling could not recover the vertical family: `drawer under
counter` (80% of 84 human VG labels) and `cabinet under counter` (39% of 70) are
the configurations VG annotates most, and *both endpoints are static*. The
objects were not in the file.

The restriction assumed the amodal pass — disable everything, re-enable one
object at a time — could only toggle moveable objects. **It cannot.**
`DisableObject` succeeds on `Cabinet`, `CounterTop`, `Drawer`, `Fridge`, `Sink`
and `Window`, all of which report `moveable=False, pickupable=False`. Verified
directly before changing anything.

A second, quieter bug fell out with it. `build_scene` disabled only the moveable
objects during the amodal pass, so the fixed fittings stayed standing and
**truncated the reference box of anything behind them** — understating both the
amodal extent and the occlusion of every object behind a counter. Both now
disable the full nameable set.

Measured on one scene-seed (`FloorPlan7` s3, 10 views), changing nothing else:

| | objects/view | relations | `above` | `under` | classes |
|---|---|---|---|---|---|
| moveable only | 6.4 | 27 | 0 | 0 | 9 |
| **+ fixed fittings** | **14.0** | **147** | **19** | **16** | **15** |

One scene-seed now yields more vertical relations than all 150 views of
`occlusion_ds2` (30 `above`, 6 `under`). The pairs are the human-annotated ones:
`drawer above cabinet` ×6, `cabinet under counter` ×3, `sink under cabinet` ×4,
`near cabinet-window` ×6.

Note the density this implies: 14.7 relations/view against VG's 5.5. The
geometric part is still at exactly the 10% annotation rate (9.8 of 91 available
pairs); the excess is the `on`/`in` support edges, which bypass the cap because
`parentReceptacles` asserts them. Effective rate is 16% against VG's 10%.
Consider subtracting the support count from the budget so the cap means what it
says.

#### Scene selection is now measured, not hand-applied

`survey_scenes.py` ranks all 120 floor plans by
`nameable_moveable x min(surfaces, seeds)` and picks by room-size bucket. Run
against the hand-picked living rooms it reproduces three of five (201, 215, 230
against 201, 204, 215, 218, 230), which is close enough to trust it on the room
types nobody inspected by hand.

| family | picked | nameable | reachable |
|---|---|---|---|
| kitchen | 7, 1, 20, 5, 3 | 20–27 | 110–324 |
| living | 201, 215, 230, 227, 228 | 23–30 | 172–468 |
| bedroom | 311, 326, 313, 318, 328 | 16–20 | 59–266 |
| bathroom | 429, 430, 420, 419, 427 | 9–11 | 33–118 |

Bathrooms are thin by construction (9–11 nameable moveable objects against a
kitchen's 20–27, 3.1 relations/view against 8.8) and are included for
`sink`/`toilet`/`towel` coverage rather than volume.

`build_occlusion_dataset.py --rooms all` builds all four families.

#### The real limit is scene composition, not annotation

`occlusion_pipeline.py` scatters clone objects *across horizontal surfaces*,
physically settled. Everything therefore ends up coplanar, and the numbers show
it: `behind` pairs have a median true vertical separation of **−0.002 m** and a
horizontal separation of 1.56 m — objects side by side on one table, which the
derivation then ordered by depth and called `behind`. Nothing is ever placed
*over* anything, so genuine `above`/`under` configurations essentially do not
occur, and the 107 `above` / 44 `under` labels were incidental geometry.

**No relabelling can fix this.** To measure the vertical family the generator
has to stage it, so `above`/`under` describe a real spatial fact with 2D `cover`
in VG's range rather than 0.000.

*Resolved by `occlusion_ds3` — see below.* The staging turned out not to need
clones placed on shelves at all: the vertical relations were already in the
scenes, between the fixed fittings the target list was excluding.

---

### Class mapping errors are not synonym problems

Every class in the GT is a valid VG150 word (40 used, 0 invalid). The asymmetry
runs the other way: **62.7% of the objects EGTR uses in its top-100 triplets match
no GT object at all**, so the effective budget is ~27 triplets, not 100.
Breakdown of the 20000 triplets over 200 views: 18.9% touch an architectural
surface, 54.9% touch something else unmatched. The top offenders are `tile`
(5693 predictions, 126 matched), `light` (4568/147), `room` (2486/132), `roof`
(847/13), plus `handle` and `cat`.

Root cause: **VG150 has no `floor`, `wall` or `ceiling`.** EGTR detects the floor
and, lacking the word, says `tile` or `room`; the ceiling becomes `roof`. Those
are correct detections in VG's vocabulary against a GT that has no such object —
THOR's `Floor` was never in our object set. Not recommended to add it: any choice
of word is arbitrary, and labelling the floor `tile` would put half the scene
`on tile` and corrupt the one predicate that works.

This is dilution, not a ceiling: 27 effective triplets against 6.8 GT relations
still leaves 4x headroom. It is a contributor to the 59%/20% figures above rather
than a fourth independent loss.

Two mappings were measurably wrong and are fixed (`relabel_multiview.py` reapplies
a mapping change in place — pure function of `thor_type`, no THOR instance, no PNG
touched):

| was | now | evidence |
|---|---|---|
| `Sofa`/`Stool`/`Footstool` → `seat` | → `chair` | `seat` named 0 of 29 Sofas and 0 of 12 Stools; `chair` took 15 and 5. GT `seat` scored 0.0% agreement over all 41 instances. `seat` is predicted 234 times — on Chairs (29) and ArmChairs (14). It names a chair's seat, not seating furniture. |
| `GarbageCan` → `basket` | → `box` | `basket` 7 of 57, against `box` 16 and `bag` 9 |

Two candidates with *larger* apparent margins were deliberately rejected, because
the mapping table must follow VG's vocabulary and not the model's mistakes:

- `Curtains` → `window` (12 v `curtain` 4). The model is confusing a curtain with
  the window behind it. Sibling mappings confirm the current one is right:
  `Blinds`→`curtain` hits 21, `ShowerCurtain`→`curtain` hits 6.
- `Lettuce` → `fruit` (5 v `vegetable` 0). A lettuce is a vegetable and
  `vegetable` is in VG150. This is a model error at n=5.

The fix moved **exact match** (8.6 → 9.1 on GT A, 11.2 → 11.7 on GT B) and left
the synonym rows unchanged, because the synonym groups were already covering for
it. That is the point: it is a correctness fix, not a recall win, and it makes the
exact-match column mean what it says.

It also collapsed synonym tier A from five groups to two. With the mapping
corrected, {chair, seat, bench}, {lamp, light} and {bag, basket, box} contributed
+0.1, 0.0 and 0.0 points of R@100 — they existed only to paper over the bad
mapping. Trimming them cost 0.1 points (13.3 → 13.2) and removed three arbitrary
concessions. What remains is {table, desk} (+1.2, the only substantial group) and
{cabinet, drawer} (+0.2).

---

## How the current dataset is built

**This section is the complete build specification for `occlusion_ds4`.** Every
value used is stated here; nothing needs to be looked up elsewhere. The sections
after it are history (what earlier versions did differently) and the section
before it is rationale (why the relation definition is what it is).

One command builds it:

```bash
python build/sgg/build_occlusion_dataset.py --rooms all --out occlusion_ds4
```

Everything that command resolves to, from the constants and CLI defaults in
`build/sgg/build_occlusion_dataset.py`:

| | value |
|---|---|
| room families (`--rooms all`) | kitchen, living, bedroom. **Not bathroom** — excluded from `DEFAULT_FAMILIES`, buildable with `--rooms bathroom` |
| kitchens | FloorPlan 7, 1, 10, 5, 16, 17, 18, 21, 8, 6 |
| living rooms | FloorPlan 201, 215, 230, 204, 203, 227, 218, 224, 223, 228 |
| bedrooms | FloorPlan 311, 326, 330, 307, 323, 321, 328, 302, 303, 305 |
| seeds | 1, 2, 3 → 90 scene-seed records |
| views per record | 10 → 900 views |
| resolution | 800 × 600 |
| field of view | 60° **vertical** |
| camera height | uniform in [1.0, 1.5] m per view (`HEIGHT_RANGE`) |
| camera distance | horizontal [1.2, 2.0] m to the focus surface (`radius`) |
| clutter added | 16 small + 6 floor-standing duplicates (`small_extra`, `large_extra`) |
| surfaces cluttered | 3 (`--surfaces`), min spacing 0.18 m |
| occlusion ceiling | 0.90 (`MAX_OCCLUSION`) |
| size floor | 1728 px = `0.06²` × 800 × 600 (`MIN_EXTENT_FRACTION`) |

Scene lists were picked by `survey_scenes.py`, ranking all 120 floor plans by
`nameable_moveable × min(surfaces, seeds)` and then requiring **distinct focus
regions ≥ seeds** — of the first 30 candidates, 11 offered fewer than three
separable regions and FloorPlan301 offered exactly one, which would have made all
three of its seeds the same sample.

### The six steps

`build()` in `build/sgg/build_occlusion_dataset.py`, run once per (scene, seed) pair.

1. **Open THOR** with `renderInstanceSegmentation=True` and
   `visibilityDistance=15.0`. Every measurement downstream is a pixel count off
   an instance mask, so segmentation is not optional and the visibility cap has
   to be far enough away not to cull anything in frame.

2. **`pick_focus`** — choose the one surface this seed builds around, from the
   receptacles ≥ 0.15 m², deduped by ground position, indexed `seed % n`. The
   focus fixes everything downstream: which surfaces get clutter, where furniture
   may stand, and what every camera aims at. It used to be whichever surface was
   nearest THOR's default agent spawn, which is a property of the *scene* and not
   of the seed — three seeds then rebuilt and photographed the same corner.

3. **`arrange`** — add clutter, leaving the room physically settled.
   `auto_spawn` reads what the scene already contains and asks
   `InitialRandomSpawn` for duplicates of it (16 small, 6 floor-standing by
   default), skipping any type `THOR_TO_VG150` cannot name — an object that can
   occlude but never be scored just spends the placement budget. The scene's own
   objects are snapshotted first and restored afterwards, because
   `InitialRandomSpawn` relocates about 30 of a scene's 77 objects as a side
   effect of duplicating anything. Each duplicate goes in with
   `PlaceObjectAtPoint` and is **verified by a non-empty `parentReceptacles`**,
   retried once, and disabled if it still will not seat — `lastActionSuccess`
   reports success for a placement it resolved by shoving the object upwards, so
   it cannot be trusted on its own.

4. **`camera_poses`** — take reachable positions whose *horizontal* distance to
   the focus falls in [1.2, 2.0] m, aim each at the arrangement with jitter, and
   draw the height uniformly from `HEIGHT_RANGE`. Aiming rather than sampling yaw
   freely: a random heading from a random position usually faces a wall, and an
   empty view costs the same to render as a useful one. If no reachable position
   satisfies the band the sampler falls back to `reachable[:count]` and ignores
   the band entirely — check that this is not firing before tightening the radius.

5. **`build_scene`** — two passes over the same pose list.

   *Everything present.* Aim the third-party camera at each pose, save
   `view_NN.png`, and record every object's visible pixel count and box from the
   instance mask. Object metadata (`position`, `distance`) is read here, with the
   scene whole — these are facts about the scene and must not be read while half
   of it is disabled.

   *One object at a time.* Disable every nameable object, then re-enable one,
   sweep the whole pose list, disable it again, and move on. That yields the
   object's **reference** silhouette: its extent against the static room with
   nothing else in the frame. Hence

   ```
   occlusion = 1 - visible_px / reference_px
   ```

   The disabled state survives a camera move, which is why the loop is
   per-target-then-per-pose rather than the reverse. `DisableObject` costs 21 ms
   against `SetObjectPoses`' 38 ms, and unlike `SetObjectPoses` it does not
   invalidate every objectId.

6. **`annotate_relations`** — `on` and `in` come from `parentReceptacles`, which
   the simulator asserts directly. Everything else is derived by
   `vg_gt.vg_calibrated_relations` from the **amodal** boxes. Each relation
   carries both endpoints' occlusion and an `annotation` field recording which of
   the two paths produced it.

`scene.json` is then written beside the PNGs.

### What counts as a relation

Step 6 in full, because it is the part most easily misread. Relations arrive by
two separate paths.

**`on` and `in` come from `parentReceptacles`**, which the simulator asserts.
Nothing is derived; these are the one predicate family whose 3D definition already
agreed with VG's annotation convention.

**Everything else is derived in two stages**, and the split is the whole design:

> **3D geometry decides whether a relation is TRUE. Human class-pair statistics
> decide whether it is worth ANNOTATING.** The 2D bands never invent a relation —
> they only decide which true ones a human would have bothered to write down.

That split exists because the 2D box features *cannot name a relation*: run back
over VG's own labels they recover the human's predicate 55.3% of the time against
a 68.8% always-say-`on` baseline, i.e. worse than a constant, while the class pair
alone recovers 80.8%. See the section above for the full measurement.

The gates, all from `vg_gt.py`:

| constant | value | what it does |
|---|---|---|
| `MIN_LINEAR_EXTENT` | 0.06 | an object below this fraction of the frame's linear extent is not a candidate. VG annotators drew boxes down to √area = 0.057 of the frame at the 5th percentile |
| `VG_ANNOTATION_RATE` | 0.10 | rank candidate pairs by human typicality, keep the top 10%. The measured rate *is* the selection criterion — a median VG image has 11 objects (55 pairs) and 5.5 relations |
| `SAME_CLASS_EXEMPT` | `{near}` | same-class pairs are dropped (humans relate them at 0.25× the base rate), except `near`, where 7.9% ≈ the 9.5% baseline |
| `NEAR_MAX_3D_M` | 3.0 m | a loose bound so `near` cannot fire across a room. Not the old metric threshold doing the work — at 3 m it excludes opposite-wall pairs and nothing else |
| `VERTICAL_MIN_DY_M` | 0.05 m | `above`/`under` sign check with centroid-jitter tolerance. Magnitude is left to the 2D band |
| `VERTICAL_MAX_HORIZONTAL_M` | 0.8 m | "above" means one thing is *over* another, not merely higher somewhere else in the room |

Direction is not free either: the subject is the smaller box (measured at 70–74%
across all four predicates), and only one direction of a pair is emitted — humans
label both directions only 2–7% of the time, and the earlier GT emitting both
doubled the denominator.

Derivable predicates are `on`, `in`, `near`, `above`, `under`, `behind`,
`in front of` (`TARGET_PREDICATES` in `vg_conventions.py`). What ds4 actually
contains: `on` 3104, `behind` 1354, `near` 968, `above` 763, `in front of` 536,
`under` 510, and **`in` 0** — THOR containers stay closed, so nothing is ever
visibly inside anything.

Two switches exist and are **off** by default. `rank_by="human"` is on (it is the
10% ordering above). `name_by="human"` would also pick the predicate by class-pair
argmax: it scores higher (R@100 22.2% against 15.9% on `occlusion_ds2`, exact
match) but collapses the ground truth onto `on` and `near` — `behind` falls
199 → 8 and `under` 44 → 2 — which makes mR@K unreportable. Do not turn it on to
improve a number.

### Two filters, and occlusion is not one of them

Only two things are dropped at build time:

| filter | value | why |
|---|---|---|
| `reference_px < MIN_EXTENT_FRACTION × W × H` | `0.06²` → 1728 px at 800×600 | below this the ratio measures mask aliasing, not occlusion. An unfiltered first run called a 359 px plate "88% occluded", which says more about how far away it was |
| `1 − visible/reference > MAX_OCCLUSION` | 0.90 | nothing recovers an object with almost no pixels left |

**Occlusion itself never filters anything.** A relation that is true of the scene
is annotated whether or not this particular camera can see it. The unoccluded
relations are the control the occluded ones are measured against, and stratifying
recall by occlusion needs the whole range present — that is what makes the
annotation *amodal*, and it is the reason the dataset can ask whether extra views
recover what one view cannot.

### Three decisions that are easy to get wrong

**Relation geometry uses the amodal box, not the visible one.** `vg_gt` decides a
predicate partly from `cover` and `area_ratio`, and the visible box of a heavily
occluded object is a fragment. Using it would have the same relation classified
differently from different viewpoints — an artefact of the occlusion rather than a
fact about the scene.

**Both boxes are stored anyway.** A detector shown an 80%-occluded object predicts
roughly its *visible* extent, so scoring that against the amodal box fails at any
sensible IoU. Scoring against the visible box alone has the opposite problem: it
shrinks as occlusion grows, so IoU gets *easier* the more hidden the object is,
rewarding exactly what the benchmark exists to penalise. `eval_occlusion.py`
therefore takes whichever of the two gives the better IoU.

**The agent is parked out of shot before every frame.** Third-party cameras render
the agent's body — 4.3% of the pixels in one 800×800 view. It is not cosmetic: the
agent occludes real objects, so part of the "occlusion" being measured was the
robot itself, and because it has no entry in `metadata["objects"]` it appears in no
instance mask and `DisableObject` cannot touch it. `park_agent` puts it behind the
camera plane, re-parked per pose because "behind the camera" moves with the camera.

### What each record stores

`scene.json` is self-describing on purpose, so annotations can be re-derived
without THOR (`relabel_occlusion.py`) and so a record cannot be misread later:
`seed`, `min_reference_px`, `max_occlusion`, `intrinsics` (fx = fy =
(h/2)/tan(vfov/2), principal point at centre, no skew — THOR exposes no way to
change either), and `convention` (Unity left-handed, Y up, yaw clockwise from +Z,
`fieldOfView` is **vertical**). `added` and `dropped` record which clutter seated
and which would not. Per view: `camera`, then per object `bbox_amodal`,
`bbox_visible`, `reference_px`, `visible_px`, `occlusion`, `distance`,
`vg150_class`, `parent_receptacles`, `moveable`, `pickupable`; then per relation
`subject`/`predicate`/`object`, both endpoint occlusions, and `annotation`
recording which of the two paths produced it.

### What this produces, and what it cannot

900 views, 8125 object-views, 7235 relations, 1123 unique objects, 35 VG150
classes, 8.0 relations per view. The per-scene-seed yields for the actual build
are in `experiments/build_logs/ds4_build.log`; the family breakdown and the
recall results are in the `occlusion_ds4` section below.

Three limits are properties of this build method, not of the data volume, so
raising `--seeds` or `--views` will not move them:

- **`in` is unobtainable.** THOR containers stay closed. Fixing it means opening
  cabinets and fridges before capture.
- **Clutter ends up coplanar.** `arrange` scatters duplicates across horizontal
  surfaces and lets physics settle them, so `behind` pairs have a median true
  vertical separation of −0.002 m — objects side by side on one table, ordered by
  depth and called `behind`. Genuine `above`/`under` comes from the fixed fittings
  (`drawer under counter`, `cabinet under counter`), not from the clutter.
- **Multi-view headroom is 236 relations.** Of 7235, 1525 are hard (an endpoint
  ≥ 0.5 occluded) and only 236 of those are recoverable from some other view. Any
  multi-view or NVS claim measured here rests on that subset.

Deeper background, if the above is not enough: *Root cause found: the benchmark
never annotated fixed fittings* (why `arrange` includes fittings and why the
amodal pass disables them too), *Scene selection is now measured, not
hand-applied* (how the lists were derived), *The real limit is scene composition,
not annotation* (the coplanarity finding in full) — all earlier in this file,
under the ground-truth story.

---

## `occlusion_ds3` — the dataset the fixes produced

`build_occlusion_dataset.py --rooms all --out occlusion_ds3`, 20 scenes x 3 seeds
x 10 views over all four room families. `occlusion_ds2` is left in place for
comparison. 59 of 60 scene-seeds are usable; seed variance is wide (best 159
relations, worst 7 — `pick_focus` sometimes rotates onto a surface with nothing
around it, so ~8 of 60 seeds contribute almost nothing).

| | `occlusion_ds2` | `occlusion_ds3` |
|---|---|---|
| views | 150 | **600** |
| unique objects | 193 | **994** |
| relations | 797 | **3737** (4.7x) |
| VG150 classes | 18 | **37** |
| relations / view | 5.3 | 6.2 (VG: 5.5) |
| `above` | 30 | **337** |
| `under` | 6 | **215** |
| `near` | 108 | 582 |

| family | views | obj/view | rel/view | classes | `above` | `under` |
|---|---|---|---|---|---|---|
| kitchen | 150 | 9.2 | 7.0 | 22 | 135 | 129 |
| living | 150 | 8.7 | 7.6 | 21 | 59 | 36 |
| bedroom | 150 | 7.7 | 6.8 | 23 | 67 | 36 |
| bathroom | 150 | 5.1 | 3.6 | 13 | 76 | 14 |

Kitchens carry the vertical family as predicted — 264 of the 552 `above`/`under`
relations. Bathrooms are thin exactly as `survey_scenes.py` said and are there
for `sink`/`toilet`/`towel` coverage, not volume.

**`in` is still zero across all 600 views.** THOR containers stay closed, so
nothing is ever visibly inside anything, while EGTR spends 14.8% of its output
budget on `in` — its third most common predicate. Fixing it means opening
cabinets and fridges before capture.

### Result — baseline EGTR on `occlusion_ds3`

| | `occlusion_ds2` | `occlusion_ds3` |
|---|---|---|
| R@100, `--accept exact` | 17.3% | **20.1%** |
| mR@100, `--accept exact` | 7.9% | **12.4%** |
| R@100, `--accept human` | 29.7% | 29.3% |
| mR@100, `--accept human` | 27.0% | 24.6% |
| R@100, class-agnostic | — | 33.2% |

mR@100 up 57% at exact match, and every predicate now rests on a usable `n`:

| predicate | GT | R@100 exact | R@100 human |
|---|---|---|---|
| `on` | 1622 | 38.3% | 41.4% |
| `behind` | 726 | 1.5% | 20.8% |
| `near` | 582 | 10.0% | 15.3% |
| `above` | 337 | 2.7% | 14.2% |
| `in front of` | 255 | 10.2% | 38.4% |
| `under` | 215 | **11.6%** | 17.7% |

`under` was 0.0% on n=6; it is now 11.6% on n=215. `behind` stays near zero at
exact match for the reason recorded earlier — EGTR barely predicts it, and our
derivation keeps true depth where the convention uses frame height.

### THE OCCLUSION TABLE IS CONFOUNDED — do not quote it as it stands

This is the measurement the dataset exists to make, and the marginal band table
cannot support it:

| band | GT | R@100 | object detection |
|---|---|---|---|
| clear (0.00–0.05) | 513 | **5.3%** | 59.4% |
| light (0.05–0.25) | 845 | 23.9% | 62.8% |
| moderate (0.25–0.50) | 1407 | **25.6%** | 57.9% |
| heavy (0.50–0.81) | 972 | 16.6% | 41.0% |

Recall *peaks at moderate occlusion*. The `clear` band is not the easy control
the design assumes — it is a different population, for three measured reasons:

1. **Predicate composition.** `clear` is 10% `on`; `moderate` is 53% `on`.
   Weighting each band by per-predicate recall alone predicts 10.2 / 20.6 /
   23.5 / 21.1%, which is almost exactly the observed shape.
2. **Object scale.** `clear` objects are 40% smaller — median area fraction
   0.0136 against 0.0225. An object nothing occludes in a cluttered room is
   usually one standing alone, small and far away.
3. **Pair proposal.** Pairs are localised at **30.8%** in `clear` against ~60%
   in light and moderate, at identical median 3D separation (0.58–0.62 m).
   EGTR proposes pairs among objects that are *visually entangled*; endpoints
   that do not occlude each other rarely get a triplet at all.

Point 3 is the structural one: **endpoint occlusion is itself the cue that gets
the pair proposed**, so it cannot serve as a clean independent variable. Less
occlusion removes the evidence that the two objects are related.

Holding predicate *and* endpoint area fixed (smaller endpoint in 0.008–0.055 of
frame) gives the curve that survives:

| `on`, area-matched | light | moderate | heavy |
|---|---|---|---|
| R@100 | 41.7% (n=187) | 40.4% (n=329) | **31.0%** (n=200) |

**The defensible result is a 26% relative drop from light to heavy with
composition and scale controlled.** The `clear` → `light` rise is an artifact.
Report the controlled curve; do not quote the four-band marginal.

### Next: make `clear` a real control

The band comparison should be paired, not cross-sectional. The pipeline already
renders every target alone during the amodal pass, so an unoccluded counterpart
of each relation is nearly free — the same objects, the same predicates, the same
pixel scale, with the occluders removed. That converts a confounded four-band
table into a within-relation comparison and removes all three confounds at once.

---

## `occlusion_ds4` — current, 900 views

`build_occlusion_dataset.py --rooms all --out occlusion_ds4`, **30 scenes x 3
seeds x 10 views**, 800x600, vfov 60. All 90 scene-seeds are usable, against 59
of 60 for `occlusion_ds3`. Per-scene-seed yields are in
`experiments/build_logs/ds4_build.log` rather than repeated here.

Two generator constants changed. Both are documented at their definitions in
`build/sgg/build_occlusion_dataset.py`; what follows is what they did to the data.

**`--rooms all` no longer means all four families.** `DEFAULT_FAMILIES` is
kitchen / living / bedroom at **ten** scenes each, ranked by `survey_scenes.py`
and then filtered on a criterion the five-scene lists never needed: *distinct
focus regions ≥ seeds*. Of the first 30 candidates, 11 offered fewer than three
separable focus regions and FloorPlan301 offered exactly one — all three of its
seeds would have been the same sample. Seeds sharing a focus are correlated, so
counting them as independent overstates n, which is the trap already recorded
above for poses within a target.

Bathrooms are excluded, not deleted. On ds3 their 150 views bought 540 relations
of which 297 were `on` and 14 `under`, and FloorPlan420 has only 14 reachable
positions inside the camera band, so all ten of its views came from nearly one
spot — a room whose viewpoints cannot vary is not a multi-view occlusion sample.
The cost is that `toilet` and `towel` leave the vocabulary entirely.
`--rooms bathroom` still builds them.

**Camera height is a uniform 1.0–1.5 m band**, not three discrete levels (0.85,
1.10, 1.35). Ten views drawn from three levels replicated each height 3–4 times
per scene, so height could not be used as a covariate; and 1.0–1.5 m is the band
a mobile robot's sensor actually sits in.

### The occlusion mix moved, as the `HEIGHT_RANGE` comment predicted

That comment warns that raising the camera floor off 0.85 m *reduces*
inter-object occlusion, and requires the band distribution to be re-checked after
any change to it. Re-checked, as a share of relations by max endpoint occlusion:

| | `<0.25` | 0.25–0.50 | 0.50–0.75 | `≥0.75` |
|---|---|---|---|---|
| `occlusion_ds3` | 36.3% | 37.7% | 22.9% | 3.1% |
| `occlusion_ds4` | 41.6% | 37.3% | **14.7%** | **6.4%** |

**Dropping bathrooms is not the cause.** Recomputing ds3 over
kitchen/living/bedroom only gives 36.0 / 37.3 / 23.5 / 3.3% — unchanged. The
shift is the camera band.

In absolute n nothing got harder to measure, because the dataset is 1.9x larger:
the 0.50–0.75 band grew 857 → 1063 and `≥0.75` grew 115 → **462**. The severe
band being four times better sampled is the one measurement ds4 genuinely fixes;
see the recall note below.

Also unchanged from ds3, and still worth stating: **`in` is zero across all 900
views.** THOR containers stay closed.

### Composition

| | `occlusion_ds2` | `occlusion_ds3` | `occlusion_ds4` |
|---|---|---|---|
| views | 150 | 600 | **900** |
| unique objects | 193 | 708 | **1123** |
| relations | 797 | 3737 | **7235** |
| VG150 classes | 18 | 37 | 35 |
| objects / view | — | 7.7 | 9.0 |
| relations / view | 5.3 | 6.2 | **8.0** (VG: 5.5) |
| `on` | — | 1622 | 3104 |
| `behind` | — | 726 | 1354 |
| `near` | 108 | 582 | 968 |
| `above` | 30 | 337 | 763 |
| `in front of` | — | 255 | 536 |
| `under` | 6 | 215 | 510 |

35 classes against ds3's 37 is the bathroom drop, not a regression.

| family | views | obj/view | rel/view | classes | `above` | `under` | `near` |
|---|---|---|---|---|---|---|---|
| kitchen | 300 | 10.0 | 8.6 | 24 | 461 | 345 | 250 |
| living | 300 | 9.3 | 9.0 | 23 | 132 | 52 | 370 |
| bedroom | 300 | 7.8 | 6.5 | 25 | 170 | 113 | 348 |

Kitchens carry the vertical family harder than ever — 806 of the 1273
`above`/`under` relations, 63%.

**The density warning from the fixed-fittings section is now worse, not better.**
8.0 relations/view against VG's 5.5. Restricting ds3 to the same three families
gives 7.1, so about half the rise is the bathroom drop and half is the new scene
list. `on` is a stable 43% of relations in both datasets, so the composition did
not change — the annotation budget is simply being exceeded by more, and the
cause is the one already recorded: `parentReceptacles` asserts `on`/`in`, which
bypasses the 10% cap. Subtracting the support count from the budget is still the
open fix.

### Result — baseline EGTR on `occlusion_ds4`

R@100 / mR@100, ds3 re-run under the current `eval_occlusion.py` so the two
columns are comparable:

| config | ds3 R@100 | ds4 R@100 | ds3 mR@100 | ds4 mR@100 |
|---|---|---|---|---|
| `exact` | 20.1 | **20.2** | 12.4 | **12.5** |
| `human` | 29.3 | 28.9 | 24.6 | 24.1 |
| `exact` + class-agnostic | 33.2 | 34.4 | 19.2 | 19.5 |
| `human` + class-agnostic | 47.0 | 46.7 | 37.9 | 35.9 |

**Nothing moved.** The dataset nearly doubled, the scene list changed, the camera
band changed, and every headline figure is inside a point of ds3. That is the
result: ds4 buys **n and a better-sampled severe band**, not a different number.
Do not present it as an improvement over ds3.

Per predicate, R@100:

| predicate | ds4 GT | `exact` | `human` | (ds3 `exact`) |
|---|---|---|---|---|
| `on` | 3104 | 38.7% | 41.7% | 38.3% |
| `behind` | 1354 | 2.3% | 19.5% | 1.5% |
| `near` | 968 | 10.4% | 15.2% | 10.0% |
| `above` | 763 | 3.3% | 11.9% | 2.7% |
| `in front of` | 536 | 5.6% | 35.6% | 10.2% |
| `under` | 510 | **14.9%** | 20.8% | 11.6% |

`in front of` is the one predicate that fell (10.2% → 5.6% at exact match) while
its `human` figure held at ~36%, which is a wording disagreement widening rather
than a detection loss. `behind` stays near zero at exact match for the reason
recorded earlier — our derivation keeps true depth where the convention uses
frame height.

### The four-band table is still confounded, and one band is now trustworthy

R@100 by band, exact match:

| | 0.00–0.25 | 0.25–0.50 | 0.50–0.75 | 0.75+ |
|---|---|---|---|---|
| ds3 | 16.9% (n=1358) | 25.6% (n=1407) | 16.9% (n=857) | 13.9% (n=115) |
| ds4 | 17.0% (n=3012) | 26.8% (n=2698) | 20.0% (n=1063) | **4.1%** (n=462) |

Recall still peaks at *moderate* occlusion, so **the warning above applies to ds4
unchanged**: report the composition- and scale-controlled curve, never this
marginal. All three measured confounds (predicate composition, object scale, pair
proposal being caused by occlusion) are properties of the generator and EGTR, and
neither changed.

What did change is the `0.75+` cell. ds3 put it at 13.9% on n=115 — above the
`0.50–0.75` band, which never made sense. On n=462 it is 4.1%, below every other
band, and the object-detection rate underneath it is 25.8%. The earlier figure
was small-sample noise. Object detection by band on ds4: 56.9 / 55.6 / 44.9 /
25.8%.

### Multi-view headroom is small

`build_occlusion_dataset.recoverable()` counts, over all 7235 relations: 5710
clear, **1525 hard** (an endpoint ≥ 0.5 occluded), of which only **236 are
recoverable** — some other view of the same scene sees both endpoints under 0.25.
The rest are hard from every viewpoint the sampler visited.

So the set that extra views can rescue and one view cannot is 236 relations, 3.3%
of the dataset. Any multi-view or NVS result measured on ds4 is measured on that
subset, and 236 is a small n to carry a claim. Raising it means staging the
occlusion rather than letting it arise from scattered clutter — which is what
`build/sgg/find_walkaround.py` and, in the end, `find_cases.py`'s staged occluders exist
to do, at the cost of leaving this dataset behind.

### `occlusion_ds4` is not backed up in git

`.gitignore` excludes `occlusion_ds*/` on the grounds that the generator plus a
seed reproduces them. That holds for ds4 **only against the constants currently
in `build/sgg/build_occlusion_dataset.py`** — the ten-scene lists, `HEIGHT_RANGE =
(1.0, 1.5)`, `radius = (1.2, 2.0)`, `MAX_OCCLUSION = 0.90`. Changing any of them
without rebuilding makes the on-disk ds4 unreproducible from HEAD. Rebuild cost
is ~2 hours of THOR; keep a copy outside git.
