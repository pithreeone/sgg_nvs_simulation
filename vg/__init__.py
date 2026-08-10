"""
VG150 vocabulary and the priors fitted on it.  No AI2-THOR, no torch.

The bottom of the dependency graph: every other package imports from here and
this one imports nothing local.  Grouped because the four modules answer one
question -- what words can the model say, and which of them would a human
annotator have bothered to write down -- and because the two fitted JSONs must
travel with the code that reads them (both resolve their path against
`__file__`, so moving the pair is safe and splitting it is not).

    vg150.py           THOR objectType <-> VG150 class, and the relation
                       vocabulary with its occlusion families
    vg_conventions.py  which relations a VG150 annotator actually writes
    vg_gt.py           ground-truth relations for a THOR scene, gated by
                       `vg_prior.json` (fitted on VG train)
    vg_pair_prior.py   `vg_pair_prior.json`, 315642 VG relations, which pairs
                       of classes are worth asserting a relation between
"""
