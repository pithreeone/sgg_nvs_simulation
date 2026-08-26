"""
build_robotic_task -- the generators that make a case list.

A case is a POINTER TO A QUESTION plus a pinned scene: an instruction, the
objects that make it true, and a camera pose it is true from.  Everything here
writes one; nothing here reads a model.  The runners that do -- `fuse_live.py`,
`eval_move.py`, the probes -- live at the repo root and take a case list as
input.

    build_tabletop.py   procedural tabletop, target + SAME-ASSET DISTRACTOR +
                        occluder.  The relation has to say which copy, so this
                        is the list that measures relational grounding.
                        -> cases_hard
    build_slot.py       procedural tabletop, target + landmark + blocker, with
                        the target visible only over an interval of viewpoints.
                        One target, so the relation is stated but not needed;
                        this is the list that measures viewpoint choice.
                        -> cases_slot

TWO LISTS, and they are built to need different evidence: `cases_slot`'s target
cannot be NAMED from the start pose, `cases_hard`'s is named perfectly and is
indistinguishable from its twin, so only the RELATION picks it out.  A viewpoint
rule that works on one and not the other has not been shown to generalise.

An older iTHOR line -- `find_cases.py` searching stock floor plans and
`freeze_cases.py` pinning what it found -- was deleted along with the lists it
produced (`cases_easy`, `cases_easy2`, `cases_tabletop`).  `build_tabletop.py`'s
docstring says why it could not survive: an iTHOR scene holds no two instances of
one objectType, so it cannot produce a same-class distractor, and all 71 frozen
iTHOR cases had zero.  Recover with `git log -- build/robot/`.
"""
