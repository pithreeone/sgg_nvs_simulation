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
                        -> cases_easy2, cases_hard, cases_tabletop
    build_slot.py       procedural tabletop, target + landmark + blocker, with
                        the target visible only over an interval of viewpoints.
                        One target, so the relation is stated but not needed;
                        this is the list that measures viewpoint choice.
                        -> cases_slot
    find_cases.py       the older iTHOR line: search stock floor plans for a
    freeze_cases.py     nameable object something can be slid in front of, then
                        pin what was found.  -> cases_wide, cases_behind

`build_tabletop.py`'s docstring says why the procedural pair exists at all: an
iTHOR scene holds no two instances of one objectType, so `find_cases` cannot
produce a same-class distractor, and all 71 frozen iTHOR cases had zero.
"""
