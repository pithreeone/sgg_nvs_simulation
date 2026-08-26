# robot_movement

Does choosing the viewpoint matter, and by how much?

Every run is `eval_move.py` walking ONE policy for three fixed steps from the
same start pose, graded at each pose by the instruction-conditioned top-1
(`robot/policy/grounding.py`).  One run, one row: `--policy` names the arm.

    python eval_move.py --cases datasets/robot/cases_slot_v2.json \
        --steps 3 --policy reveal --out results/robot_movement/slot_v2_reveal.json

## The two case lists

`slot_v2` is `datasets/robot/cases_slot_v2.json`, built for this table: every
case is one the robot gets WRONG standing still and right from its best
viewpoint, so the whole table measures what moving buys.  `stay` is 5/40 rather
than 0 because the generator screens with `fuse_live.conditioned` and the
experiment grades with `grounding.single_frame`; the two agree on 35 of 40.

`hard` is the older `datasets/robot/cases_hard.json`, kept because its target
and its distractor are the SAME CLASS -- appearance cannot separate them and
only correspondence can.  It has no such screen, so `stay` is 15/40 and its
columns mix "can moving fix a wrong answer" with "can moving avoid breaking a
right one".  Read it against `slot_v2`, not merged with it.

It was also staged at 800x600 and is run here at 600x600, so the
`staged_occlusion` recorded in the case file describes a wider frame than the
robot is given.  Grading does not read those fields -- `robot/task/measure`
measures at run time -- but the numbers in the file are not the numbers of these
runs.

## The policies

    reveal        correspondence + appearance, each standardised, added
    attrib        channel B read per view: this view's own summand for the pair
                  the fusion chose
    vlm_direct    Qwen2.5-VL answers LEFT or RIGHT from the robot's own frame.
                  The wording is measured; see `robot/policy/vlm.py`
    vlm_boxes     an earlier arm: the model returns two boxes and the SIDE is
                  arithmetic over their centres.  Stronger, but the direction is
                  not the model's decision
    random        a pose from the box the sweep spans -- same action space as
                  the method, so this isolates the CHOICE
    random-grid   any pose in the room that keeps the pair in frame
                  (`robot/world/viewgrid.py`).  A larger space on purpose: this
                  reads as system against an aimless robot, and the gap is not
                  attributable to scoring alone

## What is in a file

The header records `policy`, `steps`, `condition`, and the VLM model and
quantisation where one was used.  Then one entry per case: the start verdict,
and for the arm that walked, `per_step`, `trail`, `metres`, `revisit`, `best`.

`vlm_direct` on `slot_v2` used the 3B model.  7B fits on an 8 GB card only when
nothing else is on it: with EGTR and THOR resident it OOMs on the first frame.
