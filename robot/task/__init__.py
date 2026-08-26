"""
The task side: what is asked, how the scene was staged for it, and whether the
answer was right.  All of it reads THOR's instance masks, so nothing here may be
imported by `robot.policy`.

    task_find.py            instructions, occluder staging, and `grade`
    measure.py              what the robot's own camera can see from a pose
    find_partial_triplets.py  `qualifying`
"""
