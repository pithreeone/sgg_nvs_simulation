"""
Where the robot decides.  Reads the swept record and the instruction; never the
simulator's ground truth -- that lives in `robot.task.measure`, and a rule that
consulted it would be grading itself.

    grounding.py   one instruction, one image, the pair it names
    viewpick.py    a swept record becomes a heading and a view
    evidence.py    channel B, computed once and read two ways
    vlm.py         Qwen2.5-VL turned into one bit: step LEFT or step RIGHT
"""
