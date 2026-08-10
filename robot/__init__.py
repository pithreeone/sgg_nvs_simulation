"""
The robot, and the machinery the NVS experiments drive it with.

Layered on top of `gen/`, not beside it: `task_find` imports
`gen.occlusion_pipeline` and `drive_triplet_scene` imports three `gen` modules.
The dataset build is the lower layer and this is the one that stands on it,
which is why a flat "dataset vs experiment" split was not possible.

    robot_controller.py    base + camera on AI2-THOR; the frame geometry
    drive.py               open_scene() -- the one entry every runner starts at
    drive_triplet_scene.py measurement, ground truth, standoff poses
    find_partial_triplets.py  `qualifying`, used by the above
    task_find.py           instructions, occluder staging, and `grade`
    nvs_lemniscate.py      the sweep: trajectory, third-party render, parking
    sgg_live.py            EGTR loading, `predict`, `raw_predict`

The runnable experiments live at the repo ROOT and import from here.  Nothing
in this package is an experiment; if a file here grows a `main()` that produces
a result, it is in the wrong place.

Both invocation styles work for the two that have a `__main__`
(`drive.py`, `sgg_live.py`):

    python -m robot.drive --scene FloorPlan203
    python robot/drive.py --scene FloorPlan203

The second needs the repo root on `sys.path`; same bootstrap as `gen/`.
"""
