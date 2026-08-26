"""
The robot, and the machinery the NVS experiments drive it with.

Layered on top of `build/sgg/`, not beside it: `task_find` imports
`build.sgg.occlusion_pipeline` and `measure` imports three `gen` modules.
The dataset build is the lower layer and this is the one that stands on it,
which is why a flat "dataset vs experiment" split was not possible.

    robot_controller.py    base + camera on AI2-THOR; the frame geometry
    measure.py           replay an occlusion_ds4 view; measure what the
                           robot's own camera sees from it
    find_partial_triplets.py  `qualifying`, used by the above
    geometry.py            the frame maths, with no controller attached
    task_find.py           instructions, occluder staging, and `grade`
    nvs_lemniscate.py      the sweep: trajectory, third-party render, parking
    sgg_live.py            EGTR loading, `predict`, `raw_predict`

The runnable experiments live at the repo ROOT and import from here.  Nothing
in this package is an experiment; if a file here grows a `main()` that produces
a result, it is in the wrong place.

Both invocation styles work for the modules that have a `__main__`:

    python -m robot.sgg_live --scene FloorPlan203
    python robot/sgg_live.py --scene FloorPlan203

The second needs the repo root on `sys.path`; same bootstrap as `build/sgg/`.
"""
