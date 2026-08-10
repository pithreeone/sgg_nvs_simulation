"""
Analysis of the built datasets.  Nothing here is on the NVS-experiment path.

These read `datasets/` and report -- recall tables, ground-truth tuning, and
figures.  They are kept apart from the robot pipeline at the repo root because
mixing the two is what made it hard to tell which code was current.

Both invocation styles work:

    python -m analysis.eval_recall
    python analysis/eval_recall.py

The second only works because each runnable module bootstraps the repo root
onto `sys.path` -- Python puts the SCRIPT's directory on the path, not the cwd,
so without that `from task_find import ...` would fail from in here.  Same
convention as `gen/`.

NOT here, and each for a reason worth knowing:

  eval_occlusion.py        `../sgg_nvs/scratch/occlusion_mapback_eval.py` appends
                           the simulation directory to `sys.path` and imports it
                           by bare name for its matching rules.  Moving it would
                           break the paper's evaluation path.
  analyze_multiview.py     Named like an analysis, but it IS the shared matching
                           library -- `build_synonyms`, `iou`, `same_class`,
                           `ranked` -- imported by eval_occlusion and by two of
                           the scripts in here.
  find_partial_triplets.py Same story: `drive_triplet_scene.py` imports
                           `qualifying` from it, so it is on the live path.

Both were moved in here once and moved straight back out; the import graph, not
the file name, decides.
"""
