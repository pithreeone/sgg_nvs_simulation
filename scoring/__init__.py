"""
Scoring SGG predictions against the occlusion dataset.  A CROSS-REPO INTERFACE.

Neither dataset generation nor robotics: this answers "what is R@K on
`datasets/sgg/occlusion_ds4`", and `../sgg_nvs` reads it to produce the paper's
recall tables.  Eight files there do `from eval_occlusion import ...` by bare
name, all of them routed through `../sgg_nvs/lib/occ_eval.py:load_eval`, which
appends this directory to `sys.path`.  So this package's MODULE NAMES are part
of that contract -- renaming `eval_occlusion.py` breaks the paper's evaluation,
and moving it means editing `load_eval` in the same commit.

    eval_occlusion.py     R@K / mR@K per occlusion band; `ranked`, `matched`,
                          `band_of`, `best_iou`, `BANDS`, `KS`
    analyze_multiview.py  the shared matching primitives underneath it --
                          `build_synonyms`, `iou`, `same_class`, `ranked`

Note the robot experiment does NOT score through here.  `robot.task_find.grade`
answers a different question -- did THIS instruction ground on THAT instance --
and the two graders are deliberately separate.
"""
