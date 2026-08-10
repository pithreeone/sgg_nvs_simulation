"""
Dataset generation: everything that needs a running AI2-THOR instance.

Split from the rest of the tree because the dependency profile differs -- these
modules import `ai2thor` and `cv2`, while evaluation imports neither and runs in
the plain environment.  Keeping them apart means `python eval_occlusion.py` never
needs the simulator installed.

Both invocation styles work:

    python -m gen.build_occlusion_dataset --rooms all
    python gen/build_occlusion_dataset.py --rooms all

The second only works because each runnable module bootstraps the repo root onto
`sys.path`.  Python puts the SCRIPT's directory on the path, not the cwd, so
without that `from vg150 import ...` would fail from inside this package.
"""
