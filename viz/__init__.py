"""
viz -- the figures.  Nothing here decides anything; see `robot/` for that.

    sheets.py     tile, caption, grid -- the primitives every sheet is built of
    sweep.py      the two figures a synthesised sweep gets
    grounding.py  where `grounding.single_frame` succeeds and how it fails

and the runnable scripts, which read a file and draw it:

    show_tasks.py     what a case list actually asks
    show_slot.py      what `build_slot.py` built, before trusting the numbers
    plot_paths.py     the two arms' routes, from above
    plot_viewdist.py  the viewpoint distribution per case
    plot_viewhits.py  which sweep views got the triplet right

Both invocation styles work; each script bootstraps the repo root onto
sys.path, the same convention as `analysis/` and `gen/`.
"""
