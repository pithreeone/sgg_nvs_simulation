"""
The two scene builders, named for what each one fills.

    build/sgg/     -> datasets/sgg/     rendered multi-view scenes with occlusion
                                        and relation annotations; 2.3 GB of PNGs,
                                        not tracked, rebuilt from a seed
    build/robot/   -> datasets/robot/   case lists for the movement experiments:
                                        which asset at which coordinate, 512 KB
                                        of JSON, tracked, and THOR builds the
                                        room from it at run time

They share `robot/` and `vg/` and nothing else -- neither imports the other.
"""
