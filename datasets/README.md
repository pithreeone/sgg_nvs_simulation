# datasets

Everything that is INPUT to something else. Nothing here is code and nothing
here is a result; results live in `../nvs_pilot/` and in `../../sgg_nvs/results/`.

Split by which line of work it feeds:

| | what it holds | read by |
|---|---|---|
| `sgg/` | rendered images, 22 GB | the SGG occlusion benchmark — `../scoring/`, `../analysis/`, `../../sgg_nvs/` |
| `robot/` | scene lists, 512 KB of JSON, no images | the robot viewpoint experiments — `../eval_move.py`, `../fuse_live.py` |

`sgg/` is git-ignored and reproducible from the generators in `../gen/`, at a
cost — rebuilding `occlusion_ds4` is about two hours of THOR. Keep backups
outside git rather than assuming git has them.

`robot/` IS tracked. THOR builds the room at run time from these numbers, so
there are no images to store, and every number in this repo was measured against
one specific list — regenerating with a different seed does not reproduce it.

## Live

| | size | built by | read by |
|---|---|---|---|
| `sgg/occlusion_ds4/` | 465 MB | `../gen/build_occlusion_dataset.py` | `../../sgg_nvs/scratch/occlusion_channels.py` and `mapback_cache.py` — this is the `--gt`/`--ref_root` every fusion experiment scores against |
| `sgg/occlusion_ds4_nvs/` | 20 GB | Stable Virtual Camera, `../../sgg_nvs/script/run_nvs_occlusion.sh` | same; the synthesised views the fusion reads. `lemniscate_az30el15_v20_cs1.0` is the sweep the current channel runs use |

`sgg/occlusion_ds4` is where the paper's numbers come from. Do not move or rebuild
it without checking `../../sgg_nvs/my_script/` first — a dozen scripts name it
by path.

## Superseded

Kept because a rebuild is expensive, not because anything reads them. Delete if
the disk is needed.

| | size | why it is not live |
|---|---|---|
| `occlusion_ds2/` | 88 MB | first occlusion build; superseded by ds3, then ds4 |
| `occlusion_ds3/` | 291 MB | superseded by ds4 |
| `multiview/` `multiview_ref/` | 1.4 GB / 90 MB | the multi-view phase that preceded the occlusion datasets |
| `walkaround/` | 7.3 MB | `../gen/find_walkaround.py` output: stock scenes where furniture already half-hides a target. The finding it produced — 30 scenes yield three cases, all chair-behind-table — is written up in `../find_cases.py`, which is why that file stages occluders instead |

## Not here

The 59 frozen robot tasks are `../nvs_pilot/cases_frozen.json`. They are a
pointer to a question (scene, pose, instruction, occluder type), not rendered
data, so they are tracked in git and live with the results they generated.
