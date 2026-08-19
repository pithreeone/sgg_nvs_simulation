#!/bin/bash
# One real frame -> one (x, y, theta) step.  See move_once.py.
#
#   DIR=nvs_pilot/real/20260817_01_box_behind_bag/step0 sbatch slurm/move_once.sh
#   DIR=.../step1 EGTR=1 sbatch slurm/move_once.sh
#
# No THOR, so none of eval_move.sh's Vulkan checks.
#SBATCH --job-name=move-once
#SBATCH --account=MST115123
#SBATCH --partition=dev
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/u3997923/sgg_nvs_simulation/slurm/logs/move-once-%j.out
#SBATCH --error=/home/u3997923/sgg_nvs_simulation/slurm/logs/move-once-%j.err

set -eo pipefail

REPO=/home/u3997923/sgg_nvs_simulation

# --- the experiment ---------------------------------------------------------
DIR="${DIR:-nvs_pilot/real/20260817_01_box_behind_bag/step0}"
TASK="${TASK:-box,behind,bag}"          # subject,predicate,object; VG150 words only

# --- the camera -------------------------------------------------------------
CAMERA_HEIGHT="${CAMERA_HEIGHT:-0.15}"  # metres above the floor
PITCH="${PITCH:-0}"                     # degrees, positive down
FOV="${FOV:-42.5}"                      # VERTICAL fov; D435 colour is 42.5, THOR 60
DEPTH_SCALE="${DEPTH_SCALE:-0.001}"     # metres per unit in a 16-bit depth PNG

# --- the policy -------------------------------------------------------------
BEARING="${BEARING:-reveal}"            # see robot/viewpick.py
POOL="${POOL:-both}"                    # how channel B pools across views: mean / max / both
W="${W:-0}"                             # mixing weight on channel B; 0 = A+C only
BETA="${BETA:-1}"                       # compute B and print the diagnostic sweeps
PAIR_IOU="${PAIR_IOU:-0.15}"            # reject a pair that is one object twice
VIEWS="${VIEWS:-20}"
MAX_AZ="${MAX_AZ:-30}"
MAX_EL="${MAX_EL:-15}"

# --- diagnostics ------------------------------------------------------------
EGTR="${EGTR:-0}"                       # 1 = report detections, write egtr.png
AT="${AT:-}"                            # "x0,y0,x1,y1": what does EGTR call this?
CLASSES="${CLASSES:-}"                  # "box,bag": best query per name

# ---------------------------------------------------------------------------

module load miniconda3
module load cuda/12.4
module load gcc/11.5.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sgg_nvs

export HF_HUB_OFFLINE=1
# Node-local: see eval_move.sh.
export TMPDIR="/tmp/${USER}-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

ARGS=(--dir "$DIR" --task "$TASK" --bearing "$BEARING"
      --camera-height "$CAMERA_HEIGHT" --pitch "$PITCH" --fov "$FOV"
      --depth-scale "$DEPTH_SCALE"
      --views "$VIEWS" --max-az "$MAX_AZ" --max-el "$MAX_EL"
      --pair-iou "$PAIR_IOU" --beta "$BETA" --w "$W" --pool "$POOL")
[ "$EGTR" = "1" ] || ARGS+=(--no-egtr)
[ -n "$AT" ] && ARGS+=(--at "$AT")
[ -n "$CLASSES" ] && ARGS+=(--classes "$CLASSES")

echo "node:   $(hostname)"
echo "dir:    $DIR"
echo "task:   $TASK        bearing: $BEARING"
echo "camera: h=$CAMERA_HEIGHT  pitch=$PITCH  fov=$FOV"
echo "start:  $(date -Is)"

cd "$REPO"
python -u move_once.py "${ARGS[@]}" ${EXTRA:-}

echo "done:   $(date -Is)"
