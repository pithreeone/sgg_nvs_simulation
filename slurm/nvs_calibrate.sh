#!/bin/bash
# WHICH WAY IS LEFT.  Score SEVA's four azimuth/elevation sign conventions
# against THOR's own sweep of the same poses; see robot/nvs_seva.py.
#
#   sbatch slurm/nvs_calibrate.sbatch
#   CASE=3 sbatch slurm/nvs_calibrate.sbatch      # a different scene
#
# Four sweeps through a 5 GB diffusion model, so it is its own job rather than
# something `eval_move` does at startup.  It only has to be run again if the
# trajectory parameterisation changes.
#SBATCH --job-name=nvs-calib
#SBATCH --account=MST115123
#SBATCH --partition=dev
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/u3997923/sgg_nvs_simulation/slurm/logs/nvs-calib-%j.out
#SBATCH --error=/home/u3997923/sgg_nvs_simulation/slurm/logs/nvs-calib-%j.err

set -eo pipefail

REPO=/home/u3997923/sgg_nvs_simulation
CASES="${CASES:-datasets/robot/cases_slot.json}"
CASE="${CASE:-0}"
DUMP="${DUMP:-$REPO/slurm/logs/calib-${SLURM_JOB_ID}}"

module load miniconda3
module load cuda/12.4
module load gcc/11.5.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sgg_nvs

export THOR_HEADLESS=1
export HF_HUB_OFFLINE=1
# Node-local: ai2thor's FIFO does not work on Weka.  See eval_move.sbatch.
export TMPDIR="/tmp/${USER}-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "node:   $(hostname)"
echo "cases:  $CASES  (case $CASE)"
echo "dump:   $DUMP"
echo "start:  $(date -Is)"

cd "$REPO"
python -u robot/nvs_seva.py --calibrate \
    --cases "$CASES" --case "$CASE" --dump "$DUMP" ${EXTRA:-}

echo "done:   $(date -Is)"
