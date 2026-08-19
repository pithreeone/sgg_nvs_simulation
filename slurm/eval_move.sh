#!/bin/bash
# eval_move on a GPU node.  Nothing here runs on the login node.
#
#   sbatch slurm/eval_move.sbatch                      # THOR's own views (the bound)
#   SYNTH=seva OUT=nvs_pilot/move_slot_seva.json \
#       sbatch slurm/eval_move.sbatch                  # Stable Virtual Camera
#   N=2 sbatch -p dev -t 01:00:00 slurm/eval_move.sbatch   # smoke test, 2 cases
#
# The run is parameterised by ENVIRONMENT (CASES, SYNTH, OUT, STEPS, BEARING, N,
# EXTRA); partition and wall clock are sbatch's own flags, so they are overridden
# on the command line as above rather than by a variable.
#
# THREE THINGS MAKE IT A SERVER RUN rather than the desktop one:
# THOR_HEADLESS=1 picks CloudRendering (Vulkan, no X server -- hgpn nodes carry
# nvidia_icd.json and libvulkan.so.1, checked); HF_HUB_OFFLINE=1 because the
# SEVA and DETR weights are already in ~/.cache/huggingface and a compute node
# with no route out should fail loudly rather than hang on a download; and the
# GPU is requested with --gpus-per-node, which this cluster requires even for a
# CPU-shaped step.
#SBATCH --job-name=eval-move
#SBATCH --account=MST115123
#SBATCH --partition=normal
#SBATCH --requeue
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/u3997923/sgg_nvs_simulation/slurm/logs/eval-move-%j.out
#SBATCH --error=/home/u3997923/sgg_nvs_simulation/slurm/logs/eval-move-%j.err

set -eo pipefail

REPO=/home/u3997923/sgg_nvs_simulation
CASES="${CASES:-nvs_pilot/cases/cases_slot.json}"
SYNTH="${SYNTH:-none}"
OUT="${OUT:-nvs_pilot/move_slot_${SYNTH}.json}"
STEPS="${STEPS:-3}"
BEARING="${BEARING:-volatility}"
N="${N:-0}"

module load miniconda3
# EGTR JIT-compiles its deformable attention kernel on first use.  Without these
# the build fails and it silently falls back to the pytorch path, which is not
# numerically the same run.  12.4 matches torch 2.6.0+cu124; the system gcc 8.5
# is too old to compile it.
module load cuda/12.4
module load gcc/11.5.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sgg_nvs

# `vulkaninfo` IN ITS OWN ENVIRONMENT, ON PURPOSE.  ai2thor shells out to it to
# map the CUDA device index onto a Vulkan one, so it only has to be on PATH --
# and `conda install -c conda-forge vulkan-tools` into `sgg_nvs` once solved
# `python 3.10` with GraalPy, replaced CPython, and broke every pip wheel in the
# env (numpy: `undefined symbol: _Py_TrueStruct`).  A separate env cannot do
# that.  Nothing here imports from it.
VKTOOLS="${VKTOOLS:-$HOME/.conda/envs/vktools/bin}"
[ -d "$VKTOOLS" ] && export PATH="$VKTOOLS:$PATH"
command -v vulkaninfo >/dev/null \
  || { echo "FAIL: vulkaninfo not on PATH; ai2thor's CloudRendering needs it."
       echo "      conda create -y -n vktools -c conda-forge vulkan-tools"; exit 1; }

# CloudRendering, and the weights that are already on disk.
export THOR_HEADLESS=1
export HF_HUB_OFFLINE=1
# NODE-LOCAL, AND IT HAS TO BE.  ai2thor talks to Unity over a FIFO it creates
# under TMPDIR, and a named pipe on Weka (/work, /home) does not work: the two
# processes open it, neither ever sees the other's bytes, and the run hangs with
# both at a couple of seconds of CPU and the Unity log sitting happily at
# "Setup Scene called True".  That cost a job to find.  SEVA's per-sweep scratch
# lands here too, which is also where it should be.
export TMPDIR="/tmp/${USER}-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# SOME NODES CANNOT START CLOUDRENDERING.  Unity comes up, spins on the GPU and
# never opens its FIFO, so the job would sit there for its whole wall clock --
# measured: hgpn04 started a controller in 3.4 s while hgpn03, idle GPU, timed
# out at 150 s.  Prove it in 90 s and hand the job back to the scheduler, which
# still does the placing; no node is named here.
if ! timeout 90 python -c "
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
Controller(scene='FloorPlan1', platform=CloudRendering, width=300,
           height=300).stop()" >/dev/null 2>&1; then
    echo "THOR will not start on $(hostname); requeueing"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

echo "node:   $(hostname)"
echo "cases:  $CASES  (n=${N})"
echo "synth:  $SYNTH"
echo "out:    $OUT"
echo "start:  $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# WHICH GPU UNITY IS ABOUT TO RENDER ON.  ai2thor maps the CUDA device index to
# a Vulkan one through these UUIDs and caches the answer in
# ~/.ai2thor/cuda-vulkan-mapping.json -- one file, hard-coded path, shared by
# every job on every node.  cgroup isolation should mean each job sees exactly
# one GPU renumbered to 0, so the cache is always {0: 0}; if that ever stops
# being true, this is the line that says so.
echo "cuda:   $(nvidia-smi -L)"
echo "vulkan: $(vulkaninfo --summary 2>/dev/null | grep -c deviceUUID) device(s)"

cd "$REPO"
python -u eval_move.py \
    --cases "$CASES" \
    --n "$N" \
    --steps "$STEPS" \
    --bearing "$BEARING" \
    --synth "$SYNTH" \
    --out "$OUT" \
    ${EXTRA:-}

echo "done:   $(date -Is)"
