#!/usr/bin/env bash
# Run the recurrent G1+Wuji DoorMan task overnight with two environments.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SETUP="/home/yimin/miniconda3/etc/profile.d/conda.sh"

# Override these from the terminal when needed, for example:
# NUM_ENVS=4096 NUM_MINI_BATCHES=16 MAX_ITERATIONS=3000 RUN_NAME=overnight_4096 ./scripts/train_overnight_2env.sh
NUM_ENVS="${NUM_ENVS:-2}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-4}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-overnight_2env}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

cd "${REPO_ROOT}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "doorman" ]]; then
  source "${CONDA_SETUP}"
  conda activate doorman
fi

# Isaac Sim headless can still probe an inherited desktop display during Kit
# startup.  On this machine that X11 probe can crash inside XOpenDisplay.
unset DISPLAY
unset WAYLAND_DISPLAY
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

exec python -u scripts/rsl_rl/train.py \
  --task Template-G1-Wuji-Doorman-v0 \
  --num_envs "${NUM_ENVS}" \
  --device cuda:0 \
  --headless \
  --seed "${SEED}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --experiment_name g1_wuji_doorman_recurrent \
  --run_name "${RUN_NAME}" \
  --logger tensorboard \
  agent.save_interval="${SAVE_INTERVAL}" \
  agent.algorithm.num_mini_batches="${NUM_MINI_BATCHES}"
