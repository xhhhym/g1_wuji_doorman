#!/usr/bin/env bash
# Run G1+Wuji DoorMan recurrent PPO training.
#
# Usage: ./scripts/train_doorman.sh [NUM_ENVS]
#
# The Isaac Lab Python environment must already be active. Settings can also be
# supplied as environment variables; command-line NUM_ENVS takes precedence.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NUM_ENVS="${NUM_ENVS:-2}"
if [[ $# -gt 0 && "$1" != -* ]]; then
  NUM_ENVS="$1"
  shift
fi

MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-4}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-doorman_${NUM_ENVS}env_seed${SEED}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_wuji_doorman_recurrent}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
DEVICE="${DEVICE:-cuda:0}"
TASK_ID="${TASK_ID:-Template-G1-Wuji-Doorman-v0}"
DRY_RUN="${DRY_RUN:-0}"

for value_name in NUM_ENVS MAX_ITERATIONS NUM_MINI_BATCHES SAVE_INTERVAL; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${value_name} must be a positive integer, got '${value}'." >&2
    exit 2
  fi
done

MODEL_PATH="${REPO_ROOT}/source/g1_wuji_doorman/g1_wuji_doorman/models/model_stand.pt"
ROBOT_USD="${REPO_ROOT}/source/g1_wuji_doorman/g1_wuji_doorman/assets/robots/g1_wuji/g1_wuji_no_merge.usd"

if [[ ! -f "$MODEL_PATH" ]] || [[ "$(stat -c %s "$MODEL_PATH")" -lt 1000000 ]]; then
  echo "ERROR: HOMIE checkpoint is missing or is still a Git LFS pointer." >&2
  echo "Run: git lfs install && git lfs pull" >&2
  exit 3
fi

if [[ ! -f "$ROBOT_USD" ]] || [[ "$(stat -c %s "$ROBOT_USD")" -lt 1000 ]]; then
  echo "ERROR: G1+Wuji runtime USD is missing or is still a Git LFS pointer." >&2
  echo "Run: git lfs install && git lfs pull" >&2
  exit 4
fi

if ! python - <<'PY'
import importlib.util
import sys

missing = [
    name
    for name in ("isaaclab", "isaaclab_rl", "isaaclab_tasks", "rsl_rl", "g1_wuji_doorman")
    if importlib.util.find_spec(name) is None
]
if missing:
    print("Missing Python packages: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
then
  echo "Activate the Isaac Lab environment and install this project with:" >&2
  echo "  python -m pip install -e ${REPO_ROOT}/source/g1_wuji_doorman" >&2
  echo "  python -m pip install rsl-rl-lib==3.0.1 tensorboard" >&2
  exit 5
fi

cd "$REPO_ROOT"
unset DISPLAY
unset WAYLAND_DISPLAY
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMAND=(
  python -u scripts/rsl_rl/train.py
  --task "$TASK_ID"
  --num_envs "$NUM_ENVS"
  --device "$DEVICE"
  --headless
  --seed "$SEED"
  --max_iterations "$MAX_ITERATIONS"
  --experiment_name "$EXPERIMENT_NAME"
  --run_name "$RUN_NAME"
  --logger tensorboard
  "agent.save_interval=${SAVE_INTERVAL}"
  "agent.algorithm.num_mini_batches=${NUM_MINI_BATCHES}"
  "$@"
)

echo "Starting DoorMan training:"
echo "  task=${TASK_ID}"
echo "  experiment=${EXPERIMENT_NAME}"
echo "  envs=${NUM_ENVS} iterations=${MAX_ITERATIONS} mini_batches=${NUM_MINI_BATCHES}"
echo "  seed=${SEED} device=${DEVICE} run=${RUN_NAME}"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '  %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${COMMAND[@]}"
