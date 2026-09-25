#!/usr/bin/env bash
# Train only the first curriculum phase: move the left palm near the handle.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK_ID="Template-G1-Wuji-Doorman-Reach-v0"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_wuji_doorman_reach}"
export RUN_NAME="${RUN_NAME:-reach_only_${1:-2}env_seed${SEED:-42}}"

exec "${SCRIPT_DIR}/train_doorman.sh" "$@"
