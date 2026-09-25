#!/usr/bin/env bash
# Backward-compatible two-environment entry point.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NUM_ENVS="${NUM_ENVS:-2}"
exec "${SCRIPT_DIR}/train_doorman.sh" "$@"
