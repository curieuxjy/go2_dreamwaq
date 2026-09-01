#!/usr/bin/env bash
# Run Waq (CENet) training only.
# All env vars (NUM_ENVS, MAX_ITER, SEED, LOGGER, EXTRA_ARGS, etc.) are forwarded.
#
# Usage:
#   ./run_waq.sh
#   NUM_ENVS=4096 MAX_ITER=50000 ./run_waq.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env TASKS="waq" "$SCRIPT_DIR/run_all_trainings.sh"
