#!/usr/bin/env bash
# Run Base training only.
# All env vars (NUM_ENVS, MAX_ITER, SEED, LOGGER, EXTRA_ARGS, etc.) are forwarded.
#
# Usage:
#   ./run_base.sh
#   NUM_ENVS=4096 MAX_ITER=50000 ./run_base.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env TASKS="base" "$SCRIPT_DIR/run_all_trainings.sh"
