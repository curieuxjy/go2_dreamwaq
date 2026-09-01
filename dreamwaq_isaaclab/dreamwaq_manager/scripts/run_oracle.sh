#!/usr/bin/env bash
# Run Oracle training only.
# All env vars (NUM_ENVS, MAX_ITER, SEED, LOGGER, EXTRA_ARGS, etc.) are forwarded.
#
# Usage:
#   ./run_oracle.sh
#   NUM_ENVS=4096 MAX_ITER=50000 ./run_oracle.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env TASKS="oracle" "$SCRIPT_DIR/run_all_trainings.sh"
