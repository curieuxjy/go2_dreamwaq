#!/usr/bin/env bash
# Train DreamWaQ Direct Go2 Base model.
# Usage: ./run_base.sh
# Override: NUM_ENVS=4096 MAX_ITER=1500 ./run_base.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env TASKS="base" "$SCRIPT_DIR/run_all_trainings.sh"
