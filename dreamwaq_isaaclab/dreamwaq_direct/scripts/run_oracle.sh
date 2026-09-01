#!/usr/bin/env bash
# Train DreamWaQ Direct Go2 Oracle model.
# Usage: ./run_oracle.sh
# Override: NUM_ENVS=4096 MAX_ITER=1500 ./run_oracle.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env TASKS="oracle" "$SCRIPT_DIR/run_all_trainings.sh"
