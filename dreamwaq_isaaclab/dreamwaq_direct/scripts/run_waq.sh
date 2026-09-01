#!/usr/bin/env bash
# Train DreamWaQ Direct Go2 Waq model (CENet).
# Usage: ./run_waq.sh
# Override: NUM_ENVS=4096 MAX_ITER=3000 ./run_waq.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env TASKS="waq" "$SCRIPT_DIR/run_all_trainings.sh"
