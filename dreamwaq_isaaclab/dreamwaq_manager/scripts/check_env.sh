#!/usr/bin/env bash
# Visually verify the environment by running train.py with GUI enabled.
# Early-iteration policy is near-random (init_std=1.0), so this doubles as a
# "random action" preview before committing to a real training run.
#
# Usage:
#   ./check_env.sh                        # default: Base task, 16 envs, 50 iters
#   TASK=oracle ./check_env.sh            # base | oracle | waq
#   NUM_ENVS=8 MAX_ITER=200 ./check_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYBIN="${PYBIN:-$HOME/IsaacLab/_isaac_sim/python.sh}"
ISAACLAB_SCRIPTS="${ISAACLAB_SCRIPTS:-$HOME/IsaacLab/scripts/reinforcement_learning/rsl_rl}"

TASK_SHORT="${TASK:-base}"
NUM_ENVS="${NUM_ENVS:-16}"
MAX_ITER="${MAX_ITER:-50}"
SEED="${SEED:-42}"

case "$TASK_SHORT" in
    base)   TASK_ID="DreamWaQ-Manager-Go2-Base-v0"   ;;
    oracle) TASK_ID="DreamWaQ-Manager-Go2-Oracle-v0" ;;
    waq)    TASK_ID="DreamWaQ-Manager-Go2-Waq-v0"    ;;
    *) echo "[ERR] Unknown TASK: $TASK_SHORT (expected: base | oracle | waq)"; exit 1 ;;
esac

echo "==========================================="
echo " Env check (GUI) — $TASK_ID"
echo " envs=$NUM_ENVS  iters=$MAX_ITER  seed=$SEED"
echo "==========================================="

cd "$SCRIPT_DIR"
PYTHONPATH="$ISAACLAB_SCRIPTS:${PYTHONPATH:-}" \
    "$PYBIN" rsl_rl/train.py \
    --task="$TASK_ID" \
    --num_envs="$NUM_ENVS" \
    --max_iterations="$MAX_ITER" \
    --seed="$SEED" \
    --logger=tensorboard
