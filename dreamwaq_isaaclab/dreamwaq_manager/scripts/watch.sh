#!/usr/bin/env bash
# Watch the training environment in the Isaac Sim viewport (no trained model needed).
# Runs a random policy so you can visually inspect terrain, robot, and rewards.
#
# Usage:
#   ./watch.sh              # Base mode (default)
#   ./watch.sh oracle       # Oracle mode
#   ./watch.sh waq          # Waq mode
#   NUM_ENVS=16 ./watch.sh  # Override env count

set -euo pipefail

PYBIN="${PYBIN:-$HOME/IsaacLab/_isaac_sim/python.sh}"
ISAACLAB_SCRIPTS="${ISAACLAB_SCRIPTS:-$HOME/IsaacLab/scripts/reinforcement_learning/rsl_rl}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:-base}"
NUM_ENVS="${NUM_ENVS:-4}"

case "$MODE" in
    base)   TASK="DreamWaQ-Manager-Go2-Base-Play-v0"   ;;
    oracle) TASK="DreamWaQ-Manager-Go2-Oracle-Play-v0" ;;
    waq)    TASK="DreamWaQ-Manager-Go2-Waq-Play-v0"    ;;
    *) echo "Usage: $0 [base|oracle|waq]"; exit 1 ;;
esac

echo "============================================"
echo " DreamWaQ Manager — Watch Mode"
echo " Task:    $TASK"
echo " Envs:    $NUM_ENVS"
echo " Mode:    $MODE"
echo "============================================"
echo ""
echo " Controls:"
echo "   Mouse drag    — rotate camera"
echo "   Scroll        — zoom"
echo "   Middle drag   — pan"
echo "   Ctrl+C        — exit"
echo ""

cd "$SCRIPT_DIR"
PYTHONPATH="$ISAACLAB_SCRIPTS:${PYTHONPATH:-}" \
exec "$PYBIN" rsl_rl/watch.py \
    --task="$TASK" \
    --num_envs="$NUM_ENVS"
