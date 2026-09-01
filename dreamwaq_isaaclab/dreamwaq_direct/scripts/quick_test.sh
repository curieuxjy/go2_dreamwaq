#!/usr/bin/env bash
# Quick smoke test: run each task with few envs & iterations to verify setup.
#
# Usage:
#   ./quick_test.sh                          # default: 64 envs, 30 iters, all tasks
#   NUM_ENVS=128 MAX_ITER=50 ./quick_test.sh
#   TASKS="base" ./quick_test.sh             # single task only

set -euo pipefail
trap 'kill -- -$$' INT TERM

# Kill stale Isaac Sim processes
stale_pids=$(pgrep -f 'kit/python/bin/python3.*train.py' 2>/dev/null || true)
if [[ -n "$stale_pids" ]]; then
    echo "[WARN] Killing stale Isaac Sim processes: $stale_pids"
    kill -9 $stale_pids 2>/dev/null || true
    sleep 2
fi

# ---------- config ----------
PYBIN="${PYBIN:-$HOME/IsaacLab/_isaac_sim/python.sh}"
ISAACLAB_SCRIPTS="${ISAACLAB_SCRIPTS:-$HOME/IsaacLab/scripts/reinforcement_learning/rsl_rl}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITER="${MAX_ITER:-30}"
SEED="${SEED:-42}"
LOGGER="${LOGGER:-tensorboard}"   # tensorboard for quick tests (no wandb noise)
EXTRA_ARGS="${EXTRA_ARGS:-}"
TASKS="${TASKS:-base oracle waq}"

GPU_FREE_THRESHOLD="${GPU_FREE_THRESHOLD:-1500}"
GPU_WAIT_TIMEOUT="${GPU_WAIT_TIMEOUT:-60}"
GPU_POLL_INTERVAL="${GPU_POLL_INTERVAL:-3}"

# ---------- helpers ----------
color_info()  { printf '\033[1;36m[INFO]\033[0m  %s\n' "$*"; }
color_warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
color_err()   { printf '\033[1;31m[ERR ]\033[0m  %s\n' "$*"; }
color_ok()    { printf '\033[1;32m[ OK ]\033[0m  %s\n' "$*"; }

gpu_used_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

wait_for_gpu_idle() {
    local waited=0
    color_info "Waiting for GPU memory < ${GPU_FREE_THRESHOLD} MiB …"
    while true; do
        local used
        used="$(gpu_used_mib || echo 999999)"
        if [[ "$used" =~ ^[0-9]+$ ]] && (( used < GPU_FREE_THRESHOLD )); then
            color_ok "GPU idle (used=${used} MiB)"
            return 0
        fi
        if (( waited >= GPU_WAIT_TIMEOUT )); then
            color_warn "GPU wait timeout (used=${used} MiB). Continuing."
            return 0
        fi
        printf '\r  used=%s MiB  waited=%ds   ' "$used" "$waited"
        sleep "$GPU_POLL_INTERVAL"
        waited=$(( waited + GPU_POLL_INTERVAL ))
    done
}

task_id_for() {
    case "$1" in
        base)   echo "DreamWaQ-Direct-Go2-Base-v0"   ;;
        oracle) echo "DreamWaQ-Direct-Go2-Oracle-v0" ;;
        waq)    echo "DreamWaQ-Direct-Go2-Waq-v0"    ;;
        *) color_err "Unknown task: $1"; return 1 ;;
    esac
}

run_one() {
    local short="$1"
    local task_id
    task_id="$(task_id_for "$short")" || exit 1

    color_info "────────────────────────────────────────"
    color_info " Quick test [$short] → $task_id"
    color_info " envs=$NUM_ENVS  iters=$MAX_ITER"
    color_info "────────────────────────────────────────"

    local cmd=( "$PYBIN" rsl_rl/train.py
        --task="$task_id"
        --headless
        --num_envs="$NUM_ENVS"
        --seed="$SEED"
        --logger="$LOGGER"
        --max_iterations="$MAX_ITER"
    )
    [[ -n "$EXTRA_ARGS" ]] && read -r -a extra <<< "$EXTRA_ARGS" && cmd+=( "${extra[@]}" )

    color_info "CMD: ${cmd[*]}"
    local start_ts end_ts
    start_ts=$(date +%s)

    if ( cd "$SCRIPT_DIR" && PYTHONPATH="$ISAACLAB_SCRIPTS:${PYTHONPATH:-}" "${cmd[@]}" ); then
        end_ts=$(date +%s)
        color_ok "[$short] passed in $((end_ts - start_ts))s"
        return 0
    else
        end_ts=$(date +%s)
        color_err "[$short] FAILED after $((end_ts - start_ts))s"
        return 1
    fi
}

# ---------- preflight ----------
color_info "=== quick_test preflight ==="
[[ -x "$PYBIN" ]] || { color_err "Python not executable: $PYBIN"; exit 1; }
[[ -d "$ISAACLAB_SCRIPTS" ]] || { color_err "ISAACLAB_SCRIPTS missing: $ISAACLAB_SCRIPTS"; exit 1; }
[[ -f "$SCRIPT_DIR/rsl_rl/train.py" ]] || { color_err "train.py not found in $SCRIPT_DIR"; exit 1; }

color_info "TASKS    = $TASKS"
color_info "NUM_ENVS = $NUM_ENVS"
color_info "MAX_ITER = $MAX_ITER"
color_info "LOGGER   = $LOGGER"

# ---------- main ----------
overall_start=$(date +%s)
passed=0
failed=0
declare -a results=()

for t in $TASKS; do
    wait_for_gpu_idle
    if run_one "$t"; then
        results+=( "$t: PASS" )
        passed=$((passed + 1))
    else
        results+=( "$t: FAIL" )
        failed=$((failed + 1))
    fi
    sleep 2
done

overall_end=$(date +%s)

# ---------- summary ----------
echo
color_info "════════════════════════════════════════"
color_info " Quick test done — ${passed} passed, ${failed} failed  ($(( overall_end - overall_start ))s)"
color_info "════════════════════════════════════════"
for r in "${results[@]}"; do
    if [[ "$r" == *PASS ]]; then color_ok "$r"; else color_err "$r"; fi
done

if (( failed > 0 )); then exit 1; fi
exit 0
