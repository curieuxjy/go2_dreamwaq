#!/usr/bin/env bash
# Sequentially run Base → Oracle → Waq full trainings.
# Each run waits until the previous one is fully done and the GPU has cooled down
# (memory drained) before launching the next.
#
# Usage:
#   ./run_all_trainings.sh                # default: 4096 envs, default max_iterations
#   NUM_ENVS=2048 MAX_ITER=1500 ./run_all_trainings.sh
#   TASKS="base waq" ./run_all_trainings.sh   # subset only
#
# Requires: ~/IsaacLab/_isaac_sim/python.sh and PYTHONPATH set up per SETUP.md.

set -euo pipefail

# Ensure Ctrl-C kills the entire process tree (including Isaac Sim child processes)
cleanup() {
    echo "[INFO] Caught signal, killing all child processes..."
    pkill -P $$ 2>/dev/null || true
    pgrep -f 'kit/python/bin/python3.*train.py' | xargs kill -9 2>/dev/null || true
    exit 1
}
trap cleanup INT TERM

# Kill any leftover Isaac Sim processes from previous crashed runs
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
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-5000}"     # all tasks: 100k iterations
SEED="${SEED:-42}"
LOGGER="${LOGGER:-wandb}"         # tensorboard | wandb | neptune
EXTRA_ARGS="${EXTRA_ARGS:-}"       # any additional CLI flags
TASKS="${TASKS:-base oracle waq}"  # ordered list

# Disable wandb Sentry telemetry to prevent DNS-related init timeouts.
# Set WANDB_MODE=offline if your network fully blocks wandb.ai.
export WANDB_ERROR_REPORTING="${WANDB_ERROR_REPORTING:-false}"
export WANDB__DISABLE_STATS="${WANDB__DISABLE_STATS:-true}"
export SENTRY_DSN="${SENTRY_DSN:-}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"

# GPU memory thresholds (MiB) — wait until used memory drops below this between runs
GPU_FREE_THRESHOLD="${GPU_FREE_THRESHOLD:-1500}"
GPU_WAIT_TIMEOUT="${GPU_WAIT_TIMEOUT:-180}"   # seconds
GPU_POLL_INTERVAL="${GPU_POLL_INTERVAL:-5}"

LOG_ROOT="$PROJECT_ROOT/logs/rsl_rl"
mkdir -p "$LOG_ROOT"

# ---------- helpers ----------
color_info()  { printf '\033[1;36m[INFO]\033[0m  %s\n' "$*"; }
color_warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
color_err()   { printf '\033[1;31m[ERR ]\033[0m  %s\n' "$*"; }
color_ok()    { printf '\033[1;32m[ OK ]\033[0m  %s\n' "$*"; }

gpu_snapshot() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found"
        return
    fi
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
               --format=csv,noheader,nounits \
        | awk -F', ' '{printf "  GPU%s %s | mem %s/%s MiB | util %s%% | %s°C\n", $1,$2,$3,$4,$5,$6}'
}

gpu_used_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '
}

wait_for_gpu_idle() {
    local waited=0
    color_info "Waiting for GPU memory to drop below ${GPU_FREE_THRESHOLD} MiB (timeout ${GPU_WAIT_TIMEOUT}s)…"
    while true; do
        local used
        used="$(gpu_used_mib 2>/dev/null || echo 999999)"
        if [[ "$used" =~ ^[0-9]+$ ]] && (( used < GPU_FREE_THRESHOLD )); then
            color_ok "GPU idle (used=${used} MiB)"
            return 0
        fi
        if (( waited >= GPU_WAIT_TIMEOUT )); then
            color_warn "Timeout waiting for GPU idle (last used=${used} MiB). Continuing anyway."
            return 0
        fi
        printf '\r  used=%s MiB  waited=%ds   ' "$used" "$waited"
        sleep "$GPU_POLL_INTERVAL"
        waited=$(( waited + GPU_POLL_INTERVAL ))
    done
}

task_id_for() {
    case "$1" in
        base)   echo "DreamWaQ-Manager-Go2-Base-v0"   ;;
        oracle) echo "DreamWaQ-Manager-Go2-Oracle-v0" ;;
        waq)    echo "DreamWaQ-Manager-Go2-Waq-v0"    ;;
        *) color_err "Unknown task: $1"; return 1 ;;
    esac
}

run_one() {
    local short="$1"
    local task_id
    task_id="$(task_id_for "$short")" || exit 1
    local task_log_dir="$LOG_ROOT/$task_id"
    mkdir -p "$task_log_dir"
    local logfile="$task_log_dir/console.log"

    color_info "============================================================"
    color_info " Starting [$short] → $task_id"
    color_info " Log dir: $task_log_dir/"
    color_info "============================================================"
    gpu_snapshot

    local cmd=( "$PYBIN" rsl_rl/train.py
        --task="$task_id"
        --headless
        --num_envs="$NUM_ENVS"
        --seed="$SEED"
        --logger="$LOGGER"
    )
    [[ -n "$MAX_ITER"    ]] && cmd+=( --max_iterations="$MAX_ITER" )
    [[ -n "$EXTRA_ARGS"  ]] && read -r -a extra <<< "$EXTRA_ARGS" && cmd+=( "${extra[@]}" )

    color_info "CMD: ${cmd[*]}"
    local start_ts end_ts
    start_ts=$(date +%s)

    if ( cd "$SCRIPT_DIR" && PYTHONPATH="$ISAACLAB_SCRIPTS:${PYTHONPATH:-}" stdbuf -oL "${cmd[@]}" ) 2>&1 | tee "$logfile"; then
        end_ts=$(date +%s)
        color_ok "[$short] finished in $((end_ts - start_ts))s"
    else
        end_ts=$(date +%s)
        color_err "[$short] FAILED after $((end_ts - start_ts))s — see $logfile"
        return 1
    fi
}

# ---------- preflight ----------
color_info "=== preflight ==="
[[ -x "$PYBIN" ]] || { color_err "Python not executable: $PYBIN"; exit 1; }
[[ -d "$ISAACLAB_SCRIPTS" ]] || { color_err "ISAACLAB_SCRIPTS missing: $ISAACLAB_SCRIPTS"; exit 1; }
[[ -f "$SCRIPT_DIR/rsl_rl/train.py" ]] || { color_err "train.py not found in $SCRIPT_DIR"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || color_warn "nvidia-smi not found — GPU monitoring disabled"

color_info "PYBIN            = $PYBIN"
color_info "ISAACLAB_SCRIPTS = $ISAACLAB_SCRIPTS"
color_info "SCRIPT_DIR       = $SCRIPT_DIR"
color_info "TASKS            = $TASKS"
color_info "NUM_ENVS         = $NUM_ENVS"
color_info "MAX_ITER         = ${MAX_ITER:-<task default>}"
color_info "SEED             = $SEED"
color_info "LOGGER           = $LOGGER"
color_info "LOG_ROOT         = $LOG_ROOT"
gpu_snapshot

# ---------- main loop ----------
overall_start=$(date +%s)
declare -a results=()

for t in $TASKS; do
    wait_for_gpu_idle
    if run_one "$t"; then
        results+=( "$t: OK" )
    else
        results+=( "$t: FAIL" )
        color_warn "Continuing to next task despite failure."
    fi
    sleep 3   # let processes fully release before polling
done

overall_end=$(date +%s)

# ---------- summary ----------
echo
color_info "============================================================"
color_info " ALL DONE — total $(( overall_end - overall_start ))s"
color_info "============================================================"
for r in "${results[@]}"; do
    if [[ "$r" == *OK ]]; then color_ok "$r"; else color_err "$r"; fi
done
color_info "Logs: $LOG_ROOT"
gpu_snapshot
