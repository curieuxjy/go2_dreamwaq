#!/usr/bin/env bash
# Stage-1 full pipeline: train all six PPO tasks to completion, and record a tracking
# video from the matching *-Play-v0 config right after each run finishes.
#
# All six use the SAME budget (3000 iters / 4096 envs) so the Base vs Oracle vs Waq
# comparison is not confounded by training length.
#
# Usage:
#   ./run_full_pipeline.sh                 # all six
#   TASKS="rough:waq flat:waq" ./run_full_pipeline.sh
#   NUM_ENVS=2048 MAX_ITER=1000 ./run_full_pipeline.sh
#
# wandb 는 online 으로 기록한다 (project=lec_dreamwaq). 인증이 없으면
# WANDB_MODE=offline ./run_full_pipeline.sh 로 돌린 뒤 나중에 sync 한다.

set -uo pipefail
set -m          # 백그라운드 작업마다 독립 프로세스 그룹 → hang 시 그룹째 kill

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-3000}"
SEED="${SEED:-42}"
VIDEO_LENGTH="${VIDEO_LENGTH:-400}"     # 400 steps = 1 episode = 20 s
VIDEO_ENVS="${VIDEO_ENVS:-32}"
# rough:terrainoracle = 논문이 말한 oracle(높이맵 정책). flat 은 지형 정보가 없어 해당 없음.
TASKS="${TASKS:-rough:basedwq rough:oracledwq rough:waq rough:terrainoracle flat:basedwq flat:oracledwq flat:waq}"
LOG_DIR="${LOG_DIR:-$ROOT/pipeline_logs}"

# hang 감시: 로그가 STALL_SECS 동안 갱신되지 않으면 죽이고 다음 task 로 넘어간다.
# 정상 iteration 은 ~10s. 부팅 + 지형 생성이 길 수 있어 900s 로 여유를 둔다.
# (이 감시는 아래 WANDB_CONSOLE=off 가 전제다 — wandb 가 stdout 을 가로채면 로그가
#  갱신되지 않아 멀쩡한 학습을 hang 으로 오판한다.)
STALL_SECS="${STALL_SECS:-900}"
CHECK_SECS="${CHECK_SECS:-60}"
# Isaac Sim 부팅 실패(hang / iteration 0 회 후 exit 0)는 재시도하면 대개 된다.
TRAIN_TRIES="${TRAIN_TRIES:-3}"

# Isaac Sim / IsaacLab entry points. Override if your install differs.
# 번들 kit python 을 쓴다 (CLAUDE.md 규칙). 맨 python 은 PATH 에 없다.
PYBIN="${PYBIN:-$HOME/IsaacLab/_isaac_sim/python.sh}"
ISAACLAB_SCRIPTS="${ISAACLAB_SCRIPTS:-$HOME/IsaacLab/scripts/reinforcement_learning/rsl_rl}"

export WANDB_MODE="${WANDB_MODE:-online}"
# wandb online 은 기본적으로 콘솔을 가로채(wrap_raw) stdout 을 자기 output.log 로 빼돌린다.
# 그러면 우리 학습 로그가 더 이상 자라지 않아 (a) 진행 상황을 볼 수 없고
# (b) 위 stall 감시가 멀쩡한 학습을 hang 으로 오판한다. 지표 로깅에는 영향이 없다.
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$LOG_DIR"

info() { printf '\033[1;36m[INFO]\033[0m  %s\n' "$*"; }
ok()   { printf '\033[1;32m[ OK ]\033[0m  %s\n' "$*"; }
err()  { printf '\033[1;31m[ERR ]\033[0m  %s\n' "$*"; }

# Stage 1 비교축: 공식 env 레시피 위에서 actor 관측만 다른 3변형 x 2지형.
# (원본 DreamWaQ 레시피인 DreamWaQ-Manager-Go2-* 는 보행에 실패한다 — report.qmd §1)
pkg_dir_for()  { echo dreamwaq_manager; }
task_id_for()  {
    local terrain="$1" variant="$2"          # terrain=flat|rough, variant=basedwq|oracledwq|waq
    local t="${terrain^}"                    # flat -> Flat
    case "$variant" in
        basedwq)   printf 'DreamWaQ-BaseDwq-%s-PPO'   "$t" ;;
        oracledwq) printf 'DreamWaQ-OracleDwq-%s-PPO' "$t" ;;
        waq)       printf 'DreamWaQ-Waq-Official-%s-PPO' "$t" ;;
        terrainoracle) printf 'DreamWaQ-TerrainOracle-%s-PPO' "$t" ;;
        *) return 1 ;;
    esac
}
# agent cfg 의 experiment_name — 체크포인트/tfevents/영상이 쌓이는 폴더 이름
exp_name_for() {
    local terrain="$1" variant="$2"
    local t="${terrain^}"
    case "$variant" in
        basedwq)   printf 'BaseDwq-Official-%s-PPO-v0'   "$t" ;;
        oracledwq) printf 'OracleDwq-Official-%s-PPO-v0' "$t" ;;
        waq)       printf 'Waq-Official-%s-PPO-v0'       "$t" ;;
        terrainoracle) printf 'TerrainOracle-Official-%s-PPO-v0' "$t" ;;
        *) return 1 ;;
    esac
}

# 로그 파일 mtime 을 감시하며 실행한다. STALL_SECS 무갱신이면 프로세스 그룹째 죽이고 124 반환.
# (Waq/CENet 경로에서 간헐적 native hang 이 보고된 적이 있어 무인 실행에는 필수다.)
run_watched() {
    local logf="$1" pkg="$2"; shift 2
    : > "$logf"
    (
        cd "$ROOT/$pkg/scripts" || exit 127
        export PYTHONPATH="$ISAACLAB_SCRIPTS:${PYTHONPATH:-}"
        exec "$PYBIN" "$@"
    ) > "$logf" 2>&1 &
    local pid=$! age rc
    while kill -0 "$pid" 2>/dev/null; do
        sleep "$CHECK_SECS"
        age=$(( $(date +%s) - $(stat -c %Y "$logf" 2>/dev/null || echo 0) ))
        if (( age > STALL_SECS )); then
            err "  ↳ ${logf##*/}: ${age}s 동안 무갱신 — hang 으로 판단, kill"
            kill -KILL -- "-$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            return 124
        fi
    done
    wait "$pid"; rc=$?
    return "$rc"
}

declare -a RESULTS=()

# 진행 상황을 파일로도 남긴다. 콘솔 로그를 지켜보는 쪽(사람이든 에이전트든)이 끊겨도
# 이 파일만 보면 지금 몇 번째 task 가 어디까지 갔는지 알 수 있다.
STATUS_FILE="$LOG_DIR/STATUS.txt"
status_line() { printf '%s  %-22s %s\n' "$(date '+%F %T')" "$1" "$2" >> "$STATUS_FILE"; }

run_one() {
    local stack="$1" variant="$2"
    local pkg base_id train_task play_task
    pkg="$(pkg_dir_for "$stack")"
    base_id="$(task_id_for "$stack" "$variant")" || { err "unknown task: $stack:$variant"; return 1; }
    train_task="${base_id}-v0"
    play_task="${base_id}-Play-v0"
    local tag="${stack}_${variant}"

    info "════════════════════════════════════════"
    info " TRAIN  $train_task   envs=$NUM_ENVS iters=$MAX_ITER"
    info "════════════════════════════════════════"

    # Isaac Sim 의 부팅은 간헐적으로 실패한다 — 로그가 멈춘 채 매달리거나(watchdog 이 124 로
    # 잡는다), iteration 을 한 번도 안 돌고 exit 0 으로 끝난다(2026-08-30 flat:basedwq).
    # 둘 다 설정 문제가 아니라 재시도하면 그냥 된다. 예전에는 실패를 기록만 하고 다음 task 로
    # 넘어가서 사람이 나중에 손으로 다시 돌려야 했다 — 그 수동 개입을 없앤다.
    local exp run_dir trc attempt why
    exp="$(exp_name_for "$stack" "$variant")"
    for (( attempt = 1; attempt <= TRAIN_TRIES; attempt++ )); do
        (( attempt > 1 )) && info " RETRY  $train_task   시도 $attempt/$TRAIN_TRIES ($why)"
        run_watched "$LOG_DIR/${tag}_train.log" "$pkg" \
            rsl_rl/train.py --task="$train_task" --headless \
                --num_envs="$NUM_ENVS" --seed="$SEED" --max_iterations="$MAX_ITER" \
                --logger=wandb --log_project_name=lec_dreamwaq
        trc=$?

        run_dir="$(ls -1dt "$ROOT/$pkg/logs/rsl_rl/$exp"/*/ 2>/dev/null | head -1)"
        why=""
        if (( trc == 124 )); then
            why="STALLED"
        elif (( trc != 0 )); then
            why="rc=$trc"
        elif [[ -z "$run_dir" ]] \
             || ! compgen -G "$run_dir/events.out.tfevents.*" > /dev/null \
             || ! compgen -G "$run_dir/model_*.pt" > /dev/null; then
            # rc=0 인데 산출물이 없다. watchdog 은 '로그가 멈추는 hang' 만 잡으므로 이건 못 건다.
            why="산출물 없음"
        else
            break
        fi

        err "$tag: 학습 실패 ($why) — see $LOG_DIR/${tag}_train.log"
        # 반쯤 만들어진 run 폴더를 지운다. 남겨 두면 다음 시도가 만든 폴더와 섞이고,
        # play.py 의 '최신 run' 해석이 빈 폴더를 집는다.
        if [[ -n "$run_dir" ]] && ! compgen -G "$run_dir/events.out.tfevents.*" > /dev/null; then
            rm -rf "$run_dir"
        fi
        run_dir=""
        (( attempt < TRAIN_TRIES )) && sleep 20
    done

    if [[ -z "$run_dir" ]]; then
        err "$tag: 학습이 ${TRAIN_TRIES}번 모두 실패했다 ($why)"
        RESULTS+=( "$tag: TRAIN FAILED x${TRAIN_TRIES} ($why)" )
        status_line "$tag" "TRAIN FAILED x${TRAIN_TRIES} ($why)"
        return 1
    fi
    ok "$tag: training done${attempt:+ (시도 $attempt)}"
    status_line "$tag" "trained (시도 $attempt)"

    info " VIDEO  $play_task   ${VIDEO_LENGTH} steps, tracking env 0"
    # No --load_run/--checkpoint: the cfg defaults (load_run=".*", load_checkpoint="model_.*.pt")
    # resolve to the newest run and newest checkpoint, i.e. the run we just finished.
    # 영상 단계는 Vulkan 크래시(exit 139)로 간헐 실패하지만 체크포인트와 무관하므로
    # 1회 재시도하고, 그래도 실패하면 기록만 남기고 다음 task 로 넘어간다.
    local vattempt vrc=1
    for vattempt in 1 2; do
        run_watched "$LOG_DIR/${tag}_video.log" "$pkg" \
            rsl_rl/play.py --task="$play_task" --headless \
                --num_envs="$VIDEO_ENVS" --video --video_length="$VIDEO_LENGTH" \
                --track_agent --track_env_index=0
        vrc=$?
        (( vrc == 0 )) && break
        err "$tag: video 시도 $vattempt 실패 (rc=$vrc)"
    done

    # 방금 쓴 run 폴더에서 직접 찾는다 (시간창 기반 탐색은 긴 play 에서 오탐한다).
    local mp4
    mp4="$(find "${run_dir:-/nonexistent}" -name '*.mp4' 2>/dev/null | sort | tail -1)"

    if (( vrc != 0 )) || [[ -z "$mp4" ]]; then
        err "$tag: 학습은 성공, 영상 없음 — see $LOG_DIR/${tag}_video.log"
        RESULTS+=( "$tag: trained, VIDEO FAILED" )
        status_line "$tag" "trained, VIDEO FAILED"
        return 0
    fi
    ok "$tag: video → $mp4"
    RESULTS+=( "$tag: OK  $mp4" )
    status_line "$tag" "OK  $mp4"
    return 0
}

info "pipeline start — tasks: $TASKS"
info "envs=$NUM_ENVS iters=$MAX_ITER seed=$SEED  WANDB_MODE=$WANDB_MODE  STALL=${STALL_SECS}s"
info "체크포인트/tfevents/영상 → $ROOT/dreamwaq_manager/logs/rsl_rl/<experiment_name>/<timestamp>/"
info "wandb 로컬 사본        → $ROOT/dreamwaq_manager/wandb/"
info "파이프라인 콘솔 로그    → $LOG_DIR/"
info "진행 상황 파일        → $STATUS_FILE  (세션이 끊겨도 여기만 보면 된다)"
start_ts=$(date +%s)

: > "$STATUS_FILE"
status_line "pipeline" "start — tasks: $TASKS (envs=$NUM_ENVS iters=$MAX_ITER tries=$TRAIN_TRIES)"

for entry in $TASKS; do
    status_line "${entry%%:*}_${entry##*:}" "TRAIN 시작"
    run_one "${entry%%:*}" "${entry##*:}"
done

status_line "pipeline" "done"

end_ts=$(date +%s)
echo
info "════════════════════════════════════════"
info " pipeline done in $(( (end_ts - start_ts) / 60 )) min"
info "════════════════════════════════════════"
for r in "${RESULTS[@]}"; do
    if [[ "$r" == *OK* ]]; then ok "$r"; else err "$r"; fi
done
info "wandb (${WANDB_MODE}) → project 'lec_dreamwaq'"
info "산출물 → $ROOT/dreamwaq_manager/logs/rsl_rl/"
