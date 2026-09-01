#!/usr/bin/env python3
"""Stage 5 · task02 — tfevents 에서 3종 비교 지표를 뽑는다.  `L2 · BUILD`

강사 배포 자산(tensorboard 로그)이 있으면 그것을 읽고, 없으면 합성 데이터로 시연한다.
그림까지 그리는 완성본은 `dreamwaq_manager/scripts/compare_runs.py` 에 있다.

    python solution.py                    # 배포 자산이 있으면 자동으로 찾는다
    python solution.py --logdir <경로>     # 다른 곳에 풀었다면
"""
from __future__ import annotations

import argparse
from pathlib import Path

TRACK_TAG = "Episode_Reward/track_lin_vel_xy_exp"
TRACK_WEIGHT = 1.5  # 공식 Go2 레시피(IsaacLab UnitreeGo2RoughEnvCfg)의 track_lin_vel_xy_exp 가중치.
                   # DreamWaQ 레시피(velocity_env_cfg.py)의 track_lin_vel_xy 는 1.0 이다 — 다른 항이다.

CONFIGS = ["Base", "Waq", "Oracle"]
EXPERIMENTS = {
    "Base": "BaseDwq-Official-Rough-PPO-v0",
    "Waq": "Waq-Official-Rough-PPO-v0",
    "Oracle": "OracleDwq-Official-Rough-PPO-v0",
}
DEFAULT_LOGDIR = Path(__file__).resolve().parents[3] / "dreamwaq_manager" / "logs" / "rsl_rl"


def summarize(series: dict[str, list[float]], weight: float = TRACK_WEIGHT,
              last_k: int = 300) -> dict[str, float]:
    """각 config 의 최종 underlying tracking 과 두 델타를 구한다.

    Args:
        series: {config 이름: track_lin_vel_xy_exp 로그값 리스트}. 없는 config 는 빠져 있거나 빈 리스트.
        weight: 보상 가중치. 이걸로 나눠 underlying (0~1) 로 되돌린다.
        last_k: 최종값을 낼 때 평균낼 마지막 로그 개수. 기본 300.

    왜 300 인가: 이 지표는 iteration 마다 sd ~0.03 으로 출렁인다. 마지막 3점만 평균내면
    평균의 표준오차가 0.017 로 재려는 차이(~0.01)보다 커진다. 배포 자산(seed42, 4000 it)에서
    rough 의 Waq-Base 는 last-3 이 +0.030, last-400 이 +0.0073 으로 **크기가 4배** 부풀고,
    flat 은 last-3 이 -0.012, last-400 이 +0.0007 로 **부호가 뒤집힌다.**

    Returns:
        {"Base": .., "Waq": .., "Oracle": .., "Waq-Base": .., "Oracle-Waq": ..}
        데이터가 없는 항목은 키 자체를 넣지 않는다.
    """
    # ex:begin id=stage5-summary level=2 stage=stage5_compare inline_hints=2 task=보상 로그에서 underlying 지표와 두 델타를 계산한다
    #   hint: underlying = 로그값 / weight — 가중치를 되돌려야 run 끼리 비교할 수 있다
    #   hint: 최종값은 마지막 last_k 개의 평균이다. 몇 점만 쓰면 노이즈가 차이보다 커져 부호가 뒤집힌다
    #   hint: 값이 last_k 개보다 적으면 있는 것만 평균낸다. 빈 리스트는 건너뛴다 (0 나누기)
    #   hint: 'Waq-Base' 는 Waq 와 Base 가 둘 다 있을 때만 넣는다
    #   hint: 빠진 config 는 결과에 아예 넣지 않는다 (0.0 으로 채우면 그림이 거짓말을 한다)
    out: dict[str, float] = {}
    for cfg in CONFIGS:
        vals = series.get(cfg) or []
        if not vals:
            continue
        tail = vals[-last_k:]
        out[cfg] = sum(tail) / len(tail) / weight

    if "Waq" in out and "Base" in out:
        out["Waq-Base"] = out["Waq"] - out["Base"]
    if "Oracle" in out and "Waq" in out:
        out["Oracle-Waq"] = out["Oracle"] - out["Waq"]
    return out
    # ex:end


def load_run_series(logdir: Path) -> dict[str, list[float]]:
    """배포 자산에서 config 별 track 보상 로그를 읽는다. 없으면 빈 dict."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("[warn] tensorboard 가 없다 — 합성 데이터로 시연한다")
        return {}

    series: dict[str, list[float]] = {}
    for cfg, exp in EXPERIMENTS.items():
        exp_dir = logdir / exp
        if not exp_dir.is_dir():
            continue
        events = sorted(exp_dir.glob("*/events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
        if not events:
            continue
        ea = event_accumulator.EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
        ea.Reload()
        if TRACK_TAG in ea.Tags().get("scalars", []):
            series[cfg] = [e.value for e in ea.Scalars(TRACK_TAG)]
    return series


def demo_series() -> dict[str, list[float]]:
    """배포 자산이 없을 때 쓰는 합성 로그. rough 배포 런의 최종값 근처를 흉내낸다."""
    return {
        "Base": [0.30 * TRACK_WEIGHT, 0.52 * TRACK_WEIGHT, 0.545 * TRACK_WEIGHT, 0.549 * TRACK_WEIGHT],
        "Waq": [0.30 * TRACK_WEIGHT, 0.52 * TRACK_WEIGHT, 0.551 * TRACK_WEIGHT, 0.557 * TRACK_WEIGHT],
        "Oracle": [0.32 * TRACK_WEIGHT, 0.55 * TRACK_WEIGHT, 0.576 * TRACK_WEIGHT, 0.582 * TRACK_WEIGHT],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR)
    args = ap.parse_args()

    series = load_run_series(args.logdir) if args.logdir.is_dir() else {}
    if series:
        print(f"배포 자산에서 읽음: {args.logdir}")
        print(f"  찾은 run: {', '.join(series)}")
    else:
        print("배포 자산이 없어 합성 데이터로 시연한다 (수치는 예시일 뿐이다)")
        series = demo_series()

    result = summarize(series)

    print("\n=== rough 최종 underlying tracking ===")
    for cfg in CONFIGS:
        v = result.get(cfg)
        print(f"  {cfg:8} {v:.3f}" if v is not None else f"  {cfg:8}   -")

    print("\n=== 두 델타 ===")
    for key, label in (("Waq-Base", "CENet 의 순수 기여"), ("Oracle-Waq", "남은 격차")):
        v = result.get(key)
        print(f"  {key:12} {v:+.3f}   {label}" if v is not None else f"  {key:12}    -    {label}")

    if all(c in result for c in CONFIGS):
        ordered = result["Base"] < result["Waq"] < result["Oracle"]
        print(f"\nBase < Waq < Oracle 인가? {'예 — 논문 순서와 일치' if ordered else '아니오 — 해석이 필요하다'}")

    print("\n[PASS] 지표 계산 완료. 그림은 dreamwaq_manager/scripts/compare_runs.py 로 그린다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
