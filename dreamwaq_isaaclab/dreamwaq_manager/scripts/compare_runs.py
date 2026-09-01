#!/usr/bin/env python3
"""Base / Waq / Oracle 비교 그림과 요약표를 만든다 (flat + rough).

rsl_rl 이 남긴 tensorboard event 파일을 읽어 강의용 산출물을 만든다.

    figures/curves_flat.png          속도추종 곡선 (3종)
    figures/curves_rough.png         속도추종 곡선 (3종)
    figures/reward_{flat,rough}.png  총 에피소드 보상 곡선
    figures/terrain_level_rough.png  지형 커리큘럼 진행
    figures/bar_final.png            최종 속도추종 막대
    figures/summary.csv              최종값 + Waq-Base / Oracle-Waq

핵심 지표 `underlying` 은 보상 로그를 가중치로 되돌린 값이다.

    underlying = (Episode_Reward/track_lin_vel_xy_exp) / 1.5
               = mean exp(-||v_cmd - v||^2 / 0.25)   in [0, 1]   (1.0 = 완벽 추종)

가중치를 나눠 주지 않으면 보상 설계가 바뀔 때 값이 같이 흔들려 비교가 안 된다.

강의의 결론 두 줄이 여기서 나온다.

    Waq - Base    = CENet 의 순수 기여   (같은 env, actor 관측만 다르다)
    Oracle - Waq  = 남은 격차            (추정이 진짜 특권정보에 얼마나 못 미치는가)

Isaac Lab 번들 python 으로 돌린다 (tensorboard 가 필요하다):

    cd dreamwaq_manager && ~/IsaacLab/_isaac_sim/python.sh scripts/compare_runs.py

아직 학습하지 않은 run 은 경고만 내고 건너뛰므로, 학습 도중에도 돌려 볼 수 있다.
"""
from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing import event_accumulator  # noqa: E402


def _use_korean_font() -> None:
    """제목에 한글을 쓰므로 CJK 폰트를 찾아 쓴다. 없으면 조용히 기본 폰트로 둔다."""
    import matplotlib.font_manager as fm

    have = {f.name for f in fm.fontManager.ttflist}
    for name in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic"):
        if name in have:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트는 유니코드 마이너스가 깨진다


_use_korean_font()

TRACK_TAG = "Episode_Reward/track_lin_vel_xy_exp"
TERRAIN_TAG = "Curriculum/terrain_levels"
REWARD_TAG = "Train/mean_reward"
TRACK_WEIGHT = 1.5  # 공식 Go2 레시피(IsaacLab UnitreeGo2RoughEnvCfg)의 track_lin_vel_xy_exp 가중치.
                   # DreamWaQ 레시피(velocity_env_cfg.py)의 track_lin_vel_xy 는 1.0 이다 — 다른 항이다.

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGDIR = REPO_ROOT / "dreamwaq_manager" / "logs" / "rsl_rl"
DEFAULT_OUTDIR = REPO_ROOT / "figures"

# 순서 = 하한 -> 상한. Waq 를 가운데 두어야 두 델타가 눈에 들어온다.
CONFIGS = ["Base", "Waq", "Oracle"]
COLORS = {"Base": "#888888", "Waq": "#1f77b4", "Oracle": "#2ca02c"}
OBS_DIM = {"Base": 45, "Waq": 64, "Oracle": 48}
EXPERIMENTS = {
    "flat": {
        "Base": "BaseDwq-Official-Flat-PPO-v0",
        "Waq": "Waq-Official-Flat-PPO-v0",
        "Oracle": "OracleDwq-Official-Flat-PPO-v0",
    },
    "rough": {
        "Base": "BaseDwq-Official-Rough-PPO-v0",
        "Waq": "Waq-Official-Rough-PPO-v0",
        "Oracle": "OracleDwq-Official-Rough-PPO-v0",
    },
}


def latest_event_file(logdir: Path, experiment: str) -> Path | None:
    exp_dir = logdir / experiment
    if not exp_dir.is_dir():
        return None
    runs = sorted((p for p in exp_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    for run in reversed(runs):
        events = sorted(run.glob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
        if events:
            return events[-1]
    return None


def load_scalar(ev: Path, tag: str) -> tuple[list[int], list[float]]:
    ea = event_accumulator.EventAccumulator(str(ev), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def smooth(ys: list[float], n: int) -> list[float]:
    """이동평균. 곡선이 튀어서 3종 비교가 안 보이는 것을 막는다."""
    if n <= 1 or len(ys) < n:
        return ys
    out: list[float] = []
    win: deque = deque(maxlen=n)
    for y in ys:
        win.append(y)
        out.append(sum(win) / len(win))
    return out


def tail_window(n_points: int, last_k: int | None) -> int:
    """평균낼 꼬리 길이. 기본은 런의 마지막 10% (최소 50 포인트)."""
    if last_k is not None:
        return max(1, last_k)
    return max(50, n_points // 10)


def final_value(ys: list[float], last_k: int | None = None) -> float | None:
    """수렴 구간의 평균.

    이 지표는 iteration 마다 sd ~0.03 으로 출렁인다. 마지막 몇 점만 평균내면
    표준오차가 재려는 차이(~0.02)보다 커져서 **부호가 뒤집힌다.**
    실제로 last-3 은 Waq-Base=+0.04, last-100 은 -0.02 가 나온다.
    그래서 기본값을 런의 마지막 10% 로 둔다.
    """
    if not ys:
        return None
    tail = ys[-tail_window(len(ys), last_k):]
    return sum(tail) / len(tail)


def final_stats(ys: list[float], last_k: int | None = None) -> tuple[float, float, int] | None:
    """(평균, 평균의 표준오차, 사용한 포인트 수). 차이가 노이즈인지 판단하는 데 쓴다."""
    if not ys:
        return None
    n = tail_window(len(ys), last_k)
    tail = ys[-n:]
    m = sum(tail) / len(tail)
    if len(tail) < 2:
        return m, float("nan"), len(tail)
    var = sum((v - m) ** 2 for v in tail) / (len(tail) - 1)
    return m, (var / len(tail)) ** 0.5, len(tail)


def plot_curves(logdir: Path, terrain: str, outdir: Path, smooth_n: int,
                last_k: int | None = None,
                stats_out: dict[str, tuple[float, float, int]] | None = None) -> dict[str, float]:
    finals: dict[str, float] = {}
    fig = plt.figure(figsize=(7, 4.5))
    for cfg in CONFIGS:
        ev = latest_event_file(logdir, EXPERIMENTS[terrain][cfg])
        if ev is None:
            print(f"[warn] {terrain}/{cfg}: run 이 없다 ({EXPERIMENTS[terrain][cfg]}) — 건너뜀")
            continue
        steps, vals = load_scalar(ev, TRACK_TAG)
        if not steps:
            print(f"[warn] {terrain}/{cfg}: 태그 {TRACK_TAG} 없음 — 건너뜀")
            continue
        under = [v / TRACK_WEIGHT for v in vals]
        fv = final_value(under, last_k)
        if fv is not None:
            finals[cfg] = fv
        if stats_out is not None:
            s = final_stats(under, last_k)
            if s is not None:
                stats_out[cfg] = s
        plt.plot(steps, smooth(under, smooth_n), color=COLORS[cfg], lw=1.8,
                 label=f"{cfg} (obs {OBS_DIM[cfg]})  final={fv:.3f}")
    if not finals:  # 학습 전이면 빈 그림을 남기지 않는다
        plt.close(fig)
        print(f"[warn] curves_{terrain}: 데이터 없음 — 건너뜀")
        return finals
    plt.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.5)
    plt.xlabel("iteration")
    plt.ylabel("underlying tracking  (1.0 = perfect)")
    plt.title(f"PPO {terrain.capitalize()} — Base vs Waq vs Oracle")
    plt.ylim(0, 1.02)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = outdir / f"curves_{terrain}.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[ok] {out}")
    return finals


def plot_reward_curves(logdir: Path, terrain: str, outdir: Path, smooth_n: int) -> None:
    plt.figure(figsize=(7, 4.5))
    any_data = False
    for cfg in CONFIGS:
        ev = latest_event_file(logdir, EXPERIMENTS[terrain][cfg])
        if ev is None:
            continue
        steps, vals = load_scalar(ev, REWARD_TAG)
        if not steps:
            continue
        any_data = True
        plt.plot(steps, smooth(vals, smooth_n), color=COLORS[cfg], lw=1.8,
                 label=f"{cfg} (obs {OBS_DIM[cfg]})  final={final_value(vals):.2f}")
    if not any_data:
        plt.close()
        print(f"[warn] reward_{terrain}: 데이터 없음 — 건너뜀")
        return
    plt.xlabel("iteration")
    plt.ylabel("mean episode reward (sum)")
    plt.title(f"PPO {terrain.capitalize()} — 총 보상: Base vs Waq vs Oracle")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = outdir / f"reward_{terrain}.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[ok] {out}")


def plot_terrain_level(logdir: Path, outdir: Path, smooth_n: int) -> None:
    """지형 커리큘럼. 잘 걷는 정책일수록 더 어려운 지형으로 승급한다."""
    plt.figure(figsize=(7, 4.5))
    any_data = False
    for cfg in CONFIGS:
        ev = latest_event_file(logdir, EXPERIMENTS["rough"][cfg])
        if ev is None:
            continue
        steps, vals = load_scalar(ev, TERRAIN_TAG)
        if not steps:
            continue
        any_data = True
        plt.plot(steps, smooth(vals, smooth_n), color=COLORS[cfg], lw=1.8, label=cfg)
    if not any_data:
        plt.close()
        print("[warn] terrain_level: 데이터 없음 — 건너뜀")
        return
    plt.xlabel("iteration")
    plt.ylabel("terrain level (curriculum)")
    plt.title("PPO Rough — 지형 커리큘럼 승급")
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = outdir / "terrain_level_rough.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[ok] {out}")


def plot_bars(finals: dict[str, dict[str, float]], outdir: Path) -> None:
    import numpy as np

    terrains = [t for t in ("flat", "rough") if finals.get(t)]
    if not terrains:
        print("[warn] bar_final: 최종값 없음 — 건너뜀")
        return
    x = np.arange(len(terrains))
    width = 0.25
    plt.figure(figsize=(7, 4.5))
    for i, cfg in enumerate(CONFIGS):
        ys = [finals[t].get(cfg, 0.0) for t in terrains]
        bars = plt.bar(x + (i - 1) * width, ys, width, color=COLORS[cfg], label=cfg)
        for b, y in zip(bars, ys):
            if y > 0:
                plt.text(b.get_x() + b.get_width() / 2, y + 0.01, f"{y:.3f}",
                         ha="center", va="bottom", fontsize=8)
    plt.xticks(x, [t.capitalize() for t in terrains])
    plt.ylabel("final underlying tracking")
    plt.title("PPO 최종 속도추종 — Base vs Waq vs Oracle")
    plt.ylim(0, 1.05)
    plt.legend(fontsize=9)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out = outdir / "bar_final.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[ok] {out}")


def write_summary(finals: dict[str, dict[str, float]], outdir: Path,
                  stats: dict[str, dict[str, tuple[float, float, int]]] | None = None) -> None:
    """최종값과 두 델타를 CSV 로 남기고 화면에도 찍는다."""
    stats = stats or {}
    out = outdir / "summary.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["terrain", "Base", "Waq", "Oracle", "Waq-Base", "Oracle-Waq"])
        for terrain in ("flat", "rough"):
            f = finals.get(terrain, {})
            b, waq, o = f.get("Base"), f.get("Waq"), f.get("Oracle")
            w.writerow([
                terrain,
                "" if b is None else f"{b:.4f}",
                "" if waq is None else f"{waq:.4f}",
                "" if o is None else f"{o:.4f}",
                "" if (b is None or waq is None) else f"{waq - b:+.4f}",
                "" if (o is None or waq is None) else f"{o - waq:+.4f}",
            ])
    print(f"[ok] {out}")

    def cell(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "  -  "

    print("\n=== 최종 underlying tracking (수렴 구간 평균) ===")
    print(f"{'terrain':8} {'Base':>7} {'Waq':>7} {'Oracle':>7} {'Waq-Base':>10} {'Oracle-Waq':>11}")
    for terrain in ("flat", "rough"):
        f = finals.get(terrain, {})
        b, waq, o = f.get("Base"), f.get("Waq"), f.get("Oracle")
        d1 = f"{waq - b:+.3f}" if (b is not None and waq is not None) else "   -  "
        d2 = f"{o - waq:+.3f}" if (o is not None and waq is not None) else "   -  "
        print(f"{terrain:8} {cell(b):>7} {cell(waq):>7} {cell(o):>7} {d1:>10} {d2:>11}")

    # 차이가 노이즈인지 판단할 수 있게 표준오차를 함께 낸다.
    for terrain in ("flat", "rough"):
        st = stats.get(terrain, {})
        if not st:
            continue
        print(f"\n--- {terrain} 수렴 구간 통계 (평균 ± 평균의 표준오차) ---")
        for cfg in CONFIGS:
            if cfg in st:
                m, se, n = st[cfg]
                print(f"  {cfg:8} {m:.4f} ± {se:.4f}   (마지막 {n} iteration 평균)")
        if "Waq" in st and "Base" in st:
            d = st["Waq"][0] - st["Base"][0]
            se = (st["Waq"][1] ** 2 + st["Base"][1] ** 2) ** 0.5
            verdict = "유의" if abs(d) > 2 * se else "노이즈와 구분 안 됨"
            print(f"  Waq-Base   {d:+.4f} ± {se:.4f}  → {verdict}")
        if "Oracle" in st and "Waq" in st:
            d = st["Oracle"][0] - st["Waq"][0]
            se = (st["Oracle"][1] ** 2 + st["Waq"][1] ** 2) ** 0.5
            verdict = "유의" if abs(d) > 2 * se else "노이즈와 구분 안 됨"
            print(f"  Oracle-Waq {d:+.4f} ± {se:.4f}  → {verdict}")
        print("  주의: 단일 seed 다. 표준오차는 '한 런 안에서의 출렁임'만 재며,"
              " seed 간 변동은 포함하지 않는다.")

    rough = finals.get("rough", {})
    if all(k in rough for k in CONFIGS):
        ordered = rough["Base"] < rough["Waq"] < rough["Oracle"]
        print(f"\nrough 에서 Base < Waq < Oracle 인가? "
              f"{'예 — 논문 순서와 일치' if ordered else '아니오 — 해석이 필요하다'}")
        print("  Waq - Base   = CENet 의 순수 기여")
        print("  Oracle - Waq = 남은 격차 (추정 vs 진짜 특권정보)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Base/Waq/Oracle 비교 그림과 요약표")
    ap.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR,
                    help="rsl_rl 로그 루트 (기본: dreamwaq_manager/logs/rsl_rl)")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                    help="그림/CSV 출력 폴더 (기본: figures/)")
    ap.add_argument("--smooth", type=int, default=5, help="곡선 이동평균 창 크기")
    ap.add_argument("--last-k", type=int, default=None,
                    help="최종값을 낼 때 평균낼 마지막 iteration 수 (기본: 런의 10%%, 최소 50). "
                         "작게 주면 노이즈에 휘둘려 부호가 뒤집힐 수 있다")
    args = ap.parse_args()

    if not args.logdir.is_dir():
        raise SystemExit(f"로그 폴더가 없다: {args.logdir}\n  먼저 학습을 돌린다 (run_full_pipeline.sh)")
    args.outdir.mkdir(parents=True, exist_ok=True)

    finals: dict[str, dict[str, float]] = {}
    stats: dict[str, dict[str, tuple[float, float, int]]] = {}
    for terrain in ("flat", "rough"):
        stats[terrain] = {}
        finals[terrain] = plot_curves(args.logdir, terrain, args.outdir, args.smooth,
                                      args.last_k, stats[terrain])
        plot_reward_curves(args.logdir, terrain, args.outdir, args.smooth)
    plot_terrain_level(args.logdir, args.outdir, args.smooth)
    plot_bars(finals, args.outdir)
    write_summary(finals, args.outdir, stats)


if __name__ == "__main__":
    main()
