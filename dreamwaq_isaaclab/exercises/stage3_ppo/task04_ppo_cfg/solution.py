#!/usr/bin/env python3
"""Stage 3 · task04 — PPO cfg 와 이 repo 의 실전 이슈.  `L1 · FILL`

task03 에서 "우리에게 없는 것" 으로 넘긴 두 가지를 여기서 마무리한다.

  1. desired_kl 적응형 학습률  <- 빈칸
  2. log_std clamp             <- 이 repo 가 실제로 겪은 문제. 읽고 실행해 본다

    python solution.py
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

# 이 repo 의 실제 PPO cfg. 값이 바뀌면 여기 출력도 따라 바뀐다.
_CFG_PATH = (
    Path(__file__).resolve().parents[3]
    / "dreamwaq_manager/source/dreamwaq_manager/dreamwaq_manager/tasks/locomotion/config/go2"
    / "agents/rsl_rl_ppo_cfg.py"
)

# train.py 의 몽키패치가 쓰는 값과 같아야 한다.
LOG_STD_MIN, LOG_STD_MAX = -5.0, 0.0


def adapt_learning_rate(kl_mean: float, learning_rate: float, desired_kl: float = 0.01) -> float:
    """KL 을 보고 학습률을 조절한다 (rsl_rl `PPO.update` 의 adaptive 스케줄).

    Args:
        kl_mean: 이전 정책과 현재 정책 사이 KL 의 미니배치 평균
        learning_rate: 현재 학습률
        desired_kl: 목표 KL (이 repo 는 0.01)

    Returns:
        갱신된 학습률. 목표의 절반~2배 사이면 그대로 둔다.
    """
    # ex:begin id=ppo-cfg level=1 stage=stage3_ppo task=KL 을 보고 학습률을 올리거나 내리는 규칙을 채운다
    #   hint: KL 이 desired_kl 의 2배보다 크면 너무 많이 움직인 것 -> lr 을 1.5 로 나눈다
    #   hint: KL 이 desired_kl 의 절반보다 작으면 너무 적게 움직인 것 -> lr 에 1.5 를 곱한다
    #   hint: 그 사이(절반~2배)는 손대지 않는다 — 이 '죽은 구간' 이 진동을 막는다
    #   hint: 하한 1e-5, 상한 1e-2 로 묶는다 (max / min 을 쓴다)
    #   hint: kl_mean 이 0 이면 올리지 않는다 (rsl_rl 의 kl_mean > 0.0 조건)
    if kl_mean > desired_kl * 2.0:
        learning_rate = max(1e-5, learning_rate / 1.5)
    elif kl_mean < desired_kl / 2.0 and kl_mean > 0.0:
        learning_rate = min(1e-2, learning_rate * 1.5)
    return learning_rate
    # ex:end


def clamp_log_std(log_std: torch.Tensor) -> torch.Tensor:
    """log_std 를 [-5, 0] 으로 묶고 NaN/inf 를 되돌린다. (train.py 의 몽키패치와 같다)

    이 함수는 주어진다 — 채울 필요 없다. 아래 main() 에서 왜 필요한지 직접 본다.
    """
    log_std = torch.nan_to_num(log_std, nan=0.0, posinf=0.0, neginf=LOG_STD_MIN)
    return log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)


def _read_cfg_values() -> dict[str, str]:
    """cfg 파일에서 algorithm 블록의 값을 그대로 읽어 온다 (isaaclab import 없이)."""
    if not _CFG_PATH.exists():
        return {}
    text = _CFG_PATH.read_text(encoding="utf-8")
    block = re.search(r"algorithm = RslRlPpoAlgorithmCfg\((.*?)\n    \)", text, re.S)
    if not block:
        return {}
    return dict(re.findall(r"(\w+)\s*=\s*([^,\n]+),", block.group(1)))


def main() -> int:
    print("=" * 72)
    print(" 1. 이 repo 의 PPO cfg  (rsl_rl_ppo_cfg.py 에서 직접 읽음)")
    print("=" * 72)

    meaning = {
        "clip_param": "surrogate 의 ratio 를 1±eps 로 자른다. 논문 §III-B",
        "value_loss_coef": "value 손실 가중치",
        "entropy_coef": "엔트로피 보너스. 탐험을 유지한다",
        "num_learning_epochs": "같은 롤아웃을 몇 번 재사용하는가 (on-policy 의 타협)",
        "num_mini_batches": "epoch 당 미니배치 수",
        "learning_rate": "초기 학습률. adaptive 면 여기서 출발해 움직인다",
        "schedule": "adaptive 면 desired_kl 로 lr 을 조절",
        "gamma": "할인율",
        "lam": "GAE 람다",
        "desired_kl": "목표 KL. 이 값 기준으로 lr 이 오르내린다",
        "max_grad_norm": "gradient clipping",
        "use_clipped_value_loss": "critic 예측도 old value 기준으로 자른다",
    }
    values = _read_cfg_values()
    if values:
        for k, v in values.items():
            print(f"  {k:<24} = {v.strip():<10}  {meaning.get(k, '')}")
    else:
        print("  (cfg 파일을 찾지 못했다 — repo 밖에서 실행했을 수 있다)")

    print("\n  이 값들은 임의로 고른 것이 아니라 DreamWaQ 논문 §III-B 와 같다.")

    print("\n" + "=" * 72)
    print(" 2. 적응형 학습률 — KL 을 보고 lr 이 움직인다")
    print("=" * 72)

    desired_kl = 0.01
    lr = 1.0e-3
    # 학습 중 흔히 보는 KL 흐름: 처음엔 크게 움직이다가(정책이 급변) 점점 잦아든다.
    kl_trace = [0.05, 0.03, 0.02, 0.012, 0.008, 0.004, 0.002, 0.001, 0.0005, 0.0]
    print(f"  desired_kl = {desired_kl},  시작 lr = {lr:.2e}")
    print(f"  {'KL':>8}  {'판정':<10} {'lr':>10}")
    for kl in kl_trace:
        new_lr = adapt_learning_rate(kl, lr, desired_kl)
        verdict = "낮춤" if new_lr < lr else ("높임" if new_lr > lr else "유지")
        print(f"  {kl:8.4f}  {verdict:<10} {new_lr:10.3e}")
        lr = new_lr

    assert lr > 1e-3, "KL 이 작아지면 lr 이 올라가야 한다"
    print("\n  [PASS] KL 이 크면 조심스럽게, 작으면 과감하게 — 스스로 보폭을 정한다.")

    print("\n" + "=" * 72)
    print(" 3. log_std clamp — 이 repo 가 실제로 겪은 문제")
    print("=" * 72)
    print("""
  actor 의 액션 분포는 학습 가능한 log_std 를 갖는다. 그런데 적응형 lr 과
  좁은 관측 분포가 겹치면 log_std 가 학습 도중 극단으로 흘러간다.
  한 번 ±inf 가 되면 Normal 표집이 NaN 을 뱉고 정책 전체가 죽는다 —
  그것도 몇 시간 학습한 뒤에.""")

    drift = torch.tensor([0.0, 0.0])
    print(f"\n  {'step':>6}  {'clamp 없음':>22}  {'clamp 있음':>22}")
    guarded = drift.clone()
    for step in range(1, 6):
        drift = drift - 2.0                      # 폭주하는 gradient 를 흉내낸다
        guarded = clamp_log_std(guarded - 2.0)   # 매 optimizer step 뒤에 거는 훅과 같다
        print(f"  {step:6d}  log_std={drift[0]:8.2f} std={drift[0].exp():8.2e}"
              f"  log_std={guarded[0]:6.2f} std={guarded[0].exp():6.3f}")

    nan_case = clamp_log_std(torch.tensor([float("-inf"), float("nan")]))
    print(f"\n  -inf, nan 이 들어오면 -> {nan_case.tolist()}  (되살아난다)")

    assert torch.isfinite(nan_case).all()
    assert float(guarded[0]) >= LOG_STD_MIN
    print(f"\n  [PASS] log_std 가 [{LOG_STD_MIN:.0f}, {LOG_STD_MAX:.0f}] 안에 머문다.")
    print("\n  실제 코드: dreamwaq_manager/scripts/rsl_rl/train.py 의")
    print("            _register_std_clamp_for_all_runners() — optimizer step 후 훅으로 건다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
