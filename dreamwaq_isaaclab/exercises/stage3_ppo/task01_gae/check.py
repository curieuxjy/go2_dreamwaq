#!/usr/bin/env python3
"""빠른 검증 — ppo-gae.  (Isaac Sim 불필요, ~1초)

    python check.py                 # starter.py
    python check.py --solution      # ../ppo.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOLUTION = HERE.parent / "ppo.py"
STARTER = HERE / "starter.py"


def load(path: Path):
    if not path.exists():
        raise SystemExit(f"파일이 없다: {path}\n  make_exercise.py --id ppo-gae 로 생성한다")
    spec = importlib.util.spec_from_file_location(f"_ppo_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def reference_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """검사기 자체의 독립 구현 — 학습자 코드와 대조한다."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    acc = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        nxt = last_value if t == T - 1 else values[t + 1]
        nd = 1.0 - dones[t]
        delta = rewards[t] + gamma * nxt * nd - values[t]
        acc = delta + gamma * lam * nd * acc
        adv[t] = acc
    return adv + values, adv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"검사 대상: {path.relative_to(REPO_ROOT)}\n")
    gae = load(path).compute_gae

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    T, N = 8, 4
    torch.manual_seed(0)
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = torch.zeros(T, N)
    last_value = torch.randn(N)

    try:
        ret, adv = gae(rewards, values, dones, last_value)
    except NotImplementedError:
        print("  아직 TODO(ppo-gae) 가 비어 있다. starter.py 를 채운다.")
        return 1

    check("1. 반환 shape 이 (T, N) 두 개", ret.shape == (T, N) and adv.shape == (T, N),
          f"returns={tuple(ret.shape)} adv={tuple(adv.shape)}")

    check("2. returns == advantages + values", torch.allclose(ret, adv + values, atol=1e-6))

    # 독립 구현과 대조
    ref_ret, ref_adv = reference_gae(rewards, values, dones, last_value)
    check("3. 독립 구현과 값이 일치", torch.allclose(adv, ref_adv, atol=1e-5),
          f"최대 오차 {(adv - ref_adv).abs().max():.2e}")

    # lam=0 이면 1-step TD 오차와 정확히 같아야 한다.
    _, adv0 = gae(rewards, values, dones, last_value, gamma=0.99, lam=0.0)
    nxt = torch.cat([values[1:], last_value.unsqueeze(0)], dim=0)
    td = rewards + 0.99 * nxt - values
    check("4. lam=0 이면 1-step TD 오차와 같다", torch.allclose(adv0, td, atol=1e-5),
          f"최대 오차 {(adv0 - td).abs().max():.2e}")

    # gamma=lam=1, done 없음 -> 어드밴티지는 미래 보상 합 - V(s_t)
    _, adv1 = gae(rewards, values, dones, last_value, gamma=1.0, lam=1.0)
    mc = torch.zeros_like(rewards)
    for t in range(T):
        mc[t] = rewards[t:].sum(0) + last_value - values[t]
    check("5. gamma=lam=1 이면 몬테카를로와 같다", torch.allclose(adv1, mc, atol=1e-4),
          f"최대 오차 {(adv1 - mc).abs().max():.2e}")

    # done 이 부트스트랩을 끊는가 — 이 실습의 핵심 함정
    d = torch.zeros(T, N)
    d[3] = 1.0
    _, adv_d = gae(rewards, values, d, last_value)
    _, ref_d = reference_gae(rewards, values, d, last_value)
    check("6. done 스텝에서 부트스트랩을 끊는다", torch.allclose(adv_d, ref_d, atol=1e-5),
          f"최대 오차 {(adv_d - ref_d).abs().max():.2e}")

    # done 직전 스텝의 어드밴티지가 done 여부에 따라 달라져야 한다
    check("7. done 이 실제로 결과를 바꾼다", not torch.allclose(adv_d[3], adv[3], atol=1e-6),
          "done=1 인데 값이 그대로다 — (1 - done) 을 곱하지 않았다")

    # 마지막 스텝은 last_value 로 부트스트랩한다
    other = last_value + 10.0
    _, adv_o = gae(rewards, values, dones, other)
    check("8. 마지막 스텝이 last_value 를 쓴다", not torch.allclose(adv_o[-1], adv[-1], atol=1e-6),
          "last_value 를 바꿔도 마지막 어드밴티지가 그대로다")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)}개 실패\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32m전부 통과.\033[0m 이어서 task02_ppo_update 로 간다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
