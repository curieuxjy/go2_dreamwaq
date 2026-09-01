#!/usr/bin/env python3
"""빠른 검증 — reward-power-distribution.  (Isaac Sim 불필요, ~1초)

    python check.py                 # starter/rewards.py
    python check.py --solution      # 완성본
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fake_isaaclab import load_module  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOLUTION = (
    REPO_ROOT
    / "dreamwaq_manager/source/dreamwaq_manager/dreamwaq_manager/tasks/locomotion/mdp/rewards.py"
)
STARTER = HERE / "starter" / "rewards.py"

NUM_ENVS, NUM_JOINTS = 4, 12


class Cfg:
    def __init__(self, name="robot", joint_ids=None):
        self.name = name
        self.joint_ids = slice(None) if joint_ids is None else joint_ids


class FakeEnv:
    """rewards.py 가 읽는 것만 흉내낸다: env.scene[name].data.{applied_torque, joint_vel}"""

    def __init__(self, torque: torch.Tensor, joint_vel: torch.Tensor):
        data = type("D", (), {"applied_torque": torque, "joint_vel": joint_vel})()
        asset = type("A", (), {"data": data})()
        self.scene = type("S", (), {"__getitem__": lambda _s, _k: asset})()
        self.num_envs, self.device = torque.shape[0], "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"검사 대상: {path.relative_to(REPO_ROOT)}\n")
    fn = load_module(path).power_distribution_l2

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    ones = torch.ones(NUM_ENVS, NUM_JOINTS)
    try:
        r = fn(FakeEnv(ones, ones), asset_cfg=Cfg())
    except NotImplementedError:
        print("  아직 TODO(reward-power-distribution) 가 비어 있다. starter/rewards.py 를 채운다.")
        return 1

    check("1. 반환이 (num_envs,) 스칼라 벡터", tuple(r.shape) == (NUM_ENVS,), f"shape={tuple(r.shape)}")

    # 모든 관절이 같은 파워를 쓰면 분산 0 -> 페널티 0. 이 항의 정의 그 자체다.
    check("2. 모든 관절 파워가 같으면 0", torch.allclose(r, torch.zeros(NUM_ENVS), atol=1e-8),
          f"{r[0].item():.6f}")

    # 파워 크기가 커도 '균등'하면 여전히 0 — 총량이 아니라 불균형을 벌한다.
    r = fn(FakeEnv(ones * 10.0, ones * 10.0), asset_cfg=Cfg())
    check("3. 파워가 커도 균등하면 0 (총량이 아니라 불균형을 본다)",
          torch.allclose(r, torch.zeros(NUM_ENVS), atol=1e-6), f"{r[0].item():.6f}")

    # 알려진 값: 파워가 관절마다 [0..11] 이면 var (unbiased) = 13.0, 제곱 = 169.0
    ramp = torch.arange(NUM_JOINTS, dtype=torch.float32).repeat(NUM_ENVS, 1)
    r = fn(FakeEnv(ramp, torch.ones_like(ramp)), asset_cfg=Cfg())
    expected = torch.var(ramp, dim=-1).square()
    check("4. 알려진 분포에서 값이 정확 (var^2)", torch.allclose(r, expected, atol=1e-4),
          f"{r[0].item():.4f} vs {expected[0].item():.4f}")

    # 제곱을 빠뜨렸는지 잡는다: var 만 반환하면 위 기대값과 sqrt 만큼 다르다.
    check("5. 분산을 한 번 더 제곱했다",
          not torch.allclose(r, torch.var(ramp, dim=-1), atol=1e-4),
          "var 를 그대로 반환했다 — torch.square 가 빠졌다")

    # 절댓값을 쓰면 안 된다: 부호 있는 파워의 분산이어야 한다.
    signed = ramp.clone(); signed[:, ::2] *= -1.0
    r_signed = fn(FakeEnv(signed, torch.ones_like(signed)), asset_cfg=Cfg())
    r_abs = fn(FakeEnv(signed.abs(), torch.ones_like(signed)), asset_cfg=Cfg())
    check("6. 절댓값을 쓰지 않았다 (부호 있는 파워의 분산)",
          not torch.allclose(r_signed, r_abs, atol=1e-4),
          "부호를 죽여 절댓값 분산을 쟀다")

    r = fn(FakeEnv(ramp, torch.ones_like(ramp)), asset_cfg=Cfg(joint_ids=[0, 1, 2]))
    exp3 = torch.var(ramp[:, :3], dim=-1).square()
    check("7. asset_cfg.joint_ids 로 관절을 선택한다", torch.allclose(r, exp3, atol=1e-5),
          f"{r[0].item():.4f} vs {exp3[0].item():.4f}")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)}개 실패\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32m전부 통과.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
