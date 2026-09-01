#!/usr/bin/env python3
"""빠른 검증 — reward-joint-power.  (Isaac Sim 불필요, ~1초)

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
    fn = load_module(path).joint_power_l1

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
        print("  아직 TODO(reward-joint-power) 가 비어 있다. starter/rewards.py 를 채운다.")
        return 1

    check("1. 반환이 (num_envs,) 스칼라 벡터", tuple(r.shape) == (NUM_ENVS,), f"shape={tuple(r.shape)}")
    check("2. 토크=1, 각속도=1, 관절 12개 → 12", torch.allclose(r, torch.full((NUM_ENVS,), 12.0)),
          f"{r[0].item():.4f}")

    # 토크가 0 이면 파워도 0 (가만히 버티는 것 자체는 벌하지 않는다... 는 아니지만 수식상 0)
    r = fn(FakeEnv(torch.zeros(NUM_ENVS, NUM_JOINTS), ones), asset_cfg=Cfg())
    check("3. 토크 0 이면 0", torch.allclose(r, torch.zeros(NUM_ENVS)), f"{r[0].item():.4f}")

    # 각속도가 0 이면 파워 0 — 정지 상태로 버티는 토크는 이 항으로 벌하지 않는다.
    r = fn(FakeEnv(ones * 5.0, torch.zeros(NUM_ENVS, NUM_JOINTS)), asset_cfg=Cfg())
    check("4. 각속도 0 이면 0 (정지 유지 토크는 무벌)", torch.allclose(r, torch.zeros(NUM_ENVS)),
          f"{r[0].item():.4f}")

    # 부호가 달라도 소비량은 같다 — 절댓값을 썼는지 본다.
    neg_t, neg_v = -ones * 2.0, -ones * 3.0
    r_pp = fn(FakeEnv(ones * 2.0, ones * 3.0), asset_cfg=Cfg())
    r_nn = fn(FakeEnv(neg_t, neg_v), asset_cfg=Cfg())
    r_np = fn(FakeEnv(neg_t, ones * 3.0), asset_cfg=Cfg())
    check("5. 부호와 무관 (절댓값을 썼다)",
          torch.allclose(r_pp, r_nn) and torch.allclose(r_pp, r_np),
          f"(+,+)={r_pp[0]:.2f} (-,-)={r_nn[0]:.2f} (-,+)={r_np[0]:.2f}")

    # joint_ids 를 존중하는가
    r = fn(FakeEnv(ones, ones), asset_cfg=Cfg(joint_ids=[0, 1, 2]))
    check("6. asset_cfg.joint_ids 로 관절을 선택한다", torch.allclose(r, torch.full((NUM_ENVS,), 3.0)),
          f"{r[0].item():.4f} (관절 3개만 골랐으니 3.0 이어야 한다)")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)}개 실패\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32m전부 통과.\033[0m 이어서 reward-power-distribution 으로 간다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
