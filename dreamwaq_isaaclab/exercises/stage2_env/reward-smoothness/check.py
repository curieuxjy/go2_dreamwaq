#!/usr/bin/env python3
"""빠른 검증 — reward-smoothness.

Isaac Sim 없이 돈다. `env` 는 action_manager 만 갖는 최소 스텁으로 대체한다.

    python check.py                 # starter/rewards.py 를 검사
    python check.py --solution      # 완성본을 검사
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

NUM_ENVS, NUM_ACTIONS = 8, 12


class FakeActionManager:
    def __init__(self, num_envs: int, num_actions: int):
        self.action = torch.zeros(num_envs, num_actions)
        self.prev_action = torch.zeros(num_envs, num_actions)


class FakeEnv:
    """rewards.py 가 실제로 건드리는 최소한만 흉내낸다."""

    def __init__(self, num_envs: int = NUM_ENVS, num_actions: int = NUM_ACTIONS):
        self.num_envs = num_envs
        self.device = "cpu"
        self.action_manager = FakeActionManager(num_envs, num_actions)

    def step(self, new_action: torch.Tensor) -> None:
        """한 스텝 진행: t-1 ← t, t ← new_action."""
        am = self.action_manager
        am.prev_action = am.action.clone()
        am.action = new_action.clone()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"검사 대상: {path.relative_to(REPO_ROOT)}\n")
    mod = load_module(path)

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def fresh():
        env = FakeEnv()
        term = mod.action_smoothness_l2(cfg=None, env=env)
        term.prev_prev_action = torch.zeros(NUM_ENVS, NUM_ACTIONS)
        return env, term

    # --- 1. 액션이 계속 0 이면 저크도 0 --------------------------------------
    env, term = fresh()
    try:
        r = term(env)
    except NotImplementedError:
        print("  아직 TODO(reward-smoothness) 가 비어 있다. starter/rewards.py 를 채운다.")
        return 1
    check("1. 반환이 (num_envs,) 스칼라 벡터", tuple(r.shape) == (NUM_ENVS,), f"shape={tuple(r.shape)}")
    check("2. 액션이 0 이면 저크 0", torch.allclose(r, torch.zeros(NUM_ENVS)), f"{r.tolist()[:3]}")

    # --- 3. 등속 변화(선형 램프)면 저크가 0 -----------------------------------
    # a_t = t*c 이면 2차 차분은 정확히 0 이다. "부드러운 움직임은 벌하지 않는다".
    env, term = fresh()
    c = torch.full((NUM_ENVS, NUM_ACTIONS), 0.3)
    for t in range(1, 5):
        env.step(c * t)
        r = term(env)
    check("3. 선형 램프(등속 변화)면 저크 0", torch.allclose(r, torch.zeros(NUM_ENVS), atol=1e-6),
          f"{r[0].item():.6f}")

    # --- 4. 알려진 수열로 값이 정확한가 ---------------------------------------
    # a = 0, 0, 1 → jerk = 1 - 2*0 + 0 = 1 → sum over 12 joints of 1^2 = 12
    env, term = fresh()
    env.step(torch.zeros(NUM_ENVS, NUM_ACTIONS))  # t-1 ← 0, t ← 0
    term(env)  # prev_prev ← 0
    env.step(torch.ones(NUM_ENVS, NUM_ACTIONS))  # t-1 ← 0, t ← 1
    r = term(env)
    check("4. a=(0,0,1) → 저크^2 합 = 12", torch.allclose(r, torch.full((NUM_ENVS,), 12.0)),
          f"{r[0].item():.4f} (기대 12.0)")

    # --- 5. t-2 버퍼가 실제로 갱신되는가 --------------------------------------
    # 갱신을 빠뜨리면 prev_prev 가 계속 0 이라 아래 값이 달라진다.
    # a = 0, 1, 2 → jerk = 2 - 2*1 + 0 = 0  (선형이므로 0)
    env, term = fresh()
    env.step(torch.zeros(NUM_ENVS, NUM_ACTIONS))
    term(env)
    env.step(torch.ones(NUM_ENVS, NUM_ACTIONS))
    term(env)
    env.step(torch.full((NUM_ENVS, NUM_ACTIONS), 2.0))
    r = term(env)
    check("5. t-2 버퍼 갱신됨 (a=0,1,2 → 저크 0)", torch.allclose(r, torch.zeros(NUM_ENVS), atol=1e-6),
          f"{r[0].item():.4f} — 버퍼를 갱신하지 않으면 48 이 나온다")

    # --- 6. 별칭(aliasing) 없이 clone 했는가 ----------------------------------
    env, term = fresh()
    env.step(torch.ones(NUM_ENVS, NUM_ACTIONS))
    term(env)
    before = term.prev_prev_action.clone()
    env.action_manager.prev_action.mul_(99.0)  # 원본을 흔들어 본다
    check("6. prev_prev_action 이 별칭이 아니다 (clone 됨)",
          torch.allclose(term.prev_prev_action, before),
          "prev_action 을 바꿨더니 prev_prev_action 도 같이 바뀌었다 — .clone() 이 빠졌다")

    # --- 7. reset 이 버퍼를 비우는가 ------------------------------------------
    env, term = fresh()
    env.step(torch.ones(NUM_ENVS, NUM_ACTIONS))
    term(env)
    term.reset(env_ids=None)
    check("7. reset() 이 t-2 버퍼를 0 으로 되돌린다",
          torch.allclose(term.prev_prev_action, torch.zeros(NUM_ENVS, NUM_ACTIONS)))

    print()
    if failures:
        print(f"\033[1;31m{len(failures)}개 실패\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32m전부 통과.\033[0m 이어서 reward-foot-clearance 로 간다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
