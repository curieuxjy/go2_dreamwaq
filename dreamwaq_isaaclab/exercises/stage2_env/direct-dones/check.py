#!/usr/bin/env python3
"""빠른 검증 — direct-dones.  (Isaac Sim 불필요, ~2초)

`DreamWaQEnv._get_dones` 를 가짜 `self` 로 직접 호출한다.

    python check.py                 # starter/dreamwaq_env.py
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
SRC_DIR = REPO_ROOT / "dreamwaq_direct/source/dreamwaq_direct/dreamwaq_direct/tasks/locomotion"
SOLUTION = SRC_DIR / "dreamwaq_env.py"
STARTER = HERE / "starter" / "dreamwaq_env.py"

N = 4                 # envs
HIST = 3              # contact sensor history_length
BODIES = 5            # 몸통 1 + 발 4 라고 하자
BASE_IDS = [0]        # _termination_contact_ids — 몸통만
MAX_EP = 1000
THRESHOLD = 1.0


class Cfg:
    termination_contact_force = THRESHOLD


class FakeSelf:
    """_get_dones 가 만지는 것만 갖춘 최소 객체."""

    def __init__(self, forces: torch.Tensor, ep_len: torch.Tensor):
        data = type("D", (), {"net_forces_w_history": type("W", (), {"torch": forces})()})()
        self._contact_sensor = type("CS", (), {"data": data})()
        self._termination_contact_ids = BASE_IDS
        self.cfg = Cfg()
        self.episode_length_buf = ep_len
        self.max_episode_length = MAX_EP
        self._last_episode_dones = torch.zeros(N, 2, dtype=torch.bool)


def make(force_at=None, ep_len=None):
    """force_at: {(env, hist, body): 힘크기} — 지정한 곳에만 힘을 준다."""
    forces = torch.zeros(N, HIST, BODIES, 3)
    for (e, h, b), mag in (force_at or {}).items():
        forces[e, h, b, 2] = mag  # z 방향 힘
    ep = torch.zeros(N, dtype=torch.long) if ep_len is None else ep_len
    return FakeSelf(forces, ep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true", help="완성본을 검사한다")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"검사 대상: {path.relative_to(REPO_ROOT)}\n")
    mod = load_module(path, extra_paths=(str(SRC_DIR),))
    get_dones = mod.DreamWaQEnv._get_dones

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    # --- 1. 아무 일도 없으면 둘 다 False ------------------------------------------
    try:
        contact, timeout = get_dones(make())
    except NotImplementedError:
        print("  아직 TODO(direct-dones) 가 비어 있다. starter/dreamwaq_env.py 를 채운다.")
        return 1

    check("1. 반환 shape 이 (num_envs,) 두 개다",
          tuple(contact.shape) == (N,) and tuple(timeout.shape) == (N,),
          f"contact {tuple(contact.shape)}, timeout {tuple(timeout.shape)}")
    check("1. 힘도 없고 시간도 안 됐으면 둘 다 False",
          not contact.any() and not timeout.any())

    # --- 2. 시간 초과 ---------------------------------------------------------------
    ep = torch.tensor([0, MAX_EP - 2, MAX_EP - 1, MAX_EP])
    _, timeout = get_dones(make(ep_len=ep))
    check("2. episode_length >= max-1 인 env 만 timeout",
          timeout.tolist() == [False, False, True, True], f"{timeout.tolist()}")

    # --- 3. 몸통 접촉 ---------------------------------------------------------------
    contact, _ = get_dones(make({(1, 0, 0): 5.0}))
    check("3. 몸통에 임계값 초과 힘이 걸린 env 만 종료",
          contact.tolist() == [False, True, False, False], f"{contact.tolist()}")

    # --- 4. 임계값 경계 --------------------------------------------------------------
    # 원본은 > 1.0 이다. 정확히 1.0 은 종료가 아니다.
    contact, _ = get_dones(make({(0, 0, 0): THRESHOLD}))
    check("4. 정확히 임계값이면 종료가 아니다 (> 이지 >= 가 아니다)",
          not contact[0], f"{contact.tolist()}")
    contact, _ = get_dones(make({(0, 0, 0): THRESHOLD + 0.01}))
    check("4. 임계값을 조금이라도 넘으면 종료", bool(contact[0]))

    # --- 5. history 는 최대값으로 본다 -------------------------------------------------
    # 3 프레임 중 한 프레임에서만 세게 부딪힌 경우. 평균을 쓰면 놓친다.
    contact, _ = get_dones(make({(2, 1, 0): 9.0}))
    check("5. history 3 중 한 프레임만 부딪혀도 종료 (max, 평균이 아니다)",
          bool(contact[2]), f"{contact.tolist()}")

    # --- 6. 몸통이 아닌 바디는 무시한다 --------------------------------------------------
    # 발(body 1~4)에 큰 힘이 걸리는 것은 정상 보행이다. 이걸로 종료하면 걷지를 못한다.
    contact, _ = get_dones(make({(0, 0, 1): 500.0, (1, 2, 4): 500.0}))
    check("6. 발 접촉은 종료가 아니다 (_termination_contact_ids 로 골라낸다)",
          not contact.any(), f"{contact.tolist()}")

    # --- 7. 힘은 벡터 크기로 잰다 ---------------------------------------------------------
    # 성분별이 아니라 norm 이어야 한다. (0.8, 0.8, 0.8) 은 각 성분은 1 미만이지만 norm 은 1.39 다.
    forces = torch.zeros(N, HIST, BODIES, 3)
    forces[3, 0, 0] = torch.tensor([0.8, 0.8, 0.8])
    contact, _ = get_dones(FakeSelf(forces, torch.zeros(N, dtype=torch.long)))
    check("7. 힘은 성분이 아니라 벡터 norm 으로 잰다",
          bool(contact[3]), f"norm={float(torch.linalg.norm(forces[3, 0, 0])):.3f} 인데 {contact.tolist()}")

    # --- 8. 둘은 독립이다 -------------------------------------------------------------
    ep = torch.tensor([MAX_EP, 0, 0, 0])
    contact, timeout = get_dones(make({(1, 0, 0): 5.0}, ep_len=ep))
    check("8. timeout 과 base_contact 는 서로 영향을 주지 않는다",
          timeout.tolist() == [True, False, False, False]
          and contact.tolist() == [False, True, False, False],
          f"timeout {timeout.tolist()}, contact {contact.tolist()}")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)}개 실패\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32m전부 통과.\033[0m Stage 2 실습을 모두 마쳤다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
