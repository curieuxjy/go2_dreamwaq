#!/usr/bin/env python3
"""빠른 검증 — direct-obs-scale.  (Isaac Sim 불필요, ~2초)

`DreamWaQEnv._get_observations` 를 가짜 `self` 로 직접 호출한다. 실제 env 를
띄우지 않고 메서드 하나만 떼어 검사하는 방식이다.

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
SOLUTION = (
    REPO_ROOT
    / "dreamwaq_direct/source/dreamwaq_direct/dreamwaq_direct/tasks/locomotion/dreamwaq_env.py"
)
STARTER = HERE / "starter" / "dreamwaq_env.py"

N, J = 4, 12
# 프로젝트 기본값 (dreamwaq_env_cfg.py)
S_LIN, S_ANG, S_POS, S_VEL = 2.0, 0.25, 1.0, 0.05


NUM_RAYS = 187


class Cfg:
    obs_scale_height = 5.0
    state_space = 235
    obs_scale_lin_vel = S_LIN
    obs_scale_ang_vel = S_ANG
    obs_scale_dof_pos = S_POS
    obs_scale_dof_vel = S_VEL
    obs_noise = False       # 노이즈를 끄면 결정론적으로 검사할 수 있다
    system_delay = False
    include_lin_vel_in_obs = False


class FakeSelf:
    """_get_observations 가 만지는 것만 갖춘 최소 객체."""

    def __init__(self, lin, ang, grav, jpos, jvel, jdef, cmd, act):
        data = type("D", (), {
            "root_lin_vel_b": lin, "root_ang_vel_b": ang, "projected_gravity_b": grav,
            "joint_pos": jpos, "joint_vel": jvel, "default_joint_pos": jdef,
        })()
        self._robot = type("R", (), {"data": data})()
        self.cfg = Cfg()
        self._commands = cmd
        self._actions = act
        self._true_lin_vel_b = torch.zeros_like(lin)
        self._obs_history_buf = torch.zeros(N, 5, 45)
        # critic(privileged) 관측용 — height scanner 와 push 임펄스
        hs = type("HD", (), {
            "pos_w": torch.zeros(N, 3),
            "ray_hits_w": torch.zeros(N, NUM_RAYS, 3),
        })()
        self._height_scanner = type("HS", (), {"data": hs})()
        self._disturb_force = torch.zeros(N, 3)
        self._command_vis = None      # GUI 마커 (headless 검사에서는 없음)
        self._heading_vis = None
        self.num_envs, self.device = N, "cpu"


def make(lin_v=0.0, ang_v=0.0, jvel_v=0.0, jpos_v=0.0):
    return FakeSelf(
        lin=torch.full((N, 3), lin_v),
        ang=torch.full((N, 3), ang_v),
        grav=torch.zeros(N, 3),
        jpos=torch.full((N, J), jpos_v),
        jvel=torch.full((N, J), jvel_v),
        jdef=torch.zeros(N, J),
        cmd=torch.zeros(N, 4),
        act=torch.zeros(N, J),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"검사 대상: {path.relative_to(REPO_ROOT)}\n")
    get_obs = load_module(path, extra_paths=(SOLUTION.parent,)).DreamWaQEnv._get_observations

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def policy_of(obj) -> torch.Tensor:
        out = get_obs(obj)
        return out["policy"] if isinstance(out, dict) else out

    try:
        obs = policy_of(make())
    except NotImplementedError:
        print("  아직 TODO(direct-obs-scale) 가 비어 있다. starter/dreamwaq_env.py 를 채운다.")
        return 1

    check("1. 관측이 45차원 (Base/Waq)", obs.shape == (N, 45), f"shape={tuple(obs.shape)}")

    # 관측 배치: ang_vel(3) gravity(3) commands(3) joint_pos(12) joint_vel(12) actions(12)
    ANG, GRAV, CMD, JPOS, JVEL = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 21), slice(21, 33)

    obs = policy_of(make(ang_v=4.0))
    check("2. ang_vel 에 obs_scale_ang_vel(0.25) 적용",
          torch.allclose(obs[:, ANG], torch.full((N, 3), 4.0 * S_ANG), atol=1e-6),
          f"{obs[0, ANG].tolist()} (기대 {4.0 * S_ANG})")

    obs = policy_of(make(jvel_v=20.0))
    check("3. joint_vel 에 obs_scale_dof_vel(0.05) 적용",
          torch.allclose(obs[:, JVEL], torch.full((N, J), 20.0 * S_VEL), atol=1e-6),
          f"{obs[0, JVEL][0].item():.4f} (기대 {20.0 * S_VEL})")

    obs = policy_of(make(jpos_v=0.5))
    check("4. joint_pos 에 obs_scale_dof_pos(1.0) 적용",
          torch.allclose(obs[:, JPOS], torch.full((N, J), 0.5 * S_POS), atol=1e-6),
          f"{obs[0, JPOS][0].item():.4f}")

    # gravity 는 스케일하지 않는다 (원본도 1.0)
    s = make()
    s._robot.data.projected_gravity_b = torch.full((N, 3), -1.0)
    obs = policy_of(s)
    check("5. gravity 는 스케일하지 않는다", torch.allclose(obs[:, GRAV], torch.full((N, 3), -1.0), atol=1e-6),
          f"{obs[0, GRAV].tolist()}")

    # true_lin_vel 은 스케일 '전' 값이어야 한다 — CENet 의 학습 타깃이므로 중요하다
    s = make(lin_v=3.0)
    policy_of(s)
    check("6. _true_lin_vel_b 는 스케일 전 원값을 보관한다",
          torch.allclose(s._true_lin_vel_b, torch.full((N, 3), 3.0), atol=1e-6),
          f"{s._true_lin_vel_b[0].tolist()} — 스케일 후 값을 저장하면 CENet 타깃이 틀어진다")

    # --- 핵심: 스케일링이 없으면 무슨 일이 벌어지는가 ---
    # 실제 로봇 규모: joint_vel ~ ±20 rad/s, 명령 ~ ±1
    s = make(jvel_v=20.0)
    s._commands = torch.ones(N, 4)
    obs = policy_of(s)
    jvel_mag = obs[:, JVEL].abs().mean().item()
    cmd_mag = obs[:, CMD].abs().mean().item()
    check("7. 스케일 후 joint_vel 이 명령을 압도하지 않는다 (비 < 3)",
          jvel_mag / max(cmd_mag, 1e-9) < 3.0,
          f"joint_vel/명령 = {jvel_mag / max(cmd_mag, 1e-9):.1f}배 — 스케일을 빠뜨리면 20배가 된다")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)}개 실패\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32m전부 통과.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
