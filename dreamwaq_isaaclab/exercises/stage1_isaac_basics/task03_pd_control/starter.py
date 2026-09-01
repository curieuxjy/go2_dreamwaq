#!/usr/bin/env python3
"""task03 — PD 제어와 액션 스케일  [L1 · FILL]

task02에서 로봇은 주저앉았다. 관절 목표를 한 번도 주지 않았기 때문이다.
여기서는 목표를 주고, 그 목표가 정책의 액션으로부터 어떻게 계산되는지 본다.

DreamWaQ의 액션 해석 (velocity_env_cfg.py 의 JointPositionActionCfg):

    joint_target = default_joint_pos + action * 0.25
                   └─ use_default_offset=True 가 이 덧셈을 한다
                                            └─ scale=0.25

정책은 "이 관절을 몇 rad 로" 가 아니라 "기본 자세에서 얼마나 벗어날지" 를 낸다.

로봇 두 대를 나란히 세운다. 하나는 이 프로젝트가 쓰는 무른 게인(Kp=20),
하나는 뻣뻣한 게인(Kp=80)이다. 둘 다 action=0, 즉 "기본 자세를 유지하라"는
같은 명령을 받는다. 결과가 왜 다른지가 이 실습의 핵심이다.

    python starter.py
    python starter.py --viz kit    # GUI 로 두 대를 나란히 보기
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="task03 — PD 제어")
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext

from isaaclab_assets import UNITREE_GO2_CFG  # isort:skip

SPAWN_HEIGHT = 0.42
ACTION_SCALE = 0.25  # JointPositionActionCfg(scale=0.25)
SOFT_KP, SOFT_KD = 20.0, 0.5  # DreamWaQ 논문 값 — go2_base_cfg.py 가 쓰는 값
STIFF_KP, STIFF_KD = 80.0, 2.0  # 대조군


def spawn(name: str, y: float, kp: float, kd: float) -> Articulation:
    cfg = UNITREE_GO2_CFG.copy()
    cfg.prim_path = f"/World/{name}"
    cfg.init_state.pos = (0.0, y, SPAWN_HEIGHT)
    cfg.actuators["base_legs"].stiffness = kp
    cfg.actuators["base_legs"].damping = kd
    return Articulation(cfg=cfg)


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1.0 / 200.0))
    sim.set_camera_view(eye=[2.0, 0.0, 1.0], target=[0.0, 0.0, 0.3])

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    soft = spawn("Soft", y=+0.6, kp=SOFT_KP, kd=SOFT_KD)
    stiff = spawn("Stiff", y=-0.6, kp=STIFF_KP, kd=STIFF_KD)

    sim.reset()
    print(f"[INFO] Soft Kp={SOFT_KP} Kd={SOFT_KD}  vs  Stiff Kp={STIFF_KP} Kd={STIFF_KD}")
    print(f"[INFO] 둘 다 action=0 — '기본 자세를 유지하라'\n")

    # ── TODO(pd-target) ─ level L1 ─────────────────────────────
    # action 으로부터 관절 목표를 계산한다
    #   hint: joint_target = default_joint_pos + action * ACTION_SCALE
    #   hint: robot.data.default_joint_pos 는 (num_envs, 12) 모양이다
    #   hint: action 이 0 이면 목표는 기본 자세 그 자체가 된다
    # 통과 기준은 이 실습 폴더의 README.md 를 본다.
    raise NotImplementedError("TODO(pd-target)")
    # ─────────────────────────────────────────────────────────────────────

    zero = torch.zeros_like(soft.data.default_joint_pos)
    target_soft = action_to_target(soft, zero)
    target_stiff = action_to_target(stiff, zero)

    for _ in range(args_cli.steps):
        soft.set_joint_position_target(target_soft)
        stiff.set_joint_position_target(target_stiff)
        soft.write_data_to_sim()
        stiff.write_data_to_sim()
        sim.step()
        soft.update(sim.get_physics_dt())
        stiff.update(sim.get_physics_dt())

    def report(robot: Articulation, target: torch.Tensor, label: str) -> tuple[float, float]:
        h = robot.data.root_pos_w[0, 2].item()
        err = (robot.data.joint_pos - target).abs().max().item()
        print(f"  {label:6s} 높이 {h:.3f} m | 관절 추종 오차 최대 {err:.3f} rad")
        return h, err

    h_soft, e_soft = report(soft, target_soft, "Soft")
    h_stiff, e_stiff = report(stiff, target_stiff, "Stiff")
    print()

    # --- 스스로 검증 ---------------------------------------------------------
    # 액션 ±1 이 관절을 ±0.25 rad 만 움직인다.
    one = torch.ones_like(zero)
    delta = (action_to_target(soft, one) - soft.data.default_joint_pos).abs().max().item()
    assert abs(delta - ACTION_SCALE) < 1e-6, f"액션 스케일이 잘못됐다: {delta:.4f}"
    print(f"  [PASS] action=+1 → 관절 목표가 기본 자세에서 {delta:.2f} rad 만 벗어난다")

    assert abs(action_to_target(soft, zero) - soft.data.default_joint_pos).max() < 1e-9
    print(f"  [PASS] action=0 → 목표 = 기본 자세")

    # 뻣뻣한 쪽이 더 잘 따라간다. 이게 Kp 의 정의다.
    assert e_stiff < e_soft, f"Kp 를 올렸는데 추종이 나빠졌다 ({e_stiff:.3f} >= {e_soft:.3f})"
    print(f"  [PASS] Kp↑ → 추종 오차↓  ({e_soft:.3f} → {e_stiff:.3f} rad)")

    assert h_stiff > h_soft, f"뻣뻣한 쪽이 더 낮다 ({h_stiff:.3f} <= {h_soft:.3f})"
    print(f"  [PASS] Kp↑ → 자세 유지↑  (높이 {h_soft:.3f} → {h_stiff:.3f} m)")

    print(
        "\n[OK] task03 통과.\n"
        f"    Kp={SOFT_KP} 는 자기 무게조차 다 버티지 못해 {e_soft:.2f} rad 처진다.\n"
        "    DreamWaQ 가 이 무른 게인을 쓰는 것은 실수가 아니다 — 실제 Go2 의\n"
        "    관절이 그렇고, 부족한 만큼은 정책이 매 스텝 액션으로 메우도록 학습된다.\n"
        "    (게인을 올리면 sim 에서는 잘 서지만 실기 이전 시 그만큼 어긋난다.)\n"
        "    다음은 task04_terrain 이다."
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
