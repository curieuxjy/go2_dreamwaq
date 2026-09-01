#!/usr/bin/env python3
"""task02 — Go2 스폰하고 상태 읽기  [L1 · FILL]

로봇 하나를 씬에 올리고, 시뮬레이션이 도는 동안 그 상태를 읽는다.
Stage 2에서 만들 관측(observation)은 전부 여기서 읽는 값들의 조합이다.

    python starter.py
    python starter.py --viz kit    # GUI 로 보기
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="task02 — Go2 스폰")
parser.add_argument("--steps", type=int, default=200)
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

# DreamWaQ 의 스폰 높이. 논문은 0.34 m 지만 이 프로젝트는 0.42 m 를 쓴다 —
# 생성 지형의 boxes(최대 0.1 m)를 xy ±0.5 m 리셋 랜덤화와 함께 넘기 위해서다.
SPAWN_HEIGHT = 0.42


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1.0 / 200.0))
    sim.set_camera_view(eye=[2.0, 2.0, 1.0], target=[0.0, 0.0, 0.3])

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # ── TODO(spawn-go2) ─ level L1 ─────────────────────────────
    # UNITREE_GO2_CFG 를 복사해 prim 경로와 스폰 높이를 지정하고 Articulation 을 만든다
    #   hint: cfg 는 반드시 .copy() 로 복사한다. 원본을 고치면 같은 프로세스의 다른 코드에 영향이 간다
    #   hint: prim_path 는 "/World/Robot" 으로 한다
    #   hint: init_state.pos 는 (x, y, z) 튜플이다. z 에 SPAWN_HEIGHT 를 넣는다
    #   hint: Articulation(cfg=...) 로 만든다
    # 통과 기준은 이 실습 폴더의 README.md 를 본다.
    raise NotImplementedError("TODO(spawn-go2)")
    # ─────────────────────────────────────────────────────────────────────

    # reset() 전에는 robot.data 를 읽을 수 없다. 물리 씬이 아직 없기 때문이다.
    sim.reset()
    print("[INFO] Go2 스폰 완료\n")

    # --- 로봇이 무엇으로 이루어졌는가 -----------------------------------------
    print(f"  관절 {robot.num_joints}개: {robot.joint_names}")
    print(f"  바디 {robot.num_bodies}개: {robot.body_names}\n")

    # 기본 자세. 정책의 액션은 이 값으로부터의 '오프셋'으로 해석된다.
    default_q = robot.data.default_joint_pos[0]
    print(f"  기본 관절 자세 (rad): {[f'{v:.2f}' for v in default_q.tolist()]}\n")

    # --- 시뮬레이션을 돌리며 상태를 읽는다 -------------------------------------
    # 액션을 주지 않으므로 로봇은 기본 자세를 유지하려다 중력에 눌려 주저앉는다.
    for step in range(args_cli.steps):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

        if step % 50 == 0:
            pos = robot.data.root_pos_w[0]
            lin = robot.data.root_lin_vel_b[0]
            grav = robot.data.projected_gravity_b[0]
            print(
                f"  step {step:3d} | 높이 {pos[2]:.3f} m"
                f" | 몸통기준 속도 ({lin[0]:+.2f}, {lin[1]:+.2f}, {lin[2]:+.2f})"
                f" | 중력투영 z {grav[2]:+.3f}"
            )

    # --- 스스로 검증 ---------------------------------------------------------
    print()
    assert robot.num_joints == 12, f"Go2 는 관절이 12개여야 한다: {robot.num_joints}"
    print(f"  [PASS] 관절 12개 (다리 4개 × hip/thigh/calf 3)")

    assert "base" in robot.body_names, f"'base' 바디가 없다: {robot.body_names}"
    print(f"  [PASS] 'base' 바디 존재 — 종료 조건에서 이 바디의 접촉을 본다")

    feet = [b for b in robot.body_names if b.endswith("foot")]
    assert len(feet) == 4, f"발이 4개여야 한다: {feet}"
    print(f"  [PASS] 발 4개: {feet}")

    # 지면 아래로 꺼지지는 않아야 한다 (꺼졌다면 스폰 높이나 지면 설정이 잘못된 것).
    height = robot.data.root_pos_w[0, 2].item()
    assert 0.0 < height < SPAWN_HEIGHT + 0.1, f"로봇 높이가 이상하다: {height:.3f} m"
    print(f"  [PASS] 최종 높이 {height:.3f} m — 지면 위에 있다")

    # 몸통 기준 중력 z: 똑바로 서 있으면 -1, 완전히 옆으로 누우면 0 에 가깝다.
    # Stage 2 의 관측 45차원 중 3개가 바로 이 projected_gravity_b 다.
    gz = robot.data.projected_gravity_b[0, 2].item()
    assert -1.05 <= gz <= 0.05, f"중력투영 z 가 범위를 벗어났다: {gz:.3f}"
    print(f"  [PASS] 중력투영 z = {gz:.3f}  (똑바로=-1, 옆으로 누움=0)")

    print(
        "\n[OK] task02 통과.\n"
        f"    관절 목표를 한 번도 주지 않았으므로 로봇은 중력에 눌려 주저앉았다\n"
        f"    (높이 {SPAWN_HEIGHT:.2f} → {height:.3f} m, 중력투영 z {gz:+.3f}).\n"
        "    이걸 서 있게 만드는 것이 다음 task03_pd_control 이다."
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
