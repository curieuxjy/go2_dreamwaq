#!/usr/bin/env python3
"""task05 — 센서: 접촉과 height scan  [L2 · BUILD]

Stage 1의 마지막이자 가장 중요한 실습이다. 여기서 읽는 두 값이
Stage 2에서 만들 **종료 조건**과 **특권 관측**의 원재료다.

    ContactSensor  → 몸통이 지면에 닿았는가 → 종료 조건
    RayCaster      → 로봇 주변 지형의 높이 → Oracle 의 특권 관측 187차원

height scan은 단순히 광선이 맞은 지점의 z 가 아니다. **몸통 높이를 기준으로 한
상대 높이**여야 로봇이 어디에 있든 같은 의미를 갖는다. 그 변환을 직접 만든다.

    python starter.py
    python starter.py --viz kit
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="task05 — 센서")
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, SimulationContext

from isaaclab_assets import UNITREE_GO2_CFG  # isort:skip

SPAWN_HEIGHT = 0.42
# 프로젝트와 동일한 스캔 격자: 0.1 m 간격, 1.6 x 1.0 m → 17 x 11 = 187 광선
SCAN_RESOLUTION, SCAN_SIZE = 0.1, [1.6, 1.0]
RAY_START_HEIGHT = 20.0  # 광선을 로봇 위 20 m 에서 아래로 쏜다
BASE_HEIGHT_TARGET = 0.30  # DreamWaQ 의 목표 몸통 높이


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1.0 / 200.0))
    sim.set_camera_view(eye=[2.0, 2.0, 1.2], target=[0.0, 0.0, 0.3])

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = UNITREE_GO2_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.init_state.pos = (0.0, 0.0, SPAWN_HEIGHT)
    robot_cfg.actuators["base_legs"].stiffness = 80.0  # 서 있어야 스캔이 의미 있다
    robot_cfg.actuators["base_legs"].damping = 2.0
    robot = Articulation(cfg=robot_cfg)

    # --- 접촉 센서: 모든 바디를 감시한다 ----------------------------------------
    contact = ContactSensor(
        ContactSensorCfg(prim_path="/World/Robot/.*", history_length=3, track_air_time=True)
    )

    # --- 광선 센서: 몸통에 붙어 아래를 훑는다 ------------------------------------
    scanner = RayCaster(
        RayCasterCfg(
            prim_path="/World/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, RAY_START_HEIGHT)),
            ray_alignment="yaw",  # 로봇이 회전해도 격자는 몸통 yaw 를 따라간다
            pattern_cfg=patterns.GridPatternCfg(resolution=SCAN_RESOLUTION, size=SCAN_SIZE),
            mesh_prim_paths=["/World/defaultGroundPlane"],
            debug_vis=False,
        )
    )

    sim.reset()
    default_q = robot.data.default_joint_pos.clone()
    print(f"[INFO] 광선 {scanner.num_rays}개 ({SCAN_SIZE[0]}m x {SCAN_SIZE[1]}m, {SCAN_RESOLUTION}m 간격)\n")

    dt = sim.get_physics_dt()
    for _ in range(args_cli.steps):
        robot.set_joint_position_target(default_q)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        contact.update(dt)
        scanner.update(dt)

    # ── 1. 접촉: 어느 바디가 지면에 닿아 있는가 ────────────────────────────────
    # net_forces_w_history: (num_envs, history_length, num_sensors, 3)
    forces = contact.data.net_forces_w_history.torch
    force_mag = torch.linalg.norm(forces, dim=-1).max(dim=1)[0]  # history 최대 → (envs, sensors)

    base_idx, _ = contact.find_sensors("base")
    feet_idx, feet_names = contact.find_sensors(".*foot")
    base_force = force_mag[0, base_idx].max().item()
    feet_force = force_mag[0, feet_idx]

    print("  [접촉]")
    print(f"    base 접촉력      {base_force:8.2f} N   ← 종료 조건이 이 값을 본다 (> 1 N 이면 종료)")
    for name, f in zip(feet_names, feet_force.tolist()):
        print(f"    {name:10s} 접촉력 {f:8.2f} N")

    # ── 2. height scan: 몸통 기준 상대 높이 ────────────────────────────────────
    ray_hits_z = scanner.data.ray_hits_w[..., 2]  # (num_envs, num_rays) — 월드 좌표 z
    base_z = robot.data.root_pos_w[:, 2]  # (num_envs,)

    # ── TODO(height-scan) ─ level L2 ─────────────────────────────
    # 광선이 맞은 월드 z 를 '몸통 기준 상대 높이'로 바꾼다
    #   hint: 관측은 "내 몸통이 목표 높이보다 얼마나 높은가"를 알려주어야 한다
    #   hint: 프로젝트의 실제 식은 base_z - hit_z - BASE_HEIGHT_TARGET 이다 (mdp.height_scan)
    #   hint: base_z 는 (num_envs,) 이고 hit_z 는 (num_envs, num_rays) 다. 브로드캐스트를 맞춘다
    #   hint: 평지에서 몸통이 정확히 목표 높이면 결과가 0 이 되어야 한다
    # 통과 기준은 이 실습 폴더의 README.md 를 본다.
    raise NotImplementedError("TODO(height-scan)")
    # ─────────────────────────────────────────────────────────────────────

    print("\n  [height scan]")
    print(f"    광선 수        {height_scan.shape[1]}")
    print(f"    몸통 높이      {base_z[0].item():.3f} m")
    print(f"    맞은 지점 z    평균 {ray_hits_z[0].mean().item():+.3f} m (평지이므로 거의 0)")
    print(f"    상대 높이      평균 {height_scan[0].mean().item():+.3f} m, "
          f"범위 [{height_scan[0].min().item():+.3f}, {height_scan[0].max().item():+.3f}]")

    # --- 스스로 검증 ---------------------------------------------------------
    print()
    assert scanner.num_rays == 187, f"광선이 187개여야 한다 (17 x 11): {scanner.num_rays}"
    print(f"  [PASS] 광선 187개 — Oracle 의 특권 관측 187차원이 바로 이것이다")

    assert feet_force.sum().item() > 1.0, f"발이 지면을 안 딛고 있다: {feet_force.tolist()}"
    print(f"  [PASS] 발이 지면을 딛고 있다 (합 {feet_force.sum().item():.1f} N)")

    assert base_force < 1.0, f"몸통이 지면에 닿아 있다 ({base_force:.2f} N) — 종료 조건이 걸릴 상태다"
    print(f"  [PASS] 몸통은 지면에 안 닿았다 ({base_force:.2f} N < 1 N)")

    # 평지이므로 맞은 지점의 z 는 전부 0 근처여야 한다.
    assert ray_hits_z[0].abs().max().item() < 0.05, "평지인데 맞은 지점 z 가 흩어져 있다"
    print(f"  [PASS] 평지에서 맞은 지점 z 가 모두 0 근처")

    # 핵심: 상대 높이 = 몸통높이 - 지형높이 - 목표높이.
    expected = base_z[0].item() - 0.0 - BASE_HEIGHT_TARGET
    assert abs(height_scan[0].mean().item() - expected) < 0.02, (
        f"상대 높이가 기대와 다르다: {height_scan[0].mean().item():+.3f} vs {expected:+.3f}"
    )
    print(f"  [PASS] 상대 높이 = 몸통높이 - 지형높이 - {BASE_HEIGHT_TARGET}  ({expected:+.3f} m)")

    print(
        "\n[OK] task05 통과 — Stage 1 완료.\n"
        "    이제 관측의 재료가 전부 손에 있다:\n"
        "      몸통 속도/자세, 관절 각도/각속도 (task02)\n"
        "      액션 → 관절 목표          (task03)\n"
        "      지형과 env_origins        (task04)\n"
        "      접촉력, height scan       (task05)\n"
        "    Stage 2 에서 이것들을 45/48/64 차원 관측으로 조립한다."
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
