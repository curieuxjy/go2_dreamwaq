#!/usr/bin/env python3
"""task04 — 지형 생성과 env_origins  [L1 · FILL]

지금까지는 무한 평면 위에서 놀았다. DreamWaQ의 "implicit terrain imagination"은
울퉁불퉁한 지형이 있어야 의미가 생긴다.

Isaac Lab의 지형은 두 가지다.

    terrain_type="plane"      단일 무한 평면. 가볍고 디버깅에 좋다
    terrain_type="generator"  sub-terrain 타일을 num_rows × num_cols 격자로 생성

generator 격자에서 **행(row)이 난이도**다. 지형 커리큘럼은 로봇이 잘하면 다음
행으로 올려보내고 못하면 내린다 (`mdp.terrain_levels_vel`).

    python solution.py
    python solution.py --viz kit    # GUI 로 지형 보기
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="task04 — 지형")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporterCfg

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1.0 / 200.0))
    sim.set_camera_view(eye=[14.0, 14.0, 10.0], target=[0.0, 0.0, 0.0])

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # --- ROUGH_TERRAINS_CFG 안에 무엇이 들어 있는가 -----------------------------
    print("[INFO] ROUGH_TERRAINS_CFG 의 sub-terrain:")
    for name, sub in ROUGH_TERRAINS_CFG.sub_terrains.items():
        print(f"    {name:22s} 비율 {sub.proportion:.2f}")
    print()

    # ex:begin id=terrain-cfg level=1 stage=stage1_isaac_basics task=생성 지형 TerrainImporterCfg 를 만든다
    #   hint: prim_path 는 "/World/ground" 로 한다. height scanner 가 이 경로를 mesh 로 참조한다
    #   hint: terrain_type 은 "generator", terrain_generator 는 ROUGH_TERRAINS_CFG 의 .copy() 를 쓴다
    #   hint: collision_group=-1 로 두어야 모든 env 가 이 지형과 충돌한다 (전역 그룹)
    #   hint: 격자 크기는 terrain_generator.num_rows / num_cols 로 정한다. 여기서는 5 x 5
    terrain_gen = ROUGH_TERRAINS_CFG.copy()
    terrain_gen.num_rows = 5
    terrain_gen.num_cols = 5
    terrain_gen.curriculum = True

    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen,
        collision_group=-1,
        max_init_terrain_level=4,
        debug_vis=False,
    )
    # ex:end

    # TerrainImporter 는 env 개수만큼 원점(env_origins)을 지형 위에 배치한다.
    terrain_cfg.num_envs = args_cli.num_envs
    terrain_cfg.env_spacing = 3.0
    terrain = terrain_cfg.class_type(terrain_cfg)

    sim.reset()
    print(f"[INFO] 지형 생성 완료 ({terrain_gen.num_rows} x {terrain_gen.num_cols} 격자)\n")

    origins = terrain.env_origins  # (num_envs, 3)
    print(f"  env_origins shape: {tuple(origins.shape)}")
    print(f"  x 범위 [{origins[:, 0].min():+.2f}, {origins[:, 0].max():+.2f}] m")
    print(f"  y 범위 [{origins[:, 1].min():+.2f}, {origins[:, 1].max():+.2f}] m")
    print(f"  z 범위 [{origins[:, 2].min():+.2f}, {origins[:, 2].max():+.2f}] m  ← 지형 높이차\n")

    # --- 스스로 검증 ---------------------------------------------------------
    assert tuple(origins.shape) == (args_cli.num_envs, 3), f"env_origins 모양이 이상하다: {origins.shape}"
    print(f"  [PASS] env 마다 원점이 하나씩 있다 ({args_cli.num_envs}개)")

    # 평면이라면 모든 원점의 z 가 0 이다. 생성 지형이면 타일마다 높이가 다르다.
    assert origins[:, :2].abs().max() > 1.0, "원점들이 한 점에 몰려 있다 — 지형 격자가 안 만들어졌다"
    print(f"  [PASS] 원점이 지형 격자에 흩어져 있다")

    # 로봇 스폰 높이 0.42 m 의 근거: 지형의 boxes 가 최대 0.1 m 다.
    n_sub = len(ROUGH_TERRAINS_CFG.sub_terrains)
    assert n_sub >= 4, f"sub-terrain 이 너무 적다: {n_sub}"
    print(f"  [PASS] sub-terrain {n_sub}종 — 평지/경사/계단/블록 등이 섞여 있다")

    # 커리큘럼의 토대: 행이 난이도다.
    assert terrain_gen.curriculum, "curriculum=True 여야 행이 난이도로 정렬된다"
    print(f"  [PASS] curriculum=True — 행(row)이 난이도 단계가 된다")

    print(
        "\n[OK] task04 통과.\n"
        f"    {terrain_gen.num_rows}개 행이 난이도 단계다. 학습 중 mdp.terrain_levels_vel 이\n"
        "    잘 걷는 로봇을 윗행으로, 못 걷는 로봇을 아랫행으로 옮긴다.\n"
        "    학습 로그의 Curriculum/terrain_levels 가 바로 이 행 번호의 평균이다.\n"
        "    다음은 task05_sensors — 마지막이자 가장 중요한 실습이다."
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
