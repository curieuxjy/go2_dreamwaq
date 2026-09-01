#!/usr/bin/env python3
"""task01 — 빈 씬 띄우기  [L0 · READ]

Isaac Lab 스크립트가 항상 갖는 3단 구조를 확인한다.

    1) AppLauncher 로 Omniverse 앱을 먼저 띄운다
    2) 그 다음에야 isaaclab.* 을 import 할 수 있다
    3) SimulationContext 를 만들고 reset() 한 뒤 step() 을 돈다

이 순서는 바꿀 수 없다. Isaac Sim의 확장(extension)들이 앱이 뜬 뒤에 등록되기
때문에, import 를 위로 올리면 ModuleNotFoundError 가 난다. 이 프로젝트의
train.py / play.py 도 전부 같은 모양이다.

    python solution.py              # 기본이 headless 다
    python solution.py --viz kit    # GUI 창으로 보기

주의: 예전 자료에서 보이는 `--headless` 플래그는 이 버전에서 deprecated 다.
아무것도 안 주면 이미 headless 이고, GUI 를 원하면 `--viz kit` 을 준다.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="task01 — 빈 씬 띄우기")
parser.add_argument("--steps", type=int, default=200, help="시뮬레이션 스텝 수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


def main() -> None:
    # dt=1/200 은 이 프로젝트가 쓰는 물리 스텝이다 (dreamwaq_env_cfg.py 와 동일).
    # 정책은 decimation=4 마다 한 번 동작하므로 제어 주기는 50 Hz 가 된다.
    sim_cfg = SimulationCfg(dt=1.0 / 200.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    # 바닥과 조명. 이 둘이 없으면 로봇이 무한히 떨어지고 화면은 새까맣다.
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # reset() 이 물리 씬을 실제로 만들고 "재생" 상태로 바꾼다.
    # 이 호출 전에는 어떤 asset 의 data 도 읽을 수 없다.
    sim.reset()
    print("[INFO] 씬 준비 완료")

    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        sim.step()
        step += 1

    # --- 스스로 검증 ---------------------------------------------------------
    sim_dt = sim.get_physics_dt()
    assert abs(sim_dt - 1.0 / 200.0) < 1e-9, f"물리 dt 가 1/200 이 아니다: {sim_dt}"
    print(f"  [PASS] 물리 dt = {sim_dt:.6f} s  ({1 / sim_dt:.0f} Hz)")

    assert step == args_cli.steps, f"{args_cli.steps} 스텝을 돌지 못했다: {step}"
    print(f"  [PASS] {step} 스텝 시뮬레이션 완료 = {step * sim_dt:.2f} 초")

    print("\n[OK] task01 통과. 다음은 task02_spawn_go2 다.")


if __name__ == "__main__":
    main()
    simulation_app.close()
