#!/usr/bin/env python3
"""빠른 검증 — reward-foot-clearance.

Isaac Sim 없이 돈다. env / asset / height sensor 를 최소 스텁으로 대체한다.

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

NUM_ENVS, NUM_FEET, NUM_RAYS = 4, 4, 187
DESIRED = 0.12


class Cfg:
    def __init__(self, name: str, body_ids=None):
        self.name = name
        self.body_ids = body_ids


class FakeAsset:
    def __init__(self, feet_pos_z: torch.Tensor, feet_vel_xy: torch.Tensor):
        n, f = feet_pos_z.shape
        self.data = type("D", (), {})()
        pos = torch.zeros(n, f, 3)
        pos[:, :, 2] = feet_pos_z
        vel = torch.zeros(n, f, 3)
        vel[:, :, 0] = feet_vel_xy[..., 0]
        vel[:, :, 1] = feet_vel_xy[..., 1]
        vel[:, :, 2] = 99.0  # z 속도를 섞어 넣는다 — 답에 반영되면 안 된다
        self.data.body_pos_w = pos
        self.data.body_lin_vel_w = vel


class FakeSensor:
    def __init__(self, terrain_z: torch.Tensor, with_nan: bool = False):
        hits = torch.zeros(NUM_ENVS, NUM_RAYS, 3)
        hits[:, :, 2] = terrain_z[:, None]
        if with_nan:
            hits[:, 0, 2] = float("nan")
            hits[:, 1, 2] = float("inf")
        self.data = type("D", (), {"ray_hits_w": hits})()


class FakeScene:
    def __init__(self, asset, sensor):
        self._asset, self.sensors = asset, {"height_scanner": sensor}

    def __getitem__(self, _name):
        return self._asset


class FakeEnv:
    def __init__(self, asset, sensor):
        self.scene = FakeScene(asset, sensor)
        self.num_envs, self.device = NUM_ENVS, "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"검사 대상: {path.relative_to(REPO_ROOT)}\n")
    mod = load_module(path)
    fn = mod.foot_clearance_l2

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def call(feet_z, speed, terrain_z=None, use_sensor=True, with_nan=False):
        terrain = torch.zeros(NUM_ENVS) if terrain_z is None else terrain_z
        vel = torch.zeros(NUM_ENVS, NUM_FEET, 2)
        vel[..., 0] = speed  # vx 만 준다 → 측면속도 = |vx|
        env = FakeEnv(FakeAsset(feet_z, vel), FakeSensor(terrain, with_nan))
        return fn(
            env,
            asset_cfg=Cfg("robot", body_ids=list(range(NUM_FEET))),
            height_sensor_cfg=Cfg("height_scanner") if use_sensor else None,
            desired_height=DESIRED,
        )

    flat = torch.full((NUM_ENVS, NUM_FEET), DESIRED)  # 정확히 목표 높이
    try:
        r = call(flat, speed=1.0)
    except NotImplementedError:
        print("  아직 TODO(reward-foot-clearance) 가 비어 있다. starter/rewards.py 를 채운다.")
        return 1

    check("1. 반환이 (num_envs,) 스칼라 벡터", tuple(r.shape) == (NUM_ENVS,), f"shape={tuple(r.shape)}")
    check("2. 발이 정확히 목표 높이면 페널티 0", torch.allclose(r, torch.zeros(NUM_ENVS), atol=1e-6),
          f"{r[0].item():.6f}")

    # 발이 멈춰 있으면(측면속도 0) 높이가 틀려도 벌하지 않는다 — 딛고 선 발은 봐준다.
    r = call(torch.zeros(NUM_ENVS, NUM_FEET), speed=0.0)
    check("3. 측면속도 0 이면 페널티 0 (딛고 선 발)", torch.allclose(r, torch.zeros(NUM_ENVS), atol=1e-6),
          f"{r[0].item():.6f}")

    # 알려진 값: 발 높이 0, 목표 0.12, 속도 1.0, 발 4개 → 4 * 0.12^2 * 1 = 0.0576
    r = call(torch.zeros(NUM_ENVS, NUM_FEET), speed=1.0)
    check("4. 높이 0 / 속도 1 / 발 4개 → 4 x 0.12^2 = 0.0576",
          torch.allclose(r, torch.full((NUM_ENVS,), 4 * DESIRED**2), atol=1e-6),
          f"{r[0].item():.6f} (기대 {4 * DESIRED**2:.4f})")

    # 속도에 선형 비례한다.
    r1 = call(torch.zeros(NUM_ENVS, NUM_FEET), speed=1.0)
    r2 = call(torch.zeros(NUM_ENVS, NUM_FEET), speed=2.0)
    check("5. 페널티가 측면속도에 선형 비례", torch.allclose(r2, 2 * r1, atol=1e-6),
          f"{r1[0].item():.4f} → {r2[0].item():.4f}")

    # 핵심: 지형이 올라가면 '지형 위 높이'로 보정되어야 한다.
    # 발 월드z = 0.5, 지형 = 0.38 → 지형 위 0.12 = 목표 → 페널티 0.
    r = call(torch.full((NUM_ENVS, NUM_FEET), 0.5), speed=1.0,
             terrain_z=torch.full((NUM_ENVS,), 0.5 - DESIRED))
    check("6. 지형 높이를 빼서 '지형 위 높이'로 본다",
          torch.allclose(r, torch.zeros(NUM_ENVS), atol=1e-5),
          f"{r[0].item():.6f} — 월드 z 를 그대로 쓰면 0 이 안 나온다")

    # NaN/inf 가 섞인 ray hit 을 그대로 쓰면 결과가 오염된다.
    r = call(flat, speed=1.0, with_nan=True)
    check("7. ray hit 의 NaN/inf 를 0 으로 정리한다", bool(torch.isfinite(r).all()),
          f"{r.tolist()} — torch.nan_to_num 이 빠졌다")

    # z 속도는 무시해야 한다 (스텁이 vz=99 를 넣어 두었다).
    r = call(torch.zeros(NUM_ENVS, NUM_FEET), speed=1.0)
    check("8. 측면속도에 z 성분을 넣지 않았다",
          torch.allclose(r, torch.full((NUM_ENVS,), 4 * DESIRED**2), atol=1e-6),
          f"{r[0].item():.4f} — vz=99 가 새어 들어왔다")

    # height_sensor_cfg=None 이면 지형 보정 없이 월드 z 를 쓴다 (평지 전용 폴백).
    r = call(torch.zeros(NUM_ENVS, NUM_FEET), speed=1.0, use_sensor=False)
    check("9. 센서가 None 이면 월드 z 폴백",
          torch.allclose(r, torch.full((NUM_ENVS,), 4 * DESIRED**2), atol=1e-6),
          f"{r[0].item():.4f}")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)}개 실패\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32m전부 통과.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
