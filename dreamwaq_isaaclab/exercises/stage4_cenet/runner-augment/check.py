#!/usr/bin/env python3
"""빠른 검증 — runner-augment.

Isaac Sim 없이 돈다. `_build_actor_obs` 를 클래스에서 떼어내 가짜 self 로 부른다
(러너 전체를 띄우려면 env 가 필요하지만, 이 메서드는 self 의 몇 가지만 쓴다).

    python check.py                 # starter/dreamwaq_runner.py 를 검사
    python check.py --solution      # 완성본을 검사 (검사기 자체를 점검할 때)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOLUTION = (
    REPO_ROOT
    / "dreamwaq_manager/source/dreamwaq_manager/dreamwaq_manager/algorithms/dreamwaq_runner.py"
)
STARTER = HERE / "starter" / "dreamwaq_runner.py"

NUM_BASE, NUM_VEL, LATENT, NUM_SCAN, BATCH = 45, 3, 16, 187, 8
NORM_GAIN = 2.0  # 가짜 정규화: base 관측만 이 배수로 바뀐다 → 어디에 적용됐는지 눈에 보인다


def load_build_actor_obs(path: Path):
    """Load the module and return the *unbound* `_build_actor_obs`."""
    if not path.exists():
        raise SystemExit(f"파일이 없다: {path}\n  exercises/tools/make_exercise.py --id runner-augment 로 생성한다")
    spec = importlib.util.spec_from_file_location(f"_runner_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.OnPolicyRunnerWaq._build_actor_obs


class FakeSelf:
    """`_build_actor_obs` 가 실제로 건드리는 것만 갖춘 가짜 러너."""

    def __init__(self, use_exteroception: bool):
        self.use_exteroception = use_exteroception
        self.num_base_obs = NUM_BASE
        # 주어진 NaN 경고 블록이 읽는 플래그. True 로 두면 경고를 한 번도 안 찍으므로
        # 검사 출력이 깨끗하다 (경고 자체는 이 실습의 채점 대상이 아니다).
        self._nan_warned = True

    def _norm_base(self, base_obs):
        return base_obs * NORM_GAIN

    def _extract_extero(self, src_obs):
        return src_obs[:, NUM_BASE:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true", help="완성본을 검사한다")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"검사 대상: {path.relative_to(REPO_ROOT)}\n")
    build = load_build_actor_obs(path)

    torch.manual_seed(0)
    base = torch.randn(BATCH, NUM_BASE)
    vel = torch.randn(BATCH, NUM_VEL)
    ctx = torch.randn(BATCH, LATENT)
    src_flat = base.clone()
    src_rough = torch.cat([base, torch.randn(BATCH, NUM_SCAN)], dim=-1)

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> bool:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)
        return cond

    def skip(name: str, why: str) -> None:
        """앞 검사가 실패하면 뒤 검사는 판정할 수 없다 — FAIL 로 세면 오진이 된다."""
        print(f"  [\033[1;33mSKIP\033[0m] {name}  — {why}")

    # --- 1. flat: 45 + 3 + 16 = 64 ----------------------------------------------
    try:
        out = build(FakeSelf(use_exteroception=False), base, vel, ctx, src_flat)
    except NotImplementedError:
        print("  아직 TODO(runner-augment) 가 비어 있다. starter/dreamwaq_runner.py 를 채운다.")
        return 1
    check("1. flat 에서 actor 관측이 64 차원이다 (45+3+16)",
          tuple(out.shape) == (BATCH, NUM_BASE + NUM_VEL + LATENT),
          f"실제 {tuple(out.shape)}")
    if tuple(out.shape) != (BATCH, NUM_BASE + NUM_VEL + LATENT):
        print("\n\033[1;31m차원이 달라 이후 검사를 진행할 수 없다.\033[0m")
        return 1

    # --- 2. 순서: [norm(base), vel, context] --------------------------------------
    # 4·5 번은 고정 인덱스로 특정 자리를 찌른다. 순서가 틀렸으면 그 검사들은 "무엇이 틀렸는지"
    # 를 오진하므로(예: 순서 버그를 "nan_to_num 을 빠뜨렸다"로 보고), 여기서 걸리면 건너뛴다.
    order_ok = check("2. 앞 45 는 정규화된 base 관측이다",
                     torch.allclose(out[:, :NUM_BASE], base * NORM_GAIN, atol=1e-6),
                     "base 를 정규화하지 않았거나 순서가 다르다")
    order_ok &= check("2. 그 다음 3 은 vel_input 이다 (정규화하지 않는다)",
                      torch.allclose(out[:, NUM_BASE:NUM_BASE + NUM_VEL], vel, atol=1e-6),
                      "vel 에도 정규화를 먹였거나 context 를 먼저 이었다")
    order_ok &= check("2. 마지막 16 은 context_vec 이다 (정규화하지 않는다)",
                      torch.allclose(out[:, NUM_BASE + NUM_VEL:], ctx, atol=1e-6),
                      "context 에 정규화를 먹였거나 vel 과 순서가 바뀌었다")

    # --- 3. rough: exteroception 이 맨 뒤에 붙는다 ---------------------------------
    out_r = build(FakeSelf(use_exteroception=True), base, vel, ctx, src_rough)
    want_r = NUM_BASE + NUM_VEL + LATENT + NUM_SCAN
    check("3. rough 에서 height_scan(187) 이 뒤에 붙어 251 차원이다",
          tuple(out_r.shape) == (BATCH, want_r), f"실제 {tuple(out_r.shape)}")
    if tuple(out_r.shape) == (BATCH, want_r):
        check("3. height_scan 은 맨 뒤에 그대로 붙는다",
              torch.allclose(out_r[:, NUM_BASE + NUM_VEL + LATENT:], src_rough[:, NUM_BASE:], atol=1e-6))
        check("3. 앞 64 는 flat 일 때와 같다",
              torch.allclose(out_r[:, :NUM_BASE + NUM_VEL + LATENT], out, atol=1e-6))

    # --- 4. NaN / inf 를 흘려보내지 않는다 -----------------------------------------
    # CENet 이 발산하면 NaN 이 그대로 actor 로 들어가 정책이 통째로 망가진다.
    bad_vel = vel.clone()
    bad_vel[0, 0] = float("nan")
    bad_vel[1, 1] = float("inf")
    bad_vel[2, 2] = float("-inf")
    out_bad = build(FakeSelf(use_exteroception=False), base, bad_vel, ctx, src_flat)
    # 이 항목만 자리와 무관하다 — 어디에 붙였든 유한하면 통과한다.
    check("4. NaN/inf 가 남아 있지 않다", bool(torch.isfinite(out_bad).all()),
          "torch.nan_to_num 을 빠뜨렸다")

    blocked = "앞 검사(2. 순서)가 실패해 판정 불가 — 고정 자리를 찌르는 검사다"
    if order_ok:
        check("4. nan → 0.0", float(out_bad[0, NUM_BASE]) == 0.0, f"실제 {float(out_bad[0, NUM_BASE])}")
        check("4. +inf → 10.0", float(out_bad[1, NUM_BASE + 1]) == 10.0, f"실제 {float(out_bad[1, NUM_BASE + 1])}")
        check("4. -inf → -10.0", float(out_bad[2, NUM_BASE + 2]) == -10.0, f"실제 {float(out_bad[2, NUM_BASE + 2])}")
    else:
        skip("4. nan → 0.0 / +inf → 10.0 / -inf → -10.0", blocked)

    # --- 5. 정상 값은 클램프되지 않는다 ---------------------------------------------
    # nan_to_num 은 NaN/inf 만 건드린다. 유한한 큰 값까지 자르면 학습 신호가 왜곡된다.
    if order_ok:
        big_ctx = torch.full((BATCH, LATENT), 12345.0)
        out_big = build(FakeSelf(use_exteroception=False), base, vel, big_ctx, src_flat)
        check("5. 유한한 큰 값은 그대로 통과한다 (clamp 가 아니다)",
              torch.allclose(out_big[:, NUM_BASE + NUM_VEL:], big_ctx, atol=1e-6),
              "torch.clamp 를 썼을 수 있다 — nan_to_num 이어야 한다")
    else:
        skip("5. 유한한 큰 값은 그대로 통과한다 (clamp 가 아니다)", blocked)

    print()
    if failures:
        print(f"\033[1;31m{len(failures)}개 실패\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32m전부 통과.\033[0m Stage 4 실습을 모두 마쳤다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
