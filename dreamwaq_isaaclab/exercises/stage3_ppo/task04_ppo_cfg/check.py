#!/usr/bin/env python3
"""빠른 검증 — ppo-cfg.

Isaac Sim 없이 순수 파이썬으로 돈다. 1초면 끝난다.

    python check.py                 # starter.py 를 검사
    python check.py --solution      # 완성본을 검사 (검사기 자체를 점검할 때)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOLUTION = HERE / "solution.py"
STARTER = HERE / "starter.py"

DESIRED_KL = 0.01
LR0 = 1.0e-3


def load_module(path: Path):
    if not path.exists():
        raise SystemExit(f"파일이 없다: {path}\n  exercises/tools/make_exercise.py --id ppo-cfg 로 생성한다")
    spec = importlib.util.spec_from_file_location(f"_cfg_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true", help="완성본을 검사한다")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"검사 대상: {path.relative_to(REPO_ROOT)}\n")
    mod = load_module(path)
    adapt = mod.adapt_learning_rate

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def approx(a: float, b: float, tol: float = 1e-12) -> bool:
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

    try:
        got = adapt(0.05, LR0, DESIRED_KL)
    except NotImplementedError:
        print("  아직 TODO(ppo-cfg) 가 비어 있다. starter.py 를 채운다.")
        return 1

    # --- 1. KL 이 크면 낮춘다 ------------------------------------------------------
    check("1. KL > desired*2 이면 lr / 1.5", approx(got, LR0 / 1.5),
          f"{LR0:.3e} → {got:.6e} (기대 {LR0 / 1.5:.6e})")

    # --- 2. KL 이 작으면 높인다 ----------------------------------------------------
    got = adapt(0.001, LR0, DESIRED_KL)
    check("2. KL < desired/2 이면 lr * 1.5", approx(got, LR0 * 1.5),
          f"{LR0:.3e} → {got:.6e} (기대 {LR0 * 1.5:.6e})")

    # --- 3. 죽은 구간은 건드리지 않는다 ----------------------------------------------
    # desired/2 = 0.005 ~ desired*2 = 0.02 사이. 경계값도 함께 본다.
    for kl in (0.005, 0.008, 0.01, 0.015, 0.02):
        got = adapt(kl, LR0, DESIRED_KL)
        check(f"3. KL={kl} 은 lr 유지", approx(got, LR0), f"{got:.6e}")

    # --- 4. 하한 / 상한 ------------------------------------------------------------
    check("4. 하한 1e-5 아래로 내려가지 않는다", approx(adapt(0.5, 1e-5, DESIRED_KL), 1e-5),
          f"{adapt(0.5, 1e-5, DESIRED_KL):.6e}")
    check("4. 상한 1e-2 위로 올라가지 않는다", approx(adapt(0.0001, 1e-2, DESIRED_KL), 1e-2),
          f"{adapt(0.0001, 1e-2, DESIRED_KL):.6e}")

    # --- 5. kl_mean == 0 이면 올리지 않는다 -----------------------------------------
    # rsl_rl 의 `kl_mean > 0.0` 조건. 이게 없으면 KL 이 0 일 때도 lr 이 계속 커진다.
    check("5. KL == 0 이면 lr 을 올리지 않는다", approx(adapt(0.0, LR0, DESIRED_KL), LR0),
          f"{adapt(0.0, LR0, DESIRED_KL):.6e}")

    # --- 6. desired_kl 을 바꾸면 경계도 따라 움직인다 ---------------------------------
    # desired=0.1 이면 KL=0.05 는 죽은 구간(0.05~0.2)이라 그대로여야 한다.
    check("6. desired_kl 을 키우면 같은 KL 이 죽은 구간이 된다",
          approx(adapt(0.05, LR0, 0.1), LR0), f"{adapt(0.05, LR0, 0.1):.6e}")

    # --- 7. 반복 적용이 rsl_rl 과 같은 궤적을 그린다 -----------------------------------
    trace = [0.05, 0.03, 0.02, 0.004, 0.002]
    lr, want = LR0, LR0
    for kl in trace:
        lr = adapt(kl, lr, DESIRED_KL)
        if kl > DESIRED_KL * 2.0:
            want = max(1e-5, want / 1.5)
        elif kl < DESIRED_KL / 2.0 and kl > 0.0:
            want = min(1e-2, want * 1.5)
    check("7. 연속 적용 궤적이 rsl_rl 규칙과 일치", approx(lr, want),
          f"{lr:.6e} vs {want:.6e}")

    # --- 8. 원본 값을 망가뜨리지 않는다 (반환값으로만 전달) -------------------------------
    check("8. 부작용 없이 값을 반환한다", isinstance(adapt(0.05, LR0, DESIRED_KL), float))

    # --- 9. log_std clamp (주어진 함수 — 회귀 검사) --------------------------------------
    import torch

    clamped = mod.clamp_log_std(torch.tensor([-99.0, 99.0, float("nan"), float("-inf")]))
    check("9. clamp_log_std 가 [-5, 0] 으로 묶고 NaN 을 없앤다",
          bool(torch.isfinite(clamped).all())
          and float(clamped.min()) >= -5.0 and float(clamped.max()) <= 0.0,
          f"{clamped.tolist()}")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)}개 실패\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32m전부 통과.\033[0m Stage 3 실습을 모두 마쳤다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
