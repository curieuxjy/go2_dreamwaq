#!/usr/bin/env python3
"""빠른 검증 — stage5-summary.

합성 로그만 쓴다. 배포 자산도 Isaac Sim 도 필요 없다.

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

W = 1.5  # TRACK_WEIGHT


def load_module(path: Path):
    if not path.exists():
        raise SystemExit(f"파일이 없다: {path}\n  exercises/tools/make_exercise.py --id stage5-summary 로 생성한다")
    spec = importlib.util.spec_from_file_location(f"_s5_{path.stem}", path)
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
    summarize = load_module(path).summarize

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def approx(a: float, b: float, tol: float = 1e-9) -> bool:
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

    # 마지막 3개가 각각 0.4 / 0.5 / 0.6 (underlying) 이 되도록 만든 로그.
    flat3 = {
        "Base": [9.9, 0.3 * W, 0.4 * W, 0.5 * W],   # 마지막 3개 평균 = 0.4
        "Waq": [9.9, 0.4 * W, 0.5 * W, 0.6 * W],    # = 0.5
        "Oracle": [9.9, 0.5 * W, 0.6 * W, 0.7 * W],  # = 0.6
    }

    try:
        r = summarize(flat3, last_k=3)
    except NotImplementedError:
        print("  아직 TODO(stage5-summary) 가 비어 있다. starter.py 를 채운다.")
        return 1

    # --- 1. 가중치를 되돌린다 ---------------------------------------------------------
    check("1. underlying = 로그값 / 1.5", approx(r.get("Base", 0), 0.4),
          f"Base={r.get('Base')} (기대 0.4). 가중치로 나누지 않았을 수 있다")
    check("1. 세 config 모두 계산된다",
          approx(r.get("Waq", 0), 0.5) and approx(r.get("Oracle", 0), 0.6),
          f"Waq={r.get('Waq')}, Oracle={r.get('Oracle')}")

    # --- 2. 마지막 last_k 개의 평균이다 (마지막 한 점이 아니다) ----------------------------
    spike = {"Base": [0.4 * W, 0.4 * W, 0.4 * W, 0.4 * W, 9.0 * W]}
    r2 = summarize(spike, last_k=3)
    check("2. 최종값은 마지막 last_k 개 평균이다",
          approx(r2["Base"], (0.4 + 0.4 + 9.0) / 3),
          f"{r2['Base']:.4f} — 마지막 한 점만 봤거나 전체 평균을 냈다")

    # --- 3. 두 델타 -----------------------------------------------------------------
    check("3. Waq-Base = Waq − Base", approx(r["Waq-Base"], 0.1), f"{r.get('Waq-Base')}")
    check("3. Oracle-Waq = Oracle − Waq", approx(r["Oracle-Waq"], 0.1), f"{r.get('Oracle-Waq')}")

    # --- 4. 델타 부호가 방향을 담는다 -----------------------------------------------------
    # CENet 이 오히려 나쁠 수도 있다 — 이 프로젝트의 이전 런(logs/_archive_3000it_stairheavy)이
    # 실제로 Waq-Base = -0.0166 이었다. 부호를 지우면 그 사실을 못 본다.
    worse = {"Base": [0.6 * W] * 3, "Waq": [0.4 * W] * 3, "Oracle": [0.7 * W] * 3}
    r3 = summarize(worse, last_k=3)
    check("4. Waq 가 Base 보다 나쁘면 Waq-Base 가 음수",
          r3["Waq-Base"] < 0, f"{r3['Waq-Base']:+.3f} — abs() 를 쓰면 방향이 사라진다")

    # --- 5. 빠진 run 을 0 으로 채우지 않는다 ------------------------------------------------
    partial = {"Base": [0.4 * W] * 3}
    r4 = summarize(partial, last_k=3)
    check("5. 없는 config 는 키 자체가 없다 (0.0 으로 채우지 않는다)",
          "Waq" not in r4 and "Oracle" not in r4, f"{sorted(r4)}")
    check("5. 짝이 없으면 델타도 넣지 않는다",
          "Waq-Base" not in r4 and "Oracle-Waq" not in r4, f"{sorted(r4)}")

    # --- 6. 빈 리스트도 '없음' 으로 다룬다 ----------------------------------------------------
    r5 = summarize({"Base": [0.4 * W] * 3, "Waq": [], "Oracle": [0.6 * W] * 3}, last_k=3)
    check("6. 빈 리스트는 없는 것으로 친다",
          "Waq" not in r5 and "Base" in r5 and "Oracle" in r5, f"{sorted(r5)}")
    check("6. Waq 가 없으면 두 델타 모두 없다",
          "Waq-Base" not in r5 and "Oracle-Waq" not in r5, f"{sorted(r5)}")

    # --- 7. 로그가 last_k 개보다 적어도 동작한다 ------------------------------------------
    r6 = summarize({"Base": [0.45 * W]}, last_k=3)
    check("7. 로그가 1개뿐이어도 그 값으로 계산한다", approx(r6["Base"], 0.45), f"{r6.get('Base')}")

    # --- 7b. 기본 last_k 가 충분히 크다 ------------------------------------------------
    # 이 지표는 iteration 마다 sd~0.03 으로 출렁인다. 기본값이 작으면 노이즈가 차이보다 커진다
    # (배포 자산의 flat 은 last-3 이 -0.012, last-400 이 +0.0007 로 부호가 뒤집힌다).
    long_run = [0.9 * W] * 2700 + [0.1 * W] * 300   # 앞 2700 은 0.9, 뒤 300 은 0.1
    r7 = summarize({"Base": long_run})
    check("7b. 기본 last_k 가 100 이상이다 (몇 점만 쓰면 노이즈에 뒤집힌다)",
          r7["Base"] < 0.5, f"기본값 평균={r7['Base']:.3f} — 뒤 300 점(0.1)만 보면 0.1 근처여야 한다")

    # --- 8. weight / last_k 를 인자로 바꿀 수 있다 ------------------------------------------
    r7 = summarize({"Base": [1.0, 3.0]}, weight=1.0, last_k=1)
    check("8. weight 와 last_k 인자가 실제로 반영된다", approx(r7["Base"], 3.0),
          f"{r7.get('Base')} (기대 3.0) — 상수를 하드코딩했을 수 있다")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)}개 실패\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32m전부 통과.\033[0m 이어서 task03_interpret 로 간다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
