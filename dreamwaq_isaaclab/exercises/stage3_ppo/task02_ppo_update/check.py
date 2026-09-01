#!/usr/bin/env python3
"""빠른 검증 — ppo-losses.  (Isaac Sim 불필요, ~1초)

    python check.py                 # starter.py
    python check.py --solution      # ../ppo.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOLUTION = HERE.parent / "ppo.py"
STARTER = HERE / "starter.py"

B = 256
CLIP = 0.2


def load(path: Path):
    if not path.exists():
        raise SystemExit(f"파일이 없다: {path}\n  make_exercise.py --id ppo-losses 로 생성한다")
    spec = importlib.util.spec_from_file_location(f"_ppo_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"검사 대상: {path.relative_to(REPO_ROOT)}\n")
    losses = load(path).ppo_losses

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    torch.manual_seed(0)
    lp = torch.randn(B) * 0.1
    adv = torch.randn(B)
    val = torch.randn(B)
    ret = torch.randn(B)
    ent = torch.rand(B) + 0.5

    try:
        total, sur, vl, el = losses(lp, lp.clone(), adv, val, ret, ent)
    except NotImplementedError:
        print("  아직 TODO(ppo-losses) 가 비어 있다. starter.py 를 채운다.")
        return 1

    check("1. 네 값이 모두 스칼라", all(x.dim() == 0 for x in (total, sur, vl, el)),
          f"{[tuple(x.shape) for x in (total, sur, vl, el)]}")

    # 정책이 안 바뀌었으면 ratio=1 -> surrogate = -mean(A)
    check("2. ratio=1 일 때 surrogate = -mean(advantage)",
          torch.allclose(sur, -adv.mean(), atol=1e-5), f"{sur.item():.6f} vs {-adv.mean().item():.6f}")

    # value loss 는 MSE
    check("3. value_loss = mean((value - returns)^2)",
          torch.allclose(vl, (val - ret).pow(2).mean(), atol=1e-6), f"{vl.item():.6f}")

    # entropy 는 손실에 음수로
    check("4. entropy_loss = -mean(entropy)", torch.allclose(el, -ent.mean(), atol=1e-6),
          f"{el.item():.6f} vs {-ent.mean().item():.6f}")

    # 총합 = surrogate + c_v * value + c_e * entropy
    t2, s2, v2, e2 = losses(lp, lp.clone(), adv, val, ret, ent,
                            value_loss_coef=2.0, entropy_coef=0.05)
    check("5. total = surrogate + c_v*value + c_e*entropy",
          torch.allclose(t2, s2 + 2.0 * v2 + 0.05 * e2, atol=1e-5),
          f"{t2.item():.6f} vs {(s2 + 2.0 * v2 + 0.05 * e2).item():.6f}")

    # --- 클리핑이 실제로 걸리는가 (이 실습의 핵심) ---
    # ratio 를 크게 벗어나게 만든다: log_prob - old = +1 -> ratio = e ~ 2.718
    old = torch.zeros(B)
    big = torch.ones(B)
    pos_adv = torch.ones(B)      # 어드밴티지가 양수
    neg_adv = -torch.ones(B)

    s_pos = losses(big, old, pos_adv, val, ret, ent, clip_param=CLIP)[1]
    # A>0, ratio>1+eps -> clip 이 걸려 surrogate = -(1+eps)*A
    check("6. A>0, ratio 가 상한을 넘으면 클립된다",
          torch.allclose(s_pos, torch.tensor(-(1.0 + CLIP)), atol=1e-5),
          f"{s_pos.item():.6f} (기대 {-(1.0 + CLIP):.4f}) — 클립 없으면 -2.718")

    s_neg = losses(-big, old, neg_adv, val, ret, ent, clip_param=CLIP)[1]
    # A<0, ratio<1-eps -> clip 이 걸려 surrogate = -(1-eps)*A = +(1-eps)
    check("7. A<0, ratio 가 하한을 밑돌면 클립된다",
          torch.allclose(s_neg, torch.tensor(1.0 - CLIP), atol=1e-5),
          f"{s_neg.item():.6f} (기대 {1.0 - CLIP:.4f})")

    # min() 을 빠뜨리고 clip 만 쓰면 여기서 걸린다:
    # A>0 인데 ratio 가 아래로 벗어나면 클립하지 '않은' 쪽이 더 작으므로 그대로 써야 한다.
    s_low = losses(-big, old, pos_adv, val, ret, ent, clip_param=CLIP)[1]
    ratio_low = torch.exp(-big - old)[0]
    check("8. A>0, ratio 가 하한을 밑돌면 클립하지 않는다 (min 을 썼다)",
          torch.allclose(s_low, -ratio_low, atol=1e-5),
          f"{s_low.item():.6f} (기대 {-ratio_low.item():.4f}) — clip 만 쓰면 {-(1 - CLIP):.4f} 이 나온다")

    # 어드밴티지에는 클립을 걸지 않는다
    s_a = losses(lp, lp.clone(), adv * 10.0, val, ret, ent)[1]
    check("9. advantage 에는 클립을 걸지 않는다",
          torch.allclose(s_a, -(adv * 10.0).mean(), atol=1e-4), f"{s_a.item():.6f}")

    # gradient 가 흐르는가
    lp_g = lp.clone().requires_grad_(True)
    val_g = val.clone().requires_grad_(True)
    losses(lp_g, lp.detach(), adv, val_g, ret, ent)[0].backward()
    check("10. log_prob 과 value 로 gradient 가 흐른다",
          lp_g.grad is not None and val_g.grad is not None
          and lp_g.grad.abs().sum() > 0 and val_g.grad.abs().sum() > 0)

    print()
    if failures:
        print(f"\033[1;31m{len(failures)}개 실패\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32m전부 통과.\033[0m 이제 `python starter.py` 로 실제 학습을 돌려 본다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
