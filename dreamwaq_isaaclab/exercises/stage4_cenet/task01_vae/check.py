#!/usr/bin/env python3
"""빠른 검증 — vae-core.

Isaac Sim 없이 순수 torch 로 돈다. 1초면 끝난다.

    python check.py                 # starter.py 를 검사
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
SOLUTION = HERE / "solution.py"
STARTER = HERE / "starter.py"

INPUT_DIM, LATENT_DIM, BATCH = 8, 2, 64


def load_module(path: Path):
    """L3 라 starter 는 모듈 최상단에서 raise 한다 — import 자체가 실패할 수 있다."""
    if not path.exists():
        raise SystemExit(f"파일이 없다: {path}\n  exercises/tools/make_exercise.py --id vae-core 로 생성한다")
    spec = importlib.util.spec_from_file_location(f"_vae_{path.stem}", path)
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
    try:
        mod = load_module(path)
    except NotImplementedError:
        print("  아직 TODO(vae-core) 가 비어 있다. starter.py 에 ToyVAE 를 쓴다.")
        return 1

    if not hasattr(mod, "ToyVAE"):
        print("  ToyVAE 클래스가 없다. README.md 의 API 를 그대로 맞춘다.")
        return 1
    ToyVAE = mod.ToyVAE

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def make(beta: float = 1.0):
        torch.manual_seed(0)
        return ToyVAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, beta=beta)

    torch.manual_seed(0)
    x = torch.randn(BATCH, INPUT_DIM)

    # --- 1. encode → (mu, logvar) -----------------------------------------------
    net = make()
    mu, logvar = net.encode(x)
    check("1. encode 가 mu, logvar 를 준다", tuple(mu.shape) == (BATCH, LATENT_DIM)
          and tuple(logvar.shape) == (BATCH, LATENT_DIM),
          f"mu {tuple(mu.shape)}, logvar {tuple(logvar.shape)}")
    check("1. mu 와 logvar 는 서로 다른 값이다 (같은 절반을 두 번 쓰지 않았다)",
          not torch.allclose(mu, logvar, atol=1e-6))

    # --- 2. reparameterize ------------------------------------------------------
    mu_f = torch.zeros(BATCH, LATENT_DIM, requires_grad=True)
    lv_tiny = torch.full((BATCH, LATENT_DIM), -30.0)  # std ~ 0 → z ~ mu
    z_tiny = net.reparameterize(mu_f, lv_tiny)
    check("2. logvar 가 아주 작으면 z ≈ mu (std = exp(0.5*logvar))",
          torch.allclose(z_tiny, mu_f, atol=1e-4), f"최대 오차 {(z_tiny - mu_f).abs().max():.3e}")

    lv0 = torch.zeros(BATCH, LATENT_DIM)
    z_a = net.reparameterize(mu_f, lv0)
    z_b = net.reparameterize(mu_f, lv0)
    check("2. 같은 인자로 두 번 부르면 다른 표본이 나온다",
          not torch.allclose(z_a, z_b, atol=1e-6), "무작위 표집을 빠뜨렸다")
    check("2. 표본의 표준편차가 대략 1 이다 (logvar=0 → std=1)",
          0.8 < float(z_a.std().detach()) < 1.25, f"실제 {float(z_a.std().detach()):.3f}")

    z_a.sum().backward()
    check("2. mu 로 gradient 가 흐른다 (reparameterization trick)",
          mu_f.grad is not None and float(mu_f.grad.abs().sum()) > 0)

    # --- 3. decode / forward ----------------------------------------------------
    net = make()
    x_hat = net.decode(torch.randn(BATCH, LATENT_DIM))
    check("3. decode: (B, latent) → (B, input)", tuple(x_hat.shape) == (BATCH, INPUT_DIM),
          f"실제 {tuple(x_hat.shape)}")

    out = net.forward(x)
    ok4 = isinstance(out, tuple) and len(out) == 4
    check("3. forward 가 (x_hat, mu, logvar, z) 4개를 준다", ok4,
          f"len={len(out) if hasattr(out, '__len__') else '?'}")
    if ok4:
        xh, m, lv, z = out
        check("3. forward 반환 shape 이 맞다",
              tuple(xh.shape) == (BATCH, INPUT_DIM) and tuple(m.shape) == (BATCH, LATENT_DIM)
              and tuple(lv.shape) == (BATCH, LATENT_DIM) and tuple(z.shape) == (BATCH, LATENT_DIM))
        # decode(mu) 로 써도 위 검사는 전부 통과한다 (shape 이 같다). 표집이 forward 경로에
        # 실제로 들어가 있는지는 같은 x 를 두 번 넣어 봐야 드러난다.
        xh2, _, _, _ = net.forward(x)
        check("3. 같은 x 로 forward 를 두 번 부르면 x_hat 이 달라진다",
              not torch.allclose(xh, xh2, atol=1e-6),
              "표집이 forward 경로에 없다 — decode(z) 가 아니라 decode(mu) 를 썼을 수 있다")

    # --- 4. ELBO ----------------------------------------------------------------
    net = make()
    total, recon, kl = net.loss(x)
    check("4. loss 가 (total, recon, kl) 을 준다",
          all(torch.is_tensor(t) and t.dim() == 0 for t in (total, recon, kl)))
    check("4. total == recon + kl", torch.allclose(total, recon + kl, atol=1e-5),
          f"total={total.item():.6f} vs {recon.item() + kl.item():.6f}")

    # q == p 이면 KL == 0. encode 를 가로채 mu=0, logvar=0 을 강제한다.
    net = make()
    zeros = torch.zeros(BATCH, LATENT_DIM)
    net.encode = lambda _x: (zeros.clone().requires_grad_(True), zeros.clone().requires_grad_(True))
    _, _, kl0 = net.loss(x)
    check("4. mu=0, logvar=0 이면 kl == 0", abs(kl0.item()) < 1e-6, f"kl={kl0.item():.6f}")

    def kl_with(mu_val: float, lv_val: float, beta: float = 1.0, latent: int = LATENT_DIM) -> float:
        torch.manual_seed(0)
        n = ToyVAE(input_dim=INPUT_DIM, latent_dim=latent, beta=beta)
        m = torch.full((BATCH, latent), mu_val, requires_grad=True)
        lv = torch.full((BATCH, latent), lv_val, requires_grad=True)
        n.encode = lambda _x: (m, lv)
        return n.loss(x)[2].item()

    check("4. mu 가 0 에서 멀어지면 kl 증가", kl_with(1.0, 0.0) > kl_with(0.0, 0.0) + 1e-6)
    check("4. logvar 가 0 에서 멀어지면 kl 증가", kl_with(0.0, 1.5) > kl_with(0.0, 0.0) + 1e-6)
    # recon 은 nn.MSELoss() = 차원당 평균이다. KL 도 같은 축이어야 두 항의 비율이 차원 수에
    # 휘둘리지 않는다. 판별법: 차원당 값을 고정한 채 잠재 차원만 늘려 본다 — 평균이면 그대로,
    # 합이면 비례해 커진다.
    check("4. kl 은 잠재 차원으로 '평균'한다 (mu=1, logvar=0 → 0.5)",
          abs(kl_with(1.0, 0.0) - 0.5) < 1e-4,
          f"0.5 를 기대했는데 {kl_with(1.0, 0.0):.4f} — 합이면 {LATENT_DIM * 0.5} 가 나온다")
    check("4. 잠재 차원을 2 → 4 로 늘려도 kl 이 그대로다 (합이면 2배가 된다)",
          abs(kl_with(1.0, 0.0) - kl_with(1.0, 0.0, latent=LATENT_DIM * 2)) < 1e-4,
          f"latent {LATENT_DIM} → {kl_with(1.0, 0.0):.4f}, "
          f"latent {LATENT_DIM * 2} → {kl_with(1.0, 0.0, latent=LATENT_DIM * 2):.4f}")

    # --- 5. beta 는 KL 에만 --------------------------------------------------------
    k1, k2 = kl_with(1.0, 0.5, beta=1.0), kl_with(1.0, 0.5, beta=2.0)
    check("5. beta 2배 → kl 정확히 2배", abs(k2 - 2 * k1) < 1e-4, f"{k1:.6f} → {k2:.6f}")

    # --- 6. 완벽 복원이면 recon == 0 -------------------------------------------------
    net = make()
    net.decode = lambda _z: x
    _, recon0, _ = net.loss(x)
    check("6. x_hat == x 이면 recon == 0", abs(recon0.item()) < 1e-6, f"recon={recon0.item():.6f}")

    # --- 7. 실제로 학습이 된다 --------------------------------------------------------
    net = make()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    data = mod.make_toy_batch(256, seed=1)
    first = None
    for step in range(200):
        t, _, _ = net.loss(data)
        opt.zero_grad()
        t.backward()
        opt.step()
        if step == 0:
            first = t.item()
    check("7. 200 step 학습으로 손실이 줄어든다", t.item() < first,
          f"{first:.4f} → {t.item():.4f}")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)}개 실패\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32m전부 통과.\033[0m 이어서 cenet-forward 실습으로 간다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
