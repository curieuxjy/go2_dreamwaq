#!/usr/bin/env python3
"""Stage 4 · task01 — 순수 VAE (toy 데이터).

CENet 은 VAE 다. 지형도, 로봇도, Isaac Sim 도 없이 VAE 만 먼저 세운다.
여기서 쓴 encode / reparameterize / decode / ELBO 가 다음 실습(cenet-forward,
cenet-loss)에서 그대로 CENet 이 된다.

    python solution.py                # toy 데이터로 짧게 학습하고 손실이 줄어드는지 본다
    python solution.py --beta 10.0    # 규제를 세게 걸어 posterior collapse 를 눈으로 본다
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

INPUT_DIM, HIDDEN_DIM, LATENT_DIM = 8, 32, 2


def make_toy_batch(batch: int, input_dim: int = INPUT_DIM, seed: int | None = None) -> torch.Tensor:
    """2차원 잠재 구조를 input_dim 차원으로 펼친 toy 관측.

    진짜 자유도는 2뿐인데 8차원으로 관측된다 — VAE 가 찾아내야 하는 상황을 만든 것이다.
    DreamWaQ 로 치면 "지형의 진짜 상태는 몇 개 수로 요약되는데 로봇은 225차원 이력만 본다"에 해당한다.
    """
    gen = torch.Generator().manual_seed(0)  # 펼치는 행렬은 항상 같아야 한다
    mixer = torch.randn(LATENT_DIM, input_dim, generator=gen)
    if seed is not None:
        torch.manual_seed(seed)
    z_true = torch.randn(batch, LATENT_DIM)
    return z_true @ mixer + 0.05 * torch.randn(batch, input_dim)


# ex:begin id=vae-core level=3 stage=stage4_cenet task=toy 데이터용 VAE 를 통째로 구현한다 — encode / reparameterize / decode / ELBO
#   hint: encoder 는 입력을 2*latent_dim 으로 보내고, 앞 절반을 mu, 뒤 절반을 logvar 로 가른다
#   hint: reparameterize 는 z = mu + eps * std, std = exp(0.5 * logvar), eps ~ N(0, I) 다. mu 로 gradient 가 흘러야 한다
#   hint: KL(q||p) = -0.5 * sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j)) — 그 뒤 잠재 차원으로 '평균'내고 배치로도 평균낸다. recon 이 MSE 평균이라 축을 맞추는 것이다
#   hint: loss 가 돌려주는 kl 에는 self.beta 를 이미 곱해 둔다 (total = recon + kl). CENet 과 같은 규약이다
#   hint: 요구되는 정확한 API 는 이 폴더의 README.md 에 적혀 있다
class ToyVAE(nn.Module):
    """입력 -> 잠재분포 -> 표집 -> 복원. CENet 의 뼈대와 같다."""

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, beta=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # 인코더는 mu 와 logvar 를 한 번에 낸다 — 그래서 출력이 2 * latent_dim 이다.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.split(self.latent_dim, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # 표집을 mu + eps*std 로 쓰면 무작위성이 eps 에만 남아 mu/logvar 로 gradient 가 흐른다.
        # z ~ N(mu, sigma^2) 에서 그냥 뽑으면 미분할 수 없다 — 이것이 reparameterization trick 이다.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def loss(self, x):
        x_hat, mu, logvar, _ = self.forward(x)

        recon_loss = nn.MSELoss()(x_hat, x)

        # recon 은 nn.MSELoss() — 입력 차원당 평균이다. KL 도 같은 축(잠재 차원당 평균)으로
        # 축약해야 두 항의 비율이 차원 수에 휘둘리지 않는다. CENet 과 같은 규약이다.
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = klds.mean(-1).mean() * self.beta

        return recon_loss + kl_loss, recon_loss, kl_loss
# ex:end


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=1.0, help="KL 가중치. 1.0 과 10.0 을 비교해 본다")
    args = ap.parse_args()

    torch.manual_seed(0)
    net = ToyVAE(beta=args.beta)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    print(f"beta = {args.beta}")

    data = make_toy_batch(512, seed=1)
    first = last = None
    for step in range(300):
        total, recon, kl = net.loss(data)
        opt.zero_grad()
        total.backward()
        opt.step()
        if step == 0:
            first = total.item()
        if step % 60 == 0:
            print(f"  step {step:3d}  total={total.item():8.4f}  recon={recon.item():8.4f}  kl={kl.item():7.4f}")
        last = total.item()

    print(f"\n첫 손실 {first:.4f} → 마지막 손실 {last:.4f}")
    assert last < first, "학습이 손실을 줄이지 못했다"
    print("[PASS] toy VAE 가 학습된다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
