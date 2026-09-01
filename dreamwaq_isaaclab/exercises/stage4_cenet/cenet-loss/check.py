#!/usr/bin/env python3
"""빠른 검증 — cenet-loss.

Isaac Sim 없이 순수 torch 텐서로 돈다. 1초면 끝난다.

    python check.py                 # starter/cenet.py 를 검사
    python check.py --solution      # 완성본을 검사 (검사기 자체를 점검할 때)
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOLUTION = (
    REPO_ROOT
    / "dreamwaq_manager/source/dreamwaq_manager/dreamwaq_manager/algorithms/cenet.py"
)
STARTER = HERE / "starter" / "cenet.py"

OBS_HISTORY, LATENT, NUM_VEL, NUM_OBS, BATCH = 225, 16, 3, 45, 64


def load_cenet(path: Path):
    """Import a CENet module from an arbitrary path without touching sys.path."""
    if not path.exists():
        raise SystemExit(f"파일이 없다: {path}\n  exercises/tools/make_exercise.py --id cenet-loss 로 생성한다")
    spec = importlib.util.spec_from_file_location(f"_cenet_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.CENet


def make_net(CENet, beta: float = 1.0, latent: int = LATENT, num_obs: int = NUM_OBS):
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):  # CENet.__init__ prints its MLP structure
        return CENet(
            num_learning_epochs=1,
            num_mini_batches=1,
            input_dim=OBS_HISTORY,
            latent_dim1=NUM_VEL + latent * 2,
            latent_dim2=NUM_VEL + latent,
            output_dim=num_obs,
            beta=beta,
            device="cpu",
        )


def leaf(t: torch.Tensor) -> torch.Tensor:
    """A differentiable stand-in: `update()` calls backward(), so fakes need a grad path."""
    return t.clone().detach().requires_grad_(True)


class Batch:
    """update() 가 소비하는 미니배치 하나를 흉내낸다."""

    def __init__(self, obs_hist, true_vel, true_onext):
        self._data = [(obs_hist, true_vel, true_onext)]

    def __call__(self, *_args, **_kwargs):
        return iter(self._data)


class FakeStorage:
    def __init__(self, batch):
        self.mini_batch_generator = batch
        self.cleared = False

    def clear(self):
        self.cleared = True


def run_update(net, obs_hist, true_vel, true_onext):
    """CENet.update() 를 합성 배치 하나로 돌리고 (total, vel, recon, kl) 을 받는다."""
    net.storage = FakeStorage(Batch(obs_hist, true_vel, true_onext))
    net.train_mode()
    return net.update()


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true", help="완성본을 검사한다")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"검사 대상: {path.relative_to(REPO_ROOT)}\n")
    CENet = load_cenet(path)

    torch.manual_seed(0)
    obs_hist = torch.randn(BATCH, OBS_HISTORY)
    true_vel = torch.randn(BATCH, NUM_VEL)
    true_onext = torch.randn(BATCH, NUM_OBS)

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    # --- 1. 스칼라로 축약되는가 -------------------------------------------------
    net = make_net(CENet)
    try:
        total, vel, recon, kl = run_update(net, obs_hist, true_vel, true_onext)
    except NotImplementedError:
        print("  아직 TODO(cenet-loss) 가 비어 있다. starter/cenet.py 를 채운다.")
        return 1
    except RuntimeError as exc:
        # 흔한 실수: `kl_loss = klds.mean(1)` 로 두면 shape 이 (B,) 라 update() 안의
        # total_loss.backward() 가 터진다. 그대로 두면 검사기가 죽어 FAIL 이 한 줄도 안 나온다.
        check("1. 네 손실이 스칼라로 반환된다", False)
        print()
        if "scalar outputs" in str(exc):
            print("\033[1;31m손실이 스칼라가 아니다 — 배치 평균(`.mean(0)`)을 빠뜨렸을 수 있다.\033[0m")
            print("  `klds.mean(1)` 은 아직 (B,) 다. 잠재 차원 평균 뒤에 배치 평균이 한 번 더 필요하다.")
            print("  `total_loss.backward()` 는 스칼라에만 쓸 수 있어서 여기서 터진다.")
        else:
            print("\033[1;31mupdate() 가 RuntimeError 로 죽었다.\033[0m")
        print(f"  torch 원문: {exc}")
        return 1
    check("1. 네 손실이 스칼라로 반환된다", all(isinstance(x, float) for x in (total, vel, recon, kl)))

    # --- 2. 완벽 예측이면 MSE 항이 0 -------------------------------------------
    # est == true 를 강제하기 위해 forward 를 가로채 정답을 그대로 돌려준다.
    net = make_net(CENet)
    mu0 = torch.zeros(BATCH, LATENT)
    net.forward = lambda _oh: (leaf(true_onext), leaf(true_vel), leaf(mu0), leaf(mu0), leaf(mu0))
    _, vel_exact, recon_exact, kl_exact = run_update(net, obs_hist, true_vel, true_onext)
    check("2. est == true 이면 vel_loss == 0", approx(vel_exact, 0.0), f"vel_loss={vel_exact:.6f}")
    check("2. est == true 이면 recon_loss == 0", approx(recon_exact, 0.0), f"recon_loss={recon_exact:.6f}")

    # --- 3. q == p 이면 KL == 0 ------------------------------------------------
    check("3. mu=0, logvar=0 이면 kl_loss == 0", approx(kl_exact, 0.0), f"kl_loss={kl_exact:.6f}")

    # --- 4·5. 결정론적 fake forward 로 KL 의 성질을 본다 -------------------------
    # 실제 forward 는 reparameterize 안에서 randn_like 를 뽑으므로 호출마다 결과가 달라진다.
    # 손실 함수 자체만 보려면 forward 를 고정값으로 대체해야 한다.
    def losses_with(mu_val: float, logvar_val: float, beta: float = 1.0, latent: int = LATENT):
        n = make_net(CENet, beta=beta, latent=latent)
        mu = torch.full((BATCH, latent), mu_val)
        lv = torch.full((BATCH, latent), logvar_val)
        est_onext, est_vel = true_onext + 0.5, true_vel + 0.3  # 일부러 틀린 예측
        n.forward = lambda _oh: (leaf(est_onext), leaf(est_vel), leaf(mu), leaf(lv), leaf(mu))
        return run_update(n, obs_hist, true_vel, true_onext)

    kl_base = losses_with(0.0, 0.0)[3]
    kl_mu = losses_with(1.0, 0.0)[3]
    kl_lv = losses_with(0.0, 1.5)[3]
    check("4. mu 가 0 에서 멀어지면 kl 증가", kl_mu > kl_base + 1e-6, f"{kl_mu:.6f} <= {kl_base:.6f}")
    check("4. logvar 가 0 에서 멀어지면 kl 증가", kl_lv > kl_base + 1e-6, f"{kl_lv:.6f} <= {kl_base:.6f}")

    # --- 4b. 리덕션 축 — 이 실습의 핵심 -------------------------------------------
    # 부등호만 보면 합이든 평균이든 통과한다. 정작 학습을 무너뜨리는 것은 "recon 과 KL 이
    # 서로 다른 축으로 축약되는 것"이다. nn.MSELoss() 는 차원당 평균이므로 KL 도 차원당
    # 평균이어야 두 항의 비율이 차원 수에 휘둘리지 않는다.
    #
    # 판별법: 차원당 값을 고정한 채 차원 수만 늘려 본다. 평균이면 값이 그대로고,
    # 합이면 차원 수에 비례해 커진다.
    def recon_with_dims(num_obs: int) -> float:
        n = make_net(CENet, num_obs=num_obs)
        t_onext = torch.zeros(BATCH, num_obs)
        e_onext = t_onext + 1.0  # 원소마다 오차 1 → 차원당 평균이면 recon == 1.0
        mu = torch.zeros(BATCH, LATENT)
        n.forward = lambda _oh: (leaf(e_onext), leaf(true_vel), leaf(mu), leaf(mu), leaf(mu))
        return run_update(n, obs_hist, true_vel, t_onext)[2]

    kl_l16 = losses_with(1.0, 0.0, latent=LATENT)[3]
    kl_l32 = losses_with(1.0, 0.0, latent=LATENT * 2)[3]
    recon_d45 = recon_with_dims(NUM_OBS)
    recon_d90 = recon_with_dims(NUM_OBS * 2)

    kl_is_mean = approx(kl_l16, kl_l32, 1e-4)
    recon_is_mean = approx(recon_d45, recon_d90, 1e-4)
    check("4. kl 은 잠재 차원(dim=1)으로 '평균'한다 (mu=1, logvar=0 → 0.5)",
          approx(kl_l16, 0.5, 1e-4),
          f"0.5 를 기대했는데 {kl_l16:.4f} — 합이면 {LATENT * 0.5} 가 나온다")
    check("4. 잠재 차원을 16 → 32 로 늘려도 kl 이 그대로다 (합이면 2배가 된다)",
          kl_is_mean, f"latent 16 → {kl_l16:.4f}, latent 32 → {kl_l32:.4f}")
    check("4. recon 과 kl 의 리덕션 축이 같다 (둘 다 차원당 평균)",
          kl_is_mean and recon_is_mean,
          f"recon 45dim {recon_d45:.4f} / 90dim {recon_d90:.4f}, "
          f"kl 16dim {kl_l16:.4f} / 32dim {kl_l32:.4f}")

    # --- 5. beta 는 KL 에만, 선형으로 --------------------------------------------
    b1 = losses_with(1.0, 0.5, beta=1.0)
    b2 = losses_with(1.0, 0.5, beta=2.0)
    check("5. beta 2배 → kl 정확히 2배", approx(b2[3], 2 * b1[3], 1e-4),
          f"beta=1 → {b1[3]:.6f}, beta=2 → {b2[3]:.6f}")
    check("5. beta 를 바꿔도 vel/recon 은 불변",
          approx(b1[1], b2[1], 1e-4) and approx(b1[2], b2[2], 1e-4),
          f"vel {b1[1]:.6f}/{b2[1]:.6f}, recon {b1[2]:.6f}/{b2[2]:.6f}")

    # --- 6. 총합이 세 항의 합인가 (맨 위에서 돌린 실제 forward 결과로 본다) --------
    check("6. total == vel + recon + kl", approx(total, vel + recon + kl, 1e-4),
          f"total={total:.6f} vs sum={vel + recon + kl:.6f}")
    # β=1 에서만 재면 구멍이 남는다. β 를 `total_loss` 쪽에만 곱하고 반환하는 `kl_loss` 에는
    # 안 곱한 구현(또는 그 반대)은 β=1 에서 두 값이 같아 그냥 통과한다. 5번도 못 잡는다 —
    # 5번은 `kl_loss` 가 β 에 비례하는지만 보기 때문이다. β≠1 로 한 번 더 잰다.
    check("6. beta≠1 에서도 total == vel + recon + kl (β 를 두 군데에 따로 곱하지 않았다)",
          approx(b2[0], b2[1] + b2[2] + b2[3], 1e-4),
          f"beta=2 에서 total={b2[0]:.6f} vs vel+recon+kl={b2[1] + b2[2] + b2[3]:.6f}")

    # --- 7. gradient 가 encoder·decoder 양쪽에 흐르는가 --------------------------
    net = make_net(CENet)
    run_update(net, obs_hist, true_vel, true_onext)
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.encoder.parameters())
    dec_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.decoder.parameters())
    check("7. encoder 에 gradient 가 흐른다", enc_grad)
    check("7. decoder 에 gradient 가 흐른다", dec_grad)

    print()
    if failures:
        print(f"\033[1;31m{len(failures)}개 실패\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32m전부 통과.\033[0m 이어서 runner-augment 실습으로 간다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
