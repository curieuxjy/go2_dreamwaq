#!/usr/bin/env python3
"""빠른 검증 — cenet-forward.

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
        raise SystemExit(f"파일이 없다: {path}\n  exercises/tools/make_exercise.py --id cenet-forward 로 생성한다")
    spec = importlib.util.spec_from_file_location(f"_cenet_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.CENet


def make_net(CENet, latent_dim1: int = NUM_VEL + LATENT * 2):
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):  # CENet.__init__ prints its MLP structure
        return CENet(
            input_dim=OBS_HISTORY,
            latent_dim1=latent_dim1,
            latent_dim2=NUM_VEL + LATENT,
            output_dim=NUM_OBS,
            device="cpu",
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true", help="완성본을 검사한다")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"검사 대상: {path.relative_to(REPO_ROOT)}\n")
    CENet = load_cenet(path)

    torch.manual_seed(0)
    obs_hist = torch.randn(BATCH, OBS_HISTORY)

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    net = make_net(CENet)
    try:
        out = net.forward(obs_hist)
    except NotImplementedError:
        print("  아직 TODO(cenet-forward) 가 비어 있다. starter/cenet.py 를 채운다.")
        return 1

    # --- 1. 반환 형태 -----------------------------------------------------------
    ok_tuple = isinstance(out, tuple) and len(out) == 5
    check("1. (est_next_obs, est_vel, mu, logvar, context_vec) 5개를 반환한다", ok_tuple,
          f"받은 것: {type(out).__name__} len={len(out) if hasattr(out, '__len__') else '?'}")
    if not ok_tuple:
        print("\n\033[1;31m반환 형태가 달라 이후 검사를 진행할 수 없다.\033[0m")
        return 1
    est_next_obs, est_vel, mu, logvar, context_vec = out

    # --- 2. shape ---------------------------------------------------------------
    shapes = {
        "est_next_obs (B,45)": (est_next_obs, (BATCH, NUM_OBS)),
        "est_vel (B,3)": (est_vel, (BATCH, NUM_VEL)),
        "mu (B,16)": (mu, (BATCH, LATENT)),
        "logvar (B,16)": (logvar, (BATCH, LATENT)),
        "context_vec (B,16)": (context_vec, (BATCH, LATENT)),
    }
    for label, (tensor, want) in shapes.items():
        check(f"2. {label}", tuple(tensor.shape) == want, f"실제 {tuple(tensor.shape)}")

    # --- 3. 인코더 출력을 3 / 16 / 16 으로 정확히 갈랐는가 -------------------------
    # 인코더는 결정론적이므로 직접 돌려 기대값을 만든다.
    h = net.encoder(obs_hist)
    check("3. est_vel 은 인코더 출력의 앞 3 차원이다",
          torch.allclose(est_vel, h[:, :NUM_VEL], atol=1e-6))
    check("3. mu 는 그 다음 16 차원이다",
          torch.allclose(mu, h[:, NUM_VEL:NUM_VEL + LATENT], atol=1e-6))
    check("3. logvar 는 마지막 16 차원이다",
          torch.allclose(logvar, h[:, NUM_VEL + LATENT:], atol=1e-6),
          "mu 와 logvar 의 순서가 바뀌었을 수 있다")

    # --- 4. 디코더에 들어간 것이 [est_vel, context_vec] 인가 ----------------------
    # 디코더에 forward pre-hook 을 걸어 입력을 붙잡는다.
    seen: dict[str, torch.Tensor] = {}
    net2 = make_net(CENet)
    net2.decoder.register_forward_pre_hook(lambda _mod, args: seen.update(latent=args[0]))
    _, est_vel2, mu2, _, ctx2 = net2.forward(obs_hist)
    latent = seen.get("latent")
    check("4. 디코더 입력이 19 차원이다 (3 + 16)",
          latent is not None and tuple(latent.shape) == (BATCH, NUM_VEL + LATENT),
          f"실제 {None if latent is None else tuple(latent.shape)}")
    if latent is not None and tuple(latent.shape) == (BATCH, NUM_VEL + LATENT):
        check("4. 디코더 입력 앞 3 = est_vel",
              torch.allclose(latent[:, :NUM_VEL], est_vel2, atol=1e-6))
        check("4. 디코더 입력 뒤 16 = context_vec (mu 가 아니다)",
              torch.allclose(latent[:, NUM_VEL:], ctx2, atol=1e-6),
              "reparameterize 결과가 아니라 mu 를 그대로 이었을 수 있다")
        check("4. 디코더에 mu 를 그대로 잇지 않았다",
              not torch.allclose(latent[:, NUM_VEL:], mu2, atol=1e-6))

    # --- 5. 재매개변수화가 실제로 표집하는가 --------------------------------------
    # 같은 입력을 두 번 넣으면 mu/logvar 는 같고 context_vec 만 달라야 한다.
    net3 = make_net(CENet)
    _, _, mu_a, lv_a, ctx_a = net3.forward(obs_hist)
    _, _, mu_b, lv_b, ctx_b = net3.forward(obs_hist)
    check("5. 같은 입력에 mu / logvar 는 결정론적이다",
          torch.allclose(mu_a, mu_b, atol=1e-6) and torch.allclose(lv_a, lv_b, atol=1e-6))
    check("5. context_vec 은 호출마다 달라진다 (표집)",
          not torch.allclose(ctx_a, ctx_b, atol=1e-6),
          "reparameterize 를 쓰지 않고 mu 를 그대로 넘겼을 수 있다")

    # --- 6. context 파라미터가 홀수면 AssertionError ------------------------------
    net4 = make_net(CENet, latent_dim1=NUM_VEL + 33)  # 33 은 절반으로 갈 수 없다
    try:
        net4.forward(obs_hist)
        raised = False
    except AssertionError:
        raised = True
    except Exception:  # noqa: BLE001 — 다른 예외는 통과로 치지 않는다
        raised = False
    check("6. context 파라미터가 홀수면 AssertionError", raised)

    # --- 7. gradient 가 encoder·decoder 양쪽에 흐른다 ------------------------------
    net5 = make_net(CENet)
    est_next_obs5, est_vel5, _, _, _ = net5.forward(obs_hist)
    (est_next_obs5.sum() + est_vel5.sum()).backward()
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net5.encoder.parameters())
    dec_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net5.decoder.parameters())
    check("7. encoder 에 gradient 가 흐른다", enc_grad)
    check("7. decoder 에 gradient 가 흐른다", dec_grad)

    print()
    if failures:
        print(f"\033[1;31m{len(failures)}개 실패\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32m전부 통과.\033[0m 이어서 cenet-loss 실습으로 간다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
