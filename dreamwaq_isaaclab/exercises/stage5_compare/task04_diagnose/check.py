#!/usr/bin/env python3
"""빠른 검증 — stage5-collapse-probe.

**합성 체크포인트만 채점한다.** 학습 산출물도 Isaac Sim 도 필요 없다. 손으로 만든 가중치라
정답을 해석적으로 알 수 있어서, 축을 잘못 잡거나 프로브 분포를 잘못 쓰면 잡힌다.

진짜 체크포인트가 있으면 마지막에 **찍어만 준다 — 채점하지 않는다.** 어느 런이 정상인지는
파일 이름으로 알 수 없고(정상이라고 이름 붙인 런이 나중에 붕괴한 적이 있다), 임계값을
런에 맞춰 고르는 순간 그 임계값은 아무것도 판정하지 못하기 때문이다. README 를 본다.

    ~/IsaacLab/_isaac_sim/python.sh check.py              # starter.py 를 검사
    ~/IsaacLab/_isaac_sim/python.sh check.py --solution   # 완성본을 검사
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOLUTION = HERE / "solution.py"
STARTER = HERE / "starter.py"

INPUT_DIM, HIDDEN1, HIDDEN2, OUT_DIM = 225, 128, 64, 35
LATENT_IN = 19  # 디코더 입력 = est_vel(3) + z(16)
BIG = 10.0  # ELU 를 항등으로 만들기 위한 바이어스. 10 + N(0,1) 이 음수일 확률은 사실상 0 이다.
E_ABS_NORMAL = math.sqrt(2.0 / math.pi)  # E|x|, x ~ N(0,1) = 0.7979


def load_module(path: Path):
    if not path.exists():
        raise SystemExit(
            f"파일이 없다: {path}\n"
            "  python3 exercises/tools/make_exercise.py --id stage5-collapse-probe 로 생성한다"
        )
    spec = importlib.util.spec_from_file_location(f"_s5d_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def synth(
    pick_row: int | None = None,
    logvar_bias: float = 0.0,
    dec_vel: float = 1.0,
    dec_z: float = 1.0,
    dec_cols: list[float] | None = None,
) -> dict:
    """해석적으로 정답을 아는 체크포인트를 만든다.

    인코더: 앞 두 층은 바이어스 BIG 덕분에 ELU 가 항등으로 동작하며 입력의 0번 성분만 실어
    나른다 (h1[0] = BIG + x0, 나머지 유닛은 0). 마지막 층에서 원하는 출력 한 줄만 골라
    `-BIG` 로 상수를 지우면 그 출력은 정확히 **x0 ~ N(0,1)** 이 된다.
    pick_row=None 이면 마지막 층이 전부 0 — 입력이 무엇이든 상수를 뱉는 '죽은' 인코더다.

    디코더: 1층 가중치의 0번 행에만 값을 넣으므로 **열 노름 = 그 값의 절대값**이다.
    dec_cols 로 열마다 따로 주거나, dec_vel/dec_z 로 앞 3열 / 뒤 16열을 한 번에 준다.
    """
    w0 = torch.zeros(HIDDEN1, INPUT_DIM); w0[0, 0] = 1.0
    w2 = torch.zeros(HIDDEN2, HIDDEN1); w2[0, 0] = 1.0
    w4 = torch.zeros(OUT_DIM, HIDDEN2)
    b4 = torch.zeros(OUT_DIM)
    b4[19:35] = logvar_bias
    if pick_row is not None:
        w4[pick_row, 0] = 1.0
        b4[pick_row] = -BIG

    cols = dec_cols if dec_cols is not None else [dec_vel] * 3 + [dec_z] * 16
    if len(cols) != LATENT_IN:
        raise AssertionError(f"dec_cols 는 {LATENT_IN} 개여야 한다")
    dec_w = torch.zeros(HIDDEN2, LATENT_IN)
    dec_w[0] = torch.tensor(cols)

    return {
        "encoder.0.weight": w0, "encoder.0.bias": torch.full((HIDDEN1,), BIG),
        "encoder.2.weight": w2, "encoder.2.bias": torch.zeros(HIDDEN2),
        "encoder.4.weight": w4, "encoder.4.bias": b4,
        "decoder.0.weight": dec_w, "decoder.0.bias": torch.zeros(HIDDEN2),
        # 뒷단 층도 섞여 있는 것이 진짜 체크포인트다 — 첫 Linear 만 골라야 한다.
        "decoder.2.weight": torch.zeros(HIDDEN1, HIDDEN2), "decoder.2.bias": torch.zeros(HIDDEN1),
    }


def nan_checkpoint() -> dict:
    """모든 텐서가 NaN 인 체크포인트. 지어낸 상황이 아니다 — `_archive_3000it_stairheavy` 의
    flat Waq 런이 it 500 부터 정확히 이 상태였고, 그런데도 보상 곡선은 멀쩡했다."""
    return {k: torch.full_like(v, float("nan")) for k, v in synth().items()}


def alive_encoder(**kw) -> dict:
    """mu 16차원이 전부 입력을 따라 움직이는 인코더 — |mu| 가 판정선을 크게 넘는다."""
    sd = synth(**kw)
    sd["encoder.4.weight"][3:19, 0] = 1.0
    sd["encoder.4.bias"][3:19] = -BIG
    return sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true", help="완성본을 검사한다")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"검사 대상: {path.relative_to(REPO_ROOT)}\n")
    mod = load_module(path)
    probe = mod.diagnose_checkpoint

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def close(a: float, b: float, rel: float = 0.03) -> bool:
        return abs(a - b) <= rel * max(1.0, abs(b))

    try:
        r_mu = probe(synth(pick_row=3), num_samples=8192, seed=0)   # mu[0] = x0
    except NotImplementedError:
        print("  아직 TODO(stage5-collapse-probe) 가 비어 있다. starter.py 를 채운다.")
        return 1

    # --- 1. 반환 규약 -----------------------------------------------------------------
    nums = {"mu_abs", "sigma", "kl", "est_vel_std", "dec_w_vel", "dec_w_z"}
    check("1. 여섯 수치와 mechanism 을 모두 돌려준다", nums | {"mechanism"} <= set(r_mu),
          f"받은 키: {sorted(r_mu)}")
    check("1. 수치는 파이썬 float 이다 (텐서가 아니다)",
          all(isinstance(r_mu.get(k), float) for k in nums),
          f"{ {k: type(r_mu.get(k)).__name__ for k in sorted(nums)} }")
    check("1. mechanism 은 MECHANISMS 중 하나다", r_mu.get("mechanism") in mod.MECHANISMS,
          f"{r_mu.get('mechanism')!r} — 가능한 값: {mod.MECHANISMS}")

    # --- 2. 인코더 프로브는 표준정규다 ---------------------------------------------------
    # mu 16차원 중 한 줄만 x0 이고 나머지는 0 이므로  E|mu| = E|x0| / 16.
    # 균등분포나 상수 입력을 넣으면 이 값이 안 맞는다.
    check("2. 프로브 입력이 N(0,1) 이다", close(r_mu["mu_abs"], E_ABS_NORMAL / 16),
          f"|mu|={r_mu['mu_abs']:.5f} (기대 {E_ABS_NORMAL / 16:.5f})")

    # --- 3. KL 은 16차원 '합' 이다 --------------------------------------------------------
    # logvar=0, mu[0]=x0 이면  KL = 0.5 * sum mu_j^2 = 0.5 * x0^2  ->  기대값 0.5.
    check("3. KL = -0.5*sum(1+logvar-mu^2-exp(logvar))", close(r_mu["kl"], 0.5),
          f"kl={r_mu['kl']:.4f} (기대 0.500). 16 으로 나눴다면 0.031 이 나온다")
    # logvar 만 상수 0.5 로 올린 인코더:  KL = 0.5*16*(exp(.5) - 1 - .5) = 1.18977
    r_lv = probe(synth(logvar_bias=0.5), num_samples=4096, seed=0)
    check("3. logvar 항도 KL 에 들어간다", close(r_lv["kl"], 8 * (math.exp(0.5) - 1.5)),
          f"kl={r_lv['kl']:.4f} (기대 {8 * (math.exp(0.5) - 1.5):.4f})")
    check("3. sigma = exp(0.5*logvar) 의 평균", close(r_lv["sigma"], math.exp(0.25)),
          f"sigma={r_lv['sigma']:.4f} (기대 {math.exp(0.25):.4f}) — logvar 를 그대로 쓰면 0.5 다")

    # --- 4. est_vel_std 는 '샘플 축' 표준편차다 -------------------------------------------
    r_vel = probe(synth(pick_row=0), num_samples=8192, seed=0)  # est_vel[0] = x0
    check("4. est_vel_std = 샘플 축 std 의 3차원 평균", close(r_vel["est_vel_std"], 1.0 / 3),
          f"{r_vel['est_vel_std']:.4f} (기대 {1 / 3:.4f}) — 차원 축으로 std 를 내면 다른 값이 나온다")
    check("4. 출력이 상수면 est_vel_std 가 0 이다", r_mu["est_vel_std"] < 1e-6,
          f"{r_mu['est_vel_std']:.3e}")

    # --- 5. 디코더 열 노름 -----------------------------------------------------------------
    # 이 실습이 새로 재는 축이다. 인코더만 보면 두 붕괴가 구별되지 않는다.
    r_w = probe(synth(dec_vel=2.0, dec_z=0.5), num_samples=64, seed=0)
    check("5. dec_w_vel / dec_w_z 는 열 노름의 평균이다",
          close(r_w["dec_w_vel"], 2.0) and close(r_w["dec_w_z"], 0.5),
          f"|W_vel|={r_w['dec_w_vel']:.4f} (기대 2.0), |W_z|={r_w['dec_w_z']:.4f} (기대 0.5)")
    # 열 0,1,2 는 1.0, 열 3 만 10.0, 나머지 15개는 0.  올바른 3/16 분할이면 (1.0, 0.625) 다.
    # 4/15 로 잘못 가르면 (3.25, 0.0) 이 나온다.
    r_split = probe(synth(dec_cols=[1.0, 1.0, 1.0, 10.0] + [0.0] * 15), num_samples=64, seed=0)
    check("5. 앞 3열이 v̂, 뒤 16열이 z 다 (경계가 한 칸 밀리면 잡힌다)",
          close(r_split["dec_w_vel"], 1.0) and close(r_split["dec_w_z"], 10.0 / 16),
          f"|W_vel|={r_split['dec_w_vel']:.4f} (기대 1.0), "
          f"|W_z|={r_split['dec_w_z']:.4f} (기대 {10 / 16:.4f}); 4/15 로 갈랐다면 3.25 / 0.0 이다")

    # --- 6. 두 기전을 구별한다 --------------------------------------------------------------
    # 합성 값은 판정선에서 한 자릿수 넘게 떨어뜨려 둔다. 실측한 통과 범위는
    #   MU_ALIVE : 0 < x < 0.81      (붕괴 합성의 |mu| 가 정확히 0 이라 아래는 무한대)
    #   Z_USED   : 4.30e-4 < x <= 0.435  (가장 좁은 쪽이 기본값 1e-2 에서 23배 = 1.4 자릿수)
    # 이다 — "임계값과 무관"이 아니라 "여유가 한 자릿수 넘는다"가 정확한 표현이다.
    r_alive = probe(alive_encoder(), num_samples=2048, seed=0)
    check("6. mu 가 입력을 따라 움직이면 alive", r_alive["mechanism"] == "alive",
          f"{r_alive['mechanism']!r} (|mu|={r_alive['mu_abs']:.3e})")
    # 디코더는 z 를 크게 읽는데(|W_z|/|W_vel| = 0.44) 인코더 mu 가 죽은 경우 = KL 과압.
    r_kl = probe(synth(dec_vel=4.39, dec_z=1.91), num_samples=2048, seed=0)
    check("6. mu 는 죽었는데 |W_z| 가 살아 있으면 kl_overpressure",
          r_kl["mechanism"] == "kl_overpressure",
          f"{r_kl['mechanism']!r} (|mu|={r_kl['mu_abs']:.3e}, |W_z|/|W_vel|="
          f"{r_kl['dec_w_z'] / r_kl['dec_w_vel']:.3f})")
    # 디코더가 v̂ 만 읽는 경우(|W_z|/|W_vel| = 4e-4) = 디코더가 z 를 버렸다.
    r_dec = probe(synth(dec_vel=10.24, dec_z=0.0044), num_samples=2048, seed=0)
    check("6. mu 도 |W_z| 도 죽었으면 decoder_dropped_z",
          r_dec["mechanism"] == "decoder_dropped_z",
          f"{r_dec['mechanism']!r} (|mu|={r_dec['mu_abs']:.3e}, |W_z|/|W_vel|="
          f"{r_dec['dec_w_z'] / r_dec['dec_w_vel']:.2e})")
    # NaN 은 모든 부등호에서 False 라, 걸러내지 않으면 마지막 else 로 굴러떨어져
    # 있지도 않은 기전을 설명하게 된다.
    r_nan = probe(nan_checkpoint(), num_samples=64, seed=0)
    check("6. 가중치가 NaN 이면 nan 으로 이름 붙인다", r_nan["mechanism"] == "nan",
          f"{r_nan['mechanism']!r} — NaN 을 먼저 걸러내지 않으면 decoder_dropped_z 로 새어 나간다")
    # 판정 순서: mu 가 살아 있으면 디코더가 z 를 안 읽어도 붕괴가 아니다. actor 는 디코더가
    # 아니라 context 를 받기 때문이다 (|W_z| 가 죽는 것은 '앞으로 죽는다'는 조짐이다).
    #
    # 순서를 실제로 잡는 것은 아래 두 조합 중 '|mu| 살아 있음 + |W_z| 살아 있음' 뿐이다.
    # '|mu| 살아 있음 + |W_z| 죽음' 은 사다리 순서를 뒤집어도(|W_z| 분기를 |mu| 앞에 두어도)
    # 같은 답이 나와서 순서를 검사하지 못한다 — 그래서 둘 다 본다.
    r_order = probe(alive_encoder(dec_vel=4.39, dec_z=1.91), num_samples=2048, seed=0)
    check("6. 판정 순서는 mu 가 먼저다 (|W_z| 가 살아 있어도 mu 를 먼저 본다)",
          r_order["mechanism"] == "alive",
          f"{r_order['mechanism']!r} — |W_z| 분기를 |mu| 분기보다 먼저 두면 kl_overpressure 가 된다. "
          "|mu| 를 먼저 보고, 죽었을 때만 |W_z| 로 기전을 가른다")
    r_order2 = probe(alive_encoder(dec_vel=10.0, dec_z=0.001), num_samples=2048, seed=0)
    check("6. mu 가 살아 있으면 |W_z| 가 죽어도 alive 다 (아직은 조짐일 뿐이다)",
          r_order2["mechanism"] == "alive",
          f"{r_order2['mechanism']!r} — actor 가 받는 것은 디코더 출력이 아니라 context 다")

    # --- 7. 재현 가능해야 한다 -------------------------------------------------------------
    sd = synth(pick_row=3)
    a = probe(sd, num_samples=64, seed=1)
    b = probe(sd, num_samples=64, seed=1)
    c = probe(sd, num_samples=64, seed=2)
    check("7. 같은 seed 는 같은 수치를 낸다", a == b, f"{a} vs {b}")
    check("7. 다른 seed 는 다른 표본을 쓴다", a["mu_abs"] != c["mu_abs"],
          "seed 인자를 안 쓰고 상수를 하드코딩했을 수 있다")

    # 전역 RNG 를 '건드렸는가' 가 아니라 '프로브 seed 에 따라 다르게 건드렸는가' 를 본다.
    # build_encoder() 가 nn.Linear 를 만들며 전역 RNG 를 정해진 양만큼 소비하므로 before/after
    # 비교로는 정답도 FAIL 이 난다. probe 가 전역 seed 를 쓰면 두 상태가 갈린다.
    torch.manual_seed(1234)
    probe(sd, num_samples=64, seed=7)
    state_a = torch.random.get_rng_state()
    torch.manual_seed(1234)
    probe(sd, num_samples=64, seed=99)
    state_b = torch.random.get_rng_state()
    check("7. 전역 난수 상태를 오염시키지 않는다", torch.equal(state_a, state_b),
          "torch.manual_seed / torch.randn 을 전역으로 썼다 — Generator 를 쓴다")

    # --- 8. num_samples 가 진짜 표본 수다 ---------------------------------------------------
    def spread(n: int) -> float:
        vals = [probe(sd, num_samples=n, seed=s)["mu_abs"] for s in range(5)]
        return max(vals) - min(vals)

    s_small, s_big = spread(32), spread(8192)
    check("8. num_samples 를 늘리면 추정이 안정된다", s_small > 3 * s_big,
          f"seed 5개 산포: n=32 에서 {s_small:.2e}, n=8192 에서 {s_big:.2e} — "
          "표본 수를 무시하고 상수를 썼을 수 있다")

    # --- 9. 진짜 체크포인트: 찍기만 한다 (채점하지 않는다) ------------------------------------
    print("\n  --- 실제 체크포인트 (있으면 마지막 것만 찍는다. 채점 대상이 아니다) ---")
    runs = mod.find_runs()
    if not runs:
        print("  [건너뜀] logs/ 아래에서 Waq run 을 못 찾았다.")
        print("           학습 산출물은 gitignore 대상이라 repo 에 없다 — 위 합성 검사만으로 통과다.")
    for run in runs:
        ckpts = mod.checkpoints(run)
        blob = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        if "cenet_state_dict" not in blob:
            continue
        m = mod.diagnose_checkpoint(blob["cenet_state_dict"], num_samples=2048, seed=0)
        print(f"  {mod.label_for(run)}")
        print(f"    {ckpts[-1].name}: |mu|={m['mu_abs']:.3e}  |W_vel|={m['dec_w_vel']:.3f}  "
              f"|W_z|={m['dec_w_z']:.4f}  →  {m['mechanism']}")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)}개 실패\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32m전부 통과.\033[0m 같은 프로브로 자기 학습 런도 진단할 수 있다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
