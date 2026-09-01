#!/usr/bin/env python3
"""Stage 5 · task04 — CENet 잠재가 죽었는지, 죽었다면 **어느 쪽이 먼저 죽었는지** 판정한다.  `L2 · BUILD`

이 프로젝트의 CENet 잠재변수 z(16차원)는 **두 번 붕괴했고, 두 번의 기전이 서로 달랐다.**

    (a) KL 과압        디코더는 여전히 z 를 읽으려는데(|W_z| 가 크다) 인코더가 얼어붙는다.
                       KL 항의 압력이 recon 이 z 로 보내는 신호보다 셀 때 이렇게 된다.
    (b) 디코더가 z 를   v̂ 는 vel_loss 로 직접 지도학습되므로 재구성 경쟁에서 z 를 이긴다.
        버린다         디코더가 v̂ 만 읽기 시작하면 z 로 가는 recon 그래디언트가 사라지고,
                       그러면 약한 KL 로도 z 가 죽는다.

둘 다 겉으로는 똑같이 보인다 — `|mu| → 0`, `sigma → 1`, `KL → 0`. **인코더만 보면 구별할 수
없다.** 처방이 정반대인데도 그렇다: (a)는 β 를 낮춰야 하고, (b)는 β 를 낮춰도 소용이 없다
(z 에게 할 일을 만들어 줘야 한다). 그래서 이 실습은 **디코더 1층 가중치를 함께 잰다.**

    디코더 1층 가중치 W 의 열 j  =  디코더가 입력 j 에서 읽어 가는 양
    입력은 [v̂(3), z(16)] 이므로  |W_vel| = 앞 3열, |W_z| = 뒤 16열

Isaac Sim 도 시뮬레이터도 필요 없다. 체크포인트의 `cenet_state_dict` 하나면 순수 torch 로
끝난다.

    python starter.py                    # 자산이 풀려 있으면 Waq 런들을 찾아 훑는다
    python starter.py --run <경로>        # run 폴더를 직접 준다 (여러 번 줄 수 있다)
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import torch
import torch.nn as nn

# CENet 구조 (algorithms/cenet.py 와 같아야 한다)
INPUT_DIM = 225  # obs_history = 45 x H(5)
HIDDEN1, HIDDEN2 = 128, 64
NUM_VEL, NUM_LATENT = 3, 16
OUT_DIM = NUM_VEL + 2 * NUM_LATENT  # 35 = est_vel(3) + mu(16) + logvar(16)

# 판정선 두 개. 둘 다 **읽기 보조**이지 진리가 아니다 — 왜 임계값으로 건강을 단정하면
# 안 되는지는 README 의 '임계값으로 건강을 단정하지 않는다' 절에 적어 두었다.
MU_ALIVE = 1e-2  # |mu| 가 이 위면 인코더가 입력에 반응하고 있다
Z_USED = 1e-2  # |W_z| / |W_vel| 가 이 위면 디코더가 아직 z 를 읽고 있다

MECHANISMS = ("alive", "kl_overpressure", "decoder_dropped_z", "nan")
VERDICTS = {
    "alive": "살아 있다 — 인코더가 입력에 반응한다",
    "kl_overpressure": (
        "붕괴 (KL 과압) — 디코더는 아직 z 를 원하는데(|W_z| 가 살아 있다) 인코더 mu 가 눌렸다. "
        "β 를 낮추는 것이 처방이다"
    ),
    "decoder_dropped_z": (
        "붕괴 (디코더가 z 를 버렸다) — |W_z| 가 0 으로 죽고 v̂ 쪽만 커졌다. "
        "β 를 낮춰도 안 산다. z 에게 할 일이 있어야 한다"
    ),
    "nan": (
        "CENet 가중치가 NaN 이다 — 잠재는 논할 것도 없다. 러너가 actor 관측을 "
        "nan_to_num 으로 0 으로 바꿔 넣으므로 보상 곡선은 멀쩡해 보인다"
    ),
}

REPO_ROOT = Path(__file__).resolve().parents[3]
LOGS = REPO_ROOT / "dreamwaq_manager" / "logs"
# 어디를 뒤질지만 정해 둔다. **어느 런이 정상인지는 여기에 적지 않는다** — 그것이 이 스크립트가
# 판정할 대상이지 미리 아는 값이 아니다. 없는 폴더는 조용히 건너뛴다.
SEARCH_ROOTS = [
    LOGS / "rsl_rl",
    LOGS / "_archive_4000it_collapsedwaq",
    LOGS / "_archive_4000it_clip1_norew",
    LOGS / "_archive_3000it_stairheavy" / "rsl_rl",
]


def build_encoder(cenet_state_dict: dict) -> nn.Sequential:
    """체크포인트의 `cenet_state_dict` 에서 인코더만 되살린다."""
    enc = nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN1), nn.ELU(),
        nn.Linear(HIDDEN1, HIDDEN2), nn.ELU(),
        nn.Linear(HIDDEN2, OUT_DIM),
    )
    weights = {k[len("encoder."):]: v for k, v in cenet_state_dict.items() if k.startswith("encoder.")}
    if not weights:
        raise KeyError("state_dict 에 'encoder.*' 키가 없다 — CENet 체크포인트가 맞는지 확인한다")
    enc.load_state_dict(weights)
    enc.eval()
    return enc


def decoder_input_weight(cenet_state_dict: dict) -> torch.Tensor:
    """디코더의 **첫 Linear** 가중치 `[hidden, 19]` 를 꺼낸다.

    보통 `decoder.0.weight` 지만 인덱스를 못 박지 않는다 — 실험용 게이트가 앞에 끼면
    번호가 하나씩 밀린다. 번호가 가장 작은 2차원 `decoder.*.weight` 를 고른다.
    """
    found = [
        (int(m.group(1)), k)
        for k, v in cenet_state_dict.items()
        if (m := re.fullmatch(r"decoder\.(\d+)\.weight", k)) and v.ndim == 2
    ]
    if not found:
        raise KeyError("state_dict 에 'decoder.<i>.weight' 가 없다 — CENet 체크포인트가 맞는지 확인한다")
    return cenet_state_dict[min(found)[1]]


def diagnose_checkpoint(cenet_state_dict: dict, num_samples: int = 2048, seed: int = 0) -> dict:
    """체크포인트 하나를 진단한다 — 인코더 민감도 + 디코더 사용량 + 기전 판정.

    Args:
        cenet_state_dict: 체크포인트의 `cenet_state_dict`.
        num_samples: 인코더에 넣을 프로브 개수. 많을수록 통계가 안정된다.
        seed: 프로브 난수 시드. **같은 시드는 같은 수치를 내야 한다** — 판정을 재현할 수 있어야
            하고, 전역 난수 상태를 건드려서도 안 된다.

    Returns:
        {# --- 인코더: 입력을 바꾸면 출력이 움직이는가 ---------------------------------
         "mu_abs":      mu 절대값의 평균          붕괴하면 0 으로 눌린다  (핵심 지표)
         "sigma":       exp(0.5*logvar) 의 평균    붕괴하면 1 로 눌린다
         "kl":          KL(q||N(0,I)), 16차원 **합**, 샘플 평균, 단위 nats. 붕괴하면 0
         "est_vel_std": 추정 속도 3차원의 표준편차 평균. 붕괴하면 입력이 변해도 안 움직인다
         # --- 디코더: 무엇을 읽고 있는가 ----------------------------------------------
         "dec_w_vel":   디코더 1층 가중치의 **앞 3열** 열노름 평균 (v̂ 를 읽는 양)
         "dec_w_z":     디코더 1층 가중치의 **뒤 16열** 열노름 평균 (z 를 읽는 양)
         # --- 둘을 합쳐 기전을 이름 붙인다 --------------------------------------------
         "mechanism":   MECHANISMS 중 하나 (문자열)}

    `mechanism` 은 네 가지다 — `alive` / `kl_overpressure` / `decoder_dropped_z` / `nan`.
    마지막 것이 있는 이유는 실제로 있었기 때문이다: `_archive_3000it_stairheavy` 의 flat Waq 런은
    it 500 부터 CENet 14개 텐서가 **전부 NaN** 인데 보상은 35 까지 올랐다. 러너가 actor 관측을
    `nan_to_num(..., nan=0.0)` 으로 씻어 넣기 때문이다 (`dreamwaq_runner.py:203`) — 즉 그 런은
    3000 iteration 내내 "Base + 0 이 19개" 였다. NaN 을 먼저 걸러내지 않으면 이 런이
    `decoder_dropped_z` 로 이름 붙어 없는 기전을 설명하게 된다.

    왜 진짜 관측이 아니라 N(0,1) 프로브인가: 재는 것이 "정확도"가 아니라 **입력 민감도**이기
    때문이다. 관측은 정규화되어 들어가므로 N(0,1) 은 그럴듯한 입력 분포이고, 시뮬레이터 없이
    어떤 체크포인트에도 같은 잣대를 댈 수 있다.

    왜 디코더는 프로브가 아니라 가중치인가: 디코더가 입력 j 에서 읽어 가는 양은 1층 가중치의
    **j 번째 열 노름**에 전부 들어 있다. 열이 0 이면 그 입력은 뒷단에 아무 영향도 못 준다.
    """
    # ── TODO(stage5-collapse-probe) ─ level L2 ─────────────────────────────
    # CENet 체크포인트 하나에서 인코더 민감도와 디코더 사용량을 재고, 두 붕괴 기전을 구별한다
    #   hint: 난수는 전역 torch.manual_seed 가 아니라 seed 로 만든 torch.Generator 에서 뽑는다 — 판정이 남의 난수 상태를 바꾸면 안 된다
    #   hint: 수치는 전부 파이썬 float, mechanism 만 문자열이다 (텐서를 그대로 돌려주면 검사기가 잡는다)
    #   나머지 힌트는 README.md 의 '힌트' 절에 접혀 있다 — 먼저 안 보고 해 본다.
    # 통과 기준은 이 실습 폴더의 README.md 를 본다.
    raise NotImplementedError("TODO(stage5-collapse-probe)")
    # ─────────────────────────────────────────────────────────────────────


def probe_encoder(cenet_state_dict: dict, num_samples: int = 2048, seed: int = 0) -> dict[str, float]:
    """인코더 4지표만 추린다. `PAPER.md` §4 가 프로브 정의를 이 이름으로 인용한다."""
    m = diagnose_checkpoint(cenet_state_dict, num_samples, seed)
    return {k: m[k] for k in ("mu_abs", "sigma", "kl", "est_vel_std")}


def checkpoints(run_dir: Path) -> list[Path]:
    """run 폴더의 `model_<iter>.pt` 를 iteration 순으로 준다.

    파일명 정렬은 쓰지 않는다 — 문자열로 정렬하면 `model_1000` 이 `model_500` 보다 앞선다.
    """
    found = [(int(m.group(1)), p) for p in run_dir.glob("model_*.pt")
             if (m := re.fullmatch(r"model_(\d+)", p.stem))]
    return [p for _, p in sorted(found)]


def probe_run(run_dir: Path, num_samples: int = 2048, seed: int = 0) -> list[tuple[int, dict]]:
    """run 폴더의 모든 체크포인트를 훑는다. [(iteration, 지표), ...]"""
    trace = []
    for ckpt in checkpoints(run_dir):
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "cenet_state_dict" not in blob:
            continue  # Base / Oracle 런에는 CENet 이 없다
        trace.append((int(blob.get("iter", -1)), diagnose_checkpoint(blob["cenet_state_dict"], num_samples, seed)))
    return trace


def verdict(trace: list[tuple[int, dict]]) -> str:
    """마지막 체크포인트로 판정한다.

    **왜 마지막인가**: 붕괴는 초기화 직후가 아니라 학습 도중에 일어난다. 보관된 붕괴 런들은
    it 0 에서 |mu| 가 3e-2 ~ 1e-1 로 멀쩡했다. 첫 체크포인트로 판정했다면 전부 정상이라고
    썼을 것이다 — task02 의 '평균 창' 교훈과 같은 종류의 실수다.
    """
    if not trace:
        return "판정 불가 — CENet 체크포인트가 없다"
    return VERDICTS[trace[-1][1]["mechanism"]]


def find_runs(roots: list[Path] | None = None) -> list[Path]:
    """`<root>/<experiment_name>/<timestamp>/` 중 CENet 이 있을 법한 run 폴더를 모은다.

    experiment 이름에 `Waq` 가 든 것만 본다 (Base/Oracle 체크포인트에는 CENet 이 없다).
    experiment 당 가장 최근 timestamp 하나씩.
    """
    out: list[Path] = []
    for root in roots or SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for exp in sorted(root.iterdir()):
            if not exp.is_dir() or "Waq" not in exp.name:
                continue
            stamps = sorted(p for p in exp.iterdir() if p.is_dir() and any(p.glob("model_*.pt")))
            if stamps:
                out.append(stamps[-1])
    return out


def label_for(run_dir: Path) -> str:
    try:
        return str(run_dir.relative_to(LOGS))
    except ValueError:
        return str(run_dir)


def report(run_dir: Path, num_samples: int, seed: int) -> None:
    print(f"\n=== {label_for(run_dir)}")
    if not run_dir.is_dir():
        print("    [건너뜀] 폴더가 없다 (README.md 의 '자산' 절 참조)")
        return
    trace = probe_run(run_dir, num_samples, seed)
    if not trace:
        print("    [건너뜀] `cenet_state_dict` 를 가진 체크포인트가 없다")
        return
    print(f"    {'iter':>6} {'|mu|':>10} {'sigma':>8} {'KL(nats)':>10} "
          f"{'|W_vel|':>9} {'|W_z|':>9} {'W_z/W_vel':>10}  기전")
    for it, m in trace:
        ratio = m["dec_w_z"] / m["dec_w_vel"] if m["dec_w_vel"] else float("nan")
        print(f"    {it:>6} {m['mu_abs']:>10.3e} {m['sigma']:>8.4f} {m['kl']:>10.3e} "
              f"{m['dec_w_vel']:>9.4f} {m['dec_w_z']:>9.4f} {ratio:>10.2e}  {m['mechanism']}")
    print(f"    판정: {verdict(trace)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="CENet 체크포인트 포렌식")
    ap.add_argument("--run", type=Path, action="append", default=None,
                    help="run 폴더. 여러 번 줄 수 있다. 생략하면 logs/ 아래에서 Waq 런을 찾는다")
    ap.add_argument("--num-samples", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = args.run or find_runs()
    if not runs:
        print("검사할 run 이 없다. 학습 산출물을 dreamwaq_manager/logs/ 아래에 풀거나")
        print("  --run <폴더> 로 직접 지정한다 (README.md 의 '자산' 절).")
        return 0

    for run in runs:
        report(run, args.num_samples, args.seed)

    print("\n|mu| 하나로는 두 붕괴가 같아 보인다. 갈라 주는 것은 디코더의 |W_z| 다 (PAPER.md §4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
