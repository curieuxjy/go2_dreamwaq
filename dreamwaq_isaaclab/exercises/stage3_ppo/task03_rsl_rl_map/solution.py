#!/usr/bin/env python3
"""Stage 3 · task03 — 내가 짠 PPO 가 rsl_rl 의 어디인가.  `L0 · READ`

쓸 코드는 없다. 실행하면 **설치된 rsl_rl 의 실제 소스**를 꺼내 우리 `ppo.py` 와 나란히 보여준다.
직접 짜 본 뒤에 읽어야 의미가 있으므로 task01·task02 를 먼저 끝내고 온다.

    python solution.py
"""
from __future__ import annotations

import inspect
import sys
import textwrap

C_HEAD = "\033[1;36m"
C_OURS = "\033[1;33m"
C_RSL = "\033[1;32m"
C_OFF = "\033[0m"


def rule(title: str) -> None:
    print(f"\n{C_HEAD}{'─' * 78}\n {title}\n{'─' * 78}{C_OFF}")


def snippet(src: str, start: str, end: str | None = None, max_lines: int = 24) -> str:
    """소스에서 start 앵커부터 end 앵커 직전까지 잘라낸다.

    줄 번호가 아니라 앵커 문자열로 찾으므로 rsl_rl 버전이 조금 달라도 웬만하면 따라간다.
    """
    lines = src.splitlines()
    try:
        i = next(n for n, ln in enumerate(lines) if start in ln)
    except StopIteration:
        return f"  (이 rsl_rl 버전에서 '{start}' 를 찾지 못했다 — 소스를 직접 열어 본다)"
    j = len(lines)
    if end is not None:
        for n in range(i + 1, len(lines)):
            if end in lines[n]:
                j = n
                break
    return textwrap.dedent("\n".join(lines[i:min(j, i + max_lines)]))


def show(label: str, color: str, body: str) -> None:
    print(f"\n{color}[{label}]{C_OFF}")
    for ln in body.splitlines():
        print(f"    {ln}")


def main() -> int:
    try:
        import rsl_rl
        from rsl_rl.algorithms import PPO
        from rsl_rl.runners import OnPolicyRunner
    except ImportError:
        print("rsl_rl 을 import 하지 못했다. Isaac Sim 번들 python 으로 실행한다:")
        print("  ~/IsaacLab/_isaac_sim/python.sh solution.py")
        return 1

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    import ppo as ours  # noqa: PLC0415  — 우리 최소 구현

    print(f"rsl_rl 위치 : {rsl_rl.__file__}")
    print(f"우리 구현   : {ours.__file__}")

    ppo_src = inspect.getsource(PPO)

    # ── 1. GAE ────────────────────────────────────────────────────────────────
    rule("1. GAE — compute_gae()  ↔  PPO.compute_returns()")
    show("우리 ppo.py · compute_gae", C_OURS,
         snippet(inspect.getsource(ours.compute_gae), "last_advantage = torch.zeros_like", "returns = advantages"))
    show("rsl_rl · PPO.compute_returns", C_RSL,
         snippet(ppo_src, "advantage = 0", "# Compute the advantages"))
    print("""
    같은 재귀다. 이름만 다르다.

      우리                    rsl_rl
      ----                    ------
      last_advantage          advantage
      not_done                next_is_not_terminal
      next_value              next_values
      returns = adv + values  st.returns[step] = advantage + st.values[step]

    한 가지 다른 점: rsl_rl 은 어드밴티지 정규화를 compute_returns 안에서 끝낸다
    (우리는 train() 에서 평탄화한 뒤에 했다). 수식은 같다.""")

    # ── 2. 손실 ────────────────────────────────────────────────────────────────
    rule("2. 손실 — ppo_losses()  ↔  PPO.update() 안쪽")
    show("우리 ppo.py · ppo_losses", C_OURS,
         snippet(inspect.getsource(ours.ppo_losses), "ratio = torch.exp", "# ex:end"))
    show("rsl_rl · PPO.update (surrogate + value)", C_RSL,
         snippet(ppo_src, "# Surrogate loss", "# Symmetry loss", max_lines=18))
    print(f"""
    {C_OURS}부호 규약이 겉보기에 다르지만 같은 식이다.{C_OFF}

      우리    : -min( ratio*A , clip(ratio)*A )
      rsl_rl  :  max( -A*ratio , -A*clip(ratio) )

    -min(x, y) == max(-x, -y) 이므로 완전히 동치다. 둘 다 "최대화할 목적함수에
    음수를 붙여 최소화 문제로 바꾼" 것이다.

    {C_OURS}진짜 다른 점 두 가지.{C_OFF}

      1. clipped value loss — rsl_rl 은 use_clipped_value_loss 가 True 면
         critic 예측도 old value 기준 ±clip_param 으로 제한한다.
         우리 cfg 는 use_clipped_value_loss=True 이므로 실제로는 이 가지를 탄다.
      2. entropy 부호 — rsl_rl 은 loss 에서 빼고(- entropy_coef * entropy)
         우리는 미리 음수로 만들어 더했다(entropy_loss = -entropy.mean()). 같은 결과다.""")

    # ── 3. 적응형 학습률 ────────────────────────────────────────────────────────
    rule("3. 우리에게 없는 것 — desired_kl 적응형 학습률")
    show("rsl_rl · PPO.update (adaptive lr)", C_RSL,
         snippet(ppo_src, "if self.desired_kl is not None", "# Surrogate loss", max_lines=20))
    print("""
    우리 toy PPO 는 lr 을 1e-3 으로 고정했다. rsl_rl 은 매 미니배치마다 이전 정책과
    현재 정책의 KL 을 재서 학습률을 스스로 조절한다.

      KL 이 목표의 2배보다 크다  -> 너무 많이 움직였다 -> lr 을 1.5 로 나눈다
      KL 이 목표의 절반보다 작다 -> 너무 적게 움직였다 -> lr 에 1.5 를 곱한다

    이것이 다음 실습(task04)의 주제다.""")

    # ── 4. 학습 루프 ────────────────────────────────────────────────────────────
    rule("4. 학습 루프 — train()  ↔  OnPolicyRunner.learn()")
    print(f"""
    {C_OURS}우리 train(){C_OFF}                        {C_RSL}rsl_rl{C_OFF}
    ------------------------------      ------------------------------
    for it in range(iterations)         OnPolicyRunner.learn()
      1. 롤아웃 수집                      alg.act() / alg.process_env_step()
      2. compute_gae(...)                alg.compute_returns(obs)
      3. epoch x minibatch 업데이트        alg.update()
      history.append(mean_reward)        logging / wandb / tensorboard

    OnPolicyRunner.learn 은 {len(inspect.getsource(OnPolicyRunner.learn).splitlines())} 줄이다.
    대부분이 로깅·체크포인트·멀티GPU 처리이고, 알고리즘 자체는 위 세 줄이 전부다.

    이 프로젝트의 OnPolicyRunnerWaq 는 바로 이 learn() 을 오버라이드해서
    롤아웃 루프 한가운데에 CENet 을 끼워 넣은 것이다 (Stage 4 참고).""")

    # ── 5. 대응표 ───────────────────────────────────────────────────────────────
    rule("5. 한눈에")
    print("""
    ppo.py (우리)          rsl_rl                                   파일
    ---------------------  ---------------------------------------  ----------------------
    ActorCritic            MLPModel actor / critic (별도 인스턴스)    modules/
    log_std 파라미터        GaussianDistributionCfg(std_type="log")   isaaclab_rl cfg
    compute_gae            PPO.compute_returns                       algorithms/ppo.py
    ppo_losses             PPO.update 안쪽                            algorithms/ppo.py
    train() 루프            OnPolicyRunner.learn                      runners/on_policy_runner.py
    buf_* 리스트            RolloutStorage                            storage/rollout_storage.py
    (없음)                  desired_kl 적응형 lr                       algorithms/ppo.py
    (없음)                  clipped value loss                        algorithms/ppo.py

    [PASS] 대조 완료. 이어서 task04_ppo_cfg 로 간다.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
