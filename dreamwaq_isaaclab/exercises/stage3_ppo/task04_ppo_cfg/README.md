# task04_ppo_cfg — cfg 값의 의미와 이 repo 의 실전 이슈  `L1 · FILL`

**Stage 3 · PPO**

## 목표

task03 에서 "우리에겐 없다" 로 넘긴 **적응형 학습률**을 직접 구현한다.
그리고 이 repo 가 실제로 겪은 문제인 **log_std 폭주**를 눈으로 본다.

## 채울 곳

```
starter.py  →  adapt_learning_rate()  안의  TODO(ppo-cfg)
```

빈칸은 이것 하나다. `clamp_log_std()` 는 주어진다 — 읽고 실행만 한다.

## 적응형 학습률이 왜 필요한가

PPO 의 clipping 은 정책이 한 번에 너무 많이 변하는 것을 막는다. 그런데 **얼마나 변했는지**는
문제마다, 학습 단계마다 다르다. 고정 학습률은 초반엔 너무 크고 후반엔 너무 작다.

`rsl_rl` 은 매 미니배치마다 이전 정책과 현재 정책의 KL 을 재서 보폭을 스스로 정한다.

```
KL > desired_kl * 2    너무 많이 움직였다   ->  lr / 1.5   (하한 1e-5)
KL < desired_kl / 2    너무 적게 움직였다   ->  lr * 1.5   (상한 1e-2)
그 사이                 적당하다             ->  그대로
```

가운데 **죽은 구간**(`desired/2 ~ desired*2`)이 핵심이다. 이게 없으면 lr 이 매 스텝
올랐다 내렸다 진동한다. 이 repo 는 `desired_kl = 0.01`, `schedule = "adaptive"` 다.

## 검증

```bash
cd exercises/stage3_ppo/task04_ppo_cfg
python check.py            # 빠른 검증 — Isaac Sim 불필요, 1초
python starter.py          # cfg 값 출력 + lr 궤적 + log_std 폭주 시연
```

통과 기준 (14개):

1. `KL > desired*2` 이면 `lr / 1.5`
2. `KL < desired/2` 이면 `lr * 1.5`
3. **죽은 구간(경계 포함)에서는 lr 이 그대로** — `0.005, 0.008, 0.01, 0.015, 0.02` 다섯 값
4. 하한 `1e-5`, 상한 `1e-2` 를 넘지 않는다
5. **`KL == 0` 이면 올리지 않는다** ← `rsl_rl` 의 `kl_mean > 0.0` 조건. 빠뜨리기 쉽다
6. `desired_kl` 을 바꾸면 경계도 따라 움직인다 (하드코딩하면 걸린다)
7. 연속 적용 궤적이 `rsl_rl` 규칙과 일치
8. 부작용 없이 값을 반환한다
9. `clamp_log_std` 회귀 검사

## 힌트

<details>
<summary>경계 조건 두 개를 조심한다</summary>

- 부등호가 `>` 인가 `>=` 인가 — `rsl_rl` 은 `kl_mean > desired_kl * 2.0` 이다.
  `KL == 0.02` 는 **죽은 구간**이지 낮추는 구간이 아니다.
- `kl_mean > 0.0` 조건을 빼면, KL 이 정확히 0 일 때도 "너무 적게 움직였다" 로 판정해
  lr 을 계속 키운다. 정책이 완전히 수렴한 뒤에 lr 이 폭주하는 원인이 된다.
</details>

<details>
<summary>하한 / 상한</summary>

`max(1e-5, lr / 1.5)` 와 `min(1e-2, lr * 1.5)`. 방향을 헷갈리지 않는다 —
**내릴 때는 `max` 로 바닥을 막고, 올릴 때는 `min` 으로 천장을 막는다.**
</details>

## log_std clamp — 이 repo 가 실제로 겪은 문제

`starter.py` 를 실행하면 3번 절에서 직접 보게 된다.

actor 의 액션 분포는 학습 가능한 `log_std` 를 갖는다. 적응형 lr 과 좁은 관측 분포가
겹치면 이 값이 학습 도중 극단으로 흘러간다. 한 번 `±inf` 가 되면 `Normal` 표집이 `NaN` 을
뱉고 정책 전체가 죽는다 — **그것도 몇 시간 학습한 뒤에.**

그래서 `train.py` 가 optimizer 에 훅을 걸어 매 step 뒤에 `log_std` 를 `[-5, 0]` 으로 묶는다.

```
dreamwaq_manager/scripts/rsl_rl/train.py
  _register_std_clamp_for_all_runners()   <- optimizer.register_step_post_hook
```

상한이 `0.0` 인 이유는 원래 "액션이 `[-1, 1]` 로 clip 되므로(`clip_actions = 1.0`)
`std > 1` 은 표집 대부분이 잘려 나가 탐험에 도움이 안 된다" 였다.
**지금은 `clip_actions = 4.0` 이라 그 전제가 사라졌다** — 그 ±1 clip 이 오히려 `std` 를
상한에 붙여 놓은 **원인**이었기 때문이다 (로그확률·엔트로피는 자르기 **전** 가우시안으로
계산되므로 `σ>1` 의 추가 노이즈는 공짜인데 엔트로피 보너스는 계속 붙는다 → 상한 아래에
평형점이 없다. `PAPER.md` §6). clip 4.0 에서는 `σ*≈0.374` 로 안쪽에 평형점이 생기고,
clamp 는 더 이상 묶이지 않는 **NaN 방지 안전장치**로만 남는다.

> `dreamwaq_manager` 와 `dreamwaq_direct` 의 `train.py` 양쪽에 같은 몽키패치가 있다.

## cfg 값은 논문 값이다

`starter.py` 1번 절이 `rsl_rl_ppo_cfg.py` 에서 값을 **직접 읽어** 출력한다.
`clip=0.2`, `γ=0.99`, `λ=0.95`, `lr=1e-3`, `desired_kl=0.01`, `epochs=5`, `minibatches=4` —
임의로 고른 것이 아니라 DreamWaQ 논문 §III-B 가 명시한 값이다.

## 다음

Stage 3 완료다. [Stage 4 — CENet](../../stage4_cenet/) 으로 간다.
