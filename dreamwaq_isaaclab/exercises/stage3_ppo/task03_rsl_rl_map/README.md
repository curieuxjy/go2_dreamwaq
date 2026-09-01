# task03_rsl_rl_map — 내가 짠 게 여기 이 클래스다  `L0 · READ`

**Stage 3 · PPO**

## 목표

쓸 코드는 없다. **읽는다.**

task01·task02 에서 GAE 와 PPO 손실을 직접 썼다. 그런데 이 프로젝트는 `rsl_rl` 의 PPO 를
그대로 쓴다. 내가 쓴 것이 저 라이브러리의 **어디**인지 짚고 넘어가지 않으면,
앞으로 학습이 이상할 때 어디를 열어 봐야 할지 알 수 없다.

먼저 task01·task02 를 끝내고 온다. 직접 써 보지 않고 읽으면 남는 게 없다.

## 실행

```bash
cd exercises/stage3_ppo/task03_rsl_rl_map
python solution.py
```

이 스크립트는 설명을 하드코딩하지 않는다. **설치된 `rsl_rl` 의 소스를 `inspect` 로 직접 꺼내**
우리 `ppo.py` 와 나란히 출력한다. 그래서 rsl_rl 버전이 올라가면 출력도 따라 바뀐다.

## 무엇을 보게 되는가

| # | 우리 `ppo.py` | `rsl_rl` |
|:---:|---|---|
| 1 | `compute_gae` | `PPO.compute_returns` |
| 2 | `ppo_losses` | `PPO.update` 안쪽 |
| 3 | (없음) | `desired_kl` 적응형 학습률 |
| 4 | `train()` 루프 | `OnPolicyRunner.learn` |

## 읽으면서 확인할 것

<details>
<summary>1. 부호가 다른데 왜 같은 식인가</summary>

```
우리    : -min( ratio*A , clip(ratio)*A )
rsl_rl  :  max( -A*ratio , -A*clip(ratio) )
```

`-min(x, y) == max(-x, -y)` 다. 둘 다 "최대화하고 싶은 목적함수에 음수를 붙여
최소화 문제로 바꾼" 것뿐이다. 논문의 수식은 최대화 형태로 쓰여 있고,
optimizer 는 최소화만 하므로 어딘가에서 한 번 뒤집힌다.
</details>

<details>
<summary>2. clipped value loss — 우리에겐 없는 가지</summary>

우리는 `(value - returns)²` 만 썼다. `rsl_rl` 은 `use_clipped_value_loss` 가 켜져 있으면
critic 예측도 **롤아웃 당시 값 기준 ±clip_param** 으로 제한한다.

이 repo 의 cfg 는 `use_clipped_value_loss=True` 이므로 **실제로는 이 가지를 탄다.**
critic 이 한 번의 업데이트로 너무 멀리 뛰는 것을 막는 장치다.
</details>

<details>
<summary>3. 어드밴티지 정규화를 어디서 하는가</summary>

우리는 `train()` 에서 평탄화한 뒤 전체 배치로 정규화했다.
`rsl_rl` 은 `compute_returns` 끝에서 한다 (`normalize_advantage_per_mini_batch` 가 꺼져 있을 때).

위치만 다르고 수식은 같다. 다만 **미니배치별로** 정규화하는 옵션도 있는데,
그 경우 통계가 배치마다 달라져 결과가 미세하게 달라진다.
</details>

<details>
<summary>4. learn() 이 79 줄인데 알고리즘은 3 줄이다</summary>

`OnPolicyRunner.learn` 의 대부분은 로깅·체크포인트·멀티GPU 처리다.
알고리즘 골격은 우리 `train()` 과 똑같이 세 줄이다.

```
alg.act() / alg.process_env_step()   <- 롤아웃
alg.compute_returns(obs)             <- GAE
alg.update()                         <- epoch x minibatch
```

**이 사실이 Stage 4 로 이어진다.** 이 프로젝트의 `OnPolicyRunnerWaq` 는 바로 이
`learn()` 을 오버라이드해서, 롤아웃 루프 한가운데에 CENet 호출을 끼워 넣은 것이다.
알고리즘을 바꾼 게 아니라 **관측을 바꾼** 것이다.
</details>

## 통과 기준

`python solution.py` 가 `[PASS] 대조 완료` 로 끝나면 된다. 검사할 코드가 없으므로
`check.py` 도 없다.

## 다음

- [`task04_ppo_cfg`](../task04_ppo_cfg/) — 위 3번(적응형 학습률)을 직접 구현한다 (L1)
