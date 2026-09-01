# Stage 3 — 학습 알고리즘 PPO

이 프로젝트는 `rsl_rl` 의 PPO 를 **그대로 쓴다.** 직접 구현한 것이 없다.
그래서 이 stage 는 **분리해서** 다룬다.

```
1) 최소 PPO 를 직접 구현      (L2)  <- Isaac Sim 없이 도는 독립 스크립트
2) rsl_rl PPO 를 읽고 매핑     (L0)  <- "내가 짠 게 여기 이 클래스다"
3) cfg 하이퍼파라미터 실습     (L1)
```

## toy 환경 — 로코모션의 축소판

CartPole 같은 무관한 문제 대신, **DreamWaQ 와 같은 "명령 속도 추종"** 문제를 쓴다.
보상 함수까지 논문 Table I 과 같은 형태다.

| toy (`toy_env.py`) | DreamWaQ |
|---|---|
| `v` (1D 속도) | `base_lin_vel` |
| `v_cmd` | `velocity_commands` |
| `a` (가속도) | 관절 목표 오프셋 (`action * 0.25`) |
| `exp{-4 e²}` | `track_lin_vel_xy_exp` (`std=√0.25`) |
| `drag` | 지면 마찰 / 중력 |

```bash
python toy_env.py
```

```
완벽 추종 보상 : 1.0000
정지 시 보상   : 0.4426   <- 학습이 이걸 못 넘으면 실패
랜덤 정책 평균 : 0.5333
```

**`0.4426` 이 이 stage 의 하한선이다.** DreamWaQ 에서 "서 있기가 최적" 이 되는 실패와
같은 발상 — 정책이 아무것도 배우지 않으면 이 근처에 머문다.

## 실습 목록

| # | 실습 | 레벨 | 검사 |
|:---:|---|:---:|:---:|
| 01 | [`task01_gae`](task01_gae/) | **L2** | 8 |
| 02 | [`task02_ppo_update`](task02_ppo_update/) | **L2** | 10 |
| 03 | [`task03_rsl_rl_map`](task03_rsl_rl_map/) | **L0** | — |
| 04 | [`task04_ppo_cfg`](task04_ppo_cfg/) | **L1** | 14 |

01·02 는 `ppo.py` 한 파일의 서로 다른 부분을 채운다. `starter.py` 는 자기 실습
부분만 비어 있고 나머지는 정답이 들어 있으므로, **어느 쪽부터 해도 된다.**

03 은 읽기만 한다 — 01·02 를 끝낸 뒤에 해야 의미가 있다. 04 는 03 에서 "우리에게
없는 것" 으로 넘긴 적응형 학습률을 직접 채운다. **01·02 → 03 → 04 순서를 지킨다.**

## 다 채우면 실제로 학습이 돌아간다

```bash
cd task01_gae && python starter.py
```

```
정지 시 보상(하한선) : 0.4426

  iter    0 | 평균 보상 0.4379 | std 1.011
  iter   20 | 평균 보상 0.8670 | std 0.944
  ...
  iter  199 | 평균 보상 0.9562 | std 0.727

최종 평균 보상 : 0.9455  (하한선 0.4426, 완벽 추종 1.0)
학습 성공
```

CPU 로 40초쯤 걸린다. `std` 가 1.0 에서 서서히 줄어드는 것도 같이 본다 —
정책이 확신을 갖게 되면서 탐험을 줄이는 과정이다. (Stage 2 의 log_std 이야기와 이어진다.)

## 실제 프로젝트와의 대응

`ppo.py` 는 규모만 줄였을 뿐 골격이 같다.

| `ppo.py` | 실제 프로젝트 |
|---|---|
| `ActorCritic` 64-64 | actor/critic 512-256-128 (`rsl_rl_ppo_cfg.py`) |
| `log_std` 파라미터 | `init_noise_std=1.0` + `train.py` 의 log_std clamp |
| `compute_gae` | `rsl_rl` `RolloutStorage.compute_returns` |
| `ppo_losses` | `rsl_rl` `PPO.update` |
| `train()` 루프 | `OnPolicyRunner.learn()` |
| `clip=0.2, γ=0.99, λ=0.95, lr=1e-3` | **논문 §III-B 와 동일** |

마지막 줄이 중요하다 — toy 스크립트의 PPO 하이퍼파라미터는 임의로 고른 것이 아니라
논문이 명시한 값이다.
