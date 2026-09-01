# task02 — PPO 손실 함수  `L2 · BUILD`

## 목표

PPO 를 PPO 답게 만드는 **clipped surrogate objective** 를 만든다.

```
ratio = exp(log_prob - old_log_prob)
surrogate = -mean( min( ratio·A,  clip(ratio, 1-ε, 1+ε)·A ) )
value_loss = mean( (V - returns)² )
entropy_loss = -mean(entropy)
total = surrogate + c_v · value_loss + c_e · entropy_loss
```

## 채울 곳

`starter.py` 의 `TODO(ppo-losses)` — `ppo_losses` 본문.

**핵심은 `min` 이다**

`clip` 만 쓰면 안 된다. `min(unclipped, clipped)` 이어야 한다.

| 상황 | 왜 |
|---|---|
| `A>0`, `ratio` 가 **위로** 벗어남 | 클립 → 좋은 행동이라도 한 번에 너무 밀지 않는다 |
| `A>0`, `ratio` 가 **아래로** 벗어남 | **클립 안 함** → 되돌릴 여지를 남긴다 |
| `A<0`, `ratio` 가 아래로 벗어남 | 클립 |
| `A<0`, `ratio` 가 위로 벗어남 | 클립 안 함 |

`min` 은 "**항상 비관적인 쪽**을 택한다"는 뜻이다. 검사 8번이 정확히 이걸 잡는다.

## 검증

```bash
python check.py
```

| # | 통과 기준 |
|:---:|---|
| 1 | 네 값이 모두 스칼라 |
| 2 | `ratio=1` 일 때 `surrogate = -mean(A)` |
| 3 | `value_loss` = MSE |
| 4 | `entropy_loss = -mean(entropy)` — **부호 주의** |
| 5 | `total` 이 계수로 가중 합 |
| 6 | `A>0`, `ratio` 상한 초과 → 클립됨 |
| 7 | `A<0`, `ratio` 하한 미만 → 클립됨 |
| 8 | **`A>0`, `ratio` 하한 미만 → 클립 안 함** (`min` 을 썼다) |
| 9 | advantage 에는 클립을 걸지 않는다 |
| 10 | gradient 가 흐른다 |

## 부호 하나로 학습이 뒤집힌다

- **surrogate**: 목적함수는 `ratio·A` 를 **최대화**하는 것이지만 optimizer 는 최소화하므로 `-` 를 붙인다
- **entropy**: 엔트로피는 **크게** 만들고 싶으므로 손실에는 `-` 로 들어간다.
  부호를 뒤집으면 정책이 즉시 결정론적으로 붕괴해 탐험이 사라진다

## 다 하면

`python starter.py` 로 실제 학습이 돈다 (약 40초). 보상이 0.44 → 0.95 로 오르면 성공.
