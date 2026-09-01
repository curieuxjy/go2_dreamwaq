# reward-joint-power — 관절 파워 페널티  `L1 · FILL`

## 목표

로봇이 **에너지를 낭비하지 않게** 만드는 항.

```
reward = sum_j |tau_j| * |theta_dot_j|
```

논문 Table I 의 **Joint power**, 가중치 `−2×10⁻⁵`.

## 채울 곳

`starter/rewards.py` 의 `TODO(reward-joint-power)` — `joint_power_l1` 본문.

| 읽을 것 | 뜻 |
|---|---|
| `asset.data.applied_torque` | 실제로 관절에 가해진 토크 |
| `asset.data.joint_vel` | 관절 각속도 |
| `asset_cfg.joint_ids` | 대상 관절 (전체가 아닐 수 있다) |

둘 다 **절댓값**을 취한다 — 파워를 "소비량"으로 보므로 부호는 의미가 없다.

## 검증

```bash
python check.py
```

| # | 통과 기준 |
|:---:|---|
| 1 | 반환이 `(num_envs,)` |
| 2 | 토크=1, 각속도=1, 관절 12개 → `12` |
| 3 | 토크 0 이면 0 |
| 4 | **각속도 0 이면 0** — 정지 유지 토크는 이 항으로 벌하지 않는다 |
| 5 | **부호와 무관** (절댓값을 썼다) |
| 6 | `asset_cfg.joint_ids` 존중 |

## 왜 이 항이 있는가

논문 §II-B-4:

> The complex reward function for learning a locomotion policy usually includes a motor power
> minimization term.

4번이 흥미롭다 — 각속도가 0이면 아무리 큰 토크를 써도 페널티가 0이다. **가만히 버티는 것은
이 항으로 벌하지 않는다.** 그래서 다음 실습의 `power_distribution` 이 따로 필요해진다.

## 다음

[`reward-power-distribution`](../reward-power-distribution/) — 총량이 아니라 **불균형**을 벌하는 항.
