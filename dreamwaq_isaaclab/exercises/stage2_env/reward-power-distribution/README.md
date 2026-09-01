# reward-power-distribution — 파워 분포 페널티  `L2 · BUILD`

## 목표

**특정 모터에만 부하가 쏠리는 것**을 벌한다.

```
reward = var_j(tau_j * theta_dot_j)^2
```

논문 Table I 의 **Power distribution**. 앞 실습(`joint_power`)이 파워의 **총량**을 벌한다면,
이 항은 관절 사이의 **불균형**을 벌한다.

## 채울 곳

`starter/rewards.py` 의 `TODO(reward-power-distribution)` — `power_distribution_l2` 본문.

**앞 실습과 다른 점 둘**

1. **절댓값을 쓰지 않는다** — 부호 있는 파워 그대로의 분산이다
2. 합이 아니라 **관절 축(`dim=-1`)의 분산**을 보고, 그것을 **한 번 더 제곱**한다

## 검증

```bash
python check.py
```

| # | 통과 기준 |
|:---:|---|
| 1 | 반환이 `(num_envs,)` |
| 2 | 모든 관절 파워가 같으면 0 |
| 3 | **파워가 커도 균등하면 0** — 총량이 아니라 불균형을 본다 |
| 4 | 알려진 분포에서 `var²` 값이 정확 |
| 5 | **분산을 한 번 더 제곱했다** (`torch.square` 누락 탐지) |
| 6 | **절댓값을 쓰지 않았다** |
| 7 | `asset_cfg.joint_ids` 존중 |

3번이 이 항의 정의 그 자체다. 5·6은 흔한 실수를 직접 겨냥한다.

## 왜 이 항이 있는가 — 논문의 설명

논문 §II-B-4:

> However, this reward minimizes the overall power without considering each motor's power usage
> balance. Consequently, in the long run, some motors might overheat faster than others.
> Therefore, we introduced a power distribution reward to **reduce motor overheating in the real
> world** by penalizing motors' power with high variance over all motors used on the robot.

시뮬레이터 안에서는 모터가 타지 않는다. **이 항은 실기 배포를 위한 것이다.**
논문 §III-G의 실외 실험에서 언덕을 오를 때 "front legs' motors may easily overheat and
enter the overheat protection mode" 라고 보고한 것과 이어진다.

## 가중치에 얽힌 이야기 (읽어둘 것)

이 항의 가중치는 **논문과 저자 구현이 10배 다르다.**

| | 값 |
|---|---|
| 논문 Table I | `−10⁻⁵` |
| 저자의 `legged_gym` 설정 (`a1_config.py`) | `−1.0e−6` |

어느 쪽이 맞는가? 이 repo 는 처음에 논문 값을 썼다가 **재고 나서 `−1e−6` 으로 바꿨다.**
근거가 셋이다.

1. **나머지 11항은 소수점까지 일치한다.** 그 설정 클래스에는 `"""SAME reward functions
   with the paper"""` 라는 주석까지 붙어 있다. 12항 중 1항만 정확히 10배 어긋나는 것은
   설계 변경보다 **인쇄 오타**로 읽는 쪽이 자연스럽다.
2. **크기를 실제로 재 봤다.** 512 env × 300 step 프로브에서 가중치를 적용한 per-step `|r|`:

   | 항 | 평균 | 최대 |
   |---|---:|---:|
   | `track_lin_vel_xy_exp` (추종, 이 env 의 주 목적) | 14.2 | 75 |
   | `power_distribution` @ `−1e−5` | **16.2** | **8447** |
   | `power_distribution` @ `−1e−6` | 1.62 | 845 |

   `−1e−5` 에서는 **페널티가 주 목적보다 크다.** 정규화 항이 아니라 지배 항이다.
   `var(τ·θ̇)²` 는 토크·속도의 **4제곱** 스케일이라 이렇게 폭주한다.
3. 실제로 `−1e−5` 로 학습하면 지형 커리큘럼이 오르지 않았다 (`PAPER.md` §2).

**교훈은 "논문이 맞다"도 "코드가 맞다"도 아니다.** 둘이 다르면 **어느 쪽이 맞는지 재 보라**는
것이다. 여기서는 항의 크기를 다른 항과 나란히 재 보는 것만으로 답이 나왔다.
논문의 표를 그대로 옮기면 재현되리라는 가정이 이 프로젝트에서 여러 번 깨졌다.
