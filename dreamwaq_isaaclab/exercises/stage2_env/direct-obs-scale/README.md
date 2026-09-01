# direct-obs-scale — 관측 스케일링  `L1 · FILL`

**Stage 2 에서 유일하게 Direct 스택을 다루는 실습이다.** Manager 는 관측을 `ObsTerm`
목록으로 선언하지만, Direct 는 `_get_observations` 안에서 텐서를 직접 조립한다.
같은 일을 두 API 가 어떻게 다르게 하는지 보는 자리다.

## 목표

관측의 각 그룹을 **O(1) 크기로 맞춘다.**

```
lin_vel   x 2.0     joint_pos x 1.0
ang_vel   x 0.25    joint_vel x 0.05
gravity, actions 는 이미 O(1) — 스케일하지 않는다
```

## 채울 곳

`starter/dreamwaq_env.py` 의 `TODO(direct-obs-scale)` — 네 줄이다.

## 왜 이게 중요한가 — 실제로 났던 버그

`debugging.qmd` 가 기록한 포팅 버그 #1 이다.

> **관측 스케일 누락** — raw `dof_vel`(±20 rad/s)이 명령(±1)을 압도

숫자로 보면 명확하다. 스케일하지 않으면 관측 45차원 중 12개(관절 속도)가
**명령보다 20배 큰 값**을 갖는다. 신경망 입장에서 명령은 거의 보이지 않는 잡음이 된다.

결과는 **정책이 명령을 무시하고 가만히 서 있는 것.** Stage 3 toy 환경의
"정지 시 보상 0.4426 하한선" 과 정확히 같은 실패다.

검사 7번이 이 비율을 직접 잰다.

## 검증

```bash
python check.py
```

| # | 통과 기준 |
|:---:|---|
| 1 | 관측이 45차원 |
| 2 | `ang_vel` × 0.25 |
| 3 | `joint_vel` × 0.05 |
| 4 | `joint_pos` × 1.0 |
| 5 | **gravity 는 스케일하지 않는다** |
| 6 | **`_true_lin_vel_b` 는 스케일 *전* 원값을 보관** |
| 7 | **스케일 후 `joint_vel`/명령 비율 < 3** (스케일 없으면 20배) |

6번이 미묘하다. `_true_lin_vel_b` 는 **CENet 의 학습 타깃**이다 (Stage 4).
스케일 후 값을 저장하면 CENet 이 엉뚱한 스케일의 속도를 배우게 된다.
원본도 스케일 직전에 `.clone()` 으로 원값을 떠 둔다.

## 검사기가 Isaac Sim 없이 도는 방법

`_get_observations` 는 메서드지만, 필요한 속성만 갖춘 가짜 `self` 를 만들어
**언바운드로 직접 호출**한다.

```python
get_obs = load_module(path).DreamWaQEnv._get_observations
get_obs(fake_self)      # env 를 띄우지 않는다
```

`exercises/tools/fake_isaaclab.py` 가 `isaaclab.*` import 를 meta path finder 로
가로채므로 25초짜리 Isaac Sim 기동이 필요 없다. 2초면 끝난다.
