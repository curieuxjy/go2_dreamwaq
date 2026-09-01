# task02 — Go2 스폰하고 상태 읽기  `L1 · FILL`

## 목표

로봇을 씬에 올리고 그 상태를 읽는다. **Stage 2에서 만들 관측 45차원은
전부 여기서 읽는 값들의 조합이다.**

## 채울 곳

`starter.py` 의 `TODO(spawn-go2)` — 3~4줄이다.

`UNITREE_GO2_CFG` 를 복사해 다음을 지정하고 `Articulation` 을 만든다.

| 항목 | 값 |
|---|---|
| `prim_path` | `"/World/Robot"` |
| `init_state.pos` | `(0.0, 0.0, SPAWN_HEIGHT)` |

`SPAWN_HEIGHT = 0.42` 는 이 프로젝트가 쓰는 값이다. 논문은 0.34 m 지만,
생성 지형의 `boxes`(최대 0.1 m)를 xy ±0.5 m 리셋 랜덤화와 함께 넘기려면 더 높아야 한다.

> `.copy()` 를 빼먹지 않는다. `UNITREE_GO2_CFG` 는 모듈 전역 객체라 직접 고치면
> 같은 프로세스의 다른 코드가 영향을 받는다.

## 실행

```bash
python starter.py              # 내 코드
python solution.py             # 정답
python solution.py --viz kit   # GUI 로 주저앉는 모습 보기
```

## 통과 기준

1. 관절 12개 (다리 4 × hip/thigh/calf)
2. `base` 바디 존재 — **종료 조건이 이 바디의 접촉을 본다**
3. 발 4개 (`*_foot`) — foot clearance 보상이 이 바디들을 쓴다
4. 최종 높이가 지면 위 (지면 아래로 꺼지지 않았다)
5. `projected_gravity_b[2]` 가 `[-1.05, 0.05]` 범위

## 읽고 넘어갈 것

출력에 찍히는 값들이 곧 관측의 재료다.

| 값 | 의미 | 관측에서 |
|---|---|---|
| `root_lin_vel_b` | 몸통 기준 선속도 | **Oracle 만** 직접 받는다. Base/Waq 는 못 본다 |
| `projected_gravity_b` | 몸통 기준 중력 방향 | 3차원. 자세를 알려준다 |
| `joint_pos` / `joint_vel` | 관절 각도 / 각속도 | 12 + 12 = 24차원 |

**액션을 한 번도 주지 않으면 로봇은 주저앉는다.** 당연한 결과지만 중요하다 —
서 있는 것조차 매 스텝 관절 목표를 주어야 가능하다. 그게 task03이다.
