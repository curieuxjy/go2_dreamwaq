# Stage 2 — 학습 Env 클래스 (Manager vs Direct)

Stage 1에서 모은 재료(관절 상태 · 액션 · 지형 · 센서)를 **학습 환경**으로 조립한다.

이 강의의 차별점은 **같은 기능을 두 API로 두 번** 구현해 보는 것이다.

| 대상 | Manager 방식 | Direct 방식 | 레벨 |
|---|---|---|:---:|
| 보상 | `rewards.py` + `RewTerm` 가중치 | `_get_rewards` 인라인 | **L2** |
| 관측 | `ObsGroup` + `ObsTerm` | `_get_observations` 인라인 | L2 |
| 종료 | `DoneTerm(illegal_contact)` | `_get_dones` | **L2** |
| 리셋 / randomization | `EventTerm` | `_reset_idx` | L0 |

`velocity_env_cfg.py`(442줄)와 `dreamwaq_env.py`(834줄)를 통째로 시키지 않는다.
**핵심 조각만 파내고** 나머지는 제공한다.

## 실습 목록

| 실습 | 레벨 | 대상 | 무엇을 배우는가 |
|---|:---:|---|---|
| [`reward-joint-power`](reward-joint-power/) | L1 | `rewards.py` | 에너지 페널티. 각속도 0 이면 무벌이라는 성질 |
| [`reward-power-distribution`](reward-power-distribution/) | **L2** | `rewards.py` | 총량이 아닌 **불균형**을 벌하는 항 — 실기 모터 과열 대비 |
| [`reward-smoothness`](reward-smoothness/) | **L2** | `rewards.py` | 액션 저크(2차 차분), 상태를 들고 있는 보상 항 |
| [`reward-foot-clearance`](reward-foot-clearance/) | **L2** | `rewards.py` | 지형 위 발 높이, 속도 가중이라는 설계 |
| [`direct-obs-scale`](direct-obs-scale/) | L1 | `dreamwaq_env.py` (**Direct**) | 관측 스케일링 — 실제로 났던 포팅 버그 #1 |
| [`direct-dones`](direct-dones/) | **L2** | `dreamwaq_env.py` (**Direct**) | 종료 조건. 4차원 접촉 힘을 `(N,)` 로 줄이기 |

권장 순서는 위에서 아래다. `joint-power` 로 몸을 풀고, `power-distribution` 에서
"총량 vs 불균형"의 대비를 잡은 뒤, 상태를 들고 있는 `smoothness` 로 넘어간다.
관측(`direct-obs-scale`)과 종료(`direct-dones`)는 보상 4개를 끝낸 뒤에 한다.

### Manager 의 관측/종료 config 는 왜 실습이 없나

`velocity_env_cfg.py` 는 **import 만으로 Isaac Sim kit 커널을 띄운다.** 그래서
빠른 검증(`check.py`)을 붙일 수 없다. 대신 같은 개념을 **Direct 쪽 인라인 코드**로
다룬다 (`direct-obs-scale`, `direct-dones`) — 오히려 텐서가 눈에 보여 배우기 좋다.

Manager 의 `ObsGroup`/`DoneTerm` 선언 방식은 `dreamwaq_manager/README.md` 와
`velocity_env_cfg.py` 를 **읽는 것(L0)** 으로 갈음한다.

## 프로덕션 소스는 건드리지 않는다

Stage 2의 실습은 전부 **spec 파일 방식**이다. `dreamwaq_manager/` 의 소스에는
마커도 주석도 들어가지 않는다. 실습 정의는 `exercises/specs/*.toml` 에 있고,
생성기가 실제 소스를 읽어 starter 를 만든다.

```bash
python ../tools/make_exercise.py --id reward-smoothness
```

## 검증은 Isaac Sim 없이

`rewards.py` 는 `isaaclab.*` 을 import 하지만, 실습이 검사하는 계산은 순수 torch 다.
`exercises/tools/fake_isaaclab.py` 가 import 만 통과시키는 스텁을 심어 **1초 만에** 검사한다.

```bash
cd reward-smoothness
python check.py              # 내 코드
python check.py --solution   # 완성본 (검사기 자체 점검용)
```
