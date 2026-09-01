# Stage 1 — Isaac Sim / Isaac Lab 살펴보기

DreamWaQ를 만들기 전에, 그 아래에 깔린 시뮬레이터를 손으로 만져 본다.
전부 **독립 실행 스크립트**다. 프로젝트 코드는 아직 건드리지 않는다.

## 왜 이 순서인가

Stage 2에서 만들 학습 환경은 결국 이 다섯 조각의 조합이다.

```
씬(sim) + 로봇(articulation) + 액션(PD 제어) + 지형(terrain) + 센서(sensor)
   ↓          ↓                    ↓                ↓              ↓
task01     task02              task03           task04         task05
                                                                  │
                    Stage 2 의 관측·보상·종료가 전부 여기서 나온다 ┘
```

## 실습 목록

| # | 실습 | 레벨 | 무엇을 얻는가 |
|:---:|---|:---:|---|
| 01 | [`task01_empty_scene`](task01_empty_scene/) | **L0** | `AppLauncher` → `SimulationContext` → step 루프의 뼈대 |
| 02 | [`task02_spawn_go2`](task02_spawn_go2/) | **L1** | `Articulation`, 관절 순서, `data.*` 로 상태 읽기 |
| 03 | [`task03_pd_control`](task03_pd_control/) | **L1** | 액션 → 관절 목표 → PD 토크. DreamWaQ의 액션 스케일 0.25의 의미 |
| 04 | [`task04_terrain`](task04_terrain/) | **L1** | plane vs generator, `env_origins`, 지형 커리큘럼의 토대 |
| 05 | [`task05_sensors`](task05_sensors/) | **L2** | `ContactSensor` / `RayCaster` — Stage 2 관측·종료의 원재료 |

앞의 넷은 가볍게 지나간다. **05만 제대로 붙잡는다** — Stage 2에서 그대로 쓴다.

## 실행 방법

모든 스크립트는 Isaac Sim 번들 python(또는 conda env)으로 실행한다.

```bash
cd exercises/stage1_isaac_basics/task01_empty_scene
python solution.py             # 정답 실행 — 무엇이 나와야 하는지 먼저 본다
python starter.py              # 내 코드 실행
python solution.py --viz kit   # GUI 창으로 보기
```

**기본이 headless 다.** GUI 를 원하면 `--viz kit` 을 준다. 처음 한 번은 GUI로 보는
것을 권한다 — 로봇이 실제로 어떻게 움직이는지 눈으로 봐야 감이 온다.

> 예전 자료나 이 repo의 `train.py --headless` 에서 보이는 `--headless` 플래그는
> 현재 Isaac Lab 에서 deprecated 다. 동작은 하지만 경고가 뜬다.

각 스크립트는 **스스로 검증한다**. 끝까지 돌면 `[PASS]` 줄들이 찍히고,
틀리면 그 자리에서 `AssertionError` 와 함께 무엇이 기대와 달랐는지 알려준다.

> Stage 1의 검증은 Isaac Sim 기동이 필요해 한 번에 ~25초 걸린다.
> Stage 2부터는 대부분 Isaac Sim 없이 1초 만에 도는 `check.py` 가 붙는다.

## starter 는 생성물이다

`starter.py` 를 직접 고쳐도 되지만, 되돌리고 싶으면 언제든 다시 만든다.

```bash
python ../../tools/make_exercise.py --id <실습 id>
```

정답은 같은 폴더의 `solution.py` 에 있다. 막히면 열어 봐도 된다 —
다만 **먼저 20분은 스스로 붙잡아 본다**.
