# Stage 5 — inference 실험 결과 비교

앞의 네 stage 에서 만든 것이 **실제로 무엇을 만들어 냈는지** 확인한다.

## 전제: 학습자는 full training 을 돌리지 않는다

배포 자산을 만든 실제 학습은 4096 envs × **4000 iter** 였고, RTX 4080 16GB 기준
rough 1 run 이 **약 2시간 50분**, flat 1 run 이 약 33분이었다. 비교에 필요한 6 run 이면
**10시간이 넘는다.**

그래서 학습자는 자기 구현이 **"돌아간다"** 만 확인하고 (`quick_test` 규모, 64 envs / 30 iter),
**결과 해석은 강사가 미리 돌린 배포 자산**으로 한다.

```bash
cd dreamwaq_manager && ./scripts/quick_test.sh     # 내 구현이 도는지만 (64 envs / 30 iters)
```

## 배포 자산

`logs/` 와 `figures/` 는 gitignore 대상이라 이 repo 에 들어 있지 않다. 강사가 따로 배포한다.

```
dreamwaq_manager/logs/
└── rsl_rl/                                  ← Stage 5 실습이 쓰는 유일한 폴더
    ├── BaseDwq-Official-Rough-PPO-v0/
    │   └── <timestamp>_DreamWaQ-BaseDwq-Rough-PPO-v0_seed42_envs4096/
    │       ├── model_0.pt … model_3999.pt        체크포인트 9개 (save_interval=500, 4000 it)
    │       ├── events.out.tfevents.*             tensorboard 로그
    │       ├── params/{env,agent}.yaml           재현용 설정 — 실제로 반영된 값이 여기 있다
    │       ├── git/{IsaacLab,lec_dreamwaq}.diff  학습 당시 소스 diff
    │       └── videos/play/rl-video-step-0.mp4   추종 영상 20초 (400 스텝)
    ├── OracleDwq-Official-Rough-PPO-v0/…        actor 48 (+ 참 base 선속도)
    ├── Waq-Official-Rough-PPO-v0/…              actor 64 (+ CENet est_vel·context)
    ├── TerrainOracle-Official-Rough-PPO-v0/…    actor 232 (+ 높이맵) — 논문이 말한 oracle
    ├── BaseDwq-Official-Flat-PPO-v0/…
    ├── OracleDwq-Official-Flat-PPO-v0/…
    └── Waq-Official-Flat-PPO-v0/…               일곱 run 모두 같은 구조
```

> **`rsl_rl/` 하나뿐이다.** 개발 중에는 절제·프로브 실험 폴더가 `logs/` 아래에 여럿
> 쌓이지만(`_archive_*`, `_cenet_regression`, `_*_probe` 등) 전부 **중간 산출물이라
> 정리 시점에 지운다.** 실습은 `rsl_rl/` 만 쓴다. task04 의 실측표에 나오는 붕괴 런들도
> 그 정리로 사라졌으므로, **그 표는 강사가 당시 재 둔 기록이지 재현 대상이 아니다** —
> 실습은 자산 없이도 통과한다(실제 체크포인트 구간은 채점 대상이 아니다).

받은 압축을 **`dreamwaq_manager/logs/` 아래에 그대로** 푼다. `rsl_rl/` 바로 아래의
폴더 이름(`experiment_name`)이 `play.py` 와 `compare_runs.py` 가 run 을 찾는 열쇠이므로
바꾸지 않는다. 그 아래 timestamp 폴더 이름은 자유다 — 도구가 가장 최근 것을 고른다.

다른 곳에 풀었다면 경로를 인자로 넘기면 된다.

```bash
cd dreamwaq_manager
~/IsaacLab/_isaac_sim/python.sh scripts/compare_runs.py --logdir /어디/에/풀었든/rsl_rl
```

`compare_runs.py` 는 `figures/` 를 다시 만든다 (`curves_{flat,rough}.png`,
`reward_{flat,rough}.png`, `terrain_level_rough.png`, `bar_final.png`, `summary.csv`).
학습을 안 한 run 은 경고만 내고 건너뛴다.

## 실습 목록

| # | 실습 | 레벨 | 무엇을 하는가 |
|:---:|---|:---:|---|
| 01 | [`task01_play_compare`](task01_play_compare/) | **L0** | 체크포인트 3종을 눈으로 비교 (영상) |
| 02 | [`task02_curves`](task02_curves/) | **L2** | tfevents 에서 곡선을 뽑아 비교 플롯 + 요약표 |
| 03 | [`task03_interpret`](task03_interpret/) | 서술 | 두 델타를 해석한다 |
| 04 | [`task04_diagnose`](task04_diagnose/) | **L2** | 체크포인트로 CENet 잠재의 붕괴 기전을 가른다 |

task04 는 위 배포 자산이 **없어도 통과한다** (합성 체크포인트로 채점한다). 자산이 있으면
보관 런들을 찾아 찍어 주기만 하며, 경로는 `--run` 으로 직접 줄 수도 있다.
task04 README 의 "실측 — 보관된 네 런" 표는 위 두 `_archive_4000it_*` 와
`_archive_3000it_stairheavy` 를 **전부 받았을 때** 그대로 재현되는 값이다. 세 폴더 중 일부만
받았다면 그 줄만 안 찍힐 뿐, 채점에는 영향이 없다.

## 이 stage 의 결론 두 줄

셋은 **같은 env, 같은 하이퍼파라미터, 같은 예산**으로 학습했다. 다른 것은 actor 관측뿐이다.
그래서 차이를 관측 설계의 효과로 읽을 수 있다.

```
Waq − Base     = CENet 의 순수 기여
                 (지형·속도를 추정해 준 것이 얼마나 도움이 되었나)

Oracle − Waq   = 남은 격차
                 (추정이 진짜 특권정보에 얼마나 못 미치나)
```

논문이 맞다면 **Base < Waq < Oracle** 순서가 나와야 한다.

실측 (seed 42, 4000 it, 수렴 구간 = 마지막 400 iteration 평균):

| 지형 | Base (45) | Waq (64) | Oracle (48) | Waq−Base | Oracle−Waq |
|---|---|---|---|---|---|
| **rough** | 0.5492 ± 0.0015 | 0.5565 ± 0.0015 | 0.5822 ± 0.0017 | **+0.0073 ± 0.0021** (유의) | +0.0257 ± 0.0023 |
| **flat** | 0.9383 ± 0.0003 | 0.9390 ± 0.0003 | 0.9460 ± 0.0002 | +0.0007 ± 0.0004 (**노이즈**) | +0.0070 ± 0.0004 |

rough 에서는 순서가 나왔고 flat 에서는 안 나왔다. **왜 그런지가 task03 의 질문이다.**
`±` 는 **한 run 안의 출렁임**만 재며 seed 간 변동은 포함하지 않는다 — 단일 seed 다.

## 지표: underlying tracking

총 보상(`Train/mean_reward`)은 보상 항의 합이라 **무엇이 좋아졌는지** 알려주지 않는다.
그래서 속도 추종 항만 떼어내 가중치로 되돌린다.

```
underlying = Episode_Reward/track_lin_vel_xy_exp ÷ 1.5
           = mean exp(-‖v_cmd − v‖² / 0.25)      ∈ [0, 1]
```

`1.0` 이면 명령 속도를 완벽히 따라간 것이다. 가중치(1.5)를 나눠 주지 않으면
보상 설계가 바뀔 때 값이 같이 흔들려 run 끼리 비교가 안 된다.

가중치 **1.5 는 공식 레시피**(Isaac Lab `UnitreeGo2RoughEnvCfg`)의 값이다. Stage 2 에서 읽은
`velocity_env_cfg.py` 의 DreamWaQ 레시피는 같은 항이 `track_lin_vel_xy` 이고 **1.0** 이므로
그 계열 run 을 분석할 때는 1.0 으로 나눠야 한다 (`PAPER.md` §2). 헷갈리면 그 run 의
`params/env.yaml` 을 열어 `weight` 를 직접 본다.

## 함께 보면 좋은 지표

rough 3종의 실측값을 함께 적는다 (같은 수렴 구간 평균). **전부 같은 방향을 가리키지는 않는다** —
그것이 task03 에서 다룰 거리다.

| 태그 | 왜 보는가 | Base | Waq | Oracle |
|---|---|---|---|---|
| `Curriculum/terrain_levels` | 잘 걷는 정책일수록 어려운 지형으로 승급한다 | 2.984 | 2.997 | **3.163** |
| `Train/mean_episode_length` | 넘어지면 짧아진다. 1000 에 가까울수록 안 넘어진다 | 936.5 | **918.7** | 950.0 |
| `Episode_Termination/base_contact` | 몸통 접촉 종료 비율 (낮을수록 좋다) | 0.093 | **0.123** | 0.068 |
| `Metrics/base_velocity/error_vel_xy` | 속도 오차 자체 (낮을수록 좋다) | 0.423 | 0.410 | 0.405 |
