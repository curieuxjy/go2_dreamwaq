# lec_dreamwaq

강의용으로 분리한 **DreamWaQ** ([arXiv:2301.10602](https://arxiv.org/abs/2301.10602)) 의 Isaac Sim / Isaac Lab 구현.
사족보행 로봇 **Unitree Go2** 를 대상으로 하며, 핵심은 지형 추정을 담당하는 **CENet**(Context-aided Estimator Network) 이다.

원본 통합 repo(IsaacGym 레거시 · ROS 2 sim2sim 배포 · Quarto 문서 사이트 포함)는
[curieuxjy/IsaacLab_DreamWaQ](https://github.com/curieuxjy/IsaacLab_DreamWaQ) 에 있다.
이 repo 는 그중 **Isaac Sim 기반 RL 소스만** 추출한 것이다.

## 패키지

| 디렉토리 | API | 상태 |
|---|---|---|
| [`dreamwaq_manager/`](dreamwaq_manager/) | `ManagerBasedRLEnv` | **주 스택**. 논문의 몸통 접촉 종료를 그대로 사용 |
| [`dreamwaq_direct/`](dreamwaq_direct/)   | `DirectRLEnv`       | 동일 알고리즘의 Direct API 포팅 (cross-check용). 종료 조건·보상·관측 모두 Manager 와 동일 ([KNOWN_ISSUES.md](dreamwaq_direct/KNOWN_ISSUES.md)) |

**두 패키지는 코드를 공유하지 않는다.** 각자 `algorithms/{cenet,estnet,dreamwaq_runner}.py` 사본을
따로 갖고 있고, 다른 것은 **Isaac Lab 작성 방식뿐** 알고리즘은 같아야 한다. 그래서 알고리즘을
고칠 때는 **양쪽에 다 반영**해야 한다 (한 번 어긋난 적이 있어 나중에 back-port 했다).
동작이 같으므로 체크포인트는 양쪽 모두 upstream `deploy_sim2sim` 스택과 호환된다.

학습과 결과 표는 전부 `dreamwaq_manager` 로 만든다 — `run_full_pipeline.sh` 가 Manager 전용이라
`dreamwaq_direct/` 에는 학습 산출물이 없다.

## 파일 구조

```
lec_dreamwaq/
├── run_full_pipeline.sh          6개 task × (학습 → 영상), Manager 전용
├── dreamwaq_manager/             주 스택 (모든 결과의 출처)
│   ├── scripts/
│   │   ├── rsl_rl/{train,play,watch,collect_velocity}.py
│   │   ├── compare_runs.py       tfevents → figures/ 그림 + summary.csv
│   │   ├── eval_checkpoints.py   체크포인트 전수 평가 → best policy
│   │   └── quick_test.sh         64 envs / 30 iters 스모크
│   ├── source/dreamwaq_manager/dreamwaq_manager/
│   │   ├── algorithms/           cenet.py · estnet.py · dreamwaq_runner.py
│   │   └── tasks/locomotion/
│   │       ├── velocity_env_cfg.py    DreamWaQ 원본 레시피 env
│   │       ├── terrains.py            지형 비율 헬퍼 (6종 균등)
│   │       ├── mdp/rewards.py         커스텀 보상 항
│   │       └── config/go2/            env cfg + PPO cfg + gym 등록
│   └── logs/                     gitignore — 체크포인트·tfevents·영상·wandb
├── dreamwaq_direct/              cross-check 스택 (같은 구조, algorithms 사본 별도)
├── exercises/                    Stage 1~5 강의 실습
└── figures/                      compare_runs.py 산출물
```

## 실행 흐름

```
run_full_pipeline.sh
  └─ 6개 task 각각:
       ├─ train.py  (4096 envs)
       │    └─ Base/Oracle → OnPolicyRunner,  Waq → OnPolicyRunnerWaq (CENet 결합)
       │         └─ logs/rsl_rl/<experiment_name>/<timestamp>/
       │              model_*.pt · events.out.tfevents.* · params/*.yaml
       └─ play.py --video   →  videos/play/rl-video-step-0.mp4

학습이 끝난 뒤:
  compare_runs.py       tfevents → 곡선·막대 그림 + summary.csv (Waq−Base, Oracle−Waq)
  eval_checkpoints.py   체크포인트를 전부 굴려 best policy 선정
  collect_velocity.py   CENet 추정 속도 vs 실제 속도 궤적 (.npz)
```

Waq 만 rollout 안쪽이 다르다 — 관측 이력을 CENet 에 넣어 `est_vel(3)` 과 `context(16)` 을 얻고,
정규화한 base 관측 45 에 이어 붙여 actor 입력 64 를 만든다. 자세한 순서는 `CLAUDE.md` 참조.

## 설치

Isaac Sim 6.0 바이너리 + IsaacLab `release/3.0.0-beta2` (번들 Python 3.12.13) 가 전제다.
전체 절차는 [`setup.qmd`](setup.qmd) 참조.

```bash
cd ~/Documents/lec_dreamwaq/dreamwaq_manager
~/IsaacLab/_isaac_sim/python.sh -m pip install -e source/dreamwaq_manager
```

> 이 repo 를 새 위치로 옮겼거나 원본 repo 에서 이미 `-e` 설치를 해 두었다면,
> editable 설치가 **이전 경로를 가리키므로** 위 명령으로 다시 설치해야 한다.

## 실행

```bash
cd dreamwaq_manager

# smoke test (64 envs / 30 iters) — 풀 학습 전에 먼저 실행
./scripts/quick_test.sh

# 학습
python scripts/rsl_rl/train.py --task=DreamWaQ-Manager-Go2-Base-v0 --headless

# 평가
python scripts/rsl_rl/play.py --task=DreamWaQ-Manager-Go2-Base-Play-v0 \
    --load_run=FOLDER --checkpoint=N
```

## Task (알고리즘은 PPO 하나, 관측 설계 3종)

강의 범위는 **PPO × {Base, Oracle, Waq}** 3가지 비교뿐이다. 각 task 에 `-Play-v0` 평가 변형이 있다.

| 변형 | actor 관측 | 의미 |
|---|---|---|
| **Base** | 45 (proprioception only) | 하한 — 지형·속도 정보 없음 |
| **Oracle** | 48 (+ 실제 base linear velocity) | 상한 — 특권 정보를 직접 받음 |
| **Waq** | 64 (45 + CENet 추정 속도 3 + context 16) | DreamWaQ — CENet 이 특권 정보를 추정 |

**실제 실험축은 공식 env 레시피 위의 6변형**(3종 × flat/rough)이다. 셋이 같은 env 를 쓰고
actor 관측만 다르므로, 성능 차이가 곧 관측 설계의 효과가 된다.

```
DreamWaQ-{BaseDwq,OracleDwq,Waq-Official}-{Flat,Rough}-PPO-v0   (+ -Play-v0)   <- 실험 대상
```

원본 DreamWaQ 레시피(`DreamWaQ-Manager-Go2-{Base,Oracle,Waq}-v0`)도 등록되어 있지만
**보행에 실패해 폐기**했다 (몸통 접촉 종료 78%). 근거 로그는 `pipeline_logs_dreamwaq_env_failed/` 에 있다.
Direct 스택(`DreamWaQ-Direct-Go2-*`)은 cross-check 용이라 원본 레시피 3종만 제공한다.

전체 목록은 `python scripts/list_envs.py`, 자세한 구조 설명은
[`dreamwaq_manager/README.md`](dreamwaq_manager/README.md) 참조.

## 학습과 결과 비교

```bash
./run_full_pipeline.sh      # 6종 학습 + 각 run 영상 (기본 3000 iters / 4096 envs / wandb online)

cd dreamwaq_manager && ~/IsaacLab/_isaac_sim/python.sh scripts/compare_runs.py
                            # tfevents → figures/ 비교 그림 + summary.csv
```

`summary.csv` 의 `Waq-Base` 가 **CENet 의 순수 기여**, `Oracle-Waq` 가 **남은 격차**다.

## 논문 대조

원문(arXiv:2301.10602)의 네트워크 구조·보상 가중치·PPO 하이퍼파라미터·CENet 손실·
도메인 랜덤화를 **코드 값과 나란히** 정리한 표가 [`PAPER.md`](PAPER.md) 에 있다.

알아둘 것 — **지금 학습에 쓰는 공식 env 레시피는 논문 보상 12항 중 5항이 빠져 있고
2항은 가중치가 1.5배다.** 원본 DreamWaQ 레시피가 걷지 못해 공식 레시피를 채택한 결과다.
코드의 β annealing 과 AdaBoot 램프도 논문에 없는 것이다. 자세한 것은 `PAPER.md` §2, §4, §5.

## 강의 실습

단계별 빈칸 채우기 실습이 [`exercises/`](exercises/) 에 있다 (Stage 1~5, 강도 L0~L3).
프로덕션 소스는 건드리지 않고 생성기가 starter 를 만든다 —
[`exercises/README.md`](exercises/README.md) 참조.

## 포함되지 않은 것

- **PPO 외 알고리즘 / 확장** — SAC 스택과 DreamWaQ++(CENetPlus)는 강의 혼동을 막기 위해 제거했다.
  필요하면 git 히스토리(`a2d2774`) 또는 upstream repo 에서 볼 수 있다.
  (official-env 비교 6변형은 한때 함께 제거했다가 `c8df7b4` 에서 **되살렸다** — 실제 실험축이다.)
- **학습 로그 / 체크포인트** (`logs/`) — gitignore 대상이며 이 repo 에 없다.
- **IsaacGym 원본** (`dreamwaq/`), **ROS 2 sim2sim 배포** (`deploy_sim2sim/`), **Quarto 문서 사이트**
  (`index.qmd`, `comparison.qmd`, `plan.qmd`, `report.qmd`) — upstream repo 에 있다.

하드웨어 기준: RTX 4080 16GB 에서 4096 envs 안정 (8192 는 OOM).
