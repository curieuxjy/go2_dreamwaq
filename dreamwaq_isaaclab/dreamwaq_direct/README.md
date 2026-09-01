# dreamwaq_direct (IsaacLab — DirectRLEnv)

> 종료 조건은 원본 IsaacGym DreamWaQ 와 동일하다 — **몸통(base) 접촉력 > 1 N**
> (`terminate_after_contacts_on = ["base"]`). [`../dreamwaq_manager/`](../dreamwaq_manager/) 도
> 같은 판정을 쓰므로 두 스택의 수치는 서로 비교 가능하다.
> 4096 envs 생성 지형에서 두 스택이 동등하게 학습함을 실측 확인했다. 그 과정에서 고친
> collision filtering 버그는 [`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md) 참조.

> Independent implementation of [DreamWaQ](https://arxiv.org/abs/2301.10602), ported from the original IsaacGym code ([upstream `dreamwaq/`](https://github.com/curieuxjy/IsaacLab_DreamWaQ/tree/main/dreamwaq)) to **Isaac Lab 3.0.0-beta2 (Isaac Sim 6.0)** using the `DirectRLEnv` API. Robot platform: **Unitree Go2**.

이 문서는 원본 IsaacGym README([upstream `dreamwaq/README.md`](https://github.com/curieuxjy/IsaacLab_DreamWaQ/blob/main/dreamwaq/README.md)) 의 *Main Code Structure* 와 평행하게 작성되었으며, 원본 IsaacGym 프로젝트의 각 구성 요소가 IsaacLab `DirectRLEnv` 레이아웃의 어디로 매핑되었는지 설명한다.

---

## Table of Contents

| Section | Description |
|---------|-------------|
| [Quick Start](#quick-start) | 설치 / 학습 / 추론 명령 |
| [Task Options](#task-options) | Base / Oracle / Waq 옵션 |
| [Structure Mapping](#structure-mapping-isaacgym--isaaclab-directrlenv) | 원본 ↔ Direct 매핑 표 |
| [File-by-file correspondence](#file-by-file-correspondence) | 파일 단위 대응 |
| [Key Implementation Details](#key-implementation-details) | DirectRLEnv 이식 시 주요 결정 |
| [See Also](#see-also) | 관련 문서 링크 |

---

## Quick Start

### Installation

```bash
cd dreamwaq_direct
pip install -e source/dreamwaq_direct
```

IsaacLab(release/3.0.0-beta2) + Isaac Sim 6.0 이 사전 설치되어 있어야 한다. 환경 셋업 전체 절차는 [`../setup.qmd`](../setup.qmd) 참조.

### Training

```bash
cd scripts
python train.py --task=DreamWaQ-Direct-Go2-Base-v0   --headless
python train.py --task=DreamWaQ-Direct-Go2-Oracle-v0 --headless
python train.py --task=DreamWaQ-Direct-Go2-Waq-v0    --headless
```

편의 스크립트:
- `run_base.sh`, `run_oracle.sh`, `run_waq.sh` — 단일 태스크 학습
- `run_all_trainings.sh` — 세 태스크 순차 실행 (default 4096 envs / 5k iters / wandb)
- `quick_test.sh` — 64 envs / 30 iters smoke test
- `watch.{py,sh}` — 학습 진행 모니터링

### Play

```bash
python play.py --task=DreamWaQ-Direct-Go2-Waq-Play-v0 --load_run=FOLDER --checkpoint=N
```

체크포인트는 `logs/rsl_rl/[experiment_name]/[timestamp]/` 에 저장된다.

---

## Task Options

| Task ID | Actor obs | Critic obs (state) | Memo |
|---------|:---------:|:------------------:|------|
| `DreamWaQ-Direct-Go2-Base-v0`   | 45      | 0   | blind, no privileged obs |
| `DreamWaQ-Direct-Go2-Oracle-v0` | 48      | 238 | true `lin_vel` 포함 + privileged height scan |
| `DreamWaQ-Direct-Go2-Waq-v0`    | 45 → 64 | 238 | CENet (`est_vel + context`) + privileged |

각 task 에 대응하는 `-Play-v0` variant 가 별도로 등록되어 있으며, `num_envs=50`, `terrain.curriculum=False`, `obs_noise=False`, `system_delay=False`, push/disturb 비활성화 상태로 평가/시각화용이다.

원본 ↔ Direct task ID 매핑:

| 원본 (`--task=...`) | DirectRLEnv (`--task=...`) |
|---|---|
| `a1_base`   | `DreamWaQ-Direct-Go2-Base-v0`   (+ `-Play-v0`) |
| `a1_oracle` | `DreamWaQ-Direct-Go2-Oracle-v0` (+ `-Play-v0`) |
| `a1_waq`    | `DreamWaQ-Direct-Go2-Waq-v0`    (+ `-Play-v0`) |

---

## Structure Mapping (IsaacGym → IsaacLab DirectRLEnv)

원본 `dreamwaq/` 는 두 라이브러리로 분리되어 있다:
- `legged_gym/` — 시뮬레이션 환경 (env / config / scripts / utils / resources)
- `rsl_rl/`    — RL 알고리즘 (PPO / actor-critic / runner / VAE)

IsaacLab 으로 이식하면서, **시뮬레이터 plumbing / PPO 학습 루프 / actor-critic 모듈 / robot URDF 자산은 IsaacLab 이 제공**하기 때문에, `dreamwaq_direct/` 는 DreamWaQ 고유 로직 (env logic, CENet, EstNet, custom runner) 만 담는다.

### Top-level package layout

```
dreamwaq_direct                           # 외부 프로젝트 루트 (Isaac Lab 템플릿 레이아웃)
├── scripts/
│   ├── list_envs.py / zero_agent.py / random_agent.py   # IsaacLab 정본 헬퍼
│   ├── run_{base,oracle,waq}.sh          # task별 wrapper
│   ├── run_all_trainings.sh              # 세 태스크 순차 실행
│   ├── quick_test.sh                     # 64 envs / 30 iters smoke test
│   ├── watch.sh                          # 학습 모니터링
│   └── rsl_rl/                           # legged_gym/legged_gym/scripts/  대응
│       ├── train.py                      # ← legged_gym/.../scripts/train.py
│       ├── play.py                       # ← legged_gym/.../scripts/play.py
│       ├── watch.py
│       └── cli_args.py                   # IsaacLab RSL-RL CLI 헬퍼 (동봉)
│
└── source/dreamwaq_direct/               # 설치 가능한 확장: pip install -e source/dreamwaq_direct
    ├── config/extension.toml             # Kit 확장 매니페스트 + deps
    ├── docs/CHANGELOG.rst
    ├── pyproject.toml                    # build-system only
    ├── setup.py                          # extension.toml 읽음 + find_packages()
    ├── README.md
    └── dreamwaq_direct/
        ├── __init__.py
        ├── algorithms/                   # rsl_rl/rsl_rl/  대응 (필요한 부분만)
        │   ├── cenet.py                  # ← rsl_rl/.../vae/cenet.py
        │   ├── estnet.py                 # ← rsl_rl/.../vae/estnet.py
        │   └── dreamwaq_runner.py        # ← rsl_rl/.../runners/on_policy_runner.py
        │                                 #   (OnPolicyRunnerWAQ만 추출)
        │
        └── tasks/locomotion/             # legged_gym/legged_gym/envs/  대응
            ├── dreamwaq_env.py           # ← envs/base/legged_robot.py
            ├── dreamwaq_env_cfg.py       # ← envs/base/legged_robot_config.py
            └── config/go2/
                ├── __init__.py           # ← envs/__init__.py (gym.register)
                ├── go2_env_cfg.py        # ← envs/go2/go2_config.py
                └── agents/
                    └── rsl_rl_ppo_cfg.py # PPO + CENet runner config
```

---

## File-by-file correspondence

### Environment / config

| 원본 (`dreamwaq/`) | Direct (`dreamwaq_direct/`) | Notes |
|---|---|---|
| `legged_gym/envs/__init__.py` (`task_registry.register`) | `tasks/locomotion/config/go2/__init__.py` | gymnasium 네이티브 `gym.register`. `task_registry` 폐지. |
| `legged_gym/envs/go2/go2_config.py` (`Go2RoughCfg`) | `tasks/locomotion/config/go2/go2_env_cfg.py` | `Go2{Base,Oracle,Waq}DirectCfg` (+ `_PLAY` variant) — `DreamWaQDirectEnvCfg` 상속. PD override (stiffness=20.0, damping=0.5), spawn `0.42 m`, terrain 축소 등 Go2 specific. |
| `legged_gym/envs/base/legged_robot.py` (`LeggedRobot`) | `tasks/locomotion/dreamwaq_env.py` (`DreamWaQEnv`) | `DirectRLEnv` 상속. 원본의 단일 `step()` 을 IsaacLab API 의 `_pre_physics_step` / `_apply_action` / `_get_observations` / `_get_rewards` / `_get_dones` / `_reset_idx` 로 분할. 모든 로직은 한 파일 inline. |
| `legged_gym/envs/base/legged_robot_config.py` (`LeggedRobotCfg`, `LeggedRobotCfgPPO`) | `tasks/locomotion/dreamwaq_env_cfg.py` (`DreamWaQDirectEnvCfg`, `EventCfg`) | `DirectRLEnvCfg` 상속. 보상 스케일 / 도메인 랜덤화 (`EventCfg`) / CENet 옵션 (`len_obs_history`, `num_context`, `num_est_vel`) / 커리큘럼 등 모두 포함. |
| `legged_gym/utils/task_registry.py` | (제외) | gymnasium native registration 으로 대체. |
| `legged_gym/utils/terrain.py` (custom Terrain class) | (제외) | `isaaclab.terrains.TerrainImporterCfg` + `ROUGH_TERRAINS_CFG` 로 대체. |
| `legged_gym/utils/logger.py` (matplotlib) | (제외) | wandb / tensorboard 사용. |
| `legged_gym/resources/robots/a1/` (URDF + meshes) | (제외) | A1 → Go2 platform 변경. `isaaclab_assets.UNITREE_GO2_CFG` (USD asset) 사용. |

### Scripts

| 원본 | Direct | Notes |
|---|---|---|
| `legged_gym/scripts/train.py` | `scripts/rsl_rl/train.py` | log_std clamp `[-5, 2]` monkey-patch 동일 (policy std 폭발 방지). |
| `legged_gym/scripts/play.py` | `scripts/rsl_rl/play.py` | |
| `legged_gym/scripts/mini_test.py` | (제외) | |

### Algorithms (RSL-RL)

| 원본 (`dreamwaq/rsl_rl/`) | Direct (`dreamwaq_direct/algorithms/`) | Notes |
|---|---|---|
| `rsl_rl/algorithms/ppo.py` | (제외) | `isaaclab_rl.rsl_rl` (`RslRlPpoAlgorithmCfg`) 재사용. |
| `rsl_rl/modules/actor_critic.py` | (제외) | `RslRlMLPModelCfg` 사용 (hidden=[512,256,128], activation=elu). |
| `rsl_rl/runners/on_policy_runner.py` (`OnPolicyRunner`) | (제외 — IsaacLab 기본 runner) | Base / Oracle 모델용. |
| `rsl_rl/runners/on_policy_runner.py` (`OnPolicyRunnerWAQ`) | `algorithms/dreamwaq_runner.py` (`OnPolicyRunnerWaq`) | Waq 모델 전용. CENet 학습 루프 통합. |
| `rsl_rl/runners/on_policy_runner.py` (`OnPolicyRunnerEst`) | (제외) | EstNet 비교 모델은 사용하지 않음. |
| `rsl_rl/utils/rms.py` (`RunningMeanStd`) | `algorithms/cenet.py` 내부 | CENet 의 normal prior 학습용. |
| `rsl_rl/vae/cenet.py` | `algorithms/cenet.py` (`CENet`, `CenetRolloutStorage`) | 거의 그대로 이식. |
| `rsl_rl/vae/estnet.py` | `algorithms/estnet.py` (`EstNet`) | 거의 그대로 이식. |

### PPO Runner Config (Direct 신규)

`tasks/locomotion/config/go2/agents/rsl_rl_ppo_cfg.py` 는 `RslRlOnPolicyRunnerCfg` 기반 PPO + CENet 설정을 보유한다. 원본의 `LeggedRobotCfgPPO` 파편을 IsaacLab 표준 cfg 클래스 형태로 재구성한 것.

| 클래스 | 적용 task | `class_name` | `max_iterations` |
|---|---|---|---|
| `Go2BasePPORunnerCfg`   | `Base-v0`   | (default `OnPolicyRunner`) | 1500 |
| `Go2OraclePPORunnerCfg` | `Oracle-v0` | (default `OnPolicyRunner`) | 1500 |
| `Go2WaqPPORunnerCfg`    | `Waq-v0`    | `OnPolicyRunnerWaq` | 5000 |

`Go2WaqPPORunnerCfg` 는 추가로 `waq` (학습 hyperparams) 와 `vae` (CENet learning rate / β / scheduler) dict 를 정의한다.

---

## Key Implementation Details

### DirectRLEnv 이식 결정

- 원본의 `LeggedRobot.step()` 단일 메서드를 IsaacLab `DirectRLEnv` 가 강제하는 6 개 hook 으로 분할:
  - `_pre_physics_step(actions)` — clip + PD target 계산
  - `_apply_action()`            — joint position target 적용
  - `_get_observations()`        — body-frame obs / obs noise / system delay / obs history / privileged obs (height scan 187 ray)
  - `_get_rewards()`             — DreamWaQ 13 개 reward (paper 그대로)
  - `_get_dones()`               — base contact threshold + timeout
  - `_reset_idx(env_ids)`        — randomized state, terrain curriculum, command resample, episode reward logging
- Manager 패키지 (`dreamwaq_manager/`) 와 달리, Manager classes (Reward / Observation / Termination / Curriculum) 를 사용하지 않고 **모든 로직을 `DreamWaQEnv` 한 클래스 내부에 inline** 작성. 원본 IsaacGym `LeggedRobot` 와 1:1 대응을 유지하기 위함.

### Robot / scene

- `UNITREE_GO2_CFG` (USD asset, `isaaclab_assets`) + PD override (`stiffness=20.0`, `damping=0.5`).
- Spawn height `0.42 m` — `boxes` sub-terrain (≤ 0.1 m) + xy ±0.5 m reset 랜덤화 시 충돌 방지 마진 확보. (paper 의 0.34 m 보다 높음.)
- Height scanner: `RayCasterCfg` (`/Robot/base` offset z=20 m), `GridPattern` resolution=0.1, size=[1.6, 1.0] → 187 rays.

### Terrain

- `ROUGH_TERRAINS_CFG` (10 levels × 20 types, 6 sub-terrains: pyramid_stairs / pyramid_stairs_inv / boxes / random_rough / hf_pyramid_slope / hf_pyramid_slope_inv).
- Go2 크기에 맞춰 `boxes.grid_height_range = (0.025, 0.1)`, `random_rough.noise_range = (0.01, 0.06)`, `noise_step = 0.01` 로 축소.
- `curriculum = True`, `max_init_terrain_level = 4` — 학습 초기에 ≤ 4 레벨에서 시작 → reset 시 `+1` (이동 거리 ≥ commanded distance / 2) / `-1` (≤ 1/4) 로 진행.

### Privileged observation (`state_space = 238`)

```
lin_vel(3) + ang_vel(3) + gravity(3) + commands(3)
+ joint_pos(12) + joint_vel(12) + actions(12)
+ height_scan(187) + disturb_force(3)
```

### CENet input

```
obs_history (5 timesteps × 45 obs = 225)  →  est_vel(3) + context(16)
```

### Domain randomization (`EventCfg`)

`physics_material` (friction 0.2–1.25), `add_base_mass` (-1 ~ +2 kg), `base_com` (±5 cm), `randomize_pd_gains` (×0.9–1.1), `push_robot` (1 Hz, ±1 m/s), `reset_base` (xy ±0.5 m, yaw ±π, vel ±0.5).

### Manager 패키지와의 alignment

보상 / 관측 / 도메인 랜덤화 / 리셋 로직은 `dreamwaq_manager/` 와 1:1 alignment 를 유지한다. 차이점 분석은 upstream repo 의 `comparison.qmd` 참조.

---

## See Also

- [`../README.md`](../README.md) — 이 repo 개요
- [`../setup.qmd`](../setup.qmd) — IsaacLab 설치 / 자산 다운로드
- [`../dreamwaq_manager/`](../dreamwaq_manager/) — 동일 알고리즘의 ManagerBasedRLEnv 구현 (활성 스택)

아래 문서는 이 강의용 repo 에는 포함되지 않으며 [upstream repo](https://github.com/curieuxjy/IsaacLab_DreamWaQ) 에 있다:

- `index.qmd` — 프로젝트 전체 개요 (Quarto 홈)
- `comparison.qmd` — Manager ↔ Direct 비교 분석
- `plan.qmd` — IsaacGym → IsaacLab 마이그레이션 계획
- `report.qmd` — 실험 결과 리포트
- `dreamwaq/README.md` — IsaacGym 원본 README (read-only reference)
