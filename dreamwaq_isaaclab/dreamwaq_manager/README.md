# dreamwaq_manager (IsaacLab — ManagerBasedRLEnv)

> Independent implementation of [DreamWaQ](https://arxiv.org/abs/2301.10602), ported from the original IsaacGym code ([upstream `dreamwaq/`](https://github.com/curieuxjy/IsaacLab_DreamWaQ/tree/main/dreamwaq)) to **Isaac Lab 3.0.0-beta2 (Isaac Sim 6.0)** using the `ManagerBasedRLEnv` API. Robot platform: **Unitree Go2**.

이 문서는 원본 IsaacGym README([upstream `dreamwaq/README.md`](https://github.com/curieuxjy/IsaacLab_DreamWaQ/blob/main/dreamwaq/README.md)) 의 *Main Code Structure* 와 평행하게 작성되었으며, 원본 IsaacGym 프로젝트의 각 구성 요소가 IsaacLab `ManagerBasedRLEnv` 레이아웃의 어디로 매핑되었는지 설명한다. 또한 [`../dreamwaq_direct/README.md`](../dreamwaq_direct/README.md) 와 짝을 이루는 ManagerBased 구현이며, 두 패키지는 동일한 reward / observation / domain randomization 의미를 공유한다.

---

## Table of Contents

| Section | Description |
|---------|-------------|
| [Quick Start](#quick-start) | 설치 / 학습 / 추론 명령 |
| [Task Options](#task-options) | Base / Oracle / Waq 옵션 |
| [Structure Mapping](#structure-mapping-isaacgym--isaaclab-managerbasedrlenv) | 원본 ↔ Manager 매핑 표 |
| [File-by-file correspondence](#file-by-file-correspondence) | 파일 단위 대응 |
| [Manager API mapping](#manager-api-mapping) | Manager 클래스가 원본 코드를 어떻게 흡수하는지 |
| [Key Implementation Details](#key-implementation-details) | ManagerBasedRLEnv 이식 시 주요 결정 |
| [Direct vs Manager](#direct-vs-manager-comparison-with-dreamwaq_direct) | 비교 |
| [See Also](#see-also) | 관련 문서 링크 |

---

## Quick Start

### Installation

```bash
cd dreamwaq_manager
pip install -e source/dreamwaq_manager
```

IsaacLab(release/3.0.0-beta2) + Isaac Sim 6.0 이 사전 설치되어 있어야 한다. 환경 셋업 전체 절차는 [`../setup.qmd`](../setup.qmd) 참조.

### Training

```bash
cd scripts
python train.py --task=DreamWaQ-Manager-Go2-Base-v0   --headless
python train.py --task=DreamWaQ-Manager-Go2-Oracle-v0 --headless
python train.py --task=DreamWaQ-Manager-Go2-Waq-v0    --headless
```

편의 스크립트:
- `run_base.sh`, `run_oracle.sh`, `run_waq.sh` — 단일 태스크 학습
- `run_all_trainings.sh` — 세 태스크 순차 실행 (default 4096 envs / 100k iters / wandb)
- `quick_test.sh` — 64 envs / 30 iters smoke test
- `watch.{py,sh}` — 학습 진행 모니터링

### Play

```bash
python play.py --task=DreamWaQ-Manager-Go2-Waq-Play-v0 --load_run=FOLDER --checkpoint=N
```

체크포인트는 `logs/rsl_rl/[experiment_name]/[timestamp]/` 에 저장된다.

---

## Task Options

| Task ID | Actor obs | Critic obs (state) | Memo |
|---------|:---------:|:------------------:|------|
| `DreamWaQ-Manager-Go2-Base-v0`   | 45      | (없음, symmetric AC) | blind |
| `DreamWaQ-Manager-Go2-Oracle-v0` | 48      | 238 | true `lin_vel` 포함 + privileged height scan |
| `DreamWaQ-Manager-Go2-Waq-v0`    | 45 → 64 | 235 | CENet (`est_vel + context`) + privileged |

각 task 에 대응하는 `-Play-v0` variant 가 별도로 등록되어 있으며, `num_envs=50`, `terrain.curriculum=False`, observation noise/push 비활성, debug viz on 상태로 평가/시각화용이다.

원본 ↔ Manager task ID 매핑:

| 원본 (`--task=...`) | ManagerBasedRLEnv (`--task=...`) |
|---|---|
| `a1_base`   | `DreamWaQ-Manager-Go2-Base-v0`   (+ `-Play-v0`) |
| `a1_oracle` | `DreamWaQ-Manager-Go2-Oracle-v0` (+ `-Play-v0`) |
| `a1_waq`    | `DreamWaQ-Manager-Go2-Waq-v0`    (+ `-Play-v0`) |

---

## Structure Mapping (IsaacGym → IsaacLab ManagerBasedRLEnv)

원본 `dreamwaq/` 는 두 라이브러리로 분리되어 있다:
- `legged_gym/` — 시뮬레이션 환경 (env / config / scripts / utils / resources)
- `rsl_rl/`    — RL 알고리즘 (PPO / actor-critic / runner / VAE)

ManagerBasedRLEnv 로 이식하면서, **시뮬레이터 plumbing / PPO 학습 루프 / actor-critic 모듈 / robot URDF 자산은 IsaacLab 이 제공**하며, 추가로 IsaacLab 의 Manager 추상화 (`ObservationManager`, `RewardManager`, `EventManager`, `CommandManager`, `CurriculumManager`, `TerminationManager`) 를 적극 활용해 원본의 `LeggedRobot` 클래스를 cfg 클래스 컴포지션으로 분해한다.

### Top-level package layout

```
dreamwaq_manager                          # 외부 프로젝트 루트 (Isaac Lab 템플릿 레이아웃)
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
└── source/dreamwaq_manager/              # 설치 가능한 확장: pip install -e source/dreamwaq_manager
    ├── config/extension.toml             # Kit 확장 매니페스트 + deps
    ├── docs/CHANGELOG.rst
    ├── pyproject.toml                    # build-system only
    ├── setup.py                          # extension.toml 읽음 + find_packages()
    ├── README.md
    └── dreamwaq_manager/
        ├── __init__.py
        ├── algorithms/                   # rsl_rl/rsl_rl/  대응 (필요한 부분만)
        │   ├── cenet.py                  # ← rsl_rl/.../vae/cenet.py
        │   ├── estnet.py                 # ← rsl_rl/.../vae/estnet.py
        │   └── dreamwaq_runner.py        # ← rsl_rl/.../runners/on_policy_runner.py
        │                                 #   (OnPolicyRunnerWAQ만 추출)
        │
        ├── tasks/locomotion/             # legged_gym/legged_gym/envs/  대응
        │   ├── velocity_env_cfg.py       # ← envs/base/legged_robot.py + legged_robot_config.py
        │   │                             #   (Scene/Cmd/Action/Obs/Event/Reward/Term/Curriculum
        │   │                             #    Cfg 클래스 7종으로 분해)
        │   ├── mdp/
        │   │   ├── __init__.py           # IsaacLab 기본 mdp + 본 패키지 custom 모두 노출
        │   │   ├── rewards.py            # ← legged_robot.py:_reward_*  (custom 4종)
        │   │   ├── events.py             # ← legged_robot.py:_push_robots (disturb 트래킹)
        │   │   └── observations.py       # ← legged_robot.py:compute_observations 의 disturb_force 슬롯
        │   └── config/go2/
        │       ├── __init__.py           # ← envs/__init__.py (gym.register)
        │       ├── go2_base_cfg.py       # ← envs/go2/go2_config.py:Go2RoughBaseCfg
        │       ├── go2_oracle_cfg.py     # ← envs/go2/go2_config.py:Go2RoughOracleCfg
        │       ├── go2_waq_cfg.py        # ← envs/go2/go2_config.py:Go2RoughWaqCfg
        │       └── agents/
        │           └── rsl_rl_ppo_cfg.py # PPO + CENet runner config
        │
        └── utils/                        # 보조 유틸 (rerun visualizer 등)
```

---

## File-by-file correspondence

### Environment / config

| 원본 (`dreamwaq/`) | Manager (`dreamwaq_manager/`) | Notes |
|---|---|---|
| `legged_gym/envs/__init__.py` (`task_registry.register`) | `tasks/locomotion/config/go2/__init__.py` | gymnasium 네이티브 `gym.register`. `task_registry` 폐지. |
| `legged_gym/envs/go2/go2_config.py` (`Go2Rough{Base,Oracle,Waq}Cfg`) | `tasks/locomotion/config/go2/go2_{base,oracle,waq}_cfg.py` | `DreamWaQVelocityRoughEnvCfg` 상속. Go2 specific overrides (PD gains, spawn height, terrain 축소). |
| `legged_gym/envs/base/legged_robot.py` (`LeggedRobot`) | `tasks/locomotion/velocity_env_cfg.py` (Manager Cfg 7종) + `mdp/` | Manager 가 step loop / reset / reward / obs 를 모두 흡수. 원본의 `LeggedRobot.step()` 안의 모든 함수가 `ObservationManager` / `RewardManager` / `EventManager` 호출로 분해됨. |
| `legged_gym/envs/base/legged_robot_config.py` (`LeggedRobotCfg`) | `tasks/locomotion/velocity_env_cfg.py` (`DreamWaQVelocityRoughEnvCfg` + 7 Cfg 클래스) | `ManagerBasedRLEnvCfg` 상속. domain_rand → `EventCfg`, rewards.scales → `RewardsCfg`, noise → `Unoise(n_min, n_max)`. |
| `legged_gym/utils/task_registry.py` | (제외) | gymnasium native registration 으로 대체. |
| `legged_gym/utils/terrain.py` (custom Terrain class) | (제외) | `isaaclab.terrains.TerrainImporterCfg` + `ROUGH_TERRAINS_CFG` 로 대체. |
| `legged_gym/utils/logger.py` (matplotlib) | (제외) | wandb / tensorboard 사용. |
| `legged_gym/resources/robots/a1/` (URDF + meshes) | (제외) | A1 → Go2 platform 변경. `isaaclab_assets.UNITREE_GO2_CFG` (USD asset) 사용. |

### Scripts

| 원본 | Manager | Notes |
|---|---|---|
| `legged_gym/scripts/train.py` | `scripts/rsl_rl/train.py` | log_std clamp `[-5, 2]` monkey-patch 동일 (policy std 폭발 방지). |
| `legged_gym/scripts/play.py` | `scripts/rsl_rl/play.py` | |
| `legged_gym/scripts/mini_test.py` | (제외) | |

### Algorithms (RSL-RL)

| 원본 (`dreamwaq/rsl_rl/`) | Manager (`dreamwaq_manager/algorithms/`) | Notes |
|---|---|---|
| `rsl_rl/algorithms/ppo.py` | (제외) | `isaaclab_rl.rsl_rl` (`RslRlPpoAlgorithmCfg`) 재사용. |
| `rsl_rl/modules/actor_critic.py` | (제외) | `RslRlMLPModelCfg` 사용 (hidden=[512,256,128], activation=elu). |
| `rsl_rl/runners/on_policy_runner.py` (`OnPolicyRunner`) | (제외 — IsaacLab 기본 runner) | Base / Oracle 모델용. |
| `rsl_rl/runners/on_policy_runner.py` (`OnPolicyRunnerWAQ`) | `algorithms/dreamwaq_runner.py` (`OnPolicyRunnerWaq`) | Waq 모델 전용. CENet 학습 루프 통합. |
| `rsl_rl/runners/on_policy_runner.py` (`OnPolicyRunnerEst`) | (제외) | EstNet 비교 모델은 사용하지 않음. |
| `rsl_rl/utils/rms.py` (`RunningMeanStd`) | `algorithms/cenet.py` 내부 | CENet 의 normal prior 학습용. |
| `rsl_rl/vae/cenet.py` | `algorithms/cenet.py` (`CENet`, `CenetRolloutStorage`) | 거의 그대로 이식. |
| `rsl_rl/vae/estnet.py` | `algorithms/estnet.py` (`EstNet`) | 거의 그대로 이식. |

`dreamwaq_direct` 와 동일한 algorithms/ 코드를 사용 — 두 패키지의 학습된 체크포인트는 호환된다.

---

## Manager API mapping

원본 `LeggedRobot` 의 단일 클래스 안에 모든 step / reset / reward 가 들어있던 것을 IsaacLab 의 7개 Manager Cfg 로 분해한 매핑이다. 모든 Cfg 클래스는 `velocity_env_cfg.py` 에 정의된다.

| 원본 함수 / 메서드 | Manager Cfg | 위치 |
|---|---|---|
| `_compute_torques`, action scale | `ActionsCfg.joint_pos` (`mdp.JointPositionActionCfg`) | `velocity_env_cfg.py:ActionsCfg` |
| `_resample_commands` + `_post_physics_step_callback` (heading) | `CommandsCfg.base_velocity` (`mdp.UniformVelocityCommandCfg`) | `velocity_env_cfg.py:CommandsCfg` |
| `compute_observations` (actor obs) | `ObservationsCfg.PolicyCfg` (ObsTerm × 6) | `velocity_env_cfg.py:ObservationsCfg.PolicyCfg` |
| `compute_observations` (privileged) | `ObservationsCfg.CriticCfg` (ObsTerm × 9) | `velocity_env_cfg.py:ObservationsCfg.CriticCfg` |
| `_reward_*` (13개) | `RewardsCfg` (RewTerm × 13) | `velocity_env_cfg.py:RewardsCfg` + custom in `mdp/rewards.py` |
| `check_termination` (base contact + timeout) | `TerminationsCfg.{time_out, base_contact}` | `velocity_env_cfg.py:TerminationsCfg` |
| `_update_terrain_curriculum` | `CurriculumCfg.terrain_level` (`mdp.terrain_levels_vel`) | `velocity_env_cfg.py:CurriculumCfg` |
| `_process_rigid_shape_props` (friction) | `EventCfg.physics_material` (`mdp.randomize_rigid_body_material`) | `velocity_env_cfg.py:EventCfg` |
| `_process_rigid_body_props` (mass + CoM) | `EventCfg.{add_base_mass, base_com}` | `velocity_env_cfg.py:EventCfg` |
| `_init_buffers` PD scaling | `EventCfg.randomize_pd_gains` (`mdp.randomize_actuator_gains`) | `velocity_env_cfg.py:EventCfg` |
| `_push_robots` + `disturb_force` 트래킹 | `EventCfg.push_robot` (`mdp.push_robot_with_disturb`) + `mdp.last_disturb_force` ObsTerm | `mdp/events.py`, `mdp/observations.py` |
| `_reset_root_states` | `EventCfg.reset_base` (`mdp.reset_root_state_uniform`) | `velocity_env_cfg.py:EventCfg` |
| `_reset_dofs` | `EventCfg.reset_robot_joints` (`mdp.reset_joints_by_scale`) | `velocity_env_cfg.py:EventCfg` |

### Custom rewards / events / obs 추가 위치

원본의 4개 custom reward 와 disturb_force 추적용 event/obs 는 IsaacLab 기본에 없어 새로 작성:

| 함수 | 파일 | 원본 |
|---|---|---|
| `joint_power_l1` | `mdp/rewards.py` | `legged_robot.py:_reward_joint_power` (L1602-1606) |
| `power_distribution_l2` | `mdp/rewards.py` | `legged_robot.py:_reward_power_distribution` (L1608-1611) |
| `foot_clearance_l2` | `mdp/rewards.py` | `legged_robot.py:_reward_foot_clearance` (L1699-1711) |
| `action_smoothness_l2` (jerk) | `mdp/rewards.py` (ManagerTermBase, t-2 buffer) | `legged_robot.py:_reward_smoothness` (L1629-1637) |
| `push_robot_with_disturb` | `mdp/events.py` | `legged_robot.py:_push_robots` (L793-802) |
| `last_disturb_force` | `mdp/observations.py` | `legged_robot.py:355` (`disturb_force` slot in priv obs) |

---

## Key Implementation Details

### Manager 패턴

- **장점**: cfg 만 수정해서 reward/event 조합 변경 가능 (코드 수정 없이 ablation). IsaacLab 기본 mdp 함수 재사용성 높음.
- **단점**: 원본 IsaacGym 의 inline 흐름과 직접 비교가 어려움. cfg → manager 호출 사이클이 indirection 단계 추가.
- **`dreamwaq_direct` 와 alignment 유지**: 두 구현은 reward / observation / domain randomization 의미상 동일. wandb log key 도 동일 (`Episode_Reward/track_lin_vel_xy` 등).

### Robot / scene

- `UNITREE_GO2_CFG` (USD asset, `isaaclab_assets`) + PD override (`stiffness=20.0`, `damping=0.5`).
- Spawn height `0.42 m` — `boxes` sub-terrain (≤ 0.1 m) + xy ±1 m reset 랜덤화 시 충돌 방지 마진 확보. (paper 의 0.34 m 보다 높음.)
- Height scanner: `RayCasterCfg` (`/Robot/base` offset z=20 m), `GridPattern` resolution=0.1, size=[1.6, 1.0] → 187 rays.

### Terrain

- `ROUGH_TERRAINS_CFG` (10 levels × 20 types, 6 sub-terrains).
- Go2 크기에 맞춰 `boxes.grid_height_range`, `random_rough.noise_range/step` 축소.
- `terrain_levels_vel` 커리큘럼 (`CurriculumCfg.terrain_level`) — reset 시 거리 기반 ±1 level.

### Privileged observation

원본 IsaacGym 의 critic 입력 = `obs_buf ⊕ privileged_obs_buf` (runner 에서 concat) 와 동일하게 구성:

```
[clean actor obs] + disturb_force(3) + heights(187)
Oracle: 48 + 190 = 238
Waq   : 45 + 190 = 235  (lin_vel 제외, Go2WaqEnvCfg 에서 critic.base_lin_vel = None)
```

### Domain randomization (`EventCfg`)

| Term | mode | params | 원본 |
|---|---|---|---|
| `physics_material` | startup | friction U(0.2, 1.25), 64 buckets | `_process_rigid_shape_props` |
| `add_base_mass` | startup | base mass + U(-1, +2) kg | `_process_rigid_body_props` |
| `base_com` | startup | CoM xyz ± 5 cm, all bodies | `_process_rigid_body_props` |
| `randomize_pd_gains` | startup | stiffness/damping × U(0.9, 1.1) | `_init_buffers` (per-env scaling) |
| `reset_base` | reset | xy ±1 m, vel ±0.5 (lin+ang) | `_reset_root_states` |
| `reset_robot_joints` | reset | pos × U(0.5, 1.5), vel = 0 | `_reset_dofs` |
| `push_robot` | interval (1 s) | xyz lin_vel ±1.0, **disturb tracked** | `_push_robots` |

### Observation noise

원본 `_get_noise_scale_vec` 의 effective 값 (= `noise_scales × noise_level × obs_scales`):
- `lin_vel`: ±0.2 (Oracle 만 actor obs)
- `ang_vel`: ±0.05
- `gravity`: ±0.05
- `dof_pos`: ±0.01
- `dof_vel`: ±0.075

Variant 별 enable_corruption:
- **Base, Oracle**: `enable_corruption = True` (원본 `add_noise = True` 기본)
- **Waq**: `enable_corruption = False` (원본 `Go2RoughWaqCfg.noise.add_noise = False`)

---

## Direct vs Manager (comparison with `dreamwaq_direct`)

| 항목 | `dreamwaq_direct` | `dreamwaq_manager` |
|---|---|---|
| 기반 API | `DirectRLEnv` | `ManagerBasedRLEnv` |
| 메인 파일 | `dreamwaq_env.py` (단일 클래스 inline) | `velocity_env_cfg.py` (7 Cfg 클래스) |
| Reward 코드 | 모두 inline (`_get_rewards`) | RewTerm × 13 (IsaacLab 기본 + custom) |
| Domain randomization | inline IsaacGym-style 함수 (`_randomize_friction` 등) | `EventCfg` + manager 함수 |
| Reset | `_reset_idx` 안에 `_reset_root_states` / `_reset_dofs` 직접 호출 | `EventManager` 가 reset 모드 EventTerm 자동 호출 |
| Push event | `step()` 안에서 interval check | `EventCfg.push_robot` (interval mode) |
| Step 루프 | `_pre_physics_step` / `_apply_action` / `_get_*` hook | Manager 가 자동 호출 |
| 원본 IsaacGym 과 직접 비교 | 쉬움 (1:1 매핑) | manager 추상화 통과 필요 |
| Cfg 만으로 ablation | 어려움 (코드 수정 필요) | 쉬움 (RewTerm/EventTerm enable/disable) |

두 패키지는 **동일한 결과를 내야** 하며, 서로의 cross-check 역할을 한다. 차이는 upstream repo 의 `comparison.qmd` 에서 분석.

---

## See Also

- [`../README.md`](../README.md) — 이 repo 개요
- [`../setup.qmd`](../setup.qmd) — IsaacLab 설치 / 자산 다운로드
- [`../dreamwaq_direct/`](../dreamwaq_direct/) — 동일 알고리즘의 DirectRLEnv 구현 (cross-check용, 종료 조건 차이 있음)

아래 문서는 이 강의용 repo 에는 포함되지 않으며 [upstream repo](https://github.com/curieuxjy/IsaacLab_DreamWaQ) 에 있다:

- `index.qmd` — 프로젝트 전체 개요 (Quarto 홈)
- `comparison.qmd` — Manager ↔ Direct 비교 분석
- `plan.qmd` — IsaacGym → IsaacLab 마이그레이션 계획
- `report.qmd` — 실험 결과 리포트
- `dreamwaq/README.md` — IsaacGym 원본 README (read-only reference)
