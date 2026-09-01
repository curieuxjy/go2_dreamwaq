# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project Overview

Independent implementation of **DreamWaQ** (quadrupedal locomotion via implicit terrain imagination)
on Isaac Sim / Isaac Lab. Core: **CENet** (Context-aided Estimator Network) for terrain estimation.

This repo is the **lecture extract** of [curieuxjy/IsaacLab_DreamWaQ](https://github.com/curieuxjy/IsaacLab_DreamWaQ) —
only the Isaac Sim RL packages. The IsaacGym legacy reference (`dreamwaq/`), the ROS 2 sim2sim
deployment (`deploy_sim2sim/`), and the Quarto docs site (`index.qmd`, `comparison.qmd`, `plan.qmd`,
`report.qmd`) live upstream, not here.

| Package | API | Description |
|---------|-----|-------------|
| `dreamwaq_manager/` | ManagerBasedRLEnv | Manager-composed env — **primary stack**. All training and all reported results come from here |
| `dreamwaq_direct/`  | DirectRLEnv       | Same algorithm expressed through the Direct API (cross-check). Same rewards/observations/termination. See `dreamwaq_direct/KNOWN_ISSUES.md` |

**The two packages do NOT share code — each has its own copy** of `algorithms/{cenet,estnet,dreamwaq_runner}.py`
and its own env definition. Only the *Isaac Lab authoring style* differs; the algorithm is meant to be
identical, so **any algorithmic change must be applied to both** (project rule below). The copies drifted
once already: the CENet fidelity fixes landed in Manager only and were back-ported later.

`dreamwaq_direct/` currently has **no trained artifacts** — `run_full_pipeline.sh` is Manager-only
(`pkg_dir_for()` hard-codes it).

## Common Setup

- IsaacLab at `~/IsaacLab/` on branch `release/3.0.0-beta2` (symlink `_isaac_sim` → Isaac Sim 6.0 binary). Use the bundled kit Python via `./isaaclab.sh -p` (not a system venv).
- Python 3.12 (Isaac Sim bundled), `rsl-rl-lib` ≥ 3.1
- **Hardware (RTX 4080 16GB)**: stable at 4096 envs; 8192 OOM-kills
- Models: `logs/rsl_rl/[experiment_name]/[timestamp]/` (gitignored; **not shipped with this repo**)
- WandB project: `lec_dreamwaq`
- No test suite or linter
- **Local assets (optional)**: to avoid S3 download hangs, mirror Go2 USD + UI markers to `~/IsaacLab/data/Assets/Isaac/6.0/` and patch `user.config.json` `asset_root` to the local path (see `setup.qmd` §6)
- `debug_vis=False` for headless training; `True` only in `*_PLAY` configs
- Editable installs are path-bound — after moving this repo, re-run `pip install -e source/dreamwaq_manager`

---

## Repo layout

```
lec_dreamwaq/
├── run_full_pipeline.sh          6 tasks x (train -> record video), Manager only
├── dreamwaq_manager/             PRIMARY stack (all results come from here)
│   ├── scripts/
│   │   ├── rsl_rl/{train,play,watch,collect_velocity}.py
│   │   ├── compare_runs.py       tfevents -> figures/ + summary.csv
│   │   ├── eval_checkpoints.py   every checkpoint -> best policy
│   │   └── quick_test.sh         64 envs / 30 iters smoke
│   ├── source/dreamwaq_manager/dreamwaq_manager/
│   │   ├── algorithms/           cenet.py · estnet.py · dreamwaq_runner.py
│   │   └── tasks/locomotion/
│   │       ├── velocity_env_cfg.py       DreamWaQ env (original recipe)
│   │       ├── terrains.py               equal_proportion_terrains()
│   │       ├── mdp/rewards.py            custom reward terms
│   │       └── config/go2/
│   │           ├── go2_{base,oracle,waq}_cfg.py    original recipe (ABANDONED)
│   │           ├── go2_waq_official_cfg.py         official-env variants (ACTUAL experiments)
│   │           ├── agents/rsl_rl_ppo_cfg.py        PPO runner cfgs
│   │           └── __init__.py                     gym registration
│   └── logs/                     gitignored — checkpoints · tfevents · videos · wandb
├── dreamwaq_direct/              cross-check stack, same layout, own copy of algorithms/
├── exercises/                    Stage 1-5 lecture exercises (see below)
└── figures/                      compare_runs.py output
```

## Training → evaluation flow

```
run_full_pipeline.sh
  └─ for each of 6 tasks (rough/flat x Base/Oracle/Waq):
       ├─ scripts/rsl_rl/train.py  --headless --num_envs=4096 --max_iterations=$MAX_ITER
       │    └─ OnPolicyRunner (Base/Oracle)  |  OnPolicyRunnerWaq (Waq, adds CENet)
       │         └─ writes logs/rsl_rl/<experiment_name>/<timestamp>/
       │              model_*.pt · events.out.tfevents.* · params/*.yaml · videos/
       └─ scripts/rsl_rl/play.py --video      (records videos/play/rl-video-step-0.mp4)

then, offline:
  scripts/compare_runs.py       tfevents -> figures/*.png + figures/summary.csv
  scripts/eval_checkpoints.py   rolls out EVERY checkpoint -> figures/best_policy.csv
  scripts/rsl_rl/collect_velocity.py   CENet est_vel vs true vel traces (.npz)
```

**Waq rollout inner loop** (`OnPolicyRunnerWaq.learn`, the part that differs from stock PPO):

```
base_obs, true_vel  <- observation
obs_rms.update(base_obs);      obs_history = obs_rms(history buffer)   # normalized
true_vel_rms.update(true_vel); true_vel    = true_vel_rms(true_vel)    # normalized target
cenet.before_action(obs_history, true_vel) -> est_next_obs, est_vel, mu, logvar, context
AdaBoot: vel_input = est_vel (prob boot_prob) else true_vel
actor obs = [obs_rms(base_obs)(45), vel_input(3), context(16)] (+ height_scan on rough)
env.step -> cenet.after_action(obs_rms(next_base_obs))   # recon target on the SAME scale
...rollout ends...
cenet.update()  (5 epochs x 4 minibatches, PPO 와 동일 — cfg 의 vae 블록이 기본값 1x1 을 덮어쓴다)  +  alg.update()  (5 epochs x 4 minibatches)
```

---

## `dreamwaq_manager/` — ManagerBasedRLEnv

Setup: `cd dreamwaq_manager && pip install -e source/dreamwaq_manager` (Isaac Lab template layout — package lives under `source/dreamwaq_manager/`)

Train/Play (from the `dreamwaq_manager/` project root):
```bash
python scripts/rsl_rl/train.py --task=DreamWaQ-Manager-Go2-Base-v0 --headless
python scripts/rsl_rl/train.py --task=DreamWaQ-Manager-Go2-Base-v0 --headless --num_envs=64 --max_iterations=100
python scripts/rsl_rl/play.py  --task=DreamWaQ-Manager-Go2-Base-Play-v0 --load_run=FOLDER --checkpoint=N
```

Tasks — PPO only, two families (all with `-Play-v0` variants):

| Family | Task ids | Use |
|---|---|---|
| **Official env** (comparison axis) | `DreamWaQ-{BaseDwq,OracleDwq,Waq-Official}-{Flat,Rough}-PPO-v0` | **The actual experiments.** Official Isaac Lab env recipe; only the actor observation differs across the three |
| Original DreamWaQ recipe | `DreamWaQ-Manager-Go2-{Base,Oracle,Waq}-v0` | **Abandoned** — does not learn to walk (78% trunk-contact terminations). Kept registered for reference; see `pipeline_logs_dreamwaq_env_failed/` |

Actor obs: Base 45 (proprio) / Oracle 48 (+ true base lin_vel) / Waq 64 (45 + CENet est_vel 3 + context 16).
`class_name` selects the runner: default `OnPolicyRunner` (Base/Oracle) or `OnPolicyRunnerWaq` (Waq).
`run_full_pipeline.sh` trains the six official-env tasks (default 3000 iters / 4096 envs / wandb online)
and records a tracking video after each.

Layout:
- `tasks/locomotion/velocity_env_cfg.py` — base `ManagerBasedRLEnvCfg` (DreamWaQ env; has `DWQ_OFF_PUSH`/`DWQ_OFF_REW` env-var gates)
- `tasks/locomotion/mdp/rewards.py` — custom rewards (joint_power, power_distribution, foot_clearance, smoothness)
- `tasks/locomotion/config/go2/` — `go2_{base,oracle,waq}_cfg.py` + gymnasium registration (`__init__.py`)
- `tasks/locomotion/config/go2/agents/rsl_rl_ppo_cfg.py` — the PPO runner cfgs (original 3 + official-env 6)
- `algorithms/{cenet.py, estnet.py, dreamwaq_runner.py}` — CENet, EstNet, `OnPolicyRunnerWaq`
- Uses `UNITREE_GO2_CFG` with PD override (stiffness=20.0, damping=0.5)

Scripts: `run_{base,oracle,waq}.sh` (wrappers), `run_all_trainings.sh` (sequential, default 4096 envs / 100k iters / wandb), `quick_test.sh` (64 envs / 30 iters), `watch.{py,sh}`, `collect_velocity.py` (Waq CENet velocity traces), `compare_runs.py` (tfevents → comparison figures + `figures/summary.csv`). Override: `NUM_ENVS=2048 MAX_ITER=50000 ./run_base.sh`.

**Scope rule**: PPO only. The SAC stack and DreamWaQ++ (`CENetPlus`) were removed for the lecture
repo — do not reintroduce them without being asked. They remain in git history (commit `a2d2774`)
and in the upstream repo. (The official-env comparison matrix was removed then **restored** in
`c8df7b4` — it is the actual experiment axis, keep it.)

---

## `dreamwaq_direct/` — DirectRLEnv

Same algorithm on the Direct API, kept as a cross-check of `dreamwaq_manager`. Termination is
**identical to Manager and to the original** — trunk contact force > 1 N on `base`.

**Do not re-add the `if self.device == "cpu":` guard around `scene.filter_collisions()` in
`_setup_scene`.** The official Direct examples have it; with it, inter-env collisions are never
filtered on GPU and the contact sensor reports phantom forces at 4096 envs on rough terrain
(`ep_len` pins at 1). `InteractiveScene` filters unconditionally on PhysX, which is why Manager was
unaffected. See `dreamwaq_direct/KNOWN_ISSUES.md`.

Setup: `cd dreamwaq_direct && pip install -e source/dreamwaq_direct` (Isaac Lab template layout — package lives under `source/dreamwaq_direct/`)

Train/Play (from the `dreamwaq_direct/` project root):
```bash
python scripts/rsl_rl/train.py --task=DreamWaQ-Direct-Go2-Base-v0 --headless
python scripts/rsl_rl/play.py  --task=DreamWaQ-Direct-Go2-Base-Play-v0 --load_run=FOLDER --checkpoint=N
```

Tasks: `DreamWaQ-Direct-Go2-{Base,Oracle,Waq}-v0` (+ Play variants).

Layout:
- `tasks/locomotion/dreamwaq_env.py` — core `DreamWaQEnv(DirectRLEnv)` (all logic inline):
  - `_pre_physics_step` / `_apply_action`: clip + PD targets
  - `_get_observations`: body-frame vel/gravity, obs noise, (optional) system delay, obs history, privileged obs w/ 187-ray height scan
  - `_get_rewards`: 13 DreamWaQ rewards (base_height via 187-ray mean)
  - `_get_dones`: trunk contact-force termination (`terminate_after_contacts_on=["base"]`, `termination_contact_force=1.0`) + timeout — same as the original legged_gym `check_termination` and as Manager's `mdp.illegal_contact`
  - `_reset_idx`: randomized state, terrain curriculum, resample commands, log episode sums
  - `get_obs_history()` / `get_true_vel()`: for CENet runner
- `tasks/locomotion/dreamwaq_env_cfg.py` — base `DirectRLEnvCfg`
  - `obs_noise=True` (disable for play); `system_delay=False` (enable for original paper)
- `tasks/locomotion/config/go2/go2_env_cfg.py` — Go2 configs + Play variants
- `algorithms/` — **its own copy** of CENet / EstNet / runner, kept algorithmically identical to
  Manager's. Not an import; a parallel file. Diff them after touching either
  (`diff dreamwaq_{manager,direct}/source/*/*/algorithms/dreamwaq_runner.py`). Legitimate
  differences: package name, `_extract_true_vel` (Direct reads `env.get_true_vel()`, Manager reads
  `scene["robot"].data.root_lin_vel_b`), and exteroception (Manager-only, for rough height_scan)

Scripts: `run_{base,oracle,waq}.sh`, `run_all_trainings.sh` (default 4096 envs / 5k iters / wandb), `quick_test.sh`, `watch.{py,sh}`.

---

## `exercises/` — lecture exercises

Staged fill-in-the-blank exercises built by carving pieces out of the two packages. Levels
**L0 READ / L1 FILL / L2 BUILD / L3 DESIGN**. See `exercises/README.md`.

| Stage | Topic | Exercises |
|---|---|---|
| 1 `stage1_isaac_basics` | Isaac Sim/Lab basics | task01–05 (empty scene → spawn → PD → terrain → sensors) |
| 2 `stage2_env` | Env class (Manager vs Direct) | 4 rewards + `direct-obs-scale` + `direct-dones` |
| 3 `stage3_ppo` | PPO | `task01_gae`, `task02_ppo_update`, `task03_rsl_rl_map`, `task04_ppo_cfg` |
| 4 `stage4_cenet` | CENet (VAE) | `task01_vae`, `cenet-forward`, `cenet-loss`, `runner-augment` |
| 5 `stage5_compare` | result comparison | needs trained artifacts |

**Never hand-edit `starter.py` / `starter/`** — they are generated. Two declaration mechanisms:

- **spec toml** (`exercises/specs/<id>.toml`) for production modules — the generator reads the real
  source and replaces the verbatim `body` with a TODO. Production sources stay byte-clean. If the
  source changes and `body` no longer matches, generation **fails loudly**; refresh `body`.
- **inline `# ex:begin` markers** for standalone lecture scripts (Stage 1/3, `task01_vae`).

```bash
python exercises/tools/make_exercise.py --list          # all exercises
python exercises/tools/make_exercise.py --id <id>       # regenerate one
python exercises/tools/make_exercise.py --all --check   # drift gate — run after touching any source
```

Every exercise has a fast `check.py` (~1 s, no Isaac Sim) validated both ways: it must fail on the
starter (TODO message) and pass with `--solution`. `exercises/tools/fake_isaaclab.py` stubs
`isaaclab*`/`omni`/`carb` imports so production modules load in ~1 s instead of a 25 s kit boot.

> Run checkers with the bundled kit python (`~/IsaacLab/_isaac_sim/python.sh check.py`) — the system
> `python3` has no torch.

---

## 논문 대조 — `PAPER.md`

DreamWaQ 원문(arXiv:2301.10602)의 네트워크 구조 / 보상 가중치 / PPO 하이퍼파라미터 /
CENet 손실 / 도메인 랜덤화 / 커리큘럼을 **코드 값과 나란히 정리해 둔 표**가 루트의
[`PAPER.md`](PAPER.md) 에 있다. 논문 PDF 를 다시 받지 말고 여기부터 본다.

핵심만: **지금 학습되는 공식 env 레시피는 논문 보상 12항 중 5항이 없고 2항은 가중치가 1.5배다.**
그리고 코드의 **β annealing 과 AdaBoot 램프는 논문에 없는 것**이다. 수치를 바꾸면
`PAPER.md` 도 같이 고친다.

## 검증 에이전트 3종 (`.claude/agents/`)

구현을 다듬는 루프를 셋으로 나눠 둔다. 서로 역할이 겹치지 않는다.

| 에이전트 | 역할 | 코드 수정 |
|---|---|:---:|
| `paper-auditor` | 논문 대비 구현이 맞는지 비판적으로 판정 | ✗ |
| `implementer` | 코드 수정 + 실험 실행 + 수치 보고 | ✓ |
| `learner` | 학부생 수준으로 실습을 직접 풀며 이해 가능성 검증 | ✗ |

돌아가는 방식:

```
paper-auditor  ──지적──▶  implementer  ──수치 보고──▶  paper-auditor
                              ▲                            ▲
                    "이렇게 고쳐 주세요"            "논문에선 왜 이런가요"
                              └────────  learner  ─────────┘
```

- 알고리즘/설정을 바꾼 뒤 → `paper-auditor` 로 논문 정합 확인
- 실습을 추가·수정한 뒤 → `learner` 로 이해 가능성 확인 (**반드시 통과시킨다**)
- `learner` 나 `paper-auditor` 의 요청은 → `implementer` 가 반영

**세 에이전트 모두 GPU 를 공유한다.** 학습 중에는 읽기 전용 작업만 시킨다.

---

## Key Details

- Experiment names: the `experiment_name` in `rsl_rl_ppo_cfg.py` — `{BaseDwq,OracleDwq,Waq}-Official-{Flat,Rough}-PPO-v0`
  for the real experiments; `DreamWaQ-{Manager,Direct}-Go2-{Base,Oracle,Waq}-v0` for the abandoned recipe
- Manager & Direct keep **byte-separate but behaviourally identical** CENet/EstNet/runner, so a
  checkpoint from either loads in the upstream `deploy_sim2sim` stack
- Meant to be identical across both: obs noise, reset randomization, rewards, terrain curriculum,
  CENet normalization (`obs_rms` on base obs + CENet history, `true_vel_rms` on the velocity target,
  recon target stored normalized), AdaBoot ramp, log_std clamp
- **Rough terrain uses equal sub-terrain proportions** (`terrains.equal_proportion_terrains`, 1/6 each)
  instead of Isaac Lab's stairs-heavy 20/20/20/20/10/10 default
- **Spawn height `init_state.pos.z = 0.42 m`** (higher than paper's 0.34 m) to clear `boxes` sub-terrain (≤0.1 m) combined with xy ±0.5 m reset randomization
- `train.py` (both) includes a **log_std clamp [-5, 0]** monkey-patch (optimizer step-post hook) to
  prevent policy std explosion. Upper bound 0 (std ≤ 1) because actions are clipped to [-1, 1]

---

## Behavioral Guidelines

Bias toward caution over speed. Use judgment on trivial tasks.

**1. Think before coding.** State assumptions; if uncertain, ask. Surface multiple interpretations instead of picking silently. If something is unclear, stop and name what's confusing.

**2. Simplicity first.** Minimum code that solves the problem — no speculative features, no single-use abstractions, no error handling for impossible scenarios. If 200 lines could be 50, rewrite.

**3. Surgical changes.** Touch only what you must. Don't "improve" adjacent code, refactor working code, or reformat. Match existing style. Every changed line should trace to the user's request. Remove only orphans *your* changes created; mention (don't delete) pre-existing dead code.

**4. Goal-driven execution.** Define verifiable success criteria before implementing. For multi-step work, state a brief plan with per-step checks, then loop until verified.

## Project-Specific Rules

- **Manager ↔ Direct alignment**: the two packages hold *separate copies* of the algorithm, so any
  change to rewards/observations/reset/CENet/runner must be applied to **both** unless intentionally
  diverging — and say so in a comment when diverging. Verify with a diff; the copies have silently
  drifted before.
- **Korean prose** in `setup.qmd` and the READMEs — match existing style.
- **Smoke test first**: run `quick_test.sh` (64 envs / 30 iters) before kicking off full training.
- **One GPU**: a 4096-env run saturates the RTX 4080. Don't launch a second Isaac Sim process
  (smoke test, play, eval) while training runs — it OOMs. Queue it instead.
- **Comparing runs**: never read a "final" value off the last few logged points. The tracking metric
  fluctuates with sd ≈ 0.03 per iteration, so a 3-point mean has a larger standard error than the
  effect being measured — it flipped the sign of `Waq−Base` once. `compare_runs.py` averages the
  last 10% and prints ± standard error; keep it that way.
- **Respect hardware**: don't raise `num_envs` above 4096 on RTX 4080 without checking VRAM headroom.
- **Headless = `debug_vis=False`**; enable only in `*_PLAY` configs.
- **Don't commit checkpoints or WandB logs**; `logs/` is gitignored.
