# 실습 (exercises)

DreamWaQ를 Isaac Sim / Isaac Lab 위에서 **직접 쌓아 올리는** 5단계 실습이다.
완성본은 이 repo의 `dreamwaq_manager/` · `dreamwaq_direct/` 에 있고, 실습은 그 코드를
단계별로 도려내 만든 것이다.

## 실습 강도 — 4단계

각 실습에는 레벨 라벨이 붙는다. 지금 무엇을 요구받는지 먼저 확인하고 시작한다.

| 레벨 | 이름 | 하는 일 |
|:---:|---|---|
| **L0** | READ | 실행해서 출력만 확인한다. 코드는 읽기만 한다 |
| **L1** | FILL | `TODO` 빈칸 1~5줄을 채운다. 골격은 주어진다 |
| **L2** | BUILD | 시그니처와 docstring, 테스트만 주어진다. 본문을 전부 쓴다 |
| **L3** | DESIGN | 요구사항만 주어진다. 파일을 통째로 쓴다 |

L0에 시간을 쓰지 않는다. 중요한 것은 L2·L3에 몰려 있다.

## 검증은 두 계층

반복 속도가 학습 경험을 좌우한다. Isaac Sim 기동에만 25초가 걸리므로,
가능한 검증은 전부 Isaac Sim 없이 돌게 만들었다.

| | 명령 | 소요 | 무엇을 보는가 |
|---|---|---|---|
| **빠른 검증** | `python check.py` | ~1초 | shape·수식·경계조건. 합성 텐서만 사용 |
| **느린 검증** | `quick_test.sh` / `python solution.py` | ~40초~ | 실제 env가 뜨고 몇 iteration 도는지 |

빠른 검증을 먼저 통과시키고, 마지막에 한 번 느린 검증을 돌린다.

느린 검증은 실습별 스크립트가 아니라 **단계별로 다르다.**

- **Stage 1** — 각 `solution.py` / `starter.py` 가 스스로 Isaac Sim 을 띄우고 `[PASS]` 를 찍는다
- **Stage 2·4** — 프로젝트 모듈을 고치므로 `dreamwaq_manager/scripts/quick_test.sh` (64 envs / 30 iters)

> **내가 채운 `starter/` 는 학습에 들어가지 않는다.** Stage 2·4 의 `starter/` 는 프로덕션
> 소스를 잘라 만든 **읽기·연습용 사본**이고, `train.py` / `quick_test.sh` 가 import 하는 것은
> 설치된 패키지의 원본(`dreamwaq_manager/source/.../algorithms/cenet.py` 등)이다. 즉 느린
> 검증은 **내 답이 아니라 프로덕션 소스가 도는지**를 본다. 자기 답으로 실제 학습을 돌려 보고
> 싶으면 프로덕션 소스의 해당 **본문만** 자기 것으로 바꾼다 — starter 파일을 통째로 덮어쓰면
> 실습을 만들며 가려 둔 것(`DWQ_CENET_*` 게이트, 잠재 진단 로깅 등)까지 함께 지워진다.
> 되돌릴 때는 `git checkout` 이 있다.
- **Stage 3** — `python starter.py` 가 toy 환경에서 실제로 학습한다 (CPU 40초)

> 검사기는 **번들 kit python** 으로 돌린다 — 시스템 `python3` 에는 torch 가 없다.
> ```bash
> ~/IsaacLab/_isaac_sim/python.sh check.py
> ```

## 단계 구성

| Stage | 주제 | 형식 | 실습 | 구성 |
|:---:|---|---|:---:|---|
| **1** | Isaac Sim / Lab 살펴보기 | 독립 스크립트 | 5 | L0 ×1 · L1 ×3 · L2 ×1 |
| **2** | 학습 Env 클래스 (Manager vs Direct) | 프로젝트 모듈 부분완성 | 6 | L1 ×2 · L2 ×4 |
| **3** | 학습 알고리즘 PPO | 독립 스크립트 + 코드 독해 | 4 | L0 ×1 · L1 ×1 · L2 ×2 |
| **4** | 지형 context 학습 CENet (VAE) | 독립 → 프로젝트 통합 | 4 | **L2 ×3 · L3 ×1** |
| **5** | inference 실험 결과 비교 | 배포 자산 분석 | 4 | L0 ×1 · L2 ×2 · 서술 ×1 |

전체 목록은 `python exercises/tools/make_exercise.py --list` 로 확인한다.

### Stage 1 — Isaac Sim / Lab 살펴보기

빈 씬 → Go2 스폰 → 관절 PD 제어 → 지형 생성 → 센서(contact / raycaster) 값 읽기.

앞의 넷은 L0~L1이다. **마지막 "센서 값 읽기"만 L2**로 올린다 — Stage 2에서 관측을
만들 때 그대로 쓰는 손기술이기 때문이다.

### Stage 2 — 학습 Env 클래스

이 강의의 차별점은 **같은 기능을 두 API로 두 번** 구현해 보는 것이다.

| 대상 | Manager 방식 | Direct 방식 | 레벨 |
|---|---|---|:---:|
| 관측 | `ObsGroup` + `ObsTerm` | `_get_observations` 인라인 | L2 |
| 보상 | `rewards.py` + `RewTerm` 가중치 | `_get_rewards` 인라인 | L2 |
| 종료 | `DoneTerm(illegal_contact)` | `_get_dones` | **L2** |
| 리셋 / randomization | `EventTerm` | `_reset_idx` | L0 |

`velocity_env_cfg.py`(442줄)와 `dreamwaq_env.py`(834줄)를 통째로 시키지 않는다.
**보상 4개 + Direct 관측 스케일링 + Direct 종료**만 파내고 나머지는 제공한다.
Manager 쪽 `ObsGroup`/`DoneTerm` 은 import 만으로 kit 커널이 떠서 빠른 검증을 붙일 수 없다 —
읽기(L0)로 갈음한다. 자세한 이유는 `stage2_env/README.md`.

### Stage 3 — PPO

이 프로젝트는 `rsl_rl`의 PPO를 그대로 쓴다. 그래서 **분리해서** 다룬다.

1. **최소 PPO를 직접 구현** (L2/L3) — Isaac Sim 없이 도는 독립 스크립트.
   GAE, clipped surrogate, value loss, entropy 각각 빈칸
2. **`rsl_rl` PPO를 읽고 매핑** (L0) — "내가 짠 게 여기 이 클래스다"
3. **cfg 실습** (L1) — `RslRlPpoAlgorithmCfg` 값의 의미, `desired_kl` 적응형 lr,
   그리고 이 repo의 실전 이슈인 log_std clamp

### Stage 4 — CENet (VAE) ← 최대 강도

강의 제목이 걸린 부분이다. 4단으로 쌓는다.

1. **순수 VAE** `task01_vae` (L3) — toy 데이터. encode / reparameterize / decode / ELBO
2. **CENet 구조로 확장** `cenet-forward` (L2) — `obs_history(225)` → `est_vel(3)` + `context(16)`
3. **손실 세 항** `cenet-loss` (L2) — vel / recon / KL, 리덕션 축 정합과 실효 β
4. **runner 통합** `runner-augment` (L2) — actor 관측 45 → 64 증강, rollout storage 타이밍,
   `update()` 호출 지점

toy VAE 와 CENet 의 차이는 두 군데뿐이다 — `est_vel` 가지가 붙고, 복원 대상이 자기 자신이
아니라 **다음 스텝 관측**이다. 자세한 것은 `stage4_cenet/README.md`.

### Stage 5 — inference 실험 결과 비교

**전제: 학습자는 full training을 돌리지 않는다.** 배포 자산은 4096 envs × **4000 iter**
로 돌렸고 rough 1 run 이 약 2시간 50분, 6 run 이면 10시간이 넘는다. 대신 강사가 미리 돌린
**6개 run의 체크포인트 · 영상 · tensorboard 로그를 배포 자산으로 쓴다**.

학습자는 자기 구현으로 `quick_test` 규모(64 envs / 30 iter)에서 "돌아간다"만 확인하고,
결과 해석은 배포 자산으로 한다.

- `play.py`로 Base / Oracle / Waq 체크포인트 비교 + 영상 (L0)
- tensorboard 로그에서 곡선을 뽑아 3종 비교 플롯 (L2)
- **Waq − Base = CENet의 순수 기여**, **Oracle − Waq = 남은 격차** 해석 (서술)
- 결과가 뒤집혔을 때 **체크포인트 포렌식**으로 CENet 붕괴를 판정 (L2)

실측 결론: rough 에서 `Base(0.5492) < Waq(0.5565) < Oracle(0.5822)` 로 논문 순서가 나왔고,
flat 에서는 `Waq−Base` 가 노이즈와 구분되지 않는다 (Base 가 이미 0.938 로 천장 근처다).
**둘 다 교재다** — 왜 그런지가 Stage 5 의 질문이다.

## 디렉토리 규약

```
exercises/
├── README.md              # 이 문서
├── tools/
│   └── make_exercise.py   # solution → starter 생성기
├── stage1_isaac_basics/
├── stage2_env/
├── stage3_ppo/
├── stage4_cenet/
└── stage5_compare/
```

각 실습 폴더는 다음을 갖는다.

```
taskNN_<name>/               # 독립 실습 (Stage 1·3, task01_vae)
├── README.md      # 목표 · 레벨 · 힌트 · 통과 기준
├── solution.py    # 완성본 (마커가 들어 있다)
├── starter.py     # 학습자가 채우는 파일 (생성물 — 직접 편집하지 않는다)
└── check.py       # 빠른 검증 (L0 실습은 없다)

<id>/                        # 프로젝트 모듈 실습 (Stage 2·4)
├── README.md
├── starter/<원본파일명>.py   # 생성물 — 프로젝트 소스의 사본에 TODO 를 뚫은 것
└── check.py
```

## starter는 손으로 만들지 않는다

`starter.py`는 **항상 solution에서 생성**한다. 두 벌을 손으로 관리하면 반드시 어긋난다.

```bash
python exercises/tools/make_exercise.py --list           # 실습 전체 보기
python exercises/tools/make_exercise.py --id cenet-loss  # starter 생성
python exercises/tools/make_exercise.py --all --check    # 최신인지 검사 (CI용)
```

선언 방식은 두 가지고, **solution 이 프로덕션 코드인지 강의 전용 코드인지**로 갈린다.

### 1. spec 파일 — 프로젝트 모듈 (Stage 2·4)

`dreamwaq_manager/` · `dreamwaq_direct/` 의 **소스는 한 글자도 건드리지 않는다.**
실습 정의는 `exercises/specs/<id>.toml` 에 두고, `body` 에 잘라낼 구간을 그대로 적는다.

```toml
id     = "cenet-loss"
level  = 2
stage  = "stage4_cenet"
source = "dreamwaq_manager/.../cenet.py"
task   = "CENet 의 세 손실 항을 완성한다"
hints  = ["...", "..."]
body   = '''            mse_loss = nn.MSELoss()
...
'''
```

생성기가 **실제 소스를 읽어** `body` 를 찾아 TODO 로 바꾼다. 소스가 바뀌어 `body` 를
못 찾으면 **큰 소리로 실패한다** — 조용히 어긋난 starter 가 나오는 일이 없다.
그때는 `body` 를 현재 코드로 갱신하면 된다.

### 2. 인라인 마커 — 독립 실습 (Stage 1·3)

이 파일들은 애초에 강의 자료라 마커를 심어도 잃을 게 없다. 마커는 주석이므로
`solution.py` 는 그대로 실행된다.

```python
# ex:begin id=pd-target level=1 stage=stage1_isaac_basics task=관절 목표를 계산한다
#   hint: joint_target = default_joint_pos + action * ACTION_SCALE
return robot.data.default_joint_pos + action * ACTION_SCALE
# ex:end
```

`solution.py` 와 `starter.py` 가 실습 폴더 안에 나란히 있다.
