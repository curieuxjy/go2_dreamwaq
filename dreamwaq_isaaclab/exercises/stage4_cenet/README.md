# Stage 4 — 지형 context 학습 CENet (VAE)

**이 강의의 제목이 걸린 단계다.** 앞의 세 stage 는 여기를 위한 준비였다.

DreamWaQ 가 푸는 문제는 하나다 — **로봇은 자기 몸통 속도를 재는 센서도, 지형을 보는
카메라도 없다.** 관절 각도와 IMU 만 있다. 그런데 거친 지형을 걸으려면 둘 다 필요하다.

CENet 의 답: **과거 관측 이력(225) 만으로 "다음에 무슨 일이 일어날지"를 복원하도록
강제하면**, 그 잠재 벡터에 지형 정보가 저절로 스며든다. 이것이 논문이 말하는
*implicit terrain imagination* 이다. 지형을 직접 배우라고 시키지 않는다.

## 3단으로 쌓는다

```
task01_vae     순수 VAE            (L3)  <- toy 데이터. Isaac Sim·로봇 없음
cenet-forward  CENet 구조로 확장    (L2)  <- 실제 cenet.py 의 forward
cenet-loss     CENet 의 손실        (L2)  <- 실제 cenet.py 의 update
runner-augment 러너에 결합          (L2)  <- actor 관측 45 -> 64
```

각 단계는 앞 단계에서 쓴 것을 그대로 키운다.

| | task01 (toy) | CENet (실제) |
|---|---|---|
| 입력 | `x` (8) | `obs_history` (225 = 45 × 5 프레임) |
| 잠재 | `mu, logvar` (2) | `mu, logvar` (16) + **`est_vel` (3)** |
| 출력 | `x_hat` (8) — 자기 자신 | `est_next_obs` (45) — **다음 스텝** |
| 손실 | recon + β·KL | recon + β·KL + **vel** |

차이는 두 군데뿐이다.

1. **`est_vel` 가지가 하나 더 붙는다.** 인코더 출력의 앞 3 차원을 속도 추정에 쓰고,
   실제 base linear velocity 로 지도학습한다. 이것이 "속도 센서 없이 속도를 아는" 부분이다.
2. **복원 대상이 자기 자신이 아니라 다음 스텝 관측이다.** 자기 자신을 복원하면
   압축만 배운다. 다음을 복원해야 *동역학* 을 배우고, 거기에 지형이 들어 있다.

## 실습 목록

| # | 실습 | 레벨 | 방식 | 검사 |
|:---:|---|:---:|---|:---:|
| 01 | [`task01_vae`](task01_vae/) | **L3** | 독립 스크립트 | 20 |
| 02 | [`cenet-forward`](cenet-forward/) | **L2** | 실제 `cenet.py` | 18 |
| 03 | [`cenet-loss`](cenet-loss/) | **L2** | 실제 `cenet.py` | 14 |
| 04 | [`runner-augment`](runner-augment/) | **L2** | 실제 `dreamwaq_runner.py` | 12 |

02~04 는 `dreamwaq_manager/` 의 **프로덕션 소스를 그대로 잘라** 만든 것이다
(`exercises/specs/*.toml`). 소스는 한 글자도 건드리지 않는다.

## 순서를 지킨다

`task01_vae` → `cenet-forward` → `cenet-loss` → `runner-augment`.

01 을 건너뛰고 02 로 가면 reparameterization trick 을 실제 코드 안에서 처음 만나게 된다.
toy 에서 한 번 손으로 써 보고 가는 편이 훨씬 빠르다.

## 검증

전부 Isaac Sim 없이 1초 안에 끝난다. 다만 **번들 kit python 으로 돌려야 한다** —
시스템 `python3` 에는 torch 가 없다.

```bash
PY=~/IsaacLab/_isaac_sim/python.sh
cd exercises/stage4_cenet/task01_vae && $PY check.py
cd ../cenet-forward   && $PY check.py
cd ../cenet-loss      && $PY check.py
cd ../runner-augment  && $PY check.py
```

`--solution` 을 붙이면 완성본(프로덕션 소스)을 검사한다. 검사기가 이상하다 싶을 때 쓴다.

starter 가 없다면 생성한다.

```bash
python3 exercises/tools/make_exercise.py --id vae-core
python3 exercises/tools/make_exercise.py --id cenet-forward
python3 exercises/tools/make_exercise.py --id cenet-loss
python3 exercises/tools/make_exercise.py --id runner-augment
```

## 다 하고 나면

Waq task 가 실제로 돈다. Base(45) 와 달리 actor 가 64 차원을 받는다.

> **아래 명령이 도는 것은 내가 채운 `starter/cenet.py` 가 아니다.** 이 stage 의 `starter/` 는
> 프로덕션 소스를 잘라 만든 **읽기·연습용 사본**이고, `train.py` 가 import 하는 것은 설치된
> 패키지 쪽 원본(`dreamwaq_manager/source/dreamwaq_manager/dreamwaq_manager/algorithms/`)이다.
> 그래서 이 명령이 확인해 주는 것은 "**Waq 배선이 돈다**" 이지 "내 답이 맞다" 가 아니다 —
> 내 답의 정오는 `check.py` 가 판정한다.
>
> 자기 답으로 진짜 학습을 돌려 보고 싶으면 프로덕션 소스의 **해당 함수 본문만** 자기 것으로
> 바꾼다. starter 파일을 통째로 복사해 덮으면 실습을 만들며 가려 둔 것(`DWQ_CENET_*` 실험
> 게이트, 잠재 진단 로깅 `cenet_kl_active_dims` / `cenet_kl_nats` / `cenet_dec_w_*`)까지
> 함께 지워진다. 되돌릴 때는 `git checkout` 이 있다.

```bash
cd dreamwaq_manager
python scripts/rsl_rl/train.py --task=DreamWaQ-Waq-Official-Rough-PPO-v0 \
    --headless --num_envs=64 --max_iterations=30
```

> **`-Official-` 이 처음 보이는 이유.** repo 에는 env 레시피가 두 벌 있다. 논문 보상을
> 그대로 옮긴 `DreamWaQ-Manager-Go2-*` 레시피는 **걷지 못해 폐기됐고**(몸통 접촉 종료 78%),
> 실제 학습·비교는 Isaac Lab 공식 Go2 보상을 쓰는 `*-Official-*` 계열로 돌린다.
> CENet 코드는 양쪽이 동일하므로 이 stage 에서 쓴 것이 그대로 들어간다.
> 대조표는 [`PAPER.md`](../../PAPER.md) §0 과 루트 [`README.md`](../../README.md) 의 Task 절에 있다.

**여기서 확인할 것은 "성능" 이 아니라 "돌아간다" 뿐이다.** 제대로 된 비교는
강사가 미리 돌려 둔 **4000 iteration** 산출물로 Stage 5 에서 한다.
