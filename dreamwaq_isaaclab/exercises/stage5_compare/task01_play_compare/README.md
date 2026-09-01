# task01_play_compare — 체크포인트 3종을 눈으로 본다  `L0 · READ`

**Stage 5 · 결과 비교**

## 목표

쓸 코드는 없다. **본다.**

이 코스의 제1 교훈: **지표만 보지 말고 영상으로 보행을 확인한다.** 보상 곡선이 올라가는데
로봇은 무릎으로 기어가고 있는 경우가 실제로 있다. 숫자로는 알 수 없다.

## 준비

배포 자산을 `dreamwaq_manager/logs/` 아래에 풀어 둔다 ([Stage 5 README](../README.md) 참조).
각 run 폴더에 `videos/play/rl-video-step-0.mp4` 가 이미 들어 있다 — 먼저 그것부터 본다.

## 1. 배포된 영상 3개를 나란히 본다

```bash
cd dreamwaq_manager/logs/rsl_rl
ls */*/videos/play/*.mp4
```

rough 3종이 이 stage 의 본 비교축이다 (flat 3종도 같은 구조로 들어 있다).

| run (`experiment_name`) | actor 관측 | Play task id |
|---|---|---|
| `BaseDwq-Official-Rough-PPO-v0` | 45 — 지형도 속도도 모른다 | `DreamWaQ-BaseDwq-Rough-PPO-Play-v0` |
| `Waq-Official-Rough-PPO-v0` | 64 — CENet 이 추정 | `DreamWaQ-Waq-Official-Rough-PPO-Play-v0` |
| `OracleDwq-Official-Rough-PPO-v0` | 48 — 실제 속도를 그냥 받는다 | `DreamWaQ-OracleDwq-Rough-PPO-Play-v0` |

## 2. 직접 돌려 본다

영상은 20초(400 스텝) 한 편뿐이다. 더 보고 싶으면 직접 돌린다.
`--load_run` 에는 `experiment_name` 아래의 **timestamp 폴더 이름**을 준다.

```bash
cd dreamwaq_manager

# 화면으로 보기 (viewport)
python scripts/rsl_rl/play.py --task=DreamWaQ-BaseDwq-Rough-PPO-Play-v0 \
    --num_envs=16 --track_agent \
    --load_run=2026-08-09_15-03-53_DreamWaQ-BaseDwq-Rough-PPO-v0_seed42_envs4096 \
    --checkpoint=model_3999.pt

# 새로 영상 찍기 (headless)
python scripts/rsl_rl/play.py --task=DreamWaQ-Waq-Official-Rough-PPO-Play-v0 \
    --headless --num_envs=32 --video --video_length=400 \
    --track_agent --track_env_index=0 \
    --load_run=2026-08-09_20-38-55_DreamWaQ-Waq-Official-Rough-PPO-v0_seed42_envs4096 \
    --checkpoint=model_3999.pt
```

`--debug_vis` 를 붙이면 **명령 속도 화살표**와 **height scanner 광선**이 함께 보인다
(이 세 task 의 `*_PLAY` 설정은 debug_vis 를 켜지 않는다 — play.py 의 플래그가 켠다).
명령 화살표와 실제 진행 방향이 얼마나 어긋나는지가 바로 속도 추종 성능이다.
카메라는 `--cam_distance` / `--cam_height` 로 조절한다.

## 3. 학습 도중과 비교한다

`save_interval=500` 에 4000 iteration 이라 체크포인트가 **9개** 있다
(`model_0` / `500` / `1000` / `1500` / `2000` / `2500` / `3000` / `3500` / `3999`).
초반과 최종을 비교하면 무엇이 늘었는지 보인다.

```bash
python scripts/rsl_rl/play.py --task=... --checkpoint=model_500.pt  ...   # 비틀거린다
python scripts/rsl_rl/play.py --task=... --checkpoint=model_3999.pt ...   # 걷는다
```

## 무엇을 볼 것인가

체크리스트를 들고 본다. 그냥 보면 "다 비슷한데?" 로 끝난다.

| 관점 | 구체적으로 |
|---|---|
| **명령 추종** | 화살표 방향과 실제 진행 방향이 맞는가. 옆걸음·제자리 회전 명령에서 특히 |
| **넘어짐** | 거친 지형 경계나 계단에서 몸통이 닿는가 |
| **발 들기** | 발을 끌지 않고 드는가 (`foot_clearance` 보상이 노리는 것) |
| **떨림** | 관절이 고주파로 떨리는가 (`smoothness` 보상이 노리는 것). 실기에서 모터를 태운다 |
| **자세** | 몸통 높이가 일정한가, 주저앉아 걷지 않는가 |

## 답이 정해지지 않은 질문

- Base 는 지형도 속도도 모르는데 **왜 그럭저럭 걷는가?**
  (힌트: 관절 각도 이력에 지형 정보가 간접적으로 들어 있다. 그래서 Base 가 0 이 아니다)
- Oracle 이 상한인데도 완벽(1.0)이 아니다. rough 에서 0.582 다. **무엇이 남았는가?**
- Waq 의 보행이 Base 와 Oracle 중 어느 쪽을 더 닮았는가? 숫자(task02)와 인상이 일치하는가?
- 숫자를 미리 보면: rough 에서 Waq 는 Base 보다 **속도 추종은 낫지만**
  (0.5565 vs 0.5492) **더 자주 넘어진다** (`base_contact` 0.123 vs 0.093,
  `mean_episode_length` 919 vs 936). 영상에서 그 차이가 보이는가?
  더 공격적으로 걷는 것인가, 아니면 그냥 불안정한 것인가?

## 통과 기준

영상 3개를 보고 위 체크리스트로 차이를 말할 수 있으면 된다. `check.py` 는 없다.

## 다음

- [`task02_curves`](../task02_curves/) — 인상을 숫자로 확인한다 (L2)
