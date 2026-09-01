# task02_curves — 곡선과 요약표  `L2 · BUILD`

**Stage 5 · 결과 비교**

## 목표

task01 에서 **눈으로 본** 차이를 **숫자로** 확인한다.

핵심 함수 `summarize()` 를 쓴다. 보상 로그에서 비교 가능한 지표를 만들고,
강의의 결론인 두 델타를 계산한다.

## 왜 총 보상을 그대로 쓰면 안 되는가

`Train/mean_reward` 는 **여러 보상 항의 합**이다 (공식 레시피는 10항을 로깅한다).
값이 올랐어도 속도 추종이 좋아진 것인지 에너지 페널티가 줄어든 것인지 알 수 없다.
게다가 스케일이 임의라 "얼마나 잘하는지" 감이 없다.

그래서 속도 추종 항만 떼어내 **가중치로 되돌린다.**

```
underlying = Episode_Reward/track_lin_vel_xy_exp ÷ 1.5
           = mean exp(-‖v_cmd − v‖² / 0.25)      ∈ [0, 1]
```

이제 `1.0` 은 완벽 추종이라는 절대적 의미를 갖는다. 가중치를 나눠 주지 않으면
보상 설계를 바꿀 때 값이 같이 흔들려 run 끼리 비교가 안 된다.

> **1.5 는 어디서 온 수인가.** 배포 자산 6런은 전부 공식 레시피(`*-Official-*`)로 돌렸고,
> 그 가중치는 Isaac Lab 의 `UnitreeGo2RoughEnvCfg`
> (`isaaclab_tasks/.../locomotion/velocity/config/go2/rough_env_cfg.py`) 가 정한다.
> Stage 2 에서 본 `velocity_env_cfg.py` 의 DreamWaQ 레시피는 항 이름이
> `track_lin_vel_xy` 이고 가중치가 **1.0** 이다 — 그 런을 분석한다면 `weight=1.0` 을
> 넘겨야 한다. 실제 run 의 값은 `logs/.../params/env.yaml` 에서 확인한다
> (`weight: 1.5`). 대조표는 `PAPER.md` §2.

## 채울 곳

```
starter.py  →  summarize()  안의  TODO(stage5-summary)
```

```python
def summarize(series: dict[str, list[float]],
              weight: float = 1.5,
              last_k: int = 300) -> dict[str, float]:
```

| 인자 | |
|---|---|
| `series` | `{"Base": [로그값...], "Waq": [...], "Oracle": [...]}` — 없는 config 는 빠져 있거나 빈 리스트 |
| `weight` | 보상 가중치. 이걸로 나눠 underlying 으로 되돌린다 |
| `last_k` | 최종값을 낼 때 평균낼 마지막 로그 개수. **기본 300** — 아래 "왜 300 인가" 참조 |

반환 (rough 배포 자산의 실제 값):

```python
{"Base": 0.5496, "Waq": 0.5567, "Oracle": 0.5824,
 "Waq-Base": +0.0071, "Oracle-Waq": +0.0257}
```

**데이터가 없는 항목은 키 자체를 넣지 않는다.** `0.0` 으로 채우면 아직 학습하지 않은 run 이
"성능 0" 으로 그려져 그림이 거짓말을 한다.

## 왜 300 인가 — 이 실습에서 제일 중요한 부분

`underlying` 은 iteration 마다 **sd ≈ 0.03 으로 출렁인다.** 우리가 재려는 차이는 ~0.01 이다.
**노이즈가 신호보다 크다.**

마지막 3점만 평균내면 평균의 표준오차가 `0.03/√3 ≈ 0.017` 이라 차이의 두 배쯤 된다.
아래는 배포 자산(seed 42, 4000 it)을 창 크기만 바꿔 가며 다시 잰 것이다.
`compare_runs.py --last-k K` 로 직접 재현할 수 있다.

**rough** — 부호는 안 뒤집히지만 **크기가 4배 부풀려진다.**

| 창 크기 | Base | Waq | Oracle | **Waq−Base** | 판정 |
|---|---|---|---|---|---|
| last 3 | 0.5341 | 0.5643 | 0.5796 | **+0.0302** ± 0.0186 | 노이즈와 구분 안 됨 |
| last 30 | 0.5451 | 0.5536 | 0.5842 | **+0.0085** ± 0.0082 | 노이즈와 구분 안 됨 |
| last 100 | 0.5487 | 0.5603 | 0.5843 | **+0.0116** ± 0.0046 | 유의 |
| last 400 (기본) | 0.5492 | 0.5565 | 0.5822 | **+0.0073** ± 0.0021 | 유의 |

**flat** — 여기서는 **부호가 뒤집힌다.**

| 창 크기 | Base | Waq | Oracle | **Waq−Base** | 판정 |
|---|---|---|---|---|---|
| last 3 | 0.9412 | 0.9290 | 0.9402 | **−0.0121** ± 0.0110 | 노이즈와 구분 안 됨 |
| last 400 (기본) | 0.9383 | 0.9390 | 0.9460 | **+0.0007** ± 0.0004 | 노이즈와 구분 안 됨 |

last-3 만 보고 rough 를 썼다면 CENet 의 기여를 **실제의 4배**로 적었을 것이다.
flat 이라면 아예 "CENet 이 해롭다" 고 썼을 것이다. 창 크기 하나가 결론을 만든다.

> 이건 가정이 아니라 이 프로젝트가 실제로 겪은 일이다. 보관된 이전 런
> (`logs/_archive_3000it_stairheavy/`) 에서는 rough 의 `Waq−Base` 가
> last-3 에서 **+0.0398 ± 0.0193**, last-300 에서 **−0.0166 ± 0.0025** 로 **정반대**였다.
> 더 나쁜 것은 last-3 도 `|d| > 2·se` 라 **"유의" 판정을 받았다**는 점이다 —
> 표준오차를 함께 보는 것만으로는 부족하고, **창이 수렴 구간을 덮어야** 한다.
> 그 런은 CENet 이 붕괴해 있었다 — 원인 추적은
> [`task04_diagnose`](../task04_diagnose/) 에서 한다.

수렴 구간에서 충분히 많은 점을 평균내고, **차이를 표준오차와 함께 보고**한다.
`compare_runs.py` 가 `평균 ± 표준오차` 와 "유의 / 노이즈와 구분 안 됨" 판정을 함께 찍는 이유다.
그 기본 창은 고정된 300 이 아니라 **런 길이의 10%** (최소 50) 라, 4000 it 런에서는 400 이 된다.
여기 `summarize()` 는 실습이라 기본값을 300 으로 고정해 둔다.

## 검증

```bash
cd exercises/stage5_compare/task02_curves
python3 check.py           # 빠른 검증 — 합성 로그만 사용, 1초 (torch 도 필요 없다)
~/IsaacLab/_isaac_sim/python.sh starter.py   # 배포 자산이 있으면 실제 수치로 출력
```

`starter.py` 는 tfevents 를 읽으므로 tensorboard 가 있는 번들 python 으로 돌린다.
`check.py` 는 합성 로그만 쓰므로 시스템 `python3` 으로도 된다.

통과 기준 (13개):

1. `underlying = 로그값 / weight` — 가중치를 되돌린다
2. **최종값은 마지막 `last_k` 개의 평균** + **기본값이 100 이상**이어야 한다
3. `Waq-Base`, `Oracle-Waq` 계산
4. **Waq 가 Base 보다 나쁘면 델타가 음수** — `abs()` 를 쓰면 방향이 사라진다
5. 없는 config 는 키 자체가 없다. 짝이 없으면 델타도 없다
6. 빈 리스트도 "없음" 으로 다룬다
7. 로그가 `last_k` 개보다 적어도 동작한다
8. `weight` / `last_k` 인자가 실제로 반영된다 (상수 하드코딩 금지)

## 그림까지 그리는 완성본

`summarize()` 는 지표만 낸다. 곡선·막대·CSV 까지 만드는 전체 구현은 이미 있다.

```bash
cd dreamwaq_manager
~/IsaacLab/_isaac_sim/python.sh scripts/compare_runs.py
```

```
figures/curves_{flat,rough}.png     속도추종 곡선 (3종)
figures/reward_{flat,rough}.png     총 에피소드 보상 곡선
figures/terrain_level_rough.png     지형 커리큘럼 승급
figures/bar_final.png               최종 속도추종 막대
figures/summary.csv                 최종값 + 두 델타
```

`compare_runs.py` 를 읽으면 `summarize()` 가 어디에 해당하는지 보인다 —
`plot_curves()` 의 `final_value` 부분과 `write_summary()` 다.

## 힌트

<details>
<summary>tfevents 를 직접 읽는 법</summary>

`solution.py` 의 `load_run_series()` 에 이미 있다 (채울 필요 없다).

```python
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator(str(path), size_guidance={"scalars": 0})
ea.Reload()
vals = [e.value for e in ea.Scalars("Episode_Reward/track_lin_vel_xy_exp")]
```

`size_guidance={"scalars": 0}` 이 "전부 읽어라" 라는 뜻이다. 기본값은 샘플링해서 버린다.
</details>

<details>
<summary>빠진 config 를 어떻게 다루나 (통과 기준 5·6·7)</summary>

- 값이 `last_k` 개보다 적으면 **있는 것만** 평균낸다. 빈 리스트는 건너뛴다 — `sum([])/len([])` 은 0 나누기다.
- `"Waq-Base"` 는 `Waq` 와 `Base` 가 **둘 다** 있을 때만 넣는다. `"Oracle-Waq"` 도 같다.
- 빠진 config 는 결과 dict 에 **키 자체를 넣지 않는다.** `0.0` 으로 채우면 아직 학습하지 않은
  run 이 "성능 0" 으로 그려져 그림이 거짓말을 한다.
</details>

<details>
<summary>마지막 k 개 평균</summary>

`vals[-last_k:]` 는 리스트가 `last_k` 보다 짧아도 있는 만큼만 준다. 그래서 길이 검사가 따로 필요 없다.
다만 **빈 리스트**는 걸러야 한다 — `sum([]) / len([])` 은 0 나누기다.
</details>

## 다음

- [`task03_interpret`](../task03_interpret/) — 이 숫자가 무슨 뜻인지 쓴다
- [`task04_diagnose`](../task04_diagnose/) — 숫자가 기대와 다를 때 원인을 찾는다
