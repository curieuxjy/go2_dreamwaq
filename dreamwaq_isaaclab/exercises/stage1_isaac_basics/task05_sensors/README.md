# task05 — 센서: 접촉과 height scan  `L2 · BUILD`

**Stage 1 의 마지막이자 가장 중요한 실습이다.** 여기서 읽는 두 값이
Stage 2 에서 만들 종료 조건과 특권 관측의 원재료다.

```
ContactSensor  →  몸통이 지면에 닿았는가        →  종료 조건
RayCaster      →  로봇 주변 지형의 높이 187개   →  Oracle 의 특권 관측
```

## 채울 곳

`starter.py` 의 `TODO(height-scan)` — 한 줄이지만 **의미를 이해해야 맞다.**

`ray_hits_w[..., 2]` 는 광선이 맞은 지점의 **월드 좌표 z** 다. 이걸 그대로 관측에
넣으면 로봇이 언덕 위에 있을 때와 계곡에 있을 때 같은 지형이 전혀 다른 숫자가 된다.

관측은 **"내 몸통이 발밑 지형보다 얼마나 높은가"** 를 알려주어야 한다.

```
height_scan = base_z - hit_z - BASE_HEIGHT_TARGET
```

- `base_z` 는 `(num_envs,)`, `hit_z` 는 `(num_envs, num_rays)` — 브로드캐스트를 맞춘다
- `BASE_HEIGHT_TARGET = 0.30` 을 빼므로 **목표 높이로 서 있으면 결과가 0** 이다
- 이것이 프로젝트의 `mdp.height_scan` 이 하는 일이다

## 실행

```bash
python starter.py
python solution.py --viz kit
```

## 통과 기준

1. 광선 187개 (`17 x 11`)
2. 발 4개가 지면을 딛고 있다 (접촉력 합 > 1 N)
3. 몸통은 지면에 안 닿았다 (< 1 N)
4. 평지에서 맞은 지점 z 가 모두 0 근처
5. 상대 높이 = 몸통높이 − 지형높이 − 0.30

## 실행하면 나오는 것

```
base 접촉력       0.00 N   ← 종료 조건이 이 값을 본다 (> 1 N 이면 종료)
FL_foot 접촉력   39.46 N
FR_foot 접촉력   42.25 N
RL_foot 접촉력   33.85 N
RR_foot 접촉력   32.47 N       합 148 N ≈ Go2 무게 15 kg × 9.81
```

접촉력 합이 로봇 무게와 맞아떨어진다. 물리가 제대로 돌고 있다는 뜻이다.

## 왜 187인가

`0.1 m` 간격, `1.6 x 1.0 m` 격자 → `17 x 11 = 187`.

이 187차원이 **Oracle 만 받는 특권 정보**다.

| 모델 | actor 관측 | height scan |
|---|---:|---|
| Base | 45 | ✗ |
| Oracle | 48 | ✗ (critic 만 235차원으로 받는다) |
| Waq | 64 | ✗ — **CENet 이 과거 관측만으로 추정한다** |

DreamWaQ 의 핵심 주장이 여기 있다. actor 는 셋 다 height scan 을 직접 못 본다.
Waq 는 그 대신 과거 관측 이력에서 지형 맥락을 **상상해낸다**. 그게 Stage 4다.

## 주의: 접촉 센서는 바디 순서가 다르다

`ContactSensor` 의 바디 인덱스는 `Articulation` 의 것과 **다르다.** 섞어 쓰면
엉뚱한 바디를 읽는다 (이 프로젝트에서 실제로 났던 버그다).

```python
feet_idx, _ = contact.find_sensors(".*foot")   # 접촉 센서용
feet_ids, _ = robot.find_bodies(".*foot")      # articulation 용 — 다른 값일 수 있다
```
