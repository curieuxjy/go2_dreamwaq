# task04 — 지형 생성과 env_origins  `L1 · FILL`

## 목표

무한 평면을 벗어난다. DreamWaQ 의 "implicit terrain imagination" 은
울퉁불퉁한 지형이 있어야 의미가 생긴다.

```
terrain_type="plane"      단일 무한 평면. 가볍고 디버깅에 좋다
terrain_type="generator"  sub-terrain 타일을 num_rows x num_cols 격자로 생성
```

**격자의 행(row)이 난이도다.** 이것이 지형 커리큘럼의 토대다.

## 채울 곳

`starter.py` 의 `TODO(terrain-cfg)` — `TerrainImporterCfg` 구성 5~8줄.

| 항목 | 값 | 왜 |
|---|---|---|
| `prim_path` | `"/World/ground"` | height scanner 가 이 경로를 mesh 로 참조한다 |
| `terrain_type` | `"generator"` | |
| `terrain_generator` | `ROUGH_TERRAINS_CFG.copy()` | 원본을 고치면 안 된다 |
| `collision_group` | `-1` | 전역 그룹 — 모든 env 가 이 지형과 충돌한다 |
| `num_rows` / `num_cols` | `5` / `5` | |

## 실행

```bash
python starter.py
python solution.py --viz kit   # 지형 격자 보기
```

## 통과 기준

1. `env_origins` 가 `(num_envs, 3)` 모양
2. 원점들이 격자에 흩어져 있다 (한 점에 몰려 있지 않다)
3. sub-terrain 이 4종 이상
4. `curriculum=True`

## 실행하면 나오는 것

`ROUGH_TERRAINS_CFG` 의 구성:

| sub-terrain | 비율 |
|---|---:|
| `pyramid_stairs` / `pyramid_stairs_inv` | 0.20 / 0.20 |
| `boxes` | 0.20 |
| `random_rough` | 0.20 |
| `hf_pyramid_slope` / `_inv` | 0.10 / 0.10 |

`env_origins` 의 z 범위가 `[-1.38, +0.71] m` 로 벌어진다 — 타일마다 높이가 다르다.
**스폰 높이 0.42 m 의 근거가 여기 있다.** `boxes` 가 최대 0.1 m 이고
리셋 시 xy 를 ±0.5 m 흔들기 때문에, 낮게 스폰하면 지형에 박힌다.

## 커리큘럼과의 연결

학습 중 `mdp.terrain_levels_vel` 이 잘 걷는 로봇을 윗행으로, 못 걷는 로봇을
아랫행으로 옮긴다. 학습 로그의 **`Curriculum/terrain_levels`** 가 바로 이 행 번호의
평균이다 — Stage 5 에서 이 곡선을 읽는다.
