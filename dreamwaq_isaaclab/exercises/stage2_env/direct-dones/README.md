# direct-dones — 종료 조건  `L2 · BUILD`

**Stage 2 · 학습 Env 클래스**

## 목표

에피소드를 언제 끝낼지 정한다. 두 가지다.

| | 무엇 | 성격 |
|---|---|---|
| `time_out` | 정해진 스텝 수를 다 썼다 | **성공적 종료** — 부트스트랩해야 한다 |
| `base_contact` | 몸통이 바닥에 닿았다 = 넘어졌다 | **실패 종료** |

이 둘을 구분하는 것이 중요하다. 시간이 다 돼서 끊긴 에피소드의 마지막 상태는 "나쁜 상태"가
아니다. 그래서 `_get_dones` 는 둘을 **따로** 반환하고, 학습기는 timeout 에 대해서만
가치를 부트스트랩한다. 하나로 합쳐 버리면 "오래 살아남는 것" 자체에 잘못된 페널티가 붙는다.

## 왜 몸통 접촉인가

원본 IsaacGym DreamWaQ 의 `legged_robot.py:check_termination` 과 **완전히 같다.**

```
reset_buf = any( norm(contact_forces[:, termination_contact_indices, :]) > 1.0 )
```

`terminate_after_contacts_on = ["base"]`, 임계값 `1.0 N`. Manager 스택은
`mdp.illegal_contact` 로 같은 바디·같은 임계값을 쓴다. **두 스택과 원본이 모두 일치한다.**

임계값 1 N 은 매우 낮다 — Go2 몸통이 무언가에 스치기만 해도 종료다. 넘어진 뒤에 버둥거리며
보상을 긁어모으는 것을 막으려는 것이다.

## 채울 곳

```
starter/dreamwaq_env.py  →  DreamWaQEnv._get_dones()  안의  TODO(direct-dones)
```

주어진 것:

| | shape / 의미 |
|---|---|
| `self.episode_length_buf` | `(N,)` 각 env 가 현재 에피소드에서 밟은 스텝 수 |
| `self.max_episode_length` | 최대 스텝 수 |
| `self._contact_sensor.data.net_forces_w_history.torch` | `(N, 3, bodies, 3)` — history 3 프레임 |
| `self._termination_contact_ids` | 몸통(base) 바디 인덱스 |
| `self.cfg.termination_contact_force` | `1.0` |

`base_contact`, `time_out` 두 변수를 만들면 아래 코드가 알아서 반환한다.

## 축 줄이기가 이 실습의 전부다

`net_forces_w_history` 는 4차원이다. 이것을 `(N,)` 불리언으로 줄여야 한다.

```
(N, history=3, bodies, 3)   힘 벡터
  → norm(dim=-1)            (N, 3, bodies)     크기로 바꾼다
  → max(dim=1)[0]           (N, bodies)        history 중 최대
  → > threshold             (N, bodies)        불리언
  → any(dim=1)              (N,)               바디 중 하나라도
```

**history 는 평균이 아니라 최대**를 쓴다. 3 프레임 중 한 번이라도 세게 부딪혔으면 넘어진 것이다.
평균을 쓰면 짧고 강한 충돌을 놓친다.

## 검증

```bash
cd exercises/stage2_env/direct-dones
python check.py            # 빠른 검증 — Isaac Sim 불필요, 2초
```

통과 기준 (10개):

1. 반환이 `(N,)` 두 개, 아무 일 없으면 둘 다 `False`
2. `episode_length >= max - 1` 인 env 만 timeout
3. 몸통에 임계값 초과 힘이 걸린 env 만 종료
4. **정확히 임계값이면 종료가 아니다** — `>` 이지 `>=` 가 아니다
5. **history 3 중 한 프레임만 부딪혀도 종료** — 평균을 쓰면 걸린다
6. **발 접촉은 종료가 아니다** — 이걸 놓치면 로봇이 첫 걸음에 죽는다
7. **힘은 성분이 아니라 벡터 norm** — `(0.8, 0.8, 0.8)` 은 성분은 다 1 미만이지만 norm 은 1.39 다
8. timeout 과 base_contact 는 서로 영향을 주지 않는다

## 힌트

<details>
<summary>왜 max_episode_length - 1 인가</summary>

`episode_length_buf` 는 0 부터 센다. `max_episode_length = 1000` 이면 유효한 스텝 인덱스는
`0..999` 다. 그래서 `>= 999` 에서 끊어야 정확히 1000 스텝이 된다.
`>= max_episode_length` 로 쓰면 한 스텝을 더 밟는다.
</details>

<details>
<summary>`.torch` 는 왜 붙는가</summary>

Isaac Lab 의 센서 데이터는 warp 배열(`ProxyArray`)로 올 수 있다. `.torch` 가 torch 텐서로
바꿔 준다. 이걸 빼면 이후 `torch.linalg.norm` 에서 타입 오류가 난다.
</details>

<details>
<summary>발 접촉을 걸러내는 법</summary>

`self._termination_contact_ids` 로 **인덱싱 먼저** 한다.
`net_contact_forces[:, :, self._termination_contact_ids]` 로 몸통 바디만 남긴 뒤에
축을 줄인다. 전부 다 줄인 다음 고르려 하면 이미 정보가 뭉개져 있다.
</details>

## Manager 쪽은 어떻게 하나

같은 일을 config 한 줄로 한다.

```python
base_contact = DoneTerm(
    func=mdp.illegal_contact,
    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0},
)
```

`mdp.illegal_contact` 안을 열어 보면 위에서 쓴 것과 **같은 축 줄이기**가 들어 있다.
Direct 는 그것을 직접 쓰고, Manager 는 이미 있는 항을 골라 쓴다 — 이것이 두 API 의 차이다.

Manager 쪽에 실습이 없는 이유는 [Stage 2 README](../README.md) 를 본다.

## 다음

Stage 2 완료다. [Stage 3 — PPO](../../stage3_ppo/) 로 간다.
