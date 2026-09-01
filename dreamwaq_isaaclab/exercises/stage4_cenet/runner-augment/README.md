# runner-augment — actor 관측 45 → 64  `L2 · BUILD`

**Stage 4 · CENet (VAE)**

## 목표

CENet 이 만든 값을 **actor 입력에 실제로 꽂는다.** 이 한 줄짜리 결합이 Base 와 Waq 를 가른다.

```
Base   : actor 관측 45                                   ← 지형도 속도도 모른다
Oracle : actor 관측 48 = 45 + 실제 base_lin_vel(3)        ← 특권 정보를 그냥 받는다 (상한)
Waq    : actor 관측 64 = 45 + est_vel(3) + context(16)   ← CENet 이 추정한다
```

`Waq − Base` 가 **CENet 의 순수 기여**, `Oracle − Waq` 가 **남은 격차**다.
Stage 5 에서 이 두 값을 실제 학습 곡선으로 확인한다.

## 채울 곳

```
starter/dreamwaq_runner.py  →  OnPolicyRunnerWaq._build_actor_obs()  안의  TODO(runner-augment)
```

이 메서드는 rollout 루프에서 매 스텝 불린다 (`learn()` 안, `self.alg.act()` 직전).

이미 주어진 것:

| | |
|---|---|
| `self.obs_rms` | base 관측(45) 용 **running mean/std 정규화기** (`rsl_rl.modules.EmpiricalNormalization`). 러너가 rollout 중 매 스텝 raw 관측으로 통계를 갱신한다 |
| `self.true_vel_rms` | 같은 종류의 정규화기지만 **실제 base linear velocity(3) 전용**. CENet 의 속도 타깃을 정규화하는 데 쓴다 |
| `self._norm_base(base_obs)` | base 관측(45)을 `obs_rms` 로 정규화 |
| `self._extract_extero(src_obs)` | `src_obs` 에서 height_scan 부분만 떼어냄 |
| `self.use_exteroception` | rough 면 `True`, flat 이면 `False` |

## 정규화를 어디에만 거는가

**base 관측 45 에만** 건다. `vel_input` 과 `context_vec` 은 정규화하지 않고 그대로 잇는다.

이유: `est_vel` 은 이미 `true_vel_rms` 로 정규화된 공간에서 학습된 값이고,
`context_vec` 은 KL 규제 덕에 대략 표준정규 근처에 머문다. 둘 다 이미 스케일이 맞는다.
여기에 `obs_rms`(45 차원용 통계)를 또 먹이면 차원 수부터 맞지 않는다.

## 검증

```bash
cd exercises/stage4_cenet/runner-augment
~/IsaacLab/_isaac_sim/python.sh check.py   # 빠른 검증 — Isaac Sim 불필요, 1초
```

> 검사기는 순수 torch 지만 **번들 kit python 으로 돌린다.** 시스템 `python3` 에는 torch 가 없다.

검사기는 `_build_actor_obs` 를 클래스에서 떼어내 **가짜 self** 로 부른다. 러너 전체를
띄우려면 env 가 필요하지만, 이 메서드는 self 의 세 가지만 쓰기 때문에 가능하다.

통과 기준 (12개):

1. flat 에서 `45 + 3 + 16 = 64` 차원
2. 순서가 `[norm(base), vel, context]` 다
   - 앞 45 는 **정규화된** base
   - 그 다음 3 은 `vel_input` **원본 그대로**
   - 마지막 16 은 `context_vec` **원본 그대로**
3. rough 에서 height_scan(187) 이 **맨 뒤**에 붙어 251 차원. 앞 64 는 flat 일 때와 같다
4. `NaN` → `0.0`, `+inf` → `10.0`, `-inf` → `-10.0`
5. **유한한 큰 값은 그대로 통과한다** — `clamp` 가 아니라 `nan_to_num` 이어야 한다

## 힌트

<details>
<summary>1단계 — 순서가 곧 규약이다</summary>

`torch.cat` 의 순서가 actor 의 입력 레이아웃을 정의한다. 학습 때와 배포 때가 같은 순서를
써야 한다. 여기서 순서를 바꾸면 체크포인트가 upstream `deploy_sim2sim` 스택과 호환되지 않는다.

`get_inference_policy()` 도 같은 메서드를 부르므로, 한 곳만 고치면 학습·평가·배포가 함께 따라온다.
</details>

<details>
<summary>2단계 — exteroception 은 조건부로</summary>

rough 지형은 critic 이 height_scan 을 받는다. `use_exteroception` 이 `True` 면
`self._extract_extero(src_obs)` 를 **맨 뒤에** 덧붙인다. 앞의 64 는 건드리지 않는다.
</details>

<details>
<summary>3단계 — 왜 nan_to_num 인가</summary>

CENet 이 발산하면 `est_vel` 이나 `context_vec` 에 `NaN` 이 뜬다. 그대로 actor 로 들어가면
정책 가중치가 통째로 `NaN` 이 되어 학습이 조용히 죽는다. 몇 시간 뒤에야 알아차린다.

`torch.nan_to_num(..., nan=0.0, posinf=10.0, neginf=-10.0)` 은 **비정상 값만** 바꾸고
정상 값은 건드리지 않는다. `clamp` 를 쓰면 멀쩡한 큰 값까지 잘라 학습 신호가 왜곡된다.

아래 rollout 타이밍 5번에 `clamp` 가 나오는데 모순이 아니다. **CENet 출력(`est_vel`,
`context_vec`)은 호출자가 `_build_actor_obs` 를 부르기 *전에* 이미 ±10 으로 clamp 해 둔다.**
`_build_actor_obs` 안으로 들어온 시점에는 이미 완성된 관측이고, 여기서 45 차원 base 관측까지
싸잡아 자르면 정상 관측이 왜곡된다. 그래서 이 안에서는 `clamp` 가 아니라 `nan_to_num` 이다.
</details>

## rollout 안에서의 호출 타이밍 (읽기)

이 메서드는 `learn()` 의 rollout 루프 한가운데 있다. 순서가 중요하다.

```
1. base_obs, true_vel 을 관측에서 꺼낸다
2. obs_rms.update(base_obs)      <- 정규화 통계를 raw 값으로 먼저 갱신
3. obs_history = 정규화된 이력
4. cenet.before_action(...)      <- forward + rollout storage 에 (history, true_vel) 저장
5. est_vel / context_vec 을 ±10 으로 clamp   <- CENet 출력만. 호출자가 미리 자른다
6. _build_actor_obs(...)         <- 여기. 이미 잘린 값을 받으므로 안에서 또 clamp 하지 않는다
7. alg.act() -> env.step()
8. cenet.after_action(정규화된 next_obs)  <- recon 타깃 저장, storage.step += 1
```

두 가지를 눈여겨본다.

- **4 와 8 이 짝이다.** `before_action` 은 `(관측 이력, 실제 속도)` 를, `after_action` 은
  `다음 스텝 관측` 을 저장한다. env 를 밟기 전과 후로 나뉘어 있어서 한 transition 이 완성된다.
- **8 에서 저장하는 recon 타깃은 `_norm_base(next_base_obs)` — 정규화된 값이다.**
  인코더 입력(정규화된 이력)과 디코더 타깃의 스케일을 맞추기 위해서다.
  여기서 raw 값을 저장하면 `recon_loss` 가 오염되어 CENet 이 제 일을 못 한다.

`cenet.update()` 는 rollout 이 끝난 뒤 PPO update 와 함께 한 번 불린다.

## 다 하고 나면

Stage 4 완료다. Waq task 가 실제로 돈다.

```bash
cd dreamwaq_manager
python scripts/rsl_rl/train.py --task=DreamWaQ-Waq-Official-Rough-PPO-v0 \
    --headless --num_envs=64 --max_iterations=30
```

> **왜 실습에서 보던 `DreamWaQ-Manager-Go2-Waq-v0` 가 아니라 `-Official-` 인가?**
> repo 에는 env 레시피가 두 벌 있다. 논문 보상을 그대로 옮긴 `Manager-Go2-*` 레시피는
> **걷지 못해서 폐기됐고**(몸통 접촉 종료 78%), 실제 실험은 Isaac Lab 공식 Go2 보상을 쓰는
> `*-Official-*` 계열로 돌린다. CENet 코드는 양쪽이 똑같으니 여기서 고친 것이 그대로 쓰인다.
> 자세한 대조는 [`PAPER.md`](../../../PAPER.md) §0 에 있다.

`Learning iteration` 이 찍히고 CENet 손실 3종이 함께 나오면 성공이다.
성능 비교는 Stage 5 에서 강사 배포 자산으로 한다.
