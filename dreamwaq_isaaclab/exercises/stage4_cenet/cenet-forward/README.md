# cenet-forward — CENet 의 forward  `L2 · BUILD`

**Stage 4 · CENet (VAE)**

## 목표

`task01_vae` 에서 세운 VAE 를 **진짜 CENet** 으로 확장한다. `CENet.forward` 를 통째로 쓴다.

toy VAE 와 다른 점은 딱 하나 — 인코더 출력에 **속도 추정 가지가 하나 더 붙는다.**

```
obs_history ─▶ encoder ─▶ h (latent_dim1)
                          │
              ┌───────────┴───────────┐
              ▼           ▼           ▼
          est_vel(v)    mu(c)     logvar(c)
              │           └─────┬─────┘
              │                 ▼
              │              reparam
              │                 ▼
              │          context_vec(c)
              └────────┬────────┘
                       ▼
                cat (latent_dim2) ─▶ decoder ─▶ est_next_obs
```

`v` 는 속도 차원, `c` 는 context 차원이다. 인코더 출력 `h` 하나에 세 조각이 들어 있고,
그것을 정확히 갈라 쓰는 것이 이 실습이다. **고정된 것은 `v` 뿐이다** — `c` 는 cfg 가 정하므로
`h` 의 마지막 축 크기에서 재야 하고, 숫자를 코드에 박아 두면 cfg 를 바꿨을 때 깨진다.
`latent_dim1` / `latent_dim2` 로 역산하지 않는다. 그 둘은 cfg 가 잘못 주어졌을 때도
서로 정합한 것처럼 보여서, 잘못된 cfg 를 잡아내지 못한다 (통과 기준 6).

## 왜 속도 가지를 붙이는가

Go2 에는 몸통 속도를 재는 센서가 없다. 그런데 보상 함수의 주항이
`track_lin_vel_xy_exp` — **명령 속도를 얼마나 잘 따라가는가** 다. 속도를 모르면 따라갈 수 없다.

그래서 CENet 은 관측 이력에서 속도를 **추정**하고, 학습 중에만 접근 가능한 실제 속도로
지도학습한다. 배포 시에는 추정값만 쓴다. 이것이 Base(45) 와 Waq(64) 를 가르는 3 차원이다.

## 채울 곳

```
starter/cenet.py  →  CENet.forward()  안의  TODO(cenet-forward)
```

같은 파일의 `CENet.update()` 도 비어 있다. **그쪽은 다음 실습(`cenet-loss`)의 몫**이라
일부러 지워 둔 것이고, 이 실습의 검사기는 `update()` 를 부르지 않는다. 지금은
`TODO(cenet-forward)` 만 채운다.

> **이 `starter/cenet.py` 는 프로덕션 소스와 완전히 같지는 않다.** 진짜 `cenet.py` 에는
> 환경변수로 켜는 실험용 스위치(`DWQ_CENET_*`)가 붙어 있는데, **전부 기본 off** 이고 이
> 실습과 아무 관계가 없어서 생성 시점에 걷어냈다. 무엇을 왜 켜는지는
> [`cenet-loss`](../cenet-loss/) README 의 '실제 소스에는 스위치가 더 있다' 절에 있다.

이미 주어진 것:

| | |
|---|---|
| `self.encoder` | `input_dim → 128 → 64 → latent_dim1` |
| `self.decoder` | `latent_dim2 → 64 → 128 → 48 → output_dim` |
| `self.reparameterize(mu, logvar)` | task01 에서 쓴 것과 같은 함수. **이미 구현되어 있다** |

## 반환 규약

```python
return est_next_obs, est_vel, mu, logvar, context_vec
```

이 순서를 지켜야 한다. `before_action()` 과 `update()` 가 이 순서로 받는다.

| 이름 | shape | 쓰이는 곳 |
|---|---|---|
| `est_next_obs` | `(B, output_dim)` | `recon_loss` 의 예측값 |
| `est_vel` | `(B, v)` | `vel_loss` + **actor 입력** |
| `mu`, `logvar` | `(B, c)` | `kl_loss` |
| `context_vec` | `(B, c)` | 디코더 입력 + **actor 입력** |

`est_vel` 과 `context_vec` 을 이은 것이 곧 actor 관측 증강분이다 (`runner-augment` 실습).

## 검증

```bash
cd exercises/stage4_cenet/cenet-forward
~/IsaacLab/_isaac_sim/python.sh check.py   # 빠른 검증 — Isaac Sim 불필요, 1초
```

> 검사기는 순수 torch 지만 **번들 kit python 으로 돌린다.** 시스템 `python3` 에는 torch 가 없다.

통과 기준 (18개):

1. 5개를 튜플로 반환한다
2. 다섯 텐서의 shape 이 맞다
3. **인코더 출력을 속도 추정 한 벌과 분포 파라미터 두 벌로 정확히 가른다** — 셋이 겹치지도
   남지도 않아야 하고, 검사기는 인코더를 따로 돌려 각 조각이 `h` 의 어느 자리인지 대조한다
4. **디코더가 받는 폭이 `latent_dim2` 이고, 그 잠재 부분이 분포의 평균이 아니라 표본이다** —
   검사기가 디코더 입력을 가로채 이 둘을 구분한다
5. 같은 입력에 `mu`/`logvar` 는 결정론적이고 `context_vec` 만 매번 달라진다 (표집이 실제로 일어난다)
6. 분포 파라미터가 두 벌로 갈리지 않는 `latent_dim1` 을 주면 조용히 넘어가지 않고 예외를 낸다
   (어떤 예외여야 하는지는 힌트 2단계에 있다)
7. encoder·decoder 양쪽으로 gradient 가 흐른다

> 어느 항목에서 걸렸는지는 검사기가 자리와 숫자까지 찍어 준다. **먼저 틀려 보고 그 출력을 읽는
> 편**이 여기서 답을 미리 읽는 것보다 남는다.

## 힌트

<details>
<summary>1단계 — 인코더 출력을 세 조각으로</summary>

한 번에 세 조각으로 자르는 것보다, **두 번에 나눠 자르는 편**이 읽기 쉽다.
먼저 속도 가지를 떼어내고, 남은 것을 다시 절반으로 가른다.

자를 폭을 상수로 박지 않는다. 고정된 것은 속도의 차원 수뿐이고, 나머지는
`h` 의 마지막 축 크기에서 재야 한다. 그래야 cfg 가 context 크기를 바꿔도
`forward` 를 고치지 않아도 된다.
</details>

<details>
<summary>2단계 — 홀수 방어</summary>

남은 조각을 절반으로 가르려면 그 개수가 짝수여야 한다. 홀수라는 것은
`latent_dim1` 을 잘못 준 것이므로 조용히 넘어가지 말고 `AssertionError` 를 낸다.
cfg 를 잘못 준 채로 몇 시간 학습하고 나서 발견하는 것보다 낫다.
</details>

<details>
<summary>3단계 — 디코더에 무엇을 넣는가</summary>

디코더가 받는 폭은 `latent_dim2` = 19 다. 무엇과 무엇을 이어야 19 가 되는지부터 센다.

19 의 뒷부분 16 자리에 들어갈 후보가 둘이다 — 분포의 **평균**(`mu`) 이거나,
거기서 뽑은 **표본**이거나. 평균을 넣으면 무작위성이 사라져 그냥 오토인코더가 되고
KL 항이 규제할 대상이 없어진다. 검사기가 이 둘을 따로 구분해서 본다.
</details>

## 왜 `.requires_grad_(True)` 가 붙어 있나

완성본을 보면 `mu`, `logvar`, `context_vec` 에 `.requires_grad_(True)` 가 붙어 있다.
`split` 결과는 인코더 출력에서 나온 값이라 이미 gradient 를 물고 있으므로 **없어도 동작한다.**
원본 DreamWaQ 구현을 그대로 옮긴 흔적이다. 없이 써도 검사는 통과한다.

## 다음

- [`cenet-loss`](../cenet-loss/) — 이 다섯 값으로 세 손실 항을 만든다 (L2)
