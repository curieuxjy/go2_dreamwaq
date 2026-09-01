# task01_vae — 순수 VAE  `L3 · DESIGN`

**Stage 4 · CENet (VAE)**

## 목표

`ToyVAE` 클래스를 **통째로** 쓴다. 골격도, 시그니처도 주어지지 않는다 — 아래 API 명세만 있다.

여기서 쓴 네 조각(encode / reparameterize / decode / ELBO)이 다음 실습에서 그대로
CENet 이 된다. 로봇도 Isaac Sim 도 없이, VAE 만 먼저 세우는 단계다.

## toy 문제

`make_toy_batch()` 는 **진짜 자유도가 2 뿐인데 8 차원으로 관측되는** 데이터를 만든다.
2차원 잠재 벡터를 고정된 랜덤 행렬로 8차원에 펼치고 약간의 노이즈를 얹은 것이다.

DreamWaQ 로 치면 "지형의 진짜 상태는 몇 개 수로 요약되는데, 로봇은 225 차원 관측 이력만
본다" 에 해당한다. VAE 가 할 일은 그 저차원 구조를 되찾는 것이다.

## 채울 곳

```
starter.py  →  TODO(vae-core)
```

L3 라 TODO 가 **모듈 최상단**에 있다. `import` 하는 순간 `NotImplementedError` 가 난다.
그 자리에 `ToyVAE` 클래스를 통째로 쓰면 된다.

`make_toy_batch()` 와 `main()` 은 이미 주어져 있다. `main()` 이 여러분의 클래스를
`ToyVAE()`, `net.loss(data)`, `net.parameters()` 로 쓰므로 **API 를 정확히 맞춰야 한다.**

## 요구 API

```python
class ToyVAE(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32, latent_dim=2, beta=1.0)
    def encode(self, x)            -> (mu, logvar)          # 각 (B, latent_dim)
    def reparameterize(self, mu, logvar) -> z               # (B, latent_dim)
    def decode(self, z)            -> x_hat                 # (B, input_dim)
    def forward(self, x)           -> (x_hat, mu, logvar, z)
    def loss(self, x)              -> (total, recon, kl)    # 스칼라 텐서 3개
```

규약 두 가지를 지킨다. **CENet 이 쓰는 규약과 같게 맞춰 둔 것이다.**

- `loss` 가 돌려주는 `kl` 에는 **`self.beta` 를 이미 곱해 둔다.** 따라서 `total = recon + kl`.
- KL 은 **잠재 차원으로 평균내고, 배치로도 평균**낸다. `recon` 이 `nn.MSELoss()` —
  즉 입력 차원당 평균이기 때문이다. **두 항의 리덕션 축을 맞추는 것**이 규약의 핵심이다
  (왜 그런지는 아래 "β 가 하는 일" 절).

## 검증

```bash
cd exercises/stage4_cenet/task01_vae
~/IsaacLab/_isaac_sim/python.sh check.py     # 빠른 검증 — Isaac Sim 불필요, 1초
~/IsaacLab/_isaac_sim/python.sh starter.py   # 직접 학습시켜 본다 (~5초)
```

> 검사기는 순수 torch 지만 **번들 kit python 으로 돌린다.** 시스템 `python3` 에는 torch 가 없다.

통과 기준 (20개):

1. `encode` 가 `mu`, `logvar` 를 각각 `(B, latent_dim)` 으로 준다 — 같은 절반을 두 번 쓰지 않는다
2. `reparameterize`
   - `logvar` 가 아주 작으면 `z ≈ mu` (즉 `std = exp(0.5 * logvar)`)
   - 같은 인자로 두 번 부르면 **다른 표본**이 나온다
   - `logvar = 0` 이면 표본의 표준편차가 대략 1
   - `mu` 로 **gradient 가 흐른다** ← 이게 핵심이다
3. `decode` / `forward`
   - shape 과 반환 개수
   - **같은 `x` 로 `forward` 를 두 번 부르면 `x_hat` 이 달라진다** ← 표집이 `forward` 경로에
     실제로 들어가 있어야 한다. `decode(mu)` 로 쓰면 shape 은 다 맞는데 여기서 걸린다
4. ELBO
   - `total == recon + kl`
   - `mu = 0, logvar = 0` 이면 `kl == 0`
   - `mu` 나 `logvar` 가 0 에서 멀어지면 `kl` 증가
   - `kl` 이 잠재 차원으로 **평균**내져 있다 (`mu=1, logvar=0` → `kl == 0.5`).
     검사기는 `latent_dim` 을 2 → 4 로 늘려도 값이 변하지 않는지로 판별한다
5. `beta` 를 2배 하면 `kl` 이 정확히 2배
6. `x_hat == x` 이면 `recon == 0`
7. 200 step 학습으로 손실이 실제로 줄어든다

## 힌트

<details>
<summary>1단계 — 인코더가 분포를 낸다</summary>

보통의 오토인코더는 잠재 **벡터**를 낸다. VAE 는 잠재 **분포**를 낸다.
그래서 인코더 출력이 `latent_dim * 2` 다 — 앞 절반이 `mu`, 뒤 절반이 `logvar`.

`torch.split` 이나 슬라이싱으로 가른다. 왜 `var` 가 아니라 `logvar` 인가?
분산은 양수여야 하는데, 신경망 출력은 부호 제약을 걸기 어렵다.
로그로 받으면 `exp` 가 알아서 양수로 만들어 준다.
</details>

<details>
<summary>2단계 — reparameterization trick</summary>

`z ~ N(mu, sigma²)` 에서 그냥 뽑으면 **미분할 수 없다.** 표집은 미분 가능한 연산이 아니다.

대신 이렇게 쓴다.

```
std = exp(0.5 * logvar)
eps ~ N(0, I)            <- 무작위성은 전부 여기로 몰아낸다
z   = mu + eps * std     <- mu, std 로는 gradient 가 그대로 흐른다
```

`logvar` 가 아니라 `0.5 * logvar` 를 `exp` 하는 이유는 `logvar = log(σ²)` 이므로
`σ = exp(0.5 · log σ²)` 이기 때문이다.
</details>

<details>
<summary>3단계 — ELBO</summary>

두 항이다.

- **복원**: `x_hat` 이 `x` 와 얼마나 다른가 → MSE
- **규제**: `q(z|x)` 가 `p(z) = N(0, I)` 에서 얼마나 벗어났는가 → KL

대각 가우시안끼리의 KL 은 닫힌 형태가 있다.

```
KL(q || p) = -0.5 * sum_j ( 1 + logvar_j - mu_j² - exp(logvar_j) )
```

교과서 형태는 `sum_j` (잠재 차원 합)이지만, 여기서는 **잠재 차원 평균**을 쓴다 —
`recon` 이 `nn.MSELoss()` = 차원당 평균이라 축을 맞춘 것이다. 그 뒤 배치 평균,
마지막에 `beta` 를 곱한다.
</details>

## β 가 하는 일

`beta` 를 키우면 규제가 세져 잠재 분포가 표준정규에 가까워진다.
너무 세면 **posterior collapse** — 잠재 벡터가 입력과 무관해져 아무 정보도 담지 못한다.
너무 약하면 그냥 오토인코더가 된다.

`beta=1.0` 으로 한 번, `beta=10.0` 으로 한 번 돌려 `recon` 이 어떻게 달라지는지 직접 본다.

```bash
~/IsaacLab/_isaac_sim/python.sh starter.py                # beta = 1.0
~/IsaacLab/_isaac_sim/python.sh starter.py --beta 10.0
```

### 무엇을 볼 것인가

| | `kl` (마지막) | `recon` (마지막) |
|---|---|---|
| `beta=1.0` | 0.59 | **0.48** |
| `beta=10.0` | **0.003** | **1.77** |

`beta=10` 에서 `kl` 이 0.003 으로 **바닥에 붙는다.** 규제만 보면 "잘 됐다" 처럼 보인다.
그런데 `recon` 은 0.48 → 1.77 로 **올랐다.**

**`kl`→0 인데 `recon` 이 따라 내려오지 않는 것이 posterior collapse 의 신호다.**
`q(z|x)` 가 입력과 무관하게 `N(0,I)` 로 주저앉아 잠재가 아무 정보도 나르지 않고,
디코더는 잠재를 무시한 채 평균만 뱉는다. 잠재를 못 쓰니 복원이 나빠질 수밖에 없다.
(`kl` 이 작으면서 `recon` 도 함께 작아진다면 그건 붕괴가 아니라 진짜로 잘 된 것이다.)

**그리고 toy 의 `beta=10` 은 실효 β 로는 40 이다** (`D/L = 8/2 = 4`, 아래 표).
옛 CENet 의 실효 β 45 와 사실상 같은 자리다. **방금 눈으로 본 것이 CENet 에서 일어난 일이다.**

### 그런데 여러분이 쓴 `beta` 는 여러분이 생각한 `beta` 가 아니다

교과서 β-VAE 는 recon 도 KL 도 **합**(sum/sum)이다. 그 기준으로 환산하면 **실효 β** 는
두 항의 리덕션 축에 따라 달라진다.

| recon 축 | KL 축 | 실효 β |
|---|---|---|
| D차원 평균 | L차원 **합** | **D · β** ← 축이 어긋난 경우 |
| D차원 평균 | L차원 평균 | (D / L) · β ← 이 실습이 쓰는 규약 |

여기 toy 는 `D=8, L=2` 라 축을 맞추면 실효 β = 4·β 이고, 어긋나게 두면 8·β 다.
차이가 2배뿐이라 toy 에서는 잘 티가 안 난다.

**CENet 에서는 티가 났다.** `D=45, L=16` 이라 축이 어긋난 옛 코드의 실효 β 는 **45·β**,
`beta_limit=4` 까지 annealing 하면 **180** 이었다. 결과는 posterior collapse — 잠재 정보량
0.02~0.14 nats, 사실상 0비트. 지금은 KL 을 차원 평균으로 바꿔 실효 β = (45/16)·β ≈ **2.81·β**
로 내렸고, annealing 도 학습 경로에서 뺐다.

> **"논문 eq.7 의 β 는 상수다" 라고 쓰지 않는 이유.** 논문은 β 에 **수치도 스케줄도 주지
> 않는다** ("constant" 라는 단어도 없다). "스케줄을 안 적었으니 상수" 는 **우리 독법**이지
> 논문 문장이 아니다. 우리는 그 독법을 따라 상수로 구현했고, annealing 은 기본 off 인
> 실험용 게이트(`DWQ_CENET_BETA_ANNEAL`) 뒤로 옮겨 두었다.
> 어디까지가 논문이고 어디부터가 우리 선택인지는 [`cenet-loss`](../cenet-loss/) 의
> "논문이 정한 것과 우리가 정한 것" 절에 표로 정리해 두었다.

> **"실효 β" 도 우리 용어다.** 논문에 없다. `recon` 을 합으로 환산했을 때 `KL_sum` 에
> 붙는 계수, 즉 `vel` 항을 빼고 **recon : KL 만** 본 비다. 그래서 같은 학습을 두고도
> 어느 단위로 적느냐에 따라 4·β 로도 8·β 로도 쓸 수 있다 — **수를 인용할 때는 단위를
> 함께 말한다.**

그래서 이 실습이 KL 을 **차원 평균**으로 못박는다. 값 자체보다 **두 항의 축을 맞춘다**는
습관이 요점이다. 전말은 [`cenet-loss`](../cenet-loss/) 실습에서 다시 다룬다.

## 다음

- [`cenet-forward`](../cenet-forward/) — 이 구조에 `est_vel` 가지를 붙여 진짜 CENet 으로 (L2)
