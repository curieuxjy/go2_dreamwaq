# MIT License
# Copyright (c) 2024 Jungyeon Lee (curieuxjy)
# https://github.com/curieuxjy
#
# Context-Aided Estimator Network (CENet) implementation
# Unofficial implementation of DreamWaQ (https://arxiv.org/abs/2301.10602)
#
# Ported from dreamwaq/rsl_rl/rsl_rl/vae/cenet.py for IsaacLab integration.

import torch
import torch.nn as nn
import torch.optim as optim


class CenetRolloutStorage:
    class Transition:
        def __init__(self):
            self.observation_histories = None
            self.true_velocities = None
            self.true_next_observations = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_history_shape, true_vel_shape, true_onext_shape, device="cpu"):
        self.device = device
        self.obs_history_shape = obs_history_shape
        self.true_vel_shape = true_vel_shape
        self.true_onext_shape = true_onext_shape

        self.observation_histories = torch.zeros(num_transitions_per_env, num_envs, *obs_history_shape, device=self.device)
        self.true_velocities = torch.zeros(num_transitions_per_env, num_envs, *true_vel_shape, device=self.device)
        self.true_next_observations = torch.zeros(num_transitions_per_env, num_envs, *true_onext_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

    def add_transitions_before_action(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observation_histories[self.step].copy_(transition.observation_histories)
        self.true_velocities[self.step].copy_(transition.true_velocities)

    def add_transitions_after_action(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.true_next_observations[self.step].copy_(transition.true_next_observations)
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observation_histories = self.observation_histories.flatten(0, 1)
        true_velocities = self.true_velocities.flatten(0, 1)
        true_next_observations = self.true_next_observations.flatten(0, 1)

        for epoch in range(num_epochs):
            for batch_idx in range(num_mini_batches):
                start = batch_idx * mini_batch_size
                end = (batch_idx + 1) * mini_batch_size
                batch_indices = indices[start:end]
                yield (
                    observation_histories[batch_indices],
                    true_velocities[batch_indices],
                    true_next_observations[batch_indices],
                )


class CENet(nn.Module):
    """Context-Aided Estimator Network.

    Encoder: obs_history(225) → est_vel(3) + mu(16) + logvar(16) = 35
    Reparameterize: mu, logvar → context_vec(16)
    Decoder: est_vel(3) + context_vec(16) = 19 → est_next_obs(45)

    Used by the DreamWaQ runner to augment actor observations with
    estimated velocity and learned context vector.
    """

    def __init__(
        self,
        num_learning_epochs=1,
        num_mini_batches=1,
        input_dim=225,
        hidden_dim1=128,
        hidden_dim2=64,
        hidden_dim3=48,
        latent_dim1=35,  # 3 + 16*2
        latent_dim2=19,  # 3 + 16
        output_dim=45,
        beta=1.0,
        beta_limit=4.0,
        learning_rate=1.0e-3,
        min_lr=1.0e-4,
        patience=100,
        factor=0.8,
        device="cpu",
    ):
        super().__init__()
        self.device = device

        # 실험용 환경변수 게이트(DWQ_CENET_*)는 이 실습을 만들 때 걷어냈다 — 전부 기본 off 라
        # 학습이 쓰는 수식은 달라지지 않는다. 게이트가 읽던 값들(free_bits / kl_reduction /
        # beta_anneal / beta_limit)도 같이 지웠다: 남겨 두면 아래 TODO 에서 KL 을 어느 축으로
        # 줄일지 정할 때 `kl_reduction == "sum"` 분기를 써야 하나 하고 헤매게 된다.
        # 왜 그런 스위치가 프로덕션 소스에 있는지는 이 실습 폴더의 README.md 를 본다.
        self.grad_clip = 1.0  # 이 파일에서 실제로 읽히는 유일한 설정값 — PPO 의 max_grad_norm 과 같은 예산
        # 진단 주석이 가리키는 문턱값(차원당 KL 이 이 위면 '살아 있는' 잠재 차원).
        # 진단 계산 자체를 아래에서 걷어냈으므로 이 실습에서는 읽는 곳이 없다.
        self.kl_active_eps = 0.01

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1), nn.ELU(),
            nn.Linear(hidden_dim1, hidden_dim2), nn.ELU(),
            nn.Linear(hidden_dim2, latent_dim1),
        )

        decoder_layers = [
            nn.Linear(latent_dim2, hidden_dim2), nn.ELU(),
            nn.Linear(hidden_dim2, hidden_dim1), nn.ELU(),
            nn.Linear(hidden_dim1, hidden_dim3), nn.ELU(),
            nn.Linear(hidden_dim3, output_dim),
        ]
        # (실험용 디코더 게이트 삽입 지점 — 이 실습에서는 걷어냈다.)
        self.decoder = nn.Sequential(*decoder_layers)

        print(f"{'CENet Structure':=^60}")
        print(f"Encoder MLP: {self.encoder}")
        print(f"Decoder MLP: {self.decoder}")
        # (게이트 설정 진단 출력은 이 실습에서 걷어냈다.)

        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        # Paper eq.7 puts a weight `beta` on the KL term but gives NEITHER a numeric value NOR
        # a schedule for it. "beta is constant" is OUR READING of that silence, not a sentence
        # from the paper. The cfg's `vae` block supplies the number.
        self.beta = beta
        self.current_epoch = 0
        self.storage = None
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor

        # Latent-health scalars. ``update()`` REPLACES this dict at the end of every call; the
        # values below are only the pre-first-update placeholders, so that a runner that reads
        # ``cenet.diagnostics`` before any update still gets every key. See update() for the
        # meaning of each key.
        _w_vel, _w_z = self._decoder_input_weight_norms()
        self.diagnostics = {
            "cenet_mu_abs": 0.0,
            "cenet_sigma": 1.0,
            "cenet_kl_active_dims": 0.0,
            "cenet_kl_nats": 0.0,
            "cenet_dec_w_vel": _w_vel,
            "cenet_dec_w_z": _w_z,
            "cenet_lr": self.learning_rate,
            "cenet_beta": self.beta,
        }

        self.transition = CenetRolloutStorage.Transition()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.factor, patience=self.patience, min_lr=self.min_lr,
        )

    def init_storage(self, num_envs, num_transitions_per_env, obs_history_shape, true_vel_shape, true_onext_shape):
        self.storage = CenetRolloutStorage(
            num_envs, num_transitions_per_env, obs_history_shape, true_vel_shape, true_onext_shape, self.device,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).requires_grad_(True)
        eps = torch.randn_like(std)
        return mu + eps * std

    def train_mode(self):
        self.encoder.train()
        self.decoder.train()

    def test_mode(self):
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, obs_history):
        h = self.encoder(obs_history)
        est_vel, context_vec_params = h.split([3, h.size(-1) - 3], dim=-1)

        last_dim = context_vec_params.size(-1)
        if last_dim % 2 != 0:
            raise AssertionError("Not even number for context vector parameters")
        mu, logvar = context_vec_params.split(last_dim // 2, dim=-1)

        mu = mu.requires_grad_(True)
        logvar = logvar.requires_grad_(True)
        context_vec = self.reparameterize(mu, logvar).requires_grad_(True)
        latent = torch.cat([est_vel, context_vec], dim=-1)

        return self.decoder(latent), est_vel, mu, logvar, context_vec

    def _decoder_input_weight_norms(self):
        """(|W_vel|, |W_z|) of the decoder's FIRST Linear, as a mean column norm per input dim.

        The column of that weight matrix belonging to input dim i carries everything the
        decoder reads from i, so ``||W[:, i]||`` is how much the decoder actually uses i.
        Averaged over the 3 velocity columns and over the 16 latent columns respectively.
        This is the pair that separated the healthy run from the collapsed one offline
        (|W_vel| / |W_z| = 4.90 / 0.61 healthy vs 6.80 / 0.011 collapsed at it500); logging it
        makes that visible live instead of needing a checkpoint probe.
        """
        first_linear = next(m for m in self.decoder if isinstance(m, nn.Linear))
        with torch.no_grad():
            col_norms = first_linear.weight.norm(dim=0)  # [latent_dim2]
            return float(col_norms[:3].mean().item()), float(col_norms[3:].mean().item())

    # (잠재 차원별 KL 진단 도우미와 게이트용 KL 항은 이 실습에서 걷어냈다 —
    #  둘 다 지금 채워야 할 KL 식을 그대로 담고 있어서 답안지가 된다.)

    def before_action(self, obs_history, true_vel):
        est_next_obs, est_vel, mu, logvar, context_vec = self.forward(obs_history)
        self.transition.observation_histories = obs_history
        self.transition.true_velocities = true_vel
        self.storage.add_transitions_before_action(self.transition)
        return est_next_obs, est_vel, mu, logvar, context_vec

    def after_action(self, next_obs):
        self.transition.true_next_observations = next_obs
        self.storage.add_transitions_after_action(self.transition)
        self.transition.clear()

    def update(self):
        mean_total_loss = 0
        mean_vel_loss = 0
        mean_recon_loss = 0
        mean_kl_loss = 0
        mean_mu_abs = 0.0
        mean_sigma = 0.0
        mean_kl_active = 0.0
        mean_kl_nats = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        total_loss = None

        for obs_history_batch, true_vel_batch, true_onext_batch in generator:
            est_onext_batch, est_vel_batch, mu_batch, logvar_batch, _ = self.forward(obs_history_batch)

            # ── TODO(cenet-loss) ─ level L2 ─────────────────────────────
            # CENet 의 세 손실 항(vel / recon / KL)과 총합을 완성한다
            #   hint: vel_loss 는 est_vel 과 true_vel 의 MSE, recon_loss 는 est_onext 와 true_onext 의 MSE 다
            #   hint: q(z|x)=N(mu, exp(logvar)) 와 p(z)=N(0, I) 사이의 KL 은 닫힌 형태다. std 가 아니라 logvar 를 받는다
            #   hint: nn.MSELoss() 는 차원 평균이다. KL 도 같은 축으로 축약해야 한다. 그리고 total_loss.backward() 는 스칼라를 요구한다
            #   hint: beta 가 곱해지는 것은 KL 항 뿐이다 (beta-VAE)
            # 통과 기준은 이 실습 폴더의 README.md 를 본다.
            raise NotImplementedError("TODO(cenet-loss)")
            # ─────────────────────────────────────────────────────────────────────

            # (실험용 게이트 오버라이드는 이 실습에서 걷어냈다.)

            self.optimizer.zero_grad()
            total_loss.backward()
            # PPO 와 같은 예산(1.0)으로 CENet gradient 도 자른다.
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            self.optimizer.step()

            mean_total_loss += total_loss.item()
            mean_vel_loss += vel_loss.item()
            mean_recon_loss += recon_loss.item()
            mean_kl_loss += kl_loss.item()

            # --- latent-health diagnostics ---------------------------------------------------
            # A collapse is invisible in cenet_kl alone: KL -> 0 also happens when the encoder
            # legitimately has nothing to say. What separates the two is WHERE it went to zero:
            # collapse pins mu at 0 AND sigma at exactly 1 (the prior) on every dim. Logging
            # |mu|, sigma and the count of dims still carrying KL makes it readable in
            # TensorBoard without a weight probe. Always on, in every gate mode.
            with torch.no_grad():
                mean_mu_abs += mu_batch.abs().mean().item()
                mean_sigma += (0.5 * logvar_batch).exp().mean().item()
                # 잠재 차원별 KL 진단(kl_active_dims / kl_nats)은 이 실습에서 걷어냈다 —
                # 그 계산이 곧 위에서 채워야 할 KL 식이기 때문이다. 프로덕션 소스에는 있다.

        if total_loss is not None:
            # ReduceLROnPlateau does float(metrics); handing it a grad-carrying tensor keeps the
            # graph alive for the scheduler's `best` and warns. The metric is a plain number.
            self.scheduler.step(total_loss.detach())
        self.current_epoch += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_total_loss /= num_updates
        mean_vel_loss /= num_updates
        mean_recon_loss /= num_updates
        mean_kl_loss /= num_updates

        _w_vel, _w_z = self._decoder_input_weight_norms()
        self.diagnostics = {
            "cenet_mu_abs": mean_mu_abs / num_updates,              # mean |mu| over dims+batch
            "cenet_sigma": mean_sigma / num_updates,                # mean exp(logvar/2); 1.0 = prior
            "cenet_kl_active_dims": mean_kl_active / num_updates,   # # dims with KL > kl_active_eps
            "cenet_kl_nats": mean_kl_nats / num_updates,            # total KL over all dims (nats)
            # How much the decoder actually reads from each half of its input. |W_z| collapsing
            # toward 0 while |W_vel| keeps growing IS the failure mode, and it shows up here
            # before mu/sigma have finished flattening.
            "cenet_dec_w_vel": _w_vel,
            "cenet_dec_w_z": _w_z,
            # The CENet has its own ReduceLROnPlateau, separate from PPO's adaptive lr, and it
            # was never logged: with min_lr=1e-4 it can silently sit 10x below the initial 1e-3.
            "cenet_lr": self.optimizer.param_groups[0]["lr"],
            # beta as USED by this update (annealing, if on, is applied after this point).
            "cenet_beta": self.beta,
        }

        self.storage.clear()

        # (β annealing 은 실험용 게이트라 이 실습에서 걷어냈다. 논문 eq.7 은 β 에 스케줄을
        #  주지 않는데, "그러니 상수" 는 논문 문장이 아니라 우리 독법이다 — README 를 본다.)

        return mean_total_loss, mean_vel_loss, mean_recon_loss, mean_kl_loss
