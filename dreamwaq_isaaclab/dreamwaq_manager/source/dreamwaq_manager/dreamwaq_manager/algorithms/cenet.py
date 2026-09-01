# MIT License
# Copyright (c) 2024 Jungyeon Lee (curieuxjy)
# https://github.com/curieuxjy
#
# Context-Aided Estimator Network (CENet) implementation
# Unofficial implementation of DreamWaQ (https://arxiv.org/abs/2301.10602)
#
# Ported from dreamwaq/rsl_rl/rsl_rl/vae/cenet.py for IsaacLab integration.

import os

import torch
import torch.nn as nn
import torch.optim as optim


class _DecoderVelGate(nn.Module):
    """Structural ablation on the decoder's velocity input. **Default: not inserted.**

    Only built when ``DWQ_CENET_DEC_VEL`` is ``zero`` or ``detach``. It sits in front of the
    decoder MLP and modifies the first ``num_vel`` entries of the latent:

    * ``zero``   — the decoder cannot see ``est_vel`` at all (removes the recon-target
      competition between the supervised velocity head and z).
    * ``detach`` — the decoder still sees ``est_vel``, but the recon loss no longer shapes the
      encoder's velocity head through it (only ``vel_loss`` does).

    Both are a DIVERGENCE FROM THE PAPER (Fig. 2 feeds ``[v_hat, z]`` into the decoder, with
    full gradient flow; the IsaacGym original does the same), so they are EXPERIMENTAL ONLY and
    stay off by default. Confirmed by the paper audit: do not turn these on for a reported run.

    Caveat: inserting this module shifts the ``decoder.<i>`` state_dict indices by one, so a
    checkpoint trained with the gate on will not load into a default CENet (and weight probes
    that read ``decoder.0.weight`` will find this module instead).
    """

    def __init__(self, mode: str, num_vel: int = 3):
        super().__init__()
        self.mode = mode
        self.num_vel = num_vel

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        vel, ctx = latent.split([self.num_vel, latent.size(-1) - self.num_vel], dim=-1)
        if self.mode == "zero":
            vel = torch.zeros_like(vel)
        else:  # "detach"
            vel = vel.detach()
        return torch.cat([vel, ctx], dim=-1)


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

        # --- experiment gates (env vars) ----------------------------------------------------
        # ALL OFF BY DEFAULT: with none of them set, every number produced by this class is
        # bit-for-bit what it was before the gates existed. They exist so the rough-Waq
        # posterior collapse (|mu| 2.9e-2 -> 6.3e-4, KL 3.7e-1 -> 5.4e-6 between two runs of the
        # SAME config) can be attacked one variable at a time on the GPU.
        #
        # DWQ_CENET_REFERENCE=1 — "IsaacGym-original alignment" preset. Restored on user
        #   instruction ("the IsaacGym implementation got the CENet right, follow it"); it
        #   re-enables things the §8-1 commit removed. Which side is right is to be decided by
        #   experiment, not by this comment. (That file, dreamwaq/rsl_rl/rsl_rl/vae/cenet.py, is
        #   THIS project's IsaacGym ancestor — not code released by the paper's authors. The
        #   paper fixes neither the KL reduction nor beta, so none of (a)-(e) is a paper
        #   requirement; the argument for them is that they are what actually walked on
        #   IsaacGym.) It flips FIVE training rules to match that file (and the Go2 cfg it ran
        #   with):
        #     (a) KL reduction  klds.sum(1)  instead of our klds.mean(1)   (16x)
        #     (b) beta annealing  beta = min(beta*1.01, beta_limit=4.0)  per update
        #         (constant beta=1.0 here). Reaches the 4.0 cap after ~139 updates.
        #     (c) min_lr = learning_rate  -> ReduceLROnPlateau becomes a no-op
        #     (d) NO gradient clipping (our clip_grad_norm_(1.0) is an addition)
        #     (e) 1 epoch x 1 minibatch per update (our cfg does 5 x 4 = 20 grad steps)
        #   CAVEAT on (c): min_lr == learning_rate is true of the IsaacGym CENet *class
        #   defaults* (1e-3 / 1e-3). Its Go2 config, the one actually trained, uses
        #   learning_rate=0.01, min_lr=0.0015 — a scheduler that CAN decay 6.7x. Those two
        #   readings disagree; this preset takes the class-default one and leaves
        #   DWQ_CENET_LR / DWQ_CENET_MIN_LR to reach the other.
        #
        # Individual gates (each overrides the preset when set explicitly):
        #   DWQ_CENET_BETA=<x>              KL weight. The paper (eq.7) never fixes a numeric
        #                                   beta, so LOWERING it is the only collapse lever that
        #                                   stays inside the paper. Our mean(1) reduction makes
        #                                   beta=1.0 equivalent to a textbook sum/sum beta of
        #                                   45/16 = 2.81, so beta=0.35 ~ textbook beta=1.
        #                                   Candidates: 0.35, 0.1.
        #   DWQ_CENET_FREE_BITS=<nats>      per-dim KL floor (free bits). The collapse mechanism
        #                                   is that the decoder stops needing z at all (decoder
        #                                   |W_z| = 0.011 at it500, never recovering), so a floor
        #                                   that stops pushing an unused dim to the prior is the
        #                                   standard fix. 0.0 = off. Candidates: 0.02 .. 0.05.
        #                                   NOTE: an ADDITION not present in the paper — record
        #                                   it in PAPER.md if a reported run uses it.
        #   DWQ_CENET_KL_REDUCTION=mean|sum reduction over latent dims after free bits.
        #   DWQ_CENET_BETA_ANNEAL=0|1       beta *= 1.01 per update, capped at beta_limit.
        #   DWQ_CENET_BETA_LIMIT=<x>        the cap (default 4.0, the IsaacGym value).
        #   DWQ_CENET_GRAD_CLIP=<norm>      0 disables clipping (IsaacGym has none).
        #   DWQ_CENET_LR=<lr>               initial Adam lr.
        #   DWQ_CENET_MIN_LR=<lr>           ReduceLROnPlateau floor (ours 1e-4 vs initial 1e-3).
        #                                   LOW PRIORITY: the "collapse was locked in by a
        #                                   decayed lr" hypothesis is already falsified — the
        #                                   measured lr trajectories of the collapsed and the
        #                                   healthy run are the same. Kept for completeness.
        #   DWQ_CENET_EPOCHS / DWQ_CENET_MINI_BATCHES   update count per rollout.
        #   DWQ_CENET_DEC_VEL=zero|detach   structural, see _DecoderVelGate. EXPERIMENTAL ONLY:
        #                                   it leaves the paper (Fig. 2 decodes from [v_hat, z]).
        self.reference_mode = os.environ.get("DWQ_CENET_REFERENCE") == "1"
        _ref = self.reference_mode

        self.free_bits = float(os.environ.get("DWQ_CENET_FREE_BITS", "0.0"))
        self.kl_reduction = os.environ.get("DWQ_CENET_KL_REDUCTION", "sum" if _ref else "mean").lower()
        if self.kl_reduction not in ("mean", "sum"):
            raise ValueError(f"DWQ_CENET_KL_REDUCTION must be 'mean' or 'sum', got {self.kl_reduction!r}")

        _beta_env = os.environ.get("DWQ_CENET_BETA")
        if _beta_env is not None:
            beta = float(_beta_env)
        self.beta_anneal = os.environ.get("DWQ_CENET_BETA_ANNEAL", "1" if _ref else "0") == "1"
        self.beta_anneal_rate = 1.01  # hard-coded in the IsaacGym original too
        self.beta_limit = float(os.environ.get("DWQ_CENET_BETA_LIMIT", beta_limit))

        _clip = float(os.environ.get("DWQ_CENET_GRAD_CLIP", "0.0" if _ref else "1.0"))
        # PPO clips at max_grad_norm=1.0; the IsaacGym CENet had no clip at all and its
        # first-layer weight norm ran away (7.6 -> 62.6 over a run). Same budget here.
        self.grad_clip = _clip if _clip > 0.0 else None

        _lr_env = os.environ.get("DWQ_CENET_LR")
        if _lr_env is not None:
            learning_rate = float(_lr_env)
        _min_lr_env = os.environ.get("DWQ_CENET_MIN_LR")
        if _min_lr_env is not None:
            min_lr = float(_min_lr_env)
        elif _ref:
            min_lr = learning_rate

        num_learning_epochs = int(os.environ.get("DWQ_CENET_EPOCHS", 1 if _ref else num_learning_epochs))
        num_mini_batches = int(os.environ.get("DWQ_CENET_MINI_BATCHES", 1 if _ref else num_mini_batches))

        self.decoder_vel_mode = os.environ.get("DWQ_CENET_DEC_VEL", "keep").lower()
        if self.decoder_vel_mode not in ("keep", "zero", "detach"):
            raise ValueError(f"DWQ_CENET_DEC_VEL must be keep|zero|detach, got {self.decoder_vel_mode!r}")
        # A latent dim counts as "alive" above this per-dim KL (nats). Only a diagnostic.
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
        if self.decoder_vel_mode != "keep":
            # num_vel=3 matches forward()'s `h.split([3, ...])`.
            decoder_layers.insert(0, _DecoderVelGate(self.decoder_vel_mode, num_vel=3))
        self.decoder = nn.Sequential(*decoder_layers)

        print(f"{'CENet Structure':=^60}")
        print(f"Encoder MLP: {self.encoder}")
        print(f"Decoder MLP: {self.decoder}")
        _non_default = (
            self.reference_mode or self.free_bits > 0.0 or self.kl_reduction != "mean"
            or self.beta_anneal or self.grad_clip != 1.0 or self.decoder_vel_mode != "keep"
            or _lr_env is not None or _min_lr_env is not None or _beta_env is not None
        )
        if _non_default:
            print(
                f"[DIAG] CENet gates: reference={self.reference_mode} free_bits={self.free_bits} "
                f"kl_reduction={self.kl_reduction} beta={beta} beta_anneal={self.beta_anneal} "
                f"beta_limit={self.beta_limit} grad_clip={self.grad_clip} lr={learning_rate} "
                f"min_lr={min_lr} epochs={num_learning_epochs}x{num_mini_batches} "
                f"dec_vel={self.decoder_vel_mode}"
            )

        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        # Paper eq.7 puts a weight `beta` on the KL term but gives NEITHER a numeric value NOR a
        # schedule for it. "beta is constant" is OUR READING of that silence, not a sentence from
        # the paper (it never says "constant"), so the constant default here is a choice -- and
        # lowering it with DWQ_CENET_BETA stays inside what the paper actually states. The
        # IsaacGym original reads the same silence the other way and DOES anneal (`beta *= 1.01`
        # capped at `beta_limit`); that path is restorable via DWQ_CENET_BETA_ANNEAL /
        # DWQ_CENET_REFERENCE and is to be settled by experiment.
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

    @staticmethod
    def _kl_per_dim(mu, logvar):
        """KL(q(z|x) || N(0, I)) per latent dim, averaged over the minibatch -> shape [latent].

        Written with ``torch.distributions`` on purpose: the closed form lives in ``update()``
        and must appear exactly ONCE in this file, because that occurrence is the body carved
        out for the ``cenet-loss`` exercise (a second copy would be a printed answer key).
        Numerically the same quantity.
        """
        q = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
        return torch.distributions.kl_divergence(q, p).mean(0)

    def _kl_term_gated(self, mu, logvar):
        """KL term used ONLY when a probe gate is on (free bits and/or sum reduction).

        Free bits (Kingma et al. 2016, "Improved Variational Inference with Inverse
        Autoregressive Flow", §free-bits): clamp the *minibatch-mean* KL of each latent
        dimension at ``self.free_bits`` nats BEFORE reducing over dims. Under the floor the
        gradient w.r.t. that dim's mu/logvar is exactly zero, so a dim the decoder is not
        using yet stops being pushed toward the prior and can be picked up again later. This
        targets the observed failure (decoder |W_z| = 0.011 at it500 and never recovering),
        which is a *dead-latent* spiral rather than an excess-KL-pressure problem.

        ``kl_reduction='sum'`` restores the IsaacGym original's reduction (16x stronger than
        our per-dim mean); the paper fixes no reduction, so it is an alignment knob with that
        implementation, not a collapse fix.
        """
        kl_per_dim = torch.clamp(self._kl_per_dim(mu, logvar), min=self.free_bits)
        kl = kl_per_dim.sum() if self.kl_reduction == "sum" else kl_per_dim.mean()
        return kl.reshape(1) * self.beta

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

            mse_loss = nn.MSELoss()
            vel_loss = mse_loss(est_vel_batch, true_vel_batch)
            recon_loss = mse_loss(est_onext_batch, true_onext_batch)

            # KL is averaged over the 16 latent dims, not summed, so that all three terms use
            # the SAME reduction axis: "mean per dimension" (vel over 3, recon over 45, KL
            # over 16). Mixing them is what killed the latent: nn.MSELoss() averages the recon
            # over 45 dims while a summed KL does not, which is the textbook sum/sum beta-VAE
            # rescaled by 1/45 -- i.e. effective beta = 45*beta (180 at the old beta_limit=4)
            # -> posterior collapse. ("Effective beta" is OUR term, not the paper's: the
            # coefficient that would sit on KL_sum if recon were rescaled to a 45-dim sum.)
            # Aligning the other way instead -- recon as a 45-dim SUM with vel left as a 3-dim
            # mean -- would blow the vel:recon ratio from 1:1 to 1:45 (1:15 = 3:45 would need
            # vel summed as well), so the mean is the choice that keeps both ratios intact.
            # Effective beta vs. textbook sum/sum: (45/16)*beta ~= 2.81*beta.
            klds = -0.5 * (1 + logvar_batch - mu_batch.pow(2) - logvar_batch.exp())
            kl_loss = klds.mean(1).mean(0, True) * self.beta

            total_loss = vel_loss + recon_loss + kl_loss

            # Probe gates (OFF by default -> this block never runs and the numbers above are
            # exactly what they always were). Deliberately an override AFTER the default term
            # instead of a branch inside it: those lines are the verbatim body of the
            # `cenet-loss` exercise, and editing them would push the gate into the starter.
            if self.free_bits > 0.0 or self.kl_reduction != "mean":
                kl_loss = self._kl_term_gated(mu_batch, logvar_batch)
                total_loss = vel_loss + recon_loss + kl_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            # Default 1.0 (same budget PPO uses); None only via DWQ_CENET_GRAD_CLIP=0 or the
            # DWQ_CENET_REFERENCE preset, where there is no clipping at all.
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
                kl_per_dim = self._kl_per_dim(mu_batch, logvar_batch)
                mean_mu_abs += mu_batch.abs().mean().item()
                mean_sigma += (0.5 * logvar_batch).exp().mean().item()
                mean_kl_active += float((kl_per_dim > self.kl_active_eps).sum().item())
                mean_kl_nats += float(kl_per_dim.sum().item())

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

        # beta annealing (IsaacGym-original behaviour, off by default): applied AFTER the losses of this
        # update, exactly where the IsaacGym original puts it, so `beta` reaches beta_limit=4.0 after
        # ~139 updates. The runner checkpoints `cenet_beta`, so a resumed run keeps its place.
        if self.beta_anneal:
            self.beta = min(self.beta * self.beta_anneal_rate, self.beta_limit)

        return mean_total_loss, mean_vel_loss, mean_recon_loss, mean_kl_loss
