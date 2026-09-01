"""DreamWaQ custom runner for IsaacLab RSL-RL integration.

This runner extends the standard OnPolicyRunner to integrate CENet training
alongside PPO. The CENet processes observation history to estimate velocity
and produce a context vector, which augments the actor's observations.

Key differences from standard OnPolicyRunner:
1. Maintains an observation history buffer (5 timesteps)
2. Runs CENet forward pass before each action to get est_vel + context
3. Augments actor obs with CENet outputs: base_obs(45) + est_vel(3) + context(16) = 64
4. Critic uses privileged observations directly from the environment
5. Trains CENet after each rollout alongside PPO updates

Usage: Set class_name="OnPolicyRunnerWaq" in the runner config and provide
CENet-specific parameters via the vae config section.
"""

from __future__ import annotations

import os
import time

import torch

from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan

from dreamwaq_manager.algorithms.cenet import CENet


class OnPolicyRunnerWaq(OnPolicyRunner):
    """On-policy runner with integrated CENet for DreamWaQ velocity estimation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # Extract DreamWaQ-specific config before parent init
        self.waq_cfg = train_cfg.get("waq", {})
        self.vae_cfg = train_cfg.get("vae", {})
        self.len_obs_history = self.waq_cfg.get("len_obs_history", 5)
        self.num_base_obs = self.waq_cfg.get("num_base_obs", 45)
        self.num_est_vel = 3
        self.num_context = 16
        self.ada_boot = self.waq_cfg.get("ada_boot", True)
        # DreamWaQ++ exteroception: if True, the policy obs is [proprio(num_base_obs), height_scan],
        # and the actor obs becomes [norm(proprio), est_vel, context, height_scan]. CENet still uses
        # only the proprioceptive base. num_extero is derived from the env's policy obs dim.
        self.use_exteroception = self.waq_cfg.get("use_exteroception", False)
        _policy_dim = env.get_observations()["policy"].shape[-1]
        self.num_extero = (_policy_dim - self.num_base_obs) if self.use_exteroception else 0

        # Augmented actor obs dim: base_obs + est_vel + context (+ exteroception)
        self.num_augmented_obs = self.num_base_obs + self.num_est_vel + self.num_context + self.num_extero

        # Temporarily patch env.get_observations so the parent __init__
        # builds the actor with the augmented dimension (64) instead of base (45).
        original_get_obs = env.get_observations

        def _get_augmented_obs():
            obs = original_get_obs()
            # Pad policy obs from 45 → 64 with zeros (est_vel + context placeholder)
            policy_obs = obs["policy"]
            padding = torch.zeros(
                policy_obs.shape[0], self.num_est_vel + self.num_context,
                device=policy_obs.device,
            )
            obs["policy"] = torch.cat([policy_obs, padding], dim=-1)
            return obs

        env.get_observations = _get_augmented_obs

        # Initialize parent (creates PPO algorithm with actor input=64, critic from critic group)
        super().__init__(env, train_cfg, log_dir, device)

        # Replace the actor's built-in 64-dim normalizer with Identity, and instead normalize
        # ONLY the base obs (45) with our own running-mean-std (``self.obs_rms``), leaving
        # est_vel(3) and context(16) raw. This matches the original DreamWaQ, which normalizes
        # the base obs (and the CENet history input) via ``obs_rms`` but feeds est_vel/context
        # raw. (The auto-built 64-dim normalizer was trained on zero-padded obs → near-zero
        # variance on the CENet dims → NaN; and leaving everything unnormalized handicapped the
        # actor vs. the normalized blind baseline — see report §6.)
        self.alg.actor.obs_normalization = False
        self.alg.actor.obs_normalizer = torch.nn.Identity()
        self.obs_rms = EmpiricalNormalization(shape=self.num_base_obs).to(device)
        # Normalize the CENet velocity target with its own running stats (matches original
        # DreamWaQ's true_vel_rms). est_vel is then predicted in normalized-velocity space.
        self.true_vel_rms = EmpiricalNormalization(shape=self.num_est_vel).to(device)
        self._nan_warned = False
        print("[INFO] Waq: actor normalizer=Identity; base-obs(45) normalized via obs_rms (est_vel/context raw)")

        # Register optimizer hook to clamp log_std_param after each step.
        # Without this, log_std can drift to extreme values during long training,
        # causing exp(log_std) to underflow to 0 or gradients to produce NaN.
        self._register_std_clamp_hook()

        # Restore original get_observations
        env.get_observations = original_get_obs

        # Initialize CENet
        obs_history_dim = self.len_obs_history * self.num_base_obs
        self.cenet = CENet(
            input_dim=obs_history_dim,
            output_dim=self.num_base_obs,
            device=device,
            **self.vae_cfg,
        ).to(device)

        # Initialize CENet storage
        num_steps = train_cfg.get("num_steps_per_env", 24)
        self.cenet.init_storage(
            num_envs=env.num_envs,
            num_transitions_per_env=num_steps,
            obs_history_shape=(obs_history_dim,),
            true_vel_shape=(3,),
            true_onext_shape=(self.num_base_obs,),
        )

        # Observation history buffer: [num_envs, len_obs_history, num_base_obs]
        self.obs_history_buf = torch.zeros(
            env.num_envs, self.len_obs_history, self.num_base_obs, device=device
        )

        # AdaBoot: boot_prob = P(feed the CENet-estimated velocity to the actor); 1 - boot_prob is
        # the chance of feeding the TRUE velocity (bootstrapping). The original DreamWaQ raises
        # boot_prob as the estimator becomes reliable. We ramp it linearly from 0 → boot_max over
        # the first ``boot_ramp_frac`` of training so the actor is bootstrapped with ground-truth
        # velocity early (when CENet is untrained) and weaned onto est_vel as it learns.
        #
        # NOTE (paper divergence, unresolved): the paper uses p_boot = 1 − tanh(CV(R)) (eq.8), a
        # performance-based rule with a negative feedback loop — a broken estimator raises CV and
        # pulls p_boot back down. This is a capped LINEAR RAMP: time-based, no feedback. The cap
        # (0.9) is what keeps a true-velocity fallback alive at all; tanh(CV) > 0 means the paper's
        # p_boot never reaches 1.0 either, and the old min(1.0, …) here did, which left the actor
        # on 100% estimator output from iteration 1500 onwards.
        self.boot_ramp_frac = self.waq_cfg.get("boot_ramp_frac", 0.5)
        self.boot_max = self.waq_cfg.get("boot_max", 0.9)
        self.boot_prob = 0.0

    def _register_std_clamp_hook(self):
        """Register a post-step hook on the optimizer to clamp log_std_param.

        This prevents the log_std from drifting to extreme values during long training,
        which would cause exp(log_std) to underflow/overflow and produce NaN.
        Upper bound 0.0 (std <= 1.0): a larger std saturates almost every action at the
        +-1 clip during sampling, so exploration degrades to max-amplitude noise and the
        policy never learns coordinated locomotion (std used to pin at the old 2.0 ceiling).
        Clamp range: [-5, 0] → std range: [~0.007, ~1.0]

        The parameter is selected by OBJECT IDENTITY, not by shape. The old test was
        ``param.shape == (self.env.num_actions,)``, which also matches the actor MLP's
        OUTPUT-LAYER BIAS (``mlp.6.bias``, shape (12,)) — so the per-joint constant offset of the
        mean action was clamped to [-5, 0] and could never be positive. Measured across the six
        2026-08 runs: all 6 x 12 = 72 output biases were <= 0, several within 1e-5 of the bound.
        """
        log_std_params = [m.log_std_param for m in self.alg.actor.modules() if hasattr(m, "log_std_param")]
        if not log_std_params:
            print("[WARN] Waq: no log_std_param on the actor — std clamp hook NOT registered")
            return

        def _clamp_std_hook(optimizer, args, kwargs):
            for param in log_std_params:
                with torch.no_grad():
                    param.clamp_(-5.0, 0.0)

        self.alg.optimizer.register_step_post_hook(_clamp_std_hook)
        print(f"[INFO] Registered log_std_param clamp hook [-5, 0] on {len(log_std_params)} actor parameter(s)")

    def _update_obs_history(self, obs_base: torch.Tensor, dones: torch.Tensor | None = None):
        """Shift history buffer and append latest observation.

        Args:
            obs_base: Current base observation [num_envs, num_base_obs].
            dones: Done flags to reset history for terminated envs.
        """
        if dones is not None:
            # Reset history for terminated environments
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_ids) > 0:
                self.obs_history_buf[reset_ids] = 0.0

        # Shift left and append new obs
        self.obs_history_buf[:, :-1] = self.obs_history_buf[:, 1:].clone()
        self.obs_history_buf[:, -1] = obs_base

    def _get_flat_obs_history(self) -> torch.Tensor:
        """Return the normalized, flattened observation history [num_envs, len * num_base_obs].

        The history buffer stores RAW base obs; we normalize each frame with ``obs_rms`` at read
        time (matching the original DreamWaQ, which normalizes the CENet history input).
        """
        return self.obs_rms(self.obs_history_buf).reshape(self.env.num_envs, -1)

    def _norm_base(self, base_obs: torch.Tensor) -> torch.Tensor:
        """Normalize the base obs (45) with obs_rms for the actor input (est_vel/context stay raw)."""
        return self.obs_rms(base_obs)

    def _extract_base_obs(self, obs) -> torch.Tensor:
        """Extract the proprioceptive base obs (first ``num_base_obs`` of the policy group)."""
        policy = obs["policy"] if (hasattr(obs, "__getitem__") and "policy" in obs.keys()) else obs
        # With exteroception the policy is [proprio(45), height_scan]; CENet/actor-base use proprio.
        return policy[:, : self.num_base_obs] if self.use_exteroception else policy

    def _extract_extero(self, obs) -> torch.Tensor:
        """Extract the exteroception (height_scan) = everything after the proprioceptive base."""
        policy = obs["policy"] if (hasattr(obs, "__getitem__") and "policy" in obs.keys()) else obs
        return policy[:, self.num_base_obs :]

    def _build_actor_obs(self, base_obs, vel_input, context_vec, src_obs) -> torch.Tensor:
        """Build the actor observation: [norm(base), vel, context] (+ exteroception)."""
        # ── TODO(runner-augment) ─ level L2 ─────────────────────────────
        # actor 관측을 45 → 64 로 증강한다 — 정규화한 base 관측에 추정 속도와 context 를 잇는다
        #   hint: self._norm_base(base_obs) 가 base 관측(45)을 obs_rms 로 정규화해 준다
        #   hint: vel_input(3) 과 context_vec(16) 은 정규화하지 않고 그대로 잇는다 — 45 + 3 + 16 = 64
        #   hint: 잇는 순서가 actor 의 입력 레이아웃이다. 학습과 배포가 같은 순서를 써야 한다
        #   hint: rough 지형은 use_exteroception 이 True 라 height_scan 을 뒤에 덧붙인다 (self._extract_extero(src_obs))
        #   hint: 결과를 actor_obs 에 담는다 — 아래의 NaN 경고와 nan_to_num 이 그 이름을 쓴다 (그 부분은 주어져 있다)
        # 통과 기준은 이 실습 폴더의 README.md 를 본다.
        raise NotImplementedError("TODO(runner-augment)")
        # ─────────────────────────────────────────────────────────────────────
        # nan=0.0 keeps training alive, but silently: a NaN CENet washes the 19 estimate dims to
        # zero and the reward curve looks normal. One archived run trained 3000 iterations as
        # "Base + 19 zeros" with nothing in the curves to show it. Say it once, loudly.
        if not self._nan_warned and not torch.isfinite(actor_obs).all():
            self._nan_warned = True
            print(
                "[WARN] Waq: non-finite actor obs — CENet output is NaN/inf and is being washed "
                "to 0. Training continues on 19 dead dims; the reward curve will NOT show this. "
                "Check Loss/cenet_* and the checkpoint's encoder weights."
            )
        return torch.nan_to_num(actor_obs, nan=0.0, posinf=10.0, neginf=-10.0)

    def _extract_true_vel(self, obs) -> torch.Tensor:
        """Return the CENet's velocity-estimation target: the true base LINEAR velocity.

        Reads the robot's body-frame linear velocity directly from the scene. Reading
        ``obs["critic"][:, :3]`` is WRONG for Waq: ``go2_waq_cfg`` removes ``base_lin_vel``
        from the critic group, so its first 3 dims are the *angular* velocity — the CENet
        was being trained to estimate the wrong quantity.
        """
        lin_vel_b = self.env.unwrapped.scene["robot"].data.root_lin_vel_b
        if not isinstance(lin_vel_b, torch.Tensor):  # warp ProxyArray
            lin_vel_b = lin_vel_b.torch
        return lin_vel_b

    def get_inference_policy(self, device: str | None = None):
        """Return a stateful policy that runs the CENet -> actor pipeline for evaluation.

        The base runner's policy feeds raw observations to the actor, but the DreamWaQ
        actor expects ``base_obs(45) + est_vel(3) + context(16) = 64`` dims produced by
        the CENet from the observation history. This override reproduces the rollout-time
        augmentation (see :meth:`learn`) so a Waq checkpoint can be evaluated/played.

        The returned callable is ``policy(obs, dones=None)``: pass the previous step's
        ``dones`` so the per-env observation history is reset on episode boundaries.
        """
        self.alg.eval_mode()
        self.cenet.test_mode()
        self.obs_rms.eval()  # freeze normalizer stats during evaluation
        self.true_vel_rms.eval()
        actor_policy = self.alg.get_policy()
        if device is not None:
            actor_policy = actor_policy.to(device)

        def policy(obs, dones=None):
            base_obs = self._extract_base_obs(obs)
            self._update_obs_history(base_obs, dones)
            obs_history = self._get_flat_obs_history()  # normalized via obs_rms
            # forward() has no rollout-storage side effects (unlike before_action()).
            _, est_vel, _, _, context_vec = self.cenet.forward(obs_history)
            est_vel = torch.clamp(est_vel, -10.0, 10.0)
            context_vec = torch.clamp(context_vec, -10.0, 10.0)
            augmented = self._build_actor_obs(base_obs, est_vel, context_vec, obs)
            # The actor consumes the observation dict (actor group = "policy"), as in learn().
            augmented_obs = obs.clone()
            augmented_obs["policy"] = augmented
            return actor_policy(augmented_obs)

        return policy

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop with integrated CENet training."""
        # Randomize initial episode lengths
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Get initial observations
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.cenet.train_mode()
        self.obs_rms.train()  # allow obs_rms to learn running mean/std during rollout
        self.true_vel_rms.train()

        # Ensure all parameters are in-synced for multi-GPU
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Initialize obs history with current observation
        base_obs = self._extract_base_obs(obs)
        for _ in range(self.len_obs_history):
            self._update_obs_history(base_obs)

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()

            # AdaBoot schedule: ramp P(use est_vel) from 0 → boot_max over the first
            # boot_ramp_frac of training (relative to this learn() call's iteration span).
            # `max(self.boot_prob, …)` makes the ramp monotonic, so a resumed run keeps the
            # boot_prob restored from the checkpoint instead of falling back to 0.
            ramp_iters = max(1.0, self.boot_ramp_frac * num_learning_iterations)
            ramp = float(min(self.boot_max, (it - start_it) / ramp_iters))
            self.boot_prob = max(self.boot_prob, ramp)

            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Get base observations and true velocity
                    base_obs = self._extract_base_obs(obs)
                    true_vel = self._extract_true_vel(obs)

                    # Update running mean/std on the raw base obs (once per step), then read the
                    # normalized history for CENet (obs_rms applied inside _get_flat_obs_history).
                    self.obs_rms.update(base_obs)
                    obs_history = self._get_flat_obs_history()

                    # Normalize the velocity target with its own running stats (true_vel_rms),
                    # matching original DreamWaQ. est_vel is then learned in normalized-velocity
                    # space; the AdaBoot true-vel branch below reuses this same normalized vector.
                    self.true_vel_rms.update(true_vel)
                    true_vel = self.true_vel_rms(true_vel)

                    # CENet forward pass
                    est_next_obs, est_vel, mu, logvar, context_vec = self.cenet.before_action(
                        obs_history, true_vel
                    )

                    # Clamp CENet outputs to prevent NaN propagation
                    est_vel = torch.clamp(est_vel, -10.0, 10.0)
                    context_vec = torch.clamp(context_vec, -10.0, 10.0)

                    # AdaBoot: probabilistically use estimated or true velocity. The draw is
                    # PER ENV — a single scalar draw per step made all 4096 envs share one
                    # source, so a rollout was either all-estimated or all-true with zero
                    # batch diversity.
                    if self.ada_boot:
                        boot_mask = torch.rand(est_vel.shape[0], 1, device=est_vel.device) < self.boot_prob
                        vel_input = torch.where(boot_mask, est_vel, true_vel)
                    else:
                        vel_input = est_vel

                    # Actor obs: norm(base) + vel + context (+ exteroception); est/ctx raw
                    augmented_actor_obs = self._build_actor_obs(base_obs, vel_input, context_vec, obs)

                    # Build augmented obs dict for RSL-RL
                    # Actor uses augmented obs, critic uses its own privileged obs
                    augmented_obs = obs.clone()
                    augmented_obs["policy"] = augmented_actor_obs

                    # Sample actions using augmented observations
                    actions = self.alg.act(augmented_obs)

                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                    # Check for NaN
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)

                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # Get next base obs for CENet after_action. Store the NORMALIZED next obs as
                    # the recon target so encoder input (normalized history) and decoder target are
                    # on the SAME scale — matching the original DreamWaQ (legacy stored the
                    # already-normalized obs). Previously the raw obs was stored, mis-scaling recon.
                    next_base_obs = self._extract_base_obs(obs)
                    self.cenet.after_action(self._norm_base(next_base_obs))

                    # Update observation history (reset for done envs)
                    self._update_obs_history(next_base_obs, dones)

                    # Build augmented obs for process_env_step (normalizer update)
                    next_true_vel = self._extract_true_vel(obs)
                    next_obs_history = self._get_flat_obs_history()
                    with torch.inference_mode():
                        _, next_est_vel, _, _, next_context_vec = self.cenet.forward(next_obs_history)
                        next_est_vel = torch.clamp(next_est_vel, -10.0, 10.0)
                        next_context_vec = torch.clamp(next_context_vec, -10.0, 10.0)

                    next_policy_obs = self._build_actor_obs(next_base_obs, next_est_vel, next_context_vec, obs)
                    next_augmented_obs = obs.clone()
                    next_augmented_obs["policy"] = next_policy_obs

                    # Process environment step (updates normalizers, records transitions)
                    self.alg.process_env_step(next_augmented_obs, rewards, dones, extras)

                    # Logging
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                    # (boot_prob follows the per-iteration ramp set at the top of the loop; the old
                    # per-step timeout-rate nudge is gone — it was overwritten by the ramp on every
                    # iteration anyway, and Direct never had it.)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns using augmented critic observations
                base_obs_final = self._extract_base_obs(obs)
                true_vel_final = self._extract_true_vel(obs)
                obs_history_final = self._get_flat_obs_history()
                with torch.inference_mode():
                    _, est_vel_final, _, _, context_vec_final = self.cenet.forward(obs_history_final)
                    est_vel_final = torch.clamp(est_vel_final, -10.0, 10.0)
                    context_vec_final = torch.clamp(context_vec_final, -10.0, 10.0)
                final_augmented_obs = obs.clone()
                final_augmented_obs["policy"] = self._build_actor_obs(
                    base_obs_final, est_vel_final, context_vec_final, obs
                )
                self.alg.compute_returns(final_augmented_obs)

            # CENet update
            mean_total_loss, mean_vel_loss, mean_recon_loss, mean_kl_loss = self.cenet.update()

            # PPO update
            loss_dict = self.alg.update()

            # Surface the CENet losses alongside the PPO ones. Without this there is no way to
            # tell whether the CENet is learning at all: a posterior collapse (encoder output
            # becomes input-invariant) is invisible in the reward curve, and shows up only as
            # "Waq is no better than Base". cenet_kl going to ~0 and staying there IS the
            # collapse signature. boot_prob is logged so the AdaBoot ramp is visible too.
            loss_dict["cenet_total"] = mean_total_loss
            loss_dict["cenet_vel"] = mean_vel_loss
            loss_dict["cenet_recon"] = mean_recon_loss
            loss_dict["cenet_kl"] = mean_kl_loss
            loss_dict["cenet_beta"] = self.cenet.beta
            loss_dict["boot_prob"] = self.boot_prob
            # Latent-health scalars + the CENet's own (previously unlogged) lr:
            # cenet_mu_abs / cenet_sigma / cenet_kl_active_dims / cenet_lr.
            loss_dict.update(self.cenet.diagnostics)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.optimizer.param_groups[0]["lr"],
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            # Save model
            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        # `current_learning_iteration` was already advanced to the last `it` inside the loop, so
        # adding num_learning_iterations again double-counted it and named the final checkpoint
        # model_5999.pt for a 3000-iteration run (Base/Oracle, on the stock runner, wrote
        # model_2999.pt). Match the stock OnPolicyRunner so all three stacks agree.
        self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos=None) -> None:
        """Save model checkpoint including CENet state."""
        # Get base save dict from parent
        saved_dict = {
            # rsl_rl >= 5's ``PPO.get_policy()`` returns the ACTOR ONLY, so this key alone left
            # every Waq checkpoint without a critic while Base/Oracle (stock ``OnPolicyRunner``,
            # which saves ``actor_state_dict`` + ``critic_state_dict``) had one. Keep the legacy
            # key for checkpoints/loaders that already expect it, and add the stock key names so a
            # Waq checkpoint carries the same contents as a Base/Oracle one (a resumed Waq run
            # used to restart the value function from scratch).
            "model_state_dict": self.alg.get_policy().state_dict(),
            "actor_state_dict": self.alg.actor.state_dict(),
            "critic_state_dict": self.alg.critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            # CENet state
            "cenet_state_dict": self.cenet.state_dict(),
            "cenet_optimizer_state_dict": self.cenet.optimizer.state_dict(),
            "cenet_beta": self.cenet.beta,
            # AdaBoot ramp position — without it a resumed run restarts the ramp at 0 and
            # re-feeds the actor 100% ground-truth velocity for another boot_ramp_frac.
            "boot_prob": self.boot_prob,
            # base-obs running mean/std (used to normalize actor base obs + CENet history)
            "obs_rms": self.obs_rms.state_dict(),
            "true_vel_rms": self.true_vel_rms.state_dict(),
        }

        # Save actor and critic normalizers if they exist
        if hasattr(self.alg.actor, "obs_normalizer") and self.alg.actor.obs_normalization:
            saved_dict["actor_normalizer"] = self.alg.actor.obs_normalizer.state_dict()
        if hasattr(self.alg.critic, "obs_normalizer") and self.alg.critic.obs_normalization:
            saved_dict["critic_normalizer"] = self.alg.critic.obs_normalizer.state_dict()

        torch.save(saved_dict, path)
        # Upload the checkpoint to wandb/neptune, as the stock ``OnPolicyRunner.save()`` does.
        # This override skipped it, so no Waq model artifact was ever uploaded. No-op for the
        # tensorboard logger and before ``init_logging_writer()``.
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True) -> None:
        """Load model checkpoint including CENet state."""
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)

        # Load actor. Old Waq checkpoints only have "model_state_dict" (actor weights under the
        # legacy key); new ones also have "actor_state_dict". Prefer the explicit key, fall back.
        self.alg.actor.load_state_dict(loaded_dict.get("actor_state_dict", loaded_dict["model_state_dict"]))
        # Critic is absent from every Waq checkpoint written before this fix — skip it then.
        if "critic_state_dict" in loaded_dict:
            self.alg.critic.load_state_dict(loaded_dict["critic_state_dict"])

        # Load CENet if available
        if "cenet_state_dict" in loaded_dict:
            self.cenet.load_state_dict(loaded_dict["cenet_state_dict"])
            if load_optimizer and "cenet_optimizer_state_dict" in loaded_dict:
                self.cenet.optimizer.load_state_dict(loaded_dict["cenet_optimizer_state_dict"])
            if "cenet_beta" in loaded_dict:
                self.cenet.beta = loaded_dict["cenet_beta"]

        # Restore the AdaBoot ramp position (the ramp in learn() is monotonic, so this sticks)
        if "boot_prob" in loaded_dict:
            self.boot_prob = float(loaded_dict["boot_prob"])

        # Load base-obs running mean/std
        if "obs_rms" in loaded_dict:
            self.obs_rms.load_state_dict(loaded_dict["obs_rms"])
        if "true_vel_rms" in loaded_dict:
            self.true_vel_rms.load_state_dict(loaded_dict["true_vel_rms"])

        # Load optimizer
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

        # Load normalizers
        if "actor_normalizer" in loaded_dict and hasattr(self.alg.actor, "obs_normalizer"):
            self.alg.actor.obs_normalizer.load_state_dict(loaded_dict["actor_normalizer"])
        if "critic_normalizer" in loaded_dict and hasattr(self.alg.critic, "obs_normalizer"):
            self.alg.critic.obs_normalizer.load_state_dict(loaded_dict["critic_normalizer"])

        # Load iteration count
        self.current_learning_iteration = loaded_dict.get("iter", 0)
