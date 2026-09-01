"""RSL-RL PPO runner configurations for Go2 DreamWaQ DirectRLEnv."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for Go2 Base model."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "DreamWaQ-Direct-Go2-Base-v0"
    logger = "wandb"
    wandb_project = "lec_dreamwaq"
    # Wrapper-level raw-action clip. `_pre_physics_step` clamps inline to the same bound
    # (`cfg.clip_actions` in dreamwaq_env_cfg.py) — keep the two in sync, and with Manager.
    #
    # Was 1.0, raised to 4.0 (PAPER.md §6). The ±1 clip was the cause of the σ ceiling: log-prob
    # and entropy use the UNCLIPPED Gaussian, so beyond σ≈1 extra noise is free while the entropy
    # bonus keeps paying, and no equilibrium below the log_std clamp exists. The old "±10 actions →
    # joint limits → collapse" worry came from the σ≤7.4 era (log_std ceiling 2.0); with the
    # ceiling now 0.0 (σ≤1), a 4000-it clip-4.0 run showed `dof_pos_limits` violations 0.00000.
    clip_actions = 4.0

    def __post_init__(self):
        super().__post_init__()
        # TEST (DWQ_CLIP100=1): original legged_gym clips at ±100 (effectively unclipped).
        # Note this only moves the wrapper clip; the env's inline clamp (cfg.clip_actions) has its
        # own DWQ_CLIP100 gate in dreamwaq_env_cfg.__post_init__ so the two stay consistent.
        import os as _os_ca
        if _os_ca.environ.get("DWQ_CLIP100"):
            self.clip_actions = 100.0

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="log",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        # True for EVERY arm. This used to default to False and only Oracle/Waq flipped it on in
        # __post_init__, which made the value function differ per arm — a confound in a comparison
        # whose only intended difference is what the ACTOR sees. Mirrors Manager (measured there:
        # rough critic is 235-dim, 187 of them height_scan with std ~0.11 vs proprio up to 3.37,
        # so without normalization the terrain signal reaches the value MLP ~30x weaker).
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 0.01 x 12 action dims kept the policy pinned at the log_std clamp ceiling (std = 1)
        # for the whole run — Loss/entropy was exactly 17.0273 (= 12 x 0.5 x ln 2*pi*e) from
        # iteration 300 to 3000 in all six runs. Max-amplitude exploration noise dominates the
        # action, which buries any gain from the 3-dim velocity estimate in measurement noise.
        # Lowering the coefficient is the conservative fix; the clamp itself stays (NaN guard).
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2OraclePPORunnerCfg(Go2BasePPORunnerCfg):
    """PPO runner config for Go2 Oracle model."""

    experiment_name = "DreamWaQ-Direct-Go2-Oracle-v0"
    # (critic.obs_normalization is True in the base cfg now — the override that used to live here
    # was the reason only some arms had it.)


@configclass
class Go2WaqPPORunnerCfg(Go2BasePPORunnerCfg):
    """PPO runner config for Go2 DreamWaQ (Waq) model.

    Uses custom OnPolicyRunnerWaq that integrates CENet training.
    Actor input: base_obs(45) + est_vel(3) + context(16) = 64
    """

    class_name = "OnPolicyRunnerWaq"
    max_iterations = 5000
    save_interval = 500
    experiment_name = "DreamWaQ-Direct-Go2-Waq-v0"

    def __post_init__(self):
        super().__post_init__()
        # Make explicit what rsl_rl was inferring (it printed the deprecated-fallback warning).
        # No-op: `resolve_obs_groups` finds no group named "actor" so it falls back to ["policy"],
        # and the Waq Direct env sets state_space > 0 so a "critic" group exists and is picked.
        # The Waq runner builds the 64-dim actor input itself and writes it into obs["policy"].
        # Mirrors Manager.
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    waq = {
        "len_obs_history": 5,
        "num_base_obs": 45,
        "ada_boot": True,
    }

    # beta is CONSTANT (paper eq.7); the old beta_limit / `beta *= 1.01` annealing is gone.
    # beta 0.35 (paper gives no value): with domain randomization on, beta 1.0 collapses the
    # 16-dim context z (|mu| < 1e-3 by iter ~300). Prescription cells (PAPER.md §4, 2026-08-20):
    # beta 0.35 keeps z alive (|mu| 0.215, KL 4.2 nats, 5/16 active dims) and the 4000-iter
    # horizon-matched run (beta035_hz4000) stayed alive with no decay through iter 2100.
    # Measured in Manager; mirrored here per the Manager<->Direct alignment rule.
    # learning_rate 0.01 -> 1e-3: the paper's Adam lr, and the same lr PPO uses here. At 0.01
    # the CENet's first-layer weight norm ran away (7.6 -> 62.6) with no gradient clip.
    vae = {
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "beta": 0.35,
        "learning_rate": 1.0e-3,
        # min_lr == learning_rate: the ReduceLROnPlateau schedule is a NO-OP by design.
        # The paper gives Adam lr 1e-3 and no schedule at all. With a 1e-4 floor the
        # estimator hit bottom by iter ~2000 and then could not track the still-moving
        # policy distribution: measured on the 2026-08-30 sweep, flat `Loss/cenet_vel`
        # got WORSE over training, 0.0311 (it 500) -> 0.0455 (it 3900), a 46% regression.
        # The 5x4 update budget makes it worse -- 20 grad steps per rollout means the loss
        # the scheduler reads last is the most over-fitted one, so the decay re-triggers.
        # (The author's own IsaacGym code floors at 1.5e-3, above our starting lr.)
        "min_lr": 1.0e-3,
        "patience": 100,
        "factor": 0.8,
    }
