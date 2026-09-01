"""RSL-RL PPO runner configurations for Go2 DreamWaQ environments."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for Go2 Base model."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "DreamWaQ-Manager-Go2-Base-v0"
    logger = "wandb"
    wandb_project = "IsaacLab_DreamWaq"
    # Clamp raw policy actions at the wrapper level (Direct enforces the same bound inline in
    # _pre_physics_step; this keeps Manager at parity).
    #
    # History — why 1.0, and why 4.0 now (PAPER.md §6):
    #   * 1.0 was chosen alongside the log_std clamp when a σ≤7.4 policy (old clamp ceiling 2.0)
    #     emitted ±10 actions → joint targets far outside limits → extreme PD torques → collapse.
    #     That failure mode belongs to the σ≤7.4 era; the clamp ceiling has since been 0.0 (σ≤1).
    #   * The ±1 clip turned out to be what pinned σ at the clamp ceiling: log-prob and entropy are
    #     computed on the UNCLIPPED Gaussian, so past σ≈1 the executed action saturates and extra
    #     noise costs nothing while the entropy bonus keeps paying. g_surr saturates at ~2.2e-3 <
    #     entropy_coef 0.005 → no equilibrium below the ceiling exists. At 4.0 the equilibrium
    #     comes back (σ*=0.374, g_tot≈4.7e-5) and rough Base underlying goes 0.549 → 0.787.
    #   * Safe because σ is separately capped at 1 by the log_std clamp, and the 4000-it run at
    #     clip 4.0 measured `dof_pos_limits` violations of 0.00000 with no collapse.
    # Reference values: original legged_gym 100.0, IsaacLab official Go2 None (unclipped).
    clip_actions = 4.0

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
        # True for EVERY arm. This used to default to False and only Waq (and the abandoned-recipe
        # Oracle) flipped it on in __post_init__, so of the six official-env runs only the two Waq
        # ones had a `critic_normalizer` in their checkpoints. That is a per-arm difference in the
        # value function, i.e. exactly the kind of confound the Base/Oracle/Waq comparison must not
        # have. It also matters most where it was missing: the rough critic is 235-dim, of which
        # 187 are height_scan with measured std ~0.11 while proprio terms reach 3.37 — unnormalized,
        # the terrain signal enters the value MLP up to ~30x weaker than proprioception and with a
        # non-zero offset. The actor has had obs_normalization=True all along.
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

    experiment_name = "DreamWaQ-Manager-Go2-Oracle-v0"
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
    experiment_name = "DreamWaQ-Manager-Go2-Waq-v0"

    def __post_init__(self):
        super().__post_init__()
        # Make explicit what rsl_rl was inferring. With obs_groups left MISSING (-> {}) the
        # library filled it by the deprecated fallback and printed
        # "This behavior will be removed in a future version". Verified to be a no-op here:
        # `resolve_obs_groups` sees no group literally named "actor", so it falls back to
        # ["policy"], and it does find a group named "critic" (every Waq env declares one), so it
        # picks ["critic"] — identical to the line below. The Waq runner assembles the 64-dim
        # actor input itself and writes it back into obs["policy"] before alg.act(), so "policy"
        # is the right actor group here even though the env's own policy group is 45-dim.
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    # DreamWaQ-specific config (passed to OnPolicyRunnerWaq)
    waq = {
        "len_obs_history": 5,
        "num_base_obs": 45,
        "ada_boot": True,
    }

    # CENet hyperparameters (passed to CENet.__init__)
    # num_mini_batches/num_learning_epochs match PPO so the CENet update
    # doesn't degenerate into a single huge minibatch when num_envs is large.
    # beta is CONSTANT (paper eq.7); the old beta_limit / `beta *= 1.01` annealing is gone.
    # beta 0.35 (paper gives no value): with domain randomization on, beta 1.0 collapses the
    # 16-dim context z (|mu| < 1e-3 by iter ~300). Prescription cells (PAPER.md §4, 2026-08-20):
    # beta 0.35 keeps z alive (|mu| 0.215, KL 4.2 nats, 5/16 active dims) and the 4000-iter
    # horizon-matched run (beta035_hz4000) stayed alive with no decay through iter 2100.
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


@configclass
class Go2WaqOfficialRoughPPORunnerCfg(Go2WaqPPORunnerCfg):
    """PPO + CENet on the OFFICIAL Go2 rough env (unified-rewards fallback experiment)."""

    experiment_name = "Waq-Official-Rough-PPO-v0"
    max_iterations = 5000
    save_interval = 500


@configclass
class Go2WaqOfficialFlatPPORunnerCfg(Go2WaqPPORunnerCfg):
    """PPO + CENet on the OFFICIAL Go2 flat env (unified-rewards fallback experiment)."""

    experiment_name = "Waq-Official-Flat-PPO-v0"
    max_iterations = 5000
    save_interval = 500


# --- Confound separation (report §4.1): DreamWaQ PPO hparams + Oracle obs, NO CENet ---
# PPO-Waq differs from PPO-Oracle in (a) network/std hparams AND (b) CENet+obs. These cells
# fix (a) to the DreamWaQ-Waq settings (Go2BasePPORunnerCfg: [512,256,128], std_type=log) and
# use Oracle observation (no CENet), so (Waq − OracleDwq) = pure CENet effect, and
# (OracleDwq − official Oracle) = the hparam/network effect.
@configclass
class Go2OracleDwqHparamsFlatPPORunnerCfg(Go2BasePPORunnerCfg):
    """DreamWaQ PPO hparams ([512,256,128], std_type=log) + official flat env (Oracle obs, no CENet)."""

    max_iterations = 5000
    experiment_name = "OracleDwq-Official-Flat-PPO-v0"

    def __post_init__(self):
        super().__post_init__()
        # critic -> ["critic"], NOT ["policy"]. `OracleOfficialFlatEnvCfg` declares its own clean
        # critic group (same 48 terms, enable_corruption=False), but that group was dead: this
        # routing sent the value function back to the corrupted actor group, so of the six runs
        # only this one logged `critic : ['policy']`. It was therefore the only arm with (a) obs
        # noise in the value input and (b) no asymmetric actor-critic at all.
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}


@configclass
class Go2OracleDwqHparamsRoughPPORunnerCfg(Go2BasePPORunnerCfg):
    """DreamWaQ PPO hparams + OracleOfficialRoughEnvCfg (actor 48 incl lin_vel, critic 235, no CENet)."""

    max_iterations = 5000
    save_interval = 500
    experiment_name = "OracleDwq-Official-Rough-PPO-v0"

    def __post_init__(self):
        super().__post_init__()
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}


# DreamWaQ-regime BLIND lower bound (deployable, proprioception-only, NO CENet). Same DreamWaQ
# hparams ([512,256,128], std_type=log) and same privileged critic as Waq/OracleDwq, but the
# actor is the 45-dim blind obs without any velocity/context estimation. This is the correct
# baseline to isolate CENet's value: (Waq − BaseDwq) = pure CENet contribution, while OracleDwq
# is the privileged (sim-only) upper bound. Mirrors DreamWaQ's ablation (oracle ≥ DreamWaQ ≥ blind).
@configclass
class Go2BaseDwqHparamsFlatPPORunnerCfg(Go2BasePPORunnerCfg):
    """DreamWaQ PPO hparams + WaqOfficialFlatEnvCfg (blind actor 45, critic 48, no CENet)."""

    max_iterations = 5000
    experiment_name = "BaseDwq-Official-Flat-PPO-v0"

    def __post_init__(self):
        super().__post_init__()
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}


@configclass
class Go2BaseDwqHparamsRoughPPORunnerCfg(Go2BasePPORunnerCfg):
    """DreamWaQ PPO hparams + WaqOfficialRoughEnvCfg (blind actor 45, critic 235, no CENet)."""

    max_iterations = 5000
    save_interval = 500
    experiment_name = "BaseDwq-Official-Rough-PPO-v0"

    def __post_init__(self):
        super().__post_init__()
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}


# --- THE PAPER'S ORACLE: terrain-aware actor (height map), no CENet ---
# DreamWaQ's headline claim ("almost as well as the oracle policy that has direct access to the
# surrounding terrain's height map", §III-B / Fig.3 caption) is about THIS arm. Our OracleDwq is
# a *velocity* oracle, about which the paper says nothing. Same DreamWaQ hparams as every other
# arm so the only difference is what the actor sees: 45 proprio + 187 height_scan.
@configclass
class Go2TerrainOracleRoughPPORunnerCfg(Go2BasePPORunnerCfg):
    """DreamWaQ PPO hparams + TerrainOracleOfficialRoughEnvCfg (actor 232 = 45 + 187, critic 235)."""

    max_iterations = 5000
    save_interval = 500
    experiment_name = "TerrainOracle-Official-Rough-PPO-v0"

    def __post_init__(self):
        super().__post_init__()
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
