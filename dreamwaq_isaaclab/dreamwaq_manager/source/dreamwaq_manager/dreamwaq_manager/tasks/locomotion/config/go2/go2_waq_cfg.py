"""Go2 DreamWaQ (Waq) environment configuration.

Waq model: actor observes 45 dims (no lin_vel), critic uses privileged observations (238 dims).
CENet takes obs_history (5x45=225) and estimates velocity + context for the actor.
The CENet augmentation happens in the custom runner (Phase 3), not in the env config.

Actor effective input: base_obs(45) + CENet_context(16) + est_vel(3) = 64 (handled by runner)
Critic input: privileged_obs(238) from critic obs group
"""

from isaaclab.utils import configclass

from dreamwaq_manager.tasks.locomotion.config.go2.go2_base_cfg import Go2BaseEnvCfg
from dreamwaq_manager.tasks.locomotion.velocity_env_cfg import ObservationsCfg


@configclass
class Go2WaqEnvCfg(Go2BaseEnvCfg):
    """Go2 DreamWaQ — CENet-based velocity estimation with privileged critic.

    Critic obs (235 dims) matches the original IsaacGym OnPolicyRunnerWAQ
    critic input = `obs_buf(45) ⊕ privileged_obs_buf(190)`. lin_vel is
    excluded from the critic to mirror the actor obs composition.
    """

    def __post_init__(self):
        super().__post_init__()

        # Add critic observation group for asymmetric training
        self.observations.critic = ObservationsCfg.CriticCfg()
        # Drop lin_vel from the critic — original Waq critic input doesn't
        # include lin_vel (the actor doesn't have it, and the runner's
        # critic input is `obs_buf ⊕ privileged_obs_buf`). Removing the
        # term sets the slot to a zero-dim tensor and the manager skips it.
        self.observations.critic.base_lin_vel = None

        # Original go2_config.py:Go2RoughWaqCfg.noise overrides add_noise = False
        # (Base/Oracle inherit add_noise = True). CENet handles its own normalization.
        self.observations.policy.enable_corruption = False


@configclass
class Go2WaqEnvCfg_PLAY(Go2WaqEnvCfg):
    """Go2 DreamWaQ environment for evaluation/play."""

    def __post_init__(self):
        super().__post_init__()

        # smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # enable debug visualization for play (disabled in training for headless)
        self.commands.base_velocity.debug_vis = True
        # disable randomization for play
        self.events.base_external_force_torque = None
        self.events.push_robot = None
