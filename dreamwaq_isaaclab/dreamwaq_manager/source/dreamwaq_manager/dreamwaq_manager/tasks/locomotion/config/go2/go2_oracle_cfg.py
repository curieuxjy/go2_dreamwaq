"""Go2 Oracle environment configuration for DreamWaQ.

Oracle model: actor observes true linear velocity (48 dims),
critic observes privileged information (height scan + full state = 238 dims).
This corresponds to Go2RoughOracleCfg in the original DreamWaQ codebase.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from dreamwaq_manager.tasks.locomotion.config.go2.go2_base_cfg import Go2BaseEnvCfg
from dreamwaq_manager.tasks.locomotion.velocity_env_cfg import ObservationsCfg
import dreamwaq_manager.tasks.locomotion.mdp as mdp


@configclass
class Go2OracleObservationsCfg:
    """Oracle observation config: actor gets true lin_vel, critic gets privileged obs."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations for Oracle: 48 dims (includes true lin_vel).

        Noise levels match the original effective values
        (legged_robot.py:_get_noise_scale_vec, noise = noise_scales × obs_scales):
            lin_vel: 0.1 × 2.0  = 0.20
            ang_vel: 0.2 × 0.25 = 0.05
            gravity: 0.05
            dof_pos: 0.01 × 1.0 = 0.01
            dof_vel: 1.5 × 0.05 = 0.075
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.075, n_max=0.075))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Use CriticCfg from base (privileged observations with height scan)
    policy: PolicyCfg = PolicyCfg()
    critic: ObservationsCfg.CriticCfg = ObservationsCfg.CriticCfg()


@configclass
class Go2OracleEnvCfg(Go2BaseEnvCfg):
    """Go2 Oracle — true velocity + privileged height observations."""

    observations: Go2OracleObservationsCfg = Go2OracleObservationsCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class Go2OracleEnvCfg_PLAY(Go2OracleEnvCfg):
    """Go2 Oracle environment for evaluation/play."""

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
        # disable observation noise for play
        self.observations.policy.enable_corruption = False
        # disable randomization for play
        self.events.base_external_force_torque = None
        self.events.push_robot = None
