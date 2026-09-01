"""Go2 Base environment configuration for DreamWaQ.

Base model: actor observes 45 dims (no linear velocity), no privileged observations.
This corresponds to Go2RoughBaseCfg in the original DreamWaQ codebase.
"""

from isaaclab.utils import configclass

from dreamwaq_manager.tasks.locomotion.velocity_env_cfg import DreamWaQVelocityRoughEnvCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class Go2BaseEnvCfg(DreamWaQVelocityRoughEnvCfg):
    """Go2 Base environment — blind locomotion without velocity estimation."""

    def __post_init__(self):
        super().__post_init__()

        # -- robot
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Override actuator gains to the DreamWaQ paper values (§III-C: Kp=28, Kd=0.7 @200Hz).
        # The legged_gym reference config uses 20.0/0.5 and keeps 27.0/0.7 commented out.
        self.scene.robot.actuators["base_legs"].stiffness = 28.0
        self.scene.robot.actuators["base_legs"].damping = 0.7
        # Spawn height 0.42 m — clears boxes sub-terrain (up to 0.1 m) plus xy ±0.5 m randomization.
        # Higher than paper's 0.34 m and IsaacLab default 0.4 m; robot settles onto terrain via physics.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.42)

        # -- height scanner prim path
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # -- scale down terrains for Go2 (smaller robot)
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # -- actions: DreamWaQ action scale 0.25 (already set in base)
        self.actions.joint_pos.scale = 0.25

        # -- rewards: foot contact body names for Go2
        # Go2 foot bodies are named ".*_foot" (not ".*FOOT")
        self.rewards.base_height.params["target_height"] = 0.30

        # -- terminations: trunk-contact (TerminationsCfg.base_contact, body_names="base" — matches Go2).


@configclass
class Go2BaseEnvCfg_PLAY(Go2BaseEnvCfg):
    """Go2 Base environment for evaluation/play."""

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
