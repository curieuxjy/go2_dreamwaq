"""Go2 DreamWaQ DirectRLEnv configurations.

Three variants matching the original DreamWaQ codebase:
- Base: blind locomotion, 45-dim obs (no lin_vel), no privileged obs
- Oracle: 48-dim obs (includes lin_vel), 238-dim privileged obs
- Waq: 45-dim obs, 238-dim privileged obs, CENet integration via runner
"""

from __future__ import annotations

from isaaclab.utils import configclass

from dreamwaq_direct.tasks.locomotion.dreamwaq_env_cfg import DreamWaQDirectEnvCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class Go2BaseDirectCfg(DreamWaQDirectEnvCfg):
    """Go2 Base — blind locomotion without velocity estimation.

    Actor obs: 45 dims (no lin_vel)
    No privileged observations (state_space = 0).
    """

    # No privileged obs for base
    state_space = 0

    def __post_init__(self):
        super().__post_init__()

        # Robot
        self.robot = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        self.robot.actuators["base_legs"].stiffness = 28.0
        self.robot.actuators["base_legs"].damping = 0.7
        # Spawn height 0.42 m — clears boxes sub-terrain (up to 0.1 m) plus xy ±0.5 m randomization.
        self.robot.init_state.pos = (0.0, 0.0, 0.42)

        # Height scanner prim path
        self.height_scanner.prim_path = "/World/envs/env_.*/Robot/base"

        # Scale down terrain obstacles for Go2 (smaller robot)
        self.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # Reward param
        self.base_height_target = 0.30


@configclass
class Go2BaseDirectCfg_PLAY(Go2BaseDirectCfg):
    """Go2 Base environment for evaluation/play."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.terrain.max_init_terrain_level = None
        if self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.num_rows = 5
            self.terrain.terrain_generator.num_cols = 5
            self.terrain.terrain_generator.curriculum = False

        # Disable randomization for play (was: events.base_external_force_torque/push_robot = None)
        self.push_robots = False
        self.randomize_friction = False
        self.randomize_base_mass = False
        self.randomize_com = False
        self.randomize_pd_gains = False
        # Disable system delay and obs noise for play
        self.system_delay = False
        self.obs_noise = False


@configclass
class Go2OracleDirectCfg(Go2BaseDirectCfg):
    """Go2 Oracle — true velocity + privileged height observations.

    Actor obs: 48 dims (lin_vel(3) + ang_vel(3) + gravity(3) + cmd(3) + dof_pos(12) + dof_vel(12) + actions(12))
    Critic obs: 238 dims (clean actor obs(48) + disturb(3) + heights(187))
    Matches original IsaacGym OnPolicyRunner critic input = obs_buf ⊕ privileged_obs_buf.
    """

    observation_space = 48
    state_space = 238
    include_lin_vel_in_obs = True

    def __post_init__(self):
        super().__post_init__()


@configclass
class Go2OracleDirectCfg_PLAY(Go2OracleDirectCfg):
    """Go2 Oracle environment for evaluation/play."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.terrain.max_init_terrain_level = None
        if self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.num_rows = 5
            self.terrain.terrain_generator.num_cols = 5
            self.terrain.terrain_generator.curriculum = False

        # Disable randomization for play (was: events.base_external_force_torque/push_robot = None)
        self.push_robots = False
        self.randomize_friction = False
        self.randomize_base_mass = False
        self.randomize_com = False
        self.randomize_pd_gains = False
        self.system_delay = False
        self.obs_noise = False


@configclass
class Go2WaqDirectCfg(Go2BaseDirectCfg):
    """Go2 DreamWaQ — CENet-based velocity estimation with privileged critic.

    Actor obs: 45 dims (no lin_vel) — augmented to 64 by runner (+ est_vel + context)
    Critic obs: 235 dims (clean actor obs(45) + disturb(3) + heights(187))
    Matches original IsaacGym OnPolicyRunnerWAQ critic input = obs_buf ⊕ privileged_obs_buf.
    CENet: obs_history(5*45=225) → est_vel(3) + context(16)
    """

    observation_space = 45
    state_space = 235
    include_lin_vel_in_obs = False

    # Original go2_config.py:Go2RoughWaqCfg.noise overrides add_noise = False
    # (Base/Oracle inherit add_noise = True from LeggedRobotCfg.noise).
    obs_noise = False

    # CENet-specific
    len_obs_history = 5
    num_context = 16
    num_est_vel = 3

    # System delay disabled by default to match Manager version
    # (original DreamWaQ paper uses system_delay=True)

    def __post_init__(self):
        super().__post_init__()


@configclass
class Go2WaqDirectCfg_PLAY(Go2WaqDirectCfg):
    """Go2 DreamWaQ environment for evaluation/play."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.terrain.max_init_terrain_level = None
        if self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.num_rows = 5
            self.terrain.terrain_generator.num_cols = 5
            self.terrain.terrain_generator.curriculum = False

        # Disable randomization for play (was: events.base_external_force_torque/push_robot = None)
        self.push_robots = False
        self.randomize_friction = False
        self.randomize_base_mass = False
        self.randomize_com = False
        self.randomize_pd_gains = False
        self.system_delay = False
        self.obs_noise = False
