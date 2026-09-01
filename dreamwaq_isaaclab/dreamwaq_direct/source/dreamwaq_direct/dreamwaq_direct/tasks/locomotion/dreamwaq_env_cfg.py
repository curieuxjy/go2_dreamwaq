"""DreamWaQ DirectRLEnv configuration.

Base configuration for DreamWaQ locomotion using IsaacLab's DirectRLEnv API.
Robot-specific configs (Go2) inherit from this.

Domain randomization is performed inline by DreamWaQEnv (matching the original
IsaacGym `legged_robot.py` flow: `_process_rigid_shape_props`, `_process_rigid_body_props`,
`_randomize_pd_gains`, `_push_robots`, `_reset_root_states`, `_reset_dofs`).
We deliberately do NOT use IsaacLab's EventManager, so `events = None` is inherited
from `DirectRLEnvCfg`.
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class DreamWaQDirectEnvCfg(DirectRLEnvCfg):
    """Base DreamWaQ DirectRLEnv configuration.

    Observation space (Base/Waq): 45 dims (no lin_vel)
        ang_vel(3) + gravity(3) + commands(3) + joint_pos(12) + joint_vel(12) + actions(12)

    State space (privileged, for critic; matches original IsaacGym
    OnPolicyRunnerWAQ critic input = clean obs ⊕ disturb(3) ⊕ heights(187)):
        Waq:    45 + 190 = 235
        Oracle: 48 + 190 = 238
        Base:   0
    """

    # --- env ---
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    clip_actions = 4.0  # raw-action clip before scaling (inline clamp in _pre_physics_step);
    # must match `clip_actions` in agents/rsl_rl_ppo_cfg.py, which clips at the wrapper level.
    # Was 1.0. The suspicion recorded here — that the tight ±1 clip (joint targets ±0.25 rad)
    # cripples the gait — was confirmed on the Manager side (PAPER.md §6): it is also what pinned
    # σ at the log_std clamp ceiling, because log-prob/entropy are computed BEFORE the clip, so
    # past σ≈1 extra exploration noise is free. 4.0 restores an interior σ equilibrium (0.374).
    # The original legged_gym uses 100.0 (effectively unclipped); still reachable via DWQ_CLIP100.
    action_space = 12
    observation_space = 45
    state_space = 238

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,  # decimation
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # --- terrain ---
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=4,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=None,
        debug_vis=False,
    )

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # --- events: disabled. Randomization is inline (see DreamWaQEnv). ---
    events = None

    # --- robot (set by subclass) ---
    robot: ArticulationCfg = MISSING

    # --- sensors ---
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # --- observation scaling (legged_robot.py:416-421, 979) ---
    # The original scales each observation group so all inputs are O(1); without this the raw
    # joint velocities (~±20 rad/s) dominate the observation while the command (±1) is drowned
    # out, and an unnormalized policy never learns to use the command -> it just stands still.
    obs_scale_lin_vel = 2.0   # base linear velocity (Oracle actor + critic)
    obs_scale_ang_vel = 0.25  # base angular velocity
    obs_scale_dof_pos = 1.0   # joint positions
    obs_scale_dof_vel = 0.05  # joint velocities
    obs_scale_height = 5.0    # height scan (privileged)
    # commands scaled by (lin_vel, lin_vel, ang_vel) = (2.0, 2.0, 0.25)

    # --- DreamWaQ-specific ---
    include_lin_vel_in_obs = False  # True for Oracle
    len_obs_history = 5
    num_context = 16
    num_est_vel = 3
    system_delay = True  # paper Table II: [0.0, 15.0] ms
    obs_noise = True  # original Go2 Base/Oracle: True; Waq overrides to False

    # --- command ranges ---
    lin_vel_x_range = (-1.0, 1.0)
    lin_vel_y_range = (-1.0, 1.0)
    ang_vel_z_range = (-1.0, 1.0)
    heading_range = (-math.pi, math.pi)
    heading_command = True
    heading_control_stiffness = 0.5
    command_resampling_time = 10.0

    # --- domain randomization (inline IsaacGym-style; matches go2_config.py:Go2RoughCfg.domain_rand) ---
    # Friction (legged_robot.py:_process_rigid_shape_props): per-env value sampled from
    # `friction_range` via `friction_num_buckets` buckets, applied to ALL shapes of the robot.
    randomize_friction = True
    friction_range = (0.2, 1.25)
    friction_num_buckets = 64
    # Base mass added (legged_robot.py:_process_rigid_body_props): props[0].mass += U(range).
    randomize_base_mass = True
    added_mass_range = (-1.0, 2.0)
    # CoM offset for ALL bodies (legged_robot.py:_process_rigid_body_props): com.{x,y,z} += U(range).
    randomize_com = True
    com_range = (-0.05, 0.05)
    # PD gain scaling: stiffness *= U(p_gains_range), damping *= U(d_gains_range).
    randomize_pd_gains = True
    p_gains_range = (0.9, 1.1)
    d_gains_range = (0.9, 1.1)
    # Motor strength factor (paper Table II). ONE shared factor per env scaling both gains,
    # which for a pure PD actuator is exactly a torque multiplier.
    randomize_motor_strength = True
    motor_strength_range = (0.9, 1.1)
    # Push event (legged_robot.py:_push_robots): every push_interval_s, add U(±max) to root vel.
    push_robots = True
    push_interval_s = 15.0
    max_push_vel_xy = 1.0

    # --- reset randomization (legged_robot.py:_reset_root_states / _reset_dofs) ---
    init_xy_range = (-1.0, 1.0)             # ±1m around terrain origin
    init_vel_range = (-0.5, 0.5)            # all 6 dims (lin+ang) ±0.5
    init_dof_pos_scale_range = (0.5, 1.5)   # default joint pos × U[0.5, 1.5]

    # --- reward scales (DreamWaQ paper) ---
    rew_track_lin_vel_xy = 1.0
    rew_track_ang_vel_z = 0.5
    rew_lin_vel_z = -2.0
    rew_ang_vel_xy = -0.05
    rew_orientation = -0.2
    rew_base_height = -1.0
    rew_dof_acc = -2.5e-7
    rew_action_rate = -0.01
    rew_smoothness = -0.01
    rew_joint_power = -2.0e-5
    # -1e-6, NOT the paper's printed -1e-5 — Table I appears to carry a 10x typo. See the same
    # note on `power_distribution` in the Manager package's velocity_env_cfg.py and PAPER.md §2.
    rew_power_distribution = -1.0e-6
    rew_foot_clearance = -0.01
    rew_dof_pos_limits = 0.0

    # --- reward params ---
    base_height_target = 0.30
    desired_foot_clearance = 0.12
    tracking_sigma = 0.25  # for exp(-error/sigma) — matches the original DreamWaQ
    only_positive_rewards = False  # do NOT clip the per-step total reward at 0.
    # `True` IS the legged_gym default, but DreamWaQ explicitly turns it off: every one of the four
    # reward classes in the authors' a1_config.py sets `only_positive_rewards = False`. Manager has
    # no such clipping, so Manager was already correct and Direct was the odd one out — the old
    # comment here claimed the opposite ("Matches the original DreamWaQ config"). Aligned 2026-08-17.
    # UNVERIFIED by training: Direct has no trained artifacts, and the GPU was busy when this landed.

    # --- termination ---
    # Original DreamWaQ termination (legged_gym `check_termination` + go2_config
    # `terminate_after_contacts_on = ["base"]`): terminate when the contact force on the
    # trunk exceeds 1 N — i.e. the base touched the terrain. Same as the manager stack.
    terminate_after_contacts_on = ["base"]
    termination_contact_force = 1.0

    def __post_init__(self):
        """Scale GPU buffers with num_envs and configure terrain."""
        self.sim.render_interval = self.decimation
        # Force curriculum terrain generation to avoid trimesh/numpy bug
        # in _generate_random_terrains (UnboundLocalError in trimesh.util.concatenate).
        # The ManagerBased version avoids this via CurriculumCfg; DirectRLEnv must set it explicitly.
        if self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.curriculum = True
        n = max(self.scene.num_envs, 4096)
        scale = n / 4096
        self.sim.physics = PhysxCfg(
            gpu_max_rigid_patch_count=int(400 * 2**15 * scale),
            gpu_max_rigid_contact_count=int(2**25 * scale),
            gpu_total_aggregate_pairs_capacity=int(2**24 * scale),
            gpu_heap_capacity=int(2**28 * scale),
            gpu_temp_buffer_capacity=int(2**26 * scale),
            enable_external_forces_every_iteration=True,
        )

        # TEST (DWQ_CLIP100=1): match the original legged_gym action clip (±100, effectively
        # unclipped). The ±1 clip × 0.25 scale restricts joint targets to default ±0.25 rad,
        # which may make a proper walking gait physically unreachable.
        import os as _os_ca
        if _os_ca.environ.get("DWQ_CLIP100"):
            self.clip_actions = 100.0
