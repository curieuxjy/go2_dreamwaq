"""DreamWaQ velocity-tracking environment configuration.

Based on IsaacLab's LocomotionVelocityRoughEnvCfg with DreamWaQ paper reward functions.
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import dreamwaq_manager.tasks.locomotion.mdp as mdp

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class DreamWaQSceneCfg(InteractiveSceneCfg):
    """Scene configuration for DreamWaQ locomotion."""

    # ground terrain
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
    # robot — set by subclass
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for DreamWaQ."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications — PD position control."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for DreamWaQ.

    Base model (policy only): 45 dims (no linear velocity).
    ang_vel(3) + gravity(3) + commands(3) + joint_pos(12) + joint_vel(12) + actions(12) = 45

    Oracle/Waq models add a critic group with privileged observations.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the actor (Base: 45 dims, no lin_vel).

        Noise levels are the *effective* values from the original IsaacGym
        legged_robot.py:_get_noise_scale_vec, where noise = noise_scales × obs_scales:
            ang_vel: 0.2 × 0.25 = 0.05
            gravity: 0.05
            dof_pos: 0.01 × 1.0 = 0.01
            dof_vel: 1.5 × 0.05 = 0.075
        """

        # NOTE: base_lin_vel is intentionally excluded for Base/Waq variants
        #
        # Sensor terms are wrapped in `mdp.delayed`, which reproduces the original's
        # system delay (paper Table II "System delay [0, 15] ms"; legged_robot.py:378-388):
        #     obs = delay * last_obs + (1 - delay) * obs,  delay = U[0,1) * 0.25
        # Commands and last_action are not measurements, so they are not delayed —
        # matching the original, which only blends the five sensor readings.
        base_ang_vel = ObsTerm(
            func=mdp.delayed, params={"func": mdp.base_ang_vel}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        projected_gravity = ObsTerm(
            func=mdp.delayed, params={"func": mdp.projected_gravity}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.delayed, params={"func": mdp.joint_pos_rel}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.delayed, params={"func": mdp.joint_vel_rel}, noise=Unoise(n_min=-0.075, n_max=0.075)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the critic (asymmetric training).

        Mirrors the original IsaacGym OnPolicyRunnerWAQ critic input
        = `obs_buf ⊕ privileged_obs_buf` (legged_robot.py:354-432) where
        `privileged_obs_buf = disturb_force(3) + heights(187) = 190`.

        Composition (clean = no noise/corruption applied):
            base_lin_vel(3)*  + base_ang_vel(3) + gravity(3) + commands(3)
            + joint_pos(12) + joint_vel(12) + actions(12)
            + disturb_force(3) + height_scan(187)
            * lin_vel only included when actor obs also has it (Oracle).
              Waq overrides this term to None in go2_waq_cfg.

        Sizes:
            Oracle: 3 + 3+3+3 + 12+12+12 + 3 + 187 = 238
            Waq   :     3+3+3 + 12+12+12 + 3 + 187 = 235
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        # disturb_force: most recent push impulse (legged_robot.py:355). Populated
        # by the custom push_robots_with_disturb event in EventCfg below.
        disturb_force = ObsTerm(func=mdp.last_disturb_force)
        # Height scan: offset=0.0 to match the original's raw `root_z - terrain_z`.
        # IsaacLab's default offset=0.5 is an AnymalC convention not in the IsaacGym source.
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Domain randomization configuration matching DreamWaQ paper."""

    # startup — randomize physics properties
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # startup — randomize actuator gains (DreamWaQ: p_gains [0.9, 1.1], d_gains [0.9, 1.1])
    randomize_pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # Motor strength factor [0.9, 1.1] (paper Table II; go2_config randomize_motor_strength).
    # Separate from randomize_pd_gains: this is ONE shared factor per env scaling both gains,
    # which for a pure PD actuator is exactly a torque multiplier.
    randomize_motor_strength = EventTerm(
        func=mdp.randomize_motor_strength,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "strength_range": (0.9, 1.1),
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # Reset base — original (legged_robot.py:_reset_root_states L761-791) uses
    # xy ±1.0 m around the terrain origin and does NOT randomize yaw. Velocity
    # is U(±0.5) on all 6 DoFs (lin + ang).
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Push (legged_robot.py:_push_robots L793-802): 3D impulse on lin_vel
    # (x, y, z), recorded as `env.disturb_force` for the privileged obs.
    push_robot = EventTerm(
        func=mdp.push_robot_with_disturb,
        mode="interval",
        interval_range_s=(15.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)}},
    )


@configclass
class RewardsCfg:
    """DreamWaQ paper reward functions (Go2RoughBaseCfg.rewards.scales).

    Reference: DreamWaQ paper Table I / Go2RoughBaseCfg in original codebase.
    """

    # -- task rewards (positive)
    # Names aligned with Direct version for consistent wandb logging
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # -- penalties (from DreamWaQ paper)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

    # -- DreamWaQ-specific rewards (custom implementations)
    joint_power = RewTerm(func=mdp.joint_power_l1, weight=-2.0e-5)
    # -1e-6, NOT the paper's printed -1e-5. Table I appears to have a 10x typo: the authors'
    # legged_gym config (a1_config.py, class annotated "SAME reward functions with the paper")
    # matches Table I to the decimal on the other 11 terms and uses -1.0e-6 for this one.
    # At -1e-5 this is not a regularizer but the DOMINANT term -- measured per-step |r| mean 16.2
    # vs tracking's 14.2, max 8447, because var(tau*qd)^2 is 4th-order in torque. See PAPER.md §2.
    power_distribution = RewTerm(func=mdp.power_distribution_l2, weight=-1.0e-6)
    smoothness = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "desired_height": 0.12,
        },
    )

    # -- base height penalty (uses IsaacLab built-in)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_height": 0.30,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for DreamWaQ."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Original DreamWaQ termination: trunk contact force > 1 N (legged_gym `check_termination`
    # with `terminate_after_contacts_on = ["base"]`). `dreamwaq_direct` uses the same condition.
    # Contact termination lets a tipped robot recover (it only dies when the trunk touches the
    # ground), unlike a tilt-angle cutoff that kills mid-push — important because DreamWaQ pushes
    # the robot with U(+-1 m/s) impulses every second.
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for DreamWaQ."""

    terrain_level = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class DreamWaQVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """DreamWaQ velocity-tracking environment configuration.

    This is the base environment config. Robot-specific configs (Go2, A1) inherit from this.
    """

    # Scene settings
    scene: DreamWaQSceneCfg = DreamWaQSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Scale GPU buffers with num_envs (base: 4096 envs)
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
        # update sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # TEST (DWQ_OFF_PUSH=1): official Go2 push regime — every 10–15 s, ±0.5 m/s, xy only.
        # DreamWaQ's original push (every 1 s, ±1 m/s on x/y/z) is 10–30x more frequent and 2x
        # stronger; isolating whether that difficulty gap is what keeps tracking depressed.
        import os as _os_p
        if _os_p.environ.get("DWQ_OFF_PUSH"):
            self.events.push_robot.interval_range_s = (10.0, 15.0)
            self.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
        # TEST (DWQ_OFF_REW=1): official Go2 reward weights — higher tracking (1.5/0.75) and the
        # movement-suppressing DreamWaQ penalties zeroed (official Go2 does not use them).
        if _os_p.environ.get("DWQ_OFF_REW"):
            self.rewards.track_lin_vel_xy.weight = 1.5
            self.rewards.track_ang_vel_z.weight = 0.75
            self.rewards.orientation.weight = 0.0
            self.rewards.base_height.weight = 0.0
            self.rewards.smoothness.weight = 0.0
            self.rewards.joint_power.weight = 0.0
            self.rewards.power_distribution.weight = 0.0
            self.rewards.foot_clearance.weight = 0.0
        # enable curriculum for terrain generator
        if getattr(self.curriculum, "terrain_level", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
