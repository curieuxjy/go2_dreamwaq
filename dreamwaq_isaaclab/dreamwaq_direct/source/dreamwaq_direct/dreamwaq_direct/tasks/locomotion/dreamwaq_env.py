"""DreamWaQ DirectRLEnv implementation.

Direct port of the original IsaacGym DreamWaQ legged_robot.py to IsaacLab's
DirectRLEnv API. All observation, reward, reset, and command logic is implemented
inline for maximum transparency and 1:1 correspondence with the original code.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, quat_rotate_inverse, yaw_quat, wrap_to_pi

from .dreamwaq_env_cfg import DreamWaQDirectEnvCfg


class DreamWaQEnv(DirectRLEnv):
    """DreamWaQ locomotion environment using DirectRLEnv.

    Matches the original IsaacGym LeggedRobot environment step loop:
    1. Apply actions as PD position targets
    2. Step physics (decimation times)
    3. Compute dones (termination + timeout)
    4. Compute rewards
    5. Reset terminated envs
    6. Compute observations

    Provides additional methods for the DreamWaQ runner:
    - get_obs_history(): flattened observation history [num_envs, len_obs_history * num_obs]
    - get_true_vel(): body-frame linear velocity [num_envs, 3]
    """

    cfg: DreamWaQDirectEnvCfg

    def __init__(self, cfg: DreamWaQDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- action buffers ---
        num_actions = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, num_actions, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, num_actions, device=self.device)
        self._prev_prev_actions = torch.zeros(self.num_envs, num_actions, device=self.device)

        # --- command buffers ---
        # [lin_vel_x, lin_vel_y, ang_vel_z, heading]
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)
        self._command_time_left = torch.zeros(self.num_envs, device=self.device)

        # --- observation history buffer ---
        self._obs_history_buf = torch.zeros(
            self.num_envs, self.cfg.len_obs_history, self.cfg.observation_space,
            device=self.device,
        )

        # --- system delay buffers ---
        if self.cfg.system_delay:
            self._delay_time = torch.zeros(self.num_envs, 1, device=self.device)
            self._last_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
            self._last_ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
            self._last_projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)
            self._last_joint_pos = torch.zeros(self.num_envs, num_actions, device=self.device)
            self._last_joint_vel = torch.zeros(self.num_envs, num_actions, device=self.device)

        # --- true velocity buffer (always available for runner) ---
        self._true_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)

        # --- reward logging ---
        self._reward_names = [
            "track_lin_vel_xy",
            "track_ang_vel_z",
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
            "base_height",
            "dof_acc",
            "action_rate",
            "smoothness",
            "joint_power",
            "power_distribution",
            "foot_clearance",
            "dof_pos_limits",
        ]
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self._reward_names
        }

        # --- termination stats, in the same units as the manager stack ---
        # Isaac Lab's TerminationManager logs `_last_episode_dones.float().mean(dim=0)`: per env it
        # remembers which term(s) ended that env's most recent episode, and reports the fraction of
        # envs per term. Mirror it here so `Episode_Termination/*` is comparable across both stacks
        # (a raw count would scale with num_envs). Column order: [base_contact, time_out].
        self._termination_names = ["base_contact", "time_out"]
        self._last_episode_dones = torch.zeros(
            self.num_envs, len(self._termination_names), dtype=torch.bool, device=self.device
        )

        # --- command tracking metrics (matches Manager version's command_manager metrics) ---
        self._command_error_vel_xy = torch.zeros(self.num_envs, device=self.device)
        self._command_error_vel_yaw = torch.zeros(self.num_envs, device=self.device)

        # --- terrain curriculum flag ---
        # Skip curriculum updates on the very first reset (no meaningful data yet)
        self._init_done = False

        # --- disturb_force buffer (legged_robot.py:_init_buffers L1002) ---
        # Holds the most recent random impulse from _push_robots, exposed in the
        # privileged obs (mirrors original `self.disturb_force.reshape(-1, 3)`).
        self._disturb_force = torch.zeros(self.num_envs, 3, device=self.device)

        # --- push interval (legged_robot.py:_post_physics_step_callback L598) ---
        # Original increments common_step_counter every step; pushes when it hits
        # `push_interval = ceil(push_interval_s / dt_env)`. dt_env = sim.dt * decimation.
        if self.cfg.push_robots:
            push_dt = self.cfg.sim.dt * self.cfg.decimation
            self._push_interval_steps = max(1, int(self.cfg.push_interval_s / push_dt))
        else:
            self._push_interval_steps = None

        # --- velocity command + heading visualization (GUI only) ---
        # Green arrow: commanded body-frame xy velocity (length scales with |v_xy|).
        # Blue arrow:  current body heading (yaw-only; fixed length).
        self._command_vis = None
        self._heading_vis = None
        if self.sim.has_gui:
            cmd_vis_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
            cmd_vis_cfg.markers["arrow"].scale = (1.2, 0.4, 0.4)
            self._command_vis = VisualizationMarkers(cmd_vis_cfg)

            heading_vis_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/heading")
            heading_vis_cfg.markers["arrow"].scale = (1.2, 0.4, 0.4)
            self._heading_vis = VisualizationMarkers(heading_vis_cfg)

        # --- body indices ---
        # Index into the *robot* articulation (body_pos_w / body_lin_vel_w), whose body
        # ordering differs from the contact sensor's. Using the contact sensor's order here
        # would select the wrong bodies for the foot-clearance reward.
        self._feet_ids, _ = self._robot.find_bodies(".*foot")

        # Termination bodies, indexed in the *contact sensor's* ordering (see above — it differs
        # from the articulation's). Mirrors legged_gym `terminate_after_contacts_on = ["base"]`.
        self._termination_contact_ids, _ = self._contact_sensor.find_sensors(
            self.cfg.terminate_after_contacts_on
        )

        # --- startup domain randomization (replaces EventManager "startup" mode) ---
        # Mirrors legged_robot.py:_process_rigid_shape_props (friction),
        # _process_rigid_body_props (base mass + CoM), and per-env PD-gain scaling.
        # Runs once per env after the articulation is initialized by super().__init__.
        self._apply_startup_randomization()

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # InteractiveScene only auto-binds terrain when it's declared inside the scene cfg
        # (manager-based pattern). For DirectRLEnv we instantiate it manually, so we must
        # also wire it into the scene — otherwise scene.env_origins falls back to the
        # default grid (env_spacing-based) and EventManager's reset_root_state_uniform
        # respawns every robot at the grid layout instead of terrain.env_origins.
        self.scene._terrain = self._terrain

        self.scene.clone_environments(copy_from_source=False)
        # Filter inter-environment collisions on GPU too. The official Direct examples guard this
        # with `if self.device == "cpu"`, but InteractiveScene (used by the manager-based stack)
        # calls it unconditionally for any PhysX backend (interactive_scene.py: `if
        # self.cfg.filter_collisions and "physx" in self.physics_backend`). Without it, 4096
        # unfiltered envs overflow the PhysX broadphase pair buffers and the contact sensor
        # reports phantom forces on non-contacting bodies — every env then terminates on
        # `base_contact` at step 1. Flat terrain and small env counts hide the problem.
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_prev_actions = self._prev_actions.clone()
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-self.cfg.clip_actions, self.cfg.clip_actions)
        # PD position target: action_scale * action + default_joint_pos
        self._processed_actions = (
            self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        # Body-frame quantities (provided by IsaacLab articulation data)
        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel

        # Store true velocity for runner (raw, before scaling)
        self._true_lin_vel_b = lin_vel_b.clone()

        # Observation scaling (legged_robot.py:416-421, 979) — applied BEFORE noise so the noise
        # magnitudes below (already in obs-scale units) are correct. gravity and actions use scale 1.0.
        lin_vel_b = lin_vel_b * self.cfg.obs_scale_lin_vel
        ang_vel_b = ang_vel_b * self.cfg.obs_scale_ang_vel
        joint_pos_rel = joint_pos_rel * self.cfg.obs_scale_dof_pos
        joint_vel = joint_vel * self.cfg.obs_scale_dof_vel

        # Add observation noise (legged_robot.py:_get_noise_scale_vec at L865-884).
        # The original computes noise_vec = noise_scales * noise_level * obs_scales, i.e. the noise
        # is added to the *scaled* observation (applied just above). The effective ±ranges below are
        # these products with the default config:
        #   lin_vel: 0.1 * 1.0 * 2.0  = 0.20
        #   ang_vel: 0.2 * 1.0 * 0.25 = 0.05
        #   gravity: 0.05 * 1.0       = 0.05
        #   dof_pos: 0.01 * 1.0 * 1.0 = 0.01
        #   dof_vel: 1.5 * 1.0 * 0.05 = 0.075
        if self.cfg.obs_noise:
            # `lin_vel_b` reaches the ACTOR only when include_lin_vel_in_obs is True, i.e. Oracle.
            # Oracle is defined (CLAUDE.md / PAPER.md) as "Base + the TRUE base linear velocity" —
            # the privileged upper bound that Waq's CENet is trying to approach. Waq's AdaBoot
            # true-velocity fallback and the CENet regression target both read the simulator's
            # root_lin_vel_b with NO noise (get_true_vel()), so noising Oracle's copy handed the
            # supposed ceiling a strictly worse velocity signal than the arm it must bound.
            # The other 44 terms keep their noise: Oracle's privilege is "knows the velocity
            # exactly", not "has clean proprioception". Mirrors Manager's
            # go2_waq_official_cfg._make_oracle_velocity_exact().
            if not self.cfg.include_lin_vel_in_obs:
                lin_vel_b = lin_vel_b + torch.empty_like(lin_vel_b).uniform_(-0.2, 0.2)
            ang_vel_b = ang_vel_b + torch.empty_like(ang_vel_b).uniform_(-0.05, 0.05)
            projected_gravity = projected_gravity + torch.empty_like(projected_gravity).uniform_(-0.05, 0.05)
            joint_pos_rel = joint_pos_rel + torch.empty_like(joint_pos_rel).uniform_(-0.01, 0.01)
            joint_vel = joint_vel + torch.empty_like(joint_vel).uniform_(-0.075, 0.075)

        # Store last values before delay interpolation
        if self.cfg.system_delay:
            cur_lin_vel_b = self._delay_time * self._last_lin_vel_b + (1 - self._delay_time) * lin_vel_b
            cur_ang_vel_b = self._delay_time * self._last_ang_vel_b + (1 - self._delay_time) * ang_vel_b
            cur_gravity = self._delay_time * self._last_projected_gravity + (1 - self._delay_time) * projected_gravity
            cur_joint_pos = self._delay_time * self._last_joint_pos + (1 - self._delay_time) * joint_pos_rel
            cur_joint_vel = self._delay_time * self._last_joint_vel + (1 - self._delay_time) * joint_vel

            self._last_lin_vel_b = lin_vel_b.clone()
            self._last_ang_vel_b = ang_vel_b.clone()
            self._last_projected_gravity = projected_gravity.clone()
            self._last_joint_pos = joint_pos_rel.clone()
            self._last_joint_vel = joint_vel.clone()

            lin_vel_b = cur_lin_vel_b
            ang_vel_b = cur_ang_vel_b
            projected_gravity = cur_gravity
            joint_pos_rel = cur_joint_pos
            joint_vel = cur_joint_vel

        # Velocity commands [v_x, v_y, omega_z], scaled by (lin_vel, lin_vel, ang_vel)
        commands_3 = self._commands[:, :3] * self._commands.new_tensor(
            [self.cfg.obs_scale_lin_vel, self.cfg.obs_scale_lin_vel, self.cfg.obs_scale_ang_vel]
        )

        # Build policy observation (45 dims for Base/Waq, 48 for Oracle)
        obs_parts = []
        if self.cfg.include_lin_vel_in_obs:
            obs_parts.append(lin_vel_b)  # +3 dims for Oracle
        obs_parts.extend([
            ang_vel_b,          # 3
            projected_gravity,  # 3
            commands_3,         # 3
            joint_pos_rel,      # 12
            joint_vel,          # 12
            self._actions,      # 12
        ])
        obs = torch.cat(obs_parts, dim=-1)

        # Build privileged (critic) observation matching the original IsaacGym layout.
        # Original (legged_robot.py:354-366): privileged_obs = disturb_force(3) + heights(187) = 190.
        # In the original OnPolicyRunnerWAQ the critic input is `actor_obs_buf ⊕ privileged_obs_buf`
        # (concatenation). IsaacLab's rsl_rl wrapper takes the "critic" key directly without
        # re-concatenating with "policy", so we build state = clean_actor_obs ⊕ disturb ⊕ heights
        # here to reproduce the exact critic input the original runner sees.
        # Critic obs are the TRUE (un-noised, un-delayed) actor obs, scaled the same way.
        true_ang_vel_b = self._robot.data.root_ang_vel_b * self.cfg.obs_scale_ang_vel
        true_gravity = self._robot.data.projected_gravity_b
        true_joint_pos_rel = (
            self._robot.data.joint_pos - self._robot.data.default_joint_pos
        ) * self.cfg.obs_scale_dof_pos
        true_joint_vel = self._robot.data.joint_vel * self.cfg.obs_scale_dof_vel

        # Height scan (legged_robot.py:351-353): clip(root_z - measured_heights, -1, 1).
        # IsaacLab's mdp.height_scan helper subtracts an extra 0.5 (target-height
        # convention used by AnymalC) but the original IsaacGym formula has no offset,
        # so we omit it here to match.
        height_data = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1)
            - self._height_scanner.data.ray_hits_w[..., 2]
        ).clamp(-1.0, 1.0) * self.cfg.obs_scale_height

        # disturb_force = most recent push impulse (legged_robot.py:355 in compute_observations).
        # Updated by _push_robots, zeroed by _reset_idx. Lives in self._disturb_force.
        disturb_force = self._disturb_force

        # state = clean actor obs ⊕ privileged_obs (disturb_3 + heights_187)
        state_parts = []
        if self.cfg.include_lin_vel_in_obs:
            state_parts.append(self._true_lin_vel_b * self.cfg.obs_scale_lin_vel)  # +3 for Oracle only
        state_parts.extend([
            true_ang_vel_b,         # 3
            true_gravity,           # 3
            commands_3,             # 3
            true_joint_pos_rel,     # 12
            true_joint_vel,         # 12
            self._actions,          # 12
            disturb_force,          # 3   (privileged: original puts disturb BEFORE heights)
            height_data,            # 187
        ])
        state = torch.cat(state_parts, dim=-1)

        # Update observation history buffer
        self._obs_history_buf[:, :-1] = self._obs_history_buf[:, 1:].clone()
        self._obs_history_buf[:, -1] = obs

        observations = {"policy": obs}
        if self.cfg.state_space > 0:
            observations["critic"] = state

        # Velocity command arrow above each robot (GUI only, no-op when headless)
        if self._command_vis is not None:
            self._update_command_visualization()

        return observations

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        dt = self.step_dt

        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b

        # --- tracking rewards ---
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - lin_vel_b[:, :2]), dim=1
        )
        track_lin_vel = torch.exp(-lin_vel_error / self.cfg.tracking_sigma) * self.cfg.rew_track_lin_vel_xy

        ang_vel_error = torch.square(self._commands[:, 2] - ang_vel_b[:, 2])
        track_ang_vel = torch.exp(-ang_vel_error / self.cfg.tracking_sigma) * self.cfg.rew_track_ang_vel_z

        # --- penalties ---
        lin_vel_z = torch.square(lin_vel_b[:, 2]) * self.cfg.rew_lin_vel_z
        ang_vel_xy = torch.sum(torch.square(ang_vel_b[:, :2]), dim=1) * self.cfg.rew_ang_vel_xy
        orientation = torch.sum(torch.square(projected_gravity[:, :2]), dim=1) * self.cfg.rew_orientation

        # Base height (legged_robot.py:1589-1595): square(mean(root_z - terrain_z) - target).
        # RayCaster returns inf for rays that miss the terrain mesh; filter those out so a
        # single missed ray doesn't pollute the mean into ±inf and the reward into -inf.
        root_z = self._robot.data.root_pos_w[:, 2]
        ray_hits_z = torch.nan_to_num(
            self._height_scanner.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0
        )
        mean_terrain_z = torch.mean(ray_hits_z, dim=1)
        base_height_above_terrain = root_z - mean_terrain_z
        base_height = torch.square(base_height_above_terrain - self.cfg.base_height_target) * self.cfg.rew_base_height

        # Joint acceleration
        dof_acc = torch.sum(torch.square(self._robot.data.joint_acc), dim=1) * self.cfg.rew_dof_acc

        # Action rate
        action_rate = torch.sum(
            torch.square(self._actions - self._prev_actions), dim=1
        ) * self.cfg.rew_action_rate

        # Smoothness (jerk)
        jerk = self._actions - 2.0 * self._prev_actions + self._prev_prev_actions
        smoothness = torch.sum(torch.square(jerk), dim=1) * self.cfg.rew_smoothness

        # Joint power
        torques = self._robot.data.applied_torque
        joint_vel = self._robot.data.joint_vel
        joint_power = torch.sum(
            torch.abs(torques) * torch.abs(joint_vel), dim=1
        ) * self.cfg.rew_joint_power

        # Power distribution
        power_per_joint = torques * joint_vel
        power_dist = torch.square(torch.var(power_per_joint, dim=-1)) * self.cfg.rew_power_distribution

        # Foot clearance (legged_robot.py:1699-1711):
        #   feet_z_above_terrain = feet_pos_w[:, :, 2] - measured_heights_under_feet
        #   reward = sum((feet_z_above_terrain - desired_foot_height)^2 * lateral_speed)
        # Original samples per-foot terrain height from a precomputed height map (one
        # height sample per foot xy). IsaacLab equivalent would require 4 dedicated
        # RayCasters; here we approximate `measured_heights_under_feet` with the mean
        # of the existing 187-ray base scan (exact on flat terrain, approximate on slopes
        # — same approximation used for base_height above). No contact mask: the original
        # always penalizes regardless of contact state.
        feet_pos_w = self._robot.data.body_pos_w[:, self._feet_ids, :]
        feet_vel_w = self._robot.data.body_lin_vel_w[:, self._feet_ids, :]
        feet_z_above_terrain = feet_pos_w[:, :, 2] - mean_terrain_z[:, None]
        feet_lateral_speed = torch.sqrt(feet_vel_w[:, :, 0] ** 2 + feet_vel_w[:, :, 1] ** 2)
        foot_clearance = torch.sum(
            (feet_z_above_terrain - self.cfg.desired_foot_clearance) ** 2 * feet_lateral_speed,
            dim=1,
        ) * self.cfg.rew_foot_clearance

        # DOF position limits
        joint_pos = self._robot.data.joint_pos
        limits = self._robot.data.soft_joint_pos_limits
        out_of_limits = torch.sum(
            (joint_pos - limits[:, :, 1]).clamp(min=0.0)
            + (limits[:, :, 0] - joint_pos).clamp(min=0.0),
            dim=1,
        ) * self.cfg.rew_dof_pos_limits

        # Collect all rewards
        rewards = {
            "track_lin_vel_xy": track_lin_vel * dt,
            "track_ang_vel_z": track_ang_vel * dt,
            "lin_vel_z": lin_vel_z * dt,
            "ang_vel_xy": ang_vel_xy * dt,
            "orientation": orientation * dt,
            "base_height": base_height * dt,
            "dof_acc": dof_acc * dt,
            "action_rate": action_rate * dt,
            "smoothness": smoothness * dt,
            "joint_power": joint_power * dt,
            "power_distribution": power_dist * dt,
            "foot_clearance": foot_clearance * dt,
            "dof_pos_limits": out_of_limits * dt,
        }

        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # only_positive_rewards (legged_robot.py:compute_reward): clip the per-step total at 0.
        # It is legged_gym's default, and the usual argument for it is that the movement penalties
        # can make a step's net reward negative so standing still becomes optimal. DreamWaQ does
        # NOT use it — the authors set `only_positive_rewards = False` in every reward class of
        # a1_config.py — so the cfg default here is False and this branch is normally dead.
        # Kept as a switch because the flag is a real knob in the original code.
        if self.cfg.only_positive_rewards:
            total_reward = total_reward.clamp(min=0.0)

        # Logging (per-term sums are accumulated pre-clip, matching legged_gym)
        for name, value in rewards.items():
            self._episode_sums[name] += value

        return total_reward

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Failure termination: trunk touches the terrain.
        #
        # Identical to the original IsaacGym DreamWaQ (`legged_robot.py:check_termination`):
        #   reset_buf = any(norm(contact_forces[:, termination_contact_indices, :]) > 1.0)
        # with `terminate_after_contacts_on = ["base"]` (go2_config.py). The manager stack uses
        # `mdp.illegal_contact` with the same body/threshold, so both stacks match the original
        # and each other. Both contact sensors use history_length=3, hence the max over history.
        net_contact_forces = self._contact_sensor.data.net_forces_w_history.torch
        base_contact = torch.any(
            torch.max(
                torch.linalg.norm(net_contact_forces[:, :, self._termination_contact_ids], dim=-1), dim=1
            )[0]
            > self.cfg.termination_contact_force,
            dim=1,
        )

        # Remember which term ended each env's most recent episode (see __init__) — matches
        # TerminationManager.compute()'s `_last_episode_dones` update.
        dones = torch.stack((base_contact, time_out), dim=1)
        rows = dones.any(dim=1)
        self._last_episode_dones[rows] = dones[rows]

        return base_contact, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)

        self._robot.reset(env_ids)

        # --- Terrain curriculum (must run BEFORE the inline reset) ---
        # We measure distance walked from the previous terrain origin BEFORE the new
        # reset overwrites root_pos. self._commands still holds the expired episode's
        # command (_resample_commands runs later in this method).
        if self._init_done and self._terrain.terrain_origins is not None:
            distance = torch.norm(
                self._robot.data.root_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2],
                dim=1,
            )
            terrain_size = self.cfg.terrain.terrain_generator.size[0]
            move_up = distance > terrain_size / 2
            move_down = (
                distance
                < torch.norm(self._commands[env_ids, :2], dim=1)
                * self.max_episode_length_s
                * 0.5
            ) & ~move_up
            self._terrain.update_env_origins(env_ids, move_up, move_down)

        # super()._reset_idx still handles scene.reset(env_ids) and zeroes
        # episode_length_buf. With cfg.events = None it does NOT run any
        # EventManager-based randomization (we replace those with inline calls below).
        super()._reset_idx(env_ids)

        # --- Inline reset randomization (legged_robot.py:_reset_root_states + _reset_dofs) ---
        self._reset_root_states(env_ids)
        self._reset_dofs(env_ids)
        # Disturb force resets to zero on episode reset (no push has happened yet).
        self._disturb_force[env_ids] = 0.0

        # Spread out resets
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._prev_prev_actions[env_ids] = 0.0

        # Reset observation history
        self._obs_history_buf[env_ids] = 0.0

        # Reset system delay
        if self.cfg.system_delay:
            self._delay_time[env_ids] = torch.rand(len(env_ids), 1, device=self.device) * 0.25
            self._last_lin_vel_b[env_ids] = 0.0
            self._last_ang_vel_b[env_ids] = 0.0
            self._last_projected_gravity[env_ids] = 0.0
            self._last_joint_pos[env_ids] = 0.0
            self._last_joint_vel[env_ids] = 0.0

        # Resample commands (AFTER curriculum so curriculum uses the expired command)
        self._resample_commands(env_ids)
        self._command_time_left[env_ids] = self.cfg.command_resampling_time

        # Log episode sums
        extras = {}
        for name in self._reward_names:
            episodic_avg = torch.mean(self._episode_sums[name][env_ids])
            extras[f"Episode_Reward/{name}"] = episodic_avg / self.max_episode_length_s
            self._episode_sums[name][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = {}
        termination_stats = self._last_episode_dones.float().mean(dim=0)
        for i, name in enumerate(self._termination_names):
            extras[f"Episode_Termination/{name}"] = termination_stats[i].item()
        # Command tracking metrics
        extras["Metrics/base_velocity/error_vel_xy"] = torch.mean(
            self._command_error_vel_xy[env_ids]
        ).item()
        extras["Metrics/base_velocity/error_vel_yaw"] = torch.mean(
            self._command_error_vel_yaw[env_ids]
        ).item()
        self._command_error_vel_xy[env_ids] = 0.0
        self._command_error_vel_yaw[env_ids] = 0.0
        # Terrain curriculum level
        if self._terrain.terrain_origins is not None:
            extras["Curriculum/terrain_level"] = torch.mean(
                self._terrain.terrain_levels[env_ids].float()
            ).item()
        self.extras["log"].update(extras)

        self._init_done = True

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _resample_commands(self, env_ids: torch.Tensor):
        """Resample velocity commands for given environments."""
        n = len(env_ids)
        self._commands[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*self.cfg.lin_vel_x_range)
        self._commands[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*self.cfg.lin_vel_y_range)
        self._commands[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*self.cfg.ang_vel_z_range)
        if self.cfg.heading_command:
            self._commands[env_ids, 3] = torch.empty(n, device=self.device).uniform_(*self.cfg.heading_range)

    def _update_commands(self):
        """Update commands: resample expired commands and compute heading-based yaw rate."""
        # Decrement timers
        self._command_time_left -= self.step_dt

        # Resample expired commands
        expired = self._command_time_left <= 0.0
        expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
        if len(expired_ids) > 0:
            self._resample_commands(expired_ids)
            self._command_time_left[expired_ids] = self.cfg.command_resampling_time

        # Heading command → angular velocity
        if self.cfg.heading_command:
            forward = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
            heading = torch.atan2(
                self._robot.data.root_quat_w[:, 1] * self._robot.data.root_quat_w[:, 2]
                + self._robot.data.root_quat_w[:, 0] * self._robot.data.root_quat_w[:, 3],
                0.5 - self._robot.data.root_quat_w[:, 2] ** 2 - self._robot.data.root_quat_w[:, 3] ** 2,
            )
            heading_error = wrap_to_pi(self._commands[:, 3] - heading)
            self._commands[:, 2] = torch.clamp(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ang_vel_z_range[0],
                max=self.cfg.ang_vel_z_range[1],
            )

        # Zero small commands (standing envs)
        cmd_norm = torch.norm(self._commands[:, :2], dim=1)
        small_cmd = cmd_norm < 0.2
        self._commands[small_cmd, :2] = 0.0

    # ------------------------------------------------------------------
    # Domain randomization (inline, replacing IsaacLab's EventManager).
    # All four startup helpers + push + reset_root_states + reset_dofs are
    # 1:1 ports of the corresponding routines in
    # dreamwaq/legged_gym/legged_gym/envs/base/legged_robot.py.
    # ------------------------------------------------------------------

    def _apply_startup_randomization(self):
        """Apply per-env startup randomization once after articulation init."""
        if self.cfg.randomize_friction:
            self._randomize_friction()
        if self.cfg.randomize_base_mass:
            self._randomize_base_mass()
        if self.cfg.randomize_com:
            self._randomize_com()
        if self.cfg.randomize_pd_gains:
            self._randomize_pd_gains()
        if self.cfg.randomize_motor_strength:
            self._randomize_motor_strength()

    def _randomize_friction(self):
        """legged_robot.py:_process_rigid_shape_props (L470-495).

        Bucket-based per-env friction sampling (default 64 buckets in [0.2, 1.25]),
        applied uniformly to ALL collision shapes of the robot. Restitution = 0.
        IsaacLab's PhysX views require CPU tensors here.
        """
        materials = wp.to_torch(self._robot.root_physx_view.get_material_properties())
        # shape: (num_envs, num_shapes, 3) — [static_friction, dynamic_friction, restitution]
        device = materials.device
        fr_low, fr_high = self.cfg.friction_range
        num_buckets = self.cfg.friction_num_buckets
        bucket_friction = torch.empty(num_buckets, device=device).uniform_(fr_low, fr_high)
        bucket_ids = torch.randint(0, num_buckets, (self.num_envs,), device=device)
        per_env_friction = bucket_friction[bucket_ids]  # (num_envs,)
        materials[..., 0] = per_env_friction.unsqueeze(-1)  # static
        materials[..., 1] = per_env_friction.unsqueeze(-1)  # dynamic
        materials[..., 2] = 0.0  # restitution
        env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=device)
        self._robot.root_physx_view.set_material_properties(
            wp.from_torch(materials.contiguous()), wp.from_torch(env_ids)
        )

    def _randomize_base_mass(self):
        """legged_robot.py:_process_rigid_body_props (L557-559): props[0].mass += U(range).

        Adds a per-env scalar to the BASE body's mass only.
        """
        masses = wp.to_torch(self._robot.root_physx_view.get_masses()).clone()
        # shape: (num_envs, num_bodies)
        device = masses.device
        base_body_idx, _ = self._robot.find_bodies("base")
        low, high = self.cfg.added_mass_range
        mass_offset = torch.empty(self.num_envs, device=device).uniform_(low, high)
        masses[:, base_body_idx[0]] += mass_offset
        env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=device)
        self._robot.root_physx_view.set_masses(wp.from_torch(masses.contiguous()), wp.from_torch(env_ids))

    def _randomize_com(self):
        """legged_robot.py:_process_rigid_body_props (L562-567): com.{x,y,z} += U(range) for all bodies."""
        coms = wp.to_torch(self._robot.root_physx_view.get_coms()).clone()
        # shape: (num_envs, num_bodies, 7) — [pos(3), quat(4)]
        device = coms.device
        n_bodies = coms.shape[1]
        low, high = self.cfg.com_range
        com_offset = torch.empty(self.num_envs, n_bodies, 3, device=device).uniform_(low, high)
        coms[..., 0:3] += com_offset
        env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=device)
        self._robot.root_physx_view.set_coms(wp.from_torch(coms.contiguous()), wp.from_torch(env_ids))

    def _randomize_pd_gains(self):
        """legged_robot.py uses self.cfg.domain_rand.{p,d}_gains_range to scale gains per env.

        For ImplicitActuator (Go2 default) gains are CPU-side; we write back to sim after scaling.
        """
        p_low, p_high = self.cfg.p_gains_range
        d_low, d_high = self.cfg.d_gains_range
        for actuator in self._robot.actuators.values():
            n_envs, n_joints = actuator.stiffness.shape
            stiffness_scale = torch.empty(n_envs, n_joints, device=actuator.stiffness.device).uniform_(p_low, p_high)
            damping_scale = torch.empty(n_envs, n_joints, device=actuator.damping.device).uniform_(d_low, d_high)
            actuator.stiffness *= stiffness_scale
            actuator.damping *= damping_scale
            if isinstance(actuator, ImplicitActuator):
                self._robot.write_joint_stiffness_to_sim_index(stiffness=actuator.stiffness, joint_ids=actuator.joint_indices)
                self._robot.write_joint_damping_to_sim_index(damping=actuator.damping, joint_ids=actuator.joint_indices)

    def _randomize_motor_strength(self):
        """Motor strength factor (paper Table II; go2_config randomize_motor_strength).

        legged_gym multiplies the computed torque by a per-env factor
        (`legged_robot.py:_compute_torques`: `torques * self.motor_strengths`). For a pure PD
        actuator `tau = Kp*(q_des - q) - Kd*qd`, scaling `tau` by `m` is exactly the same as
        scaling both `Kp` and `Kd` by `m`, so we sample ONE shared factor per env.

        Distinct from `_randomize_pd_gains`, which samples Kp and Kd independently.
        """
        low, high = self.cfg.motor_strength_range
        for actuator in self._robot.actuators.values():
            n_envs, _ = actuator.stiffness.shape
            factor = torch.empty(n_envs, 1, device=actuator.stiffness.device).uniform_(low, high)
            actuator.stiffness *= factor
            actuator.damping *= factor
            if isinstance(actuator, ImplicitActuator):
                self._robot.write_joint_stiffness_to_sim_index(stiffness=actuator.stiffness, joint_ids=actuator.joint_indices)
                self._robot.write_joint_damping_to_sim_index(damping=actuator.damping, joint_ids=actuator.joint_indices)

    def _reset_root_states(self, env_ids: torch.Tensor):
        """legged_robot.py:_reset_root_states (L761-791).

        Sets pose = base_init_state + terrain_origin + xy U(±1m), velocity = U(±0.5)
        on all 6 DoFs. Yaw is NOT randomized (matching the original).
        """
        n = len(env_ids)
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        # pos + terrain origin + xy noise
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        xy_low, xy_high = self.cfg.init_xy_range
        default_root_state[:, :2] += torch.empty(n, 2, device=self.device).uniform_(xy_low, xy_high)
        # velocity (lin + ang) ±0.5
        v_low, v_high = self.cfg.init_vel_range
        default_root_state[:, 7:13] = torch.empty(n, 6, device=self.device).uniform_(v_low, v_high)
        self._robot.write_root_pose_to_sim_index(root_pose=default_root_state[:, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=default_root_state[:, 7:13], env_ids=env_ids)

    def _reset_dofs(self, env_ids: torch.Tensor):
        """legged_robot.py:_reset_dofs (L740-759): pos = default × U[0.5, 1.5], vel = 0."""
        default_joint_pos = self._robot.data.default_joint_pos[env_ids]
        s_low, s_high = self.cfg.init_dof_pos_scale_range
        scale = torch.empty_like(default_joint_pos).uniform_(s_low, s_high)
        joint_pos = default_joint_pos * scale
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _push_robots(self):
        """legged_robot.py:_push_robots (L793-802).

        Random impulse: U(±max_push_vel_xy) on all 3 lin-vel components added to root vel.
        The sampled impulse is stored in `self._disturb_force` for the privileged obs.
        """
        max_v = self.cfg.max_push_vel_xy
        self._disturb_force = torch.empty(self.num_envs, 3, device=self.device).uniform_(-max_v, max_v)
        # Add to current root velocity, write back (lin + ang).
        full_vel = self._robot.data.root_vel_w.clone()  # (N, 6) lin(3) + ang(3)
        full_vel[:, :3] += self._disturb_force
        self._robot.write_root_velocity_to_sim_index(root_velocity=full_vel)

    # ------------------------------------------------------------------
    # Visualization (GUI only)
    # ------------------------------------------------------------------

    def _update_command_visualization(self):
        """Draw arrows above each robot.

        Green arrow: body-frame xy velocity command (length = |v_xy| * 3, in world frame).
        Blue arrow:  current body heading (yaw-only quaternion, fixed length).
        Mirrors isaaclab.envs.mdp.commands.velocity_command._debug_vis_callback.
        """
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # --- green: command arrow ---
        xy_vel = self._commands[:, :2]
        default_scale = self._command_vis.cfg.markers["arrow"].scale
        cmd_scale = torch.tensor(default_scale, device=self.device).repeat(xy_vel.shape[0], 1)
        cmd_scale[:, 0] *= torch.linalg.norm(xy_vel, dim=1) * 3.0

        heading_angle = torch.atan2(xy_vel[:, 1], xy_vel[:, 0])
        zeros = torch.zeros_like(heading_angle)
        cmd_quat_b = quat_from_euler_xyz(zeros, zeros, heading_angle)
        cmd_quat_w = quat_mul(self._robot.data.root_quat_w, cmd_quat_b)
        self._command_vis.visualize(base_pos_w, cmd_quat_w, cmd_scale)

        # --- blue: heading arrow (yaw-only of body, fixed length) ---
        heading_default = self._heading_vis.cfg.markers["arrow"].scale
        heading_scale = torch.tensor(heading_default, device=self.device).repeat(self.num_envs, 1)
        heading_quat_w = yaw_quat(self._robot.data.root_quat_w)
        self._heading_vis.visualize(base_pos_w, heading_quat_w, heading_scale)

    # ------------------------------------------------------------------
    # DreamWaQ-specific accessors (called by the runner)
    # ------------------------------------------------------------------

    def get_obs_history(self) -> torch.Tensor:
        """Return flattened observation history [num_envs, len_obs_history * obs_dim]."""
        return self._obs_history_buf.reshape(self.num_envs, -1)

    def get_true_vel(self) -> torch.Tensor:
        """Return the true body-frame linear velocity [num_envs, 3], **scaled** by the lin-vel
        observation scale. This is the CENet's velocity-estimation target — it mirrors the
        original ``get_true_vel() = base_lin_vel * obs_scales.lin_vel`` (legged_robot.py:1301)."""
        return self._true_lin_vel_b * self.cfg.obs_scale_lin_vel

    # ------------------------------------------------------------------
    # Override step to update commands + apply push events
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        """Override step to update commands, apply push events, and accumulate metrics."""
        self._update_commands()

        # Push event (legged_robot.py:_post_physics_step_callback L598-601):
        # apply impulse every push_interval_s. common_step_counter ticks once per
        # outer (env) step, so dt_env = sim.dt * decimation per increment.
        if self._push_interval_steps is not None and self.common_step_counter > 0:
            if self.common_step_counter % self._push_interval_steps == 0:
                self._push_robots()

        # Accumulate command tracking metrics (normalized by max command time)
        max_command_step = self.cfg.command_resampling_time / self.step_dt
        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        self._command_error_vel_xy += (
            torch.norm(self._commands[:, :2] - lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self._command_error_vel_yaw += (
            torch.abs(self._commands[:, 2] - ang_vel_b[:, 2]) / max_command_step
        )
        return super().step(action)
