"""Custom event terms for DreamWaQ.

Provides `push_robot_with_disturb` — same impulse as IsaacLab's
`push_by_setting_velocity` but additionally records the sampled velocity
delta on `env.disturb_force` so it can be exposed in the privileged obs
(matches the original IsaacGym `self.disturb_force` flow in
`legged_robot.py:_push_robots` and `compute_observations`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def push_robot_with_disturb(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push robot via root velocity impulse, recording the sampled delta.

    Mirrors `legged_robot.py:_push_robots` (L793-802):

        self.disturb_force = torch_rand_float(-max, max, (num_envs, 3))
        self.root_states[:, 7:10] += self.disturb_force

    The recorded `env.disturb_force` is read back by the `last_disturb_force`
    observation term in the privileged obs group.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = asset.device

    # Sample per-env impulse on selected components (default: x, y, z).
    delta = torch.zeros(n, 6, device=device)
    for axis, idx in (("x", 0), ("y", 1), ("z", 2), ("roll", 3), ("pitch", 4), ("yaw", 5)):
        if axis in velocity_range:
            lo, hi = velocity_range[axis]
            delta[:, idx] = torch.empty(n, device=device).uniform_(lo, hi)

    # Lazily allocate the disturb_force buffer on the env. It mirrors the original
    # `self.disturb_force` shape (num_envs, 3) — we only store the linear part since
    # that's what the original observation uses.
    if not hasattr(env, "disturb_force") or env.disturb_force.shape != (env.num_envs, 3):
        env.disturb_force = torch.zeros(env.num_envs, 3, device=device)
    env.disturb_force[env_ids] = delta[:, :3]

    # Apply impulse on top of current root velocity.
    full_vel = asset.data.root_vel_w[env_ids].clone()  # (n, 6)
    full_vel += delta
    asset.write_root_velocity_to_sim_index(root_velocity=full_vel, env_ids=env_ids)


def randomize_motor_strength(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    strength_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Scale actuator torque output by a per-environment factor.

    DreamWaQ paper Table II lists a "Motor strength factor" of [0.9, 1.1], and the original
    `go2_config.py` sets `randomize_motor_strength = True` with `motor_strength_range = [0.9, 1.1]`.
    legged_gym implements it as a multiplier on the computed torque
    (`legged_robot.py:_compute_torques`: `torques * self.motor_strengths`).

    Isaac Lab has no torque-scaling event, but for a pure PD actuator

        tau = Kp * (q_des - q) - Kd * qd

    multiplying `tau` by `m` is *exactly* the same as multiplying both `Kp` and `Kd` by `m`.
    So we sample one shared factor per environment and scale both gains by it.

    This is distinct from `randomize_actuator_gains`, which samples Kp and Kd *independently*
    (the paper's separate "Kp factor" / "Kd factor" rows). Both are applied.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    lo, hi = strength_range
    for actuator in asset.actuators.values():
        n_envs, n_joints = actuator.stiffness.shape
        # one factor per env, shared across that robot's joints -> a pure torque multiplier
        factor = torch.empty(n_envs, 1, device=actuator.stiffness.device).uniform_(lo, hi)
        actuator.stiffness *= factor
        actuator.damping *= factor
        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_stiffness_to_sim_index(
                stiffness=actuator.stiffness, joint_ids=actuator.joint_indices
            )
            asset.write_joint_damping_to_sim_index(
                damping=actuator.damping, joint_ids=actuator.joint_indices
            )
