"""DreamWaQ-specific reward functions.

These rewards are from the DreamWaQ paper (https://arxiv.org/abs/2301.10602)
and are not available in the default IsaacLab locomotion rewards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint power consumption: sum(|torque| * |velocity|)."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
        * torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]),
        dim=1,
    )


def power_distribution_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize uneven power distribution across joints: var(torque * velocity)^2."""
    asset = env.scene[asset_cfg.name]
    joint_power = (
        asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    )
    return torch.square(torch.var(joint_power, dim=-1))


def base_height_l2_safe(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """`(h^des - h)^2` (paper Table I #8), terrain-relative, with the ray-miss guard.

    Same as :func:`isaaclab.envs.mdp.rewards.base_height_l2` except that the height-scanner
    hits are sanitized before averaging. A :class:`RayCaster` writes **inf** for rays that hit
    nothing — which happens once a robot leaves the generated terrain mesh — and the built-in
    reward averages the raw hits, so a single stray env poisons the whole PPO batch:
    ``inf`` reward -> ``inf`` return -> ``NaN`` gradients -> policy dead.

    Measured (PAPER.md §2): with the built-in, rough Base collapses at iteration ~180-224 —
    ``Loss/value`` NaN, ``Policy/mean_std`` pinned at the log_std clamp ceiling for the rest of
    the run. The instrumented run caught the exact event (``base_height=inf`` at iteration 189).
    The official ``height_scan`` *observation* survives the same inf because it is
    ``clip=(-1.0, 1.0)``; the reward had no such guard.

    :func:`foot_clearance_l2` below already sanitizes the same sensor, for the same reason.
    """
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        ray_hits_z = torch.nan_to_num(
            sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0
        )
        adjusted_target_height = target_height + torch.mean(ray_hits_z, dim=1)
    else:
        adjusted_target_height = target_height
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def foot_clearance_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg | None = None,
    desired_height: float = 0.12,
    sensor_cfg: SceneEntityCfg | None = None,  # kept for backward compat (unused)
) -> torch.Tensor:
    """Foot clearance penalty matching the original IsaacGym formulation.

    Original (legged_robot.py:_reward_foot_clearance, L1699-1711):
        feet_z_above_terrain = feet_pos[:, :, 2] - measured_heights_under_feet
        reward = sum((feet_z_above_terrain - desired_height)^2 * lateral_speed)

    The original samples per-foot terrain height from a precomputed height map.
    IsaacLab equivalent would need 4 dedicated RayCasters; we approximate
    `measured_heights_under_feet` with the mean of the existing height scanner
    rays under the base (exact on flat ground, approximate on slopes — same
    approximation used by `base_height_l2`). No contact mask: the original
    penalizes regardless of contact state.

    Args:
        asset_cfg: SceneEntityCfg targeting foot bodies.
        height_sensor_cfg: SceneEntityCfg for the base-mounted height scanner
            (e.g. the "height_scanner" sensor). If None we fall back to world Z,
            which only matches the original on perfectly flat ground.
        desired_height: target foot clearance (paper's `desired_foot_height`).
        sensor_cfg: deprecated (was the contact sensor for the in-air mask).
            Ignored — kept for backward-compatible call sites.
    """
    asset = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    feet_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    # Foot height ABOVE TERRAIN (not world Z). On uneven ground the terrain under
    # the feet differs from world Z, so we subtract the mean of the base-mounted
    # height scanner's ray hits as a flat-terrain approximation.
    if height_sensor_cfg is not None:
        height_sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]
        ray_hits_z = torch.nan_to_num(
            height_sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0
        )
        mean_terrain_z = torch.mean(ray_hits_z, dim=1)
        feet_z_above_terrain = feet_pos_w[:, :, 2] - mean_terrain_z[:, None]
    else:
        feet_z_above_terrain = feet_pos_w[:, :, 2]

    feet_lateral_speed = torch.sqrt(feet_vel_w[:, :, 0] ** 2 + feet_vel_w[:, :, 1] ** 2)
    return torch.sum(
        (feet_z_above_terrain - desired_height) ** 2 * feet_lateral_speed,
        dim=1,
    )


class action_smoothness_l2(ManagerTermBase):
    """Penalize jerk in actions: sum((a_t - 2*a_{t-1} + a_{t-2})^2).

    Second-order finite difference of the action sequence (discrete jerk).
    Matches the original DreamWaQ formulation in legged_robot.py.
    Maintains its own t-2 buffer since IsaacLab's action_manager only
    exposes t and t-1.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_prev_action = torch.zeros_like(env.action_manager.action)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.prev_prev_action.zero_()
        else:
            self.prev_prev_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        # ── TODO(reward-smoothness) ─ level L2 ─────────────────────────────
        # 액션의 저크(2차 차분)를 계산하고 t-2 버퍼를 갱신한다
        #   hint: 저크 = a_t - 2*a_{t-1} + a_{t-2} 다. 2차 유한차분이다
        #   hint: IsaacLab 의 action_manager 는 t 와 t-1 만 준다. t-2 는 self.prev_prev_action 으로 직접 들고 있어야 한다
        #   hint: 이번 호출의 prev_action 이 다음 호출의 t-2 가 된다. 반환 전에 갱신한다
        #   hint: 그냥 대입하면 같은 텐서를 참조하게 된다. .clone() 이 필요한 이유를 생각해 본다
        #   hint: 보상은 env 마다 스칼라 하나여야 한다 — 12개 관절의 저크를 제곱해 dim=1 로 합한다
        # 통과 기준은 이 실습 폴더의 README.md 를 본다.
        raise NotImplementedError("TODO(reward-smoothness)")
        # ─────────────────────────────────────────────────────────────────────
