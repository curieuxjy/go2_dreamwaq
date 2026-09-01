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
    # ── TODO(reward-foot-clearance) ─ level L2 ─────────────────────────────
    # 발 높이를 '지형 위 높이'로 바꾸고, 목표 클리어런스와의 차이를 측면 속도로 가중해 합한다
    #   hint: feet_pos_w[:, :, 2] 는 월드 z 다. 지형이 울퉁불퉁하면 그대로 쓸 수 없다
    #   hint: height_sensor 의 ray_hits_w[..., 2] 평균을 지형 높이로 근사한다 (task05 의 height scan 과 같은 발상)
    #   hint: ray hit 에는 NaN/inf 가 섞일 수 있다. torch.nan_to_num 으로 0 으로 만든다
    #   hint: height_sensor_cfg 가 None 이면 지형 보정 없이 월드 z 를 그대로 쓴다 (평지 전용 폴백)
    #   hint: 측면 속도 = sqrt(vx^2 + vy^2). z 속도는 넣지 않는다
    #   hint: 왜 속도로 가중하는가: 발을 들어 '옮기는 중'일 때만 클리어런스를 요구하고, 딛고 서 있는 발은 벌하지 않기 위해서다
    #   hint: 발 4개에 대해 dim=1 로 합해 env 당 스칼라를 만든다
    # 통과 기준은 이 실습 폴더의 README.md 를 본다.
    raise NotImplementedError("TODO(reward-foot-clearance)")
    # ─────────────────────────────────────────────────────────────────────


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
        action = env.action_manager.action
        prev_action = env.action_manager.prev_action
        jerk = action - 2.0 * prev_action + self.prev_prev_action
        # cache prev_action as the t-2 for the next call
        self.prev_prev_action = prev_action.clone()
        return torch.sum(torch.square(jerk), dim=1)
