"""Rerun logger for DreamWaQ IsaacLab evaluation and training.

Logs robot state, commands, actions, and CENet outputs to rerun for real-time visualization.

Usage:
    from dreamwaq_manager.utils.rerun_logger import RerunLogger
    logger = RerunLogger(env_index=0)
    logger.log_step(step, obs_policy, obs_critic, actions)
"""

import torch

try:
    import rerun as rr
except ImportError:
    rr = None

# Go2 joint names (4 legs x 3 joints)
JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

# Policy observation layout (45 dims)
# ang_vel(3) + gravity(3) + commands(3) + joint_pos(12) + joint_vel(12) + actions(12)
POLICY_SLICES = {
    "ang_vel": (0, 3),
    "gravity": (3, 6),
    "commands": (6, 9),
    "joint_pos": (9, 21),
    "joint_vel": (21, 33),
    "last_action": (33, 45),
}

# Critic observation layout (238 dims)
# lin_vel(3) + ang_vel(3) + gravity(3) + commands(3) + joint_pos(12) + joint_vel(12) + actions(12) + height_scan(187) + disturb_force(3)
CRITIC_SLICES = {
    "lin_vel": (0, 3),
    "ang_vel": (3, 6),
    "gravity": (6, 9),
    "commands": (9, 12),
    "joint_pos": (12, 24),
    "joint_vel": (24, 36),
    "last_action": (36, 48),
    "height_scan": (48, 235),
    "disturb_force": (235, 238),
}


class RerunLogger:
    """Logs DreamWaQ evaluation/training data to rerun."""

    def __init__(self, env_index: int = 0, app_name: str = "dreamwaq_play"):
        if rr is None:
            raise ImportError("rerun-sdk not installed. Install with: pip install rerun-sdk")
        rr.init(app_name, spawn=True)
        self.env_index = env_index
        self.dt = 0.02  # 50Hz (decimation=4, sim_dt=0.005)

    def _scalar(self, path: str, value: float):
        rr.log(path, rr.Scalar(value))

    def log_step(
        self,
        step: int,
        obs_policy: torch.Tensor,
        obs_critic: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ):
        """Log one environment step.

        Args:
            step: Current simulation step.
            obs_policy: Policy observations [num_envs, 45].
            obs_critic: Critic observations [num_envs, 238] (optional).
            actions: Actions taken [num_envs, 12] (optional).
        """
        rr.set_time_seconds("sim_time", step * self.dt)
        rr.set_time_sequence("step", step)

        idx = self.env_index
        p = obs_policy[idx].cpu()

        # -- Commands --
        s, e = POLICY_SLICES["commands"]
        self._scalar("commands/vx", p[s].item())
        self._scalar("commands/vy", p[s + 1].item())
        self._scalar("commands/yaw_rate", p[s + 2].item())

        # -- Angular velocity --
        s, e = POLICY_SLICES["ang_vel"]
        self._scalar("robot/ang_vel/x", p[s].item())
        self._scalar("robot/ang_vel/y", p[s + 1].item())
        self._scalar("robot/ang_vel/z", p[s + 2].item())

        # -- Projected gravity (orientation proxy) --
        s, e = POLICY_SLICES["gravity"]
        self._scalar("robot/gravity/x", p[s].item())
        self._scalar("robot/gravity/y", p[s + 1].item())
        self._scalar("robot/gravity/z", p[s + 2].item())

        # -- Joint positions --
        s, e = POLICY_SLICES["joint_pos"]
        for i, name in enumerate(JOINT_NAMES):
            self._scalar(f"joints/pos/{name}", p[s + i].item())

        # -- Joint velocities --
        s, e = POLICY_SLICES["joint_vel"]
        for i, name in enumerate(JOINT_NAMES):
            self._scalar(f"joints/vel/{name}", p[s + i].item())

        # -- Actions --
        if actions is not None:
            a = actions[idx].cpu()
            for i, name in enumerate(JOINT_NAMES):
                self._scalar(f"actions/{name}", a[i].item())

        # -- Critic data (true velocity, disturbance) --
        if obs_critic is not None:
            c = obs_critic[idx].cpu()
            cs, ce = CRITIC_SLICES["lin_vel"]
            self._scalar("velocity/true/vx", c[cs].item())
            self._scalar("velocity/true/vy", c[cs + 1].item())
            self._scalar("velocity/true/vz", c[cs + 2].item())

            ds, de = CRITIC_SLICES["disturb_force"]
            if obs_critic.shape[-1] >= de:
                self._scalar("robot/disturb_force/x", c[ds].item())
                self._scalar("robot/disturb_force/y", c[ds + 1].item())
                self._scalar("robot/disturb_force/z", c[ds + 2].item())

    def log_cenet(
        self,
        step: int,
        est_vel: torch.Tensor,
        true_vel: torch.Tensor,
        context_vec: torch.Tensor | None = None,
        boot_prob: float | None = None,
    ):
        """Log CENet outputs for Waq evaluation.

        Args:
            step: Current simulation step.
            est_vel: Estimated velocity [num_envs, 3].
            true_vel: True velocity [num_envs, 3].
            context_vec: Context vector [num_envs, 16] (optional).
            boot_prob: Current AdaBoot probability (optional).
        """
        rr.set_time_seconds("sim_time", step * self.dt)
        rr.set_time_sequence("step", step)

        idx = self.env_index
        ev = est_vel[idx].cpu()
        tv = true_vel[idx].cpu()

        # Estimated vs true velocity comparison
        self._scalar("velocity/estimated/vx", ev[0].item())
        self._scalar("velocity/estimated/vy", ev[1].item())
        self._scalar("velocity/estimated/vz", ev[2].item())

        # Velocity estimation error
        self._scalar("velocity/error/vx", abs(ev[0].item() - tv[0].item()))
        self._scalar("velocity/error/vy", abs(ev[1].item() - tv[1].item()))
        self._scalar("velocity/error/vz", abs(ev[2].item() - tv[2].item()))

        # Context vector
        if context_vec is not None:
            ctx = context_vec[idx].cpu()
            for i in range(ctx.shape[0]):
                self._scalar(f"cenet/context/{i:02d}", ctx[i].item())

        # AdaBoot probability
        if boot_prob is not None:
            self._scalar("cenet/boot_prob", boot_prob)

    def set_env_index(self, env_index: int):
        """Switch which environment to log (e.g., after agent tracking switch)."""
        self.env_index = env_index
