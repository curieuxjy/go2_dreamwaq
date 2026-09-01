"""Custom observation terms for DreamWaQ.

`last_disturb_force` exposes the most recent push impulse (recorded by
`push_robot_with_disturb`) to the privileged critic obs — matching the
original IsaacGym `self.disturb_force` slot in compute_observations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg


def last_disturb_force(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the most recent push impulse, shape (num_envs, 3).

    Initialized to zeros if no push has happened yet.
    """
    if not hasattr(env, "disturb_force"):
        env.disturb_force = torch.zeros(env.num_envs, 3, device=env.device)
    return env.disturb_force


class delayed(ManagerTermBase):
    """Wrap another observation term with the original's first-order system delay.

    DreamWaQ paper Table II lists a "System delay" of [0.0, 15.0] ms, and the original
    `go2_config.py` sets `domain_rand.system_delay = True`. legged_gym implements it as a
    per-environment linear blend between the previous and current measurement
    (`legged_robot.py:378-388`)::

        obs = delay_time * last_obs + (1 - delay_time) * obs
        delay_time = U[0, 1) * 0.25      # resampled on every episode reset

    Isaac Lab has no delay mechanism for the manager stack, so this term wraps an inner
    observation function and applies the same blend. `dreamwaq_direct` does it inline in
    `_get_observations`; this keeps the two stacks aligned.

    Only the *policy* group is delayed. The critic sees clean, privileged observations.

    Usage::

        base_ang_vel = ObsTerm(func=mdp.delayed, params={"func": mdp.base_ang_vel})
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._prev: torch.Tensor | None = None
        self._delay = torch.zeros(env.num_envs, 1, device=env.device)
        self._max_delay = float(cfg.params.get("max_delay", 0.25))
        self._resample(slice(None), env.num_envs)

    def _resample(self, env_ids, count: int) -> None:
        self._delay[env_ids] = torch.rand(count, 1, device=self._env.device) * self._max_delay

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._resample(slice(None), self._env.num_envs)
            if self._prev is not None:
                self._prev.zero_()
        else:
            self._resample(env_ids, len(env_ids))
            if self._prev is not None:
                self._prev[env_ids] = 0.0

    def __call__(self, env: ManagerBasedEnv, func, max_delay: float = 0.25, **kwargs) -> torch.Tensor:
        current = func(env, **kwargs)
        if self._prev is None:
            self._prev = torch.zeros_like(current)
        delayed_obs = self._delay * self._prev + (1.0 - self._delay) * current
        self._prev = current.clone()
        return delayed_obs
