"""Stage 3 공용 toy 환경 — 1D 질점의 명령 속도 추종.  [제공 코드 · 읽기만]

Isaac Sim 없이 1초 만에 도는 벡터화 환경이다. 로코모션의 본질만 남겼다.

    상태   v        현재 속도
    명령   v_cmd    추종해야 할 속도 (에피소드마다 U[-1, 1] 로 다시 뽑는다)
    액션   a        가속도 (정책 출력, [-1, 1] 로 클립)
    전이   v <- v + (a * action_scale - drag * v) * dt
    보상   exp{-4 (v_cmd - v)^2}          <- DreamWaQ 논문 Table I 과 같은 형태
    관측   [v, v_cmd, a_prev]             (3차원)

DreamWaQ 와의 대응:

    | 여기            | DreamWaQ                          |
    |-----------------|-----------------------------------|
    | v               | base_lin_vel                      |
    | v_cmd           | velocity_commands                 |
    | a               | 관절 목표 오프셋 (action * 0.25)  |
    | exp{-4 e^2}     | track_lin_vel_xy_exp (std=sqrt(0.25)) |
    | drag            | 지면 마찰 / 중력                  |

PPO 는 이 환경에서도 로코모션에서와 똑같이 동작한다. 알고리즘을 먼저 여기서 이해하고,
Stage 4 에서 진짜 로봇으로 돌아간다.
"""
from __future__ import annotations

import torch

OBS_DIM = 3
ACT_DIM = 1


class VelocityTrackingEnv:
    """벡터화된 1D 속도 추종 환경. gym 의존성 없이 torch 만 쓴다."""

    def __init__(
        self,
        num_envs: int = 64,
        episode_length: int = 64,
        dt: float = 0.05,
        action_scale: float = 4.0,
        drag: float = 1.0,
        tracking_sigma_sq: float = 0.25,  # exp(-e^2 / 0.25) = exp(-4 e^2)
        device: str = "cpu",
        seed: int | None = 0,
    ):
        self.num_envs = num_envs
        self.episode_length = episode_length
        self.dt = dt
        self.action_scale = action_scale
        self.drag = drag
        self.tracking_sigma_sq = tracking_sigma_sq
        self.device = device
        if seed is not None:
            torch.manual_seed(seed)

        self.vel = torch.zeros(num_envs, device=device)
        self.cmd = torch.zeros(num_envs, device=device)
        self.prev_action = torch.zeros(num_envs, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.reset()

    # ------------------------------------------------------------------

    def _resample_cmd(self, mask: torch.Tensor) -> None:
        n = int(mask.sum())
        if n:
            self.cmd[mask] = torch.rand(n, device=self.device) * 2.0 - 1.0

    def _obs(self) -> torch.Tensor:
        return torch.stack([self.vel, self.cmd, self.prev_action], dim=1)

    def reset(self) -> torch.Tensor:
        self.vel.zero_()
        self.prev_action.zero_()
        self.step_count.zero_()
        self._resample_cmd(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self._obs()

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """한 스텝 진행. 반환 (obs, reward, done)."""
        a = action.squeeze(-1).clamp(-1.0, 1.0)
        self.vel = self.vel + (a * self.action_scale - self.drag * self.vel) * self.dt
        self.prev_action = a

        error = self.cmd - self.vel
        reward = torch.exp(-(error**2) / self.tracking_sigma_sq)

        self.step_count += 1
        done = self.step_count >= self.episode_length
        if done.any():
            self.vel[done] = 0.0
            self.prev_action[done] = 0.0
            self.step_count[done] = 0
            self._resample_cmd(done)

        return self._obs(), reward, done


def perfect_tracking_reward() -> float:
    """완벽히 추종했을 때의 스텝 보상 = 1.0. 학습 결과를 이 값과 비교한다."""
    return 1.0


def standing_still_reward(num_samples: int = 100_000, seed: int = 0) -> float:
    """아무것도 안 하고 v=0 으로 있을 때의 기대 보상.

    학습이 '진짜로' 됐는지 판정하는 하한선이다. 이 값을 못 넘으면 정책은
    아무것도 배우지 않은 것이다. (DreamWaQ 에서 '서 있기가 최적' 이 되는 실패와 같은 발상.)
    """
    g = torch.Generator().manual_seed(seed)
    cmd = torch.rand(num_samples, generator=g) * 2.0 - 1.0
    return torch.exp(-(cmd**2) / 0.25).mean().item()


if __name__ == "__main__":
    env = VelocityTrackingEnv(num_envs=8)
    obs = env.reset()
    print(f"관측 shape: {tuple(obs.shape)}  (v, v_cmd, a_prev)")
    print(f"완벽 추종 보상 : {perfect_tracking_reward():.4f}")
    print(f"정지 시 보상   : {standing_still_reward():.4f}   <- 학습이 이걸 못 넘으면 실패")

    total = torch.zeros(env.num_envs)
    for _ in range(env.episode_length):
        _, r, _ = env.step(torch.rand(env.num_envs, 1) * 2 - 1)
        total += r
    print(f"랜덤 정책 평균 : {(total / env.episode_length).mean():.4f}")
