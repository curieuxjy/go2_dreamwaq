"""최소 PPO 구현 — Isaac Sim 없이 도는 독립 스크립트.  [Stage 3 solution]

rsl_rl 의 PPO 를 읽기 전에, 알고리즘의 뼈대를 직접 써 본다. 여기서 만든 조각들이
`rsl_rl/algorithms/ppo.py` 의 어디에 대응하는지는 `task03_rsl_rl_map/` 에서 대조한다.

    python ppo.py            # toy 환경에서 학습시켜 보상이 오르는지 확인
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# toy_env.py 는 stage3_ppo/ 에 있다. 이 파일은 거기(solution) 또는 taskNN/(starter)
# 어느 쪽에서도 실행될 수 있으므로 두 곳을 다 찾는다.
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent):
    if (_cand / "toy_env.py").exists():
        sys.path.insert(0, str(_cand))
        break

from toy_env import ACT_DIM, OBS_DIM, VelocityTrackingEnv, standing_still_reward  # noqa: E402


class ActorCritic(nn.Module):
    """DreamWaQ 와 같은 구조: actor 와 critic 이 파라미터를 공유하지 않는다.

    실제 프로젝트는 512-256-128 이지만 toy 문제라 64-64 로 줄였다.
    액션 분포는 상태에 무관한 학습 가능한 log_std 를 갖는 가우시안 —
    이것도 rsl_rl 및 DreamWaQ 와 같다.
    """

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )
        # init_noise_std = 1.0 (rsl_rl 기본값이자 DreamWaQ 값)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        dist = self.distribution(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), self.value(obs)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation.

    Args:
        rewards: (T, N)  각 스텝의 보상
        values:  (T, N)  critic 이 본 상태가치 V(s_t)
        dones:   (T, N)  그 스텝에서 에피소드가 끝났으면 1.0
        last_value: (N,) 마지막 관측의 V(s_T) — 잘린 롤아웃을 부트스트랩한다
        gamma: 할인율
        lam: GAE 람다. 0 이면 1-step TD, 1 이면 몬테카를로

    Returns:
        (returns, advantages) 둘 다 (T, N)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)

    last_advantage = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_advantage = delta + gamma * lam * not_done * last_advantage
        advantages[t] = last_advantage
    returns = advantages + values

    return returns, advantages


def ppo_losses(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    value: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_param: float = 0.2,
    value_loss_coef: float = 1.0,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO 의 세 손실 항과 총합.

    Args:
        log_prob:     (B,) 현재 정책이 그 액션에 준 log 확률
        old_log_prob: (B,) 롤아웃을 모을 때의 정책이 준 log 확률 (상수)
        advantages:   (B,) 정규화된 어드밴티지
        value:        (B,) 현재 critic 의 예측
        returns:      (B,) GAE 로 만든 회귀 타깃
        entropy:      (B,) 현재 정책의 엔트로피

    Returns:
        (total, surrogate_loss, value_loss, entropy_loss)
    """
    # ── TODO(ppo-losses) ─ level L2 ─────────────────────────────
    # PPO 의 clipped surrogate / value / entropy 손실과 총합을 완성한다
    #   hint: ratio = exp(log_prob - old_log_prob). 같은 정책이면 1 이 된다
    #   hint: surrogate 는 -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) 의 평균이다 (최대화가 아니라 최소화)
    #   hint: clip 은 ratio 에만 걸고 advantage 에는 걸지 않는다
    #   hint: value_loss 는 (value - returns)^2 의 평균 (MSE)
    #   hint: entropy 는 크게 만들고 싶으므로 손실에는 음수로 들어간다
    #   hint: total = surrogate + value_loss_coef * value_loss + entropy_coef * entropy_loss
    # 통과 기준은 이 실습 폴더의 README.md 를 본다.
    raise NotImplementedError("TODO(ppo-losses)")
    # ─────────────────────────────────────────────────────────────────────

    return total, surrogate, value_loss, entropy_loss


def train(
    iterations: int = 200,
    num_envs: int = 256,
    steps_per_env: int = 32,
    epochs: int = 5,
    minibatches: int = 4,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[ActorCritic, list[float]]:
    """PPO 학습 루프. rsl_rl 의 OnPolicyRunner.learn() 과 같은 골격이다."""
    env = VelocityTrackingEnv(num_envs=num_envs, device=device)
    model = ActorCritic().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    obs = env.reset()
    history: list[float] = []

    for it in range(iterations):
        # --- 1. 롤아웃 수집 (on-policy) ---
        buf_obs, buf_act, buf_logp, buf_rew, buf_done, buf_val = [], [], [], [], [], []
        for _ in range(steps_per_env):
            action, log_prob, value = model.act(obs)
            next_obs, reward, done = env.step(action)
            buf_obs.append(obs); buf_act.append(action); buf_logp.append(log_prob)
            buf_rew.append(reward); buf_done.append(done.float()); buf_val.append(value)
            obs = next_obs

        with torch.no_grad():
            last_value = model.value(obs)

        b_obs = torch.stack(buf_obs); b_act = torch.stack(buf_act)
        b_logp = torch.stack(buf_logp); b_rew = torch.stack(buf_rew)
        b_done = torch.stack(buf_done); b_val = torch.stack(buf_val)

        # --- 2. 어드밴티지 ---
        returns, advantages = compute_gae(b_rew, b_val, b_done, last_value)

        # 평탄화 후 어드밴티지 정규화 (rsl_rl 과 동일)
        flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731
        f_obs, f_act = flat(b_obs), flat(b_act)
        f_logp, f_ret = flat(b_logp), flat(returns)
        f_adv = flat(advantages)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        # --- 3. 여러 epoch 동안 미니배치 업데이트 ---
        batch = f_obs.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(batch, device=device)
            for mb in perm.chunk(minibatches):
                dist = model.distribution(f_obs[mb])
                log_prob = dist.log_prob(f_act[mb]).sum(-1)
                total, *_ = ppo_losses(
                    log_prob, f_logp[mb], f_adv[mb],
                    model.value(f_obs[mb]), f_ret[mb], dist.entropy().sum(-1),
                )
                optim.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

        mean_reward = b_rew.mean().item()
        history.append(mean_reward)
        if verbose and (it % 20 == 0 or it == iterations - 1):
            print(f"  iter {it:4d} | 평균 보상 {mean_reward:.4f} | std {model.log_std.exp().mean():.3f}")

    return model, history


if __name__ == "__main__":
    baseline = standing_still_reward()
    print(f"정지 시 보상(하한선) : {baseline:.4f}\n")
    _, history = train()
    final = sum(history[-10:]) / 10
    print(f"\n최종 평균 보상 : {final:.4f}  (하한선 {baseline:.4f}, 완벽 추종 1.0)")
    print("학습 성공" if final > baseline + 0.2 else "학습 실패 — 하한선을 못 넘었다")
