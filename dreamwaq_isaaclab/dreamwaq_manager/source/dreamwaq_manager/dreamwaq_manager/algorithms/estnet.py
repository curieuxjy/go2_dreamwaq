# MIT License
# Copyright (c) 2024 Jungyeon Lee (curieuxjy)
# https://github.com/curieuxjy
#
# Estimator Network (EstNet) implementation for comparison
# Ported from dreamwaq/rsl_rl/rsl_rl/vae/estnet.py for IsaacLab integration.

import torch
import torch.nn as nn
import torch.optim as optim


class EstNetRolloutStorage:
    class Transition:
        def __init__(self):
            self.observation_histories = None
            self.true_velocities = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_history_shape, true_vel_shape, device="cpu"):
        self.device = device
        self.observation_histories = torch.zeros(num_transitions_per_env, num_envs, *obs_history_shape, device=device)
        self.true_velocities = torch.zeros(num_transitions_per_env, num_envs, *true_vel_shape, device=device)
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

    def add_transitions_before_action(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observation_histories[self.step].copy_(transition.observation_histories)
        self.true_velocities[self.step].copy_(transition.true_velocities)
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observation_histories = self.observation_histories.flatten(0, 1)
        true_velocities = self.true_velocities.flatten(0, 1)

        for epoch in range(num_epochs):
            for batch_idx in range(num_mini_batches):
                start = batch_idx * mini_batch_size
                end = (batch_idx + 1) * mini_batch_size
                batch_indices = indices[start:end]
                yield observation_histories[batch_indices], true_velocities[batch_indices]


class EstNet(nn.Module):
    """Estimator Network for velocity estimation (comparison baseline).

    Simple MLP: obs_history(225) → est_vel(3)
    """

    def __init__(
        self,
        num_learning_epochs=1,
        num_mini_batches=1,
        input_dim=225,
        hidden_dim1=128,
        hidden_dim2=64,
        latent_dim1=3,
        learning_rate=0.001,
        min_lr=0.001,
        patience=100,
        factor=0.8,
        device="cpu",
    ):
        super().__init__()
        self.device = device

        self.estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1), nn.ELU(),
            nn.Linear(hidden_dim1, hidden_dim2), nn.ELU(),
            nn.Linear(hidden_dim2, latent_dim1),
        )

        print(f"{'EstNet Structure':=^60}")
        print(f"Estimator MLP: {self.estimator}")

        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.current_epoch = 0
        self.storage = None
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor

        self.transition = EstNetRolloutStorage.Transition()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.factor, patience=self.patience, min_lr=self.min_lr,
        )

    def init_storage(self, num_envs, num_transitions_per_env, obs_history_shape, true_vel_shape):
        self.storage = EstNetRolloutStorage(
            num_envs, num_transitions_per_env, obs_history_shape, true_vel_shape, self.device,
        )

    def train_mode(self):
        self.estimator.train()

    def test_mode(self):
        self.estimator.eval()

    def forward(self, obs_history):
        return self.estimator(obs_history)

    def before_action(self, obs_history, true_vel):
        est_vel = self.forward(obs_history)
        self.transition.observation_histories = obs_history
        self.transition.true_velocities = true_vel
        self.storage.add_transitions_before_action(self.transition)
        self.transition.clear()
        return est_vel

    def update(self):
        mean_vel_loss = 0
        vel_loss = None

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_history_batch, true_vel_batch in generator:
            est_vel_batch = self.forward(obs_history_batch)
            mse_loss = nn.MSELoss()
            vel_loss = mse_loss(est_vel_batch, true_vel_batch)
            self.optimizer.zero_grad()
            vel_loss.backward()
            self.optimizer.step()
            mean_vel_loss += vel_loss.item()

        if vel_loss is not None:
            self.scheduler.step(vel_loss)
        self.current_epoch += 1
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_vel_loss /= num_updates
        self.storage.clear()
        return mean_vel_loss
