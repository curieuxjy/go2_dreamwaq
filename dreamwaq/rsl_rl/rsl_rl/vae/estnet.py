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

    def __init__(self,
                 num_envs,
                 num_transitions_per_env,
                 obs_history_shape,
                 true_vel_shape,
                 device='cpu'):

        self.device = device

        self.obs_history_shape = obs_history_shape
        self.true_vel_shape = true_vel_shape
        self.observation_histories = torch.zeros(num_transitions_per_env, num_envs, *obs_history_shape, device=self.device, requires_grad=False)
        self.true_velocities = torch.zeros(num_transitions_per_env, num_envs, *true_vel_shape, device=self.device, requires_grad=False)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

    def add_transitions_before_action(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observation_histories[self.step].copy_(transition.observation_histories)
        self.true_velocities[self.step].copy_(transition.true_velocities)

        # Increment the step
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
                batch_idx = indices[start:end]

                obs_history_batch = observation_histories[batch_idx]
                true_vel_batch = true_velocities[batch_idx]

                # Yield a mini-batch of data
                yield obs_history_batch, true_vel_batch


class EstNet(nn.Module):
    def __init__(self,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 input_dim=225,  # 42 x 5
                 hidden_dim1=128,
                 hidden_dim2=64,
                 latent_dim1=3,
                 learning_rate=0.001,
                 min_lr=0.001,
                 patience=100,
                 factor=0.8,
                 device='cpu'):

        super().__init__()

        self.device = device

        # Estimator
        self.estimator = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim1),
            nn.ELU(),
            nn.Linear(in_features=hidden_dim1, out_features=hidden_dim2),
            nn.ELU(),
            nn.Linear(in_features=hidden_dim2, out_features=latent_dim1)
        )

        print('{::^60}'.format(' EstNet Structure '))
        print(f"Estimator MLP: {self.estimator}")

        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.current_epoch = 0
        self.storage = None  # initialized later
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor

        # Rollout Storage
        self.transition = EstNetRolloutStorage.Transition()

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                              mode="min",
                                                              factor=self.factor,
                                                              patience=self.patience,
                                                              min_lr=self.min_lr)

    def init_storage(self,
                     num_envs,
                     num_transitions_per_env,
                     obs_history_shape,
                     true_vel_shape,
                     ):
        self.storage = EstNetRolloutStorage(num_envs, num_transitions_per_env, obs_history_shape, true_vel_shape, self.device)


    def train_mode(self):
        self.estimator.train()

    def test_mode(self):
        self.estimator.eval()

    def forward(self, obs_history):
        # estimator process
        est_vel = self.estimator(obs_history)
        return est_vel

    def before_action(self, obs_history, true_vel):
        est_vel = self.forward(obs_history)
        self.transition.observation_histories = obs_history
        self.transition.true_velocities = true_vel
        self.storage.add_transitions_before_action(self.transition)
        self.transition.clear()
        return est_vel
  
    def update(self):
        mean_vel_loss = 0

        # Use the mini batch generator to iterate over the data
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # for est_vel_batch, true_vel_batch, est_onext_batch, true_onext_batch, mu_batch, logvar_batch in generator:
        for obs_history_batch, true_vel_batch in generator:
            est_vel_batch = self.forward(obs_history_batch)
            # loss calculation
            mse_loss = nn.MSELoss()
            vel_loss = mse_loss(est_vel_batch, true_vel_batch)
            self.optimizer.zero_grad()
            # Gradient step
            vel_loss.backward()
            self.optimizer.step()
            mean_vel_loss += vel_loss.item()

        self.scheduler.step(vel_loss)
        self.current_epoch += 1
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_vel_loss /= num_updates
        self.storage.clear()
        return mean_vel_loss