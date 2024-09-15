import torch
import torch.nn as nn
import torch.optim as optim

class CenetRolloutStorage:
    class Transition:
        def __init__(self):
            self.observation_histories = None
            self.true_velocities = None
            self.true_next_observations = None

        def clear(self):
            self.__init__()

    def __init__(self,
                 num_envs,
                 num_transitions_per_env,
                 obs_history_shape,
                 true_vel_shape,
                 true_onext_shape,
                 device='cpu'):

        self.device = device

        self.obs_history_shape = obs_history_shape
        self.true_vel_shape = true_vel_shape
        self.true_onext_shape = true_onext_shape

        self.observation_histories = torch.zeros(num_transitions_per_env, num_envs, *obs_history_shape, device=self.device, requires_grad=False)
        self.true_velocities = torch.zeros(num_transitions_per_env, num_envs, *true_vel_shape, device=self.device, requires_grad=False)
        self.true_next_observations = torch.zeros(num_transitions_per_env, num_envs, *true_onext_shape, device=self.device, requires_grad=False)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0


    def add_transitions_before_action(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observation_histories[self.step].copy_(transition.observation_histories)
        self.true_velocities[self.step].copy_(transition.true_velocities)

    def add_transitions_after_action(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.true_next_observations[self.step].copy_(transition.true_next_observations)

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
        true_next_observations = self.true_next_observations.flatten(0, 1)

        for epoch in range(num_epochs):
            for batch_idx in range(num_mini_batches):
                start = batch_idx * mini_batch_size
                end = (batch_idx + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_history_batch = observation_histories[batch_idx]
                true_vel_batch = true_velocities[batch_idx]
                true_onext_batch = true_next_observations[batch_idx]

                # Yield a mini-batch of data
                yield obs_history_batch, true_vel_batch, true_onext_batch


class CENet(nn.Module):
    def __init__(self,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 input_dim=225,  # 42 x 5
                 hidden_dim1=128,
                 hidden_dim2=64,
                 hidden_dim3=48,
                 latent_dim1=35,  # 3 + 16 x 2
                 latent_dim2=19,
                 output_dim=45,
                 beta=1,
                 beta_limit=4,
                 learning_rate=0.001,
                 min_lr=0.001,
                 patience = 100,
                 factor = 0.8,
                 device='cpu'):

        super().__init__()

        self.device = device

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim1),
            nn.ELU(),
            nn.Linear(in_features=hidden_dim1, out_features=hidden_dim2),
            nn.ELU(),
            nn.Linear(in_features=hidden_dim2, out_features=latent_dim1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim2, out_features=hidden_dim2),
            nn.ELU(),
            nn.Linear(in_features=hidden_dim2, out_features=hidden_dim1),
            nn.ELU(),
            nn.Linear(in_features=hidden_dim1, out_features=hidden_dim3),
            nn.ELU(),
            nn.Linear(in_features=hidden_dim3, out_features=output_dim)
        )

        print('{::^60}'.format(' CENet Structure '))
        print(f"Encoder MLP: {self.encoder}")
        print(f"Decoder MLP: {self.decoder}")

        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.beta = beta
        self.beta_limit = beta_limit
        self.current_epoch = 0
        self.storage = None  # initialized later
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor

        # Rollout Storage
        self.transition = CenetRolloutStorage.Transition()

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
                     true_onext_shape
                     ):

        self.storage = CenetRolloutStorage(num_envs, num_transitions_per_env, obs_history_shape,
                                           true_vel_shape, true_onext_shape, self.device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).requires_grad_(True)
        eps = torch.randn_like(std)
        # print("s", std.requires_grad)
        # print("e", eps.requires_grad)
        return mu + eps * std

    def train_mode(self):
        self.encoder.train()
        self.decoder.train()

    def test_mode(self):
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, obs_history):
        # encoder process
        h = self.encoder(obs_history)
        est_vel, context_vec_params = h.split([3, h.size(-1) - 3], dim=-1)  # 3 // 32

        last_dim = context_vec_params.size(-1)
        if last_dim % 2 == 0:
            mu, logvar = context_vec_params.split(last_dim // 2, dim=-1)
        else:
            raise AssertionError("Not even number for context vector parameters")

        mu = mu.requires_grad_(True)
        logvar = logvar.requires_grad_(True)
        context_vec = self.reparameterize(mu, logvar).requires_grad_(True)  # z: 16 dim
        latent = torch.cat([est_vel, context_vec], dim=-1)  # 19 dim

        # print("m", mu.requires_grad)
        # print("l", logvar.requires_grad)
        # print("z", context_vec.requires_grad)

        # decoder process
        return self.decoder(latent), est_vel, mu, logvar, context_vec  # est_onext


    def before_action(self, obs_history, true_vel):

        est_next_obs, est_vel, mu, logvar, context_vec = self.forward(obs_history)

        self.transition.observation_histories = obs_history
        self.transition.true_velocities = true_vel

        self.storage.add_transitions_before_action(self.transition)

        # return est_vel, context_vec
        return est_next_obs, est_vel, mu, logvar, context_vec

    def after_action(self, next_obs):
        self.transition.true_next_observations = next_obs
        # Record
        self.storage.add_transitions_after_action(self.transition)
        self.transition.clear()


    def update(self):

        mean_total_loss = 0
        mean_vel_loss = 0
        mean_recon_loss = 0
        mean_kl_loss = 0

        # Use the mini batch generator to iterate over the data
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # for est_vel_batch, true_vel_batch, est_onext_batch, true_onext_batch, mu_batch, logvar_batch in generator:
        for obs_history_batch, true_vel_batch, true_onext_batch in generator:

            # model의 forward 과정이 여기에 있어야
            est_onext_batch, est_vel_batch, mu_batch, logvar_batch, context_vec_batch = self.forward(obs_history_batch)

            # loss calculation
            mse_loss = nn.MSELoss()
            vel_loss = mse_loss(est_vel_batch, true_vel_batch)
            recon_loss = mse_loss(est_onext_batch, true_onext_batch)

            klds = -0.5 * (1 + logvar_batch - mu_batch.pow(2) - logvar_batch.exp())
            kl_loss = klds.sum(1).mean(0, True) * self.beta
            # kl_loss = (-0.5 * torch.mean(1 + logvar_batch - mu_batch.pow(2) - logvar_batch.exp())) * self.beta

            total_loss = vel_loss + recon_loss + kl_loss

            self.optimizer.zero_grad()

            # Gradient step
            total_loss.backward()

            self.optimizer.step()

            mean_total_loss += total_loss.item()
            mean_vel_loss += vel_loss.item()
            mean_recon_loss += recon_loss.item()
            mean_kl_loss += kl_loss.item()

        self.scheduler.step(total_loss)

        self.current_epoch += 1

        # ----------------- GRAD CHECK ------------------
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.sum())
        #     else:
        #         print(name, param.grad)

        num_updates = self.num_learning_epochs * self.num_mini_batches

        mean_total_loss /= num_updates
        mean_vel_loss /= num_updates
        mean_recon_loss /= num_updates
        mean_kl_loss /= num_updates

        self.storage.clear()

        # Update beta
        self.beta = min(self.beta * 1.01, self.beta_limit)
        return mean_total_loss, mean_vel_loss, mean_recon_loss, mean_kl_loss

# --- for testing the code ---
def simulate_data_and_train(model):
    # Simulate data
    num_envs = 4000
    num_steps = 100

    model.train_mode()
    # Rollout
    with torch.inference_mode():

        for _ in range(num_steps):
            # initial state O_{0 or t}
            obs = torch.randn(num_envs, 45).to(device)  # Random observation
            obs_his = torch.zeros(num_envs, 225).to(device)  # Random observation history
            obs_his[:, -45:] = obs
            true_vel = torch.randn(num_envs, 3).to(device)

            # CENet process w/ O_{t}
            est_next_obs, est_vel, mu, logvar, context_vec = model.before_action(obs_his, true_vel)

            # ACTION

            # O_{t+1}
            obs = torch.randn(num_envs, 45).to(device)  # Random observation
            model.after_action(obs)

    # Update model
    mean_total_loss, mean_vel_loss, mean_recon_loss, mean_kl_loss = model.update()

    print("Mean TOTAL loss: ", mean_total_loss)
    print("Mean VEL loss: ", mean_vel_loss)
    print("Mean RECON loss: ", mean_recon_loss)
    print("Mean KL loss: ", mean_kl_loss)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Initialize CENet
    model = CENet(device=device).to(device)

    # Save the initial weights
    initial_weights = [w.clone() for w in model.parameters()]

    # Initialize storage
    model.init_storage(
        num_envs=4000,
        num_transitions_per_env=100,
        obs_history_shape=(225,),
        true_vel_shape=(3,),
        true_onext_shape=(45,),
    )

    for _ in range(5):  # 5 times repetition
        simulate_data_and_train(model)

        # After the update, check if the weights have changed
        final_weights = [w for w in model.parameters()]
        for i, (initial, final) in enumerate(zip(initial_weights, final_weights)):
            if torch.all(torch.eq(initial, final)):
                print(f"Weights {i} did not change.")
            else:
                print(f"Weights {i} have changed.")
        initial_weights = [w.clone() for w in model.parameters()]

        # model.test_mode()
        print("Validation finished")

    print("Training finished")
