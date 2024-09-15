# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np 

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv

from rsl_rl.vae import CENet, EstNet
from rsl_rl.utils import RunningMeanStd


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.oracle = True if self.cfg["run_name"] == "oracle" else False

        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        env_cfg = self.env.cfg.env  # A1RoughCfg.env

        # set critic observation dimension
        if self.env.num_privileged_obs is not None:
            num_critic_obs = env_cfg.num_observations + env_cfg.num_privileged_obs
        else:
            num_critic_obs = env_cfg.num_observations

        # set actor observation dimension
        if self.oracle:
            num_actor_obs = env_cfg.num_observations + env_cfg.num_privileged_obs
        else:
            num_actor_obs = env_cfg.num_observations

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(num_actor_obs,
                                                       num_critic_obs,
                                                       self.env.num_actions,
                                                       **self.policy_cfg).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs,
                              self.num_steps_per_env,
                              [num_actor_obs],
                              [num_critic_obs],
                              [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.rms_dict = {}
        if self.cfg["obs_rms"]:  # Initialize later
            self.obs_rms = None
        if self.cfg["privileged_obs_rms"]:  # Initialize later
            self.privileged_obs_rms = None

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # initial state
        obs = self.env.get_observations()
        if self.cfg["obs_rms"]:
            if self.obs_rms is None:
                self.obs_rms = RunningMeanStd(shape=obs.shape[1], device=self.device)
            self.obs_rms.update(obs.detach())
            obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)

        privileged_obs = self.env.get_privileged_observations()
        if self.oracle:
            if self.cfg["privileged_obs_rms"]:
                if self.privileged_obs_rms is None:
                    self.privileged_obs_rms = RunningMeanStd(shape=privileged_obs.shape[1], device=self.device)
                self.privileged_obs_rms.update(privileged_obs.detach())
                privileged_obs = (privileged_obs - self.privileged_obs_rms.mean) / torch.sqrt(self.privileged_obs_rms.var + 1e-8)

        # critic_obs = privileged_obs if privileged_obs is not None else obs
        critic_obs = torch.cat((obs, privileged_obs), dim=-1) if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rew_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    actor_obs = critic_obs if self.oracle else obs

                    actions = self.alg.act(actor_obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    obs = obs.to(self.device)

                    if self.cfg["obs_rms"]:
                        self.obs_rms.update(obs.detach())
                        obs = (obs - self.obs_rms.mean.to(self.device)) / torch.sqrt(
                            self.obs_rms.var.to(self.device) + 1e-8)

                    if self.oracle:
                        privileged_obs = privileged_obs.to(self.device)
                        if self.cfg["privileged_obs_rms"]:
                            self.privileged_obs_rms.update(privileged_obs.detach())
                            privileged_obs = (privileged_obs - self.privileged_obs_rms.mean.to(
                                self.device)) / torch.sqrt(self.privileged_obs_rms.var.to(self.device) + 1e-8)

                    critic_obs = torch.cat((obs, privileged_obs), dim=-1) if privileged_obs is not None else obs

                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        if 'reward_cv' in infos:
                            rew_infos.append(infos['reward_cv'])

                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:

                if self.cfg["obs_rms"]:
                    self.rms_dict["obs_rms"] = self.obs_rms
                if self.cfg["privileged_obs_rms"]:
                    self.rms_dict["privileged_obs_rms"] = self.privileged_obs_rms

                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
            rew_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = ''
        info_types = [('ep_infos', 'Episode'), ('rew_infos', 'CV')]

        for info_type, prefix in info_types:
            if locs[info_type]:
                for key in locs[info_type][0]:  # one agent
                    infotensor = torch.tensor([], device=self.device)
                    for info in locs[info_type]:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(info[key], torch.Tensor):
                            info[key] = torch.Tensor([info[key]])
                        if len(info[key].shape) == 0:
                            info[key] = info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, info[key].to(self.device)))

                    value = torch.mean(infotensor)
                    self.writer.add_scalar(f'{prefix}/{key}', value, locs['it'])

                    if prefix == 'Episode':
                        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            'rms': self.rms_dict
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        if self.cfg["obs_rms"] or self.cfg["privileged_obs_rms"]:
            self.rms_info = loaded_dict['rms']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_rms(self):
        return self.rms_info if (self.cfg["obs_rms"] or self.cfg["privileged_obs_rms"]) else None


class OnPolicyRunnerWAQ:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.vae_cfg = train_cfg["vae"]
        self.device = device
        self.env = env  # LeggedRobot
       
        env_cfg = self.env.cfg.env  # A1RoughWaqCfg.env
        num_critic_obs = env_cfg.num_observations + env_cfg.num_estvel + env_cfg.num_privileged_obs
        num_actor_obs = env_cfg.num_observations + env_cfg.num_estvel + env_cfg.num_context

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(num_actor_obs,
                                                       num_critic_obs,
                                                       self.env.num_actions,
                                                       **self.policy_cfg).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        vae_class = eval(self.cfg["vae_class_name"])  # CENet
        self.cenet: CENet = vae_class(device=self.device, **self.vae_cfg).to(self.device)

        # init storage and model
        self.alg.init_storage(self.env.num_envs,
                              self.num_steps_per_env,
                              [num_actor_obs],
                              [num_critic_obs],
                              [self.env.num_actions])

        self.cenet.init_storage(self.env.num_envs,
                                self.num_steps_per_env,
                                [env_cfg.len_obs_history * env_cfg.num_observations],
                                [env_cfg.num_estvel],
                                [env_cfg.num_observations],
                                )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.rms_dict = {}
        if self.cfg["obs_rms"]: # Initialize later
            self.obs_rms = None
        if self.cfg["privileged_obs_rms"]:  # Initialize later
            self.privileged_obs_rms = None
        if self.cfg["true_vel_rms"]:
            self.true_vel_rms = None

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        self.cenet.train_mode()

        ep_infos = []
        rew_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # O_{t}
            obs = self.env.get_observations().to(self.device)

            if self.cfg["obs_rms"]:
                if self.obs_rms is None:
                    self.obs_rms = RunningMeanStd(shape=obs.shape[1], device=self.device)
                self.obs_rms.update(obs.detach())
                obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)

            privileged_obs = self.env.get_privileged_observations().to(self.device)

            if self.cfg["privileged_obs_rms"]:
                if self.privileged_obs_rms is None:
                    self.privileged_obs_rms = RunningMeanStd(shape=privileged_obs.shape[1], device=self.device)
                self.privileged_obs_rms.update(privileged_obs.detach())
                privileged_obs = (privileged_obs - self.privileged_obs_rms.mean) / torch.sqrt(self.privileged_obs_rms.var + 1e-8)

            true_vel = self.env.get_true_vel().to(self.device)

            if self.cfg["true_vel_rms"]:
                if self.true_vel_rms is None:
                    self.true_vel_rms = RunningMeanStd(shape=true_vel.shape[1], device=self.device)
                self.true_vel_rms.update(true_vel.detach())
                true_vel = (true_vel - self.true_vel_rms.mean) / torch.sqrt(self.true_vel_rms.var + 1e-8)

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    # CENet process w/ O_{t}
                    obs_history = self.env.get_observation_history()
                    if self.cfg["obs_rms"]:
                        obs_history = (obs_history - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)
                    obs_history = obs_history.reshape(self.env.num_envs, -1).to(self.device)  # for cenet [num_envs, 225]

                    est_next_obs, est_vel, mu, logvar, context_vec = self.cenet.before_action(obs_history, true_vel)

                    # AdaBoot
                    if self.cfg["ada_boot"]:
                        vel_input = est_vel if self.env.extras["episode"]["boot_prob"].item() > np.random.random() else true_vel
                    else: # Not use AdaBoot
                        vel_input = est_vel

                    # prepare observations for actor critic
                    critic_obs = torch.cat((obs, vel_input, privileged_obs), dim=-1)
                    actor_obs = torch.cat((obs, vel_input, context_vec), dim=-1)
                    obs, critic_obs, actor_obs = obs.to(self.device), critic_obs.to(self.device), actor_obs.to(self.device)

                    # A_{t}
                    actions = self.alg.act(actor_obs, critic_obs)

                    # ============================== NEXT STEP ==============================
                    # O_{t+1}
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
                    true_vel = self.env.get_true_vel().to(self.device)

                    if self.cfg["obs_rms"]:
                        self.obs_rms.update(obs.detach())
                        obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)
                    if self.cfg["privileged_obs_rms"]:
                        self.privileged_obs_rms.update(privileged_obs.detach())
                        privileged_obs = (privileged_obs - self.privileged_obs_rms.mean) / torch.sqrt(self.privileged_obs_rms.var + 1e-8)
                    if self.cfg["true_vel_rms"]:
                        self.true_vel_rms.update(true_vel.detach())
                        true_vel = (true_vel - self.true_vel_rms.mean) / torch.sqrt(self.true_vel_rms.var + 1e-8)

                    self.cenet.after_action(obs)

                    rewards, dones = rewards.to(self.device), dones.to(self.device)

                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        if 'reward_cv' in infos:
                            rew_infos.append(infos['reward_cv'])

                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            # cenet update
            mean_total_loss, mean_vel_loss, mean_recon_loss, mean_kl_loss = self.cenet.update()
            # policy update
            mean_value_loss, mean_surrogate_loss = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())

            if it % self.save_interval == 0:

                if self.cfg["obs_rms"]:
                    self.rms_dict["obs_rms"] = self.obs_rms
                if self.cfg["privileged_obs_rms"]:
                    self.rms_dict["privileged_obs_rms"] = self.privileged_obs_rms
                if self.cfg["true_vel_rms"]:
                    self.rms_dict["true_vel_rms"] = self.true_vel_rms

                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), infos=infos)

            ep_infos.clear()
            rew_infos.clear()

        self.current_learning_iteration += num_learning_iterations

        if self.cfg["obs_rms"]:
            self.rms_dict["obs_rms"] = self.obs_rms
        if self.cfg["privileged_obs_rms"]:
            self.rms_dict["privileged_obs_rms"] = self.privileged_obs_rms
        if self.cfg["true_vel_rms"]:
            self.rms_dict["true_vel_rms"] = self.true_vel_rms

        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)), infos=infos)


    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = ''
        info_types = [('ep_infos', 'Episode'), ('rew_infos', 'CV')]

        for info_type, prefix in info_types:
            if locs[info_type]:
                for key in locs[info_type][0]:  # one agent
                    infotensor = torch.tensor([], device=self.device)
                    for info in locs[info_type]:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(info[key], torch.Tensor):
                            info[key] = torch.Tensor([info[key]])
                        if len(info[key].shape) == 0:
                            info[key] = info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, info[key].to(self.device)))

                    value = torch.mean(infotensor)
                    self.writer.add_scalar(f'{prefix}/{key}', value, locs['it'])

                    if prefix == 'Episode':
                        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        # if self.cfg["obs_rms"]:
        #     self.writer.add_scalar('RMS/obs_mean', self.obs_rms.mean, locs['it'])
        #     self.writer.add_scalar('RMS/obs_var', self.obs_rms.var, locs['it'])
        # if self.cfg["privileged_obs_rms"]:
        #     self.writer.add_scalar('RMS/privileged_obs_mean', self.obs_rms.mean, locs['it'])
        #     self.writer.add_scalar('RMS/privileged_obs_var', self.obs_rms.var, locs['it'])
        # if self.cfg["true_vel_rms"]:
        #     self.writer.add_scalar('RMS/true_vel_mean', self.obs_rms.mean, locs['it'])
        #     self.writer.add_scalar('RMS/true_vel_var', self.obs_rms.var, locs['it'])

        self.writer.add_scalar('CENet/beta', self.cenet.beta, locs['it'])
        self.writer.add_scalar('CENet/learning_rate', self.cenet.optimizer.param_groups[0]['lr'], locs['it'])
        self.writer.add_scalar('CENet/kl_loss', locs['mean_kl_loss'], locs['it'])
        self.writer.add_scalar('CENet/recon_loss', locs['mean_recon_loss'], locs['it'])
        self.writer.add_scalar('CENet/vel_loss', locs['mean_vel_loss'], locs['it'])
        self.writer.add_scalar('CENet/total_loss', locs['mean_total_loss'], locs['it'])
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'CENet KL loss:':>{pad}} {locs['mean_kl_loss']:.4f}\n"""
                          f"""{'CENet reconstruction loss:':>{pad}} {locs['mean_recon_loss']:.4f}\n"""
                          f"""{'CENet velocity estimation loss:':>{pad}} {locs['mean_vel_loss']:.4f}\n"""
                          f"""{'CENet total loss:':>{pad}} {locs['mean_total_loss']:.4f}\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'CENet KL loss:':>{pad}} {locs['mean_kl_loss']:.4f}\n"""
                          f"""{'CENet reconstruction loss:':>{pad}} {locs['mean_recon_loss']:.4f}\n"""
                          f"""{'CENet velocity estimation loss:':>{pad}} {locs['mean_vel_loss']:.4f}\n"""
                          f"""{'CENet total loss:':>{pad}} {locs['mean_total_loss']:.4f}\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'cenet_state_dict': self.cenet.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'cenet_optimizer_state_dict': self.cenet.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            'rms' : self.rms_dict
        }, path)
    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.cenet.load_state_dict(loaded_dict['cenet_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.cenet.optimizer.load_state_dict(loaded_dict['cenet_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        if self.cfg["obs_rms"] or self.cfg["privileged_obs_rms"] or self.cfg["true_vel_rms"]:
            self.rms_info = loaded_dict['rms']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_rms(self):
        return self.rms_info if (self.cfg["obs_rms"] or self.cfg["privileged_obs_rms"] or self.cfg["true_vel_rms"]) else None

    def get_inference_cenet(self, device=None):
        self.cenet.test_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.cenet.encoder.to(device)
        return self.cenet

class OnPolicyRunnerEst:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.vae_cfg = train_cfg["vae"]
        self.device = device
        self.env = env  # LeggedRobot

        env_cfg = self.env.cfg.env  # A1RoughEstCfg.env

        num_critic_obs = env_cfg.num_observations + env_cfg.num_estvel + env_cfg.num_privileged_obs
        num_actor_obs = env_cfg.num_observations + env_cfg.num_estvel

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(num_actor_obs,
                                                       num_critic_obs,
                                                       self.env.num_actions,
                                                       **self.policy_cfg).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        vae_class = eval(self.cfg["vae_class_name"])  # EstNet
        self.estnet: EstNet = vae_class(device=self.device, **self.vae_cfg).to(self.device)

        # init storage and model
        self.alg.init_storage(self.env.num_envs,
                              self.num_steps_per_env,
                              [num_actor_obs],
                              [num_critic_obs],
                              [self.env.num_actions])

        self.estnet.init_storage(self.env.num_envs,
                                self.num_steps_per_env,
                                [env_cfg.len_obs_history * env_cfg.num_observations],
                                [env_cfg.num_estvel],
                                )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.rms_dict = {}
        if self.cfg["rms"]:  # Initialize later
            self.obs_rms = None
            self.privileged_obs_rms = None
            if self.cfg["true_vel_rms"]:
                self.true_vel_rms = None

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        self.estnet.train_mode()

        ep_infos = []
        rew_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # O_{t}
            obs = self.env.get_observations().to(self.device)
            if self.obs_rms is None:
                self.obs_rms = RunningMeanStd(shape=obs.shape[1], device=self.device)
            self.obs_rms.update(obs.detach())
            obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)

            privileged_obs = self.env.get_privileged_observations().to(self.device)
            if self.privileged_obs_rms is None:
                self.privileged_obs_rms = RunningMeanStd(shape=privileged_obs.shape[1], device=self.device)
            self.privileged_obs_rms.update(privileged_obs.detach())
            privileged_obs = (privileged_obs - self.privileged_obs_rms.mean) / torch.sqrt(self.privileged_obs_rms.var + 1e-8)

            true_vel = self.env.get_true_vel().to(self.device)
            if self.cfg["true_vel_rms"]:
                if self.true_vel_rms is None:
                    self.true_vel_rms = RunningMeanStd(shape=true_vel.shape[1], device=self.device)
                self.true_vel_rms.update(true_vel.detach())
                true_vel = (true_vel - self.true_vel_rms.mean) / torch.sqrt(self.true_vel_rms.var + 1e-8)


            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    # EstNet process w/ O_{t}
                    obs_history = self.env.get_observation_history()
                    obs_history = (obs_history - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)
                    obs_history = obs_history.reshape(self.env.num_envs, -1).to(self.device)  # for estnet [num_envs, 225]

                    est_vel = self.estnet.before_action(obs_history, true_vel)

                    # AdaBoot
                    if self.cfg["ada_boot"]:
                        vel_input = est_vel if self.env.extras["episode"]["boot_prob"].item() > np.random.random() else true_vel
                    else:  # Not use AdaBoot
                        vel_input = est_vel

                    # prepare observations for actor critic
                    critic_obs = torch.cat((obs, vel_input, privileged_obs), dim=-1)
                    actor_obs = torch.cat((obs, vel_input), dim=-1)

                    obs, critic_obs, actor_obs = obs.to(self.device), critic_obs.to(self.device), actor_obs.to(self.device)

                    # A_{t}
                    actions = self.alg.act(actor_obs, critic_obs)

                    # ============================== NEXT STEP ==============================
                    # O_{t+1}
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
                    self.obs_rms.update(obs.detach())
                    obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)
                    self.privileged_obs_rms.update(privileged_obs.detach())
                    privileged_obs = (privileged_obs - self.privileged_obs_rms.mean) / torch.sqrt(self.privileged_obs_rms.var + 1e-8)

                    rewards, dones = rewards.to(self.device), dones.to(self.device)

                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        if 'reward_cv' in infos:
                            rew_infos.append(infos['reward_cv'])

                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            # estnet update
            mean_vel_loss = self.estnet.update()
            # policy update
            mean_value_loss, mean_surrogate_loss = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:

                if self.cfg["obs_rms"]:
                    self.rms_dict["obs_rms"] = self.obs_rms
                if self.cfg["privileged_obs_rms"]:
                    self.rms_dict["privileged_obs_rms"] = self.privileged_obs_rms
                if self.cfg["true_vel_rms"]:
                    self.rms_dict["true_vel_rms"] = self.true_vel_rms

                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), infos=infos)

            ep_infos.clear()
            rew_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        infos = [self.obs_rms, self.privileged_obs_rms] if self.cfg["rms"] else None
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)), infos=infos)


    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = ''
        info_types = [('ep_infos', 'Episode'), ('rew_infos', 'CV')]

        for info_type, prefix in info_types:
            if locs[info_type]:
                for key in locs[info_type][0]:  # one agent
                    infotensor = torch.tensor([], device=self.device)
                    for info in locs[info_type]:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(info[key], torch.Tensor):
                            info[key] = torch.Tensor([info[key]])
                        if len(info[key].shape) == 0:
                            info[key] = info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, info[key].to(self.device)))

                    value = torch.mean(infotensor)
                    self.writer.add_scalar(f'{prefix}/{key}', value, locs['it'])

                    if prefix == 'Episode':
                        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('EstNet/learning_rate', self.estnet.optimizer.param_groups[0]['lr'], locs['it'])
        self.writer.add_scalar('EstNet/vel_loss', locs['mean_vel_loss'], locs['it'])
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'EstNet velocity estimation loss:':>{pad}} {locs['mean_vel_loss']:.4f}\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")

        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'EstNet velocity estimation loss:':>{pad}} {locs['mean_vel_loss']:.4f}\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")


        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'estnet_state_dict': self.estnet.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'estnet_optimizer_state_dict': self.estnet.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            'rms' : self.rms_dict
        }, path)


    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.estnet.load_state_dict(loaded_dict['estnet_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.estnet.optimizer.load_state_dict(loaded_dict['estnet_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        if self.cfg["obs_rms"] or self.cfg["privileged_obs_rms"] or self.cfg["true_vel_rms"]:
            self.rms_info = loaded_dict['rms']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_rms(self):
        return self.rms_info if (self.cfg["obs_rms"] or self.cfg["privileged_obs_rms"] or self.cfg["true_vel_rms"]) else None

    def get_inference_estnet(self, device=None):
        self.estnet.test_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.estnet.estimator.to(device)
        return self.estnet

