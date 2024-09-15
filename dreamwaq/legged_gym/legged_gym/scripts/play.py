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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import (
    get_args,
    export_policy_as_jit,
    task_registry,
    Custom_Logger,
    Logger,
)

import numpy as np
import torch
import cv2
import time


def play(args):
    CENET = True if args.task.split("_")[-1] == "waq" else False
    ESTNET = True if args.task.split("_")[-1] == "est" else False

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    # [smooth slope, rough slope, stairs up, stairs down, discrete]
    env_cfg.terrain.terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
    env_cfg.terrain.num_rows = 5  # level
    env_cfg.terrain.num_cols = 5  # type
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 4  # level -1
    # freezing randomization
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.system_delay = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_p_gains = False
    env_cfg.domain_rand.randomize_d_gains = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_com = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # load rms
    rms = ppo_runner.get_rms()
    if rms is not None:
        obs_rms = rms["obs_rms"]
        true_vel_rms = rms["true_vel_rms"] if "true_vel_rms" in rms else None
        # privileged_obs_rms = rms["privileged_obs_rms"] # NOT NEEDED

    # load estimator
    if CENET:
        cenet = ppo_runner.get_inference_cenet(device=env.device).to(env.device)
    if ESTNET:
        estnet = ppo_runner.get_inference_estnet(device=env.device).to(env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    # logger setting
    if CENET or ESTNET:
        logger = Custom_Logger(env.dt)
    else:
        logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 100  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards

    if RECORD_FRAMES:
        ROBOT_NAME = train_cfg.runner.run_name  # base/waq/est
        gt = time.gmtime()
        TIME_TAG = "{:02}{:02}{:02}{:02}".format(
            gt.tm_mon, gt.tm_mday, gt.tm_hour, gt.tm_min
        )
        img_idx = 0
        video_duration = 5
        num_frames = int(video_duration / env.dt)
        print(f"gathering {num_frames} frames")
        video = None

    obs = env.get_observations()

    if rms is not None:
        obs = (obs - obs_rms.mean) / torch.sqrt(obs_rms.var + 1e-8)

    if CENET or ESTNET:
        obs_history = env.get_observation_history()
        if rms is not None:
            obs_history = (obs_history - obs_rms.mean) / torch.sqrt(obs_rms.var + 1e-8)
        obs_history = obs_history.reshape(env.num_envs, -1).to(env.device)

        if TRUE_VEL:
            true_vel = env.get_true_vel().to(env.device)
            if true_vel_rms is not None:
                true_vel = (true_vel - true_vel_rms.mean) / torch.sqrt(
                    true_vel_rms.var + 1e-8
                )

    for i in range(10 * int(env.max_episode_length)):
        # cenet/estnet used
        if CENET:
            obs_history = env.get_observation_history()
            if rms is not None:
                obs_history = (obs_history - obs_rms.mean) / torch.sqrt(
                    obs_rms.var + 1e-8
                )
            obs_history = obs_history.reshape(env.num_envs, -1).to(env.device)

            _, est_vel, _, _, context_vec = cenet(obs_history.detach())

            if TRUE_VEL:
                actor_obs = torch.cat(
                    (obs.detach(), true_vel.detach(), context_vec.detach()), dim=-1
                )
            else:
                actor_obs = torch.cat(
                    (obs.detach(), est_vel.detach(), context_vec.detach()), dim=-1
                )
            actions = policy(actor_obs.detach())

        elif ESTNET:

            if TRUE_VEL:
                actor_obs = torch.cat((obs.detach(), true_vel.detach()), dim=-1)
            else:
                est_vel = estnet(obs_history.detach())
                actor_obs = torch.cat((obs.detach(), est_vel.detach()), dim=-1)

            actions = policy(actor_obs.detach())

        else:
            actions = policy(obs.detach())

        obs, _, rews, dones, infos = env.step(actions.detach())

        if rms is not None:
            obs = (obs - obs_rms.mean.to(env.device)) / torch.sqrt(
                obs_rms.var.to(env.device) + 1e-8
            )

        if RECORD_FRAMES:

            frames_path = os.path.join(
                LEGGED_GYM_ROOT_DIR, "records", ROBOT_NAME, TIME_TAG, "frames"
            )

            video_path = os.path.join(
                LEGGED_GYM_ROOT_DIR, "records", ROBOT_NAME, TIME_TAG
            )

            if not os.path.exists(frames_path):
                os.makedirs(frames_path)

            filename = os.path.join(frames_path, f"{img_idx}.png")
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img = cv2.imread(filename)

            if video is None:
                if TRUE_VEL:
                    video = cv2.VideoWriter(
                        os.path.join(
                            video_path, "{}_record_true.mp4".format(ROBOT_NAME)
                        ),
                        cv2.VideoWriter_fourcc(*"MP4V"),
                        int(1 / env.dt),
                        (img.shape[1], img.shape[0]),
                    )
                else:
                    video = cv2.VideoWriter(
                        os.path.join(
                            video_path, "{}_record_est.mp4".format(ROBOT_NAME)
                        ),
                        cv2.VideoWriter_fourcc(*"MP4V"),
                        int(1 / env.dt),
                        (img.shape[1], img.shape[0]),
                    )

            video.write(img)
            img_idx += 1

        if i < stop_state_log:
            if CENET or ESTNET:
                logger.log_states(
                    {
                        "command_x": env.commands[robot_index, 0].item(),
                        "command_y": env.commands[robot_index, 1].item(),
                        "command_yaw": env.commands[robot_index, 2].item(),
                        "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                        "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                        "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                        "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                        "est_vel_x": est_vel[robot_index, 0].item(),
                        "est_vel_y": est_vel[robot_index, 1].item(),
                        "est_vel_z": est_vel[robot_index, 2].item(),
                        "squared_error": np.sum(
                            (
                                env.base_lin_vel[robot_index, 0].item()
                                - est_vel[robot_index, 0].item()
                            )
                            ** 2
                            + (
                                env.base_lin_vel[robot_index, 1].item()
                                - est_vel[robot_index, 1].item()
                            )
                            ** 2
                            + (
                                env.base_lin_vel[robot_index, 2].item()
                                - est_vel[robot_index, 2].item()
                            )
                            ** 2
                        ),
                        "mse_vel_x": (
                            env.base_lin_vel[robot_index, 0].item()
                            - est_vel[robot_index, 0].item()
                        )
                        ** 2,
                        "mse_vel_y": (
                            env.base_lin_vel[robot_index, 1].item()
                            - est_vel[robot_index, 1].item()
                        )
                        ** 2,
                    }
                )
            else:
                logger.log_states(
                    {
                        "dof_pos_target": actions[robot_index, joint_index].item()
                        * env.cfg.control.action_scale,
                        "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                        "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                        "dof_torque": env.torques[robot_index, joint_index].item(),
                        "command_x": env.commands[robot_index, 0].item(),
                        "command_y": env.commands[robot_index, 1].item(),
                        "command_yaw": env.commands[robot_index, 2].item(),
                        "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                        "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                        "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                        "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                        "contact_forces_z": env.contact_forces[
                            robot_index, env.feet_indices, 2
                        ]
                        .cpu()
                        .numpy(),
                    }
                )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = False
    RECORD_FRAMES = True  # render a video
    TRUE_VEL = True  # inference with true base velocity not estimated base velocity
    args = get_args()
    play(args)
