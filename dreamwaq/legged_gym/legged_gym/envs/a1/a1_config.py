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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class A1RoughCfg(LeggedRobotCfg):

    class terrain(LeggedRobotCfg.terrain):
        measure_heights = False
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48  # o(45) + true_lin_vel(3)
        # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_privileged_obs = None  # d(3) + h(187)
        num_actions = 12
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.1,  # [rad]
            "RL_hip_joint": 0.1,  # [rad]
            "FR_hip_joint": -0.1,  # [rad]
            "RR_hip_joint": -0.1,  # [rad]
            "FL_thigh_joint": 0.8,  # [rad]
            "RL_thigh_joint": 1.0,  # [rad]
            "FR_thigh_joint": 0.8,  # [rad]
            "RR_thigh_joint": 1.0,  # [rad]
            "FL_calf_joint": -1.5,  # [rad]
            "RL_calf_joint": -1.5,  # [rad]
            "FR_calf_joint": -1.5,  # [rad]
            "RR_calf_joint": -1.5,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"joint": 28.0}  # [N*m/rad] # checked
        damping = {"joint": 0.7}  # [N*m*s/rad] # checked
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # checked 50Hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # checked 200Hz
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf"
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class commands(LeggedRobotCfg.commands):
        inference_policy = False
        inference_command = "x"  # "y", "mix"
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand(LeggedRobotCfg.domain_rand):
        system_delay = True
        push_robots = True
        push_interval_s = 1.0
        max_push_vel_xy = 1.0
        randomize_friction = True
        friction_range = [0.2, 1.25]  # checked
        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]  # checked
        randomize_p_gains = True
        p_gains_range = [0.9, 1.1]
        randomize_d_gains = True
        d_gains_range = [0.9, 1.1]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_com = True  # all bodies
        com_range = [-0.05, 0.05]

    class rewards(LeggedRobotCfg.rewards):
        """CAUTION: this reward configuration is NOT from the paper"""

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0

    # class normalization(LeggedRobotCfg.normalization):
    #     rms = True


class A1RoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        ada_boot = False
        run_name = "default"
        experiment_name = "rough_a1"


class A1RoughBaseCfg(A1RoughCfg):

    class terrain(A1RoughCfg.terrain):
        measure_heights = False

    class env(A1RoughCfg.env):
        num_observations = 45  # o(45)
        num_privileged_obs = None  # d(3) + h(187)

    class rewards(A1RoughCfg.rewards):
        """SAME reward functions with the paper"""

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        desired_foot_height = 0.12

        class scales(A1RoughCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2.0e-5
            base_height = -1.0
            action_rate = -0.01
            smoothness = -0.01
            power_distribution = -1.0e-6
            foot_clearance = -0.01
            termination = -0.0
            torques = -0.0
            dof_vel = -0.0
            feet_air_time = 0.0
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -0.0
            dof_pos_limits = -0.0


class A1RoughBaseCfgPPO(A1RoughCfgPPO):
    seed = 1

    class runner(A1RoughCfgPPO.runner):
        obs_rms = True
        privileged_obs_rms = False
        true_vel_rms = False
        ada_boot = False
        run_name = "base"
        experiment_name = "rough_a1"


class A1RoughOracleCfg(A1RoughBaseCfg):

    class terrain(A1RoughBaseCfg.terrain):
        measure_heights = True

    class env(A1RoughBaseCfg.env):
        num_observations = 48
        num_privileged_obs = 190  # d(3) + h(187)

    class rewards(A1RoughBaseCfg.rewards):
        """SAME reward functions with the paper"""

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)


class A1RoughOracleCfgPPO(A1RoughBaseCfgPPO):
    seed = 1

    class runner(A1RoughBaseCfgPPO.runner):
        obs_rms = True
        privileged_obs_rms = True
        true_vel_rms = False
        ada_boot = False
        run_name = "oracle"
        experiment_name = "rough_a1"


class A1RoughWaqCfg(A1RoughBaseCfg):

    class terrain(A1RoughBaseCfg.terrain):
        measure_heights = True

    class env(A1RoughBaseCfg.env):
        num_observations = 45  # o(45)
        len_obs_history = 5  # o_H
        num_context = 16  # z
        num_estvel = 3  # v
        num_privileged_obs = 190  # d(3) + h(187)

    class rewards(A1RoughBaseCfg.rewards):
        """SAME reward functions with the paper"""

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)

    class normalization(A1RoughBaseCfg.normalization):
        fixed_norm = False

    class noise(A1RoughBaseCfg.noise):
        add_noise = False
        noise_level = 1.0  # scales other values


class A1RoughCfgWaqPPO(A1RoughBaseCfgPPO):
    seed = 1

    class vae:
        beta = 1.0
        beta_limit = 4.0
        learning_rate = 0.01
        min_lr = 0.0015
        patience = 100
        factor = 0.8

    class runner(A1RoughBaseCfgPPO.runner):
        obs_rms = True
        privileged_obs_rms = True
        true_vel_rms = True
        ada_boot = True
        vae_class_name = "CENet"
        run_name = "waq"
        experiment_name = "rough_a1_waq"


class A1RoughEstCfg(A1RoughBaseCfg):

    class terrain(A1RoughBaseCfg.terrain):
        measure_heights = True

    class env(A1RoughBaseCfg.env):
        num_observations = 45  # o(45)
        len_obs_history = 5  # o_H
        num_estvel = 3  # v
        num_privileged_obs = 190  # d(3) + h(187)

    class rewards(A1RoughBaseCfg.rewards):
        """SAME reward functions with the paper"""

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)

    class noise(A1RoughBaseCfg.noise):
        add_noise = False
        noise_level = 1.0  # scales other values


class A1RoughCfgEstPPO(A1RoughBaseCfgPPO):
    seed = 1

    class vae:
        learning_rate = 0.01
        min_lr = 0.0015
        patience = 100
        factor = 0.8

    class runner(A1RoughBaseCfgPPO.runner):
        rms = True
        true_vel_rms = False
        ada_boot = True
        vae_class_name = "EstNet"
        run_name = "est"
        experiment_name = "rough_a1_est"
