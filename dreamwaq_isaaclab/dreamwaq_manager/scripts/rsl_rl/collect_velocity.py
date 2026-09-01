# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect command / true-velocity / CENet-estimated-velocity traces from a trained Waq agent.

Rolls out a Waq (CENet) policy for a fixed wall-clock duration and records, per step:
  * the commanded base velocity (vx, vy, wz) from the command manager,
  * the true base velocity (lin vx/vy/vz, ang wz) from the robot state,
  * the CENet-estimated base linear velocity (vx, vy, vz) captured from ``cenet.forward``.

Two views are saved to a ``.npz``: a single tracked agent (env 0) and the mean over the
first ``--mean_envs`` agents. Used to produce the velocity-tracking figures in the report.

Usage:
    python collect_velocity.py --task=DreamWaQ-Waq-Official-Rough-PPO-Play-v0 \
        --load_run=FOLDER --checkpoint=model_2999.pt --out=/tmp/vel_waq.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_PROJECT_ROOT)

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect velocity-tracking traces from a trained Waq agent.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--duration", type=float, default=20.0, help="Roll-out duration in seconds. Default: 20s.")
parser.add_argument("--mean_envs", type=int, default=100, help="Number of envs averaged for the mean view. Default: 100.")
parser.add_argument("--track_env_index", type=int, default=0, help="Env index for the single-agent view. Default: 0.")
parser.add_argument("--out", type=str, required=True, help="Output .npz path for the recorded traces.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import dreamwaq_manager.tasks  # noqa: F401
from dreamwaq_manager.algorithms.dreamwaq_runner import OnPolicyRunnerWaq

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Roll out the Waq policy and record velocity traces."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    import importlib.metadata as metadata

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    runner_class_name = agent_cfg.to_dict().get("class_name", "OnPolicyRunner")
    if runner_class_name != "OnPolicyRunnerWaq":
        raise ValueError(f"collect_velocity.py requires a Waq task (got class_name={runner_class_name}).")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    runner = OnPolicyRunnerWaq(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # Capture the CENet-estimated velocity (forward() return index 1) exactly as the policy uses it.
    captured: dict[str, torch.Tensor] = {}
    _orig_forward = runner.cenet.forward

    def _forward_capture(*a, **k):
        out = _orig_forward(*a, **k)
        captured["est_vel"] = out[1].detach().clone()
        return out

    runner.cenet.forward = _forward_capture

    policy = runner.get_inference_policy(device=base_env.device)

    step_dt = float(base_env.step_dt)
    num_steps = int(round(args_cli.duration / step_dt))
    mean_n = min(args_cli.mean_envs, base_env.num_envs)
    idx = args_cli.track_env_index
    print(f"[INFO] step_dt={step_dt:.4f}s -> {num_steps} steps for {args_cli.duration}s; "
          f"single agent env={idx}, mean over {mean_n} envs.")

    robot = base_env.scene["robot"]

    # Accumulate every agent's trace, then post-process: [num_steps, mean_n, 3].
    def _za():
        return np.zeros((num_steps, mean_n, 3), dtype=np.float32)

    all_cmd, all_true_lin, all_true_ang, all_est = _za(), _za(), _za(), _za()
    all_done = np.zeros((num_steps, mean_n), dtype=bool)

    obs = env.get_observations()
    dones = None
    for t in range(num_steps):
        with torch.inference_mode():
            actions = policy(obs, dones)
        obs, _, dones, _ = env.step(actions)

        cmd = base_env.command_manager.get_command("base_velocity")  # [N, 3] (vx, vy, wz)
        true_lin = robot.data.root_lin_vel_b  # [N, 3]
        true_ang = robot.data.root_ang_vel_b  # [N, 3]
        est = captured["est_vel"]  # [N, 3] estimated base lin vel

        all_cmd[t] = cmd[:mean_n].cpu().numpy()
        all_true_lin[t] = true_lin[:mean_n].cpu().numpy()
        all_true_ang[t] = true_ang[:mean_n].cpu().numpy()
        all_est[t] = est[:mean_n].cpu().numpy()
        all_done[t] = dones[:mean_n].cpu().numpy().astype(bool)

    # Single-agent view: pick the agent whose linear-velocity tracking error is the MEDIAN
    # (representative, not best/worst). Commands are per-env random, so the population SIGNED
    # mean cancels — for the population view we therefore use mean-absolute magnitudes.
    track_err = np.abs(all_cmd[:, :, :2] - all_true_lin[:, :, :2]).mean(axis=(0, 2))  # [mean_n]
    rep = int(np.argsort(track_err)[mean_n // 2])
    print(f"[INFO] representative (median-tracking) agent = {rep}; "
          f"track_err best/median/worst = {track_err.min():.3f}/{np.median(track_err):.3f}/{track_err.max():.3f}")

    rec = {
        # single representative agent (signed time series)
        "single_cmd": all_cmd[:, rep, :], "single_true_lin": all_true_lin[:, rep, :],
        "single_true_ang": all_true_ang[:, rep, :], "single_est": all_est[:, rep, :],
        "single_done": all_done[:, rep],
        # population mean-ABSOLUTE magnitudes (cancellation-free)
        "mean_cmd": np.abs(all_cmd).mean(1), "mean_true_lin": np.abs(all_true_lin).mean(1),
        "mean_true_ang": np.abs(all_true_ang).mean(1), "mean_est": np.abs(all_est).mean(1),
    }

    t_axis = np.arange(num_steps, dtype=np.float32) * step_dt
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    np.savez(
        args_cli.out,
        t=t_axis, step_dt=np.float32(step_dt), mean_envs=np.int64(mean_n),
        task=args_cli.task, rep_agent=np.int64(rep), mean_is_abs=np.bool_(True), **rec,
    )
    print(f"[INFO] Saved velocity traces -> {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
