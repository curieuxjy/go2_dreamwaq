"""Watch DreamWaQ training environment with random actions (no trained model needed).

Opens the Isaac Sim viewport so you can visually inspect:
- Terrain layout and curriculum levels
- Robot behavior under random actions
- Domain randomization effects (push, friction, mass)
- Sensor visualizations (height scanner, contact)

Usage:
    python watch.py --task=DreamWaQ-Manager-Go2-Base-Play-v0 --num_envs=4
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Watch DreamWaQ env with random policy (Manager).")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--task", type=str, default="DreamWaQ-Manager-Go2-Base-Play-v0", help="Task name.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# No headless — we want the viewport
args_cli.headless = False

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import dreamwaq_manager.tasks  # noqa: F401

from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run environment with random actions."""
    import importlib.metadata as metadata
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Enable debug visualization for commands
    if hasattr(env_cfg, "commands"):
        if hasattr(env_cfg.commands, "base_velocity"):
            env_cfg.commands.base_velocity.debug_vis = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    num_actions = env.action_space.shape[1]
    print(f"\n[INFO] Environment ready: {args_cli.task}")
    print(f"[INFO] Num envs: {args_cli.num_envs}, Action dim: {num_actions}")
    print(f"[INFO] Obs dim: {env.observation_space.shape}")
    print(f"[INFO] Running random policy — Ctrl+C to exit\n")

    obs = env.get_observations()
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.randn(args_cli.num_envs, num_actions, device=env.device) * 0.5
        obs, _, _, _ = env.step(actions)
        step += 1

        if step % 500 == 0:
            print(f"  step {step}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
