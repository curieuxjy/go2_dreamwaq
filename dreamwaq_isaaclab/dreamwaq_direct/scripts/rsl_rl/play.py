"""Script to play/evaluate a trained DreamWaQ agent (DirectRLEnv version).

Usage:
    python play.py --task=DreamWaQ-Direct-Go2-Base-Play-v0 --load_run=FOLDER --checkpoint=model_500.pt

    # track agent with video
    python play.py --task=DreamWaQ-Direct-Go2-Base-Play-v0 --load_run=FOLDER --checkpoint=model_500.pt \
        --video --enable_cameras --track_agent --track_env_index=0
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# Pin output directories to the project root regardless of cwd (see train.py for rationale).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_PROJECT_ROOT)

from isaaclab.app import AppLauncher

# local imports (cli_args.py lives alongside this script in scripts/rsl_rl/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a trained DreamWaQ agent (DirectRLEnv).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# agent tracking arguments
parser.add_argument("--track_agent", action="store_true", default=False, help="Track a specific agent with the camera.")
parser.add_argument("--track_env_index", type=int, default=0, help="Environment index of the agent to track.")
parser.add_argument("--cam_distance", type=float, default=2.5, help="Camera distance behind the tracked agent.")
parser.add_argument("--cam_height", type=float, default=1.5, help="Camera height above the tracked agent.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

# Register DreamWaQ Direct environments
import dreamwaq_direct.tasks  # noqa: F401
from dreamwaq_direct.algorithms.dreamwaq_runner import OnPolicyRunnerWaq

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def setup_agent_tracking(base_env, env_index: int, cam_distance: float, cam_height: float):
    """Configure the viewport camera to track a specific agent."""
    base_env.cfg.viewer.origin_type = "asset_root"
    base_env.cfg.viewer.asset_name = "robot"
    base_env.cfg.viewer.env_index = env_index
    base_env.cfg.viewer.eye = (cam_distance, cam_distance, cam_height)
    base_env.cfg.viewer.lookat = (0.0, 0.0, 0.0)

    if hasattr(base_env, "viewport_camera_controller") and base_env.viewport_camera_controller is not None:
        controller = base_env.viewport_camera_controller
        controller.cfg.origin_type = "asset_root"
        controller.cfg.asset_name = "robot"
        controller.cfg.env_index = env_index
        controller.default_cam_eye[:] = [cam_distance, cam_distance, cam_height]
        controller.default_cam_lookat[:] = [0.0, 0.0, 0.0]
        controller.update_view_to_asset_root("robot")
        print(f"[INFO] Camera tracking agent env_index={env_index}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    import importlib.metadata as metadata
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.track_agent:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = args_cli.track_env_index
        env_cfg.viewer.eye = (args_cli.cam_distance, args_cli.cam_distance, args_cli.cam_height)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    # get checkpoint path
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        log_dir = os.path.join(os.path.dirname(resume_path), "videos", "play")
        video_kwargs = {
            "video_folder": log_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording videos to: {log_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # setup agent tracking
    base_env = env.unwrapped
    tracked_index = args_cli.track_env_index
    if args_cli.track_agent:
        setup_agent_tracking(base_env, tracked_index, args_cli.cam_distance, args_cli.cam_height)

    # create runner and load checkpoint (Waq tasks use the custom CENet runner)
    runner_class_name = agent_cfg.to_dict().get("class_name", "OnPolicyRunner")
    is_waq = runner_class_name == "OnPolicyRunnerWaq"
    if is_waq:
        runner = OnPolicyRunnerWaq(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy
    policy = runner.get_inference_policy(device=base_env.device)

    # run evaluation loop
    obs = env.get_observations()
    dones = None
    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # the Waq policy needs the previous step's dones to reset CENet obs history
            actions = policy(obs, dones) if is_waq else policy(obs)
        obs, _, dones, _ = env.step(actions)
        step_count += 1

        # switch tracked agent if it died
        if args_cli.track_agent and dones[tracked_index]:
            alive = (~dones.bool()).nonzero(as_tuple=False).squeeze(-1)
            if len(alive) > 0:
                tracked_index = alive[0].item()
                setup_agent_tracking(base_env, tracked_index, args_cli.cam_distance, args_cli.cam_height)

        # With --video, exit once the clip is complete so the run can be scripted.
        # Without it, play.py stays an interactive infinite loop (Ctrl+C to stop).
        if args_cli.video and step_count >= args_cli.video_length:
            print(f"[INFO] Recorded {args_cli.video_length} steps — stopping.")
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
