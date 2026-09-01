"""Script to play/evaluate a trained DreamWaQ agent with RSL-RL.

Usage:
    # viewport overview (default)
    python play.py --task=DreamWaQ-Manager-Go2-Base-Play-v0 --load_run=FOLDER --checkpoint=model_500.pt

    # track a specific agent with video recording
    python play.py --task=DreamWaQ-Manager-Go2-Base-Play-v0 --load_run=FOLDER --checkpoint=model_500.pt \
        --video --enable_cameras --track_agent --track_env_index=0

    # track agent with custom camera offset
    python play.py ... --track_agent --cam_distance=2.0 --cam_height=1.0
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
parser = argparse.ArgumentParser(description="Evaluate a trained DreamWaQ agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps). 400 steps = 1 episode (20s).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# rerun visualization
parser.add_argument("--rerun", action="store_true", default=False, help="Enable rerun visualization of robot state.")
# debug visualization (command-velocity arrows + terrain height-scan hits)
parser.add_argument(
    "--debug_vis",
    action="store_true",
    default=False,
    help="Enable debug visualization: command-velocity arrows (goal vs current) and terrain height-scan ray hits.",
)
# agent tracking arguments
parser.add_argument("--track_agent", action="store_true", default=False, help="Track a specific agent with the camera.")
parser.add_argument("--track_env_index", type=int, default=0, help="Environment index of the agent to track. Default: 0.")
parser.add_argument("--cam_distance", type=float, default=2.5, help="Camera distance behind the tracked agent. Default: 2.5m.")
parser.add_argument("--cam_height", type=float, default=1.5, help="Camera height above the tracked agent. Default: 1.5m.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

# Register DreamWaQ environments
import dreamwaq_manager.tasks  # noqa: F401
from dreamwaq_manager.algorithms.dreamwaq_runner import OnPolicyRunnerWaq

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def setup_agent_tracking(base_env, env_index: int, cam_distance: float, cam_height: float):
    """Configure the viewport camera to track a specific agent.

    Args:
        base_env: The unwrapped ManagerBasedRLEnv.
        env_index: Which environment index to track.
        cam_distance: Camera distance behind the agent (meters).
        cam_height: Camera height above the agent (meters).
    """
    # Set viewer config to track the robot asset root
    base_env.cfg.viewer.origin_type = "asset_root"
    base_env.cfg.viewer.asset_name = "robot"
    base_env.cfg.viewer.env_index = env_index
    base_env.cfg.viewer.eye = (cam_distance, cam_distance, cam_height)
    base_env.cfg.viewer.lookat = (0.0, 0.0, 0.0)

    # Update the camera controller
    if hasattr(base_env, "viewport_camera_controller") and base_env.viewport_camera_controller is not None:
        controller = base_env.viewport_camera_controller
        controller.cfg.origin_type = "asset_root"
        controller.cfg.asset_name = "robot"
        controller.cfg.env_index = env_index
        controller.default_cam_eye[:] = [cam_distance, cam_distance, cam_height]
        controller.default_cam_lookat[:] = [0.0, 0.0, 0.0]
        controller.update_view_to_asset_root("robot")
        print(f"[INFO] Camera tracking agent env_index={env_index}")


def switch_tracked_agent(base_env, current_index: int, reset_buf: torch.Tensor,
                         cam_distance: float, cam_height: float) -> int:
    """Switch camera to another alive agent if the current one is reset.

    Args:
        base_env: The unwrapped ManagerBasedRLEnv.
        current_index: Currently tracked environment index.
        reset_buf: Boolean tensor of which envs were just reset.
        cam_distance: Camera distance behind the agent.
        cam_height: Camera height above the agent.

    Returns:
        New environment index being tracked.
    """
    if not reset_buf[current_index]:
        return current_index

    # Current agent died — find a live one
    alive_mask = ~reset_buf
    alive_indices = torch.where(alive_mask)[0]

    if len(alive_indices) == 0:
        # All agents reset this step — just pick 0
        new_index = 0
    else:
        new_index = alive_indices[0].item()

    print(f"[INFO] Tracked agent {current_index} reset. Switching to agent {new_index}")
    setup_agent_tracking(base_env, new_index, cam_distance, cam_height)
    return new_index


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate with RSL-RL agent."""
    # override configurations
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    import importlib.metadata as metadata
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # enable debug visualization markers before env creation (rendered into the camera/video)
    if args_cli.debug_vis:
        # command-velocity arrows: goal (commanded) vs current base velocity — as in the official velocity example
        commands_cfg = getattr(env_cfg, "commands", None)
        base_velocity_cfg = getattr(commands_cfg, "base_velocity", None) if commands_cfg is not None else None
        if base_velocity_cfg is not None:
            base_velocity_cfg.debug_vis = True
            print("[INFO] Debug vis: command-velocity arrows enabled.")
        # terrain height-scan ray hits around the robot (rough terrain only)
        height_scanner_cfg = getattr(env_cfg.scene, "height_scanner", None)
        if height_scanner_cfg is not None:
            height_scanner_cfg.debug_vis = True
            print("[INFO] Debug vis: height-scan ray hits enabled.")
        else:
            print("[INFO] Debug vis: no height_scanner in this env (flat terrain) — skipping height vis.")

    # if tracking agent, set viewer config before env creation
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
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

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

    # setup agent tracking after env creation
    base_env = env.unwrapped
    tracked_index = args_cli.track_env_index
    if args_cli.track_agent:
        setup_agent_tracking(base_env, tracked_index, args_cli.cam_distance, args_cli.cam_height)

    # setup rerun logger
    rr_logger = None
    if args_cli.rerun:
        from dreamwaq_manager.utils.rerun_logger import RerunLogger
        rr_logger = RerunLogger(env_index=tracked_index, app_name="dreamwaq_play")
        print(f"[INFO] Rerun visualization enabled (tracking env_index={tracked_index})")

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
        if args_cli.track_agent:
            reset_buf = dones.bool()
            tracked_index = switch_tracked_agent(
                base_env, tracked_index, reset_buf,
                args_cli.cam_distance, args_cli.cam_height,
            )
            if rr_logger is not None:
                rr_logger.set_env_index(tracked_index)

        # log to rerun
        if rr_logger is not None:
            obs_policy = obs["policy"] if isinstance(obs, dict) else obs
            obs_critic = obs.get("critic", None) if isinstance(obs, dict) else None
            rr_logger.log_step(step_count, obs_policy, obs_critic, actions)

        # With --video, exit once the clip is complete so the run can be scripted.
        # Without it, play.py stays an interactive infinite loop (Ctrl+C to stop).
        if args_cli.video and step_count >= args_cli.video_length:
            print(f"[INFO] Recorded {args_cli.video_length} steps — stopping.")
            break

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
