"""Script to train DreamWaQ agent with RSL-RL.

Usage:
    python train.py --task=DreamWaQ-Manager-Go2-Base-v0 --headless
    python train.py --task=DreamWaQ-Manager-Go2-Base-v0 --headless --num_envs=64
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# Pin all output directories (RSL-RL logs/, Hydra outputs/, wandb/) to the
# project root regardless of cwd. RSL-RL builds log paths via os.path.abspath
# of a relative "logs/..." string, Hydra builds outputs/ in cwd, and wandb
# defaults its dir to cwd — so chdir'ing once here covers all three.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_PROJECT_ROOT)

from isaaclab.app import AppLauncher

# local imports (cli_args.py lives alongside this script in scripts/rsl_rl/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train DreamWaQ agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
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
import time
from datetime import datetime

# Disable wandb Sentry telemetry (prevents 90s init timeout when DNS blocks sentry.io)
os.environ.setdefault("WANDB_ERROR_REPORTING", "false")
os.environ.setdefault("WANDB__DISABLE_STATS", "true")
os.environ.setdefault("SENTRY_DSN", "")
os.environ.setdefault("WANDB_INIT_TIMEOUT", "180")

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

# Monkey-patch: clamp std to prevent NaN in Normal distribution sampling
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal

# Upper bound of the log_std clamp. Default 0.0 (std <= 1.0). Env-var gated so the
# entropy-ceiling probe (PAPER.md §8-2) can sweep it without touching the shipped default.
_LOGSTD_MAX = float(os.environ.get("DWQ_LOGSTD_MAX", "0.0"))
_LOGSTD_MIN = -5.0

_original_update = GaussianDistribution.update

def _safe_update(self, mlp_output: torch.Tensor) -> None:
    mean = torch.nan_to_num(mlp_output, nan=0.0, posinf=1e6, neginf=-1e6)
    if self.std_type == "scalar":
        std = torch.nan_to_num(self.std_param, nan=1.0).clamp(min=1e-6, max=1.0).expand_as(mean)
    elif self.std_type == "log":
        # Clamp log_std parameter in-place to prevent drift to extreme values during training.
        # Without this, adaptive LR + narrow obs distribution can push log_std to ±inf mid-run,
        # causing training collapse where all logged rewards go to 0.
        # Upper bound 0.0 (std <= 1.0). Rationale as originally written: with actions clamped to
        # [-1, 1] a larger std saturates almost every sampled action at +-1, so exploration becomes
        # max-amplitude noise (std was pinning at the old ceiling 2.0 / std~7.4).
        # UPDATE (PAPER.md §6): `clip_actions` is now 4.0, so that "[-1, 1]" premise no longer
        # holds. The clip was in fact the CAUSE of the pinning — log-prob/entropy are computed on
        # the UNCLIPPED Gaussian, so past sigma~1 extra noise was free while the entropy bonus kept
        # paying, leaving no equilibrium below this ceiling. With clip 4.0 the equilibrium is back
        # (sigma* = 0.374) and the clamp no longer binds; it stays only as a NaN/divergence guard.
        # Range [-5, 0] → std in [~0.007, ~1.0] (matches OnPolicyRunnerWaq hook).
        with torch.no_grad():
            self.log_std_param.data.clamp_(_LOGSTD_MIN, _LOGSTD_MAX)
            self.log_std_param.data = torch.nan_to_num(
                self.log_std_param.data, nan=_LOGSTD_MAX, posinf=_LOGSTD_MAX, neginf=_LOGSTD_MIN
            )
        std = torch.exp(self.log_std_param).clamp(min=1e-6).expand_as(mean)
    self._distribution = Normal(mean, std)

GaussianDistribution.update = _safe_update


def _register_std_clamp_for_all_runners(runner):
    """Register log_std_param clamp hook on any runner's optimizer.

    OnPolicyRunnerWaq already does this, but OnPolicyRunner (Base/Oracle) does not.
    Without the hook, log_std_param can drift to extreme values during PPO gradient
    updates, eventually collapsing training mid-run.
    """
    if not hasattr(runner, "alg") or not hasattr(runner.alg, "optimizer"):
        return

    # Select the log_std parameter by OBJECT IDENTITY, not by shape.
    #
    # The old test was `param.ndim == 1 and param.shape[0] == num_actions`, and the actor MLP's
    # OUTPUT-LAYER BIAS (`mlp.6.bias`) has exactly that shape (12,). So the hook clamped the
    # per-joint constant offset of the mean action to [-5, 0] too — it could never be positive.
    # Measured across the six 2026-08 runs: all 6 x 12 = 72 output biases were <= 0, several
    # within 1e-5 of the boundary. Unconstrained the signs should be roughly balanced, so
    # P(all 72 negative by chance) = 2^-72. The clamp itself stays (std runaway / NaN guard);
    # only its target is narrowed.
    log_std_params = [m.log_std_param for m in runner.alg.actor.modules() if hasattr(m, "log_std_param")]
    if not log_std_params:
        print("[WARN] no log_std_param on the actor (std_type != 'log'?) — clamp hook NOT registered")
        return

    def _clamp_std_hook(optimizer, args, kwargs):
        for param in log_std_params:
            with torch.no_grad():
                param.data.clamp_(_LOGSTD_MIN, _LOGSTD_MAX)
                param.data = torch.nan_to_num(
                    param.data, nan=_LOGSTD_MAX, posinf=_LOGSTD_MAX, neginf=_LOGSTD_MIN
                )

    runner.alg.optimizer.register_step_post_hook(_clamp_std_hook)
    print(
        f"[INFO] Registered log_std_param clamp hook [{_LOGSTD_MIN}, {_LOGSTD_MAX}] on"
        f" {len(log_std_params)} actor log_std parameter(s)"
    )


# --- Diagnostic probe (PAPER.md §8-2): who wins the tug-of-war on log_std? ---------------
# loss = surrogate + c_v * value - c_H * entropy, and for a diagonal Gaussian
#   entropy = sum_i log_std_i + const   =>   d(-c_H * entropy)/d log_std_i = -c_H  exactly.
# So the entropy term is a CONSTANT upward force of magnitude entropy_coef on every dim,
# and  g_surrogate = g_total + entropy_coef.  Logging g_total therefore decomposes the
# gradient with no extra backward pass. Gated by DWQ_PROBE_STDGRAD=1.
_STDGRAD: dict = {"sum": None, "n": 0}


def _enable_stdgrad_probe(runner):
    from rsl_rl.algorithms.ppo import PPO

    param = None
    for module in runner.alg.actor.modules():
        if hasattr(module, "log_std_param"):
            param = module.log_std_param
    if param is None:
        print("[WARN] stdgrad probe: no log_std_param found — skipped")
        return

    def _accumulate(grad):
        if _STDGRAD["sum"] is None:
            _STDGRAD["sum"] = torch.zeros_like(grad)
        _STDGRAD["sum"] += grad.detach()
        _STDGRAD["n"] += 1

    param.register_hook(_accumulate)

    if getattr(PPO, "_stdgrad_probe_installed", False):
        return
    _ppo_update = PPO.update

    def _update_with_probe(self, *args, **kwargs):
        _STDGRAD["sum"], _STDGRAD["n"] = None, 0
        out = _ppo_update(self, *args, **kwargs)
        if _STDGRAD["n"] and isinstance(out, dict):
            g_total = _STDGRAD["sum"] / _STDGRAD["n"]
            g_surr = g_total + self.entropy_coef
            out["gtot_logstd"] = g_total.mean().item()
            out["gsurr_logstd"] = g_surr.mean().item()
            out["gsurr_logstd_abs"] = g_surr.abs().mean().item()
        return out

    PPO.update = _update_with_probe
    PPO._stdgrad_probe_installed = True
    print("[INFO] log_std gradient probe enabled -> Loss/{gtot,gsurr,gsurr_abs}_logstd")


# --- TEMPORARY diagnostic (DWQ_PROBE_REWMAX=1): per-term reward EXTREMES ------------------
# Episode_Reward/* is an average and hides single-sample outliers; a NaN-inducing spike is
# invisible there. This wraps RewardManager.compute to track the per-step weighted max of
# every term and print it periodically. Remove once PAPER.md §2 reward-collapse is resolved.
if os.environ.get("DWQ_PROBE_REWMAX"):
    from isaaclab.managers import RewardManager as _RM

    _rewmax: dict = {"max": None, "n": 0, "nonfinite": 0}
    _rm_compute = _RM.compute

    def _compute_probe(self, dt):
        out = _rm_compute(self, dt)
        step = self._step_reward / dt
        cur = step.abs().amax(dim=0)
        _rewmax["max"] = cur if _rewmax["max"] is None else torch.maximum(_rewmax["max"], cur)
        _rewmax["nonfinite"] += int((~torch.isfinite(step)).sum())
        if not torch.isfinite(out).all():
            _rewmax["nonfinite"] += 1
        _rewmax["n"] += 1
        if _rewmax["n"] % 240 == 0:
            pairs = sorted(
                zip(self.active_terms, _rewmax["max"].tolist()), key=lambda kv: -kv[1]
            )
            print(
                f"[REWMAX] step {_rewmax['n']} nonfinite={_rewmax['nonfinite']} :: "
                + "  ".join(f"{k}={v:.4g}" for k, v in pairs[:6]),
                flush=True,
            )
            _rewmax["max"] = None  # window max, so growth is visible
        return out

    _RM.compute = _compute_probe
    print("[INFO] reward-extreme probe enabled -> [REWMAX] lines")


from dreamwaq_manager.algorithms.dreamwaq_runner import OnPolicyRunnerWaq

from isaaclab.envs import (
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

# Register DreamWaQ environments
import dreamwaq_manager.tasks  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    import importlib.metadata as metadata
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # set wandb run name to show task variant, seed, and num_envs
    task_name = args_cli.task or "unknown"
    seed = agent_cfg.seed if agent_cfg.seed is not None else 42
    num_envs = env_cfg.scene.num_envs
    agent_cfg.run_name = f"{task_name}_seed{seed}_envs{num_envs}"

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl (use DreamWaQ runner for Waq tasks)
    agent_dict = agent_cfg.to_dict()
    # Debug: verify distribution config
    if "actor" in agent_dict:
        dist_cfg = agent_dict["actor"].get("distribution_cfg", None)
        print(f"[DEBUG] Actor distribution_cfg: {dist_cfg}")
    runner_class_name = agent_dict.get("class_name", "OnPolicyRunner")
    if runner_class_name == "OnPolicyRunnerWaq":
        runner = OnPolicyRunnerWaq(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)
        # OnPolicyRunnerWaq registers this hook itself; only standard runners need it.
        _register_std_clamp_for_all_runners(runner)

    if os.environ.get("DWQ_PROBE_STDGRAD") == "1":
        _enable_stdgrad_probe(runner)

    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
