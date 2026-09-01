#!/usr/bin/env python3
"""각 run 의 체크포인트를 전부 평가해 **best policy** 를 고른다.

`compare_runs.py` 는 학습 곡선(수렴 구간 평균)으로 비교한다. 그건 "학습이 어디까지 갔나"는
알려주지만 **배포할 정책**의 성능은 아니다. 강화학습 성능 비교는 마지막 체크포인트가 아니라
**가장 좋은 체크포인트**로 해야 한다 — 마지막이 최고라는 보장이 전혀 없기 때문이다.

이 프로젝트가 실제로 그런 경우다. Waq 의 CENet 은 학습 도중 posterior collapse 를 일으켜
**뒤로 갈수록 나빠진다.** 마지막 체크포인트로만 비교하면 CENet 을 부당하게 과소평가한다.

각 체크포인트마다 고정된 조건(같은 seed, 같은 env 수, 같은 스텝 수)으로 굴려
속도추종(underlying) 을 매 스텝 직접 계산해 재고, run 별 best 를 표로 낸다.

    cd dreamwaq_manager
    ~/IsaacLab/_isaac_sim/python.sh scripts/eval_checkpoints.py --headless
    ~/IsaacLab/_isaac_sim/python.sh scripts/eval_checkpoints.py --headless --terrain rough
    ~/IsaacLab/_isaac_sim/python.sh scripts/eval_checkpoints.py --headless --steps 400

Isaac Sim 을 체크포인트 수만큼 띄우지 않고, task 하나당 한 번만 띄워 체크포인트를 갈아끼운다.
"""

import argparse
import os
import re
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsl_rl"))
import cli_args  # isort: skip  # noqa: E402

TERRAINS = ("flat", "rough")
VARIANTS = {  # 표시이름 -> (task 접두사, experiment_name 접두사)
    "Base": ("DreamWaQ-BaseDwq-{t}-PPO", "BaseDwq-Official-{t}-PPO-v0"),
    "Waq": ("DreamWaQ-Waq-Official-{t}-PPO", "Waq-Official-{t}-PPO-v0"),
    "Oracle": ("DreamWaQ-OracleDwq-{t}-PPO", "OracleDwq-Official-{t}-PPO-v0"),
}

parser = argparse.ArgumentParser(description="체크포인트를 전부 평가해 best policy 를 고른다.")
parser.add_argument("--num_envs", type=int, default=256, help="평가에 쓸 env 수. 많을수록 분산이 준다.")
parser.add_argument("--steps", type=int, default=600, help="체크포인트당 굴릴 스텝 수 (600 = 30s).")
parser.add_argument("--terrain", choices=[*TERRAINS, "both"], default="both")
parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
parser.add_argument("--seed", type=int, default=1234, help="평가 seed (학습 seed 와 달라야 공정하다).")
parser.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(_PROJECT_ROOT), "figures", "best_policy.csv"),
                    help="결과 CSV 경로 (기본: repo 루트의 figures/, compare_runs.py 와 같은 곳)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = False
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv  # noqa: E402
import glob  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import dreamwaq_manager.tasks  # noqa: F401, E402
from dreamwaq_manager.algorithms.dreamwaq_runner import OnPolicyRunnerWaq  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def ckpt_iter(path: str) -> int:
    m = re.search(r"model_(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def latest_run_dir(experiment: str) -> str | None:
    root = os.path.join("logs", "rsl_rl", experiment)
    if not os.path.isdir(root):
        return None
    runs = sorted((os.path.join(root, d) for d in os.listdir(root)), key=os.path.getmtime)
    runs = [r for r in runs if os.path.isdir(r)]
    return runs[-1] if runs else None


def evaluate(env, runner, policy, steps: int) -> tuple[float, float]:
    """고정 스텝 굴리며 매 스텝 속도추종을 직접 계산한다. (평균, 생존율) 반환.

    ``extras["log"]["Episode_Reward/..."]`` 를 읽으면 안 된다 — 그 값은 **에피소드가 끝날 때만**
    실린다. 평가 구간(수백 스텝)보다 에피소드(1000 스텝)가 길어서, 실제로 집계되는 것은
    **일찍 끝난 = 넘어진 에피소드뿐**이라 성능이 심하게 과소평가된다.

    대신 보상 정의를 그대로 재현한다 (velocity_env_cfg 의 track_lin_vel_xy_exp):

        exp(-||v_cmd_xy - v_xy||^2 / 0.25)   in [0, 1]
    """
    base = env.unwrapped
    robot = base.scene["robot"]
    obs = env.get_observations()
    dones = None
    total, n, alive = 0.0, 0, 0.0
    with torch.inference_mode():
        for _ in range(steps):
            actions = policy(obs, dones) if runner_is_waq(runner) else policy(obs)
            obs, _, dones, _ = env.step(actions)
            cmd = base.command_manager.get_command("base_velocity")[:, :2]
            vel = robot.data.root_lin_vel_b
            if not isinstance(vel, torch.Tensor):  # warp ProxyArray
                vel = vel.torch
            err = torch.sum((cmd - vel[:, :2]) ** 2, dim=1)
            total += float(torch.exp(-err / 0.25).mean())
            alive += float((~dones.bool()).float().mean())
            n += 1
    return (total / n if n else float("nan")), (alive / n if n else float("nan"))


def runner_is_waq(runner) -> bool:
    return isinstance(runner, OnPolicyRunnerWaq)


def main() -> int:
    terrains = TERRAINS if args_cli.terrain == "both" else (args_cli.terrain,)
    variants = list(VARIANTS) if args_cli.variant == "all" else [args_cli.variant]

    rows: list[dict] = []
    for terrain in terrains:
        T = terrain.capitalize()
        for name in variants:
            task_pat, exp_pat = VARIANTS[name]
            task = task_pat.format(t=T) + "-Play-v0"
            exp = exp_pat.format(t=T)
            run_dir = latest_run_dir(exp)
            if run_dir is None:
                print(f"[warn] {terrain}/{name}: run 없음 ({exp}) — 건너뜀")
                continue
            ckpts = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")), key=ckpt_iter)
            if not ckpts:
                print(f"[warn] {terrain}/{name}: 체크포인트 없음 — 건너뜀")
                continue

            env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
            env_cfg.seed = args_cli.seed
            agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
            import importlib.metadata as md

            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, md.version("rsl-rl-lib"))
            agent_dict = agent_cfg.to_dict()

            print(f"\n=== {terrain}/{name} — 체크포인트 {len(ckpts)}개 ===")
            env = gym.make(task, cfg=env_cfg, render_mode=None)
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            cls = OnPolicyRunnerWaq if agent_dict.get("class_name") == "OnPolicyRunnerWaq" else OnPolicyRunner
            runner = cls(env, agent_dict, log_dir=None, device=agent_cfg.device)

            for ck in ckpts:
                runner.load(ck)
                policy = runner.get_inference_policy(device=env.unwrapped.device)
                env.unwrapped.reset()
                under, alive = evaluate(env, runner, policy, args_cli.steps)
                print(f"  {os.path.basename(ck):>16}  underlying={under:.4f}  alive={alive:.3f}")
                rows.append({
                    "terrain": terrain, "variant": name,
                    "checkpoint": os.path.basename(ck), "iter": ckpt_iter(ck),
                    "underlying": round(under, 4), "alive": round(alive, 4),
                })
            env.close()

    if not rows:
        print("\n평가할 run 이 없다.")
        return 1

    os.makedirs(os.path.dirname(args_cli.out) or ".", exist_ok=True)
    with open(args_cli.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["terrain", "variant", "checkpoint", "iter", "underlying", "alive"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] {args_cli.out}")

    print("\n=== best policy (체크포인트별 평가 중 최고) ===")
    print(f"{'terrain':8} {'variant':8} {'best ckpt':>16} {'underlying':>11}   {'최종 ckpt':>16} {'underlying':>11}")
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        k = (r["terrain"], r["variant"])
        if k not in best or r["underlying"] > best[k]["underlying"]:
            best[k] = r
    for (terrain, name), b in sorted(best.items()):
        last = max((r for r in rows if (r["terrain"], r["variant"]) == (terrain, name)),
                   key=lambda r: r["iter"])
        print(f"{terrain:8} {name:8} {b['checkpoint']:>16} {b['underlying']:>11.4f}   "
              f"{last['checkpoint']:>16} {last['underlying']:>11.4f}")

    for terrain in terrains:
        got = {n: best[(terrain, n)]["underlying"] for n in variants if (terrain, n) in best}
        if len(got) == 3:
            print(f"\n[{terrain}] best 기준  Waq-Base = {got['Waq'] - got['Base']:+.4f}, "
                  f"Oracle-Waq = {got['Oracle'] - got['Waq']:+.4f}")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
