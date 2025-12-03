# dreamwaq

> Independent implementation of [DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning](https://arxiv.org/abs/2301.10602)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** [Jungyeon Lee (curieuxjy)](https://github.com/curieuxjy)

I independently implemented the DreamWaQ algorithm based on the paper. The core component, **Context-aided Estimator Network (CENet)**, has been carefully implemented and verified to work as described. Feel free to explore the code and experiment with it!

---

https://github.com/curieuxjy/dreamwaq/assets/40867411/5dcea5c9-3ff3-469d-baa7-70f0852a0395

[🎥 1080 Streaming Video in YouTube](https://youtu.be/5rwFcz-lerw)

---

## Table of Contents

| Section | Description |
|---------|-------------|
| [Start Manual](#start-manual) | Project environment setup and execution instructions |
| [Main Code Structure](#main-code-structure) | Main code structure explanation |
| [Result Graphs](#result-graphs) | Training result graphs |
| [Result Motions](#result-motions) | Training result walking motion videos (gif per section) |

---

## Start Manual

### Start **w/o** this repository

> This is the initial setup for implementation project independent of this repository. To run based on this repository, please refer to the w/ execution steps below.

1. Install IsaacGym ver.4
2. Download [rsl-rl](https://github.com/leggedrobotics/rsl_rl) from github as **zip** file and install `pip install -e .`
3. Download [legged-gym](https://github.com/leggedrobotics/legged_gym) from github as **zip** file and install `pip install -e .`
4. Modify some experiment logging parts including wandb (must login with your own account)

---

### Start **w/** this repository

> Please follow the steps below when starting the project based on this repository.

1. Install IsaacGym ver.4 [isaac-gym page](https://developer.nvidia.com/isaac-gym)
2. Run `pip install -e .` in `rsl-rl/` directory
3. Run `pip install -e .` in `legged-gym/` directory
4. `ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory`
   - `export LD_LIBRARY_PATH=/home/jungyeon/anaconda3/envs/go2/lib`
5. `pip install tensorboard wandb opencv-python`
6. `AttributeError: module 'distutils' has no attribute 'version'`
   - `pip install setuptools==59.5.0`
   - (ref) https://github.com/pytorch/pytorch/issues/69894
4. Start Rough terrain locomotion learning with A1 (refer to table below)

#### Task Options

| Option | Config | Critic Obs | Actor Obs | Memo |
|--------|--------|:----------:|:---------:|------|
| `--task=a1_base` | A1RoughBaseCfg | 45 | 45 | observation without lin_vel |
| `--task=a1_oracle` | A1RoughOracleCfg | 238 | 238 | true_lin_vel + privileged(d,h) |
| `--task=a1_waq` | A1RoughBaseCfg | 238 | 64 | est_lin_vel + privileged / obs_history(timestep 5) |

---

### Start **w/** docker

> Please follow the steps below when starting via docker based on this repository.
> A driver supporting CUDA 12.1 or higher must be installed.

1. Download IsaacGym ver.4 [isaac-gym page](https://developer.nvidia.com/isaac-gym)
2. Move the downloaded `IsaacGym_Preview_4_Package.tar.gz` file to `asset/IsaacGym_Preview_4_Package.tar.gz`
3. Build docker with the following command:
   ```bash
   docker build . -t dreamwaq/dreamwaq -f docker/Dockerfile  --build-arg UID=$(id -u) --build-arg GID=$(id -g)
   ```
4. Run docker with the following command:
   ```bash
   docker run -ti --privileged -e DISPLAY=:0 -e TERM=xterm-256color -v /tmp/.X11-unix:/tmp/.X11-unix:ro --network host -v $PWD/dreamwaq:/home/user/dreamwaq --gpus all dreamwaq/dreamwaq /usr/bin/zsh
   ```

---

### Command

#### Training

```bash
python train.py --task=[TASK_NAME] --headless
```

- `--headless`: Option to run training without opening simulator window. Add this option when running on a server without display.

#### Inferencing

```bash
python play.py --task=[TASK_NAME] --load_run=[LOAD_FOLDER] --checkpoint=[CHECKPOINT_NUMBER]
```

| Parameter | Description | Example |
|-----------|-------------|---------|
| `[LOAD_FOLDER]` | Folder name inside `legged_gym/logs/[task folder]` | `Sep04_14-24-54_waq` |
| `[task folder]` | Task-specific log folder | `rough_a1/rough_a1_waq/rough_a1_est` |
| `[CHECKPOINT_NUMBER]` | Number of **model_[NUMBER].pt** file | `250` |

**Complete command example:**
```bash
python play.py --task=a1_waq --load_run=Sep04_14-24-54_waq --checkpoint=250
```

- Inferencing code to view a single agent up close: `mini_test.py` (options same as `play.py`)
- There are adjustable options in the main loop of each inferencing script, adjust True/False as needed.

#### Cross-computer Inference

If you want to inference a **model_[NUMBER].pt** file trained on a different computer:

| Step | Computer A (Training) | Computer B (Inferencing) |
|------|----------------------|--------------------------|
| 1 | - | Create a new folder named `FOLDER_NAME` in `legged_gym/logs/[task folder]` |
| 2 | Copy **model_[NUMBER].pt** | Paste to `FOLDER_NAME` |
| 3 | - | Run `python play.py --task=[TASK_NAME] --load_run=[FOLDER_NAME] --checkpoint=[NUMBER]` |

---

## Main Code Structure

- Explanation of important files in the project code. Files related to the robot platform and algorithms used in the project were selected. Please refer to the description next to each file name.
   - Robot platform used (environment): A1
   - Learning algorithm used: PPO

```
dreamwaq
│
├── legged_gym
│   ├── legged_gym
│   │   ├── envs
│   │   │   ├── __init__.py: Environment registration for training execution. Referenced by task_registry.
│   │   │   ├── a1/a1_config.py: Variable classes for A1 platform. Inherits from legged_robot_config.py classes.
│   │   │   └── base
│   │   │        ├── legged_robot.py: Base environment class for locomotion task. LeggedRobot Class
│   │   │        └── legged_robot_config.py: Variable classes for LeggedRobot. LeggedRobotCfg Class / LeggedRobotCfgPPO Class
│   │   ├── scripts
│   │   │   ├── train.py: Main training execution code. wandb settings setup. (Refer to Command-training)
│   │   │   ├── play.py: Code to check walking inference motion of multiple agents on various terrains after training. (Refer to Command-inference)
│   │   │   └── mini_test.py: Code to check walking inference motion of multiple agents on various terrains after training. (Refer to Command-inference)
│   │   └── utils
│   │       ├── logger.py: Code for matplotlib plot used in play.py and mini_test.py.
│   │       ├── task_registry.py: Connects environment and algorithm based on training environment info registered in envs/__init__.py.
│   │       └── terrain.py: Terrain class for walking. Referenced by LeggedRobot.
│   │
│   └── resources/robots/a1: Robot platform information (urdf&mesh)
│
└── rsl_rl
    └── rsl_rl
        ├── algorithms
        │   └── ppo.py: PPO algorithm code. Uses Actor/Critic classes from actor_critic.py.
        ├── modules
        │   └── actor_critic.py: Actor/Critic class code.
        ├── runners
        │   └── on_policy_runner.py: File containing OnPolicyRunner class with the main RL loop (learn function).
        │                            Base model uses OnPolicyRunner class, DreamWaQ model uses OnPolicyRunnerWaq class,
        │                            Estnet model uses OnPolicyRunnerEst class for training code execution.
        │                            (Classes are distinguished by modifications at the stage before the RL main loop [before actor/critic network stage])
        ├── utils
        │   └── rms.py: Running Mean Std class for CENet's normal prior distribution training.
        └── vae
            ├── cenet.py: Context-Aided Estimator Network (CENet) class.
            └── estnet.py: Estimator class for comparison model group.

```

---

## Result Graphs

Reward Graph for approximately 1000 iterations of training

![](./asset/two_models_rew.png)

### DreamWaQ model

- State plot of 1 robot agent after training
  - Row 1: Plot of x, y direction velocity and yaw direction command vs actual measured physical quantities from base state
  - Row 2: Plot of estimated velocity through CENet vs true velocity measured from simulator
  - Row 3: Error plot between estimated velocity and true velocity
    - Column 1: Squared error of each x, y, z direction component
    - Column 2, 3: Mean squared error of x, y directions

![](./asset/a1_waq_est_vel.png)

### Base model

- State plot of 1 robot agent after training (Unlike DreamWaQ, there is no estimated velocity, so the plotted graphs are different.)
  - Row 1: Plot of x, y direction velocity and yaw direction command vs actual measured physical quantities from base state
  - Row 2 Column 1/2: Position and velocity of 1 joint
  - Row 2 Column 3: Base z direction velocity
  - Row 3 Column 1: Contact force of 4 feet
  - Row 3 Column 2/3: Torque of 1 joint

![](./asset/a1_base_no_vel.png)

---

## Result Motions

> **Notice:** The videos below were recorded using the **A1 platform**. However, this repository also includes code for applying the algorithm to the **Go2 platform**.

### Walking Performance of a Reproduction Model in Different Terrains

- Smooth Slope / Rough Slope

![](./asset/1.gif)

- Stair Up / Stair Down

![](./asset/2.gif)

- Discrete / Mixed

![](./asset/3.gif)

---

### Comparative Analysis of Walking Motion Between the Reproduction Model and the Base Model

> small difference: naturalness of motion
>
> big difference: foot stuck / unstable step

- Smooth Slope(small difference)

![](./asset/4.gif)

- Rough Slope(small difference)

![](./asset/5.gif)

- Stair Up(big difference)

![](./asset/6.gif)

- Stair Down(big difference)

![](./asset/7.gif)

- Discrete(big difference)

![](./asset/8.gif)
