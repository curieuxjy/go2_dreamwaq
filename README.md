# go2_walk
Demonstration Code for [Go2](https://www.unitree.com/go2/) Walking
- PPT : [RO](https://www.canva.com/design/DAGPeSPjAwk/5edJAwXttYySYyRlV8QsNg/edit?utm_content=DAGPeSPjAwk&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
- Code Structure
```
go2_walk
├── gym_basic
├── unitree_rl_gym
├── dreamwaq
│   ├── legged_gym
│   └── rsl_rl
└── wtw
    ├── go2_gym
    ├── go2_gym_learn
    └── go2_gym_deploy
```

# Part 0. `gym_basic`
> Go2 in IsaacGym
- IsaacGym ver.4

# Part 1. `unitree_rl_gym`
> Flat Terrain
- original repository: https://github.com/unitreerobotics/unitree_rl_gym
- From: https://github.com/leggedrobotics/legged_gym

# Part 2. `dreamwaq`
> Rough Terrain
- paper: https://arxiv.org/abs/2301.10602
  - fall recovery task: https://arxiv.org/abs/2306.12712
- Training Log: https://wandb.ai/curieuxjy/dreamwaq


https://github.com/user-attachments/assets/dc52a728-4099-4625-8688-c49ac8858327


# [FYA] Part 3. `wtw`
> Sim to Real
- original repository: https://github.com/Teddy-Liao/walk-these-ways-go2
- TODO: remove [ml-logger](https://ml-logger.readthedocs.io/en/latest/)


---

## Reference
- https://github.com/abizovnuralem/go2_omniverse
- https://isaac-sim.github.io/IsaacLab/source/setup/installation/binaries_installation.html
