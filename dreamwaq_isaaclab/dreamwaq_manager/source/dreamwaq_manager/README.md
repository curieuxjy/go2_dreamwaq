# DreamWaQ Manager (Isaac Lab extension)

DreamWaQ quadrupedal locomotion environments implemented with Isaac Lab's
**ManagerBasedRLEnv** workflow, for Isaac Lab 3.0.0-beta2 / Isaac Sim 6.0.

This directory is the installable extension. Install it (with an interpreter that has
Isaac Lab available) via:

```bash
python -m pip install -e source/dreamwaq_manager
```

Run a task from the project root:

```bash
python scripts/rsl_rl/train.py --task=DreamWaQ-Manager-Go2-Base-v0 --headless
```

Registered tasks: `DreamWaQ-Manager-Go2-{Base,Oracle,Waq}-v0` (+ `-Play-v0`).
See the project-level documentation for details.
