# DreamWaQ Direct (Isaac Lab extension)

DreamWaQ quadrupedal locomotion environments implemented with Isaac Lab's
**DirectRLEnv** workflow, for Isaac Lab 3.0.0-beta2 / Isaac Sim 6.0.

This directory is the installable extension. Install it (with an interpreter that has
Isaac Lab available) via:

```bash
python -m pip install -e source/dreamwaq_direct
```

Run a task from the project root:

```bash
python scripts/rsl_rl/train.py --task=DreamWaQ-Direct-Go2-Base-v0 --headless
```

Registered tasks: `DreamWaQ-Direct-Go2-{Base,Oracle,Waq}-v0` (+ `-Play-v0`).
See the project-level documentation for details.
