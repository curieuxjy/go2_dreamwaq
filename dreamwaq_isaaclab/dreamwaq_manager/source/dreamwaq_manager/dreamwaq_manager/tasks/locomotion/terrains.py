# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared terrain helpers.

Deliberately tiny and free of ``isaaclab.utils`` imports. ``isaaclab.utils.configclass`` is both a
submodule and a re-exported decorator, so ``from isaaclab.utils import configclass`` binds the
module instead of the function whenever the submodule happens to be imported first. Keeping this
helper in its own module lets any cfg import it without perturbing that ordering.
"""

import os


def apply_equal_proportions(terrain_generator) -> None:
    """Equalize sub-terrain proportions **in place**, only when ``DWQ_EQUAL_TERRAIN=1``.

    Off by default: the stock Isaac Lab mix (stairs up/down, boxes, random_rough at 0.2 each; the
    two slopes at 0.1) is what every reported result uses.

    **Mutates the generator that is already attached to the cfg — never replace it.**
    ``UnitreeGo2RoughEnvCfg.__post_init__`` scales the sub-terrains down because the Go2 is small
    (``boxes.grid_height_range`` 0.05→0.025, ``random_rough.noise_range`` 0.02→0.01,
    ``noise_step`` 0.02→0.01) and ``LocomotionVelocityRoughEnvCfg.__post_init__`` sets
    ``curriculum = True`` when the ``terrain_levels`` curriculum term exists. Assigning a fresh
    ``ROUGH_TERRAINS_CFG.copy()`` after ``super().__post_init__()`` silently throws all of that
    away — the terrain becomes ~2x harder AND the difficulty-sorted rows turn off, which breaks the
    terrain-level curriculum entirely. That mistake was made once; don't repeat it.

    Why equalizing is an ablation and not the default: a sub-terrain's proportion decides how many
    *columns* it gets, and ``TerrainImporter`` pins each env to one column for the whole run
    (``terrain_types = arange(num_envs) // (num_envs/num_cols)`` never changes) — the curriculum
    only moves an env up and down in *difficulty*, never across types. So the proportion is the
    fraction of envs permanently assigned to that terrain, and equalizing raises the two slope
    terrains from 20% to 33% of envs combined.
    """
    if os.environ.get("DWQ_EQUAL_TERRAIN") != "1":
        return
    share = 1.0 / len(terrain_generator.sub_terrains)
    for sub in terrain_generator.sub_terrains.values():
        sub.proportion = share
    print(f"[INFO] DWQ_EQUAL_TERRAIN=1 → sub-terrain proportions equalized to {share:.4f} each")
