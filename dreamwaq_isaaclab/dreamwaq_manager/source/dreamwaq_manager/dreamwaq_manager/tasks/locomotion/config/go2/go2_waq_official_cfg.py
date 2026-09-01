# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CENet(DreamWaQ) on the OFFICIAL Go2 velocity envs — the "unified rewards" fallback.

Last-resort comparison experiment: take the official Go2 velocity env wholesale (rewards,
events/push, terminations — the recipe that demonstrably walks; SAC reached underlying
tracking 0.96 on flat), keep ONLY DreamWaQ's core idea (a blind actor + CENet velocity/context
estimation), and retrain. This isolates the CENet contribution from the DreamWaQ env design
(harsh 1 s/±1 m/s push, DreamWaQ reward set), which is suspected of capping tracking quality.

Observation design:

* ``policy`` (actor, 45 dims): the official policy group with ``base_lin_vel`` and
  ``height_scan`` removed — composition and ordering then exactly match DreamWaQ's base obs
  (ang_vel 3 + gravity 3 + commands 3 + joint_pos 12 + joint_vel 12 + actions 12).
  The Waq runners augment this to 64 dims with CENet outputs.
* ``critic`` (privileged): the full official policy content (incl. ``base_lin_vel``; rough
  also keeps ``height_scan``), without observation corruption.
"""

from isaaclab.utils import configclass  # noqa: keep BEFORE terrain imports (submodule name clash)

import os

import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg as OfficialObservationsCfg


# Harder terrain emphasizing STAIRS (steeper) and GAPS — where blind locomotion struggles and
# exteroception (height_scan / ++) should help. Curriculum-sorted (difficulty increases by row).
HARD_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.30, step_height_range=(0.12, 0.32), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25, step_height_range=(0.12, 0.32), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.20, gap_width_range=(0.1, 0.5), platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.02, 0.12), noise_step=0.02, border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.10, grid_width=0.45, grid_height_range=(0.05, 0.25), platform_width=2.0,
        ),
    },
)


# Moderated hard terrain: within Go2's physical capability so the curriculum can actually promote —
# climbable stairs (0.08-0.20 m) and steppable GAPS (0.1-0.3 m). Gaps are the key discriminator:
# a blind robot cannot detect them (steps in and falls), a perceptive one sees and adjusts.
# Pair with EASY-START (max_init_terrain_level=0) for a fair blind-vs-perception comparison.
MODERATE_HARD_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.35, gap_width_range=(0.1, 0.3), platform_width=2.0,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25, step_height_range=(0.08, 0.20), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20, step_height_range=(0.08, 0.20), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25,
        ),
    },
)


def _make_oracle_velocity_exact(env_cfg):
    """Strip observation noise from the Oracle actor's ``base_lin_vel`` — and only from it.

    Oracle is defined (CLAUDE.md, PAPER.md) as "Base + the TRUE base linear velocity": the
    privileged upper bound that Waq's CENet is trying to approach. But the official policy group
    ships ``base_lin_vel`` with ``Unoise(-0.1, 0.1)``, so as written the Oracle actor was reading a
    *corrupted* velocity while the Waq actor, on its AdaBoot true-velocity draws, reads
    ``scene["robot"].data.root_lin_vel_b`` straight from the simulator with no noise at all (and
    CENet is trained against that same clean target). The arm meant to be the ceiling therefore had
    a strictly worse velocity signal than the arm it is supposed to bound.

    The other 44 terms keep their noise. Oracle's privilege is "knows the velocity exactly", not
    "has clean proprioception" — removing all noise would change the sensing difficulty of the
    task and stop being comparable to Base/Waq.
    """
    env_cfg.observations.policy.base_lin_vel.noise = None


def _make_blind_policy_and_critic(env_cfg, keep_height_scan_in_critic: bool):
    """Drop lin_vel (+height_scan) from the actor obs; attach an uncorrupted privileged critic group."""
    # Blind actor: 45-dim DreamWaQ base obs
    env_cfg.observations.policy.base_lin_vel = None
    if env_cfg.observations.policy.height_scan is not None:
        env_cfg.observations.policy.height_scan = None

    # Privileged critic: official policy terms, no corruption
    critic = OfficialObservationsCfg.PolicyCfg()
    critic.enable_corruption = False
    if not keep_height_scan_in_critic:
        critic.height_scan = None
    env_cfg.observations.critic = critic


def _add_paper_rewards(env_cfg, has_height_scanner: bool):
    """Add the five DreamWaQ Table-I reward terms the official Go2 recipe does not have.

    PAPER.md §2: of the paper's 12 terms the official recipe is missing 7·8·9·11·12 —
    joint power, body height, foot clearance, smoothness and power distribution. Weights are
    the paper's; the implementations live in ``dreamwaq_manager...mdp.rewards`` and are the
    same objects the DreamWaQ recipe (``velocity_env_cfg.RewardsCfg``) uses, so term names and
    parameters are kept identical to that config for wandb/tensorboard tag continuity.

    Must be applied to EVERY arm (Base / Oracle / Waq, flat and rough) — the three arms differ
    only in actor observation, so a reward set on one arm alone would break the comparison.

    Args:
        has_height_scanner: rough envs carry the base-mounted ``height_scanner``, so foot and
            base height are measured relative to the terrain under the robot. The flat env
            deletes the sensor; there we fall back to world Z, which is exact on a plane at z=0.
    """
    # TEMPORARY diagnostic gate (CENet posterior-collapse regression, logs/_cenet_regression/):
    # reverts this arm to the pre-§8-4 reward set so the collapse can be attributed.
    if os.environ.get("DWQ_NO_PAPER_REW"):
        print("[DIAG] DWQ_NO_PAPER_REW=1 -> paper reward terms NOT added")
        return

    # Imported here for the same reason the terrain import below is deferred: a module-level
    # import of the package `mdp` shim pulls in isaaclab.utils.configclass as a *submodule* and
    # shadows the decorator bound at the top of this file.
    from dreamwaq_manager.tasks.locomotion import mdp

    rewards = env_cfg.rewards
    rewards.joint_power = RewTerm(func=mdp.joint_power_l1, weight=-2.0e-5)
    # weight 0.0 = MEASURED, not assumed. The r1/r2 NaN collapses were base_height's inf, not this
    # term, so the term was re-tested against safe base_height at BOTH candidate weights, one 1500-it
    # rough-Base run each, everything else identical to r3 (PAPER.md §8-4, seed 42, last-10% mean):
    #   r3  weight  0        underlying 0.750  terrain 5.56   <- best
    #   p4  no paper rewards at all      underlying 0.725  terrain 5.35   (baseline)
    #   r5  weight -1.0e-6   underlying 0.707  terrain 5.45   (author's IsaacGym reference value)
    #   r4  weight -1.0e-5   underlying 0.626  terrain 4.61   (paper Table I printed value)
    # Monotone in the weight and separated at every stage of training, not just the tail — the term
    # costs tracking here even at reference scale, where its episodic reward (-0.049) is the same
    # order as smoothness (-0.079). Single seed; re-open if a seed sweep says otherwise.
    # NOTE this is a deliberate deviation from the paper, and the metric it is judged on cannot see
    # what the term is for: the paper introduces it to keep individual motors from OVERHEATING on
    # real hardware (§III-G reports exactly that failure outdoors). Nothing overheats in sim, so
    # dropping it is free here and would not be on a robot. Turn it back on for any sim2real work.
    rewards.power_distribution = RewTerm(func=mdp.power_distribution_l2, weight=0.0)
    rewards.smoothness = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    rewards.foot_clearance = RewTerm(
        func=mdp.foot_clearance_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "height_sensor_cfg": SceneEntityCfg("height_scanner") if has_height_scanner else None,
            "desired_height": 0.12,
        },
    )
    rewards.base_height = RewTerm(
        # NOT isaaclab's base_height_l2 — that one averages raw ray hits and a single ray-miss
        # (inf) NaN-kills the PPO batch. See the docstring of base_height_l2_safe.
        func=mdp.base_height_l2_safe,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner") if has_height_scanner else None,
            "target_height": 0.30,
        },
    )
    # NOT restored here: `flat_orientation_l2` (paper -0.2). The official recipe already has the
    # term — 0.0 on rough, -2.5 on flat — so this is a weight-tuning question, not a missing term,
    # and matching the paper on flat would mean WEAKENING -2.5 to -0.2. Same class of decision as
    # the tracking weights (official 1.5/0.75 vs paper 1.0/0.5), which are deliberately untouched:
    # one variable at a time. Recorded as an open item in PAPER.md §8.


def _add_paper_randomization(env_cfg):
    """Restore the DreamWaQ Table-II domain randomization the official Go2 recipe leaves out.

    PAPER.md §6: the paper's Table II and the authors' ``a1_config.py::domain_rand`` agree on
    every row, so unlike the reward set there is no paper-vs-implementation dispute here — the
    official Isaac Lab ``EventsCfg`` (``velocity_env_cfg.py``) simply randomizes much less.
    The values below are the same ones the DreamWaQ recipe (``velocity_env_cfg.EventCfg`` in
    this package) already uses; this function ports them onto the official recipe.

    ==================  ==========================  ================================
    item                paper / authors             official recipe (before)
    ==================  ==========================  ================================
    friction            [0.2, 1.25]                 fixed 0.8 static / 0.6 dynamic
    Kp, Kd factor       x[0.9, 1.1] each            (absent)
    motor strength      x[0.9, 1.1]                 (absent)
    payload             [-1, +2] kg, ADDITIVE       x[1/1.25, 1.25] log-uniform
    CoM shift           +-50 mm, EVERY link         base only, xy +-50 mm, z +-10 mm
    push                every 1.0 s, +-1.0 m/s      every 10-15 s, +-0.5 m/s
    system delay        [0, 15] ms                  (absent)
    ==================  ==========================  ================================

    Applied to EVERY arm (Base / Oracle / Waq, flat and rough) for the same reason as
    :func:`_add_paper_rewards` — the arms differ only in actor observation.

    Two rows are deliberately NOT in the always-on set:

    * **push** is a 10-30x difficulty change on its own, so it is gated behind
      ``DWQ_PAPER_PUSH=1`` and measured as a separate variable rather than bundled in.
    * **system delay** is not implemented on the Manager stack at all (there is no
      observation-delay plumbing; the Direct stack's ``system_delay`` flag is the reference).
      Still open — see PAPER.md §6.
    """
    # TEMPORARY diagnostic gate (CENet posterior-collapse regression, logs/_cenet_regression/):
    # reverts this arm to the pre-§8-5 official randomization so the collapse can be attributed.
    if os.environ.get("DWQ_NO_PAPER_RAND"):
        print("[DIAG] DWQ_NO_PAPER_RAND=1 -> paper domain randomization NOT applied")
        return

    from dreamwaq_manager.tasks.locomotion import mdp

    events = env_cfg.events

    # -- friction [0.2, 1.25] on every body (official: fixed 0.8 / 0.6).
    # make_consistent keeps dynamic <= static, which PhysX expects; legged_gym has a single
    # friction coefficient per shape, so the pair here is the closest Isaac Lab equivalent.
    events.physics_material.params["static_friction_range"] = (0.2, 1.25)
    events.physics_material.params["dynamic_friction_range"] = (0.2, 1.25)
    events.physics_material.params["make_consistent"] = True

    # -- payload: additive [-1, +2] kg, i.e. asymmetric and in absolute kg, not a ratio
    # (`props[0].mass += uniform(-1.0, 2.0)` in _process_rigid_body_props).
    events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)
    events.add_base_mass.params["operation"] = "add"
    events.add_base_mass.params["distribution"] = "uniform"

    # -- CoM +-50 mm on EVERY link and on z too (official: base link only, z only +-10 mm).
    # The official `base_com` is a `preset()` whose alternatives are only resolved once the
    # physics backend is known, so its `.params` are not reachable from here — and mutating the
    # preset in place would hit the shared CLASS attribute that `_preset_fields` prefers over the
    # instance. Build a fresh preset instead (each `preset()` call makes its own class).
    from isaaclab_tasks.utils import preset

    events.base_com = preset(
        default=EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        ),
        newton_mjwarp=None,  # same carve-out the official cfg has (CoM writes destabilize mjwarp)
    )

    # -- Kp / Kd factors, independently sampled per env and per joint.
    # startup (not reset) mode: randomize_actuator_gains re-derives from the cached DEFAULT
    # gains on every call, so a reset-mode term here would silently wipe the motor-strength
    # factor applied below. Both terms therefore run once, in declaration order.
    events.randomize_pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    # -- motor strength x[0.9, 1.1]. Isaac Lab has no torque-scaling event; for the Go2's
    # DCMotor (tau = Kp*e_pos + Kd*e_vel, then clipped) multiplying tau by m is exactly
    # multiplying both gains by m, which is what this local term does. Must come AFTER
    # randomize_pd_gains — it multiplies on top of whatever the gains currently are.
    events.randomize_motor_strength = EventTerm(
        func=mdp.randomize_motor_strength,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "strength_range": (0.9, 1.1),
        },
    )

    # -- push: SEPARATE VARIABLE, off by default. legged_gym pushes every 1.0 s with a +-1 m/s
    # additive impulse on lin vel x/y/z; the official recipe pushes every 10-15 s at +-0.5 m/s
    # on xy only. That is 10-30x more disturbance, i.e. a large difficulty change that would
    # confound the rest of this function if switched on at the same time.
    if os.environ.get("DWQ_PAPER_PUSH"):
        events.push_robot.interval_range_s = (1.0, 1.0)
        events.push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)}


@configclass
class WaqOfficialFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    """Official Go2 flat env + blind actor / privileged critic for CENet (policy 45, critic 48)."""

    def __post_init__(self):
        super().__post_init__()
        # flat env already removed the height scanner; the critic must not reference it either
        _make_blind_policy_and_critic(self, keep_height_scan_in_critic=False)
        _add_paper_rewards(self, has_height_scanner=False)
        _add_paper_randomization(self)


class WaqOfficialFlatEnvCfg_PLAY(WaqOfficialFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class WaqOfficialRoughEnvCfg(UnitreeGo2RoughEnvCfg):
    """Official Go2 rough env + blind actor / privileged critic for CENet (policy 45, critic 235)."""

    def __post_init__(self):
        super().__post_init__()
        _make_blind_policy_and_critic(self, keep_height_scan_in_critic=True)
        # Optional ablation (DWQ_EQUAL_TERRAIN=1). Mutates the generator that
        # UnitreeGo2RoughEnvCfg already scaled down for the Go2 — do NOT assign a fresh
        # ROUGH_TERRAINS_CFG copy here, it would wipe that scaling and curriculum=True.
        # Imported inside the method: a top-level import pulls in isaaclab.terrains.config.rough,
        # which loads the isaaclab.utils.configclass *submodule* and shadows the decorator this
        # file binds at line 24 (see the note there).
        from dreamwaq_manager.tasks.locomotion.terrains import apply_equal_proportions

        apply_equal_proportions(self.scene.terrain.terrain_generator)
        _add_paper_rewards(self, has_height_scanner=True)
        _add_paper_randomization(self)


@configclass
class OracleOfficialFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    """Official Go2 flat env, DreamWaQ-Oracle observation split (single 48-dim policy group).

    The actor layout is the stock official flat one (policy includes ``base_lin_vel``, no
    height scan). Before the paper rewards were added, the Oracle-Flat task pointed straight at
    ``UnitreeGo2FlatEnvCfg``; it must not, or the Base/Waq flat arms would carry rewards and
    randomization this one lacks.

    It must also declare a ``critic`` group of its own. Inheriting the stock flat cfg means
    inheriting its *absence*, and with no critic group rsl_rl trains the value function on the
    ``policy`` group — which has ``enable_corruption=True``. Base and Waq get a clean privileged
    critic from :func:`_make_blind_policy_and_critic`, so an Oracle without one is the only arm
    of the six whose value function sees observation noise. That is not the asymmetric
    actor-critic the comparison is supposed to isolate; it is a handicap on the arm that is
    meant to be the upper bound. Measured: the 2026-08-18 sweep put flat Oracle at 0.8920 —
    tied with blind Base (0.8917) and BELOW Waq (0.9124) — while rough Oracle, which has always
    declared its critic, led its arm by +0.053. See PAPER.md §3.
    """

    def __post_init__(self):
        super().__post_init__()
        _add_paper_rewards(self, has_height_scanner=False)
        _add_paper_randomization(self)
        _make_oracle_velocity_exact(self)
        # Same content as the actor here (Oracle already sees base_lin_vel; flat has no height
        # scan), so the only thing this buys is the missing `enable_corruption = False`.
        critic = OfficialObservationsCfg.PolicyCfg()
        critic.enable_corruption = False
        critic.height_scan = None
        self.observations.critic = critic


class OracleOfficialFlatEnvCfg_PLAY(OracleOfficialFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class OracleOfficialRoughEnvCfg(UnitreeGo2RoughEnvCfg):
    """Official Go2 rough env, DreamWaQ-Oracle observation split (policy 48, critic 235).

    Oracle in DreamWaQ terms: the actor sees the TRUE base linear velocity but remains
    terrain-blind (no height scan); the critic keeps the full privileged view. This separates
    the "velocity information" axis (Oracle vs Base vs Waq) from terrain awareness — the
    official default (height scan in the actor) is reported separately as a terrain-aware
    reference, not as Oracle.
    """

    def __post_init__(self):
        super().__post_init__()
        # Same ablation hook as WaqOfficialRoughEnvCfg — must match it or the Base/Waq/Oracle
        # comparison is confounded by terrain composition. In-place, for the reason noted there.
        from dreamwaq_manager.tasks.locomotion.terrains import apply_equal_proportions

        apply_equal_proportions(self.scene.terrain.terrain_generator)
        _add_paper_rewards(self, has_height_scanner=True)
        _add_paper_randomization(self)
        # actor: official policy terms minus height_scan -> lin_vel(3)+45 = 48 dims
        self.observations.policy.height_scan = None
        _make_oracle_velocity_exact(self)
        # critic: full official policy content incl height_scan, no corruption
        critic = OfficialObservationsCfg.PolicyCfg()
        critic.enable_corruption = False
        self.observations.critic = critic


@configclass
class TerrainOracleOfficialRoughEnvCfg(UnitreeGo2RoughEnvCfg):
    """THE PAPER'S ORACLE: an actor that sees the terrain height map (policy 45+187, critic 235).

    DreamWaQ's headline claim is about THIS arm, not about our velocity Oracle. Paper Fig.3
    caption: "The oracle policy has access to the height map measurement of the robot's
    surroundings as in [12]", and §III-B: "despite walking without exteroception, DreamWaQ
    performs almost as well as the oracle policy that has direct access to the surrounding
    terrain's height map."

    Our `OracleOfficialRoughEnvCfg` is a *velocity* oracle (45 + true base lin_vel). The paper
    makes no claim about that arm, so `Oracle - Waq` was never the number the paper's claim is
    about. This class supplies the missing comparison: blind Waq vs terrain-aware policy.

    Observation-wise this is simply the stock official Go2 rough policy group (which already
    carries `height_scan`) minus the true base lin_vel, so the only thing it has over Base is
    exteroception. Everything else -- rewards, randomization, terrain, critic -- matches the
    other arms.
    """

    def __post_init__(self):
        super().__post_init__()
        from dreamwaq_manager.tasks.locomotion.terrains import apply_equal_proportions

        apply_equal_proportions(self.scene.terrain.terrain_generator)
        _add_paper_rewards(self, has_height_scanner=True)
        _add_paper_randomization(self)
        # actor: 45 proprio + 187 height_scan. Drop base_lin_vel so this arm differs from Base
        # ONLY by exteroception (otherwise it would be "terrain AND velocity" oracle).
        self.observations.policy.base_lin_vel = None
        # critic: same privileged view as every other arm.
        critic = OfficialObservationsCfg.PolicyCfg()
        critic.enable_corruption = False
        self.observations.critic = critic


class TerrainOracleOfficialRoughEnvCfg_PLAY(TerrainOracleOfficialRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


class OracleOfficialRoughEnvCfg_PLAY(OracleOfficialRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


class WaqOfficialRoughEnvCfg_PLAY(WaqOfficialRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
