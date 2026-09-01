# Notes — `dreamwaq_direct/` (DirectRLEnv)

This stack is a 1:1 port of the original IsaacGym DreamWaQ onto the `DirectRLEnv` API, and serves as
a cross-check of [`../dreamwaq_manager/`](../dreamwaq_manager/). Both stacks now behave the same at
scale. This document records the one non-obvious bug that had to be fixed to get there.

## Termination is identical across both stacks and matches the original

| | Termination |
|---|---|
| Original DreamWaQ (`legged_gym/envs/base/legged_robot.py`, `check_termination`) | `any(norm(contact_forces[:, termination_contact_indices, :]) > 1.0)` with `terminate_after_contacts_on = ["base"]` |
| `dreamwaq_manager` | `mdp.illegal_contact`, `body_names="base"`, `threshold=1.0` |
| `dreamwaq_direct` | the same expression inline in `_get_dones` (`terminate_after_contacts_on=["base"]`, `termination_contact_force=1.0`) |

Both contact sensors use `history_length=3`, so both take the max over history before thresholding.
Timeout is separate and is a truncation, not a failure — as in the original. There is no
orientation/tilt termination in either stack.

## Fixed: inter-environment collisions were never filtered on GPU

**Symptom.** On generated (rough) terrain at 4096 envs, the PhysX contact sensor reported large
forces on non-contacting bodies — e.g. ~400 N on the base while the robot floated 1 m above the
ground. With contact termination that pins mean episode length at **1.00** with ~3920/4096 envs
terminating on `base_contact` every step. Flat terrain (`terrain_type="plane"`) and small env counts
(16, 64) did *not* show it, which is what made this hard to pin down.

**Cause.** `DreamWaQEnv._setup_scene` had copied the official Direct example pattern:

```python
self.scene.clone_environments(copy_from_source=False)
if self.device == "cpu":                                  # <-- the bug
    self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
```

so on GPU — i.e. every real training run — inter-environment collisions were **never filtered**.
`InteractiveScene`, which the manager-based stack uses, has no such guard: it filters for any PhysX
backend regardless of device (`interactive_scene.py`: `if self.cfg.filter_collisions and "physx" in
self.physics_backend`). That is exactly why the manager stack was fine at the same scale, on the
same terrain, with the same `ContactSensor` config.

**Fix.** Drop the `device == "cpu"` guard so filtering always runs. One line.

**Verified** (`DreamWaQ-Direct-Go2-*-v0`, rough terrain, 4096 envs, 20 iterations, mean episode
length):

| Task | before | after |
|---|---|---|
| Base | 1.00 | 281.8 → 298.6 |
| Oracle | 1.00 | 274.6 → 283.8 |
| Waq | 1.00 | 248.4 → 282.8 |
| *Manager Base (control)* | *265.7 → 256.3* | *unchanged* |

**Note for anyone hitting this elsewhere.** The `device == "cpu"` guard is present in the official
Isaac Lab Direct examples too, and `Isaac-Velocity-Rough-Anymal-C-Direct-v0` reproduces the same
`ep_len = 1` collapse on this build. This was previously recorded in this project as an
unfixable upstream PhysX bug and was the stated reason for deprecating `dreamwaq_direct`; that
diagnosis was wrong. Worth reporting upstream.

## Side fixes from the same investigation

- `gpu_max_rigid_patch_count` raised `50 → 400 × 2^15` (PhysX patch buffer overflow at 4096 envs on
  complex terrain). Real, but a separate problem — it did not fix the phantom contacts.
- `terrain_generator.curriculum = True` is forced in `__post_init__` to avoid an
  `UnboundLocalError` in `trimesh.util.concatenate` inside `_generate_random_terrains`.
  The ManagerBased version avoids this via `CurriculumCfg`.
- Foot indices are resolved with `self._robot.find_bodies()` — the contact sensor's body ordering
  differs from the articulation's, and mixing them computed `foot_clearance` on the wrong bodies.
  Termination bodies are resolved separately with `self._contact_sensor.find_sensors()`.

## Termination logging units are unified too

`Episode_Termination/*` used to be a **raw count** of envs here and a **normalized rate** in the
manager stack, so the numbers were not comparable. `dreamwaq_direct` now mirrors Isaac Lab's
`TerminationManager`: it keeps `_last_episode_dones` (which term ended each env's most recent
episode) and logs `mean(dim=0)` — the fraction of envs per term, in [0, 1].

Measured at 1024 envs, 15 iterations, Base:

| | `base_contact` | `time_out` |
|---|---|---|
| `dreamwaq_direct` | 0.480 | 0.180 |
| `dreamwaq_manager` | 0.591 | 0.153 |
