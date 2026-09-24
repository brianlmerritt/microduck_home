# Slope mode — `roller_slope` (balanced passive descent)

> English translation of [the French source](../../../../microduck_rl/docs/superpowers/specs/2026-07-22-roller-slope-design.md).
> Source repository: `microduck_rl`; source commit: `2b581c641406a48346e696212930ea881c222c52`.
> Translation date: 2026-09-17. This is a historical design document, not a fresh set of execution instructions. Its proposals, commands and reported results retain their original context.

## Current-source notes (separate from the translation)

- `Mjlab-RollerSlope-Flat-MicroDuck` is registered; see the [task configuration](../../../../microduck_rl/src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py).
- The source's statement that `Y` is free describes its design date. In the current [inference script](../../../../microduck_rl/scripts/infer_policy.py), `Y` serves sit/sitstand or slope according to the loaded auxiliary policy. The script accepts `--slope`.
- The assertions about other runtime software are retained as historical claims, not revalidated deployment instructions.

---

Date: 2026-07-22
Status: design approved, ready for the implementation plan.

## Objective

Train a dedicated policy in which **microduck (on rollers) starts on flat ground
with a small forward impulse, rolls to a descending ramp, and glides down to the
bottom while remaining upright and balanced**. No steering during the descent:
the policy's only objective is **not to fall**.

The policy must handle increasingly steep ramps (**~2° → ~20°**) through a
difficulty curriculum.

## Agreed decisions (brainstorming)

| Topic | Decision |
|---|---|
| Behaviour | Balanced passive descent (gravity moves it forwards, no prescribed pedalling) |
| Control | None—balance only, `twist` command forced to zero |
| Approach | **A**—dedicated, isolated task (like `roller_crouch`) |
| Terrain shape | **Simple ramp**: flat start + descending ramp (no pyramid) |
| Episode scenario | Spawn on flat ground → forward impulse velocity → glide down the ramp |
| Steepness | Curriculum **0/2° → 20°** |
| Deployment | `--slope <onnx>` flag + **`Y`** key in `infer_policy.py` (Y is free) |

## Architecture

### 1. New task

File: `src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py`, cloned from
`microduck_velocity_rollers_env_cfg.py`.

- Same roller robot (`MICRODUCK_WALK_ROLLERS_ROBOT_CFG`), same physics, same
  domain randomization / noise / delays.
- **Same 61D observation** (twist + head/body zero padding) → the policy
  loads through the runtime's `--new-cmd-obs` path and remains interchangeable
  with the other roller policies.
- Register in `src/mjlab_microduck/tasks/__init__.py` through
  `register_mjlab_task`, with a PPO configuration `MicroduckRollerSlopeRlCfg`
  (`experiment_name`/`run_name` = `roller_slope`).

### 2. Custom “flat + ramp” terrain

The sloping terrains provided by mjlab are pyramids, so write a dedicated
`SubTerrainCfg` (e.g. `FlatRampTerrainCfg`) whose
`function(difficulty, spec, rng)` method constructs:

- a **flat starting area** (~1–2 m long) where the robot spawns;
- a **descending ramp** after it, with its angle **interpolated by `difficulty`**
  over `[~2°, ~20°]`.

Set up the terrain through `TerrainEntityCfg(terrain_type="generator", ...)` with
a `TerrainGeneratorCfg` that generates several difficulty levels (and thus several
ramp angles). Each environment's origin must fall **on the flat area**, with the
ramp in front of it.

> Implementation risk to address in the plan: position the spawn origin on the
> flat area (not in the centre of the tile), and orient the ramp so that
> “forwards” = “downhill”.

### 3. Command = none

Neutralize the `twist` slot: `rel_standing_envs = 1.0`, velocity ranges at 0,
`rel_heading_envs = 0.0`. Head/body retain zero padding. The policy receives no
movement instruction.

### 4. Reset & impulse velocity

- `reset_base`: spawn at rest on the flat area, nominal roller height `z`
  (~`0.1335–0.1435`, as in the roller environment).
- Inject **entry velocity** through the `velocity_range` of
  `reset_root_state_uniform` (clean state + range), **not** through
  `push_by_setting_velocity` (which adds to the current state and can make the
  free joint diverge → NaN—a lesson already learned on `roller_crouch`):
  `x ≈ (0.2, 0.5) m/s` forwards.
- Retain light random pushes during the episode (robustness), as in the roller environment.

### 5. Rewards

Core objective: “stay upright + natural posture”, preventing a lazy optimum
(avoid collapsing onto the ground to maximize stability):

- `upright` (vertical trunk)—**main reward**.
- `alive` (survival bonus per step).
- **Nominal standing pose**: reward approaching the HOME pose (pose interpolation
  mechanics reused from `roller_crouch`, but with a fixed standing target),
  to retain a normal roller stance rather than a defensive crouch.
- `feet_flat` (rollers flat on the ground).
- `body_ang_vel`, `angular_momentum` (no shaking / twisting).
- `action_rate_l2`, `neck_action_rate_l2`, `joint_torques_l2`,
  `self_collisions` (smoothness + sim-to-real).

> No speed/braking reward: descent is passive. We do not reward “going fast”,
> only “staying upright while descending”.

### 6. Terminations

- **Fall:** `bad_orientation` (trunk tilted too far).
- **Bottom reached:** `out_of_terrain_bounds` (robot has reached the bottom of
  the ramp → reset).
- `nan_state`, timeout.

### 7. Difficulty curriculum (steepness)

Progression **gentle → steep**: start on almost-flat ramps, increase the angle
towards 20° as success improves.

> Implementation risk: the standard `terrain_levels_vel` curriculum promotes
> according to distance travelled relative to commanded velocity. Here the
> command is zero, so **a custom promotion criterion is required**: promote if
> the robot survived / reached the bottom without falling; demote if it falls early.

### 8. Deployment — `Y` button

In `scripts/infer_policy.py`:

- New `--slope <onnx>` flag loading the slope policy as an additional session
  (same pattern as `--walking` / `--standing` / `--ground-pick`).
- `GLFW_KEY_Y = 89` (currently **free**—the head uses `H`), which **toggles**
  the active session to/from the slope policy.
- Add a line to keyboard help.

No existing control is broken (unlike sharing the head-control `H` key).

## Out of scope (YAGNI)

- No left/right steering or braking while descending.
- No uphill or cross-slope travel.
- No pyramid or multidirectional terrain.
- No fine-tuning from existing roller weights (train from scratch).

## Deliverables

1. `microduck_roller_slope_env_cfg.py` (environment + `FlatRampTerrainCfg` + PPO configuration).
2. Task registration in `tasks/__init__.py`.
3. Necessary custom rewards/curriculum in `tasks/mdp.py` (standing pose,
   level promotion).
4. `--slope` + `Y` key integration in `scripts/infer_policy.py`.
5. Unit tests for pure functions (ramp angle by difficulty, promotion criterion if needed).
