# Design — Roller Crouch-Glide (button-triggered “crouch while gliding”)

> English translation of [the French source](../../../../microduck_rl/docs/superpowers/specs/2026-07-17-roller-crouch-glide-design.md).
> Source repository: `microduck_rl`; source commit: `2b581c641406a48346e696212930ea881c222c52`.
> Translation date: 2026-09-17. This is a historical design document, not a fresh set of execution instructions. Its proposals, commands and reported results retain their original context.

## Current-source notes (separate from the translation)

- `Mjlab-RollerCrouch-Flat-MicroDuck` is registered. The [current configuration](../../../../microduck_rl/src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py) uses interpolated joint poses rather than the trunk-height objective proposed below, with `CROUCH_PERIOD = 5.0`, `DESCENT_END = 0.10`, `HOLD_END = 0.50`, `RISE_END = 0.60`, and action scale `1.0`.
- The historical text is internally inconsistent: its height diagram and hold calculation use a 4 s period, its runtime command specifies 5.0, and its parity note says to retain the 4.0 default. It also names training action scale `1.0` but gives deployment scale `0.8`. These values are preserved below rather than silently reconciled. Check the trained run's settings and actual runtime implementation before using a deployment command.
- Claims about the separately installed Rust runtime are historical source claims; they were not revalidated against a current runtime checkout for this translation.

---

**Date:** 2026-07-17
**Status:** design approved, ready for the implementation plan

## Context

The microduck robot can skate (roller policy, task `Mjlab-Velocity-Flat-MicroDuck-Rollers`).
We want a new movement: on a button press, it **crouches and continues gliding
on its momentum** (like a skater in a low position), holds for ~1 s, then **stands
back up** automatically and resumes skating.

A firm user constraint: **do not modify the Rust runtime**
(`apirrone/microduck_runtime`, installed as a binary). The movement must therefore
reuse a mechanism already present in the runtime.

**Key discovery:** the runtime already has a “button-triggered one-shot behaviour”
slot: `--ground-pick`. It is triggered by **button A** (rising edge), runs an ONNX
policy driven by a **phase** for a fixed duration, and then automatically returns
to the main policy. Crucially, it uses **exactly the same 61D observation layout**
as the roller policy—the two are interchangeable in the runtime. This is the ideal
mechanism, without writing a line of Rust.

Accepted compromise: the movement is **one-shot** (fixed duration, no “hold the
toggle” mode). The duration of the crouch is set by the slot's period.

## Selected approach (approach B)

Create a **new mjlab task** trained on the roller robot, performing
lower → glide while crouched → rise, driven by the ground-pick slot's phase.
Export it to ONNX and load it through `--ground-pick`. No Rust changes.

### Files involved

| File | Action |
|---|---|
| `src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py` | **New.** The environment, a roller + ground-pick hybrid. |
| `src/mjlab_microduck/tasks/mdp.py` | **Add** the `crouch_glide_height_by_phase` reward. |
| `src/mjlab_microduck/tasks/__init__.py` | **Add:** register `Mjlab-RollerCrouch-Flat-MicroDuck`. |

### Reuse (do not reinvent anything)

- **Physics / roller robot** ← `microduck_velocity_rollers_env_cfg.py`:
  `MICRODUCK_WALK_ROLLERS_ROBOT_CFG` (14 active joints + 4 passive wheels),
  contact sensor on the `roller_blade`, bearing-friction DR
  (`randomize_wheel_friction` + curriculum), 14-dimensional observations (wheels
  excluded through `SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))`),
  `action.scale=1.0`, `kp_fw=200`.
- **Phase / one-shot machinery** ← `microduck_ground_pick_env_cfg.py`:
  reuse the `microduck_mdp.GroundPickPhaseCommand` command **unchanged**
  (produces the `[cos(2πφ), sin(2πφ), 0]` that the runtime will send in the twist slot),
  zero head/body padding (`zero_command_padding`), `robot_state_is_nan`
  termination, `reset_action_history`.
- **Sim-to-real DR** ← taken unchanged from the roller environment (observation-level
  IMU misalignment, encoder bias, mass/inertia, BAM friction, armature, gentle ±0.2 pushes).

## The core: a phase-driven trapezoidal height target

The only genuinely new element. Instead of lowering the mouth (ground-pick), control
the **trunk height** (`com_height` of `trunk_base`) according to the phase, with a low plateau:

```text
height
 high  ┐                    ┌──   standing (hands back to the roller policy)
       │ \                 /
  low  │  \_______________/       crouch + glide (1 s plateau)
       └───────────────────────► phase
       0   0.375      0.625   1
```

- φ ∈ [0, 0.375]: lower towards the crouched height.
- φ ∈ [0.375, 0.625]: **hold the crouch** (= 1 s over a 4 s period) → glide.
- φ ∈ [0.625, 1.0]: rise towards the standing roller pose.

**New reward `crouch_glide_height_by_phase(env, command_name, height_low,
height_high, hold_lo=0.375, hold_hi=0.625, std=...)`** in `mdp.py`:
reads the phase from the command, calculates the target height (interpolated
high→low→high, flat on the plateau), rewards `exp(-((h_mesurée - h_cible)/std)²)`
(measured height minus target height).
Use `com_height_target` (mdp.py:694) and the existing
`interpolated/multistage height target` functions as references.

Starting values: `height_high ≈ 0.11` m (standing roller height; see the roller
`com_height_target` band 0.0935–0.1235), `height_low ≈ 0.075` m (crouched;
refine during playback). Reconstruct the phase from the command's `atan2(sin, cos)`.

## Rewards

| Reward | Role | Origin |
|---|---|---|
| `crouch_glide_height_by_phase` | Main target (high→low→high) | **new** |
| `wheel_speed` (reduced weight ~2–3) | Keep momentum, do not brake during the crouch | roller environment (`wheel_speed_reward`) |
| `upright` (≈2), `body_ang_vel` (−0.05), `angular_momentum` (−0.02) | Balance / stability | roller environment |
| `return_pose` (end of phase) | Converge towards the standing roller pose for a clean handover | adapted from `ground_pick_return_pose` |
| `feet_flat` (−2) | Flat blades → stable glide | roller environment |
| `action_rate_l2`, `neck_action_rate_l2`, `joint_torques_l2`, `self_collisions` | Smoothing / sim-to-real transfer | both environments |

**Explicitly NOT included:** `braking` (we do not want to stop), `mouth_ground_proximity`
/ `mouth_perpendicular_to_ground` (we do not touch the ground), `skating_air_time`
/ `single_support` / `glide` (no stride during the trick—we glide passively).

## Training

- `MicroduckRollerCrouchRlCfg` = copy of `MicroduckRollersRlCfg`
  (MLP 512/256/128, ELU, obs_normalization, PPO, `experiment_name="roller_crouch"`).
- Register in `tasks/__init__.py`:
  `register_mjlab_task(task_id="Mjlab-RollerCrouch-Flat-MicroDuck", ...)`.
- Run:
  ```bash
  uv run train Mjlab-RollerCrouch-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 8000
  ```
- Start episodes with a **realistic entry velocity** (the robot arrives rolling);
  otherwise it has no momentum to preserve during the crouch. Implement through
  a reset event (nonzero initial velocity) or a push at the start of the episode.

## Export + deployment (exact runtime flags)

Export ONNX (the normalizer is baked in by `export.py`), then:

```bash
microduck_runtime --variant pre-alpha --new-cmd-obs --roller \
  --model output.onnx \
  --new-dxl-imu --kp 200 --action-scale 0.8 \
  --max-linear-vel 0.6 --max-linear-vel-backward 0.5 --max-angular-vel 0.0 \
  --ground-pick roller_crouch.onnx \
  --ground-pick-period 5.0 \
  --ground-pick-kp-ratio 1.0 \
  --ground-pick-action-scale 0.8
```

Button **A** → crouch-glide, then automatically return to the roller policy.

**Training/deployment parity pitfalls (important for sim-to-real):**

- `--ground-pick-kp-ratio 1.0`: the default is **0.6** (lowers kp to 120 during the trick).
  Training uses kp=200 → force **1.0** to match.
- `--ground-pick-action-scale` must match the training `action_scale` (0.8 above).
- `--ground-pick-period 5.0` must match the trained period/movement length
  (default 4.0, which we retain).

## Risks and verification

- **One-shot, fixed duration:** the crouch lasts `ground-pick-period`, then rises
  automatically. No arbitrary hold—an accepted limitation of approach B.
- **Momentum during the trick:** the phase replaces the velocity command → **no active
  push** during the crouch. If entry momentum is too low, it slows down. Hence
  training with a realistic entry velocity.
- **Verification:**
  1. In simulation (`play`): it lowers, keeps the wheels turning during the plateau,
     rises without falling, and the final pose cleanly meets the standing roller pose.
  2. On the real robot: start at low speed, press A, observe.
  3. Confirm that the roller policy takes over cleanly after the return.

## Open questions / to confirm during implementation

- Exact `height_low` value (crouched)—adjust during playback.
- Best way to inject entry velocity into an episode (reset event vs initial push).
- Relative weight of `wheel_speed` vs `crouch_glide_height_by_phase` (keep momentum
  without preventing the crouch).
