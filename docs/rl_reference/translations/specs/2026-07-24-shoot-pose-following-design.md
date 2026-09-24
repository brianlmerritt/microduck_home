# Specification — “Kick a ball” RL task through pose tracking

> English translation of [the French source](../../../../microduck_rl/docs/superpowers/specs/2026-07-24-shoot-pose-following-design.md).
> Source repository: `microduck_rl`; source commit: `2b581c641406a48346e696212930ea881c222c52`.
> Translation date: 2026-09-17. This is a historical design document, not a fresh set of execution instructions. Its proposals, commands and reported results retain their original context.

## Current-source notes (separate from the translation)

- `Mjlab-Shoot-Flat-MicroDuck` is **not registered** in the current [task registry](../../../../microduck_rl/src/mjlab_microduck/tasks/__init__.py). The available `Mjlab-BallKick-Flat-MicroDuck` is a distinct [task with a simulated ball](../../../../microduck_rl/src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py), not proof that this no-ball pose-tracking design was implemented as written.
- The source contains revisions alongside earlier defaults: the learned weight-transfer section specifies gesture tracking with `std 0.35` and separate support-leg tracking, while the later original table still says `std=0.4`. Its references to tables “above” are retained even where those tables appear later. These are historical-document inconsistencies, not settings to combine into a new task without review.
- Runtime deployment claims below were not revalidated against a current Rust runtime checkout.

---

**Date:** 2026-07-24
**Branch:** `new_pre_alpha_ground_pick`
**Task id:** `Mjlab-Shoot-Flat-MicroDuck`

## Objective

Learn a **one-shot kick** (striking a ball) by **tracking a trajectory of joint
poses with 4 keyframes**, interpolated by phase:

```text
STAND → PIED_ARRIÈRE (wind-up) → PIED_AVANT (strike) → STAND (rest)
```

Here `PIED_ARRIÈRE` means “foot back” and `PIED_AVANT` means “foot forwards”.

- The **right leg** kicks; the **left leg** supports.
- **No simulated ball:** learn the *movement* through pose tracking (like
  `ground_pick` / crouch). If a real ball is in front of the robot at deployment,
  it gets struck.
- **Unified 61D observations**, identical to the other microduck policies → the
  exported ONNX deploys unchanged in a runtime **button slot** (one-shot: play
  the movement, then hand back to the main policy).

Same template as this branch's `ground_pick` task (phase encoded as `[cos, sin, 0]`
in the twist slot, phase-driven pose tracking, 61D observations, sim-to-real DR
inherited from velocity).

## Non-goals (YAGNI)

- No physical ball, no ball-contact/velocity reward.
- No configurable side (right only; left can be mirrored later if needed).
- No walking / fall recovery: remove all locomotion terms.

## Architecture

### File & registration

- `src/mjlab_microduck/tasks/microduck_shoot_env_cfg.py`
  - `make_microduck_shoot_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg`
  - `MicroduckShootRlCfg` (RslRlOnPolicyRunnerCfg, `experiment_name="shoot"`)
- Register in `src/mjlab_microduck/tasks/__init__.py`:
  `Mjlab-Shoot-Flat-MicroDuck` (optional `-Rough-` variant).
- Base: inherit from the velocity environment (through `make_velocity_env_cfg`,
  as ground_pick does), then aggressively strip everything related to locomotion.
- Robot: `MICRODUCK_WALK_ROBOT_CFG` (standard walking, 14 joints, no rollers).
- `action.scale = 1.0`.

### Poses (placeholders → read from the real robot using `read_pose.py`)

Dictionaries `{nom_joint: rad}`, **14 joints** (mouth excluded). At the top of the environment file.

- `STAND_POSE`: neutral stance (~simulation HOME).
- `KICK_BACK_POSE`: right hip in **backward extension** + bent right knee
  (wind-up); left leg + neck ≈ HOME.
- `KICK_FWD_POSE`: right hip **flexed forwards** + straight right knee (strike);
  left leg + neck ≈ HOME.

Start with plausible placeholders (adjustable), to be replaced by real readings.

### Command & phase

- Reuse `GroundPickPhaseCommand`: `command = [cos(2π·φ), sin(2π·φ), 0]`
  in the twist slot.
- **Period:** `SHOOT_PERIOD ≈ 2.5 s` (configurable through `cfg.period`).
- **New `randomize_phase` flag** on `GroundPickPhaseCommandCfg` /
  `GroundPickPhaseCommand`:
  - Default `True` (non-breaking: `ground_pick` keeps its current behaviour).
  - Shoot sets it to `False` → `reset()` resets φ=0 instead of `rand()`.
  - Reason: every episode starts at STAND (robot state = `default_joint_pos`)
    with φ=0 = STAND target → state/target consistency at reset (otherwise the
    policy is asked to reach a “strike” pose instantly from a stationary stance).
  - **Consistency invariant:** `STAND_POSE` MUST equal the simulator's joint
    reset pose (`HOME_FRAME` / `default_joint_pos`, nonzero: hip_pitch ±0.4579,
    ankle ±0.4530, hip_roll ±0.0873, neck/head_pitch 0.3491). Checked by
    `test_stand_pose_matches_home_standing_pose`. The initially zero placeholders
    broke this invariant (corrected after final review).

### Reset (standing height, no momentum)

- `reset_base.pose_range.z = (0.12, 0.13)`—**absolute standing height** (the
  default root `pos` in `InitialStateCfg` is (0,0,0), so reset z = 0.12–0.13 m,
  not an additive offset; identical to the working velocity environment). No fall.
- **No entry-velocity injection** (stationary kick, unlike crouch-glide).

### Unlisted inherited rewards

The table above is not exhaustive: the environment inherits a few generic,
low-weight regularizers from velocity that are not specific to shoot—
`angular_momentum` (-0.02), `dof_pos_limits`—retained (stability, negligible).
⚠️ `soft_landing` (walking reward) is **removed**: it reads the two-foot sensor
`feet_ground_contact`, removed in favour of the left-foot sensor → otherwise a
KeyError on the first step, and it is inert for a stationary kick.

### Sensor-renaming gotcha (⚠️)

Renaming the foot sensor (`feet_ground_contact` → `left_foot_ground_contact`)
breaks everything inherited from velocity/ground_pick that references that name.
Handle:

- **Critic observations** `foot_air_time`/`foot_contact`/`foot_contact_forces` →
  point them to the left-foot sensor (critic retains support information;
  otherwise KeyError during environment construction).
- **Reward** `soft_landing` → remove (see above; otherwise KeyError on the first step).

Always validate with live construction + **at least one `step()`** (the reward
manager only runs at a step), not just configuration construction or unit tests.

### ⚠️ Learned weight transfer (revision after the first training run)

Finding: the BACK/FWD poses recorded **while holding the robot by hand (two-foot
support)** keep the CoM **centred between the two feet** (~4–5 cm inside the
left foot) at every phase. With `upright` imposed, as soon as the right foot
lifts the robot tips over → no policy can hold this (geometry, not tuning).
Verified in simulation (CoM vs foot sites).

Selected fix (RL learns balance):

- `mdp.com_over_support_foot`: Gaussian reward (std 4 cm) pulling the CoM
  projection (`root_com_pos_w`) towards the support foot, **gated** by
  `mdp.kick_engagement` (0 at STAND rest, 1 during the kick). Weight 3.0.
- **Split pose tracking** (`joint_names` parameter on `kick_pose_track`/`_l1`):
  MOVEMENT = right leg + neck/head (std 0.35, tight); SUPPORT = left leg
  (std 0.9, weight 1.0, **loose**) → the policy can adduct/shift the pelvis to
  transfer weight without tracking locking the pelvis in the centre.

The “Balance / support” table above is therefore extended: add
`support_leg_pose` (1.0), `com_over_support` (3.0), and restrict
`kick_pose_track`/`kick_pose_l1` to the 9 movement joints (right side + neck).

### Objective: track the phase-interpolated pose

New **pure** function in `mdp.py`:

```python
kick_pose_target(phase, stand, back, forward, windup_end, kick_end, return_end) -> Tensor
```

Interpolate between pose vectors over 4 segments (normalized period [0,1)):

```text
[0, windup_end)        STAND   → BACK      (wind-up,      default 0.35)
[windup_end, kick_end) BACK    → FORWARD   (sharp strike, default 0.10 = "snap")
[kick_end, return_end) FORWARD → STAND     (return,       default 0.30)
[return_end, 1.0)      STAND               (rest)
```

The “snap” comes from the short strike segment: the joint target moves quickly
→ a fast foot swing. The 3 timing boundaries are configurable.

Resolve joints **by name** (`asset.find_joints([name])`)—robust to ordering.

Tracking rewards (always active, symmetric as in crouch):

| Reward | Weight | Role |
|---|---|---|
| `kick_pose_tracking` | 6.0 | Gaussian tracking `exp(-((q-cible)/std)²).mean`, std=0.4 |
| `kick_pose_l1` | 2.0 | L1 bootstrap (constant gradient early on) |

### Balance / support (single leg = tipping risk)

| Reward | Weight | Role |
|---|---|---|
| `upright` | 2.0 | vertical trunk |
| `support_foot_grounded` (left foot) | 6.0 | keep the support foot planted (single-foot sensor → `found∈{0,1}` → reward∈{0,0.5} after `/2`, so weight 6.0 ≈ maximum contribution 3.0) |
| `feet_flat` (left) | -1.0 | left blade flat |
| `self_collisions` | -1.0 | |
| `body_ang_vel` | -0.05 | |

`support_foot_grounded`: reuse ground_pick's `feet_grounded_reward` mechanism,
but restrict it to the **left foot** (contact sensor on `left_foot_collision`).

### Regularization (lighter than ground_pick—allow the snap)

| Reward | Weight | Role |
|---|---|---|
| `action_rate_l2` | -0.5 | light: a large weight would suppress the fast strike |
| `neck_action_rate_l2` | -0.5 | stable head |
| `joint_torques_l2` | -1e-3 | |

**Removed** (walking terms): `track_linear_velocity`, `track_angular_velocity`,
`air_time`, `foot_clearance`, `foot_swing_height`, `foot_slip`, `pose`.

### Observations / deployment (parity)

- **Identical 61D observations** to ground_pick/roller: `[gyro(3), projected_gravity(3),
  joint_pos(14), joint_vel(14), last_action(14), command(13)]`, with head(4)+body(6)
  slots **zero-padded** (`zero_command_padding`).
- Same sim-to-real DR inherited from velocity (CoM, mass/inertia, BAM friction,
  armature, observation-level IMU misalignment, encoder bias, ±0.3 pushes),
  with NaN-guard termination.
- Export ONNX (normalizer baked in) through the existing export script.
- Deploy in a runtime phase slot, e.g.:
  ```text
  --ground-pick shoot.onnx --ground-pick-period 2.5 \
  --ground-pick-kp-ratio 1.0 --ground-pick-action-scale <match>
  ```
  Button → kick → automatic return to the main policy.

## Tests

- `tests/test_shoot.py`—pure functions:
  - `kick_pose_target` at key points: STAND at φ=0, BACK at `windup_end`,
    FORWARD at `kick_end`, STAND during rest; interpolation at segment midpoints;
    bounds (each component between the minimum/maximum of the poses).
  - `kick_pose_tracking` / `kick_pose_l1` reward values for simple cases.
- `tests/test_shoot_cfg.py`—the environment constructs with the right command
  (`GroundPickPhaseCommand`, `randomize_phase=False`, period), expected rewards
  present / walking terms absent.
- Run: `uv run --with pytest pytest tests/ -q`.

## Training

```bash
uv run train Mjlab-Shoot-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations <N>
```

Monitor `Episode_Reward/kick_pose_tracking` (should rise). Playback: play_latest script.

## Open points / to tune during training

- **Timings** (windup/kick/return) and **period**: reasonable snap defaults,
  adjust according to achieved foot speed and stability.
- **`action_rate` weight:** tension between snap and sim-to-real smoothing;
  start light (-0.5).
- **Optional extension (not selected for v1):** small “right foot moving forwards”
  velocity reward, gated to the strike segment, to encourage power without a
  simulated ball. Add only if pose tracking alone lacks punch.
- **Deployment transitions:** if `STAND_POSE` ≠ the main policy's neutral pose,
  a slight jolt on activation/return (as noted for crouch).
