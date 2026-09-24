# Design — `roller_standup`: getting up on roller skates

> English translation of [microduck_rl/docs/superpowers/specs/2026-08-04-roller-standup-design.md](../../../../microduck_rl/docs/superpowers/specs/2026-08-04-roller-standup-design.md).
> Source repository: `microduck_rl`; source commit: `2b581c641406a48346e696212930ea881c222c52`.
> Original design date: **2026-08-04**. Translated and reviewed: **2026-09-17**.
> The historical design below retains its original values, claims, commands and code semantics. These snippets document the proposal; they are not instructions to execute as part of the current walking experiment.

## Current-source review notes — separate from the translation

- The registered task and [roller StandUp configuration](../../../../microduck_rl/src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py) exist, but the design predates later reward corrections. The current roller configuration uses **`gentle_rise = +0.02`**, rather than the design's −0.02, and **`joint_torque_rate_l2 = −0.2`**, rather than −2e-3. The reward function already returns negative acceleration magnitude, so the original negative weight rewarded abrupt acceleration. See the [translated experimental summary](../roller_standup_policy_summary.md) for the historical diagnosis and attempted fixes.
- The ordinary walking [StandUp configuration](../../../../microduck_rl/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py) has also corrected the sign; its current `gentle_rise` weight is **+0.005**. Do not copy the historical reward table into a new task unchanged.
- [`symmetry.py`](../../../../microduck_rl/src/mjlab_microduck/tasks/symmetry.py) now supports the **61D** layout (migration dated 2026-08-13 in its module commentary). Symmetry is still disabled for Roller StandUp, but the historical 51D incompatibility is no longer the reason to assume it cannot be enabled.
- In the inherited [roller configuration](../../../../microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py), wheel-friction selection now uses **`^passive_.*wheel`**, rather than the historical broad `^passive_.*`. This distinction matters when backlash joints are also passive: a wheel-only operation must not select all passive joints.
- The design's claim that the base roller `action_rate_l2` is fixed at −1.0 is historical. The current roller StandUp source explicitly replaces the base roller ramp towards −2.0 with its own −0.4 → −0.8 → −1.0 curriculum.
- The measured heights, joint indices, friction stages and **4000+ checkpoint threshold** refer to the roller model. They are **not walking or VelStand defaults**, and say nothing by themselves about the quality of the current `Mjlab-VelStand-Flat-Backlash-MicroDuck` run.
- The measurements and runtime assumptions below are preserved as the original author's record. This translation did not repeat the measurements or verify a hardware deployment. The source's separate-policy deployment proposal is historical; the parent project's intended walking/recovery/sitting extension uses one trained policy.

---

## Translated historical design

**Goal**: a dedicated policy that brings the Microduck **upright on its roller skates** after a fall (face down or on its back), then can **hold** that stance on wheels.

Port the `standup` recipe (walking duck) to the roller model. No changes to existing environments.

---

## Agreed decisions

| Decision | Choice | Rejected alternatives |
|---|---|---|
| Form | **Dedicated** episodic policy | Add recovery to the roller environment (`velstand` recipe) → a real risk of breaking the learned skating stride |
| Starting poses | **Face down + on its back + standing** | `sitting` (exists only for handoff from the `sit` policy; no roller equivalent); sides (maximum coverage but much harder convergence); omitting `standing` (the policy would get up and then fall again) |
| Free wheels | **Reversed rolling-friction curriculum** | Real friction from the start (too hard to bootstrap); prescribe a skater's technique through rewards (repository history: overly prescriptive style rewards create unwanted optima—swizzle, the crouch's lazy optimum) |
| Target pose | **HOME + measured height** | Roller-crouch's `STAND_POSE` (flagged as an open issue: differs from roller neutral → a jerk on return); pose read from the real robot (blocks development) |
| Command | **Neutralized twist** (≈ 0) | Phase command / button slot (see “Training and deployment”); controllable head |

---

## Architecture

**New file**: `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`

- `make_microduck_roller_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`
- `MicroduckRollerStandUpRlCfg` (`experiment_name="roller_standup"`)
- Task ID: `Mjlab-RollerStandUp-Flat-MicroDuck` (flat only, no rough variant)

**Derivation**: `cfg = make_microduck_velocity_rollers_env_cfg()`.

This follows the `roller_slope` pattern (246 lines), rather than `roller_crouch` (479 lines, which starts again from `make_velocity_env_cfg()` and copies all the domain-randomization blocks). This inherits the following without a risk of drifting out of sync:

- The `MICRODUCK_WALK_ROLLERS_ROBOT_CFG` robot (14 active joints + 4 passive wheels, BAM m6, kp_fw 200).
- The `feet_ground_contact` sensor (subtree mode on `ankle_{l,r}_v1`) and `self_collision` sensor.
- All domain randomization (DR): trunk + head centre of mass, mass/inertia (`pseudo_inertia`), BAM friction, armature, encoder bias, observation-level IMU misalignment, bearing friction.
- **The unified 61D observation** `[gyro(3), projected_gravity(3), joint_pos(14), joint_vel(14), last_action(14), command(13)]`—a strict requirement for runtime interchangeability.
- The `nan_state` termination (expanded guard: joints + free-joint + wheels).

The roller model **physically allows** lying down: `robot_allcollisions_rollers.xml` has collision geometries on the trunk (`np_f970`), hips, legs, head shells and jaw, in addition to the 4 tyres. Verified.

---

## Measured constants

Measured through exact kinematics (minimum mesh-vertex height of colliding geometries, `STAND` keyframe pose, trunk lowered to contact) on `scene_rollers.xml` versus `scene.xml`:

| Pose | Foot model | Roller model |
|---|---|---|
| Standing (`STAND` = HOME) | 0.1172 | **0.1407** |
| Face down (resting) | 0.0752 | 0.0752 |
| On its back (resting) | 0.0476 | 0.0475 |

Sanity check: `standup` uses `STAND_Z = 0.115`, measured **under load**, versus 0.1172 from kinematics → ~2 mm of sag. Apply the same correction, and the result falls squarely within the `reset_base z = 0.1335–0.1435` already used by the roller environment.

```python
ROLLER_STAND_Z   = 0.138   # trunk standing on wheels, under load (+23 mm vs feet)
ROLLER_PRONE_Z   = 0.075   # resting height when face down
EPISODE_LENGTH_S = 6.0
```

Ground-resting heights are **identical** for both models (the trunk shell makes contact, not the feet). This does not mean the `standup` `prone_z` range can be reused unchanged: see the note under “Reset”—`prone_z_min` differs (0.076 here, not 0.05), because a single range serves two poses (face down, on its back) whose contact heights at reset are different.

The measured quantity is indeed the one read by the rewards: `height_target_gaussian` and `height_l1_penalty` use `root_link_pos_w[:, 2]`, exactly equal to `xpos[trunk_base].z` (the free-joint is on `trunk_base`)—verified numerically.

## Joint indices

The passive wheels are **interleaved** in joint order. Actual order verified in MuJoCo (`m.jnt_qposadr`, roller model, 18 joints after the free-joint):

```text
0-4   left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle
5-6   passive_LF_wheel, passive_LR_wheel
7-10  neck_pitch, head_pitch, head_yaw, head_roll
11-15 right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
16-17 passive_RF_wheel, passive_RR_wheel
```

```python
_LEG_JOINTS   = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]   # standup: [0-4, 9-13]
_NECK_JOINTS  = [7, 8, 9, 10]                          # standup: [5-8]
_WHEEL_JOINTS = [5, 6, 16, 17]
```

Only `_LEG_JOINTS` is actually consumed (by pose rewards). `_NECK_JOINTS` and `_WHEEL_JOINTS` are declared for documentation and the index test: the neck is resolved **by name** (`neck_joint_pos_l2` calls `find_joints(r".*(neck|head).*")` at every step, specifically to tolerate the offset introduced by the wheels), and the wheels by the regex `^passive_.*`.

The handover document explicitly flags this fragility. It is locked down by a test that builds the environment and checks the joint names at those indices (see “Tests”).

---

## Rewards

### Removed from the inherited roller configuration

| Removed | Why |
|---|---|
| `wheel_speed`, `braking`, `skating_air_time`, `glide`, `single_support`, `gait_symmetry`, `forward_lean`, `heading_hold` | Stride rewards: meaningless when on the ground |
| `feet_flat` | During the rise, the skate frames are not flat → this penalty would oppose the movement |
| `hip_roll_neutral` | Getting up requires spreading the legs |
| `pose`, `com_height_target` | Replaced by the pose/height targets below |
| `upright` (base Gaussian) | Replaced by `upright_linear` + `upright_sharp` |

### Kept from the inherited roller configuration

| Reward | Weight | Role |
|---|---|---|
| `action_over_limit` | −0.5 | Task-independent sim2real protection (commands beyond joint stops) |
| `self_collisions` | −1.0 | |
| `body_ang_vel` | **−0.05** | Deliberately **light**: `standup` documents that −0.15 froze recovery (a motion blocker) |
| `angular_momentum` | −0.02 | |
| `action_rate_l2` | Curriculum −0.4 → −0.8 → −1.0 | The roller environment holds it at −1.0; reuse the `standup` ramp (gentle initially → helps bootstrap the large turning-over movement) |
| `neck_action_rate_l2` | −0.5 | Stable head |
| `neck_joint_pos_l2` | −0.5 | Keep the head upright (`roller_slope`'s choice)—**replaces** the `standup` `head_pose` command |
| `joint_torques_l2` | −1e-3 | |

### Added

| Reward | Weight | Role |
|---|---|---|
| `joint_torque_rate_l2` | −2e-3 | Jitter damping: `standup` identified it as the only damper that does not block turning over (it penalizes torque *variation*, not magnitude or trunk rotation) |

### Recovery rewards (transplanted from `standup`, remapped)

The ten terms are copied **with weights already tuned** through the iterations documented in `microduck_standup_env_cfg.py`. Only the joint indices and the two heights change. All MDP functions already exist—**nothing to write in `mdp.py`**.

| Reward | MDP function | Weight | Roller parameters | Role |
|---|---|---|---|---|
| `pose_stand_legs` | `pose_target_match` | +8.0 | `std=0.5`, `joint_indices=_LEG_JOINTS`, `target_overrides=None` (HOME) | Target joint pose |
| `pose_stand_l1` | `pose_l1_penalty` | +5.0 | `joint_indices=_LEG_JOINTS`, `target_overrides=None` | L1 bootstrap: constant gradient even far from HOME |
| `height_stand` | `height_target_gaussian` | +4.0 | `std=0.04`, `target_height=0.138` | Broad Gaussian → pulls up from the ground |
| `height_stand_sharp` | `height_target_gaussian` | +4.0 | `std=0.015`, `target_height=0.138` | Narrow Gaussian → forces the last few centimetres |
| `height_stand_l1` | `height_l1_penalty` | +30.0 | `target_height=0.138` | Makes “staying on the ground” net negative (otherwise a lazy optimum) |
| `com_upward_velocity` | `com_upward_velocity` | +3.0 | `max_height=0.148` | Pays for upward *movement* (+10 mm margin above the target, like 0.125 vs 0.115 in `standup`) |
| `gentle_rise` | `trunk_vertical_accel_penalty` | −0.02 | | Penalizes `\|a_z\|` → smooth rise at constant speed |
| `upright_linear` | `body_upright_linear` | +6.0 | | `cos(tilt)`: strong gradient while lying down |
| `upright_sharp` | `upright_gaussian_at_height` | +6.0 | `std=0.3`, `height_low=0.075`, `height_high=0.138` | Tight height-gated Gaussian → eliminates the backward lean |
| `standing_composite` | `standing_composite_score` | +15.0 | `height_std=0.04`, `upright_std=0.40`, `pose_std=0.40`, `target_height=0.138`, `joint_indices=_LEG_JOINTS` | Multiplicative height × upright × pose score |

All terms take `asset_cfg=SceneEntityCfg("robot", body_names=("trunk_base",))` wherever `standup` does.

**No impact penalties** (trunk/head) for this v1: `standup` has none; only `velstand` has them. Keep the set minimal.

---

## Observation and command

**Observation**: inherited unchanged from the roller environment (61D). No modification—that is the reason for deriving from this environment.

Add `nan_policy = "sanitize"` to the actor and critic groups, as in `roller_slope`: a rare contact makes the free-joint diverge to NaN; the observation is sanitized (→ 0) to avoid killing training, and the affected environment resets on the next step.

**Command**: the `twist` slot is neutralized, exactly as in `standup`:

```python
command = cfg.commands["twist"]
command.rel_standing_envs = 0.0
command.rel_heading_envs  = 0.0
command.heading_command   = False
command.ranges.heading    = None
command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
command.debug_vis = False
command.ranges.lin_vel_x = (-0.01, 0.01)
command.ranges.lin_vel_y = (-0.01, 0.01)
command.ranges.ang_vel_z = (-0.05, 0.05)
cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
```

The `head_pose` (4) and `body_pose` (6) slots remain **zero-padded**—the roller-family convention (`roller`, `roller_crouch`, `roller_slope`). This is an intentional departure from walking `standup`, which controls the head through a real 4D `head_pose` command (see “Risks and things to watch”).

Reason for neutralizing twist: in `scripts/infer_policy.py`, the walking `standup` policy is loaded in `--standing` alongside `--walking`, and switching is **automatic based on velocity-command magnitude** (`infer_policy.py:262`, threshold 0.05); when `standing` is active, the twist slot is left at zero (`infer_policy.py:239`). Phase slots (`ground_pick`, `fold`) are for button-triggered one-shot tricks, not recovery.

---

## Reset

Add the `set_ground_state` event (`reset` mode), inserted **after** the inherited `reset_base` and `reset_robot_joints` (event order follows dictionary insertion order):

```python
cfg.events["set_ground_state"] = EventTermCfg(
    func=microduck_mdp.set_random_ground_state,
    mode="reset",
    params={
        "face_down_prob":  0.50,   # face down — controlled by the curriculum below
        "face_up_prob":    0.00,   # on its back — introduced late (hardest)
        "sitting_prob":    0.00,   # no sitting bucket → no joint override to remap
        "standing_prob":   0.50,
        "prone_z_min":     0.076,  # see note below — not simply inherited from standup
        "prone_z_max":     0.09,
        "standing_z_min":  0.134,  # roller (versus 0.11–0.12 for feet)
        "standing_z_max":  0.144,
        "sitting_tilt_max": math.radians(10),  # ± pitch/roll noise; ALSO applies to the standing bucket
    },
)
```

Note: in `set_random_ground_state`, the `standing` bucket reuses the `sitting` bucket's quaternion—so `sitting_tilt_max` also perturbs standing starts, intentionally.

**On `prone_z_min` = 0.076 (rather than 0.05, incorrectly copied from `standup`)**: face-down and back poses share a single z range, but their measured contact heights differ—face down 0.0752, back 0.0475—so one range cannot be ideal for both. The `standup` comment justifies its `0.05` lower bound using a measured resting height of ~0.044 **after settling under gravity**; however, what matters at reset is the contact height in the HOME pose, not the resting height after falling back down. At 0.05, the face-down spawn puts the trunk shell **25 mm into the ground**, causing a contact pushout that the policy then pays for through `gentle_rise` / `joint_torque_rate_l2`. `prone_z_min = 0.076` eliminates this interpenetration, at the cost of a back start 28–42 mm above its resting height—a much gentler artefact than a contact pushout.

**No changes to `mdp.py`**: the base `reset_robot_joints` uses `joint_names=(".*",)` with `velocity_range=(0.0, 0.0)` and `default_joint_vel` (HOME_FRAME `joint_vel={".*": 0.0}`) → all 4 passive wheels are already reset to zero at every reset. Verified.

**`ground_state_mix` curriculum** (`event_param_curriculum`), using the same easy → hard logic as `standup`: back starts are introduced late and receive the most training at the end.

| Iteration | Standing | Face down | On its back |
|---|---|---|---|
| 0 | 0.50 | 0.50 | 0.00 |
| 600 | 0.35 | 0.45 | 0.20 |
| 1500 | 0.25 | 0.40 | 0.35 |
| 2500 | 0.20 | 0.40 | 0.40 |

(Steps in `common_step_counter` units = `iteration × 24`.)

**Pushes**: `push_robot` is inherited from the roller environment (±0.2 m/s, interval 3–6 s). Add the increasing `standup` curriculum to avoid disrupting bootstrapping: 0 → ±0.08 (iteration 500) → ±0.2 (iteration 1000).

**Terminations**: remove `fell_over` (the robot **starts** fallen—a tilt termination makes no sense here). `nan_state` is inherited and retained.

**Terrain**: `plane`. No rough variant for this v1—consistent with the roller environment, which has no `rough` parameter.

---

## Reversed rolling-friction curriculum

This is the only genuinely new part of the design, and the core of the question posed by the task: **the wheels roll; there is no longitudinal grip for pushing against the ground.**

The mechanism already exists and is inherited (`randomize_wheel_friction` via `dr.dof_frictionloss` on `^passive_.*` + `wheel_friction_curriculum`). In the roller environment it **increases** 0 → 0.0015. Here, make it **decrease**:

| Iteration | frictionloss | Effect |
|---|---|---|
| 0 | 0.05 | Wheels almost locked → gets up as if it had feet |
| 1000 | 0.02 | |
| 2000 | 0.008 | |
| 3000 | 0.003 | |
| 4000 | 0.0015 | The real rolling value (the roller environment's value) |

`wheel_friction_curriculum` simply applies the last stage passed (`if env.common_step_counter > stage["step"]`)—it works just as well for decreasing as increasing values. **No code to write.**

**What this curriculum tells us**: if `Episode_Reward/standing_composite` collapses when friction decreases, that clearly answers that the “grippy feet” movement does not transfer to free wheels, and a skater's technique will need guidance (intermediate knee support, one skate at a time). This is an actionable result, not a failure.

---

## Network and PPO

Identical to `standup`: actor and critic `(512, 256, 128)` elu, `obs_normalization=True` (normalizer baked into the ONNX by `export.py`), PPO `lr=1e-3`, adaptive schedule, `desired_kl=0.01`, `entropy_coef=0.01`, `gamma=0.99`, `lam=0.95`, `num_steps_per_env=24`, `save_interval=250`, `max_iterations=15_000`. **Symmetry OFF** (`SYMMETRY_CFG` is wired for the old 51D layout and breaks on 61D—the same situation as all v1.5+ environments).

---

## Tests

`tests/test_roller_standup_cfg.py`:

1. The environment builds (`play=False` and `play=True`).
2. **Joint names at `_LEG_JOINTS` / `_NECK_JOINTS` / `_WHEEL_JOINTS` indices are correct** (the guard against interleaved-wheel fragility).
3. The expected recovery rewards are present; skating rewards are absent (`wheel_speed`, `glide`, `single_support`, `feet_flat`, …).
4. `fell_over` absent, `nan_state` present.
5. The `wheel_friction` curriculum is **decreasing** and ends at 0.0015.
6. The `ground_state_mix` curriculum: final-stage probabilities sum to 1 and `face_up_prob` increases monotonically.
7. **Observation parity**: actor/critic term names and dimensions are identical to those of `make_microduck_velocity_rollers_env_cfg()` (otherwise the ONNX will not load into a slot).

Run: `uv run --with pytest pytest tests/ -q`.

---

## Training and deployment

```bash
uv run train Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 15000
```

Watch `Episode_Reward/standing_composite` (it should increase), especially its behaviour **at rolling-friction stages** (iterations 1000/2000/3000/4000).

Playback: `uv run scripts/play_latest.py`. Export: `uv run scripts/export_latest.py`.

Intended deployment: the policy in `--standing` alongside the roller policy in `--walking`, with automatic switching based on command magnitude. **Caveat**: `infer_policy.py` is the local simulation/keyboard script; the robot runtime is the Rust binary `microduck_runtime`, absent from this repository—it has not been verified here that it exposes an equivalent `--standing` with the same switching. The handover document lists only `--model`, `--ground-pick`, `--fold-policy`. To be confirmed. This does not change training: if the runtime has no such slot, the policy remains usable in a button slot (the command would then be a phase rather than zero—this would be the only point to revisit).

---

## Risks and things to watch

1. **Getting up on free wheels may be impossible without a dedicated technique.** This is the main risk. The friction curriculum is designed to answer this question clearly rather than work around it.
2. **The “on its back” bucket is hardest.** `standup` documents freezing into “do nothing” in that pose, caused by *motion blockers* (high `body_ang_vel`, excessive `action_rate`). The values reused here are from the “gets up from anywhere” version—do not increase them without reason.
3. **Zero-padded head versus a `head_pose` command.** If the policy is deployed in `--standing` and someone uses the head-control keys, `infer_policy` writes `cmd[3:7] = head_offset`, and the policy sees out-of-distribution input. This is an intentional choice to stay with the roller convention; revisit if head control during recovery proves necessary.
4. **Frictionloss 0.05 is far from reality.** Stages from 0 → 2000 iterations produce a policy that does not transfer; only checkpoints after the last stage (iteration 4000+) are deployment candidates.

## Out of scope

- Integrating recovery into the rolling policy (`velstand` recipe)—decision deferred until feasibility has been validated.
- Side-start buckets.
- Rough / uneven-terrain variant.
- Trunk/head impact penalties.
- Any changes to `roller`, `roller_crouch`, `roller_slope`, `standup`, `velstand`, or `mdp.py`.
