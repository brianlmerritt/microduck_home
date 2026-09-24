# Ground-pick through phase-interpolated pose tracking

> English translation of [the French source](../../../../microduck_rl/docs/superpowers/specs/2026-07-24-ground-pick-pose-following-design.md).
> Source repository: `microduck_rl`; source commit: `2b581c641406a48346e696212930ea881c222c52`.
> Translation date: 2026-09-17. This is a historical design document, not a fresh set of execution instructions. Its proposals, commands and reported results retain their original context.

## Current-source notes (separate from the translation)

- `Mjlab-GroundPick-Flat-MicroDuck` is registered, but the proposed pose-tracking rewrite below is **not the recipe used by the current task**. The [current configuration](../../../../microduck_rl/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py) still rewards mouth proximity and a separate return to the standing pose (`ground_pick_return_pose_legs` and `ground_pick_return_pose_neck`). The existence of generic pose-tracking helpers in `mdp.py` does not mean this task uses them.
- Runtime commands below belong to the historical design. The source puts comments after continuation backslashes in one shell example; that layout is retained, but it is not directly runnable Bash because the backslash must end its line. Runtime parity also needs verification against the actual trained checkpoint and installed runtime.

---

**Date:** 2026-07-24
**Branch:** `new_pre_alpha_ground_pick`
**Target file:** `src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py` (rewrite in place)
**Task id:** `Mjlab-GroundPick-Flat-MicroDuck` (unchanged)

## 1. Objective

Replace the current ground_pick *task-space* objective (reward the mouth
descending to the ground, then separately reward returning to standing) with a
**prescriptive pose-tracking objective**: define two target joint poses—STAND
and DOWN—and reward tracking the **phase-interpolated pose**
(STAND→DOWN→STAND).

Motivation (taken from the validated roller_crouch approach): the interpolated-pose
objective is **symmetric by construction**—“standing up” (target → STAND)
is rewarded in exactly the same way as “lowering” (target → DOWN), solving the
lazy-optimum problem where the policy lowers but rises poorly. The signal is
**dense at every phase** (a continuously moving target), unlike a fixed target
weighted by `sin`, which gives no signal at transitions.

The movement remains triggered by **button A** through the runtime's
`--ground-pick` slot (one-shot, automatic return to the main policy). The unified
61D observation stays unchanged → the policy is interchangeable in the slot.

## 2. Target poses

Resolve joints **BY NAME** (`asset.find_joints([name])`)—robust, consistent
with the roller approach. 14 joints (mouth excluded).

- **STAND_POSE** = HOME (the model's `default_joint_pos`). Source of the blend;
  do not redefine it as hard-coded values—use the model default as the source
  (blend=0). At deployment, the main policy resumes from HOME → clean return.

- **DOWN_POSE** = initial values from the **FOLD keyframe** in `scene_walk.xml`
  (deep forward fold, head lowered → mouth towards the ground). A dictionary by
  name at the top of the file, **commented as replaceable by a `read_pose.py`
  reading** of the real robot positioned with its mouth on the ground. Starting values:

  ```python
  DOWN_POSE = {
      "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": 1.57,
      "left_knee": 1.57, "left_ankle": 0.0,
      "neck_pitch": 1.0, "head_pitch": 1.0, "head_yaw": 0.0, "head_roll": 0.0,
      "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -1.57,
      "right_knee": -1.57, "right_ankle": 0.0,
  }
  ```

## 3. Phase profile (4 segments)

`GroundPickPhaseCommand`: `[cos(2πφ), sin(2πφ), 0]`, period **4.0 s**
(runtime-slot default → no period flag to change at deployment).

```text
DESCENT_END=0.15  HOLD_END=0.50  RISE_END=0.65   (4 s period)
[0, 0.15)     descent   STAND->DOWN   ~0.6 s   blend 0->1
[0.15, 0.50)  low       DOWN          ~1.4 s   blend 1
[0.50, 0.65)  rise      DOWN->STAND   ~0.6 s   blend 1->0
[0.65, 1.0)   high      STAND (rest)  ~1.4 s   blend 0
```

`blend ∈ [0,1]`: 0 = STAND (HOME), 1 = DOWN. Target = `stand + blend·(down - stand)`.
Tunable boundaries (constants at the top of the file).

**`randomize_phase=False`**: each episode starts at φ=0 (= standing), like the
button-A trigger at deployment. Since episodes reset at staggered times, the
environments naturally become decorrelated in phase (no need to randomize).
Requires adding a `randomize_phase` flag to `GroundPickPhaseCommandCfg`
(default `True` → other sit/stand tasks unchanged), respected in `reset()`.

## 4. New mdp functions (ported from roller, adapted, by name)

In `src/mjlab_microduck/tasks/mdp.py`. Use names distinct from the existing
`phase_pose_match` (the fixed-target-weighted-by-sin variant) to avoid confusion.

- **`phase_pose_blend(phase, descent_end, hold_end, rise_end) -> Tensor`**—pure,
  4-segment blend 0..1 (testable in isolation).
- **`_phase_pose_error(env, asset_cfg, command_name, target_pose, descent_end,
  hold_end, rise_end, source_pose=None) -> (cur, target)`**—resolves joints by name;
  `source_pose` = HOME (`default_joint_pos`) if `None`; calculates
  `phase = atan2(sin,cos)/2π % 1`, `blend`, then `target = source + blend·(target_pose - source)`.
- **`phase_pose_track(env, command_name, target_pose, source_pose=None, std=0.3,
  descent_end, hold_end, rise_end, asset_cfg) -> Tensor`**—Gaussian
  `exp(-((cur-target)/std)²).mean(-1)`.
- **`phase_pose_track_l1(env, ...same arguments without std...) -> Tensor`**—bootstrap
  `-(cur-target).abs().mean(-1)` (constant gradient when the Gaussian saturates).

`target_pose` = `DOWN_POSE` (dictionary by name). `source_pose=None` → HOME.

## 5. Rewards

Minimal rewrite relative to the current task—replace the return-pose mechanics,
retain stability/regularization/sim-to-real.

| Reward | Weight | Status | Role |
|---|---|---|---|
| `phase_pose_track` (std 0.3) | **6.0** | **NEW** | track the interpolated STAND↔DOWN pose |
| `phase_pose_track_l1` | **2.0** | **NEW** | L1 bootstrap |
| `mouth_ground_proximity` (std 0.10) | **1.0** | retune (was 2.0) | backstop: ensure the mouth reaches the ground if DOWN is imperfect; approach-gated (+sin) |
| `upright` | 0.2 | retained | approximately vertical trunk (weak; the robot leans) |
| `feet_grounded` | 3.0 | retained | 2 feet on the ground throughout the movement |
| `self_collisions` | -1.0 | retained | |
| `head_impact_penalty` (2 N threshold) | -0.5 | retained | no head slam (DOWN brings the head low) |
| `action_rate_l2` | -0.8→-2.0 (curriculum) | retained | smoothing |
| `neck_action_rate_l2` | -1.0 | retained | |
| `joint_torques_l2` | -5e-3 | retained | |
| `body_ang_vel` | -0.05 | retained | |
| `angular_momentum` | -0.02 | retained | |
| `soft_landing` | -1e-5 | retained | |

**Removed:** `mouth_perpendicular_to_ground`, `ground_pick_return_pose_legs`,
`ground_pick_return_pose_neck` (replaced by pose tracking).

Everything else **unchanged**: DR block (CoM/head-CoM/mass-inertia/friction/armature/
IMU-misalign/encoder-bias/pushes), 61D observations + zero head/body padding,
terminations (`nan_state`), curricula (`action_rate_weight`, `com_range`,
`head_com_range`), RlCfg (`experiment_name="ground_pick"`).

## 6. Deployment (sim-to-real parity)

```bash
microduck_runtime ... \
  --ground-pick ground_pick.onnx \
  --ground-pick-period 4.0 \       # = environment period (default, nothing to change)
  --ground-pick-kp-ratio 1.0 \     # trained at kp 200 → force 1.0 (default 0.6 lowers it to 120)
  --ground-pick-action-scale 1.0   # = environment action.scale
```

## 7. Tests

`tests/` (run `uv run --with pytest pytest tests/ -q`):

- **Pure functions:** `phase_pose_blend` at key points
  (φ=0→0, φ=0.075→0.5, φ=0.3→1, φ=0.575→0.5, φ=0.8→0, monotonic per segment);
  `phase_pose_track`/`_l1`: maximum value (cur==target) and sign.
- **Environment construction:** `make_microduck_ground_pick_env_cfg()` constructs;
  command = `GroundPickPhaseCommand` with `randomize_phase=False`, `period=4.0`;
  rewards `phase_pose_track`/`phase_pose_track_l1` present;
  `mouth_perpendicular_to_ground`/`ground_pick_return_pose_*` absent;
  `mouth_ground_proximity` present with weight 1.0.

## 8. Training / play / export

```bash
uv run train Mjlab-GroundPick-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 20000
uv run scripts/play_latest.py     # md-play
uv run scripts/export_latest.py   # normalizer baked into the ONNX
```

Monitor `Episode_Reward/phase_pose_track` (should rise).

## 9. Out of scope / notes

- **Duplicate `pose_target_match`** (mdp.py 1577 and 1914): latent, not addressed here.
- **Adjusting DOWN_POSE:** if the mouth does not reach the ground sufficiently
  with the FOLD values, adjust the dictionary (ideally using a `read_pose.py`
  reading from the real robot placed mouth-to-ground) instead of increasing
  `mouth_ground_proximity`.
- **Deployment transition:** STAND=HOME = neutral pose of the main policy →
  no jolt on return (unlike the issue noted on roller, where STAND≠HOME).
