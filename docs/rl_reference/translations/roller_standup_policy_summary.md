# `roller_standup` policy — getting up on roller skates

> English translation of [microduck_rl/docs/roller_standup_policy_summary.md](../../../microduck_rl/docs/roller_standup_policy_summary.md).
> Source repository: `microduck_rl`; source commit: `2b581c641406a48346e696212930ea881c222c52`.
> Translated and reviewed: **2026-09-17**. The historical account below retains the original values, claims, commands and caveats. Commands and code snippets are historical reference, not instructions to execute as part of the current walking experiment.

## Current-source review notes — separate from the translation

- The ordinary walking **StandUp sign bug is now fixed**: [`microduck_standup_env_cfg.py`](../../../microduck_rl/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py) assigns `gentle_rise` the positive weight **+0.005**. The historical section below saying it remains unfixed is out of date.
- The [current roller StandUp configuration](../../../microduck_rl/src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py) uses **`gentle_rise = +0.02`** and **`joint_torque_rate_l2 = −0.2`**, with no head-impact reward term. Its later correction section below takes precedence over the earlier historical reward list.
- [`symmetry.py`](../../../microduck_rl/src/mjlab_microduck/tasks/symmetry.py) now implements the **61D** observation layout; its module commentary dates the migration to 2026-08-13. Roller StandUp still leaves symmetry disabled, but the old 51D-only explanation is stale.
- The historical claim that playback **always** restarts the curriculum at stage 0 is dependency-sensitive. The installed `mjlab/rl/runner.py` saves and restores `common_step_counter` from checkpoint `infos["env_state"]`; the environment recomputes curricula on reset. Checkpoints carrying that state can therefore restore a later stage during playback. The explicit `STANDUP_PLAY_FACE_UP` override remains useful, but do not assume default playback has no back starts. Reproducing this behaviour requires the dependency version as well as the task's source commit.
- The heights, wheel friction values, wheel-interleaved joint indices and **iteration 4000 deployment threshold** belong to the roller experiment. They are **not walking or VelStand defaults**. In particular, do not use the wheel-friction threshold to judge the current `Mjlab-VelStand-Flat-Backlash-MicroDuck` run.
- Run IDs, measured reward values and reported robot outcomes below are the original author's experimental record. This translation did not rerun those experiments or independently verify those logs. The proposed deployment arrangement remains a historical proposal, rather than a validation of the real robot runtime.

---

## Translated historical document

**Goal**: the Microduck, on roller skates, starts on the ground—lying face down or on its back—and gets **upright on its wheels**, then **holds** that stance.

- **Task**: `Mjlab-RollerStandUp-Flat-MicroDuck`
- **File**: `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- **Base**: derived from the roller environment (`velocity_rollers`) → the same robot, physics/domain randomization (DR), and **61D observation**; interchangeable at runtime, loadable via `--new-cmd-obs`.
- **Spec**: `docs/superpowers/specs/2026-08-04-roller-standup-design.md` ([English translation](specs/2026-08-04-roller-standup-design.md)).
- **Blind policy**: no terrain scan; proprioception + `projected_gravity`.

## Heights (measured, not guessed)

| Pose | Foot model | Roller model |
|---|---|---|
| Standing | 0.1172 → `STAND_Z=0.115` under load | 0.1407 → **`ROLLER_STAND_Z=0.138`** |
| Face down (resting) | 0.075 | 0.075 |
| On its back (resting) | 0.048 | 0.048 |

The ground-resting heights are identical for both models: the trunk shell touches the ground, not the feet.

## ⚠️ Joint indices — the wheels are INTERLEAVED

```text
0-4   left leg          5-6   left wheels
7-10  neck / head      11-15  right leg       16-17  right wheels
```

`_LEG_JOINTS = [0-4, 11-15]`. The `standup` indices (`[0-4, 9-13]`) apply to the model **without** wheels and would point to wheels here. Locked down by `tests/test_roller_standup_cfg.py::test_joint_indices_match_actual_roller_model`.

## Reset — starting on the ground

`set_random_ground_state`: face down (`prone_z` 0.076–0.09; the lower bound is raised because the belly clears the ground only at 0.0752) / on its back / **already standing** (`standing_z` 0.134–0.144), with ±10° pitch/roll noise. No “sitting” bucket. The “standing” bucket is necessary: without it, the policy gets up but does not stay up.

**`ground_state_mix` curriculum** (easy → hard, back starts last):

| Iteration | Standing | Face down | On its back |
|---|---|---|---|
| 0 | 0.50 | 0.50 | 0.00 |
| 600 | 0.35 | 0.45 | 0.20 |
| 1500 | 0.25 | 0.40 | 0.35 |
| 2500 | 0.20 | 0.40 | 0.40 |

## Rewards

Ten terms taken from `standup` with their already tuned weights: `pose_stand_legs` (+8), `pose_stand_l1` (+5), `height_stand` (+4, std 0.04), `height_stand_sharp` (+4, std 0.015), `height_stand_l1` (+30), `com_upward_velocity` (+3), `gentle_rise` (−0.02), `upright_linear` (+6), `upright_sharp` (+6), `standing_composite` (+15). Plus `joint_torque_rate_l2` (−2e-3), the jitter damper that does not prevent turning over.

Inherited regularizers: `body_ang_vel` **−0.05** (a motion blocker; keep it LIGHT), `angular_momentum` −0.02, `action_rate_l2` (ramp −0.4 → −1.0, **not** the roller's −2.0), `neck_action_rate_l2` −0.5, `neck_joint_pos_l2` −0.5 (head upright), `joint_torques_l2` −1e-3, `action_over_limit` −0.5, `self_collisions` −1.0.

Removed: all skating rewards, plus `feet_flat` (the skate frames are not flat while rising) and `hip_roll_neutral` (getting up requires spreading the legs).

## ⚠️ The hard part: the wheels roll

There is no longitudinal grip for pushing against the ground. The **rolling-friction curriculum is REVERSED**: the roller environment increases it; here it decreases.

| Iteration | frictionloss | Effect |
|---|---|---|
| 0 | 0.05 | Wheels almost locked → gets up as if it had feet |
| 1000 | 0.02 | |
| 2000 | 0.008 | |
| 3000 | 0.003 | |
| 4000 | 0.0015 | The real rolling value |

**Watch `Episode_Reward/standing_composite` at the stages.** If it collapses, the “grippy feet” movement does not transfer to free wheels → a skater's technique will need guidance (intermediate knee support, one skate at a time). This is a result, not a failure.

**ALSO watch the robot's horizontal drift during playback**, at each friction stage. `standing_composite` sees neither `root_link_pos_w[:2]` nor horizontal velocity: a policy that gets up while sliding far from its starting point receives exactly the same score as one that gets up and stops. Until this drift has been visually measured, the outcome of the friction curriculum—the very question this environment exists to answer—is not reliable.

**Sim2real**: only checkpoints after iteration 4000 are deployment candidates. Before that, the policy relies on friction that does not exist on the real robot.

## Command

Neutralized `twist` slot: `lin_vel_x`/`lin_vel_y` ±0.01, `ang_vel_z` **±0.05** (5× wider—the same choice as `standup`). `head_pose` / `body_pose` slots are **zero-padded** (roller convention). Intended deployment: in `--standing` alongside the roller policy in `--walking`, with automatic switching based on command magnitude (`infer_policy.py:262`, threshold 0.05); the twist slot is left at zero there (`infer_policy.py:239`).

**Caveat**: `infer_policy.py` is the local simulation/keyboard script. The robot runtime is the Rust binary `microduck_runtime`, absent from the repository—it has not been verified that it exposes an equivalent `--standing` option. The crouch handover document lists only `--model`, `--ground-pick`, `--fold-policy`. To be confirmed.

## Terminations

`fell_over` **removed** (the robot starts fallen). `nan_state` inherited. `nan_policy="sanitize"` on actor/critic observations.

## Network / PPO

Actor and critic `(512, 256, 128)` elu, `obs_normalization=True`. PPO `lr=1e-3` adaptive, `desired_kl=0.01`, `gamma=0.99`, `lam=0.95`, `num_steps_per_env=24`, episode 6 s, `max_iterations=15000`. **Symmetry OFF** (`SYMMETRY_CFG` is wired for the 51D layout).

## Commands

```bash
uv run train Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 15000
uv run scripts/play_latest.py        # alias md-play
uv run scripts/export_latest.py      # alias md-export
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```

### ⚠️ Seeing back starts during playback

By default, playback **never** shows starts on the back: the playback environment is rebuilt from scratch, so `common_step_counter` returns to 0 and the curriculum applies stage 0, where `face_up_prob = 0`. You see only 50% face down / 50% standing, regardless of how mature the loaded checkpoint is. But starting on the back is the hardest case—the very one we want to inspect.

`STANDUP_PLAY_FACE_UP` forces the mix (the same pattern as `SLOPE_PLAY_DIFFICULTY` in `roller_slope`), **only on the `play=True` path**—training and its easy → hard curriculum remain untouched:

```bash
STANDUP_PLAY_FACE_UP=1.0 md-play    # 100% starts on the back
STANDUP_PLAY_FACE_UP=0.4 md-play    # the final curriculum stage's mix
STANDUP_PLAY_FACE_UP=none md-play   # default (stage 0, no starts on the back)
```

The remainder (`1 - face_up`) is split face-down:standing in the final stage's 2:1 ratio, so `0.4` exactly reproduces the end-of-training mix (0.40 / 0.20 / 0.40).

## 🔧 Correction for violent movements (after the first robot test)

**Symptoms** on a checkpoint after 4000: very abrupt movements, head striking the ground, failure to get up from the back on the robot. **Also present in simulation** → this was therefore neither a sim2real problem nor a checkpoint that was too early, but reward design.

**Root cause: `gentle_rise` rewarded violence.** `trunk_vertical_accel_penalty` already returns `-|a_z|` (`mdp.py:2171`); multiplied by the **−0.02** weight inherited from `standup`, this became a double negative, therefore `+0.02·|a_z|`—**the more abruptly the trunk accelerated, the more the policy was paid**. Confirmed by the log: `Episode_Reward/gentle_rise = +0.0118` on run `vweolw91`, the only penalty term logged as positive.

`mdp.py` mixes two sign conventions, and that is the trap:

| Term | Function returns | Correct weight |
|---|---|---|
| `height_stand_l1`, `pose_stand_l1`, `gentle_rise` | `-abs(...)`, already negative | **Positive** |
| `joint_torques_l2`, `joint_torque_rate_l2`, `action_rate_l2`, `body_impact_cost` | Positive magnitude | **Negative** |

Locked down by `test_already_negative_penalties_use_positive_weights`.

⚠️ **The walking robot's `standup` has exactly the same bug** (same function, same −0.02 weight). That explains the sequence of unsuccessful damping attempts documented in its comments (“*violent / shaky / overshoot-tip-repeat on the real robot*”): they were fighting a term actively pushing in the opposite direction. **Not fixed here**—it is a different environment, to be decided separately.

**Related structural problem.** At convergence, task rewards totalled **≈ +41.6**, saturated at 95–99%, against **≈ −1.2** for all dampers combined—including `joint_torque_rate_l2` at **−0.0002/step** and `joint_torques_l2` at **−0.0001/step**, effectively nothing. A ratio of ~35:1: no reason to be gentle.

**Current state of the corrections:**

| Term | Before | Now | Why |
|---|---|---|---|
| `gentle_rise` | −0.02 (reward) | **+0.02** (penalty) | Sign corrected; magnitude deliberately kept SMALL—`\|a_z\|` is necessarily high during a flip, so a large weight would block motion |
| `joint_torque_rate_l2` | −2e-3 | **−0.2** | The SAFE control: penalizes torque variation, not motion |
| `head_impact_penalty` | Absent | **Still absent** | Tried at −1.0; it froze the policy—see below |

### ⚠️ The head-impact penalty froze the policy—do not restore it unchanged

An attempt using the `velstand` values (`body_impact_cost`, `neck` subtree, −1.0, threshold 2.0): **the policy converged to lying still, inert.** Measured on run `d8rnko6p`:

| Term | Before (violent) | With head_impact (frozen) |
|---|---|---|
| `standing_composite` | +14.32 | **+3.26** |
| `upright_sharp` | +5.76 | +1.06 |
| `head_impact_penalty` | — | **−1.01** ← largest negative term |
| `joint_torque_rate_l2` | −0.0002 | −0.255 (therefore **not** the culprit) |

The reasoning mistake: believing that a “targeted” penalty does not restrict motion. **False here—to get up from its back, this robot pivots on its head and shoulders.** The head is the flip's support point, not collateral damage; penalizing it blocks the only available mechanism, and getting up from the back was already the case that failed.

**The lazy optimum that makes this freeze possible**: `pose_stand_legs` remained at **+7.72 out of 8** while the robot was lying down—the legs are at HOME in the lying pose, so this reward is collected almost for free. It is `height_stand_l1` (weight +30) that must make “staying on the ground” net negative; it must not be weakened.

**Hypothesis being tested**: hitting the head was a *symptom* of violent movement (the sign bug paid for abruptness, and an abrupt rise ends on the head), not a separate defect. If the slam returns now that the sign is corrected, the next attempt should use a **height-gated** penalty (as `upright_sharp` is), which spares the ground-level turning-over phase.

**Methodological lesson**: all three corrections were applied at once, so the freeze could not be attributed with certainty—only the most likely suspect could be identified. One correction at a time in future.

**Recalibration if it is still violent**: `|Δτ|²` is ~0.1 at convergence, so the contribution of `joint_torque_rate_l2` ≈ `0.1 × |weight|`. Increase **this** term, **not** `body_ang_vel` (−0.05) or `action_rate_l2` (ramp → −1.0): those block motion, and `standup` documents that at −0.15 and −1.2 respectively, they **froze** recovery from the back. If, conversely, back recovery stops working, **reduce** `joint_torque_rate_l2` first.

## Out of scope

Integrating recovery into the rolling policy (`velstand` recipe); side-start buckets; a rough-terrain variant; trunk/head impact penalties.

No reward penalizes the trunk's horizontal velocity (`root_link_lin_vel_w[:, :2]`): “getting up while rolling far away” is an unpenalized outcome that receives full credit. This is deliberate, not an oversight: a stillness reward without height gating would also penalize the translation physically required to rise from the ground—the “motion blocker” failure mode documented by `standup`. A candidate if the problem is confirmed: height-gated stillness, active only near `ROLLER_STAND_Z`.
