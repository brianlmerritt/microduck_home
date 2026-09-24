# Specification — “Spin” environment (fast rotation in place, on rollers)

> English translation of [the French source](../../../../microduck_rl/docs/superpowers/specs/2026-08-04-spin-env-design.md).
> Source repository: `microduck_rl`; source commit: `2b581c641406a48346e696212930ea881c222c52`.
> Translation date: 2026-09-17. This is a historical design document, not a fresh set of execution instructions. Its proposals, commands and reported results retain their original context.

## Current-source notes (separate from the translation)

- `Mjlab-Spin-Flat-MicroDuck` is registered. The [current configuration](../../../../microduck_rl/src/mjlab_microduck/tasks/microduck_spin_env_cfg.py) and [MDP constants](../../../../microduck_rl/src/mjlab_microduck/tasks/mdp.py) use the amended target: `SPIN_RATE_MAX = 3.0`, `SPIN_WHEEL_OMEGA_SCALE = 17.0`, drift weight `−3.0`, and launch drift scale `0.2`. The original 6 rad/s, two-turn design below is historical and explicitly amended within the source.
- The source's run results and inferred diagnostics are translated as reported, not independently reproduced. A mean episode length alone does not prove that every episode had that duration or that no episode reached later phases. Likewise, reward-ratio estimates are not direct measurements of tracking error. Preserve that distinction when applying these lessons to a new experiment.
- The runtime example retains the source's comments after continuation backslashes; it is not directly runnable Bash in that form. The source also specifies training action scale `1.0` but deployment scale `0.8`. Matching observation order alone does not establish complete deployment parity. Check the actual checkpoint and runtime before using these commands.

---

Date: 2026-08-04. Branch: `new_pre_alpha_rollers`.

> **Amendment (after the first run):** the first calibration run (500 iterations)
> showed that the robot consistently falls at around 1.16 s, well before
> braking. In response, the target was halved—`SPIN_RATE_MAX`
> 6.0 → **3.0 rad/s**, or **1 turn per cycle instead of 2**—and
> `spin_stay_in_place` increased in strength to **−3.0**, **without a speed curriculum**.
> See “Initial verification results” for the evidence and the configuration
> currently in effect.

## Goal

A new RL task teaching microduck on rollers to perform a **spin**:
~2 counterclockwise turns in place at ~6 rad/s (360°/s) *(initial target;
reduced to 3 rad/s, see the amendment)*, then stop cleanly while standing.
A **cyclic movement driven by a phase**, deployed in a runtime **one-shot button
slot**, like the existing `roller_crouch` task.

## Agreed decisions

| Question | Decision |
|---|---|
| Support | On rollers (`MICRODUCK_WALK_ROLLERS_ROBOT_CFG`, 4 passive wheels) |
| Control | One-shot button slot, command = phase `[cos(2πφ), sin(2πφ), 0]` |
| Target | ~6 rad/s, 2 turns, then brake to a stop (initial target; reduced to 3 rad/s, see the amendment) |
| Entry state | Stationary **or** rolling slowly (0 → 0.3 m/s) |
| Direction | Left only (positive yaw, counterclockwise) |
| Approach | “Outcome” objective (tracking ω_z) + decaying antisymmetric shaping |

**Runtime constraint:** the slot sends only `[cos, sin, 0]`—no free channel
for rotation direction. The policy therefore **always turns left**. A mirrored
policy could later go into another slot (button B, `--fold-policy`).

## Intended physical mechanics

On 4 passive wheels, a “clean” rotation in place uses **differential rolling**:
the left skate goes backwards, the right forwards (the wheels **roll**, they do
not skid). It is an *antisymmetric swizzle*: the legs do the opposite of each
other, rather than the mirror motion of a conventional swizzle.

Sign check for a counterclockwise rotation (frame: x forwards, y left,
z up; ω_z > 0): a point on the left (+y) has velocity `ω ẑ × y ŷ = −ω y x̂`,
so **backwards**. All 4 wheels turn positive when moving forwards (verified by
`test_wheel_direction.py`), so for a counterclockwise spin:
`ω_roues_gauche < 0`, `ω_roues_droite > 0`, hence **`ω_D − ω_G > 0`**.
Here the French subscripts denote left wheels, right wheels, right (`D`) and left (`G`).

## Selected approach (C) and why

Three approaches were considered:

- **A—pure “outcome” objective:** reward yaw velocity and let PPO find the
  movement. Risk documented in this repository: a lazy optimum /
  skidding-hopping instead of clean rolling.
- **B—“prescriptive” pose objective:** two scissoring poses interpolated by
  phase, like `roller_crouch`. Works quickly *if* the poses are good; for crouch
  they were **read from the real robot**, whereas here the movement is unknown.
  It would need to be composed by hand: expensive and risky (poses without
  useful torque achieve nothing).
- **C—A + decaying antisymmetric shaping** ← **selected**. Structure of A,
  plus two weak *shaping* terms that inject the only certain physical knowledge
  (differential rolling), whose weights decay through a curriculum to let the
  policy refine its own movement. The **pumping frequency remains free**.

## Architecture

**File:** `src/mjlab_microduck/tasks/microduck_spin_env_cfg.py`

- Factory `make_microduck_spin_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`.
- PPO configuration `MicroduckSpinRlCfg`.
- Task id `Mjlab-Spin-Flat-MicroDuck`, registered in `tasks/__init__.py`.

Clone the structure of `microduck_roller_crouch_env_cfg.py`: roller robot,
unified 61D observations, full DR, `action.scale = 1.0`, flat terrain.

**`ENABLE_SYMMETRY = False`**—mandatory: left/right symmetry augmentation would
turn a left spin into a right spin and destroy learning.

**Command:** `GroundPickPhaseCommandCfg(period=4.0, randomize_phase=False)`.
`period=4.0` is the `--ground-pick-period` default → nothing to pass to the runtime.
`randomize_phase=False` → each episode starts at φ=0 (standing), as at deployment.

## Phase envelope

The phase drives a **target yaw velocity** ω\*(φ), a trapezoid over 4 segments
(4 s period, `SPIN_RATE_MAX = 6.0` rad/s—initial target; reduced to 3 rad/s,
see the amendment; the segments and period have not changed):

```text
ACCEL_END = 0.125   [0,     0.125)  0.5 s  ω* : 0 → 6 rad/s   (launch, linear ramp)
HOLD_END  = 0.525   [0.125, 0.525)  1.6 s  ω* = 6 rad/s        (steady speed)
BRAKE_END = 0.650   [0.525, 0.650)  0.5 s  ω* : 6 → 0          (braking, linear ramp)
            1.0     [0.650, 1.0)    1.4 s  ω* = 0              (standing rest)
```

*(The ω\* = 6 rad/s values above correspond to `SPIN_RATE_MAX` = 6.0, the initial
target; see the amendment for the value currently in effect.)*

Integral over a cycle: `0.5·3 + 1.6·6 + 0.5·3 = 12.6 rad ≈ 2.0 turns`. ✅
*(At `SPIN_RATE_MAX = 6.0`, the initial target.)* General form: the integral is
`2.1 × SPIN_RATE_MAX` regardless of `rate_max` (0.25 + 1.6 + 0.25 = 2.1).
With the current target (3.0 rad/s): `2.1 × 3.0 = 6.3 rad ≈ 1 turn` per cycle—
see the amendment.

Episode = 20 s = **5 cycles**: the robot repeats launch → steady speed → braking
→ rest five times per episode. More data per episode, and the “rest” segment
also trains a clean exit from the trick. **Note (post-run):** this remains true
geometrically (20 s / 4 s), but no episode in the calibration run survived beyond
~1.16 s, only a fraction of the first cycle—see “Initial verification results”.

**Pure function** `spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)`
in `mdp.py`, next to `crouch_pose_blend`. Testable without a simulator.

**Shaping gate:** `gate(φ) = spin_rate_by_phase(φ) / rate_max ∈ [0, 1]`. Equals 0
throughout rest → no shaping term encourages scissoring then, so the robot returns
to a neutral stance. This provides a clean exit from the trick to the roller policy.

## Rewards

### Pitfalls checked in mjlab (handle explicitly)

- `body_ang_vel` (`body_angular_velocity_penalty`) penalizes only **x/y**
  (`ang_vel_xy`, comment “Don't penalize z-angular velocity”) → **retained**
  (weight −0.05): suppresses roll/pitch sway without interfering with the spin.
- `angular_momentum` (`angular_momentum_penalty`) penalizes the **3D norm** of
  angular momentum → it would directly oppose the spin. **Removed.**

### New rewards (to write in `mdp.py`)

| Reward | Weight | Definition |
|---|---|---|
| `spin_rate_track` | 6.0 | `exp(−((ω_z − ω*(φ))/std)²)`, `std = 1.5` rad/s. ω_z = trunk yaw velocity in the body frame (what the IMU observes). Main objective. |
| `spin_rate_l1` | 0.5 | `−\|ω_z − ω*(φ)\|`: constant-gradient bootstrap when the Gaussian saturates far from the target (same trick as `crouch_glide_pose_l1`) |
| `spin_stay_in_place` | −3.0 (initially −1.0, see the amendment) | Trunk `‖v_xy‖²` → “in place”, and removes entry momentum. No reference state, so robust to the 5 cycles per episode |
| `spin_wheel_differential` | 1.0 | `gate(φ) · tanh(clamp(ω_D − ω_G, min=0) / omega_scale)`, with `ω_G = (LF+LR)/2`, `ω_D = (RF+RR)/2`: rewards skates rolling in opposite directions consistent with counterclockwise rotation → turn **by rolling**, not skidding. Wheels resolved by name (`passive_LF_?wheel`, …). Current `omega_scale = 17.0` rad/s (see the calibration paragraph below) |
| `leg_antisymmetry` | 1.0 → 0.25 | `gate(φ) · (−mean\|q_G − q_D\|)` on `hip_pitch` and `knee`. ⚠️ Mirror convention: a *symmetric* pose gives `q_G + q_D ≈ 0`, so **scissoring** means `q_G ≈ q_D`. Decays through a curriculum |
| `spin_grounded` | 0.5 | `gate(φ) · 1[n_contact ≥ 2]`: both blades on the ground, preventing “jump and spin in the air”. Swizzle's `grounded_reward` cannot be reused unchanged (it weights itself by `cmd_x`, which here equals `cos(2πφ)`) |

**Calibrating `omega_scale`** (tanh saturation scale): at the target steady speed,
each skate travels at `v = ω_z · demi_voie` (half-track width), so each wheel turns
at `v / r`, with `r = 0.0175` m, and the differential is `2 · ω_z · demi_voie / r`.
The leg roots are at `y = ±0.0175` m in the roller model, but the skates are farther
apart (ankle offset): the actual half-track width must be **measured at the
`left_foot` / `right_foot` sites in simulation** during the first run. With an
estimated half-track width of ~0.03 m and `ω_z = 6` rad/s, the expected differential
was ~20 rad/s—hence the initial default `omega_scale = 20.0`. **Measurement completed
(Task 3): actual half-track width = 0.0499 m, expected differential = 34.2 rad/s,
71% above the estimate—beyond the plan's 30% threshold.** `SPIN_WHEEL_OMEGA_SCALE`
was therefore corrected to **34.0** (an intermediate value, in effect while the
target was 6 rad/s; since recalibrated to **17.0**, see the “Update” paragraph
immediately below). See “Initial verification results” below for details of the
half-track-width measurement.

**Update (post-review fix wave):** `SPIN_RATE_MAX` was reduced from 6.0 to
**3.0 rad/s** (human decision, no curriculum—see below). The effect on
`omega_scale` is a direct mechanical consequence, not an independent choice:
the expected steady-speed differential becomes
`2 · 3.0 · 0.0499 / 0.0175` = **17.1 rad/s**. Leaving `omega_scale = 34.0` would
cap the term at `tanh(17.1/34) = 0.47` of its own maximum, weakening exactly the
shaping we want to strengthen. `SPIN_WHEEL_OMEGA_SCALE` is therefore corrected
again to **17.0**, retaining the same measured half-track width (0.0499 m) as reference.

### Rewards reused from `roller_crouch` (stability / sim-to-real)

| Reward | Weight |
|---|---|
| `upright` (vertical trunk) | 2.0 |
| `feet_flat` (flat blades) | −2.0 |
| `self_collisions` | −1.0 |
| `body_ang_vel` (xy only) | −0.05 |
| `action_rate_l2` | −1.0 (curriculum −0.5 → −1.0) |
| `neck_action_rate_l2` | −0.5 |
| `joint_torques_l2` | −1e-3 |
| `neck_joint_pos_l2` **excluding `head_yaw`** | −0.2 |

**The head:** neck pitch/roll kept near neutral (sim-to-real), but `head_yaw`
**excluded** from the term → free to act as a flywheel to initiate rotation.
Implementation: `neck_joint_pos_l2` resolves its joints using the hard-coded
regex `.*(neck|head).*`; either add a regex parameter to this function or write a
`neck_joint_pos_l2_no_yaw` variant. Choice: **add a `pattern` parameter** to
`neck_joint_pos_l2` (unchanged default) to avoid duplication.

## Reset / entry state

```python
cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)
cfg.events["reset_base"].params["velocity_range"] = {"x": (0.0, 0.3)}
```

Inject through `reset_root_state_uniform`. **Never** use
`push_by_setting_velocity` in `mode="reset"`: that caused the NaNs in crouch
(`root_vel +=` applied to a potentially divergent root velocity → the base free joint blows up).

## Domain randomization

Identical to `roller_crouch`, without deviation (the repository's validated
sim-to-real recipe): trunk + head COM, mass/inertia, BAM joint friction, armature,
wheel friction, 0.2 m/s pushes every 3–6 s, 6° IMU misalignment, ±0.015 rad encoder bias.

## Observations

**Identical 61D layout** to roller / ground_pick / crouch—a condition for the
ONNX to load in the slot:
`[gyro(3), projected_gravity(3), joint_pos(14), joint_vel(14), last_action(14), command(13)]`,
with `command = [twist(3), head_pose(4), body_pose(6)]`, head/body zero-padded.

Therefore: remove `base_lin_vel` from the actor (retain for the critic), remove
`height_scan` and `foot_height`, `wheel_vel` on the critic side, passive joints
excluded from `joint_pos`/`joint_vel`, delays and noise identical to crouch.

The gyro is in the observations → the policy **observes** its own ω_z: the task is observable.

## Terminations

`time_out`, `fell_over`, `out_of_terrain_bounds` (inherited) + `nan_state`
(`microduck_mdp.robot_state_is_nan`), as in crouch.

## Curriculum

| Term | Stages |
|---|---|
| `action_rate_weight` | −0.5 (0) → −0.8 (250 iterations) → −1.0 (500 iterations) |
| `leg_antisym_weight` | 1.0 (0) → 0.5 (1500 iterations) → 0.25 (3000 iterations) |
| `com_range` | 0.003 → 0.005 (500 iterations) → 0.01 (1000 iterations) |
| `head_com_range` | 0.003 → 0.005 (500 iterations) → 0.01 (1000 iterations) |

(Iterations × 24 steps/environment, as in the other environments.)

**No target-speed curriculum:** 6 rad/s from the outset *(initial target;
reduced to 3 rad/s, still without a curriculum, see the amendment)*. See “Plan B”.

## PPO

`MicroduckSpinRlCfg` = copy of `MicroduckRollerCrouchRlCfg`: actor/critic
(512, 256, 128) elu, observation normalization, adaptive PPO lr 1e-3,
`desired_kl=0.01`, `num_steps_per_env=24`, `symmetry_cfg=None`,
`experiment_name="spin"`, `run_name="spin"`, `max_iterations=8000`.

## Tests

`tests/test_spin.py`—pure functions, without a simulator:

- `spin_rate_by_phase`: values at the boundaries of the 4 segments
  (0, rate_max, rate_max, 0, 0).
- Monotonically increasing on the launch ramp, decreasing during braking.
- **Integral over a cycle ≈ 4π** at `rate_max = 6.0` (guarantees the trapezoid's
  **shape**, `2.1 × rate_max` rad per cycle)—no longer protects the current
  target since the amendment; see the next bullet. Exact envelope value: 12.6 rad
  versus 4π = 12.566 → 1% tolerance.
- **The actual shipped target** (`mdp.SPIN_RATE_MAX`) correctly integrates to
  `2.1 × SPIN_RATE_MAX` rad per cycle, regardless of `rate_max`—added in
  7d916aa; this test fails if the target changes without considering the number
  of turns. With the current value (3.0 rad/s): 6.3 rad ≈ 1 turn.
- `gate(φ) = 0` throughout rest, `∈ [0,1]` everywhere.

`tests/test_spin_cfg.py`—the environment constructs:

- Command = `GroundPickPhaseCommand`, `period == 4.0`, `randomize_phase is False`.
- `"angular_momentum" not in cfg.rewards` (the pitfall in the rewards section).
- `symmetry_cfg is None`.
- Actor observation dimension == 61.
- **Exact observation-term ordering parity** (actor + critic) with
  `roller_crouch`, group by group—added in 7d916aa, a strict condition for the
  exported ONNX to load in the runtime slot.

Run: `uv run --with pytest pytest tests/ -q`.

## Training / deployment

```bash
uv run train Mjlab-Spin-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 8000
# monitor Episode_Reward/spin_rate_track (should rise)
uv run scripts/play_latest.py     # alias md-play
uv run scripts/export_latest.py   # ONNX, observation normalizer baked in
```

```bash
microduck_runtime --variant pre-alpha --new-cmd-obs --roller \
  --model output.onnx --new-dxl-imu --kp 200 --action-scale 0.8 \
  --ground-pick spin.onnx \
  --ground-pick-period 4.0 \      # = SPIN_PERIOD
  --ground-pick-kp-ratio 1.0 \    # default 0.6 -> force 1.0 (trained at kp 200)
  --ground-pick-action-scale 0.8  # match runtime action_scale
```

Button **A** → spin, then automatically return to the roller policy.

## Success criterion

In playback: ~2 counterclockwise turns in ~2.6 s, trunk drift < ~10 cm, robot
upright throughout, stable neutral stance during rest before the next cycle.
*(Criterion written for the initial 6 rad/s / 2-turn target; at 3 rad/s, see
the amendment, this would be ~1 turn over the active-speed duration—the criterion
has not been revised because the robot does not yet stay upright that long.)*

## Plan B if training plateaus

In order:

1. **Speed curriculum:** `SPIN_RATE_MAX` 3 → 6 rad/s (requires making
   `rate_max` controllable through a `CurriculumTermCfg` on reward parameters).
   **Partially followed:** after the calibration run, the target was lowered
   to 3 rad/s (see the amendment), but **without a curriculum**—3 rad/s is
   currently a fixed target, not a starting point for a ramp towards 6.
   The human chose to first see what the robot could do at this speed before
   considering a gradual increase.
2. Increase `spin_wheel_differential` and delay the decay of `leg_antisymmetry`.
3. Widen `std` in `spin_rate_track` (1.5 → 2.5) for a useful gradient farther away.
4. As a last resort, switch to approach B (scissoring poses composed by hand in
   a pose editor) to bootstrap the movement, then relax the constraint.

## Out of scope

- Rightward spin (mirrored policy in another slot)—later.
- Walking-foot variant (without rollers).
- Continuously speed-commanded spin (would require a runtime command channel).

## Initial verification results

### Measured half-track width and `omega_scale`

The half-track width was measured at the roller model's `left_foot` /
`right_foot` sites: **0.0499 m**, against the specification's estimate of 0.03 m.
Expected wheel differential at steady speed (6 rad/s):
`2 · 6.0 · 0.0499 / 0.0175` = **34.2 rad/s**, 71% above the 20.0 default—beyond
the plan's 30% threshold. `SPIN_WHEEL_OMEGA_SCALE` was therefore changed from
20.0 to **34.0**. Tests continue to pass `omega_scale=20.0` explicitly to remain
independent of the constant.

### Smoke run (Step 2: 5 iterations, 64 environments, NaN guard)

Completed without exception. `Episode_Termination/nan_state` remained at 0.0000
throughout, and `/tmp/mjlab/nan_dumps/` was never created. All six spin rewards
appear in the logged `Episode_Reward/` keys: `spin_rate_track`, `spin_rate_l1`,
`spin_stay_in_place`, `spin_wheel_differential`, `spin_grounded`, `leg_antisymmetry`.

Observation parity (Step 1): the spin environment's actor observation-term list
is **identical** to `roller_crouch`—8 terms, same order:
`base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, command,
head_command, body_command`. This is the condition for the exported ONNX to
load in the runtime slot.

**Usage note to retain:** the plan's example command using `--enable-nan-guard`
as a bare flag is rejected by this repository's CLI—pass `--enable-nan-guard True`.

### 500-iteration calibration run (Step 3)

4096 environments, 500 iterations, ~2.32 s/iteration, exit code 0, wandb logger
(so `scripts/play_latest.py` / `md-play` finds the run).

**What was actually established:** `Mean episode length` = **57.83 steps** out
of a 1000-step episode (20 s at 50 Hz), or **~1.16 s**.
`Episode_Termination/fell_over` ≈ **70**, `time_out = 0.0000`, `nan_state = 0`.
The robot **falls in every episode**, at phase φ ≈ 0.29—in the middle of the
steady-speed segment. It never reaches braking (φ ≥ 0.525) or rest
(φ ≥ 0.650): **71% of the cycle is never trained**.

Episode length rose from 23.98 to 57.83 steps during the run: the increase in
`Episode_Reward/spin_rate_track` (0.0291 → 0.3168) therefore mainly reflects
**longer survival**, not improved tracking. The success criterion stated for
this step in the plan (“the curve should rise”) **is not a valid signal** for
this term: a completely stationary robot already scores `6.0 × 0.405 = 2.43`
on it—the rest segment pays fully for standing still, so any policy surviving
longer mechanically captures more of that segment, regardless of tracking quality.

### Derived diagnosis—estimates, not direct measurements

The values below come from ratios between reward terms in the final log block,
cancelling the unknown normalization factor applied by the logger. Treat them
as estimates, reproducible using the same method:

**What works:** during the ~1.2 s it stays upright, the robot follows the target
fairly closely. Taking the `spin_rate_l1 / spin_rate_track` ratio
(−0.0097 / 0.3168, weights 0.5 and 6.0, `std = 1.5`), and solving
`e = 0.3674 · exp(−(e/1.5)²)`: mean absolute yaw-velocity tracking error ≈
**0.35 rad/s**, confirmed by two independent routes—this
`spin_rate_l1 / spin_rate_track` ratio, and an inverse calculation from the
reward manager's normalization. It **can initiate** the spin; it **cannot stay
upright** while doing it.

**What does not work:** the shaping block (`spin_wheel_differential` 1.0,
`spin_grounded` 0.5, `spin_stay_in_place` −1.0) totals ~1.0 weight against 6.0
for the main objective—around **13%** of what a skidding policy would forgo
by ignoring this block. And `spin_wheel_differential` is **invariant to the
instantaneous centre of rotation**: a centred spin at 6 rad/s and a pivot around
the left skate at 6 rad/s both produce a differential of 34.2—this term therefore
does **not** encode centred rolling; only `spin_stay_in_place` does.
`spin_stay_in_place` ≈ −0.0069 implies `‖v_xy‖ ≈ 0.35 m/s`: the robot is still
translating, consistent with an off-centre pivot (skate as pivot) rather than
rotation around the body centre.

### Configuration change decided after this diagnosis

Target halved—`SPIN_RATE_MAX` 6.0 → **3.0 rad/s**—and `spin_stay_in_place`
strengthened from −1.0 to **−3.0** (see the rewards table and
`SPIN_WHEEL_OMEGA_SCALE` recalibrated to 17.0 above). **Deliberately no curriculum**
on target speed: this is a first trial to see what the robot can do at half
speed before considering a gradual increase if needed.

**Reducing drift cost during launch.** Strengthening `spin_stay_in_place` to
−3.0 sharpened a flaw identified in review: it was the only spin term not
modulated by phase, so it charged full cost for transient translation during
the launch ramp—exactly when the robot needs to push against the ground to
generate angular momentum, and when entry momentum (up to 0.3 m/s) must be
**converted** into rotation. The cost is now multiplied by
`SPIN_LAUNCH_DRIFT_SCALE = 0.2` on `[0, ACCEL_END)`, then charged at full strength.
Unlike the shaping terms, it is deliberately **not** switched off at rest:
that is when stillness is the real criterion.

Step 4 (watch the movement) remains to be done, reserved for the human.

⚠️ These four tests (three new attenuation tests, one modified) have **not**
been run—the machine was reserved for something else at commit time. Run before
any long training run: `uv run --with pytest pytest tests/test_spin.py
tests/test_spin_cfg.py -q`.
