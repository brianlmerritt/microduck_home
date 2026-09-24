# Implementation brief — one policy for movement, posture and recovery

[Experiment](instructions.md) · [Data and results](data_parameters.md) · [Interaction plan](../../../docs/rl_reference/human_interaction_plan.md)

**Proposed work, not an existing registered task.** Implement this in the custom
environment/application repository used by the execution session. The root
workspace keeps the experiment record; upstream `microduck_rl` provides the
robot and reusable training components.

## Required result

One actor network, one checkpoint and one ONNX policy must support commanded
walking, commanded sit ↔ stand, and recovery from supported fallen poses. A
command layer can request movement/posture and apply stop rules, but it must not
select a different neural policy for each skill.

Use `Mjlab-Interaction-Flat-Backlash-MicroDuck` as the proposed registered ID and
`microduck_interaction_backlash` as its distinct experiment/log group. Reuse the
existing `VelStand` recipe as a starting point for walking plus recovery and the
`SitStand` posture definitions/reward ideas. Treat `StandUp` as a reference for
reset coverage and recovery evaluation. Combining these goals requires training
one new recipe; averaging checkpoint weights is not a training method here.

## One physical model for every behaviour

Use the **ground-contact backlash model** throughout the unified task:
`robot_groundcontact_backlash.xml`, represented by
`MICRODUCK_BACKLASH_ROBOT_CFG`. It supports the body contacts needed for sitting
and recovery. Apply `make_backlash_variant` with this explicit robot config.

The reported VelStand backlash run already uses this ground-contact model.
Velocity backlash instead uses `robot_walk_backlash.xml`, with different body
contacts. Treat any comparison with that task as also changing the contact
model; an improvement or regression cannot be attributed solely to adding a
posture objective. Do not swap the robot model between skills within one episode.

Preserve BAM actuation, the 50 Hz policy timing, observation normalisation and
the existing encoder-through-backlash convention. Resolve actuated joints by
their names/IDs; passive hinges are interspersed and must not become extra policy
actions. Include the same view of a joint in observations and related rewards.

Before training, verify that the standing and seated targets are stable under
the chosen model: hold each from perturbed starting states for at least three
seconds and inspect tilt and contacts, not height alone. Record the measured
target heights, joint poses and allowable tolerances.

## Command contract: retain meaningful velocity inputs

Keep the 61-value actor layout and 14 actions:

```text
angular velocity(3), projected gravity(3), joint position(14),
joint velocity(14), previous action(14),
twist(3), head pose(4), body pose(6)
```

**Proposed posture signal:** use `body_pose.z` as a desired height offset from
the measured nominal standing height, in metres. This retains its physical
meaning while extending the training range to the seated pose. The body-pose
order stays `[x, y, z, roll, pitch, yaw]`; the twist order stays `[vx, vy, yaw_rate]`.

| Request | Twist | Body-height request | Intended response |
| --- | --- | --- | --- |
| Stand / idle | All zero | Zero offset | Remain upright and stationary. |
| Walk / turn | Requested velocity within tested limits | Zero offset initially | Follow movement while maintaining balance. |
| Sit / hold sit | All zero | Measured seated height minus measured standing height | Reach the seated pose gently and hold it. |
| Stand from sitting | All zero | Return to zero offset | Rise and stabilise. |
| Recover from a fall | All zero | Zero offset | Reach stable standing from the supported fallen state. |

For illustration only, 0.115 m standing and 0.060 m sitting give an offset of
−0.055 m. **Measure on the selected model before using those numbers.** The
source configurations use different nominal-height assumptions in different
rewards; copying one value across them can create contradictory targets.

Initially train two posture endpoints and transitions between them. A learned
height request alone is not enough to define sitting: also reward the intended
joint pose, orientation, contact/support pattern and stillness. Add a deadband
or hysteresis around posture selection so noise cannot rapidly flip objectives.

Do not reuse SitStand's `twist[0] = sit_flag`: it collides with forward velocity.
Replace its posture-command sampler and any rewards that read that flag with
the unified height-target interpretation. Replace the inherited body-pose
curriculum too; otherwise it can overwrite the extended seated range with the
walking recipe's tiny offsets.

The current Velocity/VelStand body-pose tracking reward has weight zero. The
slot's existence does not establish learned height control: enabling and
training the posture objective is part of this implementation.

During a sit/stand transition, command generation suppresses translation and
turning. A later walking request first requests upright posture, waits for a
measured stable-standing condition, then permits travel. Recovery evaluation
starts with a stand request. On an unexpected fall, the application cancels
travel and requests recovery; after recovery it remains idle until a deliberate
new movement request. All these requests feed the same actor.

The observation shape staying at 61 does not make old policies understand the
extended posture range. Record this contract and its calibrated targets in the
custom task and exported metadata; test it in the training and replay adapters.

## Rewards and termination rules that do not fight each other

Start from the existing recipes, then audit every inherited term against these
cases before combining weights:

| Case | Reward / termination requirement |
| --- | --- |
| Upright walking | Retain velocity tracking and gait quality. Posture terms must not reward standing still instead of fulfilling a movement request. |
| Intentionally seated and upright | Reward the correct stable seated pose. Do not apply an unconditional standing-height penalty or low-height failure timeout. |
| Sitting down / rising | Reward progress toward the commanded posture; introduce gentleness without making all movement unprofitable. |
| Fallen while asked to stand | Permit time to discover recovery; retain failure timeouts for genuinely failed recovery. |
| Deliberately falling or repeatedly crouching | Do not let repeated recovery rewards outperform staying upright and completing the requested behaviour. |
| Resets and command changes | Reset posture/recovery bookkeeping correctly, without accumulating randomisation or paying a spurious progress bonus. |

In the pinned VelStand recipe, `fallen_too_long` can trigger below 0.08 m. That
would conflict with a correct seated target near 0.06 m. Make the condition
posture-aware: intended sitting must be distinguished from a genuinely fallen
or stuck state using orientation, pose and support as well as height. Do not
simply disable failure detection for everything below the standing height.

Likewise, condition standing-pose and gait rewards on the requested behaviour.
Disable gait/foot-air-time incentives during seated holds. Avoid world-position
or spawn-relative yaw penalties that oppose purposeful locomotion; use the
appropriate moving-frame posture measures. Check signs of self-negating
penalties before transferring their weights.

Recovery rewards need bounded progress/success credit and a clear completion
condition, rather than continuous positive payment for remaining fallen. Test
for deliberate falling, indefinite crouching, refusing to sit and sitting while
ignoring a walk request. Log each behaviour's metrics separately: a good mean
reward must not hide a missing skill.

## Curriculum within one task

Keep one observation/action interface and one physical model across stages.
Changes below alter the sampled requests/start states and reward schedule, not
which neural network controls the robot.

1. **Establish full-contact backlash walking.** Recheck the isolated velocity
   baseline on this model and ensure zero-command standing is practised.
2. **Introduce seated endpoints and transitions.** Include standing→sit,
   sitting→stand, standing holds and seated holds. Retain substantial walking
   experience so learning the new posture does not erase the gait.
3. **Expand recovery starts.** Progress from near-standing and mid-rise states
   to face-down, face-up and explicitly supported side starts. Preserve walking
   and posture transitions in each training batch.
4. **Practise mixed episodes.** Change commands within an episode: travel,
   stop, sit, rise and travel again; include disturbances and recovery followed
   by stationary standing. Re-evaluate all skills after each curriculum change.

Choose and record the stage boundaries and mixture proportions from observed
learning, rather than assuming the original independent-task schedules remain
valid together. Maintain nonzero sampling of required command inputs from the
start; keep the other command slots and their conventions intact.

A warm start from a compatible prior checkpoint is an optional experiment, not
a merge of three skill networks. Check actor, critic, normalisers, joint model
and curriculum counters when using it. First demonstrate resume within the same
custom task before comparing a cross-task warm start.

## Stop and interaction boundary

The movement actor receives proprioception and commands; it does not recognise
people or palms. The future application owns target selection, perception and
a persistent stop latch, and turns those into the movement/posture requests.

For the first simulation implementation, define stop as **cancel travel and new
expressive requests while allowing balancing and an active transition to settle
into a stable posture**. An active recovery may finish into stationary standing;
it must not restore a previous follow command. Record this operating definition
and the measured stop/settling behaviour before treating it as accepted.

Do not equate stop with pausing inference or freezing the last motor target.
Do not automatically initiate a new sit gesture on stop; stopping must work
without requiring the full sit manoeuvre. Clearing the perceived palm does not
clear the latch: a deliberate restart is needed.

Test synthetic stop requests during each behaviour before adding perception.
When recognition is implemented later, the same command rules apply to a
recognised stop from any person, including a bystander.

## Export and replay work required

Export through the upstream normalisation-aware exporter after the custom task
is registered. Confirm a single ONNX actor with 61 inputs and 14 outputs, and
evaluate every skill using that same file.

The stock `scripts/infer_policy.py` supports multiple sessions via `--walking`,
`--standing` and `--sitstand`. That switching path does not demonstrate this
single-policy requirement, and its sit command currently writes the velocity
slot. Implement a replay adapter for the unified command contract that loads
only the combined ONNX policy and exposes velocity plus seated/upright requests.

For matched CPU replay, also:

- [ ] Use the ground-contact backlash scene and the same calibrated poses.
- [ ] Match joint-position/velocity observations to the training encoder view,
  including passive backlash displacement; servo indexing alone is insufficient.
- [ ] Match BAM feedback and actuation through the backlash model, not merely
  the visible XML geometry.
- [ ] Preserve policy timing, action scaling, previous-action history and
  normalisation. Do not add action filtering without a matched training design.
- [ ] Compare observations and policy outputs on fixed states against the
  training environment, then compare motion on repeatable command sequences.

Until that adapter is validated, use the registered task's mjlab playback for
matched-physics tests and label stock CPU playback as a separate approximation.

## Implementation and evaluation checks

- [ ] Custom task appears in `list-envs` from a reproducible project install.
- [ ] Observation ordering, command units and the 61/14 shape contract are tested.
- [ ] Joint selectors exclude passive hinges while encoder observations include
  their correct physical contribution.
- [ ] Pose/height calibration is recorded and both target poses are stable.
- [ ] Each reward's sign and behaviour dependence is tested at representative
  standing, walking, seated, transitioning and fallen states.
- [ ] Intended sitting survives the failure timeout; a genuine failed recovery
  still terminates; recovery credit cannot be repeatedly farmed by sitting.
- [ ] Command sampler and curriculum exercise all required behaviours. A normal
  five-iteration smoke test only covers the initial curriculum; use explicit
  test configurations to smoke-test later stages as well.
- [ ] A deterministic evaluator can select initial pose, commanded sequence and
  seed, and reports fall rate, tracking error, posture success and recovery time.
- [ ] Resume restores the chosen custom checkpoint and its curriculum state.
- [ ] One exported policy passes the same mixed sequence without resetting the
  robot or swapping policies between skills.
- [ ] CPU parity and stop-latch tests pass before claiming an integration result.

## Source references for the implementation session

- [Task registrations](../../../microduck_rl/src/mjlab_microduck/tasks/__init__.py)
- [Walking configuration and command layout](../../../microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)
- [VelStand walking/recovery recipe](../../../microduck_rl/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py)
- [SitStand posture recipe](../../../microduck_rl/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py)
- [StandUp reset and recovery recipe](../../../microduck_rl/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py)
- [Backlash model/observation wrapper](../../../microduck_rl/src/mjlab_microduck/tasks/backlash.py)
- [Existing CPU inference implementation](../../../microduck_rl/scripts/infer_policy.py)

These references were inspected at revision
`2b581c641406a48346e696212930ea881c222c52`. Recheck the execution checkout if it differs.
