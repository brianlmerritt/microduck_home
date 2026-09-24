# Backlash VelStand → one policy for walking, sit/stand and recovery

[Experiments](../README.md) · [Data and results](data_parameters.md) · [English reference](../../../docs/rl_reference/README.md) · [Unified-task specification](unified_task_spec.md)

**17 September 2026: the user reports starting the VelStand run below.** Its
completion and behaviour have not been inspected from this learning session.
The custom task adding commanded sitting is still to be implemented.

Use this session to read, discuss and record. Run commands in your separate
execution session, with its terminal at `microduck_rl`. Use that session's VS
Code Explorer to find outputs; save complete commands in the record sheets.

## 1. The run now being explored

```bash
uv run train Mjlab-VelStand-Flat-Backlash-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 50000 \
  --agent.run-name experiment-backlash-flat-velstand-baseline \
  --agent.logger tensorboard
```

### Memory of the RTX3090 is 7884MiB /  24564MiB so plenty of headroom

This is the user's reported launch command, not a second run to start. It
selects **walking and fall recovery in one policy**, with the ground-contact
backlash robot. It does not include commanded sitting or recognising people.

| Setting | Meaning |
| --- | --- |
| `VelStand` | Walking plus recovery training; different rewards, starting states and body contacts from plain Velocity. |
| `Flat` | A flat floor. |
| `Backlash` | Mechanical play in the servo joints, with matching encoder observations and actuator feedback. |
| 4,096 environments | Robots collecting experience in parallel. |
| 50,000 iterations | Requested training budget. The pinned VelStand default is 20,000; this command deliberately overrides it. |
| TensorBoard | Local charts; no W&B run ID is needed for the local checkpoint workflow. |
| Save interval | 250 iterations in the pinned recipe, plus a save at normal completion. |

A long run provides a sequence of policies to compare. More iterations do not
guarantee monotonically better walking or recovery. Choose by observed behaviour,
keeping earlier candidates if later training loses a useful skill.

**Reading long-run timings:** the installed rsl-rl-lib 5.0.1 logger omits days
from elapsed time and ETA: `29:31:06` remaining displays as `05:31:06`. Use the
iteration count and seconds per iteration as a cross-check. For this run,
18,334 / 50,000 at about 3.4 seconds per iteration implies roughly 30 hours
remaining. See the [timing correction](data_parameters.md#reported-training-timings--18-september-2026).

## 2. Find the outputs, then open the checkpoint review

In the execution session's Explorer, find:

```text
microduck_rl/
  logs/rsl_rl/velstand/
    <timestamp>_experiment-backlash-flat-velstand-baseline/
      params/env.yaml
      params/agent.yaml
      model_....pt
      events.out.tfevents....
```

The log group is **`velstand`**, not `velocity`. Record the actual folder and
settings in [data_parameters.md](data_parameters.md). The timestamp, completed
iteration count and checkpoint filenames must come from the execution session.

**When you return, start the [checkpoint review](checkpoint_review/instructions.md).**
It provides one playback command with the correct VelStand task, a Viser
checkpoint-dropdown workflow, a short observation battery and a blank scorecard.
Start with a few checkpoints; inspect more around any clear improvement or
regression.

To compare several at once, use the
[multi-checkpoint launcher](checkpoint_review/instructions.md#3-compare-several-checkpoints-at-the-same-time).
It defaults to the newest five checkpoints at 3,000-iteration intervals;
`--num_checkpoints` changes the number of simultaneous viewers.

Standard playback includes frequent pushes and sampled head commands. It is a
useful first look, but controlled evaluation needs fixed scenarios. The review
explains how to distinguish actual recovery from an automatic reset.

## 3. Use the results to choose extra training or a targeted change

After recording checkpoint behaviour, follow
[continue training](continue_training/instructions.md). It explains:

- how to resume one chosen checkpoint into a new output folder;
- why `max_iterations` on resume means additional iterations;
- how to compare the source and continued policy;
- what changes when fine-tuning or transferring to the custom task;
- how to export the selected checkpoint using its matching task.

A repeatable evaluator is the next implementation deliverable in the
[interaction plan](../../../docs/rl_reference/human_interaction_plan.md).
It will freeze starts, commands and disturbances across checkpoints, report
walking and recovery separately, and supply evidence for the next change.

## 4. Keep the original baseline as context

The [original record](../flat_velocity_rtx3090_4096.md) gives us this starting point:

| Item | Recorded evidence |
| --- | --- |
| Task | `Mjlab-Velocity-Flat-MicroDuck`. |
| Environment count / budget | 1,024 environments, 5,000 iterations, as shown by the command and step count. |
| GPU | RTX 3090 Ti; 24,564 MiB total reported. |
| Resource observation | 3,306 MiB used in one snapshot; 10,173 steps/s at iteration 4. Neither establishes a peak or settled throughput. |
| Elapsed training time | 02:36:54. |
| Checkpoint | `logs/rsl_rl/velocity/2026-09-17_07-00-22_experiment-flat-baseline/model_4999.pt`. |
| Browser observation | Followed the command arrow but fell frequently. |

The new run changes the task, contact model, backlash, environment count and
training budget together. It can answer “is this a more useful policy for our
goal?”; it cannot isolate which change caused a difference. Replay each
checkpoint with its own task ID, and label that as a comparison in each policy's
own model.

If isolating backlash becomes useful later, run a separate matched experiment:
plain Velocity versus backlash Velocity, with the same seed, 1,024 environments,
5,000 iterations and fixed evaluation conditions. That optional experiment is
not a prerequisite to learning from the VelStand run already started.

## 5. Build the learning problem for one combined policy

The next deliverable is **one policy and one exported ONNX file** that can walk,
sit, stand from sitting and recover from a fall. The existing task families are
useful source material:

| Existing recipe | What to reuse |
| --- | --- |
| `Mjlab-VelStand-Flat-Backlash-MicroDuck` | Walking and fall recovery in one policy; appropriate body-contact model. |
| `Mjlab-SitStand-Flat-Backlash-MicroDuck` | Sit/stand target poses, transitions, gentleness and posture-dependent rewards. |
| `Mjlab-StandUp-Flat-Backlash-MicroDuck` | Recovery start states, success measures and curriculum ideas. |

The registered `VelStand` task does not include the full commanded sit/stand
behaviour. The registered `SitStand` task repurposes the first velocity slot as
a posture flag. We need a custom recipe with one consistent command meaning,
compatible rewards and a training mix that retains all skills.

Use the [unified-task specification](unified_task_spec.md) as the implementation
brief in the execution session. Proposed ID:
`Mjlab-Interaction-Flat-Backlash-MicroDuck`. **It is not registered yet.** Do not
replace a working task name with this ID until the custom package is implemented
and its registration is confirmed by `list-envs`.

- [ ] Create the custom environment/application repository and add it as a
  submodule of this workspace, following TODO 3's project structure.
- [ ] Implement one command-conditioned task using the specified body-height
  convention, full body-contact backlash model and 61-observation/14-action
  interface.
- [ ] Incorporate sitting and recovery without making intended sitting trigger
  the inherited fallen-state timeout, or paying recovery reward for sitting.
- [ ] Add deterministic command/initial-state evaluation and meaningful tests
  for rewards, transitions, joint selection and observation compatibility.
- [ ] Demonstrate a five-iteration, 64-environment smoke run of that new task,
  including all behaviours through short test curricula.
- [ ] Check capacity again on the new model/task before choosing a full batch.
- [ ] Train the same policy through the staged curriculum; record separate
  walking, sit/stand and recovery results after each stage.
- [ ] Export one ONNX file and validate matching CPU physics, encoder feedback
  and command handling before using replay as an acceptance result.

The initial budget is a bounded experiment, not a fixed completion promise.
Use a 5,000-iteration starting budget after the smoke/capacity checks; pause to
inspect each skill and adjust curriculum pacing if adding sitting or recovery
causes walking to regress. Any warm start from a prior policy is a separately
recorded fine-tuning choice, with model/interface compatibility checked first.

## 6. Evaluate the one-policy result

Use one checkpoint for every trial. The controller may select commands and apply
the stop rules, but all joint actions must come from that same learned policy.

| Test | Initial target, to record before evaluation |
| --- | --- |
| Walking and stopping | Repeat the checkpoint review's walking/stop cases under the fixed-scenario evaluator on the unified physical model. |
| Sit → hold → stand | Ten command cycles; reach each requested posture within 5 seconds, hold for 10 seconds, no unintended fall. |
| Fall recovery | Five trials each from supported face-down, face-up and side starts; reach stable standing within 8 seconds and hold for 5 seconds. |
| Mixed sequence | Walk → zero velocity → sit → hold → stand → walk, repeated five times with no reset or change of policy. |
| Interrupted movement | A stop request during walking, a posture transition or recovery suppresses new travel and preserves the stop latch through the transition. |
| No accidental restart | Lowering a simulated stop hand or completing recovery cannot restore an old following command. |

Posture completion and “stable” must use calibrated height, orientation and
motion thresholds on this robot model; record them with the results. The times
above are provisional experiment targets, not claims about present capability.
Set the evaluator's episode timeout to contain each whole sequence; the stock
20-second timeout cannot contain a long mixed sequence or several posture holds.
If side recovery is not supported, record it as unsupported rather than folding
it into an overall success score.

Human stop signalling is not implemented by these movement policies. Initially
inject a synthetic stop event to test command handling; keep its result separate
from actual flat-palm recognition. The specification defines how stop requests
interact with balance, posture transitions and fall recovery.

## 7. Later TODO — people, sensing and self-contained interaction

This is the next application layer from TODO 3. Define observable behaviours
first; ToF and camera use remain choices to investigate, not assumed hardware.
The first implementation target is following a selected person and a persistent
flat-palm stop, with expressive actions added only after those work.

Use the [human-interaction implementation plan](../../../docs/rl_reference/human_interaction_plan.md)
for the sequence and evidence needed. Existing upstream camera/ToF simulation
is described there; it is not yet connected to this experiment's policy.

- [ ] Add simulated people with controllable paths, height, posture and hand
  presentation. Include children through adults, adults bending/crouching,
  occlusion, crossing people and a bystander's stop signal.
- [ ] Start with labelled person/gesture events and known simulated positions
  to debug the application. Label these results as using simulator ground truth;
  they do not validate perception.
- [ ] Define the deliberate follow-start action and target-selection rules.
  Record following distance/speed limits, target loss and obstruction responses;
  do not silently switch targets.
- [ ] Implement flat-palm stop from the target or a bystander. Keep stop latched
  until deliberate restart and give it priority over following and expression.
- [ ] Investigate **time-of-flight (ToF)** range sensing for proximity/distance:
  specify placement, field of view, range, noise, latency, invalid readings and
  occlusion. Range alone does not identify a person or recognise a flat palm.
- [ ] Investigate a **camera** for person selection/tracking and palm recognition:
  specify mounting, field of view, image rate, latency, lighting and occlusion.
  Rendering a person in simulation does not add camera input to the gait policy.
- [ ] Route perception into target velocity, posture and stop requests initially,
  preserving the tested movement interface. Adding raw sensor inputs to the
  neural policy would be a separate observation/training/export change.
- [ ] Add obstacles and representative contact geometry; test blocked movement,
  lost targets and unreliable observations. Do not use apparent human size as
  the sole distance estimate across children, adults and posture changes.
- [ ] Move from synthetic events to simulated sensor observations, then recorded
  real input and finally robot trials. Keep the evidence from each stage distinct.
- [ ] Add a small agreed set of expressive reactions and check that they cannot
  override stopping, balance or target retention.
- [ ] Measure sustained onboard memory, computation and response latency while
  movement control runs. Desktop GPU success alone does not prove self-contained
  operation on the robot.

Completing the movement experiment supports TODO 3, but does not complete person
following, gesture recognition or physical-robot acceptance.
