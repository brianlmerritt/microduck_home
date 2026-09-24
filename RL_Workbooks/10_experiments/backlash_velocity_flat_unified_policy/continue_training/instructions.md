# Continue learning — extra training or a changed experiment?

[Checkpoint review](../checkpoint_review/instructions.md) · [Your record](data_parameters.md) · [Experiment](../instructions.md)

Runner behaviour below was checked against the learning checkout's mjlab 1.3.0
and rsl-rl-lib 5.0.1 source; verify the execution session's versions as well.

## 1. Start from a behaviour you can describe

Use the checkpoint comparison to choose a source policy and one question. For
example: “Walking is stable and face-down recovery is improving; does another
2,000 iterations improve recovery without harming walking?” Save the source
checkpoint and its evaluation before starting anything new.

| Evidence | Next action |
| --- | --- |
| Useful skill still improving, with no material regression | A bounded continuation of the same task. |
| Earlier checkpoint performs better on the same tests | Keep that checkpoint; investigate the regression before adding time. |
| Same failure persists across widely separated checkpoints | Diagnose commands, start-state coverage, reward conflict or physics; more time may repeat the same failure. |
| Results came from different random conditions | Run fixed-scenario comparisons before changing training. |
| Want commanded sitting in addition to walking/recovery | Implement the unified task and its command/reward changes; extra VelStand iterations do not add a new objective. |

## 2. Rehearse loading the chosen checkpoint

In Explorer, choose the source folder under `logs/rsl_rl/velstand` and the
checkpoint inside it. Here `load-run` takes the **folder name only**, and
`load-checkpoint` takes the **filename only**. These are different from `play`,
which takes a full path.

First do five additional iterations to check the loading/output workflow:

```bash
uv run train Mjlab-VelStand-Flat-Backlash-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run "PASTE_SOURCE_RUN_FOLDER_NAME" \
  --agent.load-checkpoint "PASTE_SOURCE_MODEL_FILENAME.pt" \
  --agent.max_iterations 5 \
  --agent.run-name experiment-velstand-resume-check \
  --agent.logger tensorboard
```

Replace both quoted placeholders with actual names. Keep the environment count
and task unchanged for this rehearsal. Confirm the printed loaded path, then
find the **new** timestamped `_experiment-velstand-resume-check` folder. Record
its output checkpoint separately from the source.

The installed runner restores model/normalisation and optimiser state, iteration
information and a saved environment step counter used by curricula. It does not
restore every robot's pose, random generator state or in-flight episode; this
is continued learning, not an exact replay of the interrupted simulation.
Verify the execution session's versions and curriculum logs. Older checkpoints
without the saved environment counter need separate handling.

## 3. Run a bounded continuation when the evidence supports it

For the example question, use the same command with **`max_iterations 2000`**
and **`run-name experiment-velstand-extra-2000`**. Start from the original chosen
checkpoint again so the five-iteration rehearsal is not an unrecorded extra
stage. Save the fully substituted command in your record before running it.

For this runner, `max_iterations` on resume means **additional iterations**.
Putting `50000` here asks for another 50,000, not a lifetime cap of 50,000.
Use the actual saved filenames and logs for numbering; checkpoint indices are
zero-based and the loaded iteration can be reused at the beginning of resume.

The task is constructed from the current source and command-line settings;
resuming does not automatically reinstate the old run's `params/env.yaml`.
Keep the source/dependencies/settings matched for a continuation, and record
any difference as a changed experiment. A familiar run name alone never resumes
training.

Afterwards compare the source and new checkpoints using the same scenarios.
Keep the source even if the new mean reward is higher. Decide whether the
observed improvement meets the question you wrote down.

## 4. Fine-tune a specific weakness

Fine-tuning means starting from learned weights while deliberately changing the
training problem or learning settings. Change one main factor, give the run a
distinct name, and record why it should address the observed failure.

Examples to investigate, not ready-made fixes:

- Poor face-up recovery: inspect how often the policy actually sees that state
  and whether the reward/timeout makes attempts worthwhile.
- Walking lost after recovery training: inspect the mixture of walking and
  fallen starts and the relative contributions of rewards.
- Instability when looking up: compare fixed head requests, training coverage
  and the competing head/posture rewards.

Keep actor/critic interfaces, joint ordering, normalisation and model assumptions
compatible with the source checkpoint. For a changed task, decide explicitly
whether to load the actor only, the critic, the optimiser and the curriculum
counter. Stock `--agent.resume True` loads the full training state by default;
it is not an actor-only transfer command. The stock lookup also searches within
the selected task's experiment log group, so a new group will not automatically
find a VelStand checkpoint.

A changed curriculum may need a deliberate starting stage instead of the old
counter. An overridden learning rate must be verified after loading optimiser
state and during the adaptive schedule; a different number in a command is not
proof of the effective rate. These choices belong in the custom implementation
and a short rehearsal before a longer run.

The [unified-task specification](../unified_task_spec.md) defines the next
capability change. Its proposed task ID is not yet registered; finish its
implementation and compatibility checks before using it in a training command.

## 5. Export the selected result

Export whichever checkpoint passed the comparison, using its matching task:

```bash
uv run scripts/export.py Mjlab-VelStand-Flat-Backlash-MicroDuck \
  --checkpoint-file "PASTE_SELECTED_VELSTAND_CHECKPOINT_PATH" \
  --num-envs 1 \
  --onnx-file "experiment-velstand-selected.onnx"
```

Record which checkpoint produced that file. Give later exports distinct names
if retaining comparisons. Export preserves the selected policy and its
normalisation; it does not improve the skill. Keep `.pt` files for continued
training: an ONNX inference file is not a full training checkpoint.

Use matching mjlab playback for the first result. Stock CPU replay's backlash
observation and actuator parity still needs the work described in the unified
spec; loading an ONNX successfully alone does not complete that check.
