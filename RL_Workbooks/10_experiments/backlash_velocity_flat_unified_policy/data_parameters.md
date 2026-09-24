# Experiment record — backlash VelStand and one combined policy

[Instructions](instructions.md) · [Checkpoint scorecard](checkpoint_review/data_parameters.md) · [Continuation record](continue_training/data_parameters.md) · [Unified-task specification](unified_task_spec.md)

Fill values from the execution session. Results start blank. A proposed setting
in the instructions is not a measured result. Save exact commands and Linux
paths so another terminal or session can reproduce the selected experiment.

## Reported launch — 17 September 2026

The user reports starting the command below in the execution session. These are
reported inputs; no completed iterations, runtime or behaviour are inferred.

```bash
uv run train Mjlab-VelStand-Flat-Backlash-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 50000 \
  --agent.run-name experiment-backlash-flat-velstand-baseline \
  --agent.logger tensorboard
```

Outputs are expected under `logs/rsl_rl/velstand`. Fill the actual path and
results below when available. Source inspected for the documentation:
`2b581c641406a48346e696212930ea881c222c52`; verify the execution checkout too.

## Reported memory — 18 September 2026

During the reported 4,096-environment VelStand run, the user observed
**7,884 MiB used / 24,564 MiB total** GPU memory: approximately **32.1% used**,
with **16,680 MiB remaining** at that snapshot. This is observed usage, not a
measured peak or a verified per-process allocation. The exact iteration and
whether the timing report below was captured simultaneously remain unrecorded.

## Reported training timings — 18 September 2026

| Item | User-reported value |
| --- | --- |
| Iteration time | 3.41 seconds |
| Elapsed training time | 17:04:22 |
| Displayed logger ETA | 05:31:06; omits whole days (see correction below). |
| Subsequently reported progress | Iteration 18,334 / 50,000. |

**Correction after checking the logger source:** the installed rsl-rl-lib 5.0.1
formats elapsed time and ETA with `time.strftime("%H:%M:%S", time.gmtime(...))`.
Hours wrap at 24 and the day count is omitted. At the reported progress and
iteration speed, the displayed ETA means approximately **29:31:06 remaining**,
not five hours. Added to the reported elapsed time, this projects **46:35:28**
total. The earlier interpretation of 22:35:28 was incorrect.

The iteration count was supplied in a subsequent message, so these readings
need not be from exactly the same log block. As a cross-check, after zero-based
iteration 18,334 there are 31,665 iterations left: at 3.41 seconds each, about
**29:59:38**. The run is approximately **36.7% complete by iteration count**.
The logger estimates from its accumulated average, whereas this cross-check
uses the latest iteration. Both support roughly 30 hours remaining.

The estimate can change, and the logger's accumulated collection/learning time
does not include all startup, logging and checkpoint-saving overhead. Its elapsed
display will also wrap after 24 hours. The diagnosis is based on reading and
reproducing the formatter; training code and the running process were not changed.

Assuming the reported 4,096 environments and the recipe's 24 steps per
environment, each iteration collects **98,304 environment transitions**.
Dividing by the reported 3.41 seconds gives approximately **28,828 transitions
per second** for that iteration. This is a derived estimate, not a directly
reported throughput metric or a steady-state benchmark.

## Baseline and comparison settings

| Item | Value |
| --- | --- |
| Execution project / machine | |
| GPU model and driver | |
| Source revision and local modifications | |
| Original baseline checkpoint path | |
| Original baseline seed, from saved settings | |
| Original baseline environment count, from saved settings | |
| Chosen backlash training environment count | |
| Chosen backlash iteration budget | |
| Actual VelStand run folder / execution status | |
| Final completed iteration / interruption reason, if any | |
| Effective save interval | |
| Chosen seed / reason for any baseline difference | |
| Evaluation commands and durations | |
| Acceptance thresholds fixed before training | |
| Initial-state / reset procedure | |

## Run records

Add a row per smoke, capacity, full-training or resumed run. Record memory as
observed usage unless you have actually measured a peak.

| Purpose | Task ID | Run folder | Environments | Iterations completed | Seed | Duration | Observed memory | Steps/s | Selected checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

## Walking comparison trials

Use one row per attempt. Record the model used and whether a fall triggered a
reset; do not count a reset as learned recovery.

| Policy / checkpoint | Model / task | Condition | Attempt | Actual command / units | Duration | Falls / resets | Observed behaviour | Criterion met? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

## Unified task implementation

| Item | Value |
| --- | --- |
| Custom repository / revision | |
| Registered task ID | |
| Experiment / log group | |
| Robot model and scene | |
| Measured nominal standing height (m) | |
| Measured seated height (m) | |
| Seated body-height offset (m) | |
| Pose, tilt and motion tolerances | |
| Command contract / version | |
| Curriculum stages and request/reset mixtures | |
| Warm-start source, if used, and compatibility checks | |
| Reward/termination test results | |
| Initial and later-stage smoke-test results | |
| Capacity result for the unified task | |
| Resume source and output checkpoint | |
| Selected unified checkpoint path | |
| Exported ONNX path | |
| Verified actor input/output shapes | |
| Evidence that all behaviours use the same policy | |
| CPU observation / BAM / motion parity results | |
| Stop and settling operating definition | |

## Unified-policy behaviour trials

Use one row per attempt for walking, sit/stand, recovery, mixed sequences and
synthetic stop cases. Keep unsupported initial states visible in the results.

| Checkpoint | Start state | Requested sequence | Attempt | Completion / settling time | Falls | Stop latch / restart outcome | Criterion met? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

## Exact commands used

Copy complete commands with actual paths below. Add a short label for each run,
playback, export or evaluation command.

```bash

```

## Interpretation and next action

| Item | Value |
| --- | --- |
| Main observed improvement or regression | |
| Which criteria pass / fail / remain untested | |
| Evidence limitations | |
| Next change and why | |
| Next people/sensor investigation | |

## Additional observations / recordings
