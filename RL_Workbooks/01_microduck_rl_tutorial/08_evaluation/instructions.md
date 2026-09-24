# 08 — Evaluate the result

[Workbook](../README.md) · [Your record](data_parameters.md)

## Learn

Training reward describes the objective being optimised. Evaluation asks whether
the resulting behaviour is useful. Use both: a rising curve can coexist with
falls, poor turning or an unwanted response to zero command.

A later checkpoint is not automatically a better one. Compare checkpoints using
the same initial conditions, commands and durations. Record failures as well as
successes. The manual trials here are a first inspection, not a statistical claim
of reliability or evidence of safe hardware behaviour.

## Do

Bring the baseline checkpoint path and acceptance checks from lesson 07 into
this lesson's record. In the execution session, confirm that file in Explorer.

This example has **one path to replace**: paste the selected checkpoint's Linux
path between the quotes. Save the completed command before running it:

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file "PASTE_CHECKPOINT_PATH_FROM_EXPLORER" \
  --num-envs 1 \
  --viewer viser
```

Use the browser viewer for an initial look, then stop it with Ctrl+C. Export
the **same checkpoint**, replacing the same single path in this command:

```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file "PASTE_CHECKPOINT_PATH_FROM_EXPLORER" \
  --num-envs 1 \
  --onnx-file "workbook-baseline.onnx"
```

After successful export, run the file just created:

```bash
uv run scripts/infer_policy.py --walking "workbook-baseline.onnx" --new-cmd-obs
```

Apply the checks recorded before training. Keep keyboard focus on the execution
terminal: in default velocity mode, Up requests forward motion, A/E request
turning, Space requests zero movement and Q quits. Record the command values
printed by the script; a key is not a measurement of achieved speed.

For a clean starting state, quit and relaunch the replay between trials. Repeat
each scenario at least three times for this first comparison. Record what happened
on each attempt. Use the same protocol if you compare another checkpoint.

## Find the result

The ONNX file appears at the execution project root. Save its path alongside the
source checkpoint in your record. If comparing policies, choose a distinct ONNX
filename for each instead of overwriting the evidence you are comparing.

The main result of this lesson is your completed trial table: requested action,
observed movement, falls, duration and whether your stated criterion was met.
Include a short recording if useful and record where it is saved. A visual
estimate should be labelled as such; it is not an instrumented tracking metric.

## Observe

Look at the selected checkpoint's training charts alongside the trials. If reward
improved but turning did not, record both. Do not choose a conclusion from only
one of those observations.

Use the results to choose a next action:

| Finding | Reasonable next action |
| --- | --- |
| Behaviour meets your provisional checks | Preserve the checkpoint, ONNX and record as the baseline. |
| Behaviour is improving but still weak | Resume that checkpoint with a bounded additional budget and evaluate again. |
| Behaviour is stuck, unstable or exploiting the task | Inspect commands, reward terms and failure cases before increasing the budget. |
| Training-viewer behaviour differs sharply from ONNX replay | Check the selected file, exporter and observation format before judging learning. |

To resume, reuse the complete command pattern in lesson 05, but put in **this
baseline run's folder name**, **this checkpoint's filename**, the tested batch
size and an explicit additional iteration count. Save that command and identify
the new output folder before comparing its results. Do not accidentally resume
the five-iteration example just because its name is already in a command.

## Continue when

You have a recorded assessment and an evidence-based next action. A failed gait
test is still a useful completed evaluation. This workbook is complete when you
can perform and explain the whole training → checkpoint → playback → export →
resume → evaluation workflow. Declaring the gait acceptable additionally requires
meeting the criteria you set.

[Return to the workbook](../README.md)
