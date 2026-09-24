# 07 — Train a walking baseline

[Workbook](../README.md) · [Your record](data_parameters.md)

## Learn

The smoke test proved that learning can run. A longer experiment gives the policy
time to discover useful behaviour. These are different outcomes: a program can
finish successfully while its robot still moves badly.

The iteration count is a budget, not a certificate that the policy is finished.
This walking recipe changes parts of its training task over time: movement
smoothing strengthens through iteration 1,500, and standing practice and head
command ranges develop through iteration 2,000. More environments do not replace
those later iterations.

| Budget | What it is for |
| --- | --- |
| 1,000 iterations | Optional early inspection of learning; stops before the later curriculum stages finish. |
| 5,000 iterations | A first longer experiment, allowing training after those stages. |
| 50,000 iterations | The pinned configuration's default; neither a required duration nor a guarantee of success. |

Choose one budget. You do not need to do the 1,000-iteration run before a
5,000-iteration run. At 1,024 environments, 5,000 iterations collect about
122.9 million transitions. Actual wall-clock time depends on measured throughput,
startup and workload; the number of simulated ducks alone does not predict it.

## Do

First fill the **planned experiment** fields in your record. Use the batch size
selected in lesson 06. Define provisional behaviour checks before you see the
result. A useful starting exercise is:

- Stay upright at zero command for 30 seconds.
- Move forward for 10 seconds without falling.
- Respond to left and right turns in separate 10-second trials.
- After a forward request, accept zero command and settle without falling;
  choose a visible-motion tolerance and time limit to assess consistently.

These are suggested learning checks, not established hardware requirements. Save
your chosen durations and tolerances so later checkpoints face the same checks.

This complete command uses **1,024 environments and 5,000 iterations**. Change
`1024` if your recorded batch choice differs, and change `5000` only if you chose
another budget. Save the resulting command before executing it:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 1024 \
  --agent.max_iterations 5000 \
  --agent.run-name workbook-flat-baseline \
  --agent.logger tensorboard
```

Run it in the execution session. This deliberately starts a **fresh** walking
policy with your selected settings. It does not load the smoke or batch-check
policies. Keep the task and reward configuration unchanged for this baseline.

## Find the result

As soon as training starts, locate the new `<timestamp>_workbook-flat-baseline`
folder in the execution session's Explorer. Open `params/agent.yaml` and
`params/env.yaml` to confirm the budget, seed and environment count. Record the
actual folder and start time.

Open the charts with the same complete command used in lesson 06 if that viewer
is not already running:

```bash
uv run tensorboard --logdir logs/rsl_rl/velocity --port 6006
```

Select the baseline run. Inspect `Train/mean_reward`, `Train/mean_episode_length`
and `Perf/total_fps`. Save your observations in the record, not just a screenshot
of the most flattering point on a curve.

After training finishes, choose a saved checkpoint in Explorer and copy its Linux
path into your record. A fresh completed 5,000-iteration run normally ends with
`model_4999.pt`; confirm the file actually exists rather than assuming completion.
If you stopped early, record that and use the last checkpoint actually saved.

## Observe

For walking, fewer early falls can lengthen episodes, but reward and episode
length alone do not prove correct movement. A policy could learn to stand still
or exploit an easy reward. Lesson 08 tests actual behaviour.

Stop and investigate invalid numerical values or persistent physics errors.
Do not treat a larger iteration budget as a fix for a broken environment.

## Continue when

You have a saved baseline checkpoint, its actual settings and completion status,
and the evaluation checks you decided before training. You can now assess the
policy even after closing the training terminal.

[Next: evaluate the result](../08_evaluation/instructions.md)
