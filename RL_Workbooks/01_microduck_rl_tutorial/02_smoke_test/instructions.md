# 02 — Train and find the result

[Workbook](../README.md) · [Your record](data_parameters.md)

## Learn

A **policy** maps observations of the robot to joint actions. During training,
the robot tries actions in simulated environments, receives rewards and updates
the policy. This project uses PPO to make those updates.

A **run** is one training session. A **checkpoint** is a saved snapshot of the
policy and learning state during that session. One run can contain several
checkpoints, allowing you to compare earlier and later stages of learning.

The smoke test uses 64 simulated ducks, all contributing experience to one
shared policy. Each iteration collects 24 steps per duck and then updates the
networks. Five iterations check that this process works. They are not enough to
expect reliable walking.

## Do

**If your smoke run already exists:** open it in the execution session's Explorer
and go directly to **Find the result** below. Record the settings from its saved
files rather than rerunning training.

**For a new smoke test:** run this complete command in the execution session,
from `microduck_rl`:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 5 \
  --agent.run-name first-smoke \
  --agent.logger tensorboard
```

| Setting | What you are choosing |
| --- | --- |
| `64` environments | How many simulated ducks gather experience in parallel. |
| `5` iterations | How many rounds of collecting experience and learning to run. |
| `first-smoke` | A label to recognise the output folder; it does not select an earlier policy. |
| `tensorboard` | Local metric logging, with no online account needed. |

Wait for the command to finish. Save the exact command in your record sheet.
Starting this command again creates a fresh policy and another run folder.

## Find the result

In **the execution session's Explorer**, expand:

```text
microduck_rl
  logs
    rsl_rl
      velocity
        <timestamp>_first-smoke
```

Choose the folder for the run you just completed. The trainer also prints its
path beside `Logging experiment in directory:`; this is a cross-check if several
similar folders exist. Refresh Explorer if it has not shown the new folder yet.

The previously verified smoke example is `2026-09-11_13-23-20_first-smoke`:

| File | What it tells you |
| --- | --- |
| `model_0.pt` | Saved state after the first learning iteration. |
| `model_4.pt` | Saved state after the fifth learning iteration; numbering starts at zero. |
| `params/agent.yaml` | Learning settings, including seed, iteration budget and run name. |
| `params/env.yaml` | The simulated environment settings, including environment count. |
| `events.out.tfevents...` | Training metrics for a chart viewer; do not edit this binary file. |
| A `.onnx` file, if automatic export succeeded | An exported policy; its presence alone does not verify playback. |

Open the YAML files in VS Code and find `seed`, `num_steps_per_env`,
`max_iterations`, `num_envs` and `logger`. Record the actual values, run folder
and checkpoint path in `data_parameters.md`. A saved parameter file records
how the run was configured, not proof that every requested iteration completed.

Select `model_4.pt` for the next lesson if your five-iteration run completed.
Use Explorer's **Copy Path** to record its full Linux path. There is no need to
open the binary checkpoint or list files in a terminal.

## Observe

Look for completed iterations and numerical rewards/losses in the training
output. `nan`, `inf`, a traceback, or a missing final checkpoint needs
investigation. Do not infer good walking from a completed smoke test.

## Continue when

You can identify the completed run and its saved checkpoint, and you have recorded
the exact path that lesson 03 will load. A completed five-iteration run has a
saved final state; it does not yet have a useful gait.

[Next: watch the checkpoint](../03_playback/instructions.md)
