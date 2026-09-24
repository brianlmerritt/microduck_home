# 05 — Continue learning

[Workbook](../README.md) · [Your record](data_parameters.md)

## Learn

Running `train` again normally starts with a fresh policy. Reusing a run name does
not continue its learning. **Resume** explicitly loads a saved checkpoint,
including the learning state needed to continue the experiment.

The source checkpoint and the new run are different objects. Resumed training
writes a new timestamped output folder. Its results must be found there; watching
the original file afterwards still shows the original five-iteration policy.

For this pinned runner, `--agent.max_iterations 5` on a resumed invocation requests
five additional learning iterations. It is not a lifetime total of five. Use
the actual output filenames rather than predicting their numbering.

## Do

In the execution session's Explorer, select the smoke checkpoint used in the
previous lessons. This example uses its **folder name** and **file name**, rather
than the full path required by `play`:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.resume True \
  --agent.load-run "2026-09-11_13-23-20_first-smoke" \
  --agent.load-checkpoint "model_4.pt" \
  --agent.max_iterations 5 \
  --agent.run-name smoke-resumed \
  --agent.logger tensorboard
```

Replace the two quoted names if your selected run differs. The trainer searches
for that run below `logs/rsl_rl/velocity`; do not put the full checkpoint path
into `--agent.load-run`. Keep the task and environment count the same for this
first rehearsal. Save the completed command in your record before running it.

## Find the result

Check the startup output says it is loading the checkpoint you intended. After
the run finishes, refresh Explorer and open the new `<timestamp>_smoke-resumed`
folder under `logs/rsl_rl/velocity`.

Record the new folder and its highest numbered checkpoint. Open its
`params/agent.yaml` to confirm `resume`, the source run/checkpoint and requested
iteration count. Keep a clear **source checkpoint → resumed output checkpoint**
pair in your record. The original folder is still available for comparison.

If you want to view the resumed policy, copy the new checkpoint path into the
same `play` command from lesson 03. Save that fully substituted command rather
than relying on the previous terminal's history.

## Observe

Five additional iterations mainly demonstrate the resume mechanism. Similar or
poor movement is not evidence that resume failed. The useful checks here are
the loaded source path, the learning iterations and the newly saved state.

## Continue when

The correct source checkpoint was loaded, another short training session finished,
and its output is recorded separately from the original smoke run.

[Next: choose the batch size](../06_batch_size/instructions.md)
