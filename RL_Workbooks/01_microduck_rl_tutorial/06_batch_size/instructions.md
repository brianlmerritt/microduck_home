# 06 — Choose the batch size

[Workbook](../README.md) · [Your record](data_parameters.md)

## Learn

More environments means more simulated ducks gathering experience for the same
policy. It does not mean training that many independent policies. With 24 steps
per duck per iteration, 256 environments produce 6,144 transitions per iteration;
1,024 produce 24,576. A transition is one observation/action/reward step.

Larger batches can use a GPU more effectively, but they also consume more memory
and change how much data the learning update receives. Four times as many ducks
does not guarantee four times the speed or the same learning result.

This lesson chooses a workable batch size before a longer experiment. It measures
resource use and throughput, not gait quality.

## Do

In the execution session, run this short test from `microduck_rl`:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 1024 \
  --agent.max_iterations 1024 \
  --agent.run-name batch-check-1024 \
  --agent.logger tensorboard
```

Keep your preferred GPU memory monitor visible while it runs. If you do not
already have one, open an additional execution-session terminal and run this
single-purpose monitor:

```bash
nvidia-smi -l 1
```

```md
## Actual output for 1024
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090 Ti     On  |   00000000:01:00.0  On |                  Off |
| 52%   55C    P2            112W /  450W | *3268MiB* / *24564MiB* |     38%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

Read **memory used / total** for the training GPU. The display includes other
GPU workloads, so note any that are active. Stop the monitor with Ctrl+C after
the tests. Its role is to establish memory headroom, not to manage project files.

If 256 completes with headroom, repeat the training command with **both** `1024`
values changed to `4096`. Consider 4,096 only after the smaller runs complete
comfortably. You do not have to use the largest possible count. Save each complete
training command in your record; there is no variable to remember between runs.

These tests start fresh policies. They do not overwrite or continue the smoke
policy. Do one test at a time so concurrent training does not distort the comparison.

## Find the result

In Explorer, each test creates a timestamped `batch-check-...` folder under
`logs/rsl_rl/velocity`. Its `params/env.yaml` records the actual environment
count. The terminal reports **Steps per second** for training iterations.

For visual comparison, start the local chart viewer in an execution terminal:

```bash
uv run tensorboard --logdir logs/rsl_rl/velocity --port 6006
```

Open its printed browser address in the execution environment. Select just the
batch-check runs and inspect `Perf/total_fps`. Despite the name, this counts
training transitions per second across environments, not rendered video frames.
The viewer also reads the TensorBoard event file in your original smoke run.

Record the run folder, environment count, highest memory usage you observed and
throughput range. The monitor samples once per second and may miss short memory
peaks; describe this as **observed usage**, not an exact allocator peak.

## Observe

Compare the later iterations as well as the first. Initial compilation and warmup
can dominate startup. Five iterations give an initial capacity/speed check, not
a precise prediction for every hour of a long training job.

If a candidate runs out of memory, keep the failure in the record and use a smaller
successful batch. Do not increase the workload while investigating a failed run.

## Continue when

Your record names a tested batch size, its observed memory/throughput, and why you
selected it. Close the chart viewer with Ctrl+C in its terminal if you are done.
The next lesson uses that saved number directly in a complete command.

[Next: train a walking baseline](../07_training/instructions.md)
