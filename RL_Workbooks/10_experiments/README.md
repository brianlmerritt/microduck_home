# RL experiments

Use this folder for an experiment's question, chosen settings, execution commands
and observed results. Carry out runs in the actual `microduck_rl` execution
session; keep the explanation and record here in the learning session.

| Experiment | Status / next action |
| --- | --- |
| [Initial flat walking](flat_velocity_rtx3090_4096.md) | Recorded: 1,024 environments, 5,000 iterations, browser playback; frequent falls. The filename says 4096, but the command and step count show 1024. GPU output shows an RTX 3090 Ti. |
| [Backlash VelStand to one policy for walking, sit/stand and recovery](backlash_velocity_flat_unified_policy/instructions.md) | User reports starting 4,096 environments / 50,000 iterations on 17 September 2026. Next: compare saved checkpoints; completion and behaviour remain unverified here. |

The original baseline record is retained as written. For the next experiment,
use its [blank data record](backlash_velocity_flat_unified_policy/data_parameters.md)
to distinguish proposed settings from measurements and completed checks.

For the running VelStand experiment, start with [checkpoint review](backlash_velocity_flat_unified_policy/checkpoint_review/instructions.md),
then [continuation and fine-tuning](backlash_velocity_flat_unified_policy/continue_training/instructions.md).
The parent project's [English RL reference](../../docs/rl_reference/README.md)
explains the task definitions and indexes the upstream design history.
