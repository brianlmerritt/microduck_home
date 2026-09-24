# 01 — Learn the Microduck training workflow

By the end, you will be able to explain what training changes, find its saved
policy, watch it, export it, resume learning, and assess a longer experiment.
The deliverable is a trained checkpoint with a record of its settings and
observed behaviour. Simulation results do not establish hardware performance.

## Your two sessions

| Session | What you do there |
| --- | --- |
| Learning session — this workspace | Read and discuss the lessons; fill in the Markdown record sheets; save your complete commands and observations. |
| Execution session — the actual `microduck_rl` environment | Browse its files in VS Code Explorer; run training, playback and export; inspect its viewers and charts. |

The sessions may use the same computer, but the workbook does not assume they
share a filesystem, terminal or conversation. Paths recorded here must refer to
files in the **execution session**. Copy a completed command there to run it.
If you ask an assistant in that session to help, supply the lesson action and
the selected paths/settings from your record; it cannot be assumed to know them.

Use **Markdown: Open Preview to the Side** to read a lesson and keep its
`data_parameters.md` open for editing. VS Code's Markdown preview and document
outline support this layout. [VS Code Markdown guide](https://code.visualstudio.com/docs/languages/markdown)

## Start here

Your five-iteration smoke run was previously confirmed. Read lesson 01 to set up
the two-session workflow, then use lesson 02 to identify and record that existing
run. Continue at lesson 03 for playback. There is no need to repeat successful
training just to fill in a worksheet.

| Order | Lesson | What you will be able to explain |
| --- | --- | --- |
| 01 | [Identify the workspace](01_workspace/instructions.md) | Which session owns the files and runs the commands. |
| 02 | [Train and find the result](02_smoke_test/instructions.md) | How a training session produces a run folder and checkpoints. |
| 03 | [Watch the checkpoint](03_playback/instructions.md) | How the selected training result becomes a playback command. |
| 04 | [Export and replay](04_export/instructions.md) | How a checkpoint becomes an ONNX policy for a different runtime. |
| 05 | [Continue learning](05_resume/instructions.md) | How resuming differs from starting another fresh run. |
| 06 | [Choose the batch size](06_batch_size/instructions.md) | What parallel environments change and how to measure the cost. |
| 07 | [Train a walking baseline](07_training/instructions.md) | How to choose a training budget and define a useful outcome. |
| 08 | [Evaluate the result](08_evaluation/instructions.md) | How to judge behaviour alongside training metrics. |

Each lesson follows **Learn → Do → Find the result → Observe → Continue when**.
Open its record sheet before doing the exercise. Fill only the values you can
establish; a blank cell means you have not recorded that result yet.

## Commands that survive closing a terminal

All command blocks are for **WSL Bash in the execution session**, with
`microduck_rl` as the working directory. Use Explorer to open a terminal on that
folder. Commands use explicit values, not shell variables or environment setup
carried over from an earlier lesson.

Copy paths from **the execution session's Explorer**. An absolute Linux path is
the least ambiguous choice; quote it when inserting it into a command. A path
beginning with `C:\` or `\\wsl.localhost\` is a Windows path, not the Linux path
expected by these WSL commands. The lesson examples use paths relative to
`microduck_rl`, such as `logs/rsl_rl/velocity/.../model_4.pt`.

Save the complete command you actually used in the lesson's record sheet. A
record sheet is a notebook, not an executable configuration file: editing it
does not automatically change a running experiment or generate commands.

The workbook selects `--agent.logger tensorboard` explicitly for new training.
That writes local charts and checkpoints without a W&B login or environment
variable. It changes logging, not the walking task or learning algorithm. Your
existing smoke run's checkpoints remain usable regardless of its logger.

## Scope and references

Examples target `microduck_rl` revision
`2b581c641406a48346e696212930ea881c222c52`, with mjlab 1.3.0. Commands were checked
against the local source; this workbook was not validated by launching another
GPU run or opening a viewer. Record your execution environment in the blank
[project record](data_parameters.md).

The [original tutorial](../../MICRODUCK_RL_TUTORIAL.md) remains a reference for RL
concepts, source organisation, terrain, rewards and house modelling. This workbook
is the route through the first training experiment. Human-interaction work will
have its own project; it is not part of these exercises.
