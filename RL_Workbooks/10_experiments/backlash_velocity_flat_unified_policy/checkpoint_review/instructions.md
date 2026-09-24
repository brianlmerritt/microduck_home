# Checkpoint review — what changed as the duck learned?

[Experiment](../instructions.md) · [Your record](data_parameters.md) · [Next: continue or change training](../continue_training/instructions.md)

Viewer and loading details were checked against the learning checkout's mjlab
1.3.0 / rsl-rl-lib 5.0.1 source. Record the versions in the execution session.

## 1. Find the run in the execution session

In VS Code Explorer, open `microduck_rl/logs/rsl_rl/velstand`, then the folder
ending in `_experiment-backlash-flat-velstand-baseline`. Its timestamp comes
from the execution machine; do not guess it. Record the task ID from the launch
command, and the seed/settings from `params/agent.yaml` and `params/env.yaml`.
Those YAML files do not store the task ID itself.

The reported launch requests 4,096 environments and 50,000 iterations. That is
the budget, not a result. The pinned VelStand recipe saves every 250 iterations
and at normal completion. Expect names such as `model_1000.pt`, `model_5000.pt`
and, after an uninterrupted fresh 50,000-iteration run, `model_49999.pt`.
Use the files actually present. An interrupted run may only have its previous
periodic save; the largest number means latest saved, not best behaviour.

## 2. Open one saved policy

Choose an existing checkpoint in Explorer. Copy its Linux path into the quoted
field and save the complete command in your record. Run from `microduck_rl`:

```bash
uv run play Mjlab-VelStand-Flat-Backlash-MicroDuck \
  --checkpoint-file "PASTE_EXISTING_VELSTAND_CHECKPOINT_PATH" \
  --num-envs 1 \
  --viewer viser
```

Open the printed viewer address. Use **VelStand** here: its body-contact model
and fall handling are part of what this policy learned. The similar Velocity
task name selects a different environment even if the network can load.

After loading, use **Reset Environment** to begin a fresh observation. A new
checkpoint's saved curriculum counter can affect the next reset's starting-state
mix. Note whether the robot starts standing, crouched or fallen. A reset that
places it upright is not a successful get-up.

The browser viewer is a separate simulation and can consume GPU resources. The
first review is simplest after the training run ends; viewing alongside a live
run may affect throughput and available memory. This lesson does not require
interrupting training.

## 3. Compare several checkpoints at the same time

Use [play_checkpoints.py](play_checkpoints.py). It selects the **five most recent
saved checkpoints at multiples of 3,000 iterations** and starts one robot in a
separate viewer for each. For example, when the latest save is iteration 22,000,
the selection is 9,000, 12,000, 15,000, 18,000 and 21,000. It excludes iteration
zero and does not require every intermediate checkpoint file to exist.

The [settings file](checkpoint_viewers.json) already points to the discovered
`2026-09-17_16-51-02_experiment-backlash-flat-velstand-baseline` run. Open it in
VS Code to change the run, task, interval or persistent default count.
`project_dir` is relative to the settings file; `run_dir` is relative to the RL
project. If the execution checkout is elsewhere, put its absolute Linux path
in `project_dir`. Keep the launcher, helper and settings file together.

In the execution session's VS Code Explorer, right-click this **checkpoint_review**
folder and choose **Open in Integrated Terminal**. Run in that Linux/WSL terminal:

```bash
python3 play_checkpoints.py
```

For a different count, override it for this launch:

```bash
python3 play_checkpoints.py --num_checkpoints 3
```

The program starts viewers in sequence to initialise them, then leaves all of
them running concurrently. Ctrl+click each printed URL to open its browser tab;
each viewer is labelled with its checkpoint. The first port is normally 8100,
and occupied ports are skipped. For a remote VS Code connection, forward the
printed ports in the **Ports** panel. Each view stays attached to its selected
checkpoint; this launch mode does not show a checkpoint-switching tab.

Use **Commands → Twist → Enable**, then **Zero**, in each viewer before manual
motion tests. Put browser windows side by side if useful. Their commands and
simulations are independent; the launcher does not synchronise test scenarios.

Return to the launch terminal and press **Enter or Ctrl+C** to stop the viewers.
Closing browser tabs alone does not stop their simulations. The launcher stops
only its own viewer processes. If startup fails, it also cleans up viewers it
already started and prints the relevant log path.

A `viewers.md` link list and per-checkpoint logs are saved under the execution
project's `logs/checkpoint_viewers/<session>/` folder. The launcher uses the
existing `.venv/bin/python`; it does not install packages or change training
files. Each viewer adds GPU/CPU work alongside training, so reduce
`--num_checkpoints` if needed. Available training VRAM alone does not establish
how many independent viewers will fit.

Optional: `python3 play_checkpoints.py --dry-run` previews the selection without
starting simulations; `--interval 6000` changes the spacing. Selection is a
snapshot of existing files when launched. Rerun later to include newly saved
checkpoints. Keep task ID and run matched when editing the settings.

### Change checkpoints within a single stock viewer

If you used the single `uv run play` command in section 2 instead, the installed
mjlab Viser viewer has a **Checkpoints** tab:

1. Press **Sync** to refresh the `.pt` files in the same run folder.
2. Select a named file in the **Checkpoint** dropdown. Loading it resets the
   environment; confirm the printed loaded filename.
3. Observe and record the actual filename before moving to another candidate.

**Use Latest** chooses the latest available save. For comparisons, select exact
filenames so each observation remains attached to the policy that produced it.
If the execution session has a different mjlab version without this tab, stop
play with Ctrl+C, replace the checkpoint path in the saved command and reopen.

Start with a manageable spread: around **1,000, 5,000, 10,000, 20,000 and the
latest available** iteration. Skip unavailable files. Add checkpoints between
two candidates when you see a useful improvement or regression. There is no
need to inspect all roughly 200 saves from a 50,000-iteration run.

## 4. Understand what standard playback is showing

This is an exploratory comparison, not yet a test with identical conditions:

- The pinned Velocity recipe, inherited by VelStand, applies random pushes
  every **0.5–1.0 seconds in play**, versus **3–6 seconds in training**. A visible
  stumble may follow an injected disturbance. Record that pushes are enabled;
  do not describe this as undisturbed walking.
- Head/body pose commands are sampled too. The custom pose command currently
  does not expose its own GUI controls, so neutral head requests cannot be
  assumed just because the velocity arrow is zero.
- The installed runner restores `common_step_counter` when loading a checkpoint;
  curricula are applied on reset. Earlier and later checkpoints can therefore
  get different spawn mixtures and randomisation settings. Long viewing also
  advances the counter. Reset does not make random starting conditions identical.
- The stock episode is **20 simulated seconds**, and VelStand also resets after
  being continuously fallen/too low for **8 seconds**. Record resets separately
  from falls and recoveries. A 30-second continuous stand cannot be established
  across a timed reset.

These details may help explain the original frequent falls in playback; they do
not establish their cause. A quiet test with fixed commands is the next way to
separate basic gait quality from disturbance handling.

## 5. Observe the same small set of behaviours

In the viewer, open **Commands → Twist**, tick **Enable**, then press **Zero**.
The manual override starts disabled: pressing Zero or changing a slider without
Enable leaves the sampled commands in control. Set `lin_vel_x` for forward/back,
`ang_vel_z` for turns, and keep `lin_vel_y` and other unused axes at zero. For a
stand or stop request, keep Enable checked and press Zero. Check these controls
again after changing checkpoints or resetting. This overrides twist only;
head/body commands remain sampled as explained above.

For a first screen, take three attempts of each feasible condition for each
selected checkpoint. Start each attempt from a reset, wait until standing for
walking tests, and record any failure to reach that starting condition. Keep
simulation speed the same. Use simulation time when available; otherwise label
timings as visual estimates.

| Condition | Request / observation | Record |
| --- | --- | --- |
| Stand | Zero twist for 10 seconds. | Falls, drift, head motion, visible disturbances and resets. |
| Forward / backward | Separately request +0.1 / −0.1 m/s for up to 10 seconds. | Actual control value, direction, stability and whether it tracks. |
| Turn left / right | Separately request +0.3 / −0.3 rad/s yaw, zero translation, for up to 10 seconds. | Correct direction, falls and unintended travel. |
| Stop | Walk briefly, then zero twist and observe for 5 seconds within the same episode. | Time to visible settling and whether travel resumes. |
| Recovery | On an observed fall or fallen start, zero twist and observe the get-up and subsequent hold. | Start orientation, time to upright, hold duration and any reset. |
| Head movement | Observe naturally sampled head changes during standing/walking. | Whether head movement coincides with loss of balance; controlled gaze tests remain pending. |

Use the velocity controls the viewer exposes. If the requested value cannot be
set exactly, record the value actually used and reproduce it. A missing face-up
or side start means **untested**, not success. For recovery, wait for a fallen
start or natural disturbance rather than pretending a reset button can place a
specific pose. Do not switch checkpoints mid-attempt.

For a useful recovery screen, look for upright standing within 8 seconds and a
5-second hold afterwards, without a reset. Distinguish recovering under a walk
command from recovering after zeroing it. Formal evaluation will automate both.

## 6. Relate behaviour to the charts

Open TensorBoard in the execution session:

```bash
uv run tensorboard --logdir logs/rsl_rl/velstand --port 6006
```

Select this run and open the printed address. Compare the candidate iterations
using tracking errors, episode length, `fallen_too_long`, `nan_state` and recovery
reward terms if present. Copy the exact tag names from this installation.
`Episode_Reward/recovery_success` is a weighted training reward, not a measured
percentage of successful recoveries. A zero-weight term is uninformative about
the skill before its curriculum activates it.

Approximate curriculum landmarks from the pinned recipe:

| Iteration | What changes in the training problem |
| --- | --- |
| 0–500 | Walking first, with fall termination active. |
| 500 onward | Falls become opportunities to recover rather than immediate tilt resets. |
| 800 | Some starts are mid-recovery crouches. |
| 1,200 | Fallen-state penalty, recovery success reward and upward-motion reward switch on. |
| 1,500–2,500 | Fallen starts expand, eventually including both face-down and face-up poses. |

There are also inherited walking/head/randomisation curricula. Curves around
these boundaries reflect changes in task difficulty and reward weights as well
as learning. Later checkpoints are not guaranteed to be better at every skill.

## 7. Choose the next experiment

Record a leading candidate and a comparison checkpoint, with one sentence
explaining the observed trade-off. Preserve both. Examples of useful findings:

- “10k walks steadily, 20k gets up more often but drifts while standing.”
- “All candidates fail face-up recovery; face-down is improving.”
- “The latest looks better, but its starting conditions differed.”

The [fixed-scenario evaluator plan](../../../../docs/rl_reference/human_interaction_plan.md#2-make-checkpoint-comparisons-repeatable)
defines the next implementation step: identical starts, commands, seeds and
disturbances per candidate, measured success and tracking, and recordings. It is
not implemented by this worksheet. Use it to confirm a promising observation
before attributing improvement to a training change.

Continue to [extra training or a targeted change](../continue_training/instructions.md).
