# Where the Microduck environments live

Checked on 2026-09-17 against `microduck_rl` commit `2b581c641406a48346e696212930ea881c222c52`. This guide describes that source; it does not certify the behaviour of a trained checkpoint.

Playback and resume behaviour was also checked against the locally installed
**mjlab 1.3.0** and **rsl-rl-lib 5.0.1** source. Record the execution session's
versions too; the task commit alone does not identify those runtime components.

An **environment** is the training situation and its rules: the robot model, ground, commands, information available to the policy, rewards, disturbances, and when an attempt ends. A **policy** is the neural network learned by training in that environment. A **checkpoint** saves a particular stage of that learning. Naming a task selects a recipe; selecting a checkpoint chooses the learned behaviour you actually watch.

## Start with the task name

For the current experiment, find `Mjlab-VelStand-Flat-Backlash-MicroDuck` in the [task registry](../../microduck_rl/src/mjlab_microduck/tasks/__init__.py). In VS Code, open that link and use **Ctrl+F** for the exact name.

| Part of the name | Meaning here |
|---|---|
| `VelStand` | Commanded walking plus learning to recover from falls, using one policy. |
| `Flat` | A plane as the training ground. This does not mean the task has no disturbances. |
| `Rough` | For task families that register it, generated uneven terrain. Check the selected factory; terrain settings differ by family. |
| `Backlash` | Adds simulated mechanical gear play to each servo, with matching encoder feedback. |
| `MicroDuck` | The robot/task family identifier. Preserve the spelling in commands. |

`Flat`, `Backlash`, and `VelStand` describe different choices. Backlash does not mean rough ground, and adding backlash does not by itself teach recovery.

The registry connects a task ID to **training environment settings**, **playback environment settings**, and **PPO learning settings**. Its `_BACKLASH_TASKS` table applies the backlash wrapper to the appropriate base factory. A document proposing a name does not register it.

## Follow the configuration to its implementation

| Open this file in VS Code | What you learn there |
|---|---|
| [Task registry](../../microduck_rl/src/mjlab_microduck/tasks/__init__.py) | The exact runnable IDs and which factories/configurations they select. |
| [Velocity configuration](../../microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py) | The walking recipe and much of the shared setup: observations, commands, rewards, randomisation, terrain and learning settings. |
| [VelStand configuration](../../microduck_rl/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py) | How the walking recipe gains recovery, ground-contact geometry, different starting states and staged training. |
| [MDP functions](../../microduck_rl/src/mjlab_microduck/tasks/mdp.py) | The actual calculations referenced by the settings: rewards, command sampling, reset events, observations and curricula. |
| [Robot constants](../../microduck_rl/src/mjlab_microduck/robot/microduck_constants.py) | Which robot XML is loaded, initial joint pose, collision choices and actuator configuration. |
| [Robot models](../../microduck_rl/src/mjlab_microduck/robot/microduck/) | MJCF/XML geometry, joints, contact shapes and physical properties. |
| [Backlash wrapper](../../microduck_rl/src/mjlab_microduck/tasks/backlash.py) | The robot swap, encoder-view observations and passive-joint handling. |
| [Actuator implementation](../../microduck_rl/src/mjlab_microduck/actuator/friction_dr_bam.py) | BAM servo behaviour, friction randomisation and encoder-through-backlash feedback. |

In a configuration, `commands` defines what is requested; `observations` defines what the network receives; `actions` defines what it controls; `rewards` defines what training encourages. `events` handles resets and disturbances, `terminations` ends attempts, and `curriculum` changes difficulty or reward weights as training advances.

For example, seeing `func=microduck_mdp.upright_progress` under `rewards` means the setting chooses the function and its weight. Open that function in `mdp.py` to see exactly how progress is calculated. This matters because a reward's name alone does not tell you its sign, units, or when it applies.

## Which task contributes which behaviour?

| Existing family | What its recipe teaches | Role in this project |
|---|---|---|
| [Velocity](../../microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py) | Follow movement commands while balancing and responding to head-pose commands. Falls normally end the attempt. | Walking reference and the basis of VelStand. |
| [VelStand](../../microduck_rl/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py) | Walk, survive a fall and learn to recover, all in one policy. | Current baseline to evaluate before changing the task. |
| [StandUp](../../microduck_rl/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py) | Recover and settle upright from its configured reset states. | Recovery design reference; a separate StandUp checkpoint is not automatically part of a VelStand network. |
| [SitStand](../../microduck_rl/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py) | Respond to a sit/stand command, transition gently, hold either posture and control the head. | Source of posture rewards and transition lessons for the proposed combined task. |

These four families have registered flat/rough and backlash variants at this commit. The custom walking + sitting + recovery task in the [unified task specification](../../RL_Workbooks/10_experiments/backlash_velocity_flat_unified_policy/unified_task_spec.md) still needs implementation. The current VelStand training run does not add commanded sitting simply by running longer.

Velocity uses `robot_walk.xml`; VelStand uses `robot_groundcontact.xml`, with body contact shapes that allow lying down and pushing off the ground. Their backlash variants select the matching `_backlash.xml` model. Some older comments call the latter an “all-collision” model, but the current constants distinguish **ground-contact** geometry from `robot_allcollisions.xml`. Read the selected constant to determine the model actually used.

The backlash models add a passive hinge in series with each of the 14 servos, with ±1° play. The wrapper makes joint observations use the servo angle plus its backlash angle, retaining 14 actuated joints. Equal input/output sizes alone do not establish playback/export parity: a deployment adapter must reproduce the model's feedback and observation conventions.

## What the policy sees

The shared walking-family actor interface is **61 input values and 14 actions**. Its observations comprise 48 values of body/joint feedback and previous actions, plus 13 command values. The critic used during training can receive extra simulator information; it is not the exported actor.

| Command block | Size | Meaning for Velocity/VelStand |
|---|---:|---|
| `twist` | 3 | Requested forward speed, sideways speed and yaw rate. Linear speeds are in m/s; yaw rate is in rad/s. |
| `head_pose` | 4 | Joint-angle offsets from HOME: neck pitch, head pitch, head yaw, head roll, in radians. |
| `body_pose` | 6 | Body offsets `[x, y, z, roll, pitch, yaw]`; translations in metres, angles in radians. The slot exists, but body-pose tracking has weight **0** in current Velocity and inherited VelStand settings. |

The presence of a command slot does not mean the policy has learned to obey it. This is particularly important for the planned sitting command. Some source comments mention tiny body-pose weights or body control; the active Velocity assignment is `weight=0.0`, and VelStand does not override it.

SitStand keeps the same observation size but repurposes the first `twist` value as `sit_flag`: **0 = stand, 1 = sit**. In a walking policy that same value means forward speed. Combining these tasks therefore requires an explicit, consistent command design and corresponding rewards. The proposed unified specification reserves a calibrated `body_pose.z` offset for posture while keeping walking speed semantics; that is planned work, not an existing feature.

The walking actor receives neither camera images nor ToF frames. The default terrain `height_scan` observation is explicitly removed in the Velocity configuration. Head movement should be requested through the trained head-command input so the policy can balance while moving its head.

## Understand a run and its curricula

In the actual training checkout's VS Code Explorer, open:

```text
microduck_rl/
  logs/rsl_rl/velstand/
    <timestamp>_experiment-backlash-flat-velstand-baseline/
      params/
        env.yaml
        agent.yaml
      model_0.pt
      model_250.pt
      ...
      events.out.tfevents...
```

`env.yaml` records the starting environment configuration after overrides. `agent.yaml` records network/PPO settings, environment steps per iteration and the requested iteration budget. The event file contains training charts; `.pt` files are checkpoints. The timestamp is discovered from this run's folder, not guessed from a tutorial example. Record the actual folder and chosen checkpoints in the workbook.

A curriculum changes settings while training. Saved starting YAML may include its schedule, but does not describe every checkpoint's active weights and reset mixture as if they were constant throughout the run. Keep the source commit, initial settings, checkpoint identity and curriculum stage together when comparing outcomes.

VelStand's active recovery progression includes:

| Iteration threshold, with 24 steps per environment | What changes |
|---:|---|
| 0 | Walking starts with fall termination enabled. |
| 500 | The fall termination limit is relaxed so falls can become recovery attempts. |
| 800 | Some resets start in a crouch, providing experience of the final rise to standing. |
| 1,200 | The fallen-state cost, recovery-success bonus and upward-velocity reward become active. |
| 1,500 / 2,000 / 2,500 | Prone reset probability rises to 15% / 30% / 45%, with face-up states progressively included. |

This is a partial map of the recipe; the inherited walking curricula also run. The active failed-recovery timeout is **8 seconds**, with the timeout's fallen condition based on height below `0.08 m` or tilt above `40°`. The gated upward-velocity recovery incentive uses tilt, while progress rewards have their own definitions. Older narrative comments still mention 5 seconds or earlier gate designs; use the assignments passed to the functions for exact current settings.

Increasing the training environment count does not multiply these iteration thresholds. Changing `num_steps_per_env` would change their relationship to iteration count because the schedules use environment steps.

Playback is an observation tool, but its conditions need checking before treating it as a benchmark. The pinned playback path restores the checkpoint's `common_step_counter`, and reset-time curriculum computation can change the spawn mix with checkpoint age. The inherited Velocity play configuration also applies pushes every **0.5–1 seconds**, compared with **3–6 seconds** during training. A reset after a fall or episode timeout can look like recovery if you only watch the robot reappear upright. Use the [checkpoint review workbook](../../RL_Workbooks/10_experiments/backlash_velocity_flat_unified_policy/checkpoint_review/instructions.md) to record those distinctions and plan repeatable comparisons.

## Where cameras and ToF fit

The source contains useful sensor simulation infrastructure, separate from the current gait observation:

| File | Existing capability | Still needed for following people |
|---|---|---|
| [sim/tof.py](../../microduck_rl/src/mjlab_microduck/sim/tof.py) | Simulated 8×8 range zones, per-zone status and distance noise. | Decide how valid ranges constrain approach speed and stopping distance; test missing/invalid measurements. |
| [sim/camera.py](../../microduck_rl/src/mjlab_microduck/sim/camera.py) | Renders a head camera and supplies frames in the expected transport format. | A person detector/tracker, target selection and tests for children and bent-over adults. |
| [sim/body_server.py](../../microduck_rl/src/mjlab_microduck/sim/body_server.py) | A simulated body interface for the external robot daemon, including sensor plumbing. | Verify model/actuator/observation parity for the chosen backlash policy before using it as an equivalent deployment test. |

These files do not constitute a registered person-following environment. A useful first design is for perception to estimate the selected person's bearing and distance, a controller to turn that estimate into movement and head commands, and the RL policy to perform the movement while balancing and recovering. The [human interaction plan](human_interaction_plan.md) lays out that progression and the experiments needed before introducing sensor inputs directly into a policy.
