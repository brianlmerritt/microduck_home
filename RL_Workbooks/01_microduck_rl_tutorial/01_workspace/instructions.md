# 01 — Identify the workspace

[Workbook](../README.md) · [Your record](data_parameters.md)

## Learn

The workbook and the training environment have different jobs. Here you keep
explanations, chosen settings and results. The execution session owns the actual
Python installation, GPU, robot models, run folders and checkpoints.

`uv run train ...` asks `uv` to launch the project's training program using its
Python environment. It does not run the workbook, and the word `run` here is
also used for the experiment that `train` creates. Working from the correct
project folder is what connects the command to the intended environment.

A checkpoint path identifies a file on the execution machine. Seeing a similarly
named folder in this learning workspace does not prove that the execution
session can use it. The first task is to establish where the real project lives.

The word **environment** has two meanings here. Your **Python environment** is
the installed software used to run the project. An **RL environment** is the
simulated learning problem: robot, terrain, observations, possible actions,
commands, rewards and rules for ending an attempt. `list-envs` lists those
learning problems, not Python installations.

## Do

1. **Learning session:** open this lesson and its `data_parameters.md` side by
   side. Keep instructions in preview and the record in an editable tab.
2. **Execution session:** locate `microduck_rl` in VS Code Explorer. Confirm it
   contains `pyproject.toml`, `uv.lock`, `src`, `scripts` and, after training,
   `logs`. Open that folder as the workspace if it makes navigation easier.
3. Right-click that folder and choose **Open in Integrated Terminal**. This
   opens a terminal at the selected folder. Use a WSL Bash terminal, not Windows
   PowerShell. [VS Code terminal basics](https://code.visualstudio.com/docs/terminal/basics)
4. Copy the folder's Linux path into your record and the
   [project record](../data_parameters.md). Run this small check there:

```bash
uv run list-envs
```

Find `Mjlab-Velocity-Flat-MicroDuck` in the result. This is the task the workbook
will train: walking on flat ground while following motion commands.

```md
+----+----------------------------------------------------+
| #  | Task ID                                            |
+----+----------------------------------------------------+
| 1  | Mjlab-BallKick-Flat-Backlash-MicroDuck             |
| 2  | Mjlab-BallKick-Flat-MicroDuck                      |
| 3  | Mjlab-Cartpole-Balance                             |
| 4  | Mjlab-Cartpole-Swingup                             |
| 5  | Mjlab-GroundPick-Flat-Backlash-MicroDuck           |
| 6  | Mjlab-GroundPick-Flat-MicroDuck                    |
| 7  | Mjlab-GroundPick-Rough-Backlash-MicroDuck          |
| 8  | Mjlab-GroundPick-Rough-MicroDuck                   |
| 9  | Mjlab-Lift-Cube-Yam                                |
| 10 | Mjlab-Lift-Cube-Yam-Depth                          |
| 11 | Mjlab-Lift-Cube-Yam-Rgb                            |
| 12 | Mjlab-Multi-Cube-Seg-Yam                           |
| 13 | Mjlab-RollerCrouch-Flat-Backlash-MicroDuck         |
| 14 | Mjlab-RollerCrouch-Flat-MicroDuck                  |
| 15 | Mjlab-RollerSlope-Flat-Backlash-MicroDuck          |
| 16 | Mjlab-RollerSlope-Flat-MicroDuck                   |
| 17 | Mjlab-RollerStandUp-Flat-MicroDuck                 |
| 18 | Mjlab-Roulade-Flat-MicroDuck                       |
| 19 | Mjlab-SitStand-Flat-Backlash-MicroDuck             |
| 20 | Mjlab-SitStand-Flat-MicroDuck                      |
| 21 | Mjlab-SitStand-Rough-Backlash-MicroDuck            |
| 22 | Mjlab-SitStand-Rough-MicroDuck                     |
| 23 | Mjlab-Spin-Flat-MicroDuck                          |
| 24 | Mjlab-StandUp-Flat-Backlash-MicroDuck              |
| 25 | Mjlab-StandUp-Flat-MicroDuck                       |
| 26 | Mjlab-StandUp-Rough-Backlash-MicroDuck             |
| 27 | Mjlab-StandUp-Rough-MicroDuck                      |
| 28 | Mjlab-Tracking-Flat-Unitree-G1                     |
| 29 | Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation |
| 30 | Mjlab-VelStand-Flat-Backlash-MicroDuck             |
| 31 | Mjlab-VelStand-Flat-MicroDuck                      |
| 32 | Mjlab-VelStand-Rough-Backlash-MicroDuck            |
| 33 | Mjlab-VelStand-Rough-MicroDuck                     |
| 34 | Mjlab-Velocity-Flat-Backlash-MicroDuck             |
| 35 | Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers     |
| 36 | Mjlab-Velocity-Flat-MicroDuck                      |
| 37 | Mjlab-Velocity-Flat-MicroDuck-Rollers              |
| 38 | Mjlab-Velocity-Flat-Unitree-G1                     |
| 39 | Mjlab-Velocity-Flat-Unitree-Go1                    |
| 40 | Mjlab-Velocity-Rough-Backlash-MicroDuck            |
| 41 | Mjlab-Velocity-Rough-MicroDuck                     |
| 42 | Mjlab-Velocity-Rough-Unitree-G1                    |
| 43 | Mjlab-Velocity-Rough-Unitree-Go1                   |
| 44 | Mjlab-Velocity-Swizzle-Backlash-MicroDuck          |
| 45 | Mjlab-Velocity-Swizzle-MicroDuck                   |
+----+----------------------------------------------------+
```

If this is the environment that already completed your smoke test, no new setup
is required. For a genuinely fresh checkout, use the repository's setup
instructions before continuing; this lesson assumes dependencies are installed.

## Find the result

The task list appears in the execution terminal. Each **Task ID** names a
registered recipe that the trainer knows how to construct. The list tells you
what you can ask it to train; it does not list trained policies or prove that a
robot has learned any of these skills. A checkpoint is the separate result of
training one of those recipes.

The `#` column is just a row number. Use the full Task ID in a command, not `36`.
The order and number of entries can change with installed packages or revisions.

### Read a task name

Take the task selected for this workbook:

```text
Mjlab - Velocity - Flat - MicroDuck
framework   goal   terrain   robot
```

| Part | Meaning |
| --- | --- |
| `Mjlab` | The framework that assembles the simulation and learning task. |
| `Velocity` | Train the robot to follow requested movement speeds and turning rates while balancing. |
| `Flat` | Use the flat-ground version of this task. |
| `MicroDuck` | Use the Microduck robot model and its associated control configuration. |
| `Backlash`, when present | Use a robot model that includes mechanical play in the servo gearing. |
| `Rollers`, when present | Use the roller-equipped robot variant, with passive wheels under its feet. |

These are naming conventions rather than a grammar that accepts any combination.
For example, the listed `Velocity-Swizzle` task has its own name structure. Choose
an exact ID that exists in the output; inserting a word does not create a task.

### Velocity: the movement being requested

**Velocity** includes speed and direction. The walking task receives requests
such as “move forward at 0.1 metres per second” or “turn at 0.3 radians per
second”. Its policy must turn those requests into coordinated joint movements.
Those example numbers are desired motion, not a guarantee of the speed achieved.

The movement command has three components: forward/backward speed, sideways speed
and rotation about the vertical axis, called **yaw**. An all-zero movement
command requests no translation or turning; remaining balanced still requires
the robot to control its joints. This task also includes head and body pose
commands, which later lessons can explore.

Velocity tracking is the movement layer that a higher-level application could
use. “Follow this person” would need another system to decide whom to follow and
what velocity to request. The `Velocity` task does not itself recognise a person,
choose a destination or navigate a house.

### Flat and Rough: the ground being practised on

**Flat** gives this walking task a level plane. It is a useful starting point
because you can inspect balance and command following without introducing
uneven-ground challenges at the same time.

**Rough** uses generated terrain. In the pinned Microduck walking recipe, the
mixture includes flat patches, small steps up to 1.5 cm, uneven blocks up to
1 cm high, and gentle slopes. This is terrain scaled for a small robot; the word
“rough” does not imply household stair climbing or arbitrary obstacle avoidance.

The flat and rough recipes pursue the same broad movement goal under different
ground conditions. Selecting `Rough` changes the training environment; it does
not automatically improve an existing flat-ground checkpoint.

### Backlash: mechanical looseness in the robot

Imagine reversing a geared joint: some movement can take up a small gap between
gear teeth before the output responds. That play is **backlash**. A controller
trained with perfectly tight gearing might react differently when the real
mechanism has some looseness.

The pinned backlash variants add a passive hinge with **±1 degree of play** at
each servo joint. “Passive” means that the policy does not get an extra motor to
command. The simulated joint readings account for the output-side movement, and
the Microduck policy still uses 14 commanded joints.

This lets you study whether a policy learns to cope with that mechanical effect.
It is not a harder terrain or a different desired skill, and the name alone does
not prove better real-world performance. The ordinary variants already model
other motor effects; adding `Backlash` specifically introduces gear play.

These four entries therefore represent distinct experiments:

| Task ID | What differs |
| --- | --- |
| `Mjlab-Velocity-Flat-MicroDuck` | Walking on a level plane using the base robot model. |
| `Mjlab-Velocity-Rough-MicroDuck` | Walking on the configured terrain mixture. |
| `Mjlab-Velocity-Flat-Backlash-MicroDuck` | Level-ground walking with gear play modelled. |
| `Mjlab-Velocity-Rough-Backlash-MicroDuck` | Terrain variation and gear play together. |

### Other Microduck goals in your output

The word following `Mjlab` often identifies a different behaviour being trained.
These are objectives, not claims that an untrained policy can already perform them.

| Name | Intended behaviour |
| --- | --- |
| `StandUp` | Recover from a fallen or low pose and reach a standing posture. |
| `VelStand` | Combine velocity-controlled walking with getting back up after a fall. |
| `SitStand` | Respond to a command to sit or stand, including both transitions. |
| `GroundPick` | Crouch, touch the ground with the mouth tip and return to standing; it does not by itself detect and grasp an arbitrary object. |
| `BallKick` | Kick a ball; this recipe does not give the actor visual ball tracking. |
| `Roulade` | Perform a forward roll and land back on the feet. |
| `Velocity-...-Rollers` | Follow movement commands using the roller-equipped model. |
| `Velocity-Swizzle` | Practise a skating motion with symmetrical leg movements and the wheels kept on the ground. |
| `RollerCrouch` | Crouch while gliding on rollers. |
| `RollerSlope` | Stay balanced while rolling down a slope. |
| `RollerStandUp` | Get up from the ground with rollers fitted. |
| `Spin` | Rotate rapidly in place on rollers. |

One naming exception illustrates why the actual recipe matters:
`Mjlab-RollerSlope-Flat-MicroDuck` contains `Flat` in its registered name but
constructs a scene with a descending ramp. The name is a useful identifier;
the configuration defines the experiment.

### Why other robots appear

`list-envs` reports tasks registered by mjlab as well as this project's Microduck
tasks. Installing the Microduck package adds its entries to that shared list.

| Other names in your output | What they refer to |
| --- | --- |
| `Cartpole-Balance` / `Cartpole-Swingup` | Small teaching problems: balance a pole on a moving cart, or swing it up before balancing. |
| `Unitree-G1` | Tasks for a humanoid robot model. |
| `Unitree-Go1` | Tasks for a four-legged robot model. |
| `Tracking` | Following a reference motion; this is motion tracking, not tracking a person with a camera. |
| `No-State-Estimation` | A tracking variant that omits selected estimated-state inputs, including base linear velocity, from the actor's observations. |
| `Lift-Cube-Yam` | Cube-lifting tasks for the Yam robot arm. `Rgb` and `Depth` select colour-image and depth-image observation variants. |
| `Multi-Cube-Seg-Yam` | A multiple-cube task using depth and segmentation information to specify the target; segmentation distinguishes objects in an image. |

These entries explain why the list is longer than the Microduck project alone.
They do not need to be run for this workbook, and their policies are not
interchangeable with Microduck checkpoints.

### Choose the task for this workbook

Use **`Mjlab-Velocity-Flat-MicroDuck`**. It gives us a clear first question:
can this policy learn balanced movement that follows a velocity request on flat
ground? Keep the same task for the initial train, replay and resume exercises.
Later, you can compare one changed factor at a time, such as terrain or backlash.

In your record sheet, identify the goal, terrain and robot variant for that ID.
Pick one other listed Microduck task and explain what would change. This is a
reading exercise; you do not need to launch more training.

For the source behind these descriptions, see the
[task registrations](../../../microduck_rl/src/mjlab_microduck/tasks/__init__.py),
[walking and terrain configuration](../../../microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)
and [backlash implementation](../../../microduck_rl/src/mjlab_microduck/tasks/backlash.py).
Use the corresponding files in your execution session if its revision differs.

Finally, in Explorer, check whether `logs/rsl_rl/velocity` contains your earlier
smoke run. Merely locate it for now; lesson 02 explains its files.

The revision is useful when comparing results. In the execution session, Source
Control's history can identify the checked-out commit; if you need a copyable
value, `git rev-parse HEAD` prints it. Put it in the project record once.

## Observe

You should be able to point to the project used by the execution terminal and
explain why commands should run there. You should also be able to distinguish a
listed training recipe from a saved trained policy, and explain what `Velocity`,
`Flat`, `Rough` and `Backlash` change. If the task is missing or imports fail,
record that output and resolve the environment before starting training.

## Continue when

The execution project is identified, its task list includes the walking task,
and its location and selected task are saved in your record. You can explain
the selected task's goal, terrain and robot variant, and close the terminal
without losing that information.

[Next: train and find the result](../02_smoke_test/instructions.md)
