# Microduck TODO

Prepare a simulation, training and development workspace ahead of the physical
Microduck's arrival. Research: [Microduck projects](docs/MICRODUCK_PROJECTS.md).

Development is now on the Windows PC with an RTX 3090 Ti, using WSL2 and VS Code
with the Codex extension. Verify the environment before treating simulation or
training as working. The 16 GB Mac Mini remains available for scaffolding and
lightweight development; local RL training is not planned there.

This root coordinates the tutorial, research and project progress. All other
projects, including custom environments and human-interaction applications,
belong in separate repositories added to this root as Git submodules.

The immediate application goal is a Microduck that can interact with people,
follow a selected person, stop when shown a flat hand in front of it, and express
a few simple reactions through movement and sound. Start with a self-contained
version running on the robot. Add optional remote support later.

Describe desired behaviour and measurable outcomes before choosing an
implementation. Sensors, distance estimation, recognition models, control
architecture and learning methods remain open decisions. The interaction project
does not require a fruit-fly model, an LLM or new locomotion training.

## 1. Test the web simulator

- [x] Install Git LFS and fetch the simulator's actual binary assets.
- [x] Install the web dependencies and launch the pinned simulator locally.
- [x] Test loading, walking/turning, reset and behaviour switching; test a gamepad
  if available.
- [x] Check the browser console and network requests for missing assets/errors,
  and run the production build.

Optional: keep a short note of the revision, browser, launch command and any
workarounds in `docs/`. This helps diagnose regressions after an upgrade or
recreate the setup on another machine; it is not a completion requirement.

Done when the local simulator loads real models/policies, controls work and the
production build passes. Complete based on the checks above.

## 2. Establish simulation and RL on Windows / WSL2 with RTX 3090 Ti

Progress reviewed on 17 September 2026: setup, CUDA checks and upstream tests
were reported complete through the tutorial; the earlier smoke run was also
confirmed from local artifacts. The
[first baseline record](RL_Workbooks/10_experiments/flat_velocity_rtx3090_4096.md)
now documents 5,000 iterations, a saved checkpoint and browser playback.
The duck followed the command arrow but fell frequently, so gait acceptance
remains open. The recorded command and step count indicate **1,024 environments**,
despite `4096` in the filename; the GPU output identifies an **RTX 3090 Ti**.

Later on 17 September, the user reported starting
`Mjlab-VelStand-Flat-Backlash-MicroDuck` with 4,096 environments and a 50,000-iteration
budget, run name `experiment-backlash-flat-velstand-baseline`. This records the
launch report, not completion or a successful gait/recovery result.

- [x] Record GPU model and total GPU memory: RTX 3090 Ti, 24,564 MiB reported.
- [ ] Record Windows and WSL distribution versions, system RAM, free disk space
  and GPU driver version.
- [x] Check out this root and its pinned submodules on the PC; install the
  dependencies required by the pinned projects.
- [x] Verify CUDA visibility from both PyTorch and MuJoCo Warp.
- [x] Load a trained movement policy in an interactive browser simulation and
  observe command response: baseline `model_4999.pt` replayed with Viser.
- [ ] Systematically verify standing, walking, both turn directions and stopping;
  measure falls and command following against explicit gait acceptance criteria.
- [ ] Confirm which robot interfaces and observations the pinned simulator
  supports, and document differences from physical hardware.
- [x] Record the baseline training and browser-playback launch commands.
- [ ] Verify simulation timing and complete the inspection/shutdown workflow.
- [x] Run upstream regression tests (reported tutorial completion; not rerun
  during this progress review).
- [x] Run a small headless training smoke test and save a checkpoint: 64
  environments, five iterations, seed 42; `model_4.pt` exists and logged scalar
  metrics are finite for all five iterations.
- [x] Check smoke-test throughput: TensorBoard records 637–786 environment
  transitions per second (`Perf/total_fps`) at 64 environments.
- [x] Record an initial memory/throughput observation at 1,024 environments:
  3,306 MiB used in one GPU snapshot and 10,173 steps/s at iteration 4. This is
  not a measured peak or a steady-state benchmark.
- [ ] Compare memory and throughput in short runs before increasing the batch
  further; the new 4,096-environment run's actual resource use and throughput
  remain to be recorded.
- [x] Train the official flat-ground walking baseline and save a checkpoint:
  5,000 iterations at 1,024 environments, recorded elapsed time 02:36:54,
  `2026-09-17_07-00-22_experiment-flat-baseline/model_4999.pt`.
- [ ] Demonstrate resuming a saved checkpoint.
- [ ] Export through the upstream ONNX exporter and replay the policy in CPU
  MuJoCo; inspect motion and command following as well as reward curves.
- [x] Record the first baseline's commands, training time, a memory snapshot and
  initial playback outcome in `RL_Workbooks/10_experiments/`.
- [ ] Complete the experiment record with versions, revision, seed, saved
  configuration, memory measurement method and repeatable evaluation results.
- [x] Prepare a VS Code checkpoint comparison lesson, blank scorecard and
  continuation/fine-tuning workflow for the reported VelStand run.
- [ ] Review selected VelStand checkpoints and record walking, stopping,
  recovery and head/balance behaviour separately.
- [ ] Implement a fixed-scenario checkpoint evaluator; verify identical
  commands, starts, disturbances and evaluation settings across candidates.

Smoke-run artifacts:
`microduck_rl/logs/rsl_rl/velocity/2026-09-11_13-23-20_first-smoke/` contains
`model_0.pt`, `model_4.pt`, saved environment/agent configurations, TensorBoard
metrics and an ONNX file. The ONNX file's presence alone does not confirm replay.

Suggested next steps:

1. Follow the
   [VelStand checkpoint review](RL_Workbooks/10_experiments/backlash_velocity_flat_unified_policy/checkpoint_review/instructions.md).
   Find the reported run under `logs/rsl_rl/velstand` and compare several saved
   policies. The task/model/batch/budget changed together; this does not isolate
   the effect of backlash relative to the original plain walking baseline.
2. Develop and test one policy for walking, commanded sit/stand and fall recovery
   using the custom-task specification in that experiment. This requires a new
   training recipe; the existing task names do not combine all three behaviours.
3. Complete resume and ONNX/CPU replay verification. Use the workbook for the
   [continuation workflow](RL_Workbooks/10_experiments/backlash_velocity_flat_unified_policy/continue_training/instructions.md)
   and the experiment's requirements for matching backlash physics and
   observations in replay. Select extra training from measured behaviour.

Done when the PC can run the interactive simulator and reproduce train →
checkpoint → resume → ONNX export → simulation replay. Human-interaction work
can begin with existing policies before the full training baseline is complete.

## 3. Initial application: self-contained human interaction

Build a small companion behaviour application that runs within Microduck's
onboard resources, without requiring a remote computer or internet connection.
The first functional scope is following and stopping. Add a few expressive
reactions after those behaviours are established.

- [x] Specify the staged simulation experiment for backlash walking, a single
  policy combining walking/sit/stand/recovery, and later people/sensor inputs:
  [experiment plan](RL_Workbooks/10_experiments/backlash_velocity_flat_unified_policy/instructions.md).
- [x] Review upstream design documents against the pinned source and create an
  [English reference collection](docs/rl_reference/README.md), preserving source
  provenance and separating historical claims from current-source annotations.
- [x] Document the [implementation sequence](docs/rl_reference/human_interaction_plan.md)
  from checkpoint evaluation through one movement policy, labelled-target
  following, sensor integration and onboard validation.
- [ ] Implement and evaluate that unified movement policy; distinguish this
  movement experiment from implementing or validating human perception.
- [ ] Create the application as a separate repository/submodule and document its
  purpose, scope and interface to the simulator and robot.
- [ ] Define how a person starts following and becomes the selected target.
  The follow-start gesture or action has not yet been agreed.
- [ ] Implement the agreed stop signal: a flat hand held in front of the robot,
  with the palm presented towards it.
- [ ] Make stop take priority over following and expressive actions; accept a
  recognised stop signal from someone other than the selected target too.
- [ ] Keep the robot stopped after a stop signal until following is deliberately
  restarted; lowering the hand must not automatically resume movement.
- [ ] Follow the selected person at an agreed distance and speed, including
  changes in direction and pauses, without switching to another person silently.
- [ ] Support people from approximately two years old through adulthood,
  including adults bending down or crouching. Do not assume a fixed human size
  or interpret a posture change as a distance change.
- [ ] Define operating limits, including following distance, speed, gesture
  visibility and the conditions under which following should stop.
- [ ] Stop predictably when the target is lost, observations become unreliable,
  or continuing movement is obstructed; agree how tracking can resume.
- [ ] Add a small agreed set of expressive reactions, such as acknowledging a
  command, showing curiosity or signalling uncertainty. These are expressive
  behaviours, not a claim to recognise human emotions.
- [ ] Ensure expressive actions do not compromise balance, override stopping or
  interfere with following.
- [ ] Measure onboard processing, memory, response time and sustained operation;
  keep interaction responsive while movement control continues reliably.

Done when the physical robot can follow the selected person and respond to the
stop palm within agreed limits, entirely onboard, with predictable handling of
uncertainty. Before arrival, record simulation and recorded-input results as
provisional evidence rather than hardware completion.

## 4. Validate interaction across people and situations

- [ ] Define measurable acceptance criteria for stop response, missed and false
  gestures, target retention, following distance, movement reliability and
  resource use before declaring the initial application complete.
- [ ] Test different human heights, small children's hands, adults standing and
  bending down, and transitions between postures.
- [ ] Test approach, retreat, turning, standing still, partial visibility,
  leaving the view and returning.
- [ ] Test multiple people, people crossing paths, a bystander giving the stop
  signal, and preventing unintended target changes.
- [ ] Test ordinary variations in lighting, clothing, backgrounds and gesture
  position, including incidental hand movements that should not start following.
- [ ] Test obstacles, blocked routes, nearby drop boundaries, falls and recovery;
  define whether deliberate restart is required after each interruption.
- [ ] Use simulation and representative recorded inputs where appropriate;
  document what each test can and cannot demonstrate.
- [ ] When hardware arrives, repeat the tests incrementally, establishing basic
  movement and stop reliability before supervised interaction with children.
- [ ] Record cases, configurations, results, limitations and failures in `docs/`.

Done when the agreed scenarios have repeatable results and any unsupported
conditions are documented. PC performance alone does not establish onboard
feasibility, and simulated people alone do not establish real gesture reliability.

## 5. Build a separate RL environment for each house floor

- [ ] Gather floor names/count, dimensions or floor plans, floor materials,
  furniture, doorways, thresholds and stair/drop locations from the owner.
- [ ] Choose how to capture geometry and create the custom environment project
  as a separate repository/submodule.
- [ ] Build one measured floor first, with simplified collision geometry and
  a separately selectable scene/task configuration.
- [ ] Repeat for every floor; give each its own maps, start/goal sets and
  evaluation cases while sharing robot and training code.
- [ ] Validate scale, collisions, clearance, spawn points and walking playback
  before beginning floor-specific RL experiments.
- [ ] Define navigation rewards/terminations and hold-out scenarios, including
  changed furniture, open/closed doors and different starting positions.

Done when every floor loads independently, has valid training/evaluation tasks
and can be reproduced from its recorded assets and configuration.

## 6. Navigate around the house

- [ ] Evaluate the Microduck mapping/planning projects in the research document
  and establish a waypoint-navigation baseline using the existing walking policy.
- [ ] Establish the observations and position estimates needed for navigation;
  distinguish debugging with perfect simulator information from evaluation
  using information available to the robot.
- [ ] Train and compare a higher-level navigation policy in each floor environment.
- [ ] Test room-to-room goals, narrow passages, blocked paths, localization loss,
  fall recovery and stopping near obstacles/drop boundaries.
- [ ] Define and record acceptance thresholds for success rate, collisions,
  falls, arrival tolerance and route time across held-out cases.
- [ ] Decide how the current floor is selected or recognised and how the robot
  moves between floors. Stair traversal is unresolved, not an assumed capability.
- [ ] Integrate navigation with following and stop behaviour so that competing
  goals cannot override a person's stop command.
- [ ] Once the robot arrives, validate sensing and the walking baseline, then
  transfer navigation incrementally from one clear room to routes on each floor.

Done when the robot can reach agreed destinations on each mapped floor with
measured reliability and predictable failure handling. Whole-house completion
also requires an explicit solution for movement between floors.

## 7. Later: optional remote support and richer interaction

- [ ] Agree which additional capabilities justify remote assistance, such as
  richer conversation, more complex requests or improved scene understanding.
- [ ] Define which capabilities remain available independently on the robot and
  which require a nearby computer or external service.
- [ ] Preserve local stopping and reliable movement when remote responses are
  delayed, unavailable or interrupted; prevent stale requests restarting motion.
- [ ] Test transitions between self-contained and remotely supported operation.
- [ ] If recognising individual household members is wanted, define this as a
  separate feature from detecting and following a selected person.
- [ ] For individual recognition, agree enrolment, storage and deletion, then
  test known/unknown handling and false matches.
- [ ] Record remote resource requirements, latency and the resulting behaviour.

Done when remote assistance adds agreed capabilities while the self-contained
follow/stop behaviour remains useful and predictable without it.
