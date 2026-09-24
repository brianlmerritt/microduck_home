# Microduck TODO

Prepare a training and simulation workspace ahead of the physical Microduck's
arrival. Research: [Microduck projects](docs/MICRODUCK_PROJECTS.md).

The 16 GB Mac Mini is for scaffolding and lightweight development. Do not plan
local RL training here. The intended training machine is a Windows PC with an
RTX 3090. All other projects, including future custom environments, belong in
separate repositories added to this root as Git submodules.

## 1. Test the web simulator

- [ ] Install Git LFS and fetch the simulator's actual binary assets.
- [ ] Install the web dependencies and launch the pinned simulator locally.
- [ ] Test loading, walking/turning, reset and behaviour switching; test a gamepad
  if available.
- [ ] Check the browser console and network requests for missing assets/errors,
  and run the production build.
- [ ] Record the revision, browser, commands and observed results in `docs/`.

Done when the local simulator loads real models/policies, controls work and the
production build passes, with a reproducible test record.

## 2. Test RL on the Windows PC with RTX 3090

- [ ] Record Windows version, system RAM, free disk space, GPU driver and GPU
  memory; confirm access to the machine.
- [ ] Confirm the OS route. First candidate: Ubuntu under WSL2 with CUDA;
  native-Windows training support has not been established. Consider native
  Linux if WSL2 cannot run the pinned stack reliably.
- [ ] Check out this root and its pinned submodules on the PC; install the
  dependencies required by the pinned RL project.
- [ ] Verify CUDA visibility from both PyTorch and MuJoCo Warp.
- [ ] Run upstream tests and a small headless training smoke test; measure memory
  and throughput before increasing parallel environments.
- [ ] Train the official flat-ground walking baseline, save a checkpoint and
  demonstrate resuming it.
- [ ] Export through the upstream ONNX exporter and replay the policy in CPU
  MuJoCo; inspect motion and command following as well as reward curves.
- [ ] Record versions, revision, seed, configuration, commands, training time,
  peak memory and replay results in `docs/`.

Done when this PC can reproduce train → checkpoint → resume → ONNX export →
simulation replay. Set the gait acceptance criteria before the full run.

## 3. Build a separate RL environment for each house floor

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

## 4. Navigate around the house

- [ ] Evaluate the Microduck mapping/planning projects in the research document
  and establish a waypoint-navigation baseline using the existing walking policy.
- [ ] Add localization and obstacle observations; distinguish debugging with
  ground-truth pose from evaluation using simulated robot sensors.
- [ ] Train and compare a higher-level navigation policy in each floor environment.
- [ ] Test room-to-room goals, narrow passages, blocked paths, localization loss,
  fall recovery and stopping near obstacles/drop boundaries.
- [ ] Define and record acceptance thresholds for success rate, collisions,
  falls, arrival tolerance and route time across held-out cases.
- [ ] Decide how the current floor is selected or recognised and how the robot
  moves between floors. Stair traversal is unresolved, not an assumed capability.
- [ ] Once the robot arrives, validate sensing and the walking baseline, then
  transfer navigation incrementally from one clear room to routes on each floor.

Done when the robot can reach agreed destinations on each mapped floor with
measured reliability and predictable failure handling. Whole-house completion
also requires an explicit solution for movement between floors.

## 5. Stretch: recognise people and react

- [ ] Clarify whether recognition means detecting a person, identifying known
  household members, or both, and choose the desired reactions.
- [ ] Establish camera ingestion and evaluate person detection/tracking.
- [ ] Add a greeting/look-towards reaction with a cooldown so it does not repeat
  continuously; keep perception separate from the balance/control loop.
- [ ] If individual recognition is wanted, agree opt-in enrolment, storage and
  deletion, then test known/unknown handling and false matches.
- [ ] Evaluate lighting changes, occlusion and multiple people, first in recorded
  or simulated scenes and later on the robot.

Done when the agreed people-related behaviour works consistently, treats
uncertain matches as unknown and does not disrupt navigation or balance.
