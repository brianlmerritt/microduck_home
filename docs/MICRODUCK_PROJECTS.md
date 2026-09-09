# Microduck projects and training research

Researched: **9 September 2026**. This is a curated survey for this workspace,
based on project READMEs and upstream documentation. Linked capabilities are
author-reported unless stated otherwise; no training runs or hardware validation
were performed for this survey. Community repositories are evolving quickly.

## Direction for this workspace

The goal is to prepare for the owner's arriving Microduck: test the browser
simulator, train on a Windows PC with an RTX 3090, build a separate simulated RL
environment for each floor of the house, and work towards navigation and later
recognising and reacting to people. The 16 GB Mac Mini is for scaffolding and
lightweight development; local RL training is deliberately out of scope.

**Recommendation:** establish the official training baseline first, then add
house scenes and a navigation layer. Keep experimental simulator ports as
references until their benefits can be measured against that baseline. The
project choices below are recommendations, not additional installed dependencies.

## Official foundation

| Project | What it provides | Relevance here |
| --- | --- | --- |
| [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) | PPO environments using mjlab/MuJoCo Warp; actuator modelling, domain randomization, recovery and locomotion tasks; ONNX export. | Primary training baseline; already a submodule. |
| [Microduck simulator](https://huggingface.co/spaces/pollen-robotics/microduck-simulator) | Browser MuJoCo WASM simulation with ONNX policy inference, keyboard/gamepad controls, walking and roller modes. | First playback smoke test; already a submodule. It runs trained policies rather than training them. |
| [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) | Robot runtime, control services, camera/media services, update system and robot documentation. | Future integration reference for deploying policies and issuing movement intents. |
| [Official policy repository](https://huggingface.co/pollen-robotics/microduck-policies) | Released ONNX behaviour policies. | Baselines for playback and comparisons against our retrained policies. |
| [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | MuJoCo Warp training framework with composable environment components. | Learn task/scene composition here; use the version selected by our RL checkout. |

The official RL tasks cover velocity tracking, walking with recovery, stand-up,
sit/stand, ground pick, ball kick, forward roll and roller behaviours, with
rough-terrain and backlash variants where available. Their common actor contract
is 61 observations and 14 actions at 50 Hz. The exporter includes observation
normalization. Preserve that contract when reusing these locomotion policies;
new navigation observations can live in a separate higher-level controller.
Source: [official training README](https://github.com/pollen-robotics/microduck_rl#readme).

Current local pins:

| Submodule | Commit | Initial upstream branch |
| --- | --- | --- |
| `microduck-simulator` | `e81974b932c7ca1819843b7bb3dcd42e2993e98e` | `main` |
| `microduck_rl` | `2b581c641406a48346e696212930ea881c222c52` | `develop` |

These are reproducibility anchors, not a claim that all linked projects use the
same interfaces or dependency versions. Git LFS assets still need fetching for
the local simulator; application dependencies are not installed.

## Training environments and alternative backends

| Project | Backend and useful features | Assessment for this project |
| --- | --- | --- |
| [jonathanhawkins/microduck-lab](https://github.com/jonathanhawkins/microduck-lab) | CPU MuJoCo plus Stable Baselines 3 PPO; optional Apple MPS policy updates; browser viewer, reward recipes, curricula and ONNX export. | Useful Apple Silicon reference, but deferred on this Mac. Physics is CPU-based, so this is not an all-Metal replacement. Its README calls it a prototyping harness with only part of upstream's domain randomization. |
| [bsprenger/microduck-rl-torch](https://github.com/bsprenger/microduck-rl-torch) | PyTorch-native physics through `mujoco-torch`; CPU, Apple MPS and CUDA paths. | Interesting Metal/MPS option to watch. README still advertises a forthcoming walking demo and cloud-job support; do not infer proven gait or sim-to-real parity from backend support. |
| [Macmachi/microduck-rl-genesis](https://github.com/Macmachi/microduck-rl-genesis) | Genesis walking-task port aimed at AMD/ROCm, retaining upstream actuator modelling. | Alternative-hardware research; little immediate advantage for our NVIDIA target. Author-reported actuator parity is narrower than full simulator or hardware equivalence. |
| [kabilankb/isaaclab-microduck](https://github.com/kabilankb/isaaclab-microduck) | Isaac Lab 3.0/Newton MJWarp tasks for walking, running, kicking and two-robot activities. | Experimental. README explicitly reports locomotion converging to standing still and missing BAM/delay/encoder-bias features. Not our walking baseline. |
| [dreamerarun/isaaclab_microduck](https://github.com/dreamerarun/isaaclab_microduck) | IsaacLab/PhysX port with 37 environments, BAM/backlash models, PPO and export. | Broader simulator comparison candidate. Published rollout is a two-iteration integration checkpoint, not a converged gait. README requires Linux/NVIDIA and points its clone command to the related `5usu/IsaacLab` branch; resolve exact lineage before adopting. |
| [jvpflum/microduck-lab](https://github.com/jvpflum/microduck-lab) | Reproducible training workspace targeting DGX Spark. | Structural reference for pinned upstream projects and repeatable runs; distinct from Jonathan Hawkins's Mac project. Its hardware recipe is not an RTX 3090 recipe. |
| [Microduck RL Ball Follow](https://github.com/yangyihai/Microduck_RL_Ball_Follow) | MuJoCo Warp target-following task, command-contract layer and interactive ball demo. | Useful intermediate task between velocity commands and goal-directed navigation; verify what target information the policy receives before applying it to camera-based control. |

### Windows PC with RTX 3090

The current [mjlab installation guide](https://mujocolab.github.io/mjlab/main/source/installation.html)
lists Linux plus NVIDIA for training, and Linux/macOS/Windows WSL for evaluation.
It does not establish native-Windows Microduck training support.

**Proposed first test:** Ubuntu under WSL2 on the Windows PC, with the pinned
official Microduck environment. NVIDIA documents CUDA support in WSL2 using the
Windows GPU driver; a separate Linux display driver must not be installed inside
WSL. This makes WSL2 a reasonable trial, but does not prove the entire pinned
Microduck stack will work. If that trial fails, investigate the failure and
consider native Linux on the same PC. Source: [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

Start with a small headless rollout/training smoke test, measure GPU memory and
throughput, then increase parallel environments. Do not budget a house scene
using a flat-ground benchmark alone: contact geometry and camera observations can
change resource requirements. Record actual RTX 3090 results rather than copying
another author's GPU timings. The exact OS route remains a setup decision.

## House scenes, mapping and navigation

| Project | Relevant capabilities | Limits and intended use |
| --- | --- | --- |
| [apirrone/microduck_maploc_rs](https://github.com/apirrone/microduck_maploc_rs) | Rust 2D ToF submap SLAM, loop closure, saved-map relocalization, A* planning and waypoint following that outputs body velocities. | Strongest direct mapping lead. Consumes odometry and pre-projected horizontal scans; sensor drivers/projection live elsewhere. README links runtime and Python sandbox integrations that were not verified accessible in this survey. Public crate alone is not a ready house-navigation installation. |
| [shaibuafeez/microduck-ai-world](https://github.com/shaibuafeez/microduck-ai-world) | Detailed MuJoCo scenes and an asynchronous vision-language layer producing bounded movement/skill intents. | Useful architecture reference for keeping perception latency outside the balance loop. Experimental scene autonomy, not evidence of robust household localization. |
| [selinayfilizp/microduck-courier](https://github.com/selinayfilizp/microduck-courier) | Apartment scene, pick/carry/place tasks, ONNX artifact and reproducible evaluation scripts. | Useful scene/reward example. Its trained v1 places the book and recipient nearly straight ahead at short fixed distances with small noise; this is not general room-to-room navigation. |
| [bihaokun/microduck-step-up-policy](https://github.com/bihaokun/microduck-step-up-policy) | Simulation-validated policy pair for a 25 mm threshold, followed by recovery. | Relevant to door thresholds. Hardware unvalidated, does not perceive the step itself, and is not evidence of household stair climbing. |
| [adityakamath/microduck_description](https://github.com/adityakamath/microduck_description) | ROS 2 URDF/xacro and meshes. | Potential model/visualization bridge if ROS becomes useful; does not supply a navigation controller. |
| [osrbot/microduck-ros2-isaac](https://github.com/osrbot/microduck-ros2-isaac) | Community ROS 2 Jazzy/Isaac Sim integration. | Reference for a future ROS-based simulation route; additional integration work compared with starting from our existing MuJoCo stack. |

### Suggested structure for the house work

This is a design proposal, not a feature supplied by the repositories above:

1. Create a separately selectable scene and RL task configuration for each floor,
   using shared robot/task code. Keep floor-specific assets, maps, spawn/goal
   sets and evaluation results separate. Separate environments need not require
   independent copies of the training code or a different gait per floor.
2. Start with measured, simplified collision geometry: rooms, corridors, doors,
   furniture, floor materials and thresholds. Add visual detail when perception
   experiments need it.
3. Establish a mapping/planning baseline above the walking controller. Then test
   a learned navigation policy that produces velocity/skill commands against that
   baseline. This isolates failures in balance, localization and route selection.
4. Use simulator ground-truth pose only as an initial debugging baseline. Later
   evaluate using observations available on the robot, with latency, noise,
   occlusion and localization uncertainty.
5. Reserve unseen start/goal pairs and changed furniture/door arrangements for
   evaluation. Measure goal success, collisions, falls, route time and recovery.
6. Treat identifying the current floor and moving between floors as explicit
   future design decisions. No surveyed project establishes autonomous traversal
   of this house's stairs.

Follow the workspace convention: any adopted external or custom project belongs
in its own repository added here as a submodule. This survey adds no projects.

## People, perception and reactions

| Project | Why it is interesting | What it does not establish |
| --- | --- | --- |
| [Official vision demo](https://huggingface.co/spaces/pollen-robotics/microduck-vision-demo/blob/main/README.md) | Streams robot camera frames to an off-robot OpenCV consumer; also documents local hosting. Source lives in the official runtime's `spaces/vision-demo/`. | A streaming/processing example, not a ready household identity-recognition system. |
| [AlexBodner/microduck-tracking](https://github.com/AlexBodner/microduck-tracking) | Multi-object tracking and target lock in the Microduck simulator. | Tracking a target across frames is different from identifying a known person; its ball-following example is a pattern to adapt. |
| [Official emotions dataset](https://huggingface.co/datasets/pollen-robotics/microduck-emotions) | Emotional body-language animations. | Possible inspiration/assets for greeting reactions; compatibility and playback must be checked separately from locomotion. |
| [joeynyc/microduck-mcp](https://github.com/joeynyc/microduck-mcp) | Agent-facing control tools with mock/simulation and hardware transport paths. | Optional orchestration reference; hardware interfaces and behaviour need validation. An agent tool interface does not solve person recognition or navigation. |

Proposed progression: detect a person, maintain a stable track, trigger a simple
greeting, then optionally distinguish enrolled household members. Decide later
whether “recognise people” means presence alone or individual identity. Keep
unknown people an explicit outcome and test false matches, occlusion and lighting
changes. If identity recognition is wanted, define opt-in enrolment, local data
storage and deletion before collecting reference images. No recognition model
has been selected or tested here.

## Discovery and adoption

[Awesome Microduck](https://github.com/joeynyc/awesome-microduck) is a useful
community discovery index. It was used to locate projects; the assessments above
refer to original project documentation rather than treating directory summaries
as proof of working features.

Suggested next candidates to inspect closely are the official robot runtime,
`microduck_maploc_rs`, and the apartment/target-following examples. Before adding
one, record its exact revision, applicable code/model/asset licenses, upstream
compatibility and a reproducible smoke test. A README or video is evidence of a
project to investigate, not our own validation result.

Implementation milestones and unresolved choices are in [TODO.md](../TODO.md).
