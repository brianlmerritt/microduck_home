# Microduck training workspace

Prepare a reproducible simulation and reinforcement-learning training environment
for a Microduck robot expected to arrive in a few months.

This repository is the project root. Other projects belong here as Git submodules.

## Upstream projects

| Submodule | Purpose | Upstream |
| --- | --- | --- |
| `microduck-simulator` | Browser simulator running MuJoCo WASM and trained ONNX policies | [Hugging Face Space](https://huggingface.co/spaces/pollen-robotics/microduck-simulator) |
| `microduck_rl` | Official PPO training environments, policy export and evaluation | [GitHub](https://github.com/pollen-robotics/microduck_rl) |

Each submodule is pinned to a commit. Initial checkouts use the upstream default
branches: simulator `main`, RL `develop`.

## Checkout

Install Git LFS before fetching simulator assets, then run from this root:

```sh
git submodule update --init --recursive
git -C microduck-simulator lfs install --local
git -C microduck-simulator lfs pull
```

## Setup status and next steps

- Both upstream repositories are registered and checked out as submodules.
- Simulator setup and initial checks are recorded as complete in the roadmap.
- Development and training now use the Windows / WSL2 PC with an RTX 3090 Ti.
  The first 5,000-iteration flat walking run completed and was replayed; frequent
  falls leave gait acceptance open.
- On 17 September 2026 the user reported starting a 4,096-environment,
  50,000-iteration backlash VelStand run. Its completion and behaviour remain to
  be recorded through the checkpoint-review workbook.
- Official training uses MuJoCo Warp and requires an NVIDIA CUDA GPU plus `uv`.
  The 16 GB Mac Mini remains available for lightweight work. The upstream
  project also supports Hugging Face Jobs.

See [TODO.md](TODO.md) for the roadmap and
[Microduck projects](docs/MICRODUCK_PROJECTS.md) for research and source links.

Start the guided exercises in the
[Microduck RL workbook](RL_Workbooks/01_microduck_rl_tutorial/README.md).

For the current run, open the
[checkpoint-review lesson](RL_Workbooks/10_experiments/backlash_velocity_flat_unified_policy/checkpoint_review/instructions.md).
The [English RL reference](docs/rl_reference/README.md) maps documents to actual
environments and preserves translated design history. The
[human-interaction plan](docs/rl_reference/human_interaction_plan.md) explains
the next implementation steps for movement, following and sensing.

See the [simulator setup](microduck-simulator/README.md),
[RL setup](microduck_rl/README.md), and
[hosted training instructions](microduck_rl/scripts/hf/README.md).
