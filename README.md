# Microduck training workspace

Prepare a reproducible simulation and reinforcement-learning training environment
for a Microduck robot expected to arrive in a few months. The root directory is
currently named `microcat` and may be renamed later.

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
- Git LFS is missing on the current Mac; simulator binary assets remain LFS
  pointers until Git LFS is installed and the assets are pulled.
- Application and Python dependencies have not been installed, and simulation
  and training have not yet been run.
- Official training uses MuJoCo Warp and requires an NVIDIA CUDA GPU plus `uv`.
  The intended training host is a Windows PC with an RTX 3090; its Linux/WSL2
  setup remains to be tested. The 16 GB Mac Mini is for scaffolding, not RL
  training. The upstream project also supports Hugging Face Jobs.

See [TODO.md](TODO.md) for the roadmap and
[Microduck projects](docs/MICRODUCK_PROJECTS.md) for research and source links.

See the [simulator setup](microduck-simulator/README.md),
[RL setup](microduck_rl/README.md), and
[hosted training instructions](microduck_rl/scripts/hf/README.md).
