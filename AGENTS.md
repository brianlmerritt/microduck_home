# Repository Guidelines

## Project Structure & Module Organization

This workspace coordinates two pinned Git submodules:

- `microduck-simulator/app/`: Vite/React browser simulator. UI lives in `src/ui/`, rendering in `src/scene/`, physics and policy logic in `src/game/`, static assets in `public/`, and tests in `test/`.
- `microduck_rl/`: Python reinforcement-learning project. Source lives in `src/mjlab_microduck/`, regression tests in `tests/`, and training/export utilities in `scripts/`. Robot models and meshes are under `src/mjlab_microduck/robot/`.
- Root `README.md`, `TODO.md`, and `docs/` describe setup, roadmap, and research. Read `microduck_rl/AGENTS.md` before changing RL code.

## Build, Test, and Development Commands

Initialize from the repository root with Git LFS installed:

```sh
git submodule update --init --recursive
git -C microduck-simulator lfs install --local
git -C microduck-simulator lfs pull
```

From `microduck-simulator/app/`:

- `npm ci`: install locked dependencies.
- `npm run dev`: serve locally at `http://localhost:5173`.
- `npm test`: run Node's built-in test runner.
- `npm run build`: create the production bundle in `dist/`.

From `microduck_rl/` (Python 3.12 and `uv` required):

- `uv sync`: install project dependencies.
- `uv run --with pytest pytest tests/`: run regression tests.
- `uv run list-envs`: list registered training tasks.
- `uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5`: smoke-test training on an NVIDIA CUDA GPU before longer runs.

## Coding Style & Naming Conventions

Match surrounding code: JavaScript/JSX uses two-space indentation, semicolons, camelCase functions, and PascalCase React components. Python uses four spaces, snake_case functions/modules, and uppercase constants. Ruff is configured in `microduck_rl/pyproject.toml`; use `uvx ruff check` and `uvx ruff format --check` on changed Python files. Keep task configurations named `microduck_*_env_cfg.py`.

## Testing Guidelines

Name simulator tests `test/*.test.js` and Python tests `tests/test_*.py`. Add regression coverage for changed behavior, especially policy interfaces, reward signs, and joint selection. Most RL tests run on CPU; training smoke tests require CUDA. No coverage percentage is configured. For simulator changes, run tests and build, then check affected controls and locomotion modes in-browser.

## Commit & Pull Request Guidelines

Root history contains only `Initial Commit`; submodule history mixes descriptive subjects with `feat:`, `fix:`, `docs:`, and `chore:` prefixes. Use concise, imperative subjects. Create a branch inside the relevant submodule, commit changes there, then update its pinned commit in the parent repository. PRs should explain behavior changes, link relevant issues, report validation and hardware limitations, and include screenshots or recordings for visual changes.
