# Microduck RL reference in English

This parent-project collection connects the upstream design documents to the environments we can actually train. It supports the walking, fall recovery and person-following project, and keeps the original source available for comparison.

Source baseline: `microduck_rl` commit **`2b581c641406a48346e696212930ea881c222c52`**, reviewed on **2026-09-17**. Translations preserve the historical design, identifiers, numerical values and examples. Separate source-check notes explain where the current implementation differs. Historical commands and implementation checklists are reference material, not tasks to execute for the current experiment.

## Suggested reading order

1. [Environment guide](environment_guide.md): where task settings live and how an environment, a policy and a checkpoint relate.
2. [Checkpoint review workbook](../../RL_Workbooks/10_experiments/backlash_velocity_flat_unified_policy/checkpoint_review/instructions.md): inspect the running VelStand experiment and record what each selected checkpoint can do.
3. [Human interaction plan](human_interaction_plan.md): progress from walking/recovery to sitting, simulated targets and person following.
4. [Roller recovery summary in English](translations/roller_standup_policy_summary.md): useful lessons about reward signs, recovery and remaining upright, with wheel-specific context retained.
5. [Head-control design](translations/specs/2026-07-27-swizzle-head-control-design.md): why head movement belongs in the policy's commands when balance matters. This upstream document was already English.

The current VelStand task already combines walking and recovery in one policy. Commanded sitting and person following require further work. Training completion, a registered task and a checked-off design plan are different kinds of evidence; checkpoint tests determine whether the behaviour meets our requirements.

## How the documents correspond to code

There are **16 upstream documents** in this review: eight specs, seven implementation plans and one recovery summary. The six French specs and French summary are translated here; the three already-English documents are copied with provenance and an unchanged source body. The six French implementation plans remain linked upstream as lower-priority historical references and are **not translated in this collection**.

“Registered” below means the task ID and configuration exist in the pinned source. It does not mean the old document describes every current setting or that a trained policy passed our tests. Unchecked plan checkboxes were not used as evidence that implementation is missing.

| Original document | English copy or translation | Relationship to pinned code |
|---|---|---|
| [Roller StandUp summary](../../microduck_rl/docs/roller_standup_policy_summary.md) | [Translated summary](translations/roller_standup_policy_summary.md) | Historical learning report for `Mjlab-RollerStandUp-Flat-MicroDuck`. General recovery lessons are relevant; some statements about ordinary StandUp are stale. |
| [2026-07-17 Roller Crouch spec](../../microduck_rl/docs/superpowers/specs/2026-07-17-roller-crouch-glide-design.md) | [Translated spec](translations/specs/2026-07-17-roller-crouch-glide-design.md) | `Mjlab-RollerCrouch-Flat-MicroDuck` is registered. Historical crouch/glide design; consult current factory for its evolved recipe. |
| [2026-07-17 Roller Crouch plan](../../microduck_rl/docs/superpowers/plans/2026-07-17-roller-crouch-glide.md) | French plan retained upstream | Same registered family. Historical implementation sequence, not the current experiment's checklist. |
| [2026-07-22 Roller Slope spec](../../microduck_rl/docs/superpowers/specs/2026-07-22-roller-slope-design.md) | [Translated spec](translations/specs/2026-07-22-roller-slope-design.md) | `Mjlab-RollerSlope-Flat-MicroDuck` is registered. Passive slope descent; task name includes `Flat` despite custom ramp terrain. |
| [2026-07-22 Roller Slope plan](../../microduck_rl/docs/superpowers/plans/2026-07-22-roller-slope.md) | French plan retained upstream | Same slope task. Historical implementation instructions. |
| [2026-07-23 Swizzle spec](../../microduck_rl/docs/superpowers/specs/2026-07-23-swizzle-env-design.md) | [English source copy](translations/specs/2026-07-23-swizzle-env-design.md) | `Mjlab-Velocity-Swizzle-MicroDuck` is registered. Current code also supports backward motion and a heading curriculum beyond this early straight-line design. |
| [2026-07-24 GroundPick spec](../../microduck_rl/docs/superpowers/specs/2026-07-24-ground-pick-pose-following-design.md) | [Translated spec](translations/specs/2026-07-24-ground-pick-pose-following-design.md) | `Mjlab-GroundPick-Flat-MicroDuck` is registered, but this proposed pose-following rewrite does not match the current mouth-proximity/return-to-standing recipe. |
| [2026-07-24 GroundPick plan](../../microduck_rl/docs/superpowers/plans/2026-07-24-ground-pick-pose-following.md) | French plan retained upstream | Proposed rewrite, not a reliable description of the current GroundPick implementation. |
| [2026-07-24 Shoot spec](../../microduck_rl/docs/superpowers/specs/2026-07-24-shoot-pose-following-design.md) | [Translated spec](translations/specs/2026-07-24-shoot-pose-following-design.md) | Proposed `Mjlab-Shoot-Flat-MicroDuck` is **not registered**. Existing `Mjlab-BallKick-Flat-MicroDuck` uses a different design, with a physical ball in simulation. |
| [2026-07-24 Shoot plan](../../microduck_rl/docs/superpowers/plans/2026-07-24-shoot-pose-following.md) | French plan retained upstream | Unregistered proposed task; do not treat its training commands as currently available. |
| [2026-07-27 Swizzle head spec](../../microduck_rl/docs/superpowers/specs/2026-07-27-swizzle-head-control-design.md) | [English source copy](translations/specs/2026-07-27-swizzle-head-control-design.md) | Head command, reward and staged ranges are implemented in the Swizzle configuration. Relevant to looking at people while balancing. |
| [2026-07-27 Swizzle head plan](../../microduck_rl/docs/superpowers/plans/2026-07-27-swizzle-head-control.md) | [English source copy](translations/plans/2026-07-27-swizzle-head-control.md) | Main configuration changes are present despite unchecked historical boxes. Runtime button/flag claims were not independently validated here. |
| [2026-08-04 Roller StandUp spec](../../microduck_rl/docs/superpowers/specs/2026-08-04-roller-standup-design.md) | [Translated spec](translations/specs/2026-08-04-roller-standup-design.md) | `Mjlab-RollerStandUp-Flat-MicroDuck` is registered. Read alongside the later summary and current recovery configuration. |
| [2026-08-04 Roller StandUp plan](../../microduck_rl/docs/superpowers/plans/2026-08-04-roller-standup.md) | French plan retained upstream | Same task family; historical instructions rather than a record of final reward tuning. |
| [2026-08-04 Spin spec](../../microduck_rl/docs/superpowers/specs/2026-08-04-spin-env-design.md) | [Translated spec](translations/specs/2026-08-04-spin-env-design.md) | `Mjlab-Spin-Flat-MicroDuck` is registered; wheel-based rotation is peripheral to the walking/following goal. |
| [2026-08-04 Spin plan](../../microduck_rl/docs/superpowers/plans/2026-08-04-spin-env.md) | French plan retained upstream | Same registered family; historical implementation sequence. |

The task mappings above come from the [registry](../../microduck_rl/src/mjlab_microduck/tasks/__init__.py) and these configurations: [RollerCrouch](../../microduck_rl/src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py), [RollerSlope](../../microduck_rl/src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py), [Swizzle](../../microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_swizzle_env_cfg.py), [GroundPick](../../microduck_rl/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py), [BallKick](../../microduck_rl/src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py), [RollerStandUp](../../microduck_rl/src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py) and [Spin](../../microduck_rl/src/mjlab_microduck/tasks/microduck_spin_env_cfg.py).

## Keeping the reference useful

When the RL submodule is updated, keep each translation's original source commit and review the separate status note against the new code. Amend an old author's claim only in an explicitly labelled annotation; do not silently turn a historical proposal into a description of a later implementation.

For current experiments, record actual results in the [experiment workbook](../../RL_Workbooks/10_experiments/backlash_velocity_flat_unified_policy/instructions.md). Proposed settings, source-reported outcomes and our measured outcomes should remain distinguishable.
