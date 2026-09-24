# 03 — Playback record

[Lesson instructions](instructions.md) · [Workbook](../README.md)

Fill in values from the execution session. Leave unknown values blank. Save
commands with real paths and explicit settings so they can be reused after a
terminal closes. These notes do not configure or run the training program.

| Item | Value |
| --- | --- |
| Task ID | |
| Source run folder | |
| Selected checkpoint (execution Linux path) | |
| Viewer address in execution session | |
| Observation duration (seconds) | |
| Model loaded successfully | |
| Movement / falls / resets observed | |
| How playback was stopped | |

## Complete playback command

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file "logs/rsl_rl/velocity/2026-09-17_07-00-22_experiment-flat-baseline/model_4999.pt" \
  --num-envs 1 \
  --viewer viser
```

## Interpretation of the observed behaviour

Followed arrow but fell over a lot

## Next action


