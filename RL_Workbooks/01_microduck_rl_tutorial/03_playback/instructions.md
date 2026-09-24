# 03 — Watch the checkpoint

[Workbook](../README.md) · [Your record](data_parameters.md)

## Learn

Playback loads the policy saved in a checkpoint and asks it to control a simulated
robot. It performs **inference**: choosing actions without updating the network.
The robot's movement is therefore evidence about what that particular checkpoint
learned. Watching it does not provide additional training.

The task name selects the robot/environment and policy interface. The checkpoint
path selects the learned weights. Both matter: a correct file loaded into an
incompatible task is not a valid evaluation.

## Do

1. **Learning session:** open the run record from lesson 02. Copy its selected
   checkpoint path into this lesson's record.
2. **Execution session:** confirm that exact file in Explorer. Copy its Linux
   path and paste it between the quotes after `--checkpoint-file` in the command
   below. Keep the same task name used for training.
3. Save the completed command in this lesson's record, then run it in the
   execution session from `microduck_rl`.

This example points to the previously verified smoke checkpoint. It can be used
unchanged only if that relative path exists in your execution project:

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file "logs/rsl_rl/velocity/2026-09-11_13-23-20_first-smoke/model_4.pt" \
  --num-envs 1 \
  --viewer viser
```

| Part of the command | Where the value comes from |
| --- | --- |
| `Mjlab-Velocity-Flat-MicroDuck` | The task used to train this checkpoint. |
| The quoted `.pt` path | The file selected in Explorer and saved in lesson 02. |
| `--num-envs 1` | Show one duck so its movement is easy to inspect. |
| `--viewer viser` | Use the browser viewer for this exercise. |

Open the browser address printed by the command. This address belongs to the
execution session. If it is on another machine, use that session's forwarded
port/address; `localhost` in the learning session may be a different machine.
VS Code's execution-session Ports view can expose the viewer's printed port.

## Find the result

The result is the live viewer; this action does not produce a newly trained
checkpoint. Confirm that the terminal reports loading the file you selected.
Record the viewer address and what you actually see.

## Observe

Watch for about 20 seconds. Does the model appear? Does the duck move, stay
upright, fall or repeatedly reset? A five-iteration policy will often fall.
Repeated resets can make a failed policy look briefly upright again, so observe
the sequence rather than a single frame.

This first lesson tests loading and viewing, not keyboard command-following.
The CPU replay lesson introduces explicit movement commands. Stop this browser
viewer with Ctrl+C in its execution terminal when you finish; closing the browser
tab alone need not stop the simulation process.

## Continue when

The selected checkpoint loads and you have recorded what it does. Poor gait is
an expected observation at this stage; failure to load the model is a separate
problem to resolve before continuing.

[Next: export and replay](../04_export/instructions.md)
