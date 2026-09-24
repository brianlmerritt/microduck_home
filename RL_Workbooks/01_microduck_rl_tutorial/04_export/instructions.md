# 04 — Export and replay

[Workbook](../README.md) · [Your record](data_parameters.md)

## Learn

A `.pt` checkpoint contains training state. An **ONNX** export packages the
policy for inference outside the training program. Export changes the format,
not how much the robot has learned.

This policy normalises its observations before using them. The repository's
exporter includes that normalisation in the ONNX graph. Hand-converting just the
network weights can omit it, giving the policy different inputs at playback.

The current walking policy uses 61 observation values and produces 14 joint
actions. The CPU inference script needs `--new-cmd-obs` to select that input
format. These interfaces must agree even when both files are called “walking”.

## Do

Use the same checkpoint that loaded successfully in lesson 03. In the execution
session, copy its path from Explorer and replace the quoted `.pt` path below if
needed. Save the completed command in this lesson's record.

Run the exporter from `microduck_rl`:

```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file "logs/rsl_rl/velocity/2026-09-11_13-23-20_first-smoke/model_4.pt" \
  --num-envs 1 \
  --onnx-file "workbook-smoke.onnx"
```

The exporter builds a training environment, so use the working GPU environment.
After it finishes, find `workbook-smoke.onnx` at the execution project's root in
Explorer. That explicit output filename supplies the input to the next command:

```bash
uv run scripts/infer_policy.py --walking "workbook-smoke.onnx" --new-cmd-obs
```

This runs CPU MuJoCo in a native viewer. It requires a working display in the
execution session, such as WSLg. The browser viewer from lesson 03 does not prove
that native display support works. If the execution session is remote without
a display, record that obstacle and resolve viewing before calling this complete.

## Find the result

The output file is the ONNX policy selected in Explorer. The live viewer is its
deployment rehearsal. Record which checkpoint produced which ONNX file; a short
name like `walking.onnx` does not identify its training origin by itself.

The inference script takes its keyboard input from **the execution terminal**,
not the viewer window. Give that terminal focus. In the default velocity mode:

| Key | Request |
| --- | --- |
| Up arrow | Forward movement. |
| A / E | Turn in either direction. |
| Space | Zero the movement commands. |
| Q | Quit the replay. |

These controls request motion; they do not directly position the body. Space is
a zero-velocity request to the learned policy, not a guarantee that an untrained
robot will stop or balance. Keep the default mode; head/body modes change the
meaning of some keys.

## Observe

Compare the general behaviour with lesson 03. The two simulators need not produce
identical trajectories, but loading errors, invalid observations or a dramatic
change deserve investigation. Falling remains unsurprising after five iterations.

## Continue when

You can connect the original checkpoint to its exported file, load that ONNX
policy, use the terminal controls and quit. Record motion as observed, without
claiming the smoke policy has learned to walk.

[Next: continue learning](../05_resume/instructions.md)
