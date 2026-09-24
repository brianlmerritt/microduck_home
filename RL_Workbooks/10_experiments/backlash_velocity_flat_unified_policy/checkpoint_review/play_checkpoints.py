#!/usr/bin/env python3
"""Open the newest interval-matching checkpoints in separate Viser viewers.

Run with --dry-run to inspect selection without importing RL packages or using
the GPU. Paths live in checkpoint_viewers.json, relative to that settings file
(project_dir) and the execution project (run_dir).
"""

import argparse
import json
import os
import re
import select
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKPOINT_NAME = re.compile(r"model_(\d+)\.pt\Z")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def select_checkpoints(run_dir, interval, count):
    """Select exact iteration multiples, newest N, presented oldest first."""
    if interval < 1 or count < 1:
        raise ValueError("checkpoint_interval and num_checkpoints must be positive")
    candidates = []
    for path in run_dir.glob("model_*.pt"):
        match = CHECKPOINT_NAME.fullmatch(path.name)
        if not match or not path.is_file():
            continue
        iteration = int(match[1])
        if iteration > 0 and iteration % interval == 0 and path.stat().st_size > 0:
            candidates.append((iteration, path))
    return [path for _, path in sorted(candidates)[-count:]]


@dataclass(frozen=True)
class Settings:
    project_dir: Path
    run_dir: Path
    task: str
    checkpoint_interval: int
    num_checkpoints: int
    base_port: int
    device: str


def read_settings(path, count=None, interval=None):
    path = path.expanduser().resolve()
    raw = json.loads(path.read_text())
    expected = {
        "project_dir",
        "run_dir",
        "task",
        "checkpoint_interval",
        "num_checkpoints",
        "base_port",
        "device",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(
            f"Settings must contain exactly: {', '.join(sorted(expected))}"
        )
    if count is not None:
        raw["num_checkpoints"] = count
    if interval is not None:
        raw["checkpoint_interval"] = interval
    for key in ("num_checkpoints", "checkpoint_interval", "base_port"):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("project_dir", "run_dir", "task", "device"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if raw["base_port"] > 65535:
        raise ValueError("base_port must be at most 65535")
    project = (path.parent / Path(raw["project_dir"]).expanduser()).resolve()
    run = (project / Path(raw["run_dir"]).expanduser()).resolve()
    if not (project / "pyproject.toml").is_file():
        raise ValueError(
            f"No execution project at {project}; edit project_dir in {path}"
        )
    if not run.is_dir():
        raise ValueError(f"Run folder does not exist: {run}; edit run_dir in {path}")
    return Settings(
        project,
        run,
        raw["task"],
        raw["checkpoint_interval"],
        raw["num_checkpoints"],
        raw["base_port"],
        raw["device"],
    )


def worker_command(settings, checkpoint, port, ready_file):
    # Keep the venv symlink path: resolving it can select the system environment.
    python = settings.project_dir / ".venv/bin/python"
    return [
        str(python),
        "-u",
        str(HERE / "checkpoint_viewer.py"),
        "--task",
        settings.task,
        "--checkpoint",
        str(checkpoint),
        "--port",
        str(port),
        "--device",
        settings.device,
        "--ready-file",
        str(ready_file),
    ]


def stop_viewers(processes):
    """Stop only the new sessions owned by this launcher, never other training."""
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def await_ready(process, ready_file, log_file, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Viewer exited with code {process.returncode}. Open its log: {log_file}"
            )
        if ready_file.is_file():
            return json.loads(ready_file.read_text())
        time.sleep(0.2)
    raise RuntimeError(f"Viewer startup exceeded {timeout}s. Open its log: {log_file}")


def write_report(log_dir, settings, viewers):
    lines = [
        "# Checkpoint viewers",
        "",
        f"Task: `{settings.task}`",
        f"Run: `{settings.run_dir}`",
        "",
        "Links work while the launcher is running. Closing a browser tab does not",
        "stop its simulator; return to the launch terminal and press Enter or Ctrl+C.",
        "",
        "| Checkpoint | Viewer | Process | Log |",
        "| --- | --- | --- | --- |",
    ]
    for item in viewers:
        lines.append(
            f"| {item['checkpoint']} | [Open viewer]({item['url']}) | "
            f"{item['pid']} | [{item['log']}]({item['log']}) |"
        )
    (log_dir / "viewers.md").write_text("\n".join(lines) + "\n")


def launch(settings, checkpoints):
    if os.name != "posix":
        raise RuntimeError("Run this launcher in the Linux/WSL execution terminal")
    python = settings.project_dir / ".venv/bin/python"
    if not os.access(python, os.X_OK):
        raise RuntimeError(
            f"Python environment missing: {python}; use your configured RL checkout"
        )
    log_root = settings.project_dir / "logs/checkpoint_viewers"
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d_%H-%M-%S_")
    log_dir = Path(tempfile.mkdtemp(prefix=timestamp, dir=log_root))
    print(f"Viewer logs and link list: {log_dir}", flush=True)
    processes, viewers = [], []
    next_port = settings.base_port
    write_report(log_dir, settings, viewers)
    try:
        for checkpoint in checkpoints:
            if next_port > 65535:
                raise RuntimeError(
                    "No remaining viewer ports; choose a lower base_port"
                )
            ready_file = log_dir / f"{checkpoint.stem}.ready.json"
            log_file = log_dir / f"{checkpoint.stem}.log"
            print(
                f"Starting {checkpoint.name} (initialisation may compile kernels)...",
                flush=True,
            )
            env = os.environ.copy()
            env.pop("_VISER_PORT_OVERRIDE", None)
            with log_file.open("w") as output:
                process = subprocess.Popen(
                    worker_command(settings, checkpoint, next_port, ready_file),
                    cwd=settings.project_dir,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            processes.append(process)
            ready = await_ready(process, ready_file, log_file)
            next_port = ready["port"] + 1  # Viser may skip an already occupied port.
            item = {
                "checkpoint": checkpoint.name,
                "url": ready["url"],
                "pid": process.pid,
                "log": log_file.name,
            }
            viewers.append(item)
            write_report(log_dir, settings, viewers)
            print(f"  {checkpoint.name}: {ready['url']}", flush=True)
        print(
            "\nAll viewers are running together. Ctrl+click their links to open tabs."
        )
        print("Press Enter or Ctrl+C here to stop these viewers.", flush=True)
        while True:
            for process, item in zip(processes, viewers):
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{item['checkpoint']} exited with code {process.returncode}; "
                        f"see {log_dir / item['log']}"
                    )
            readable, _, _ = select.select([sys.stdin], [], [], 0.5)
            if readable:
                sys.stdin.readline()
                break
    finally:
        stop_viewers(processes)
        print("Stopped this launcher's viewers.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings", type=Path, default=HERE / "checkpoint_viewers.json"
    )
    parser.add_argument(
        "--num_checkpoints",
        "--num-checkpoints",
        type=positive_int,
        help="number of newest matching checkpoints (default: 5 in settings)",
    )
    parser.add_argument(
        "--interval",
        type=positive_int,
        help="select iteration multiples of this number (default: 3000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list selection without starting viewers"
    )
    args = parser.parse_args(argv)
    try:
        settings = read_settings(args.settings, args.num_checkpoints, args.interval)
        checkpoints = select_checkpoints(
            settings.run_dir, settings.checkpoint_interval, settings.num_checkpoints
        )
        if not checkpoints:
            raise ValueError(
                f"No nonempty checkpoints at multiples of {settings.checkpoint_interval} "
                f"in {settings.run_dir} (model_0.pt is excluded)"
            )
        print(f"Task: {settings.task}\nRun: {settings.run_dir}")
        print(
            f"Newest {len(checkpoints)} available checkpoints at {settings.checkpoint_interval}-iteration intervals:"
        )
        for checkpoint in checkpoints:
            print(f"  {checkpoint.name}")
        if len(checkpoints) < settings.num_checkpoints:
            print(
                f"Only {len(checkpoints)} matching checkpoints exist; requested {settings.num_checkpoints}."
            )
        if args.dry_run:
            print("Dry run: no viewers started, no GPU used.")
            return 0
        launch(settings, checkpoints)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
