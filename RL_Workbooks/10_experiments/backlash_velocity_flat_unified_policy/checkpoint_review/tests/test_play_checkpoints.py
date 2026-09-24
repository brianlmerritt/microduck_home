"""Launcher checks without importing RL packages or allocating a GPU."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import play_checkpoints as launcher


class CheckpointSelectionTests(unittest.TestCase):
    def test_newest_matching_iterations_not_file_order_or_every_nth_file(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for number in (21000, 0, 250, 6000, 3000, 9000, 12000, 15000, 18000, 22000):
                (run / f"model_{number}.pt").write_bytes(b"checkpoint")
            (run / "model_24000.pt").touch()  # Not a finished checkpoint.
            (run / "model_27000.pt").mkdir()
            (run / "model_latest.pt").write_bytes(b"checkpoint")
            selected = launcher.select_checkpoints(run, 3000, 5)
            self.assertEqual(
                [p.name for p in selected],
                [
                    "model_9000.pt",
                    "model_12000.pt",
                    "model_15000.pt",
                    "model_18000.pt",
                    "model_21000.pt",
                ],
            )
            self.assertEqual(
                [p.name for p in launcher.select_checkpoints(run, 6000, 2)],
                ["model_12000.pt", "model_18000.pt"],
            )

    def test_fewer_matches_and_empty_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.assertEqual(launcher.select_checkpoints(run, 3000, 5), [])
            (run / "model_3000.pt").write_bytes(b"checkpoint")
            self.assertEqual(len(launcher.select_checkpoints(run, 3000, 5)), 1)

    def test_nonpositive_selection_arguments_are_rejected(self):
        for interval, count in ((0, 5), (3000, 0), (-1, 2)):
            with self.assertRaises(ValueError):
                launcher.select_checkpoints(Path("."), interval, count)


class SettingsTests(unittest.TestCase):
    def test_paths_are_relative_to_settings_and_project_not_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "execution project"
            run = project / "logs/run with spaces"
            run.mkdir(parents=True)
            (project / "pyproject.toml").touch()
            settings_file = root / "settings.json"
            settings_file.write_text(
                json.dumps(
                    {
                        "project_dir": "execution project",
                        "run_dir": "logs/run with spaces",
                        "task": "Mjlab-VelStand-Flat-Backlash-MicroDuck",
                        "checkpoint_interval": 3000,
                        "num_checkpoints": 5,
                        "base_port": 8100,
                        "device": "cuda:0",
                    }
                )
            )
            settings = launcher.read_settings(settings_file, count=2, interval=6000)
            self.assertEqual(settings.project_dir, project)
            self.assertEqual(settings.run_dir, run)
            self.assertEqual(settings.num_checkpoints, 2)
            self.assertEqual(settings.checkpoint_interval, 6000)
            command = launcher.worker_command(
                settings, run / "model_6000.pt", 8100, root / "ready.json"
            )
            self.assertEqual(command[0], str(project / ".venv/bin/python"))
            self.assertIn(str(run / "model_6000.pt"), command)

    def test_dry_run_does_not_start_a_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model_3000.pt").write_bytes(b"checkpoint")
            settings = launcher.Settings(root, root, "task", 3000, 5, 8100, "cuda:0")
            with (
                patch.object(launcher, "read_settings", return_value=settings),
                patch.object(launcher.subprocess, "Popen") as popen,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(launcher.main(["--dry-run"]), 0)
            popen.assert_not_called()


@unittest.skipUnless(os.name == "posix", "launcher targets Linux/WSL")
class ProcessLifecycleTests(unittest.TestCase):
    def test_shutdown_only_stops_owned_sessions(self):
        def sleeper():
            return subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                start_new_session=True,
            )

        unrelated, owned = sleeper(), sleeper()
        try:
            launcher.stop_viewers([owned])
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            launcher.stop_viewers([unrelated, owned])

    def test_failed_startup_stops_previously_started_viewer(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / ".venv/bin").mkdir(parents=True)
            (project / ".venv/bin/python").symlink_to(sys.executable)
            settings = launcher.Settings(
                project, project, "task", 3000, 2, 8100, "cuda:0"
            )
            fake_worker = (
                "import json, pathlib, sys, time; "
                "pathlib.Path(sys.argv[1]).write_text(json.dumps("
                "{'port': 8112, 'url': 'http://localhost:8112'})); time.sleep(60)"
            )

            def command(_settings, checkpoint, _port, ready_file):
                if checkpoint.name == "model_6000.pt":
                    return [sys.executable, "-c", "raise SystemExit(3)"]
                return [sys.executable, "-c", fake_worker, str(ready_file)]

            processes = []
            real_popen = subprocess.Popen

            def start(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                processes.append(process)
                return process

            with (
                patch.object(launcher, "worker_command", side_effect=command) as worker,
                patch.object(launcher.subprocess, "Popen", side_effect=start),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "code 3"),
            ):
                launcher.launch(
                    settings, [project / "model_3000.pt", project / "model_6000.pt"]
                )
            self.assertEqual(worker.call_args_list[1].args[2], 8113)
            self.assertEqual(len(processes), 2)
            self.assertTrue(all(p.poll() is not None for p in processes))
            report = next(
                (project / "logs/checkpoint_viewers").glob("*/viewers.md")
            ).read_text()
            self.assertIn("http://localhost:8112", report)

    def test_exited_worker_does_not_wait_for_full_startup_timeout(self):
        process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(2)"])
        process.wait()
        with self.assertRaisesRegex(RuntimeError, "code 2"):
            launcher.await_ready(
                process, Path("missing-ready.json"), Path("viewer.log")
            )


if __name__ == "__main__":
    unittest.main()
