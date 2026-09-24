"""One labelled checkpoint viewer, started by play_checkpoints.py.

Uses the installed mjlab playback path. The adapter only supplies the server
and readiness notification; no training configuration or installed file changes.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ready-file", required=True, type=Path)
    args = parser.parse_args()

    # These imports run only in the execution project's configured environment.
    import mjlab.tasks  # noqa: F401
    import viser
    from mjlab.scripts import play
    from mjlab.viewer import ViserPlayViewer

    class CheckpointViewer(ViserPlayViewer):
        def __init__(self, env, policy, **kwargs):
            self.owned_server = viser.ViserServer(
                host="127.0.0.1",
                port=args.port,
                label=f"MicroDuck — {args.checkpoint.name}",
            )
            # Keep each window attached to its label's checkpoint.
            kwargs["checkpoint_manager"] = None
            super().__init__(env, policy, viser_server=self.owned_server, **kwargs)

        def setup(self):
            super().setup()
            port = self.owned_server.get_port()
            temporary = args.ready_file.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "port": port,
                        "url": f"http://localhost:{port}",
                        "checkpoint": str(args.checkpoint),
                        "pid": os.getpid(),
                    }
                )
            )
            temporary.replace(args.ready_file)

        def close(self):
            try:
                super().close()
            finally:
                self.owned_server.stop()

    # The substitution is confined to this child process, preserving upstream
    # checkpoint loading, normalisation, curricula and simulation behaviour.
    play.ViserPlayViewer = CheckpointViewer
    play.run_play(
        args.task,
        play.PlayConfig(
            checkpoint_file=str(args.checkpoint),
            num_envs=1,
            viewer="viser",
            device=args.device,
        ),
    )


if __name__ == "__main__":
    main()
