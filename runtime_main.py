from __future__ import annotations

import argparse
from pathlib import Path

from inspection_realtime.app import InspectionRuntimeApp
from inspection_realtime.settings import DEFAULT_SETTINGS_FILE, load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_SETTINGS_FILE),
        help="Path to the runtime YAML settings file.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional override for run.max_frames.",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable the live dashboard server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_settings(Path(args.config))
    if args.max_frames is not None:
        cfg.setdefault("run", {})
        cfg["run"]["max_frames"] = args.max_frames
    if args.no_web:
        cfg.setdefault("web", {})
        cfg["web"]["enabled"] = False

    app = InspectionRuntimeApp(cfg)
    session_dir = app.run()
    print(f"Runtime session saved to: {session_dir}")


if __name__ == "__main__":
    main()
