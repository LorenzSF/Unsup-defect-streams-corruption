#!/usr/bin/env python3
"""Simple script that runs the example function from `src/python_project_template`.

This script will load values from a `.env` file (if present) using `python-dotenv`.
If no `.env` exists, it prints a short instruction and continues.
"""

from pathlib import Path

from dotenv import load_dotenv

# Repository root (used to locate .env and .env.example)
repo_root = Path(__file__).resolve().parents[1]

# Import the installed package. This requires you to run
# `pip install -e .` (inside a virtual environment) so the package
# is importable.
from python_project_template.core import summarize  # noqa: E402

# Load .env from repository root if present
env_path = repo_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"Loaded environment from {env_path}")
else:
    print("No .env file found. Copy .env.example to .env and fill values if needed:")
    print(f"  cp {repo_root / '.env.example'} {repo_root / '.env'}")


def main() -> None:
    data = [1.0, 2.0, 5.0]
    print("data =", data)
    print("summary =", summarize(data))


if __name__ == "__main__":
    main()
