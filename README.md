# Python Project Template — Getting Started

This guide explains, step by step, how to get this repository running on your computer.

## Prerequisites

- **Git:** used to clone the repository. Install from https://git-scm.com/ if you don't have it.
- **Python 3.10+ (recommended):** check with `python --version`.

## 1) Clone the repository

- **SSH (recommended if you plan to push changes):**

    1. If you don't have an SSH key, generate one and add it to your GitHub account:

        ```bash
        ssh-keygen -t ed25519 -C "your_email@example.com"
        # then copy the public key and add it to GitHub: ~/.ssh/id_ed25519.pub
        ```

        After adding the key to GitHub, clone using SSH:

        ```bash
        git clone git@github.com:your-username/your-repo.git
        ```

- **HTTPS (simpler read-only):**

    ```bash
    git clone https://github.com/your-username/your-repo.git
    ```

Why SSH vs HTTPS? SSH lets you push without typing your username/password every time; HTTPS is fine to only read the repo.

Change into the project folder:

```bash
cd python_project_template
```

## 2) Create and activate a virtual environment (venv)

Why a virtual environment? It isolates this project's Python packages from other projects and from system packages, so you can install the exact dependencies needed without affecting your system Python.

On macOS / Linux (zsh / bash):

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```


After activation your shell prompt usually changes to show the venv name. To leave the venv later use `deactivate`.

If `python` points to the wrong version, try `python3` or install a recent Python (3.10+).

## 3) Install dependencies

With pip:

```bash
pip install -e .
```


## Alternative: Installing with uv

```bash
uv sync
```



On macOS / Linux (zsh / bash):

```bash
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

## 4) Setup a `.env` file (if needed)

What is a `.env` file? It's a simple text file with KEY=VALUE lines used to store local (sometimes confidential) configuration variables (API keys, database URLs, passwords, user names, etc.). This repo treats `.env` as local-only (not committed).

If a `.env.example` exists, copy it and fill in values:

```bash
cp .env.example .env
# then open .env in a text editor and fill values
```

If there is no `.env.example`, create `.env` only when you need to store secrets or configuration. Example contents:

```
API_TOKEN=replace-me
DEBUG=True
```

Why not commit `.env`? Committing secrets is dangerous. Keep keys out of version control.

## 5) Run the example script

From the project root (with the venv activated), run this command:

```bash
python scripts/run_example.py
```

This runs a small example to show how the package is used. If the script reads `.env`, make sure you created it first.

## Common issues & quick fixes

- **Command not found / wrong Python version:** run `python --version`. If it is too old, install a newer Python and try `python3`.
- **`venv` activation doesn't change prompt:** still activated, try `python -m pip list` to confirm packages are from the venv.
- **Can't import package code in src:** ensure you installed editable mode `pip install -e .` or that you're running Python from the activated venv so `src/` is on `PYTHONPATH`.

## Where to edit code

- The package source code is in `src/python_project_template/`.
- Tests are in `tests/` and can be run with `pytest`.

## Files to look at

- [notebooks/example_notebook.ipynb](notebooks/example_notebook.ipynb) — interactive example notebook.
- [scripts/run_example.py](scripts/run_example.py) — small runnable example script.
- [requirements.txt](requirements.txt) — list of Python dependencies.

## 6) Folder structure and root files

This explains the top-level layout and what you should do with each item as a student:

- `src/` — package source. The package `python_project_template` lives under `src/python_project_template/`. Here you put all your modules (classes, libraries, etc) that are used by your scripts.
- `tests/` — unit tests for the project. Run `pytest` to execute them. You can write tests for the code that you write such that before runtime, you know the code is correct.
- `notebooks/` — example and exploratory notebooks. You mostly use notebooks for quick experiments or sharing code in e.g.workshops.
- `scripts/` — your executable scripts. Run them from the project root with the venv active (for example `python scripts/run_example.py`). Scripts assume the package is installed (editable) or that you run from the project root. You mostly have a main.py script which is the starting script for your project.
- `data/` — placeholder for example data. The folder is tracked via `data/.gitkeep`, but its contents are ignored by git. Do not commit large datasets or secrets.
- `requirements.txt` —
- `pyproject.toml` — packaging and tooling configuration used to build/install the package and configure tools (formatters, linters). lists runtime and development packages. Contains all modules that you need in your code. This to ensure that if you share your code, people know how to setup the environment correctly and reproduce your results easily.
- `pytest.ini` — pytest settings (test discovery, markers). Run `pytest` to use these defaults.
- `.env.example` / `.env` — example environment variables and local values. Copy `.env.example` to `.env` and fill in values if required. Never commit real secrets. The env.example is used such that people you share the code with know what kind of environment variables they need to run the code.
- `.gitignore` — lists files and folders that git should ignore (venv, build artifacts, data contents). We typically never push large data files, secrets, runtime files (__pycache__), or models to git.
 - `.pre-commit-config.yaml` — configuration for the `pre-commit` framework. Defines hooks (formatters, linters, and other checks) that run automatically before commits to keep code style and quality consistent. Run `pre-commit install` to enable the hooks locally.
 - `.gitlab-ci.yml` — GitLab CI/CD pipeline configuration that automates testing, linting, and building across multiple Python versions when pushing your code. It includes pytest for testing, mypy for type checking, and ruff for linting and formatting. 
 - `README.md` — The entry file where you describe what your code does and how to use it.
