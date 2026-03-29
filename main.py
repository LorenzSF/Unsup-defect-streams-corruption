from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmark_AD.pipeline import load_config
from benchmark_AD.models import available_models, model_dependencies
from benchmark_AD.pipeline import run_pipeline


DEFAULT_HISTORY_FILE = Path("data") / ".dataset_path_history.json"
DEFAULT_CONFIG_FILE = Path("src") / "benchmark_AD" / "config" / "default.yaml"
MAX_HISTORY_ENTRIES = 10
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class UserExitRequested(Exception):
    """Raised when the user requests to exit from an interactive menu."""


def _clean_path_text(raw: str) -> str:
    return raw.strip().strip('"').strip("'")


def _canonical_path(path_text: str) -> str:
    path = Path(path_text).expanduser()
    try:
        path = path.resolve(strict=False)
    except OSError:
        pass
    return str(path)


def _load_history(history_file: Path) -> List[str]:
    if not history_file.exists():
        return []

    try:
        data = json.loads(history_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    return [item for item in data if isinstance(item, str) and item.strip()]


def _save_history(history_file: Path, paths: List[str]) -> None:
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history_file.write_text(json.dumps(paths, indent=2), encoding="utf-8")


def _update_history(history_file: Path, selected_path: str) -> None:
    selected = _canonical_path(selected_path)
    selected_key = selected.lower()

    updated: List[str] = [selected]
    for item in _load_history(history_file):
        item_norm = _canonical_path(item)
        if item_norm.lower() == selected_key:
            continue
        updated.append(item_norm)
        if len(updated) >= MAX_HISTORY_ENTRIES:
            break

    _save_history(history_file, updated)


def _looks_like_dataset_dir(path: Path) -> bool:
    if not path.is_dir():
        return False

    # Fast checks first: common anomaly dataset folder conventions.
    if (path / "good").is_dir():
        return True
    for bad_name in ("bad", "defects", "defective", "anomaly", "anomalous"):
        if (path / bad_name).is_dir():
            return True

    # Fallback heuristic: image files in root or one level below.
    for ext in _IMAGE_EXTS:
        if any(path.glob(f"*{ext}")) or any(path.glob(f"*/*{ext}")):
            return True
    return False


def _discover_dataset_candidates() -> List[str]:
    candidates: List[str] = []
    seen = set()
    roots = [Path("data") / "raw", Path("data") / "processed"]

    def _add(path: Path) -> None:
        norm = _canonical_path(str(path))
        key = norm.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(norm)

    for root in roots:
        if not root.exists():
            continue
        if _looks_like_dataset_dir(root):
            _add(root)
        for child in sorted(root.iterdir()):
            if child.is_dir() and _looks_like_dataset_dir(child):
                _add(child)
        for zip_file in sorted(root.glob("*.zip")):
            if zip_file.is_file():
                _add(zip_file)
    return candidates


def _collect_dataset_options(
    current_path: str | None,
    history_file: Path,
) -> List[Tuple[str, str]]:
    options: List[Tuple[str, str]] = []
    seen = set()

    def _add(path_text: str, source: str) -> None:
        norm = _canonical_path(path_text)
        key = norm.lower()
        if key in seen:
            return
        seen.add(key)
        options.append((norm, source))

    if current_path:
        _add(current_path, "config")

    for p in _load_history(history_file):
        _add(p, "recent")

    for p in _discover_dataset_candidates():
        _add(p, "detected")
    return options


def _is_exit_choice(choice: str) -> bool:
    return choice.strip().lower() in {"q", "quit", "exit", "x"}


def _prompt_dataset_path(current_path: str | None, history_file: Path) -> str:
    options = _collect_dataset_options(current_path=current_path, history_file=history_file)

    print("\nDataset path selection")
    for idx, (path_text, source) in enumerate(options, start=1):
        status = "exists" if Path(path_text).exists() else "missing"
        print(f"  {idx}. {path_text} [{source}, {status}]")
    print("  N. Enter a new path")
    print("  Q. Exit")

    while True:
        try:
            choice = input("Choose dataset path (number or N): ").strip()
        except EOFError as exc:
            raise ValueError("Interactive selection aborted (no input available).") from exc

        if _is_exit_choice(choice):
            raise UserExitRequested()

        if choice.lower() in {"n", "new"} or (not options and choice == ""):
            entered = _clean_path_text(input("Enter dataset ZIP/folder path: "))
            if _is_exit_choice(entered):
                raise UserExitRequested()
            if not entered:
                print("Path cannot be empty.")
                continue
            selected = _canonical_path(entered)
            if not Path(selected).exists():
                print(f"Path not found: {selected}")
                continue
            _update_history(history_file, selected)
            return selected

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                selected = options[idx - 1][0]
                if not Path(selected).exists():
                    print(f"Selected path does not exist: {selected}")
                    continue
                _update_history(history_file, selected)
                return selected

        print("Invalid choice. Type a number from the list or N.")


def _infer_source_type(path_text: str) -> str:
    path = Path(path_text)
    return "zip" if path.is_file() and path.suffix.lower() == ".zip" else "folder"


def _current_model_name(cfg: Dict[str, object]) -> str | None:
    bench_models = cfg.get("benchmark", {}).get("models", [])
    if isinstance(bench_models, list) and bench_models:
        first = bench_models[0]
        if isinstance(first, dict):
            name = first.get("name")
            if isinstance(name, str):
                return name

    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        name = model_cfg.get("name")
        if isinstance(name, str):
            return name
    return None


def _model_required_modules(model_name: str) -> List[str]:
    return list(model_dependencies(model_name))


def _install_hint(module_name: str) -> str:
    hints = {
        "lightning": "lightning",
        "lightning.pytorch": "lightning",
        "anomalib": "anomalib",
        "sklearn": "scikit-learn",
        "cv2": "opencv-python",
        "FrEIA": "FrEIA",
        "kornia": "kornia",
        "transformers": "transformers",
    }
    return hints.get(module_name, module_name.split(".", 1)[0])


def _module_issue(module_name: str) -> Optional[str]:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or module_name
        return f"missing dependency '{missing}' (pip install {_install_hint(missing)})"
    except Exception as exc:
        return f"import error: {exc}"
    return None


def _model_preflight_checks(model_name: str) -> List[Tuple[str, Optional[str]]]:
    return [(module_name, _module_issue(module_name)) for module_name in _model_required_modules(model_name)]


def _print_model_preflight(model_name: str, checks: List[Tuple[str, Optional[str]]]) -> None:
    in_venv = bool(getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
    print("\nModel preflight")
    print(f"  model: {model_name}")
    print(f"  python: {sys.version.split()[0]}")
    print(f"  executable: {sys.executable}")
    print(f"  in_venv: {in_venv}")
    for module_name, issue in checks:
        status = "OK" if issue is None else f"ERROR: {issue}"
        print(f"  - {module_name}: {status}")


def _model_runtime_issue(model_name: str) -> Optional[str]:
    for module_name, issue in _model_preflight_checks(model_name):
        if issue is not None:
            return f"{module_name}: {issue}"
    return None


def _available_model_issues() -> Dict[str, Optional[str]]:
    return {name: _model_runtime_issue(name) for name in available_models()}


def _prompt_model_name(current_model: str | None) -> str:
    models = available_models()
    if len(models) == 0:
        raise ValueError("No models are registered in model registry.")

    issues = _available_model_issues()
    selectable = [m for m in models if issues.get(m) is None]
    if len(selectable) == 0:
        detail = "; ".join(f"{m}: {issues[m]}" for m in models)
        raise RuntimeError(f"No selectable models in current environment. {detail}")

    default_name = current_model if current_model in models else models[0]
    if issues.get(default_name) is not None:
        default_name = selectable[0]
    default_index = models.index(default_name) + 1

    print("\nModel selection")
    for idx, model_name in enumerate(models, start=1):
        marker = " [default]" if model_name == default_name else ""
        issue = issues.get(model_name)
        status = "" if issue is None else f" [unavailable: {issue}]"
        print(f"  {idx}. {model_name}{marker}{status}")
    print("  Q. Exit")

    while True:
        try:
            choice = input(
                f"Choose model (number or name, Enter={default_index}): "
            ).strip()
        except EOFError as exc:
            raise ValueError("Interactive model selection aborted (no input available).") from exc

        if _is_exit_choice(choice):
            raise UserExitRequested()

        if choice == "":
            if issues.get(default_name) is None:
                return default_name
            print(f"Default model is unavailable: {issues[default_name]}")
            continue
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(models):
                selected = models[idx - 1]
                issue = issues.get(selected)
                if issue is None:
                    return selected
                print(f"Model '{selected}' is unavailable: {issue}")
                continue
        if choice in models:
            issue = issues.get(choice)
            if issue is None:
                return choice
            print(f"Model '{choice}' is unavailable: {issue}")
            continue
        print("Invalid choice. Type a listed number or model name.")


def _resolve_model_cfg(cfg: Dict[str, object], model_name: str) -> Dict[str, object]:
    bench_models = cfg.get("benchmark", {}).get("models", [])
    if isinstance(bench_models, list):
        for entry in bench_models:
            if isinstance(entry, dict) and entry.get("name") == model_name:
                return dict(entry)

    base = dict(cfg.get("model", {})) if isinstance(cfg.get("model"), dict) else {}
    base["name"] = model_name
    return base


def _apply_model_selection(cfg: Dict[str, object], model_name: str) -> None:
    selected = _resolve_model_cfg(cfg, model_name)
    selected["name"] = model_name
    cfg["model"] = dict(selected)
    bench = dict(cfg.get("benchmark", {})) if isinstance(cfg.get("benchmark"), dict) else {}
    bench["models"] = [dict(selected)]
    cfg["benchmark"] = bench


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_FILE),
        help="Path to YAML config.",
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional override for cfg['dataset']['path'] (useful for local ZIP paths).",
    )
    p.add_argument(
        "--extract-dir",
        type=str,
        default=None,
        help="Optional override for cfg['dataset']['extract_dir'].",
    )
    p.add_argument(
        "--choose-dataset",
        action="store_true",
        help="Interactively choose dataset path from config/history/detected paths.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional override for model name (for example: anomalib_patchcore).",
    )
    p.add_argument(
        "--choose-model",
        action="store_true",
        help="Interactively choose one model from registry.",
    )
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable automatic interactive menus when running in a terminal.",
    )
    p.add_argument(
        "--history-file",
        type=str,
        default=str(DEFAULT_HISTORY_FILE),
        help="Path to JSON file that stores recently used dataset paths.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    cfg.setdefault("dataset", {})

    if args.dataset_path is not None:
        cfg["dataset"]["path"] = args.dataset_path
        cfg["dataset"]["source_type"] = _infer_source_type(args.dataset_path)
    if args.extract_dir is not None:
        cfg["dataset"]["extract_dir"] = args.extract_dir

    is_interactive_tty = sys.stdin.isatty()
    auto_prompt = is_interactive_tty and not args.no_interactive

    history_file = Path(args.history_file)
    dataset_path = cfg["dataset"].get("path")
    should_prompt_dataset = bool(args.choose_dataset) or (
        auto_prompt and args.dataset_path is None
    )

    if dataset_path is None:
        should_prompt_dataset = True
    else:
        dataset_exists = Path(str(dataset_path)).expanduser().exists()
        if not dataset_exists and is_interactive_tty:
            print(f"Configured dataset path does not exist: {dataset_path}")
            should_prompt_dataset = True
        if not dataset_exists and not is_interactive_tty and not should_prompt_dataset:
            raise FileNotFoundError(
                f"Dataset path not found: {dataset_path}. "
                "Use --dataset-path or --choose-dataset to select a valid path."
            )

    try:
        if should_prompt_dataset:
            selected_dataset = _prompt_dataset_path(
                str(cfg["dataset"].get("path")) if cfg["dataset"].get("path") is not None else None,
                history_file=history_file,
            )
            cfg["dataset"]["path"] = selected_dataset
            cfg["dataset"]["source_type"] = _infer_source_type(selected_dataset)

        model_name = args.model
        if model_name is not None and model_name not in available_models():
            supported = ", ".join(available_models())
            raise ValueError(f"Unknown model '{model_name}'. Supported: {supported}.")

        should_prompt_model = bool(args.choose_model) or (auto_prompt and model_name is None)
        if should_prompt_model:
            model_name = _prompt_model_name(_current_model_name(cfg))
    except UserExitRequested:
        print("Execution cancelled by user.")
        return

    if model_name is not None:
        checks = _model_preflight_checks(model_name)
        if auto_prompt:
            _print_model_preflight(model_name, checks)
        failed = [(module_name, issue) for module_name, issue in checks if issue is not None]
        if failed:
            details = "; ".join(f"{module_name}: {issue}" for module_name, issue in failed)
            raise RuntimeError(
                f"Model '{model_name}' preflight failed: {details}."
            )
        _apply_model_selection(cfg, model_name)

    if auto_prompt:
        selected_model = _current_model_name(cfg)
        print("\nRun selection")
        print(f"  dataset.path: {cfg['dataset'].get('path')}")
        print(f"  dataset.source_type: {cfg['dataset'].get('source_type', 'folder')}")
        print(f"  model: {selected_model}")
        print(f"  runtime.device: {cfg.get('runtime', {}).get('device', 'auto')}")

    out_dir = run_pipeline(cfg)
    print(f"Run complete. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
