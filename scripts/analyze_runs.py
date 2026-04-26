"""Aggregate benchmark_summary.json files for a Job-A or Job-B style run.

Generalises analyze_jobA.py: pass --prefix to pick which output folders to
ingest (e.g. ``jobA_``, ``jobA_val_defect_``, ``jobB_``,
``jobB_val_defect_``). Surfaces the industrial-relevance metrics added to
the pipeline (recall_at_fpr_1pct, macro_recall, weighted_recall,
per_defect_recall) on top of the legacy AUROC/AUPR/F1/precision/recall.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median


MODELS = ["anomalib_patchcore", "anomalib_padim", "subspacead"]
LEGACY_METRICS = [
    "auroc", "aupr", "f1", "precision", "recall", "accuracy",
    "ms_per_image", "fps", "peak_vram_mb", "fit_seconds", "predict_seconds",
    "threshold_used", "test_samples", "train_samples", "val_samples",
]
INDUSTRIAL_METRICS = [
    "recall_at_fpr_1pct", "recall_at_fpr_5pct", "macro_recall", "weighted_recall",
]


def _category_from_dirname(name: str, prefix: str) -> str:
    pattern = rf"{re.escape(prefix)}(.+)_\d{{8}}_\d{{6}}$"
    m = re.match(pattern, name)
    return m.group(1) if m else name[len(prefix):]


def _load_rows(root: Path, prefix: str) -> list[dict]:
    rows: list[dict] = []
    for d in sorted(root.glob(f"{prefix}*")):
        if "csflow" in d.name:
            continue
        # Avoid matching jobA_val_defect_* when prefix is jobA_
        if prefix == "jobA_" and d.name.startswith("jobA_val_defect_"):
            continue
        category = _category_from_dirname(d.name, prefix)
        fp = d / "benchmark_summary.json"
        if not fp.exists():
            continue
        data = json.loads(fp.read_text())
        for entry in data.get("models", []):
            if entry["model"] not in MODELS:
                continue
            row = {"category": category, "model": entry["model"]}
            for k in LEGACY_METRICS + INDUSTRIAL_METRICS:
                row[k] = entry.get(k)
            row["per_defect_recall"] = entry.get("per_defect_recall") or {}
            row["per_defect_support"] = entry.get("per_defect_support") or {}
            rows.append(row)
    return rows


def _print_table(rows: list[dict]) -> None:
    header = ["category", "model"] + LEGACY_METRICS + INDUSTRIAL_METRICS
    print("\t".join(header))
    for r in rows:
        print("\t".join(str(r.get(k)) for k in header))


def _avg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return mean(vals) if vals else float("nan")


def _med(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return median(vals) if vals else float("nan")


def _print_per_model(rows: list[dict]) -> None:
    print("\n=== PER-MODEL AGGREGATES (mean across categories) ===")
    cols = (
        "model\tn\tauroc\taupr\tf1\tprecision\trecall\trec@fpr1\trec@fpr5"
        "\tmacro_rec\tweighted_rec\tms_per_image\tfps\tpeak_vram_mb\tfit_seconds"
    )
    print(cols)
    for model in MODELS:
        sub = [r for r in rows if r["model"] == model]
        if not sub:
            continue
        print(
            f"{model}\t{len(sub)}"
            f"\t{_avg(sub,'auroc'):.4f}\t{_avg(sub,'aupr'):.4f}"
            f"\t{_avg(sub,'f1'):.4f}\t{_avg(sub,'precision'):.4f}\t{_avg(sub,'recall'):.4f}"
            f"\t{_avg(sub,'recall_at_fpr_1pct'):.4f}\t{_avg(sub,'recall_at_fpr_5pct'):.4f}"
            f"\t{_avg(sub,'macro_recall'):.4f}\t{_avg(sub,'weighted_recall'):.4f}"
            f"\t{_avg(sub,'ms_per_image'):.2f}\t{_avg(sub,'fps'):.2f}"
            f"\t{_avg(sub,'peak_vram_mb'):.1f}\t{_avg(sub,'fit_seconds'):.2f}"
        )

    print("\n=== PER-MODEL MEDIANS ===")
    print("model\tauroc_med\taupr_med\tf1_med\trec@fpr1_med\tmacro_rec_med")
    for model in MODELS:
        sub = [r for r in rows if r["model"] == model]
        if not sub:
            continue
        print(
            f"{model}"
            f"\t{_med(sub,'auroc'):.4f}\t{_med(sub,'aupr'):.4f}\t{_med(sub,'f1'):.4f}"
            f"\t{_med(sub,'recall_at_fpr_1pct'):.4f}\t{_med(sub,'macro_recall'):.4f}"
        )


def _print_per_defect(rows: list[dict]) -> None:
    """Per-(category, model, defect_type) recall table."""
    print("\n=== PER-DEFECT RECALL (test split) ===")
    print("category\tmodel\tdefect_type\tsupport\trecall")
    for r in rows:
        per = r.get("per_defect_recall") or {}
        sup = r.get("per_defect_support") or {}
        if not per:
            continue
        for dtype in sorted(per.keys()):
            print(f"{r['category']}\t{r['model']}\t{dtype}\t{sup.get(dtype,0)}\t{per[dtype]:.4f}")


def _print_winners(rows: list[dict]) -> None:
    cats = sorted({r["category"] for r in rows})
    if not cats:
        return
    print("\n=== PER-CATEGORY AUROC (winner) ===")
    print("category\tpatchcore\tpadim\tsubspacead\twinner")
    wins = {"patchcore": 0, "padim": 0, "subspacead": 0}
    ties = 0
    for c in cats:
        by_model = {r["model"]: r for r in rows if r["category"] == c}
        vals = {
            "patchcore": by_model.get("anomalib_patchcore", {}).get("auroc"),
            "padim": by_model.get("anomalib_padim", {}).get("auroc"),
            "subspacead": by_model.get("subspacead", {}).get("auroc"),
        }
        valid = {k: v for k, v in vals.items() if v is not None}
        if not valid:
            continue
        best = max(valid.values())
        winners = [k for k, v in valid.items() if v == best]
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            ties += 1
        fmt = lambda v: f"{v:.3f}" if v is not None else "-"
        winner = winners[0] if len(winners) == 1 else "tie"
        print(f"{c}\t{fmt(vals['patchcore'])}\t{fmt(vals['padim'])}\t{fmt(vals['subspacead'])}\t{winner}")
    print(f"\nWIN COUNT: {wins} ties: {ties}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "outputs",
        help="data/outputs directory containing run subfolders.",
    )
    parser.add_argument(
        "--prefix", type=str, default="jobA_",
        help="Folder prefix to ingest (jobA_, jobA_val_defect_, jobB_, jobB_val_defect_, ...).",
    )
    parser.add_argument(
        "--per-defect", action="store_true",
        help="Also dump the per-(cat, model, defect_type) recall table.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.root, args.prefix)
    if not rows:
        print(f"No runs matched prefix {args.prefix!r} under {args.root}")
        return

    _print_table(rows)
    _print_per_model(rows)
    _print_winners(rows)
    if args.per_defect:
        _print_per_defect(rows)


if __name__ == "__main__":
    main()
