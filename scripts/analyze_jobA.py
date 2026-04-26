"""Aggregate jobA benchmark_summary.json files across categories for PatchCore, PaDiM, SubspaceAD."""
import json
import re
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1] / "data" / "outputs"
MODELS = ["anomalib_patchcore", "anomalib_padim", "subspacead"]
METRICS = ["auroc", "aupr", "f1", "precision", "recall", "accuracy",
           "ms_per_image", "fps", "peak_vram_mb", "fit_seconds", "predict_seconds",
           "threshold_used", "test_samples", "train_samples"]

rows = []
for d in sorted(ROOT.glob("jobA_*")):
    name = d.name
    if "csflow" in name:
        continue
    m = re.match(r"jobA_(.+)_\d{8}_\d{6}$", name)
    category = m.group(1) if m else name
    fp = d / "benchmark_summary.json"
    if not fp.exists():
        continue
    data = json.loads(fp.read_text())
    for entry in data.get("models", []):
        if entry["model"] not in MODELS:
            continue
        row = {"category": category, "model": entry["model"]}
        for k in METRICS:
            row[k] = entry.get(k)
        rows.append(row)

# Print as TSV-ish table
header = ["category", "model"] + METRICS
print("\t".join(header))
for r in rows:
    print("\t".join(str(r[k]) for k in header))

# Summary per model
print("\n=== PER-MODEL AGGREGATES (mean across all categories) ===")
print("model\tn\tauroc\taupr\tf1\tprecision\trecall\taccuracy\tms_per_image\tfps\tpeak_vram_mb\tfit_seconds")
for model in MODELS:
    sub = [r for r in rows if r["model"] == model]
    if not sub:
        continue
    def avg(k):
        vals = [r[k] for r in sub if r[k] is not None]
        return mean(vals) if vals else float("nan")
    def med(k):
        vals = [r[k] for r in sub if r[k] is not None]
        return median(vals) if vals else float("nan")
    print(f"{model}\t{len(sub)}\t{avg('auroc'):.4f}\t{avg('aupr'):.4f}\t{avg('f1'):.4f}"
          f"\t{avg('precision'):.4f}\t{avg('recall'):.4f}\t{avg('accuracy'):.4f}"
          f"\t{avg('ms_per_image'):.2f}\t{avg('fps'):.2f}\t{avg('peak_vram_mb'):.1f}"
          f"\t{avg('fit_seconds'):.2f}")

print("\n=== PER-MODEL MEDIANS ===")
print("model\tauroc_med\taupr_med\tf1_med\tms_per_image_med\tfps_med")
for model in MODELS:
    sub = [r for r in rows if r["model"] == model]
    if not sub:
        continue
    def med(k):
        vals = [r[k] for r in sub if r[k] is not None]
        return median(vals) if vals else float("nan")
    print(f"{model}\t{med('auroc'):.4f}\t{med('aupr'):.4f}\t{med('f1'):.4f}"
          f"\t{med('ms_per_image'):.2f}\t{med('fps'):.2f}")

# Per-category winner by AUROC
print("\n=== PER-CATEGORY AUROC (winner in caps) ===")
cats = sorted({r["category"] for r in rows})
print("category\tpatchcore\tpadim\tsubspacead\twinner")
for c in cats:
    by_model = {r["model"]: r for r in rows if r["category"] == c}
    pc = by_model.get("anomalib_patchcore", {}).get("auroc")
    pd = by_model.get("anomalib_padim", {}).get("auroc")
    sd = by_model.get("subspacead", {}).get("auroc")
    vals = {"patchcore": pc, "padim": pd, "subspacead": sd}
    valid = {k: v for k, v in vals.items() if v is not None}
    winner = max(valid, key=valid.get) if valid else "-"
    pcs = f"{pc:.3f}" if pc is not None else "-"
    pds = f"{pd:.3f}" if pd is not None else "-"
    sds = f"{sd:.3f}" if sd is not None else "-"
    print(f"{c}\t{pcs}\t{pds}\t{sds}\t{winner}")

# Win counts
print("\n=== WIN COUNT (best AUROC per category) ===")
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
print(wins, "ties:", ties)

# Top-N by AUPR (mean across 3 models)
print("\n=== TOP-10 CATEGORIES BY MEAN AUPR (across 3 models) ===")
aupr_ranked = []
for c in cats:
    sub = [r for r in rows if r["category"] == c and r["aupr"] is not None]
    if not sub:
        continue
    aupr_ranked.append((c, mean(r["aupr"] for r in sub)))
aupr_ranked.sort(key=lambda x: -x[1])
for c, v in aupr_ranked[:10]:
    print(f"{c}\t{v:.4f}")

# Hardest / easiest categories (mean AUROC across 3 models)
print("\n=== HARDEST / EASIEST CATEGORIES (mean AUROC across 3 models) ===")
ranked = []
for c in cats:
    sub = [r for r in rows if r["category"] == c and r["auroc"] is not None]
    if not sub:
        continue
    ranked.append((c, mean(r["auroc"] for r in sub)))
ranked.sort(key=lambda x: x[1])
print("--- 5 hardest ---")
for c, v in ranked[:5]:
    print(f"{c}\t{v:.4f}")
print("--- 5 easiest ---")
for c, v in ranked[-5:]:
    print(f"{c}\t{v:.4f}")
