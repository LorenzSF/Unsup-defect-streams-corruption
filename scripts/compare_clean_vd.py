"""Compare JobA clean baseline vs val_defect re-run.

Outputs Tables A-D from PLAN job A_analize val_defect.md as TSV files
under data/outputs/jobA_val_defect_V1/_analysis/, plus prints sanity
checks to stdout.
"""
import json
import re
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1] / "data" / "outputs"
CLEAN_DIR = ROOT / "JobA_v0"
VD_DIR = ROOT / "jobA_val_defect_V1"
OUT_DIR = VD_DIR / "_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["anomalib_patchcore", "anomalib_padim", "subspacead"]


def load_runs(directory: Path, prefix: str) -> dict:
    """Return {(category, model): entry_dict} for the freshest run per category.

    Filenames are jobA_<cat>_<UTC> or jobA_val_defect_<cat>_<UTC>.
    """
    out = {}
    for d in sorted(directory.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if "csflow" in name or "stfpm" in name or "draem" in name or "rd4ad" in name:
            continue
        if not name.startswith(prefix):
            continue
        m = re.match(rf"{re.escape(prefix)}(.+)_\d{{8}}_\d{{6}}$", name)
        if not m:
            continue
        category = m.group(1)
        fp = d / "benchmark_summary.json"
        if not fp.exists():
            continue
        data = json.loads(fp.read_text())
        for entry in data.get("models", []):
            if entry.get("model") not in MODELS:
                continue
            key = (category, entry["model"])
            # Keep freshest by run_id timestamp suffix
            prev = out.get(key)
            if prev is None or name > prev["_dir"]:
                row = dict(entry)
                row["_dir"] = name
                out[key] = row
    return out


def fmt(v, p=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{p}f}"
    return str(v)


def delta(a, b):
    if a is None or b is None:
        return None
    return b - a


def main():
    clean = load_runs(CLEAN_DIR, "jobA_")
    vd = load_runs(VD_DIR, "jobA_val_defect_")

    cats_vd = sorted({c for c, _ in vd.keys()})
    print(f"# val_defect categories: {len(cats_vd)}")
    print(f"# clean categories     : {len(sorted({c for c, _ in clean.keys()}))}")
    print(f"# val_defect cats: {cats_vd}")

    # ---------------- Table A: per (cat, model) ----------------
    headerA = ["category", "model",
               "AUROC_clean", "AUROC_vd", "dAUROC",
               "AUPR_clean", "AUPR_vd", "dAUPR",
               "F1_clean", "F1_vd", "dF1",
               "Precision_clean", "Precision_vd",
               "Recall_clean", "Recall_vd",
               "threshold_clean", "threshold_vd",
               "val_recall_vd"]
    rowsA = []
    missing_pairs = []
    for c in cats_vd:
        for m in MODELS:
            v = vd.get((c, m))
            cl = clean.get((c, m))
            if v is None:
                missing_pairs.append((c, m, "vd"))
                continue
            if cl is None:
                missing_pairs.append((c, m, "clean"))
            row = {
                "category": c,
                "model": m,
                "AUROC_clean": (cl or {}).get("auroc"),
                "AUROC_vd": v.get("auroc"),
                "AUPR_clean": (cl or {}).get("aupr"),
                "AUPR_vd": v.get("aupr"),
                "F1_clean": (cl or {}).get("f1"),
                "F1_vd": v.get("f1"),
                "Precision_clean": (cl or {}).get("precision"),
                "Precision_vd": v.get("precision"),
                "Recall_clean": (cl or {}).get("recall"),
                "Recall_vd": v.get("recall"),
                "threshold_clean": (cl or {}).get("threshold_used"),
                "threshold_vd": v.get("threshold_used"),
                "val_recall_vd": v.get("val_recall"),
            }
            row["dAUROC"] = delta(row["AUROC_clean"], row["AUROC_vd"])
            row["dAUPR"] = delta(row["AUPR_clean"], row["AUPR_vd"])
            row["dF1"] = delta(row["F1_clean"], row["F1_vd"])
            rowsA.append(row)

    with (OUT_DIR / "tableA_per_cat_model.tsv").open("w") as f:
        f.write("\t".join(headerA) + "\n")
        for r in rowsA:
            f.write("\t".join(fmt(r[k]) for k in headerA) + "\n")

    # ---------------- Table B: per-model summary ----------------
    headerB = ["model", "n",
               "AUROC_clean_mean", "AUROC_vd_mean", "dAUROC_mean",
               "AUPR_clean_mean", "AUPR_vd_mean", "dAUPR_mean",
               "F1_clean_mean", "F1_vd_mean", "dF1_mean",
               "Recall_clean_mean", "Recall_vd_mean", "dRecall_mean",
               "Precision_clean_mean", "Precision_vd_mean",
               "F1_clean_median", "F1_vd_median",
               "Recall_clean_median", "Recall_vd_median"]
    rowsB = []
    for m in MODELS:
        sub = [r for r in rowsA if r["model"] == m
               and r["AUROC_clean"] is not None and r["AUROC_vd"] is not None]
        if not sub:
            continue
        def col(k):
            return [r[k] for r in sub if r[k] is not None]
        rowsB.append({
            "model": m,
            "n": len(sub),
            "AUROC_clean_mean": mean(col("AUROC_clean")),
            "AUROC_vd_mean": mean(col("AUROC_vd")),
            "dAUROC_mean": mean(col("dAUROC")),
            "AUPR_clean_mean": mean(col("AUPR_clean")),
            "AUPR_vd_mean": mean(col("AUPR_vd")),
            "dAUPR_mean": mean(col("dAUPR")),
            "F1_clean_mean": mean(col("F1_clean")),
            "F1_vd_mean": mean(col("F1_vd")),
            "dF1_mean": mean(col("dF1")),
            "Recall_clean_mean": mean(col("Recall_clean")),
            "Recall_vd_mean": mean(col("Recall_vd")),
            "dRecall_mean": mean(col("Recall_vd")) - mean(col("Recall_clean")),
            "Precision_clean_mean": mean(col("Precision_clean")),
            "Precision_vd_mean": mean(col("Precision_vd")),
            "F1_clean_median": median(col("F1_clean")),
            "F1_vd_median": median(col("F1_vd")),
            "Recall_clean_median": median(col("Recall_clean")),
            "Recall_vd_median": median(col("Recall_vd")),
        })

    with (OUT_DIR / "tableB_per_model.tsv").open("w") as f:
        f.write("\t".join(headerB) + "\n")
        for r in rowsB:
            f.write("\t".join(fmt(r[k]) for k in headerB) + "\n")

    # ---------------- Table C: threshold-shift ----------------
    headerC = ["model", "n",
               "ratio_mean", "ratio_median", "ratio_std",
               "ratio_min", "ratio_max"]
    rowsC = []
    for m in MODELS:
        sub = [r for r in rowsA
               if r["model"] == m
               and r["threshold_clean"] not in (None, 0)
               and r["threshold_vd"] is not None]
        if not sub:
            continue
        ratios = [r["threshold_vd"] / r["threshold_clean"] for r in sub]
        rowsC.append({
            "model": m,
            "n": len(ratios),
            "ratio_mean": mean(ratios),
            "ratio_median": median(ratios),
            "ratio_std": stdev(ratios) if len(ratios) > 1 else 0.0,
            "ratio_min": min(ratios),
            "ratio_max": max(ratios),
        })
    with (OUT_DIR / "tableC_threshold_shift.tsv").open("w") as f:
        f.write("\t".join(headerC) + "\n")
        for r in rowsC:
            f.write("\t".join(fmt(r[k]) for k in headerC) + "\n")

    # ---------------- Table D: val sanity ----------------
    headerD = ["category", "model",
               "val_samples", "val_precision", "val_recall", "val_f1",
               "val_auroc", "val_aupr", "threshold_used"]
    rowsD = []
    for c in cats_vd:
        for m in MODELS:
            v = vd.get((c, m))
            if v is None:
                continue
            rowsD.append({
                "category": c,
                "model": m,
                "val_samples": v.get("val_samples"),
                "val_precision": v.get("val_precision"),
                "val_recall": v.get("val_recall"),
                "val_f1": v.get("val_f1"),
                "val_auroc": v.get("val_auroc"),
                "val_aupr": v.get("val_aupr"),
                "threshold_used": v.get("threshold_used"),
            })
    with (OUT_DIR / "tableD_val_sanity.tsv").open("w") as f:
        f.write("\t".join(headerD) + "\n")
        for r in rowsD:
            f.write("\t".join(fmt(r[k]) for k in headerD) + "\n")

    # ---------------- Sanity checks (PLAN §8.4) ----------------
    print("\n=== SANITY CHECKS ===")
    # 1) every val_defect cell has val_recall > 0
    zero_val = [(r["category"], r["model"], r["val_recall_vd"])
                for r in rowsA if r["val_recall_vd"] in (None, 0)]
    print(f"[1] cells with val_recall == 0: {len(zero_val)}")
    for x in zero_val:
        print("    ", x)

    # 2) |dAUROC| median across all cells < 0.01
    da = [abs(r["dAUROC"]) for r in rowsA if r["dAUROC"] is not None]
    print(f"[2] |dAUROC| median = {median(da):.4f}  (target < 0.01)" if da else "[2] no dAUROC")
    print(f"    |dAUROC| mean   = {mean(da):.4f}")
    print(f"    |dAUROC| max    = {max(da):.4f}")

    # 3) per category, at least 2/3 models show dF1 > 0
    print("[3] per-category models with dF1 > 0:")
    bad_cats = []
    for c in cats_vd:
        df1s = [r["dF1"] for r in rowsA if r["category"] == c and r["dF1"] is not None]
        wins = sum(1 for d in df1s if d > 0)
        flag = "" if wins >= 2 else "  <-- FAIL"
        if wins < 2:
            bad_cats.append(c)
        print(f"    {c:20s}  {wins}/{len(df1s)} models with dF1>0{flag}")
    print(f"    categories failing 2/3 dF1>0 rule: {bad_cats}")

    # 4) AUROC drift outliers > 0.02
    print("[4] cells with |dAUROC| > 0.02:")
    outliers = [(r["category"], r["model"], r["dAUROC"]) for r in rowsA
                if r["dAUROC"] is not None and abs(r["dAUROC"]) > 0.02]
    for x in outliers:
        print(f"    {x[0]:20s}  {x[1]:22s}  dAUROC = {x[2]:+.4f}")
    if not outliers:
        print("    none")

    # 5) Threshold collapse (precision crash < 0.5)
    print("[5] cells with Precision_vd < 0.5:")
    crash = [(r["category"], r["model"], r["Precision_vd"]) for r in rowsA
             if r["Precision_vd"] is not None and r["Precision_vd"] < 0.5]
    for x in crash:
        print(f"    {x[0]:20s}  {x[1]:22s}  Precision_vd = {x[2]:.3f}")
    if not crash:
        print("    none")

    # 6) Threshold silently fell back to val_quantile (identical threshold)
    print("[6] cells with threshold_vd == threshold_clean (possible silent fallback):")
    same = [r for r in rowsA
            if r["threshold_clean"] is not None and r["threshold_vd"] is not None
            and abs(r["threshold_clean"] - r["threshold_vd"]) < 1e-6]
    for r in same:
        print(f"    {r['category']:20s}  {r['model']:22s}  thr={r['threshold_vd']:.4f}")
    if not same:
        print("    none")

    # ---------------- Headline console summary ----------------
    print("\n=== TABLE B HEADLINE (per-model means over val_defect cats) ===")
    print(f"{'model':22s}  n    AUROC clean -> vd   AUPR clean -> vd   "
          f"F1 clean -> vd      Recall clean -> vd  Prec clean -> vd")
    for r in rowsB:
        print(
            f"{r['model']:22s}  {r['n']:<3d}"
            f"  {r['AUROC_clean_mean']:.3f} -> {r['AUROC_vd_mean']:.3f}  "
            f"({r['dAUROC_mean']:+.3f})"
            f"  {r['AUPR_clean_mean']:.3f} -> {r['AUPR_vd_mean']:.3f}"
            f"  {r['F1_clean_mean']:.3f} -> {r['F1_vd_mean']:.3f}"
            f" ({r['dF1_mean']:+.3f})"
            f"  {r['Recall_clean_mean']:.3f} -> {r['Recall_vd_mean']:.3f}"
            f" ({r['dRecall_mean']:+.3f})"
            f"  {r['Precision_clean_mean']:.3f} -> {r['Precision_vd_mean']:.3f}"
        )

    print("\n=== TABLE C THRESHOLD SHIFT ===")
    print(f"{'model':22s}  n    ratio_mean  ratio_median  ratio_std  [min, max]")
    for r in rowsC:
        print(
            f"{r['model']:22s}  {r['n']:<3d}"
            f"  {r['ratio_mean']:.3f}      {r['ratio_median']:.3f}        "
            f"{r['ratio_std']:.3f}      [{r['ratio_min']:.3f}, {r['ratio_max']:.3f}]"
        )

    if missing_pairs:
        print("\n=== MISSING (cat, model, side) pairs ===")
        for x in missing_pairs:
            print("   ", x)

    print(f"\nTSVs written under: {OUT_DIR}")


if __name__ == "__main__":
    main()
