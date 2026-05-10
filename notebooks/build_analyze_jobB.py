"""Generate `analyze jobB.ipynb` — kept as a script so the notebook stays
reproducible and easy to regenerate after schema tweaks. Run once:

    python notebooks/build_analyze_jobB.py
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).with_name("analyze jobB.ipynb")


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l if l.endswith("\n") else l + "\n" for l in lines][:-1]
        + [lines[-1]],
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l if l.endswith("\n") else l + "\n" for l in lines][:-1]
        + [lines[-1]],
    }


cells: list[dict] = []

cells.append(md(
    "# Analyze JobB — trained-models validation on Deceuninck\n",
    "\n",
    "Companion to `analyze jobA.ipynb`, but focused on the **JobB** runs that re-evaluate the trained\n",
    "models against the Deceuninck dataset.\n",
    "\n",
    "The single Deceuninck category means we cannot compute per-category statistics the way JobA does —\n",
    "instead we treat each model run as one observation and dig into the per-image score distribution\n",
    "and the per-defect-type recall to understand model behavior.\n",
    "\n",
    "Sections:\n",
    "1. **Setup** — paths, imports.\n",
    "2. **Loader** — walks the JobB output dirs (baseline + seed sweep) and returns a tidy DataFrame.\n",
    "   Each row is one `(model, seed)` pair, so multi-seed runs surface as multiple rows per model.\n",
    "3. **§1 Summary metrics per model** — comparison table, generalization gap, per-defect recall.\n",
    "4. **§2 Graphical comparison** — accuracy bars, latency-vs-quality Pareto, generalization gap.\n",
    "5. **§3 Coherence / sanity checks** — flags `0.0` / `1.0` metrics, threshold position relative to\n",
    "   the actual score distribution, sample-count vs fit-time consistency.\n",
    "6. **§4 Score distributions** — per-run histogram of test scores split by `good` vs `defect`\n",
    "   with the calibrated threshold drawn on top.\n",
    "7. **§5 Confusion matrices** — derived from `predictions_*.json` so the headline F1 can be audited.\n",
    "8. **§6 JobB vs JobA reference** — quick reuse of `_analysis/compare_jobB_vs_jobA.tsv`.\n",
    "9. **§7 Multi-seed coherence** — coverage matrix, mean ± std aggregates, threshold stability,\n",
    "   saturation re-check across seeds.\n",
    "10. **§8 Engineering suggestions** — what to look at next, framed for the thesis goal."
))

cells.append(md("## 1. Setup"))

cells.append(code(
    "from __future__ import annotations\n",
    "\n",
    "import json\n",
    "import re\n",
    "from pathlib import Path\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "REPO = Path.cwd().resolve()\n",
    "if REPO.name == \"notebooks\":\n",
    "    REPO = REPO.parent\n",
    "\n",
    "OUTPUTS = REPO / \"data\" / \"outputs\"\n",
    "JOBB_DIR = OUTPUTS / \"jobB_val_defect_V1\"\n",
    "JOBB_SEED_DIR = OUTPUTS / \"jobB_val_defect_and_seed\"\n",
    "JOBA_DIR = OUTPUTS / \"jobA_val_defect_V1\"\n",
    "ANALYSIS_DIR = JOBB_DIR / \"_analysis\"\n",
    "\n",
    "JOBB_SOURCES = [d for d in (JOBB_DIR, JOBB_SEED_DIR) if d.is_dir()]\n",
    "assert JOBB_SOURCES, f\"no JobB output dirs found under {OUTPUTS}\"\n",
    "print(\"REPO         :\", REPO)\n",
    "for d in JOBB_SOURCES:\n",
    "    print(\"JOBB source :\", d)\n",
    "print(\"JOBA         :\", JOBA_DIR if JOBA_DIR.is_dir() else \"(not present)\")"
))

cells.append(md(
    "## 2. Loader\n",
    "\n",
    "Each JobB run directory holds a `benchmark_summary.json` whose `models[]` array contains one entry\n",
    "per model that ran. Some directories bundle 3 models (the Deceuninck mass-eval), others contain a\n",
    "single model — we just iterate and flatten.\n",
    "\n",
    "The seed lives in `summary['run']['seed']`; when missing we try to recover it from a `_s<N>_`\n",
    "fragment in the directory name (added by the seed-sweep driver), falling back to `42` (the original\n",
    "default). The `source` column records which top-level dir a row came from so multi-seed reruns and\n",
    "the original baseline are easy to slice apart.\n",
    "\n",
    "Per-image scores are taken from `predictions_<model>.json` and `validation_predictions_<model>.json`."
))

cells.append(code(
    "_SEED_RE = re.compile(r\"_s(\\d+)_\")\n",
    "\n",
    "\n",
    "def _extract_seed(run_dir_name: str, summary: dict) -> int:\n",
    "    seed = summary.get(\"run\", {}).get(\"seed\")\n",
    "    if seed is not None:\n",
    "        return int(seed)\n",
    "    m = _SEED_RE.search(run_dir_name)\n",
    "    return int(m.group(1)) if m else 42\n",
    "\n",
    "\n",
    "def load_jobB_runs(*directories: Path) -> pd.DataFrame:\n",
    "    rows = []\n",
    "    for directory in directories:\n",
    "        if not directory.is_dir():\n",
    "            continue\n",
    "        for run_dir in sorted(p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(\"_\")):\n",
    "            summary_path = run_dir / \"benchmark_summary.json\"\n",
    "            if not summary_path.is_file():\n",
    "                continue\n",
    "            b = json.loads(summary_path.read_text())\n",
    "            run_meta = {\n",
    "                \"source\": directory.name,\n",
    "                \"run_dir\": run_dir.name,\n",
    "                \"run_path\": str(run_dir),\n",
    "                \"run_id\": b.get(\"run\", {}).get(\"run_id\"),\n",
    "                \"seed\": _extract_seed(run_dir.name, b),\n",
    "                \"corruption_enabled\": b.get(\"corruption\", {}).get(\"enabled\"),\n",
    "                \"corruption_type\": b.get(\"corruption\", {}).get(\"type\"),\n",
    "                \"corruption_severity\": b.get(\"corruption\", {}).get(\"severity\"),\n",
    "                \"dataset_path\": b.get(\"dataset\", {}).get(\"path\"),\n",
    "                \"resize_w\": b.get(\"preprocessing\", {}).get(\"resize\", {}).get(\"width\"),\n",
    "                \"resize_h\": b.get(\"preprocessing\", {}).get(\"resize\", {}).get(\"height\"),\n",
    "            }\n",
    "            for m in b.get(\"models\", []):\n",
    "                row = {**run_meta}\n",
    "                row.update({k: v for k, v in m.items() if k not in {\"model_cfg\", \"per_defect_recall\", \"per_defect_support\"}})\n",
    "                row[\"per_defect_recall\"] = m.get(\"per_defect_recall\", {})\n",
    "                row[\"per_defect_support\"] = m.get(\"per_defect_support\", {})\n",
    "                rows.append(row)\n",
    "    return pd.DataFrame(rows)\n",
    "\n",
    "\n",
    "def load_predictions(run_path: str | Path, model: str, validation: bool = False) -> pd.DataFrame:\n",
    "    fname = (\"validation_predictions_\" if validation else \"predictions_\") + f\"{model}.json\"\n",
    "    path = Path(run_path) / fname\n",
    "    if not path.is_file():\n",
    "        return pd.DataFrame()\n",
    "    return pd.DataFrame(json.loads(path.read_text()))\n",
    "\n",
    "\n",
    "df = load_jobB_runs(*JOBB_SOURCES)\n",
    "print(f\"Loaded {len(df)} model rows from {df['run_dir'].nunique()} run directories\"\n",
    "      f\" across {df['source'].nunique()} source dirs.\")\n",
    "df[[\"source\", \"run_dir\", \"model\", \"seed\", \"train_samples\", \"val_samples\", \"test_samples\"]]"
))

cells.append(md(
    "## §1 Summary metrics per model\n",
    "\n",
    "Headline numbers, one row per model. The `_industrial` block (recall@1pct FPR / recall@5pct FPR /\n",
    "macro_recall) is the metric set the thesis uses for the rolling-window industrial benchmark."
))

cells.append(code(
    "headline_cols = [\n",
    "    \"model\", \"seed\",\n",
    "    \"train_samples\", \"val_samples\", \"test_samples\",\n",
    "    \"auroc\", \"aupr\", \"f1\", \"precision\", \"recall\", \"accuracy\",\n",
    "    \"recall_at_fpr_1pct\", \"recall_at_fpr_5pct\", \"macro_recall\", \"weighted_recall\",\n",
    "    \"threshold_mode\", \"threshold_used\",\n",
    "    \"fit_seconds\", \"predict_seconds\", \"ms_per_image\", \"fps\", \"peak_vram_mb\",\n",
    "]\n",
    "summary = df[headline_cols].copy()\n",
    "summary = summary.sort_values([\"model\", \"seed\"]).reset_index(drop=True)\n",
    "summary.round(4)"
))

cells.append(code(
    "# val (calibration) vs test (held-out) — generalization gap, per (model, seed).\n",
    "gap_cols = [\"model\", \"seed\", \"val_f1\", \"f1\", \"val_auroc\", \"auroc\", \"val_aupr\", \"aupr\", \"val_recall\", \"recall\", \"val_precision\", \"precision\"]\n",
    "gap = df[gap_cols].copy()\n",
    "for m in [\"f1\", \"auroc\", \"aupr\", \"recall\", \"precision\"]:\n",
    "    gap[f\"{m}_gap\"] = gap[m] - gap[f\"val_{m}\"]\n",
    "gap.sort_values([\"model\", \"seed\"]).set_index([\"model\", \"seed\"]).round(4)"
))

cells.append(code(
    "# Per-defect-type recall — one row per (model, seed).\n",
    "all_defects = sorted({d for r in df[\"per_defect_recall\"] for d in r})\n",
    "rows = []\n",
    "for _, r in df.iterrows():\n",
    "    rec = {\"model\": r[\"model\"], \"seed\": r[\"seed\"]}\n",
    "    for d in all_defects:\n",
    "        rec[d] = r[\"per_defect_recall\"].get(d, np.nan)\n",
    "    rows.append(rec)\n",
    "per_defect_df = pd.DataFrame(rows).sort_values([\"model\", \"seed\"]).set_index([\"model\", \"seed\"]).round(4)\n",
    "per_defect_df"
))

cells.append(code(
    "# Per-defect support — same for every model in JobB (single dataset), so we show it once.\n",
    "support = next(iter(df[\"per_defect_support\"]), {})\n",
    "support_df = pd.DataFrame({\"defect\": list(support), \"n_test_samples\": list(support.values())}).sort_values(\"n_test_samples\", ascending=False)\n",
    "support_df"
))

cells.append(md(
    "## §2 Graphical comparison\n",
    "\n",
    "Three views: a metric grid, a latency-vs-F1 Pareto, and the val→test generalization gap."
))

cells.append(code(
    "# One bar per (model, seed). Models with one seed show a single bar; multi-seed models\n",
    "# show one bar per seed so cross-seed variance is visible at a glance.\n",
    "metrics = [\"auroc\", \"aupr\", \"f1\", \"precision\", \"recall\", \"recall_at_fpr_1pct\"]\n",
    "df_plot = df.sort_values([\"model\", \"seed\"]).reset_index(drop=True)\n",
    "labels = [f\"{m.replace('anomalib_', '')}\\ns={s}\" for m, s in zip(df_plot[\"model\"], df_plot[\"seed\"])]\n",
    "fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharey=True)\n",
    "for ax, metric in zip(axes.ravel(), metrics):\n",
    "    vals = df_plot[metric].values\n",
    "    bars = ax.bar(range(len(vals)), vals, color=\"#4C72B0\")\n",
    "    ax.set_xticks(range(len(vals)))\n",
    "    ax.set_xticklabels(labels, rotation=45, ha=\"right\", fontsize=7)\n",
    "    ax.set_title(metric)\n",
    "    ax.set_ylim(0, 1.05)\n",
    "    ax.axhline(1.0, color=\"#c44e52\", lw=0.7, ls=\":\")\n",
    "    for b, v in zip(bars, vals):\n",
    "        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f\"{v:.3f}\", ha=\"center\", va=\"bottom\", fontsize=7)\n",
    "fig.suptitle(\"JobB — quality metrics per (model, seed)\", fontsize=12)\n",
    "fig.tight_layout()\n",
    "plt.show()"
))

cells.append(code(
    "# Pareto: latency vs F1, one point per (model, seed). Point colour by model,\n",
    "# annotation shows seed when there are multiple per model.\n",
    "fig, ax = plt.subplots(figsize=(9, 5.5))\n",
    "models = sorted(df[\"model\"].unique())\n",
    "cmap = plt.get_cmap(\"tab10\")\n",
    "color_by = {m: cmap(i % 10) for i, m in enumerate(models)}\n",
    "for m in models:\n",
    "    sub = df[df[\"model\"] == m]\n",
    "    ax.scatter(sub[\"ms_per_image\"], sub[\"f1\"], s=80, color=color_by[m], label=m.replace(\"anomalib_\", \"\"))\n",
    "    multi = len(sub) > 1\n",
    "    for _, r in sub.iterrows():\n",
    "        tag = f\"s={r['seed']}\" if multi else r[\"model\"].replace(\"anomalib_\", \"\")\n",
    "        ax.annotate(tag, (r[\"ms_per_image\"], r[\"f1\"]),\n",
    "                    xytext=(5, 5), textcoords=\"offset points\", fontsize=8)\n",
    "ax.set_xlabel(\"ms / image (lower is better)\")\n",
    "ax.set_ylabel(\"F1 (higher is better)\")\n",
    "ax.set_title(\"Latency vs quality — JobB Pareto (one point per seed)\")\n",
    "ax.grid(alpha=0.3)\n",
    "ax.legend(fontsize=8, loc=\"lower left\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
))

cells.append(code(
    "# Generalization gap per (model, seed). When val_f1 saturates at 1.0 across all seeds\n",
    "# (as on Deceuninck) the val→test gap is the meaningful number, not val_f1 itself.\n",
    "fig, ax = plt.subplots(figsize=(11, 5.5))\n",
    "x = np.arange(len(df_plot))\n",
    "w = 0.4\n",
    "ax.bar(x - w/2, df_plot[\"val_f1\"].values, w, label=\"val_f1 (calibration)\", color=\"#dd8452\")\n",
    "ax.bar(x + w/2, df_plot[\"f1\"].values, w, label=\"f1 (held-out test)\", color=\"#4C72B0\")\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(labels, rotation=45, ha=\"right\", fontsize=7)\n",
    "ax.set_ylim(0, 1.05)\n",
    "ax.set_ylabel(\"F1\")\n",
    "ax.set_title(\"Generalization: validation F1 vs test F1, per (model, seed)\")\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()"
))

cells.append(md(
    "## §3 Coherence / sanity checks\n",
    "\n",
    "Goal: surface anything that *should* be a decimal but landed exactly on `0.0` or `1.0`, plus a few\n",
    "consistency checks between reported numbers and the underlying samples.\n",
    "\n",
    "Why it matters: with only 478 test images, a single small bug (e.g. all scores collapsing to one\n",
    "value, or threshold falling outside the score range) can produce \"perfect\" metrics that do not\n",
    "reflect real performance."
))

cells.append(code(
    "# 3.1 — flag suspiciously round metric values, per (model, seed)\n",
    "watch = [\"auroc\", \"aupr\", \"f1\", \"precision\", \"recall\",\n",
    "         \"val_auroc\", \"val_f1\", \"val_precision\", \"val_recall\",\n",
    "         \"recall_at_fpr_1pct\", \"recall_at_fpr_5pct\",\n",
    "         \"macro_recall\", \"weighted_recall\", \"accuracy\"]\n",
    "flags = []\n",
    "EPS = 1e-9\n",
    "for _, r in df.iterrows():\n",
    "    for col in watch:\n",
    "        v = r.get(col)\n",
    "        if v is None or pd.isna(v):\n",
    "            continue\n",
    "        if abs(v - 1.0) < EPS:\n",
    "            flags.append({\"model\": r[\"model\"], \"seed\": r[\"seed\"], \"metric\": col, \"value\": v, \"note\": \"== 1.0 exactly\"})\n",
    "        elif abs(v) < EPS:\n",
    "            flags.append({\"model\": r[\"model\"], \"seed\": r[\"seed\"], \"metric\": col, \"value\": v, \"note\": \"== 0.0 exactly\"})\n",
    "        elif v > 0.9999 and v < 1.0:\n",
    "            flags.append({\"model\": r[\"model\"], \"seed\": r[\"seed\"], \"metric\": col, \"value\": v, \"note\": \"> 0.9999 (effectively saturated)\"})\n",
    "flag_df = pd.DataFrame(flags)\n",
    "if flag_df.empty:\n",
    "    print(\"No saturated metrics — all values are intermediate decimals.\")\n",
    "else:\n",
    "    print(f\"{len(flag_df)} saturated metric values:\")\n",
    "    display(flag_df.sort_values([\"model\", \"seed\", \"metric\"]).reset_index(drop=True))"
))

cells.append(code(
    "# 3.2 — sample-count and time consistency.\n",
    "# train_samples + val_samples + test_samples should match across models on the SAME dataset run.\n",
    "consistency = df.groupby(\"dataset_path\")[[\"train_samples\", \"val_samples\", \"test_samples\"]].agg([\"min\", \"max\"])\n",
    "consistency"
))

cells.append(code(
    "# 3.3 — does the calibrated threshold sit inside the actual score range on val and test?\n",
    "rows = []\n",
    "for _, r in df.iterrows():\n",
    "    test = load_predictions(r[\"run_path\"], r[\"model\"], validation=False)\n",
    "    val = load_predictions(r[\"run_path\"], r[\"model\"], validation=True)\n",
    "    if test.empty:\n",
    "        continue\n",
    "    rows.append({\n",
    "        \"model\": r[\"model\"],\n",
    "        \"seed\": r[\"seed\"],\n",
    "        \"threshold\": r[\"threshold_used\"],\n",
    "        \"val_score_min\": val[\"score\"].min() if not val.empty else None,\n",
    "        \"val_score_max\": val[\"score\"].max() if not val.empty else None,\n",
    "        \"val_score_unique\": val[\"score\"].nunique() if not val.empty else None,\n",
    "        \"test_score_min\": test[\"score\"].min(),\n",
    "        \"test_score_median\": test[\"score\"].median(),\n",
    "        \"test_score_max\": test[\"score\"].max(),\n",
    "        \"test_score_unique\": test[\"score\"].nunique(),\n",
    "        \"frac_test_above_thr\": (test[\"score\"] > r[\"threshold_used\"]).mean(),\n",
    "    })\n",
    "score_range = pd.DataFrame(rows).sort_values([\"model\", \"seed\"]).set_index([\"model\", \"seed\"]).round(6)\n",
    "score_range"
))

cells.append(code(
    "# 3.4 — class balance per (model, seed). Same dataset and split rules → all rows\n",
    "# should match within a seed; cross-seed differences here would mean the splitter is\n",
    "# allocating different test sets per seed, which is the expected behaviour for\n",
    "# stratify=True + different RNG.\n",
    "rows = []\n",
    "for _, r in df.iterrows():\n",
    "    test = load_predictions(r[\"run_path\"], r[\"model\"])\n",
    "    if test.empty:\n",
    "        continue\n",
    "    rows.append({\n",
    "        \"model\": r[\"model\"],\n",
    "        \"seed\": r[\"seed\"],\n",
    "        \"n_test\": len(test),\n",
    "        \"n_good\": int((test[\"label\"] == 0).sum()),\n",
    "        \"n_defect\": int((test[\"label\"] == 1).sum()),\n",
    "        \"defect_rate\": float((test[\"label\"] == 1).mean()),\n",
    "    })\n",
    "pd.DataFrame(rows).sort_values([\"model\", \"seed\"]).set_index([\"model\", \"seed\"]).round(4)"
))

cells.append(code(
    "# 3.5 — fit-time vs sample-count plausibility, per (model, seed).\n",
    "(df[[\"model\", \"seed\", \"train_samples\", \"fit_seconds\", \"predict_seconds\", \"ms_per_image\"]]\n",
    "   .assign(seconds_per_train_sample=lambda x: x[\"fit_seconds\"] / x[\"train_samples\"])\n",
    "   .round(2)\n",
    "   .sort_values([\"model\", \"seed\"]))"
))

cells.append(md(
    "**How to read §3:**\n",
    "\n",
    "- A `1.0000` AUROC on 478 images is not impossible but it should always be cross-checked against\n",
    "  the score histogram in §4 — perfect separability looks like two non-overlapping clusters.\n",
    "- A threshold that sits *outside* the validation score range, or where `frac_test_above_thr` is\n",
    "  close to `0` or `1`, means the calibration step was effectively a no-op (everything ends up\n",
    "  predicted as one class).\n",
    "- `test_score_unique` close to `1–2` is a hard fail: scores collapsed and the model is constant.\n",
    "- For the seconds-per-sample column, models with comparable architectures should land in the same\n",
    "  order of magnitude; an outlier signals a config drift between runs."
))

cells.append(md(
    "## §4 Score distributions\n",
    "\n",
    "One subplot per model, log-scale y to make the rare-class tail visible. Vertical line is the\n",
    "calibrated threshold; bars to its right are predicted as *anomaly*."
))

cells.append(code(
    "# Score histograms — to keep the grid manageable across many seeds we show one\n",
    "# histogram per (model, seed) but use the seed=42 baseline as the canonical row when\n",
    "# present, and only add extra seeds for models that have multi-seed coverage.\n",
    "df_hist = df.sort_values([\"model\", \"seed\"]).reset_index(drop=True)\n",
    "n = len(df_hist)\n",
    "ncols = 2\n",
    "nrows = (n + ncols - 1) // ncols\n",
    "fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows))\n",
    "axes = np.array(axes).reshape(nrows, ncols)\n",
    "for ax, (_, r) in zip(axes.ravel(), df_hist.iterrows()):\n",
    "    test = load_predictions(r[\"run_path\"], r[\"model\"])\n",
    "    if test.empty:\n",
    "        ax.set_visible(False)\n",
    "        continue\n",
    "    good_scores = test.loc[test[\"label\"] == 0, \"score\"]\n",
    "    bad_scores = test.loc[test[\"label\"] == 1, \"score\"]\n",
    "    bins = 50\n",
    "    ax.hist(good_scores, bins=bins, alpha=0.65, label=f\"good (n={len(good_scores)})\", color=\"#55a868\")\n",
    "    ax.hist(bad_scores, bins=bins, alpha=0.65, label=f\"defect (n={len(bad_scores)})\", color=\"#c44e52\")\n",
    "    ax.axvline(r[\"threshold_used\"], color=\"black\", ls=\"--\", lw=1, label=f\"thr={r['threshold_used']:.4f}\")\n",
    "    ax.set_yscale(\"log\")\n",
    "    ax.set_title(f\"{r['model']} (seed={r['seed']})\")\n",
    "    ax.set_xlabel(\"anomaly score\")\n",
    "    ax.set_ylabel(\"count (log)\")\n",
    "    ax.legend(fontsize=8, loc=\"upper left\")\n",
    "for ax in axes.ravel()[len(df_hist):]:\n",
    "    ax.set_visible(False)\n",
    "fig.suptitle(\"JobB — test score distributions per (model, seed)\", fontsize=12)\n",
    "fig.tight_layout()\n",
    "plt.show()"
))

cells.append(md(
    "## §5 Confusion matrices\n",
    "\n",
    "Recomputed from `predictions_*.json` so the headline F1 in §1 can be audited end-to-end."
))

cells.append(code(
    "def confusion(test_df: pd.DataFrame) -> pd.DataFrame:\n",
    "    cm = pd.crosstab(\n",
    "        test_df[\"label\"].map({0: \"good\", 1: \"defect\"}),\n",
    "        test_df[\"pred_is_anomaly\"].map({0: \"pred_good\", 1: \"pred_defect\"}),\n",
    "        rownames=[\"actual\"], colnames=[\"predicted\"], dropna=False,\n",
    "    )\n",
    "    for col in [\"pred_good\", \"pred_defect\"]:\n",
    "        if col not in cm.columns:\n",
    "            cm[col] = 0\n",
    "    for row in [\"good\", \"defect\"]:\n",
    "        if row not in cm.index:\n",
    "            cm.loc[row] = 0\n",
    "    return cm.loc[[\"good\", \"defect\"], [\"pred_good\", \"pred_defect\"]]\n",
    "\n",
    "for _, r in df.sort_values([\"model\", \"seed\"]).iterrows():\n",
    "    test = load_predictions(r[\"run_path\"], r[\"model\"])\n",
    "    if test.empty:\n",
    "        continue\n",
    "    cm = confusion(test)\n",
    "    tp = cm.loc[\"defect\", \"pred_defect\"]\n",
    "    fp = cm.loc[\"good\", \"pred_defect\"]\n",
    "    fn = cm.loc[\"defect\", \"pred_good\"]\n",
    "    tn = cm.loc[\"good\", \"pred_good\"]\n",
    "    prec = tp / (tp + fp) if (tp + fp) else float(\"nan\")\n",
    "    rec = tp / (tp + fn) if (tp + fn) else float(\"nan\")\n",
    "    print(f\"=== {r['model']} (seed={r['seed']}) ===\")\n",
    "    print(cm.to_string())\n",
    "    print(f\"recomputed precision={prec:.4f} recall={rec:.4f} (reported precision={r['precision']:.4f} recall={r['recall']:.4f})\")\n",
    "    print()"
))

cells.append(md(
    "## §6 JobB vs JobA reference\n",
    "\n",
    "Reuse the pre-computed comparison TSV (`_analysis/compare_jobB_vs_jobA.tsv`) — it already has the\n",
    "per-model JobA mean ± std baselines next to the JobB single number.\n",
    "\n",
    "Only `patchcore`, `padim`, and `subspacead` are present in that TSV because those are the three\n",
    "models JobA evaluated across all MVTec / Real-IAD categories. The new JobB-only models (`draem`,\n",
    "`stfpm`, `rd4ad`) need to be paired with their JobA counterparts manually."
))

cells.append(code(
    "tsv_path = ANALYSIS_DIR / \"compare_jobB_vs_jobA.tsv\"\n",
    "if tsv_path.is_file():\n",
    "    cmp_df = pd.read_csv(tsv_path, sep=\"\\t\")\n",
    "    display_cols = [\"model\", \"auroc_jobA_mean\", \"auroc_jobA_std\", \"auroc_jobB\", \"auroc_diff\",\n",
    "                    \"f1_jobA_mean\", \"f1_jobB\", \"f1_diff\", \"recall_jobA_mean\", \"recall_jobB\", \"recall_diff\",\n",
    "                    \"ms_per_image_jobA_mean\", \"ms_per_image_jobB\", \"ms_per_image_diff\"]\n",
    "    display_cols = [c for c in display_cols if c in cmp_df.columns]\n",
    "    display(cmp_df[display_cols].round(3))\n",
    "else:\n",
    "    print(f\"(missing {tsv_path} — skip)\")"
))

cells.append(code(
    "# Bonus: pair the JobB-only models against their JobA per-category mean.\n",
    "extra_models = [\"anomalib_draem\", \"anomalib_stfpm\", \"rd4ad\"]\n",
    "extra_rows = []\n",
    "if JOBA_DIR.is_dir():\n",
    "    for run_dir in sorted(p for p in JOBA_DIR.iterdir() if p.is_dir() and not p.name.startswith(\"_\")):\n",
    "        sp = run_dir / \"benchmark_summary.json\"\n",
    "        if not sp.is_file():\n",
    "            continue\n",
    "        b = json.loads(sp.read_text())\n",
    "        for m in b.get(\"models\", []):\n",
    "            if m[\"model\"] in extra_models:\n",
    "                extra_rows.append({\n",
    "                    \"jobA_run\": run_dir.name,\n",
    "                    \"model\": m[\"model\"],\n",
    "                    \"auroc\": m.get(\"auroc\"),\n",
    "                    \"f1\": m.get(\"f1\"),\n",
    "                    \"recall\": m.get(\"recall\"),\n",
    "                    \"ms_per_image\": m.get(\"ms_per_image\"),\n",
    "                    \"fit_seconds\": m.get(\"fit_seconds\"),\n",
    "                })\n",
    "if extra_rows:\n",
    "    extra_df = pd.DataFrame(extra_rows)\n",
    "    summary = extra_df.groupby(\"model\")[[\"auroc\", \"f1\", \"recall\", \"ms_per_image\", \"fit_seconds\"]].agg([\"mean\", \"std\", \"count\"]).round(3)\n",
    "    print(\"=== JobA aggregate for JobB-only models ===\")\n",
    "    display(summary)\n",
    "    print(\"\\n=== JobB per-seed numbers for the same models ===\")\n",
    "    display(df[df[\"model\"].isin(extra_models)][[\"model\", \"seed\", \"auroc\", \"f1\", \"recall\", \"ms_per_image\", \"fit_seconds\"]].sort_values([\"model\", \"seed\"]).round(3))\n",
    "else:\n",
    "    print(\"(no JobA runs covering draem/stfpm/rd4ad found)\")"
))

cells.append(md(
    "## §7 Multi-seed coherence\n",
    "\n",
    "The original JobB run was a single seed (`seed=42`). With the seed-sweep outputs now under\n",
    "`data/outputs/jobB_val_defect_and_seed/`, several models have 2–4 seeds available. This section\n",
    "answers the question: **does the multi-seed evidence change which numbers we should trust?**\n",
    "\n",
    "Coherence checks here:\n",
    "1. Coverage matrix — which `(model, seed)` pairs we actually have.\n",
    "2. Per-model mean ± std of the headline metrics (only computed for models with ≥2 seeds).\n",
    "3. Threshold stability — how much the calibrated threshold varies across seeds.\n",
    "4. Saturation re-check — which metrics still hit `1.0` in *every* seed (truly saturated)\n",
    "   versus only in some seeds (artifact of a single split)."
))

cells.append(code(
    "# §7.1 — coverage matrix: rows = model, cols = seed, value = number of runs.\n",
    "coverage = (df.assign(n=1)\n",
    "              .pivot_table(index=\"model\", columns=\"seed\", values=\"n\", aggfunc=\"sum\", fill_value=0)\n",
    "              .astype(int))\n",
    "coverage[\"total_seeds\"] = (coverage > 0).sum(axis=1)\n",
    "print(\"=== seeds available per model ===\")\n",
    "coverage"
))

cells.append(code(
    "# §7.2 — mean ± std for headline metrics, only for models with >= 2 seeds.\n",
    "agg_metrics = [\"auroc\", \"aupr\", \"f1\", \"precision\", \"recall\",\n",
    "               \"recall_at_fpr_1pct\", \"macro_recall\",\n",
    "               \"threshold_used\", \"ms_per_image\", \"peak_vram_mb\"]\n",
    "agg = (df.groupby(\"model\")[agg_metrics]\n",
    "         .agg([\"mean\", \"std\", \"min\", \"max\", \"count\"])\n",
    "         .round(4))\n",
    "# Drop single-seed models from the aggregate view so std isn't NaN-misleading.\n",
    "n_seeds = df.groupby(\"model\")[\"seed\"].nunique()\n",
    "multi_seed_models = n_seeds[n_seeds >= 2].index.tolist()\n",
    "single_seed_models = n_seeds[n_seeds < 2].index.tolist()\n",
    "if multi_seed_models:\n",
    "    print(\"=== multi-seed aggregates (n_seeds >= 2) ===\")\n",
    "    display(agg.loc[multi_seed_models])\n",
    "if single_seed_models:\n",
    "    print(f\"\\nSingle-seed only (excluded from aggregates): {single_seed_models}\")"
))

cells.append(code(
    "# §7.3 — threshold stability across seeds.\n",
    "# Coefficient of variation = std/mean. Small CV = calibration is reproducible.\n",
    "if multi_seed_models:\n",
    "    thr_stats = (df[df[\"model\"].isin(multi_seed_models)]\n",
    "                   .groupby(\"model\")[\"threshold_used\"]\n",
    "                   .agg([\"mean\", \"std\", \"min\", \"max\"]))\n",
    "    thr_stats[\"cv\"] = thr_stats[\"std\"] / thr_stats[\"mean\"]\n",
    "    thr_stats = thr_stats.round(4)\n",
    "    print(\"=== threshold_used stability across seeds ===\")\n",
    "    display(thr_stats)\n",
    "else:\n",
    "    print(\"(no multi-seed models — skip)\")"
))

cells.append(code(
    "# §7.4 — saturation re-check across seeds.\n",
    "# A metric that was '== 1.0 exactly' in seed=42 might now be intermediate at other seeds.\n",
    "# Compare: which metrics stay saturated across ALL seeds vs varied at least once?\n",
    "EPS = 1e-9\n",
    "watch = [\"auroc\", \"aupr\", \"f1\", \"precision\", \"recall\",\n",
    "         \"val_auroc\", \"val_f1\", \"val_precision\", \"val_recall\",\n",
    "         \"recall_at_fpr_1pct\", \"recall_at_fpr_5pct\",\n",
    "         \"macro_recall\", \"weighted_recall\", \"accuracy\"]\n",
    "rows = []\n",
    "for model, grp in df.groupby(\"model\"):\n",
    "    if len(grp) < 2:\n",
    "        continue\n",
    "    for col in watch:\n",
    "        vals = grp[col].dropna().values\n",
    "        if len(vals) == 0:\n",
    "            continue\n",
    "        all_one = bool(np.all(np.abs(vals - 1.0) < EPS))\n",
    "        any_one = bool(np.any(np.abs(vals - 1.0) < EPS))\n",
    "        if all_one or any_one:\n",
    "            rows.append({\n",
    "                \"model\": model,\n",
    "                \"metric\": col,\n",
    "                \"n_seeds\": len(vals),\n",
    "                \"min\": float(vals.min()),\n",
    "                \"max\": float(vals.max()),\n",
    "                \"all_seeds_eq_1\": all_one,\n",
    "                \"some_seed_eq_1\": any_one,\n",
    "            })\n",
    "sat = pd.DataFrame(rows)\n",
    "if sat.empty:\n",
    "    print(\"No multi-seed metric ever reached 1.0 — no saturation to discuss.\")\n",
    "else:\n",
    "    truly = sat[sat[\"all_seeds_eq_1\"]]\n",
    "    sometimes = sat[(sat[\"some_seed_eq_1\"]) & (~sat[\"all_seeds_eq_1\"])]\n",
    "    print(f\"Metrics that stay at exactly 1.0 across ALL seeds (truly saturated): {len(truly)}\")\n",
    "    if not truly.empty:\n",
    "        display(truly[[\"model\", \"metric\", \"n_seeds\", \"min\", \"max\"]].sort_values([\"model\", \"metric\"]).reset_index(drop=True))\n",
    "    print(f\"\\nMetrics that hit 1.0 in SOME seed but vary in others (single-seed artifact): {len(sometimes)}\")\n",
    "    if not sometimes.empty:\n",
    "        display(sometimes[[\"model\", \"metric\", \"n_seeds\", \"min\", \"max\"]].sort_values([\"model\", \"metric\"]).reset_index(drop=True))"
))

cells.append(code(
    "# §7.5 — visualization: mean ± std of f1 / auroc / recall / aupr per multi-seed model,\n",
    "# with single-seed models drawn as a single point (no error bar).\n",
    "if multi_seed_models or single_seed_models:\n",
    "    plot_metrics = [\"auroc\", \"aupr\", \"f1\", \"recall\"]\n",
    "    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(4.0 * len(plot_metrics), 5), sharey=True)\n",
    "    model_order = sorted(df[\"model\"].unique())\n",
    "    x = np.arange(len(model_order))\n",
    "    for ax, metric in zip(axes, plot_metrics):\n",
    "        means, stds, ns = [], [], []\n",
    "        for m in model_order:\n",
    "            vals = df.loc[df[\"model\"] == m, metric].values\n",
    "            means.append(np.mean(vals))\n",
    "            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)\n",
    "            ns.append(len(vals))\n",
    "        colors = [\"#4C72B0\" if n >= 2 else \"#bbbbbb\" for n in ns]\n",
    "        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor=\"black\", linewidth=0.5)\n",
    "        for xi, mn, n in zip(x, means, ns):\n",
    "            ax.text(xi, mn + 0.015, f\"n={n}\", ha=\"center\", va=\"bottom\", fontsize=8)\n",
    "        ax.set_title(metric)\n",
    "        ax.set_xticks(x)\n",
    "        ax.set_xticklabels([m.replace(\"anomalib_\", \"\") for m in model_order], rotation=30, ha=\"right\")\n",
    "        ax.set_ylim(0, 1.08)\n",
    "        ax.axhline(1.0, color=\"#c44e52\", lw=0.7, ls=\":\")\n",
    "    fig.suptitle(\"JobB — mean ± std across seeds (grey bars = single seed only, no error bar)\", fontsize=12)\n",
    "    fig.tight_layout()\n",
    "    plt.show()"
))

cells.append(code(
    "# §7.6 — per-defect recall variance across seeds (multi-seed models only).\n",
    "# Surfaces which defect classes are stable predictions vs which flip across splits.\n",
    "rows = []\n",
    "for _, r in df.iterrows():\n",
    "    for d, v in (r[\"per_defect_recall\"] or {}).items():\n",
    "        rows.append({\"model\": r[\"model\"], \"seed\": r[\"seed\"], \"defect\": d, \"recall\": v})\n",
    "long = pd.DataFrame(rows)\n",
    "if not long.empty:\n",
    "    pd_stats = (long[long[\"model\"].isin(multi_seed_models)]\n",
    "                  .groupby([\"model\", \"defect\"])[\"recall\"]\n",
    "                  .agg([\"mean\", \"std\", \"min\", \"max\", \"count\"])\n",
    "                  .round(4))\n",
    "    if not pd_stats.empty:\n",
    "        print(\"=== per-defect recall — mean ± std across seeds ===\")\n",
    "        display(pd_stats)\n",
    "    else:\n",
    "        print(\"(no multi-seed models with per-defect recall)\")"
))

cells.append(md(
    "**Reading §7:**\n",
    "\n",
    "- **§7.1 coverage** tells you how much weight each per-model aggregate carries. Anything with\n",
    "  `total_seeds < 3` should be reported as a point estimate, not a mean ± std.\n",
    "- **§7.2 aggregates** is the table that goes into the thesis. Models with `count == 1` are\n",
    "  excluded so a misleading `std=NaN` does not leak into the writeup.\n",
    "- **§7.3 threshold CV** is the calibration-stability check. A coefficient of variation under\n",
    "  ~0.05 means `val_f1` (or `val_quantile`) reliably picks the same operating point across\n",
    "  splits. If CV is large for a model, the threshold mode for that model is fragile and the\n",
    "  test-set numbers depend on which goods happen to be in val.\n",
    "- **§7.4 saturation re-check** is the *coherence* answer. Metrics in `truly` saturated stayed at\n",
    "  exactly 1.0 across *every* available seed — those are real but should still be retested with\n",
    "  more seeds. Metrics in `sometimes` were saturated only at seed=42 — those are the\n",
    "  single-seed-artifact numbers we suspected; they no longer support a `f1=1.0` claim.\n",
    "- **§7.5 visualization** is the figure for the thesis. Grey bars (single-seed) communicate that\n",
    "  the value is a point estimate, not an aggregate, without dropping the model entirely.\n",
    "- **§7.6 per-defect variance** highlights which defect classes are *robustly* recognised. A class\n",
    "  with `std≈0` across seeds is a stable prediction; a class with high std reveals that recognition\n",
    "  depends on which examples landed in train vs test."
))

cells.append(md(
    "## §8 Engineering suggestions — what to look at next\n",
    "\n",
    "Framed for the thesis goal (real-time visual defect detection that holds up on a single industrial\n",
    "dataset). These are the questions the data above cannot fully answer on its own — flagging them so\n",
    "the next iteration knows where to dig.\n",
    "\n",
    "1. **Saturated metrics need a second test set.** Any model showing `auroc == 1.0` and `f1 ≥ 0.99`\n",
    "   on a single 478-image test split should be retested against a *different* split (different\n",
    "   `seed`, or a held-out batch from a later production day). This rules out the case where the\n",
    "   train / test split happens to put all hard images on one side. The current `seed=42` is fixed —\n",
    "   running with `seed=7, 17, 123` and reporting mean ± std would convert these single-shot numbers\n",
    "   into something the thesis can defend.\n",
    "\n",
    "2. **Threshold position is the silent failure mode.** §3.3 shows the calibrated threshold relative\n",
    "   to the test score range. If a model's threshold sits below `min(test_score)` *and* above\n",
    "   `min(val_score)` it just means \"label everything as anomaly,\" which inflates recall to ≈1 while\n",
    "   precision becomes whatever the prevalence allows. Treat any model where\n",
    "   `frac_test_above_thr ∈ {<0.05, >0.95}` as effectively un-calibrated and rerun with\n",
    "   `threshold_mode=val_quantile` for comparison.\n",
    "\n",
    "3. **Generalization gap matters more than headline F1.** The `*_gap` columns in §1 are the ones to\n",
    "   put in the thesis. A model with `val_f1=0.95` and `f1=0.96` is more trustworthy than one with\n",
    "   `val_f1=0.90` and `f1=1.00`, even though the latter looks better on paper.\n",
    "\n",
    "4. **Per-defect recall is where the deceuninck story gets interesting.** Support is heavily skewed\n",
    "   (Scratch=353 vs Black-spots=16). A model that scores `recall=1.0` on Black-spots but `0.83` on\n",
    "   Scratch is mostly being judged on Scratch. Plot recall against support and only celebrate models\n",
    "   that hold the line on the high-support classes.\n",
    "\n",
    "5. **Latency is reported as `ms_per_image` averaged over the full test set, single-image batch.**\n",
    "   For a real-time deployment the relevant number is `p95_latency_ms` from `live_status_*.json`,\n",
    "   not the mean — add it to the comparison table when the SLA discussion comes up.\n",
    "\n",
    "6. **Fit-time delta vs JobA is the throughput story.** §6's `extra_models` block compares\n",
    "   `fit_seconds` JobA-mean against JobB. If JobB ran ~3× faster than the JobA average for the same\n",
    "   model, that is either (a) a smaller training set on Deceuninck (`train_samples=161` vs 200–500\n",
    "   typical for Real-IAD), or (b) the run hit an early-stopping path. Worth checking the\n",
    "   per-epoch loss curve before writing it up as a speedup.\n",
    "\n",
    "7. **Cross-check with `live_status_*.json`.** The `fail_count` field is the count of frames the\n",
    "   live consumer flagged. If `fail_count / frames_seen` differs from `1 - precision * recall / f1`\n",
    "   it usually means the live pipeline applied a different threshold or skipped frames — a\n",
    "   discrepancy here would invalidate any latency claim made from the same run.\n",
    "\n",
    "8. **Persist a `manifest.csv`.** Right now run discovery is by directory name + timestamp. A flat\n",
    "   manifest (`run_id, model, dataset, threshold_mode, seed, git_sha`) would make every future\n",
    "   notebook one `pd.read_csv` away from the answer instead of re-walking the filesystem."
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB_PATH} ({NB_PATH.stat().st_size} bytes, {len(cells)} cells)")
