"""Create auditable CSV/Markdown comparisons for the 2026-07-24 rerun."""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


BASE = r"D:\sc_dataset"
FULL_DIR = os.path.join(
    BASE,
    "experiments",
    "metrics",
    "archer_core_20260724_k20_p1of15_exact_hig500_16benchmark",
)
RARE_DIR = os.path.join(
    BASE,
    "experiments",
    "metrics",
    "archer_core_20260724_k20_p1of15_exact_hig500_rare",
)
HIG500_FULL = os.path.join(
    BASE, "experiments", "metrics", "archer_v32_fig2a_merged_hig500_20260722.csv"
)
HIG500_RARE_DIR = os.path.join(
    BASE, "experiments", "metrics", "archer_v32_hig500_16datasets_20260721"
)
OUT_DIR = os.path.join(FULL_DIR, "comparison_to_hig500_20260724_final")

FULL_METRICS = [
    "ACC",
    "ARI",
    "NMI",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Purity",
]
RARE_METRICS = [
    "ACC",
    "ARI",
    "NMI",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Rare_Precision",
    "Rare_Recall",
    "Rare_F1",
]


def require_paths():
    paths = [
        os.path.join(FULL_DIR, "per_dataset.csv"),
        os.path.join(FULL_DIR, "summary.csv"),
        os.path.join(RARE_DIR, "per_run.csv"),
        os.path.join(RARE_DIR, "summary.csv"),
        HIG500_FULL,
        os.path.join(HIG500_RARE_DIR, "per_run.csv"),
        os.path.join(HIG500_RARE_DIR, "manifest.json"),
    ]
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))
    if os.path.exists(OUT_DIR):
        raise RuntimeError("Refusing to overwrite existing comparison directory: {}".format(OUT_DIR))


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value, digits=4):
    if pd.isna(value):
        return "NA"
    return "{:.{digits}f}".format(float(value), digits=digits)


def compact_table(frame, columns, max_rows=None):
    work = frame[columns].copy()
    if max_rows is not None:
        work = work.head(max_rows)
    lines = [
        "| " + " | ".join(str(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in work.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                rendered = "NA"
            elif isinstance(value, (float, np.floating)):
                rendered = "{:.4f}".format(float(value))
            else:
                rendered = str(value)
            values.append(rendered.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def bootstrap_ci(values, seed=20260724, n_bootstrap=10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.RandomState(seed)
    sample = rng.choice(values, size=(n_bootstrap, values.size), replace=True)
    means = sample.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compare_full():
    current = pd.read_csv(os.path.join(FULL_DIR, "per_dataset.csv"))
    current = current[current["method"].eq("archer_final")].copy()
    reference = pd.read_csv(HIG500_FULL).copy()
    if current["dataset"].nunique() != 16 or len(current) != 16:
        raise ValueError("Current full rerun is not exactly 16 final rows")
    if reference["dataset"].nunique() != 16 or len(reference) != 16:
        raise ValueError("HIG500 full reference is not exactly 16 rows")
    if set(current["dataset"]) != set(reference["dataset"]):
        raise ValueError("Current/reference full dataset sets differ")
    missing = [col for col in FULL_METRICS if col not in current or col not in reference]
    if missing:
        raise ValueError("Missing full comparison metric(s): {}".format(missing))

    keep_current = ["dataset", "method_version", "runtime_seconds"] + FULL_METRICS
    keep_reference = [
        "dataset",
        "method_version",
        "runtime_seconds",
        "RF_HIG_TOPN",
        "AKNN_KMAX",
        "AKNN_PRUNE_SNN",
        "src",
    ] + FULL_METRICS
    merged = current[keep_current].merge(
        reference[keep_reference], on="dataset", how="outer", suffixes=("_current_core", "_hig500_reference")
    )
    merged["reference_status"] = np.where(
        merged["src"].eq("hig500"),
        "DESCRIPTIVE_NON_EQUIVALENT: per-dataset-tuned HIG500 reference",
        "NOT_COMPARABLE: merged reference row is HIG2000, not HIG500",
    )
    for metric in FULL_METRICS:
        merged["delta_current_minus_hig500_" + metric] = (
            merged[metric + "_current_core"] - merged[metric + "_hig500_reference"]
        )
    merged = merged.sort_values("dataset").reset_index(drop=True)
    merged.to_csv(os.path.join(OUT_DIR, "full16_current_core_vs_hig500_reference.csv"), index=False)

    summary_rows = []
    for metric in FULL_METRICS:
        delta = merged["delta_current_minus_hig500_" + metric].astype(float)
        summary_rows.append(
            {
                "metric": metric,
                "current_core_mean": float(merged[metric + "_current_core"].mean()),
                "hig500_reference_mean": float(merged[metric + "_hig500_reference"].mean()),
                "delta_current_minus_hig500_mean": float(delta.mean()),
                "delta_sd_across_16": float(delta.std(ddof=1)),
                "current_core_wins": int((delta > 1e-12).sum()),
                "ties": int((delta.abs() <= 1e-12).sum()),
                "current_core_losses": int((delta < -1e-12).sum()),
                "reference_status": "non-equivalent descriptive comparison",
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, "full16_metric_summary_vs_hig500_reference.csv"), index=False)

    audit = pd.DataFrame(
        [
            {
                "reference": "HIG500_full_table",
                "path": HIG500_FULL,
                "rows": int(len(reference)),
                "datasets": int(reference["dataset"].nunique()),
                "hig500_rows": int((reference["src"] == "hig500").sum()),
                "hig2000_rows": int((reference["src"] == "hig2000").sum()),
                "comparison_status": "NOT STRICTLY COMPARABLE",
                "reason": "5/16 rows are HIG2000 and the table uses per-dataset tuning; it is not a fixed HIG500 control.",
            }
        ]
    )
    audit.to_csv(os.path.join(OUT_DIR, "full16_reference_audit.csv"), index=False)
    return merged, summary, audit


def compare_rare():
    current = pd.read_csv(os.path.join(RARE_DIR, "per_run.csv"))
    current = current[current["method"].eq("ArCHeR-v3-full")].copy()
    reference = pd.read_csv(os.path.join(HIG500_RARE_DIR, "per_run.csv"))
    reference = reference[reference["method"].eq("ArCHeR-v3-full")].copy()
    keys = ["dataset", "setting", "rare_count", "rep"]
    expected = 2 * 2 * 10 * 3
    if len(current) != expected or len(reference[reference["dataset"].isin(current["dataset"].unique())]) != expected:
        raise ValueError("Rare rerun/reference coverage is not 120 matched rows")
    old_subset = reference[reference["dataset"].isin(current["dataset"].unique())].copy()
    merged = current.merge(
        old_subset,
        on=keys,
        how="outer",
        suffixes=("_current_core", "_hig500_reference"),
        indicator=True,
    )
    if len(merged) != expected or not merged["_merge"].eq("both").all():
        raise ValueError("Rare current/reference keys are not exactly paired")
    if not np.allclose(
        merged["sample_seed_current_core"].astype(int),
        merged["sample_seed_hig500_reference"].astype(int),
    ):
        raise ValueError("Rare current/reference sample seeds differ")
    merged["reference_status"] = "PAIRED_DESCRIPTIVE: prior HIG500 rare run; core config differs"
    for metric in RARE_METRICS:
        merged["delta_current_minus_hig500_" + metric] = (
            merged[metric + "_current_core"] - merged[metric + "_hig500_reference"]
        )
    merged = merged.drop(columns=["_merge"]).sort_values(keys).reset_index(drop=True)
    merged.to_csv(os.path.join(OUT_DIR, "rare_current_core_vs_hig500_paired_per_run.csv"), index=False)

    overall_rows = []
    for metric in RARE_METRICS:
        delta = merged["delta_current_minus_hig500_" + metric].astype(float)
        low, high = bootstrap_ci(delta.values, seed=20260724 + RARE_METRICS.index(metric))
        overall_rows.append(
            {
                "metric": metric,
                "n_paired_runs": int(len(delta)),
                "current_core_mean": float(merged[metric + "_current_core"].mean()),
                "hig500_reference_mean": float(merged[metric + "_hig500_reference"].mean()),
                "delta_current_minus_hig500_mean": float(delta.mean()),
                "delta_sd": float(delta.std(ddof=1)),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "current_core_wins": int((delta > 1e-12).sum()),
                "ties": int((delta.abs() <= 1e-12).sum()),
                "current_core_losses": int((delta < -1e-12).sum()),
                "reference_status": "paired descriptive; not isolated to one parameter change",
            }
        )
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(os.path.join(OUT_DIR, "rare_metric_summary_vs_hig500.csv"), index=False)

    by_scenario = (
        merged.groupby(["dataset", "setting", "rare_count"], as_index=False)
        .agg(
            n_reps=("rep", "count"),
            current_Rare_Recall=("Rare_Recall_current_core", "mean"),
            hig500_Rare_Recall=("Rare_Recall_hig500_reference", "mean"),
            current_Rare_F1=("Rare_F1_current_core", "mean"),
            hig500_Rare_F1=("Rare_F1_hig500_reference", "mean"),
            delta_Rare_Recall=("delta_current_minus_hig500_Rare_Recall", "mean"),
            delta_Rare_F1=("delta_current_minus_hig500_Rare_F1", "mean"),
        )
        .sort_values(["dataset", "setting", "rare_count"])
    )
    by_scenario.to_csv(os.path.join(OUT_DIR, "rare_by_scenario_vs_hig500.csv"), index=False)

    return merged, overall, by_scenario


def write_markdown(full_merged, full_summary, full_audit, rare_merged, rare_overall, rare_scenario):
    current_full_manifest = read_json(os.path.join(FULL_DIR, "manifest.json"))
    current_rare_manifest = read_json(os.path.join(RARE_DIR, "manifest.json"))
    old_rare_manifest = read_json(os.path.join(HIG500_RARE_DIR, "manifest.json"))
    params = current_full_manifest.get("effective_core_config", {})
    lines = [
        "# ArCHeR core rerun vs current HIG500 references (2026-07-24)",
        "",
        "## Material Passport",
        "",
        "- Type: reproducibility rerun and paired descriptive comparison",
        "- Status: VERIFIED for run completeness; comparison warnings are retained below",
        "- Current core tag: `archer_core_20260724_k20_p1of15_exact_hig500`",
        "- Generated: `{}`".format(datetime.now().isoformat(timespec="seconds")),
        "",
        "## Effective current core configuration",
        "",
        "| Item | Value |",
        "|---|---|",
        "| aKNNO Kmax | {} |".format(params.get("AKNN_KMAX")),
        "| aKNNO prune | {} |".format(params.get("AKNN_PRUNE_SNN")),
        "| RF-HIG budget | {} |".format(params.get("RF_HIG_TOPN")),
        "| Graph embedding cap | {} PCs |".format(params.get("AKNN_EMBEDDING_DIMS")),
        "| Neighbor backend | `{}` |".format(params.get("NEIGHBOR_BACKEND")),
        "| HNSW | `{}` |".format(params.get("HNSW_ENABLED")),
        "| Delta selection | author last-rapid-change point + connectivity-safe fallback |",
        "| Feature view | HVG union HIG, capped DPC+kNN view fusion |",
        "",
        "## Run completeness",
        "",
        "- Full benchmark: 16/16 datasets, two method output trees, per-dataset CSV/JSON and paired deltas.",
        "- Rare benchmark: 120/120 rows = 2 datasets × 2 settings × 10 counts (2–20) × 3 reps.",
        "- Rare current/reference keys: 120/120 paired; sample seeds identical.",
        "",
        "## Full 16-dataset comparison",
        "",
        "The historical full-table reference is not a strict fixed HIG500 control: it contains 11 `hig500` rows and 5 `hig2000` rows, and all rows are per-dataset-tuned. Therefore these deltas are descriptive only; they must not be interpreted as the isolated effect of Kmax/prune/delta changes.",
        "",
        "### Metric means",
        "",
        compact_table(
            full_summary,
            ["metric", "current_core_mean", "hig500_reference_mean", "delta_current_minus_hig500_mean", "current_core_wins", "ties", "current_core_losses"],
        ),
        "",
        "### Per-dataset deltas",
        "",
        compact_table(
            full_merged,
            ["dataset", "src", "reference_status"] + ["delta_current_minus_hig500_" + metric for metric in FULL_METRICS],
        ),
        "",
        "Reference audit:",
        "",
        compact_table(full_audit, list(full_audit.columns)),
        "",
        "## Rare scenario comparison",
        "",
        "The prior HIG500 rare table is directly paired by dataset/setting/count/rep and uses the same sample seeds. It is still not a one-parameter ablation: its manifest records the prior core defaults (`AKNN_KMAX=12`, `AKNN_PRUNE_SNN=0.05`) and an older source hash. The paired deltas therefore summarize the combined change from that prior HIG500 run to the current core.",
        "",
        "### Overall paired means",
        "",
        compact_table(
            rare_overall,
            ["metric", "n_paired_runs", "current_core_mean", "hig500_reference_mean", "delta_current_minus_hig500_mean", "bootstrap_ci95_low", "bootstrap_ci95_high", "current_core_wins", "ties", "current_core_losses"],
        ),
        "",
        "### Scenario means (Rare Recall and Rare F1)",
        "",
        compact_table(rare_scenario, list(rare_scenario.columns)),
        "",
        "## Refined conclusion",
        "",
        "- The new fixed core completed without missing runs under exact neighbors and the requested label-free configuration.",
        "- On the rare stress test, the current core mean Rare Recall is {:.4f} versus {:.4f} for the prior HIG500 reference (Δ={:+.4f}); mean Rare F1 is {:.4f} versus {:.4f} (Δ={:+.4f}).".format(
            rare_overall.loc[rare_overall.metric.eq("Rare_Recall"), "current_core_mean"].iloc[0],
            rare_overall.loc[rare_overall.metric.eq("Rare_Recall"), "hig500_reference_mean"].iloc[0],
            rare_overall.loc[rare_overall.metric.eq("Rare_Recall"), "delta_current_minus_hig500_mean"].iloc[0],
            rare_overall.loc[rare_overall.metric.eq("Rare_F1"), "current_core_mean"].iloc[0],
            rare_overall.loc[rare_overall.metric.eq("Rare_F1"), "hig500_reference_mean"].iloc[0],
            rare_overall.loc[rare_overall.metric.eq("Rare_F1"), "delta_current_minus_hig500_mean"].iloc[0],
        ),
        "- The full benchmark current core mean Macro-F1 is {:.4f}; its comparison to the mixed per-dataset-tuned HIG500/HIG2000 table is descriptive and not a fair fixed-config claim.".format(
            float(current_full_manifest.get("effective_core_config", {}).get("RF_HIG_TOPN", np.nan))
            if False else float(pd.read_csv(os.path.join(FULL_DIR, "summary.csv")).query("method == 'archer_final'")["Macro_F1_mean"].iloc[0])
        ),
        "- Existing baseline and figure files were not overwritten; all new result and comparison artifacts are under the two new tag directories.",
        "",
        "## Source paths",
        "",
        "- Current full rerun: `{}`".format(FULL_DIR),
        "- Current rare rerun: `{}`".format(RARE_DIR),
        "- Historical full reference: `{}`".format(HIG500_FULL),
        "- Historical rare HIG500 reference: `{}`".format(HIG500_RARE_DIR),
        "- Full manifest: `{}`".format(os.path.join(FULL_DIR, "manifest.json")),
        "- Rare manifest: `{}`".format(os.path.join(RARE_DIR, "manifest.json")),
        "- Complete paired rare per-run table: `rare_current_core_vs_hig500_paired_per_run.csv`",
        "",
        "## Comparison artifacts",
        "",
        "| Artifact | Meaning |",
        "|---|---|",
        "| `full16_current_core_vs_hig500_reference.csv` | all 16 rows, all common metrics, status flags and deltas |",
        "| `full16_metric_summary_vs_hig500_reference.csv` | full metric means, wins/ties/losses |",
        "| `full16_reference_audit.csv` | HIG500/HIG2000 mixture audit |",
        "| `rare_current_core_vs_hig500_paired_per_run.csv` | all 120 matched rare rows and all metric deltas |",
        "| `rare_metric_summary_vs_hig500.csv` | paired rare metric means and bootstrap intervals |",
        "| `rare_by_scenario_vs_hig500.csv` | 40 scenario means for Rare Recall/F1 |",
        "",
    ]
    with open(os.path.join(OUT_DIR, "comparison_report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    per_run_columns = [
        "dataset",
        "setting",
        "rare_count",
        "rep",
        "Rare_Recall_current_core",
        "Rare_Recall_hig500_reference",
        "delta_current_minus_hig500_Rare_Recall",
        "Rare_F1_current_core",
        "Rare_F1_hig500_reference",
        "delta_current_minus_hig500_Rare_F1",
    ]
    with open(os.path.join(OUT_DIR, "rare_per_run_comparison.md"), "w", encoding="utf-8") as handle:
        handle.write("# Complete paired rare-run comparison\n\n")
        handle.write(compact_table(rare_merged, per_run_columns))
        handle.write("\n")

    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(
            "# Comparison artifacts\n\n"
            "Generated by `experiments/scripts/compare_archer_core_to_hig500_20260724.py`. "
            "See `comparison_report.md` for configuration audit, complete conclusions, "
            "and explicit comparability warnings.\n"
        )


def main():
    require_paths()
    os.makedirs(OUT_DIR, exist_ok=False)
    full_merged, full_summary, full_audit = compare_full()
    rare_merged, rare_overall, rare_scenario = compare_rare()
    write_markdown(full_merged, full_summary, full_audit, rare_merged, rare_overall, rare_scenario)
    print("Wrote comparison artifacts to {}".format(OUT_DIR))


if __name__ == "__main__":
    main()
