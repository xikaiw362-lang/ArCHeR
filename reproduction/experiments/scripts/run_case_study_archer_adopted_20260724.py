"""Create an independent, label-free case-study report for the adopted ArCHeR run.

This script deliberately consumes an archived prediction file instead of rerunning
with a mixed configuration. Reference annotations are joined only after prediction
to describe within-predicted-cluster composition; they are never used for tuning,
selection, or cluster-ID remapping.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def sort_cluster_ids(values: pd.Series) -> list[str]:
    unique = [str(v) for v in values.dropna().unique()]
    try:
        return sorted(unique, key=lambda item: (0, int(item)))
    except ValueError:
        return sorted(unique)


def composition_table(joined: pd.DataFrame, category: str) -> pd.DataFrame:
    work = joined[["archer_cluster", category]].copy()
    work["archer_cluster"] = work["archer_cluster"].astype(str)
    work[category] = work[category].fillna("<missing>").astype(str)
    counts = (
        work.groupby(["archer_cluster", category], dropna=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    totals = counts.groupby("archer_cluster")["n_cells"].transform("sum")
    counts["within_predicted_cluster_fraction"] = counts["n_cells"] / totals
    counts["rank_within_predicted_cluster"] = counts.groupby("archer_cluster")["n_cells"].rank(
        method="first", ascending=False
    ).astype(int)
    order = {cluster_id: rank for rank, cluster_id in enumerate(sort_cluster_ids(work["archer_cluster"]))}
    counts["_cluster_order"] = counts["archer_cluster"].map(order)
    return counts.sort_values(
        ["_cluster_order", "n_cells", category], ascending=[True, False, True]
    ).drop(columns="_cluster_order")


def partition_metrics(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    merged = old[["barcode", "archer_cluster"]].rename(columns={"archer_cluster": "old_cluster"}).merge(
        new[["barcode", "archer_cluster"]].rename(columns={"archer_cluster": "new_cluster"}),
        on="barcode",
        how="inner",
        validate="one_to_one",
    )
    result = {
        "n_cells_compared": int(len(merged)),
        "old_only_cells": int(len(old) - len(merged)),
        "new_only_cells": int(len(new) - len(merged)),
        "old_n_clusters": int(old["archer_cluster"].nunique()),
        "new_n_clusters": int(new["archer_cluster"].nunique()),
    }
    if len(merged) == len(old) == len(new):
        try:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

            result["adjusted_rand_index_old_vs_new"] = float(
                adjusted_rand_score(merged["old_cluster"], merged["new_cluster"])
            )
            result["normalized_mutual_info_old_vs_new"] = float(
                normalized_mutual_info_score(merged["old_cluster"], merged["new_cluster"])
            )
        except Exception as exc:  # pragma: no cover - only used when sklearn is unavailable
            result["partition_metric_error"] = repr(exc)
    else:
        result["partition_metric_error"] = "cell sets differ; label-free partition metrics omitted"
    return result


def top_composition_rows(table: pd.DataFrame, category: str, cluster_id: str, limit: int = 3) -> str:
    rows = table[table["archer_cluster"] == cluster_id].head(limit)
    return "; ".join(
        f"{row[category]} ({int(row['n_cells'])}, {row['within_predicted_cluster_fraction']:.1%})"
        for _, row in rows.iterrows()
    )


def build_report(
    output_dir: Path,
    manifest: dict,
    cluster_summary: pd.DataFrame,
    big_composition: pd.DataFrame,
    partition: dict,
) -> None:
    ari = partition.get("adjusted_rand_index_old_vs_new", float("nan"))
    nmi = partition.get("normalized_mutual_info_old_vs_new", float("nan"))
    lines = [
        "# D_R_00002 ArCHeR case study (independent rerun record)",
        "",
        "## Scope and adopted result",
        "",
        "The adopted case-study result is the archived fixed HIG500 ArCHeR run. The new core was not adopted because the benchmark/rare comparison did not show a stable advantage, with rare-scene performance lower in the paired analysis.",
        "",
        "All cluster IDs below are the prediction IDs emitted by ArCHeR. The annotation columns are used only post hoc to report composition inside each predicted cluster. No reference label or marker was used for parameter selection, cluster selection, or ID remapping.",
        "",
        f"- Dataset: <code>{manifest['dataset']}</code>",
        f"- Cells: <code>{manifest['n_cells']}</code>; predicted clusters: <code>{manifest['n_predicted_clusters']}</code>",
        f"- Adopted HIG500 parameters: <code>RF_HIG_TOPN={manifest['adopted_config']['RF_HIG_TOPN']}</code>, <code>AKNN_KMAX={manifest['adopted_config']['AKNN_KMAX']}</code>, <code>AKNN_PRUNE_SNN={manifest['adopted_config']['AKNN_PRUNE_SNN']}</code>",
        f"- Effective neighbor backend in archived diagnostics: <code>{manifest['adopted_config']['effective_neighbor_backend']}</code>; HNSW: <code>{manifest['adopted_config']['HNSW_ENABLED']}</code>",
        f"- Labels used in method: <code>{manifest['labels_used_in_method']}</code>",
        "",
        "## Label-free partition audit",
        "",
        f"The archived adopted run has <code>{manifest['n_predicted_clusters']}</code> predicted clusters and <code>{manifest['n_rare_cells_by_size_rule']}</code> cells flagged by the stored size rule. For the archived-vs-new-core partition audit, ARI={ari:.4f} and NMI={nmi:.4f}; these are stability descriptors only, not tuning criteria.",
        "",
        "## Predicted cluster composition",
        "",
        "| predicted cluster ID | cells | fraction | size-rule rare cells | top Big_Cell_type composition |",
        "|---:|---:|---:|---:|---|",
    ]
    for _, row in cluster_summary.iterrows():
        cluster_id = str(row["archer_cluster"])
        lines.append(
            f"| {cluster_id} | {int(row['n_cells'])} | {row['fraction']:.1%} | {int(row['n_rare_cells_by_size_rule'])} | {top_composition_rows(big_composition, 'Big_Cell_type', cluster_id)} |"
        )
    lines.extend(
        [
            "",
            "The complete composition tables are saved as CSV, including both Big_Cell_type and the finer Cell_type annotation. They should be read as post-hoc composition summaries of the predicted IDs, not as a corrected or remapped partition.",
            "",
            "## Output files",
            "",
        ]
    )
    for name in [
        "case_study_manifest.json",
        "adopted_predictions_hig500.csv",
        "adopted_cluster_summary.csv",
        "adopted_cluster_composition_big_cell_type.csv",
        "adopted_cluster_composition_cell_type.csv",
        "partition_stability_archived_hig500_vs_new_core.json",
    ]:
        lines.append(f"- <code>{output_dir / name}</code>")
    (output_dir / "case_study_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "case_study_datasets" / "D_R_00002",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "case_study_datasets"
        / "D_R_00002"
        / "case_reruns"
        / "archer_hig500_adopted_20260724",
    )
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = dataset_dir / "exported"
    old_prediction_path = exported / "archer_predictions_hig500_repaired.csv"
    old_diag_path = exported / "archer_hig500_repaired_diagnostics.json"
    new_prediction_path = exported / "archer_predictions_hig500_aknno_archecore_v1.csv"
    new_diag_path = exported / "archer_hig500_aknno_archecore_v1_diagnostics.json"
    meta_path = exported / "meta.csv"
    required = [old_prediction_path, old_diag_path, new_prediction_path, new_diag_path, meta_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required case-study inputs are missing:\n" + "\n".join(missing))

    old = pd.read_csv(old_prediction_path)
    new = pd.read_csv(new_prediction_path)
    meta = pd.read_csv(meta_path)
    for frame, name in [(old, "archived prediction"), (new, "new-core prediction"), (meta, "metadata")]:
        if "barcode" not in frame.columns:
            raise ValueError(f"{name} has no barcode column")
        if frame["barcode"].duplicated().any():
            raise ValueError(f"{name} contains duplicate barcodes")
    if set(old["barcode"]) != set(meta["barcode"]):
        raise ValueError("Archived prediction and metadata barcode sets differ")

    old["archer_cluster"] = old["archer_cluster"].astype(str)
    new["archer_cluster"] = new["archer_cluster"].astype(str)
    joined = old.merge(meta, on="barcode", how="left", validate="one_to_one")
    if joined["Big_Cell_type"].isna().any() or joined["Cell_type"].isna().any():
        raise ValueError("Metadata join produced missing annotation values")

    cluster_summary = (
        joined.groupby("archer_cluster", sort=False)
        .agg(
            n_cells=("barcode", "size"),
            n_rare_cells_by_size_rule=("is_rare_cluster_by_size_rule", "sum"),
            mean_rare_cell_score=("rare_cell_score", "mean"),
        )
        .reset_index()
    )
    cluster_summary["fraction"] = cluster_summary["n_cells"] / len(joined)
    cluster_summary["n_rare_cells_by_size_rule"] = cluster_summary["n_rare_cells_by_size_rule"].astype(int)
    cluster_order = {cluster_id: rank for rank, cluster_id in enumerate(sort_cluster_ids(cluster_summary["archer_cluster"]))}
    cluster_summary["_cluster_order"] = cluster_summary["archer_cluster"].map(cluster_order)
    cluster_summary = cluster_summary.sort_values("_cluster_order").drop(columns="_cluster_order")
    big_composition = composition_table(joined, "Big_Cell_type")
    cell_composition = composition_table(joined, "Cell_type")
    partition = partition_metrics(old, new)

    old.to_csv(output_dir / "adopted_predictions_hig500.csv", index=False)
    cluster_summary.to_csv(output_dir / "adopted_cluster_summary.csv", index=False)
    big_composition.to_csv(output_dir / "adopted_cluster_composition_big_cell_type.csv", index=False)
    cell_composition.to_csv(output_dir / "adopted_cluster_composition_cell_type.csv", index=False)
    (output_dir / "partition_stability_archived_hig500_vs_new_core.json").write_text(
        json.dumps(json_safe(partition), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    old_diag = read_json(old_diag_path)
    new_diag = read_json(new_diag_path)
    old_params = old_diag.get("fixed_params", {})
    old_runtime = old_diag.get("runtime_details", {})
    new_params = new_diag.get("fixed_params", {})
    new_runtime = new_diag.get("runtime_details", {})
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": old_diag.get("dataset", "D_R_00002 zebrafish AGM"),
        "n_cells": int(len(old)),
        "n_genes": int(old_diag.get("n_genes", 0)),
        "n_predicted_clusters": int(old["archer_cluster"].nunique()),
        "n_rare_cells_by_size_rule": int(old["is_rare_cluster_by_size_rule"].sum()),
        "adopted_version": "archived_hig500_repaired",
        "adopted_config": {
            "RF_HIG_TOPN": old_params.get("RF_HIG_TOPN"),
            "AKNN_KMAX": old_params.get("AKNN_KMAX"),
            "AKNN_PRUNE_SNN": old_params.get("AKNN_PRUNE_SNN"),
            "NEIGHBOR_BACKEND_requested": old_params.get("NEIGHBOR_BACKEND"),
            "effective_neighbor_backend": old_runtime.get("aknno_neighbor_backend"),
            "HNSW_ENABLED": old_params.get("HNSW_ENABLED"),
            "AKNNO_EMBEDDING_N_COMPONENTS": old_runtime.get("aknno_embedding_n_components"),
            "delta_selection_rule": old_runtime.get("aknno_delta_selection_rule"),
            "optimal_delta": old_runtime.get("aknno_optimal_delta"),
        },
        "labels_used_in_method": bool(old_diag.get("labels_used_in_method", True)),
        "reference_annotation_use": "posthoc composition only; no tuning, selection, or cluster-ID remapping",
        "source_prediction": str(old_prediction_path),
        "source_diagnostics": str(old_diag_path),
        "new_core_candidate_prediction": str(new_prediction_path),
        "new_core_candidate_diagnostics": str(new_diag_path),
        "new_core_candidate_config": {
            "RF_HIG_TOPN": new_params.get("RF_HIG_TOPN"),
            "AKNN_KMAX": new_params.get("AKNN_KMAX"),
            "AKNN_PRUNE_SNN": new_params.get("AKNN_PRUNE_SNN"),
            "NEIGHBOR_BACKEND_requested": new_params.get("NEIGHBOR_BACKEND"),
            "effective_neighbor_backend": new_runtime.get("aknno_neighbor_backend"),
            "HNSW_ENABLED": new_params.get("HNSW_ENABLED"),
            "AKNNO_EMBEDDING_N_COMPONENTS": new_runtime.get("aknno_embedding_n_components"),
            "delta_selection_rule": new_runtime.get("aknno_delta_selection_rule"),
            "optimal_delta": new_runtime.get("aknno_optimal_delta"),
        },
        "partition_stability": partition,
        "interpretation_guard": [
            "Predicted cluster IDs are retained exactly as emitted by ArCHeR.",
            "Reference labels are not used to select parameters or clusters.",
            "No Hungarian mapping or other reference-driven ID remapping is applied.",
            "No marker-driven parameter selection is applied.",
        ],
    }
    (output_dir / "case_study_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    build_report(output_dir, manifest, cluster_summary, big_composition, partition)

    print(json.dumps(json_safe({"output_dir": str(output_dir), "manifest": manifest}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
