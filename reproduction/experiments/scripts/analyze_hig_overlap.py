"""Measure overlap between DPC-view and kNN-view RF-HIG selections.

The sampling design mirrors ``run_archer_ablation_multimetric_fair.py``.  The
analysis stops after the two RF gene-importance rankings, so it does not run
the final aKNNO clustering stage and never exposes ground-truth labels to the
feature-selection pipeline.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime

import igraph as ig
import numpy as np
import pandas as pd
import scanpy as sc


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from method_archer import (  # noqa: E402
    highly_variable_genes_inplace,
    log1p_inplace,
    normalize_total_inplace,
)
from method_archer_configurable import (  # noqa: E402
    compute_rf_hig_importance_with_params,
    run_dpc_with_params,
    run_knn_louvain_init_with_params,
)
from run_archer_ablation_multimetric_fair import (  # noqa: E402
    DEFAULT_DATASETS,
    infer_already_log,
    pick_labels,
    resolve_dataset_path,
    sample_cells,
)


DEFAULT_OUT_DIR = os.path.join(
    r"D:\sc_dataset", "experiments", "metrics", "hig_overlap_heldout7"
)


def frozen_default_params():
    """Return the exact configurable-backend defaults used by union-HIG."""
    return {
        "HVG_MIN_MEAN": 0.0125,
        "HVG_MAX_MEAN": 3,
        "HVG_MIN_DISP": 0.5,
        "HVG_TOPN": 2000,
        "DPC_DC_PERCENT": 8,
        "DPC_GAMMA_TOPK": 8,
        "DPC_PCA_N_COMPONENTS": 40,
        "DPC_KNN_K": 200,
        "DPC_FEATURE_MODE": "hvg",
        "DPC_CENTER_MODE": "global_gamma",
        "DPC_LOCAL_CENTER_TOPK": 3,
        "DPC_LOCAL_NEIGHBOR_K": 10,
        "DPC_LOCAL_SUPPORT_K": 5,
        "DPC_LOCAL_MUTUAL_MIN": 1,
        "DPC_LOCAL_RHO_MIN_RATIO": 1.0,
        "DPC_LOCAL_PEAK_QUANTILE": 0.75,
        "DPC_LOCAL_SCORE_QUANTILE": 0.90,
        "DPC_LOCAL_DEDUP_K": 5,
        "DPC_RARE_AWARE_TOPN": 500,
        "DPC_RARE_MIN_CELLS": 2,
        "DPC_RARE_MAX_PREVALENCE": 0.25,
        "COLLECT_QUALITY_DIAGNOSTICS": False,
        "QUALITY_SILHOUETTE_SAMPLE_SIZE": 2000,
        "RF_N_ESTIMATORS": 100,
        "RF_HIG_TOPN": 2000,
        "RF_PCA_DIMS": 50,
        "RF_OOB_DIAGNOSTICS": False,
        "INIT_KNN_K": 20,
        "INIT_LOUVAIN_RES": 1.0,
        "POST_KNN_K": 20,
        "POST_LOUVAIN_RES": 1.0,
        "DBSCAN_MIN_SAMPLES_GRID": [2, 3, 5],
        "DBSCAN_EPS_QUANTILES": [80, 85, 90, 95],
        "DBSCAN_MAX_NOISE_FRAC": 0.50,
        "OPTICS_MIN_SAMPLES": 2,
        "OPTICS_XI": 0.05,
        "OPTICS_MIN_CLUSTER_SIZE": 2,
        "AKNN_KMAX": 12,
        "AKNN_PRUNE_SNN": 1 / 20,
        "AKNN_GRAPH_MODE": "default",
        "AKNN_PCA_DIMS": list(range(1, 31)),
        "AKNN_DELTA_RANGE": np.arange(-1.0, 0.1, 0.1),
        "AKNN_CUTOFF": 5,
        "RESOLUTION_BASE": 0.28,
        "RESOLUTION_SCALE": 0.22,
        "RESOLUTION_MIN": 0.18,
        "RESOLUTION_MAX": 1.2,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_case_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    ig.set_random_number_generator(random.Random(int(seed)))


def prepare_for_hig(adata, already_log, params):
    if not already_log:
        normalize_total_inplace(adata, target_sum=1e4)
        log1p_inplace(adata)
    highly_variable_genes_inplace(
        adata,
        min_mean=params["HVG_MIN_MEAN"],
        max_mean=params["HVG_MAX_MEAN"],
        min_disp=params["HVG_MIN_DISP"],
        n_top_genes=params["HVG_TOPN"],
    )
    return adata


def overlap_row(adata, params):
    dpc_view = run_dpc_with_params(adata.copy(), params)
    knn_view = run_knn_louvain_init_with_params(
        adata.copy(), params, target_n_clusters=None
    )
    dpc_importance, _ = compute_rf_hig_importance_with_params(
        adata, params, dpc_view.obs["initial_cluster"].values
    )
    knn_importance, _ = compute_rf_hig_importance_with_params(
        adata, params, knn_view.obs["initial_cluster"].values
    )

    topn = min(int(params["RF_HIG_TOPN"]), adata.n_vars)
    dpc_top = np.argsort(-dpc_importance)[:topn]
    knn_top = np.argsort(-knn_importance)[:topn]
    intersection_count = int(
        len(np.intersect1d(dpc_top, knn_top, assume_unique=True))
    )
    union_count = int(len(np.union1d(dpc_top, knn_top)))
    overlap_coefficient = intersection_count / max(topn, 1)
    return {
        "n_genes": int(adata.n_vars),
        "view_topn": int(topn),
        "intersection_count": intersection_count,
        "union_count": union_count,
        "dpc_unique_count": int(topn - intersection_count),
        "knn_unique_count": int(topn - intersection_count),
        "jaccard": float(intersection_count / max(union_count, 1)),
        "dpc_covered_by_knn": float(overlap_coefficient),
        "knn_covered_by_dpc": float(overlap_coefficient),
        "overlap_coefficient": float(overlap_coefficient),
        "dpc_n_clusters": int(dpc_view.obs["initial_cluster"].nunique()),
        "knn_n_clusters": int(knn_view.obs["initial_cluster"].nunique()),
    }


def summarize(frame, output_dir):
    metrics = [
        "view_topn",
        "intersection_count",
        "union_count",
        "dpc_unique_count",
        "knn_unique_count",
        "jaccard",
        "overlap_coefficient",
        "dpc_n_clusters",
        "knn_n_clusters",
    ]

    def aggregate(group_columns, filename):
        if group_columns:
            grouped = frame.groupby(group_columns, as_index=False)
            rows = []
            for keys, group in grouped:
                if not isinstance(keys, tuple):
                    keys = (keys,)
                row = dict(zip(group_columns, keys))
                row["n_runs"] = int(len(group))
                for metric in metrics:
                    row[metric + "_mean"] = float(group[metric].mean())
                    row[metric + "_sd"] = float(group[metric].std(ddof=1))
                    row[metric + "_median"] = float(group[metric].median())
                    row[metric + "_min"] = float(group[metric].min())
                    row[metric + "_max"] = float(group[metric].max())
                rows.append(row)
            result = pd.DataFrame(rows)
        else:
            row = {"n_runs": int(len(frame))}
            for metric in metrics:
                row[metric + "_mean"] = float(frame[metric].mean())
                row[metric + "_sd"] = float(frame[metric].std(ddof=1))
                row[metric + "_median"] = float(frame[metric].median())
                row[metric + "_min"] = float(frame[metric].min())
                row[metric + "_max"] = float(frame[metric].max())
            result = pd.DataFrame([row])
        result.to_csv(os.path.join(output_dir, filename), index=False)

    aggregate([], "summary_overall.csv")
    aggregate(["dataset"], "summary_by_dataset.csv")
    aggregate(["setting"], "summary_by_setting.csv")
    aggregate(["rare_count"], "summary_by_rare_count.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Measure DPC/kNN RF-HIG overlap under the fair benchmark sampling design"
    )
    parser.add_argument("--dataset", action="append", choices=DEFAULT_DATASETS)
    parser.add_argument("--rare-count", action="append", type=int)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--abundant-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    datasets = args.dataset or DEFAULT_DATASETS
    rare_counts = sorted(set(args.rare_count or [2, 8, 16]))
    params = frozen_default_params()
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, "per_run_overlap.csv")

    existing = set()
    if os.path.exists(output_csv):
        old = pd.read_csv(output_csv)
        existing = set(zip(old.dataset, old.setting, old.rare_count, old.rep))

    input_paths = {}
    for dataset in datasets:
        path = resolve_dataset_path(dataset)
        input_paths[dataset] = path
        print("\n=== {} ===".format(dataset), flush=True)
        backed = sc.read_h5ad(path, backed="r")
        labels = pd.to_numeric(
            backed.obs["true_label"], errors="coerce"
        ).astype(int).values
        picked = pick_labels(labels)
        if picked is None:
            print("Skip {}: fewer than three eligible classes".format(dataset), flush=True)
            continue
        abundant_a, abundant_b, rare_challenging, rare_easier = picked
        settings = [
            ("Setting1_Challenging", rare_challenging),
            ("Setting2_Easier", rare_easier),
        ]
        already_log = infer_already_log(backed)

        for setting, rare_label in settings:
            for rare_count in rare_counts:
                for rep in range(1, int(args.reps) + 1):
                    key = (dataset, setting, rare_count, rep)
                    if key in existing:
                        continue
                    sample_seed = int(args.seed) + rare_count * 100 + rep
                    selected = sample_cells(
                        labels,
                        abundant_a,
                        abundant_b,
                        rare_label,
                        args.abundant_n,
                        rare_count,
                        sample_seed,
                    )
                    subset = backed[selected].to_memory()
                    if subset.X.dtype != np.float32:
                        subset.X = subset.X.astype(np.float32)
                    if "true_label" in subset.obs:
                        del subset.obs["true_label"]

                    set_case_seed(sample_seed)
                    print(
                        "{} | {} | rare={} | rep={}".format(
                            dataset, setting, rare_count, rep
                        ),
                        flush=True,
                    )
                    prepared = prepare_for_hig(subset, already_log, params)
                    row = {
                        "dataset": dataset,
                        "setting": setting,
                        "rare_label": int(rare_label),
                        "rare_count": int(rare_count),
                        "rep": int(rep),
                        "sample_seed": int(sample_seed),
                        "n_cells": int(prepared.n_obs),
                    }
                    row.update(overlap_row(prepared, params))
                    pd.DataFrame([row]).to_csv(
                        output_csv,
                        mode="a",
                        header=not os.path.exists(output_csv),
                        index=False,
                    )
                    existing.add(key)

    frame = pd.read_csv(output_csv)
    expected = len(datasets) * 2 * len(rare_counts) * int(args.reps)
    if len(frame) != expected:
        raise ValueError(
            "Incomplete overlap table: expected {}, observed {}".format(
                expected, len(frame)
            )
        )
    summarize(frame, args.output_dir)

    source_paths = [
        os.path.abspath(__file__),
        os.path.join(SCRIPT_DIR, "method_archer.py"),
        os.path.join(SCRIPT_DIR, "method_archer_configurable.py"),
        os.path.join(SCRIPT_DIR, "run_archer_ablation_multimetric_fair.py"),
    ]
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis": "DPC-view versus kNN-view top RF-HIG overlap",
        "datasets": datasets,
        "rare_counts": rare_counts,
        "settings": ["Setting1_Challenging", "Setting2_Easier"],
        "reps": int(args.reps),
        "abundant_n": int(args.abundant_n),
        "base_seed": int(args.seed),
        "expected_runs": expected,
        "rf_hig_topn": int(params["RF_HIG_TOPN"]),
        "label_firewall": "true_label removed before preprocessing and pseudo-label/HIG computation",
        "stopping_point": "after DPC/kNN pseudo-label views and RF-HIG rankings; aKNNO not run",
        "input_sources": input_paths,
        "input_sha256": {name: sha256_file(path) for name, path in input_paths.items()},
        "source_sha256": {path: sha256_file(path) for path in source_paths},
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    overall = pd.read_csv(os.path.join(args.output_dir, "summary_overall.csv")).iloc[0]
    print("\nCompleted {} runs".format(len(frame)), flush=True)
    print(
        "mean intersection={:.1f}, mean union={:.1f}, mean Jaccard={:.4f}, mean coverage={:.4f}".format(
            overall["intersection_count_mean"],
            overall["union_count_mean"],
            overall["jaccard_mean"],
            overall["overlap_coefficient_mean"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
