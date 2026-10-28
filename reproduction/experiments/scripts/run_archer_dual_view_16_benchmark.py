"""Fair full-data benchmark for original versus scale-adaptive ArCHeR.

Every dataset is evaluated on all cells under one frozen, label-free protocol.
The original ArCHeR and the scale-adaptive final candidate are rerun from the
same h5ad sources; historical per-dataset-tuned scores are not read.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime

import igraph as ig
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import method_archer as original  # noqa: E402
import method_archer_final as final  # noqa: E402
import run_archer_v2_evaluation as evaluation  # noqa: E402


BASE_DIR = r"D:\sc_dataset"
DEFAULT_DATASET_SOURCE = os.path.join(
    BASE_DIR,
    "experiments",
    "metrics",
    "full_metrics_16datasets_by_dataset.csv",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "experiments",
    "metrics",
    "archer_v31_scale_adaptive_16_benchmark_20260717",
)
METHOD_ORDER = ["archer_original", "archer_final"]
METHOD_LABELS = {
    "archer_original": "ArCHeR original",
    "archer_final": final.METHOD_NAME,
}
RARE_THRESHOLD = 0.02


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def primitive_config(config_class):
    result = {}
    for key in dir(config_class):
        if key.startswith("_"):
            continue
        value = getattr(config_class, key)
        if isinstance(value, (str, int, float, bool, list, tuple)):
            result[key] = json_safe(value)
    return result


def stable_seed(base_seed, dataset):
    return evaluation.stable_scenario_seed(base_seed, "full", dataset)


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    ig.set_random_number_generator(random.Random(int(seed)))


def predicted_small_cluster_mask(labels, threshold):
    labels = np.asarray(labels)
    values, counts = np.unique(labels, return_counts=True)
    small = values[counts / float(len(labels)) <= float(threshold)]
    return np.isin(labels, small)


def binary_metrics(true_mask, pred_mask, prefix):
    true_mask = np.asarray(true_mask, dtype=bool)
    pred_mask = np.asarray(pred_mask, dtype=bool)
    tp = int(np.sum(true_mask & pred_mask))
    fp = int(np.sum(~true_mask & pred_mask))
    fn = int(np.sum(true_mask & ~pred_mask))
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        prefix + "_Precision": float(precision),
        prefix + "_Recall": float(recall),
        prefix + "_F1": float(f1),
        prefix + "_TP": tp,
        prefix + "_FP": fp,
        prefix + "_FN": fn,
        prefix + "_Predicted_Count": int(np.sum(pred_mask)),
    }


def rare_oracle_metrics(true_labels, pred_labels, threshold):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    values, counts = np.unique(true_labels, return_counts=True)
    rare_values = values[counts / float(len(true_labels)) <= float(threshold)]
    rows = [
        evaluation.best_cluster_for_class(true_labels, pred_labels, value)
        for value in rare_values
    ]
    if not rows:
        return {
            "Rare_Oracle_Class_Count": 0,
            "Rare_Oracle_Macro_Precision": np.nan,
            "Rare_Oracle_Macro_Recall": np.nan,
            "Rare_Oracle_Macro_F1": np.nan,
        }
    return {
        "Rare_Oracle_Class_Count": int(len(rows)),
        "Rare_Oracle_Macro_Precision": float(np.mean([row["precision"] for row in rows])),
        "Rare_Oracle_Macro_Recall": float(np.mean([row["recall"] for row in rows])),
        "Rare_Oracle_Macro_F1": float(np.mean([row["f1"] for row in rows])),
    }


def run_one(method, model_input, already_log, seed):
    diagnostics = {}
    if method == "archer_original":
        if bool(original.Config.USE_TARGET_CLUSTER_COUNT):
            raise RuntimeError("Original target-cluster-count mode must be disabled")
        prediction = original.run_archer_pipeline(
            model_input,
            true_labels=None,
            already_log=already_log,
        )
    elif method == "archer_final":
        prediction, diagnostics = final.run_archer_final_pipeline(
            model_input,
            true_labels=None,
            already_log=already_log,
            random_seed=seed,
            return_details=True,
        )
    else:
        raise ValueError("Unknown method: {}".format(method))
    return np.asarray(prediction, dtype=int), diagnostics


def output_paths(output_dir, method, dataset):
    root = os.path.join(output_dir, method)
    os.makedirs(root, exist_ok=True)
    return {
        "metric": os.path.join(root, dataset + ".json"),
        "prediction": os.path.join(root, dataset + ".csv.gz"),
    }


def evaluate_one(output_dir, dataset, method, base_seed):
    paths = output_paths(output_dir, method, dataset)
    if os.path.exists(paths["metric"]) and os.path.exists(paths["prediction"]):
        print("[skip] {} | {}".format(dataset, method), flush=True)
        with open(paths["metric"], "r", encoding="utf-8") as handle:
            return json.load(handle)

    counts, true_labels, cell_ids, already_log, source_path = evaluation.load_full_dataset(
        dataset, source="h5ad"
    )
    model_input = evaluation.strip_observation_metadata(counts)
    del counts
    gc.collect()
    seed = stable_seed(base_seed, dataset)
    set_seed(seed)
    started = time.time()
    prediction, method_diagnostics = run_one(
        method, model_input, already_log, seed
    )
    runtime = float(time.time() - started)
    del model_input
    gc.collect()

    if len(prediction) != len(true_labels):
        raise ValueError(
            "Prediction length mismatch for {} / {}: {} != {}".format(
                dataset, method, len(prediction), len(true_labels)
            )
        )

    global_metrics, _ = evaluation.compute_metrics(true_labels, prediction)
    true_rare = np.zeros(len(true_labels), dtype=bool)
    values, counts = np.unique(true_labels, return_counts=True)
    rare_values = values[counts / float(len(true_labels)) <= RARE_THRESHOLD]
    if len(rare_values):
        true_rare = np.isin(true_labels, rare_values)
    size_rule = predicted_small_cluster_mask(prediction, RARE_THRESHOLD)

    metrics = dict(global_metrics)
    metrics.update({
        "dataset": dataset,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "seed": int(seed),
        "runtime_seconds": runtime,
        "source_path": source_path,
        "rare_threshold": RARE_THRESHOLD,
        "rare_true_class_count": int(len(rare_values)),
        "rare_true_cell_count": int(np.sum(true_rare)),
        "rare_true_cell_fraction": float(np.mean(true_rare)),
    })
    metrics.update(binary_metrics(true_rare, size_rule, "SizeRule_Rare"))
    metrics.update(rare_oracle_metrics(true_labels, prediction, RARE_THRESHOLD))
    for key in [
        "method_version",
        "scale_route",
        "scale_route_threshold",
        "effective_hig_mode",
        "init_embedding_mode",
        "init_neighbor_backend",
        "init_neighbor_backend_fallback",
        "aknno_embedding_mode",
        "aknno_neighbor_backend",
        "aknno_neighbor_backend_fallback",
    ]:
        value = method_diagnostics.get(key)
        if value is not None:
            metrics[key] = value

    prediction_frame = pd.DataFrame({
        "cell_id": np.asarray(cell_ids).astype(str),
        "true_label": np.asarray(true_labels),
        "pred_label": prediction,
        "true_rare": true_rare.astype(np.uint8),
        "size_rule_rare_pred": size_rule.astype(np.uint8),
    })
    prediction_frame.to_csv(paths["prediction"], index=False, compression="gzip")
    with open(paths["metric"], "w", encoding="utf-8") as handle:
        json.dump(json_safe(metrics), handle, indent=2, ensure_ascii=False)
    print(
        "[done] {} | {} | ARI={:.4f} MacroF1={:.4f} RareOracleF1={} | {:.1f}s".format(
            dataset,
            method,
            metrics["ARI"],
            metrics["Macro_F1"],
            (
                "{:.4f}".format(metrics["Rare_Oracle_Macro_F1"])
                if pd.notna(metrics["Rare_Oracle_Macro_F1"])
                else "NA"
            ),
            runtime,
        ),
        flush=True,
    )
    return metrics


def build_outputs(output_dir, datasets, base_seed):
    rows = []
    for dataset in datasets:
        for method in METHOD_ORDER:
            path = output_paths(output_dir, method, dataset)["metric"]
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as handle:
                rows.append(json.load(handle))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No metrics were produced")
    frame["method"] = pd.Categorical(frame["method"], categories=METHOD_ORDER, ordered=True)
    frame = frame.sort_values(["dataset", "method"]).reset_index(drop=True)
    frame["method"] = frame["method"].astype(str)
    frame.to_csv(os.path.join(output_dir, "per_dataset.csv"), index=False)

    numeric = [
        "ACC",
        "ARI",
        "NMI",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "SizeRule_Rare_Precision",
        "SizeRule_Rare_Recall",
        "SizeRule_Rare_F1",
        "Rare_Oracle_Macro_Precision",
        "Rare_Oracle_Macro_Recall",
        "Rare_Oracle_Macro_F1",
        "runtime_seconds",
    ]
    summary_rows = []
    for method in METHOD_ORDER:
        subset = frame[frame["method"] == method]
        row = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_datasets": int(len(subset)),
            "rare_eligible_datasets": int(np.sum(subset["rare_true_class_count"] > 0)),
        }
        for metric in numeric:
            row[metric + "_mean"] = float(pd.to_numeric(subset[metric], errors="coerce").mean())
            row[metric + "_sd"] = float(pd.to_numeric(subset[metric], errors="coerce").std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    wide = frame.pivot(index="dataset", columns="method", values=numeric)
    delta_rows = []
    for dataset in datasets:
        if dataset not in wide.index:
            continue
        row = {"dataset": dataset}
        for metric in numeric:
            original_value = wide.loc[dataset, (metric, "archer_original")]
            final_value = wide.loc[dataset, (metric, "archer_final")]
            row["final_minus_original_" + metric] = float(final_value - original_value)
        delta_rows.append(row)
    pd.DataFrame(delta_rows).to_csv(os.path.join(output_dir, "paired_delta.csv"), index=False)
    return frame, summary


def write_manifest(output_dir, datasets, base_seed, source_path):
    input_sources = {
        dataset: os.path.join(
            BASE_DIR, "data", "benchmarks", dataset, "sc_cluster_fixed_result.h5ad"
        )
        for dataset in datasets
    }
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": "full 16-dataset fair comparison: original versus scale-adaptive ArCHeR v3.1",
        "datasets": list(datasets),
        "methods": METHOD_ORDER,
        "method_labels": METHOD_LABELS,
        "method_sources": {
            "archer_original": os.path.join(SCRIPT_DIR, "method_archer.py"),
            "archer_final": os.path.join(SCRIPT_DIR, "method_archer_final.py"),
        },
        "method_sha256": {
            "archer_original": sha256_file(os.path.join(SCRIPT_DIR, "method_archer.py")),
            "archer_final": sha256_file(os.path.join(SCRIPT_DIR, "method_archer_final.py")),
        },
        "implementation_source": os.path.join(SCRIPT_DIR, "method_archer_configurable.py"),
        "implementation_sha256": sha256_file(
            os.path.join(SCRIPT_DIR, "method_archer_configurable.py")
        ),
        "final_method_version": final.METHOD_VERSION,
        "runner_source": os.path.abspath(__file__),
        "runner_sha256": sha256_file(os.path.abspath(__file__)),
        "source_dataset_list": source_path,
        "input_sources": input_sources,
        "input_sha256": {name: sha256_file(path) for name, path in input_sources.items()},
        "seed": int(base_seed),
        "seed_scheme": "sha256_dataset_full_v1",
        "rare_threshold": RARE_THRESHOLD,
        "scale_route_rule": "DPC+kNN union-HIG when n_cells<=10000; kNN-only HIG otherwise",
        "neighbor_backend_rule": "auto: HNSW for large data when installed, otherwise PyNNDescent; exact/KDTree fallback",
        "dual_view_hig_rule": "top RF-HIG from DPC pseudo-labels union top RF-HIG from kNN-Louvain pseudo-labels on the dual-view route",
        "dual_view_hig_topn_per_view": 2000,
        "label_firewall": "true labels are removed from AnnData.obs before both model calls and used only after prediction",
        "historical_metric_exclusion": "no Figure A or per-dataset-tuned ArCHeR metrics are read",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "original_config": primitive_config(original.Config),
        },
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(json_safe(manifest), handle, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Full 16-dataset scale-adaptive ArCHeR benchmark")
    parser.add_argument("--datasets-from", default=DEFAULT_DATASET_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    datasets = pd.read_csv(args.datasets_from)["dataset"].drop_duplicates().tolist()
    if len(datasets) != 16:
        raise ValueError("Expected 16 datasets, observed {}".format(len(datasets)))
    os.makedirs(args.output_dir, exist_ok=True)
    write_manifest(args.output_dir, datasets, args.seed, args.datasets_from)

    for dataset in datasets:
        print("\n=== {} ===".format(dataset), flush=True)
        for method in METHOD_ORDER:
            evaluate_one(args.output_dir, dataset, method, args.seed)

    frame, summary = build_outputs(args.output_dir, datasets, args.seed)
    print("\nCompleted {} dataset-method runs".format(len(frame)), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
