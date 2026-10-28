"""Unified, resumable evaluation runner for the frozen ArCHeR v2 pipeline.

The runner has an explicit label firewall: true labels are removed from AnnData
metadata before the model is called and are used only after prediction.  It can
evaluate the 16 full benchmark datasets, the two controlled rare-cell stress
test datasets, or both.  No ablations are included in this runner.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime

import anndata as ad
import igraph
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from method_archer_v2 import get_default_params, run_archer_v2_pipeline


BASE_DIR = r"D:\sc_dataset"
BENCHMARK_DIR = os.path.join(BASE_DIR, "data", "benchmarks")
DEFAULT_DATASET_SOURCE = os.path.join(
    BASE_DIR,
    "experiments",
    "metrics",
    "full_metrics_16datasets_by_dataset.csv",
)
SEED_SCHEME = "sha256_dataset_scenario_v1"


def stable_scenario_seed(base_seed, *parts):
    """Derive an order-independent 31-bit seed from scenario identity."""
    payload = "|".join([str(int(base_seed))] + [str(part) for part in parts])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(value % (2 ** 31 - 1))


DEFAULT_OUTPUT_ROOT = os.path.join(
    BASE_DIR,
    "experiments",
    "metrics",
    "archer_v2",
)
DEFAULT_RARE_DATASETS = [
    "Quake_10x_Limb_Muscle",
    "Quake_Smart-seq2_Heart",
]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _source_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def prepare_run_directory(output_root, run_name, seed):
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    method_source = os.path.join(SCRIPT_DIR, "method_archer_v2.py")
    runner_source = os.path.abspath(__file__)
    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(seed),
        "seed_scheme": SEED_SCHEME,
        "method_source": method_source,
        "method_sha256": _source_hash(method_source),
        "runner_source": runner_source,
        "runner_sha256": _source_hash(runner_source),
        "params": get_default_params(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "anndata": ad.__version__,
            "igraph": igraph.__version__,
        },
        "label_firewall": (
            "All observation metadata are removed before model invocation; "
            "true labels are used only for post-prediction metrics."
        ),
    }
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("method_sha256") != manifest["method_sha256"]:
            raise RuntimeError(
                "Refusing to mix outputs from different method source hashes in {}"
                .format(run_dir)
            )
        if existing.get("runner_sha256") != manifest["runner_sha256"]:
            raise RuntimeError(
                "Refusing to mix outputs from different runner source hashes in {}"
                .format(run_dir)
            )
    else:
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(manifest), handle, indent=2, ensure_ascii=False)
    return run_dir


def infer_already_log(adata):
    try:
        if "log1p" in adata.uns:
            return True
    except Exception:
        pass
    if "total_counts" in adata.obs:
        try:
            total_counts = adata.obs["total_counts"]
            if float(total_counts.max()) > 30000 or float(total_counts.median()) > 2000:
                return False
        except Exception:
            pass
    rows = min(1000, adata.n_obs)
    columns = min(200, adata.n_vars)
    block = adata.X[:rows, :columns]
    if hasattr(block, "toarray"):
        block = block.toarray()
    block = np.asarray(block, dtype=np.float64)
    finite = block[np.isfinite(block)]
    if len(finite) == 0:
        return False
    positive = finite[finite > 0]
    if len(positive) == 0:
        return False
    integer_fraction = float(np.mean(np.abs(positive - np.round(positive)) < 1e-6))
    if integer_fraction > 0.98 and float(np.max(positive)) > 10:
        return False
    return float(np.max(positive)) <= 30


def _read_label_table(label_path, fallback_cell_ids=None):
    labels = pd.read_csv(label_path, header=None)
    if labels.shape[1] >= 2:
        labels = labels.iloc[:, :2]
        labels.columns = ["cell_id", "true_label"]
        labels["cell_id"] = labels["cell_id"].astype(str)
    else:
        labels.columns = ["true_label"]
        if fallback_cell_ids is None or len(fallback_cell_ids) != len(labels):
            raise ValueError("One-column label file cannot be aligned safely: {}".format(label_path))
        labels["cell_id"] = np.asarray(fallback_cell_ids).astype(str)
    labels["true_label"] = pd.to_numeric(labels["true_label"], errors="coerce")
    labels = labels.dropna(subset=["true_label"])
    labels["true_label"] = labels["true_label"].astype(int)
    return labels.set_index("cell_id")


def _labels_from_adata_or_file(adata, dataset_dir):
    if "true_label" in adata.obs:
        values = pd.to_numeric(adata.obs["true_label"], errors="coerce")
        if values.notna().all():
            return values.astype(int).to_numpy()
    label_path = os.path.join(dataset_dir, "label.csv.gz")
    if not os.path.exists(label_path):
        raise FileNotFoundError("Missing labels for {}".format(dataset_dir))
    table = _read_label_table(label_path, adata.obs_names)
    cells = pd.Index(adata.obs_names.astype(str))
    missing = cells.difference(table.index)
    extra = table.index.difference(cells)
    if len(missing) or len(extra):
        raise ValueError(
            "h5ad/label mismatch for {} (missing={}, extra={})".format(
                dataset_dir,
                len(missing),
                len(extra),
            )
        )
    return table.loc[cells, "true_label"].to_numpy(dtype=int)


def load_h5ad_dataset(dataset_name, backed=False):
    dataset_dir = os.path.join(BENCHMARK_DIR, dataset_name)
    path = os.path.join(dataset_dir, "sc_cluster_fixed_result.h5ad")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    adata = ad.read_h5ad(path, backed="r" if backed else None)
    true_labels = _labels_from_adata_or_file(adata, dataset_dir)
    already_log = infer_already_log(adata)
    return adata, true_labels, np.asarray(adata.obs_names.astype(str)), already_log, path


def load_csv_dataset(dataset_name):
    dataset_dir = os.path.join(BENCHMARK_DIR, dataset_name)
    matrix_path = os.path.join(dataset_dir, "count_matrix.csv.gz")
    label_path = os.path.join(dataset_dir, "label.csv.gz")
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(matrix_path)
    columns = pd.read_csv(matrix_path, nrows=0, engine="python").columns
    dtype_map = {column: np.float32 for column in columns[1:]}
    counts = pd.read_csv(
        matrix_path,
        index_col=0,
        dtype=dtype_map,
        engine="python",
    ).T
    counts.index = counts.index.astype(str)
    counts.columns = counts.columns.astype(str)
    if counts.values.dtype != np.float32:
        counts = counts.astype(np.float32)
    labels = _read_label_table(label_path, counts.index)
    missing = counts.index.difference(labels.index)
    extra = labels.index.difference(counts.index)
    if len(missing) or len(extra):
        raise ValueError(
            "count/label mismatch for {} (missing={}, extra={})".format(
                dataset_name,
                len(missing),
                len(extra),
            )
        )
    true_labels = labels.loc[counts.index, "true_label"].to_numpy(dtype=int)
    return counts, true_labels, counts.index.to_numpy(), False, matrix_path


def load_full_dataset(dataset_name, source="auto"):
    count_path = os.path.join(BENCHMARK_DIR, dataset_name, "count_matrix.csv.gz")
    if source == "csv" or (source == "auto" and os.path.exists(count_path)):
        return load_csv_dataset(dataset_name)
    return load_h5ad_dataset(dataset_name, backed=False)


def strip_observation_metadata(adata):
    if getattr(adata, "isbacked", False):
        adata = adata.to_memory()
    else:
        adata = adata.copy()
    adata.obs = pd.DataFrame(index=adata.obs_names.copy())
    return adata


def encode_labels(labels):
    encoded, uniques = pd.factorize(pd.Series(labels).astype(str), sort=True)
    if np.any(encoded < 0):
        raise ValueError("Labels contain missing values.")
    return encoded.astype(int), uniques.astype(str).to_numpy()


def hungarian_map(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    true_ids = np.unique(y_true)
    pred_ids = np.unique(y_pred)
    overlap = np.zeros((len(pred_ids), len(true_ids)), dtype=np.int64)
    for pred_index, pred_id in enumerate(pred_ids):
        mask = y_pred == pred_id
        values, counts = np.unique(y_true[mask], return_counts=True)
        true_positions = {value: index for index, value in enumerate(true_ids)}
        for value, count in zip(values, counts):
            overlap[pred_index, true_positions[value]] = int(count)
    row_ind, col_ind = linear_sum_assignment(-overlap)
    mapping = {int(pred_ids[row]): int(true_ids[col]) for row, col in zip(row_ind, col_ind)}
    for pred_id in pred_ids:
        if int(pred_id) not in mapping:
            mask = y_pred == pred_id
            values, counts = np.unique(y_true[mask], return_counts=True)
            mapping[int(pred_id)] = int(values[np.argmax(counts)])
    mapped = np.asarray([mapping[int(value)] for value in y_pred], dtype=int)
    return mapped


def best_cluster_for_class(y_true, y_pred, target_label):
    true_positive = np.asarray(y_true) == target_label
    best = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "cluster_size": 0,
        "pred_label": None,
    }
    for pred_label in np.unique(y_pred):
        predicted = np.asarray(y_pred) == pred_label
        tp = int(np.sum(predicted & true_positive))
        fp = int(np.sum(predicted & ~true_positive))
        fn = int(np.sum(~predicted & true_positive))
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if (f1, recall, precision) > (best["f1"], best["recall"], best["precision"]):
            best = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "cluster_size": int(np.sum(predicted)),
                "pred_label": int(pred_label),
            }
    return best


def compute_metrics(y_true_raw, y_pred, target_rare_label=None):
    y_true, true_names = encode_labels(y_true_raw)
    y_pred, _ = encode_labels(y_pred)
    mapped = hungarian_map(y_true, y_pred)
    labels = np.unique(y_true)
    purity_numerator = 0
    for pred_label in np.unique(y_pred):
        _, counts = np.unique(y_true[y_pred == pred_label], return_counts=True)
        purity_numerator += int(np.max(counts))
    metrics = {
        "ACC": float(np.mean(mapped == y_true)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "Macro_Precision": float(
            precision_score(y_true, mapped, labels=labels, average="macro", zero_division=0)
        ),
        "Macro_Recall": float(
            recall_score(y_true, mapped, labels=labels, average="macro", zero_division=0)
        ),
        "Macro_F1": float(
            f1_score(y_true, mapped, labels=labels, average="macro", zero_division=0)
        ),
        "Purity": float(purity_numerator / max(float(len(y_true)), 1.0)),
        "n_cells": int(len(y_true)),
        "n_true_clusters": int(len(labels)),
        "n_pred_clusters": int(len(np.unique(y_pred))),
    }
    class_counts = np.bincount(y_true)
    smallest_label = int(np.where(class_counts == np.min(class_counts))[0][0])
    smallest = best_cluster_for_class(y_true, y_pred, smallest_label)
    metrics.update({
        "Smallest_Class_Size": int(class_counts[smallest_label]),
        "Smallest_Class_Precision": smallest["precision"],
        "Smallest_Class_Recall": smallest["recall"],
        "Smallest_Class_F1": smallest["f1"],
        "Smallest_Predicted_Cluster_Size": smallest["cluster_size"],
    })

    if target_rare_label is not None:
        raw_strings = pd.Series(y_true_raw).astype(str).to_numpy()
        matches = np.where(true_names == str(target_rare_label))[0]
        if len(matches) != 1:
            raise ValueError("Rare label could not be encoded uniquely: {}".format(target_rare_label))
        rare_encoded = int(matches[0])
        rare_true = y_true == rare_encoded
        rare_pred = mapped == rare_encoded
        tp = int(np.sum(rare_true & rare_pred))
        fp = int(np.sum(~rare_true & rare_pred))
        fn = int(np.sum(rare_true & ~rare_pred))
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        rare_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        best = best_cluster_for_class(y_true, y_pred, rare_encoded)
        metrics.update({
            "Rare_Precision": float(precision),
            "Rare_Recall": float(recall),
            "Rare_F1": float(rare_f1),
            "Rare_TP": tp,
            "Rare_FP": fp,
            "Rare_FN": fn,
            "Best_Rare_Precision": best["precision"],
            "Best_Rare_Recall": best["recall"],
            "Best_Rare_F1": best["f1"],
            "Best_Rare_Cluster_Size": best["cluster_size"],
        })
    class_rows = []
    for label in labels:
        true_mask = y_true == label
        pred_mask = mapped == label
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        class_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_rows.append({
            "true_label": str(true_names[int(label)]),
            "class_size": int(np.sum(true_mask)),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(class_f1),
        })
    return metrics, class_rows


def extract_model_diagnostics(details):
    selected_index = int(details.get("selected_round_index", len(details["rounds"]) - 1))
    selected = details["rounds"][selected_index]
    topology = selected.get("topology_anchors", {})
    cleanup = selected.get("cleanup", {})
    clustering = selected.get("clustering", {})
    view_graphs = selected.get("view_graphs", {})
    topology_records = topology.get("top_candidates", [])
    selected_purified = [
        record
        for record in topology_records
        if record.get("selected", False) and record.get("purified", False)
    ]
    return {
        "initial_n_clusters": int(details.get("initial_n_clusters", 0)),
        "final_n_clusters": int(details.get("final_n_clusters", 0)),
        "selected_round_index": selected_index,
        "round_cluster_counts": details.get("round_cluster_counts", []),
        "round_partition_ari": details.get("round_partition_ari", []),
        "round_acceptance": details.get("round_acceptance", []),
        "round_rejection_reasons": details.get("round_rejection_reasons", []),
        "round_supported_new_splits": details.get(
            "round_supported_new_splits", []
        ),
        "round_maximum_allowed_cluster_count": details.get(
            "round_maximum_allowed_cluster_count", []
        ),
        "dpc_total_centers": int(details.get("dpc", {}).get("dpc_total_center_count", 0)),
        "dpc_local_centers": int(details.get("dpc", {}).get("dpc_local_center_count", 0)),
        "topology_candidate_count": int(topology.get("candidate_count", 0)),
        "topology_selected_count": int(topology.get("selected_count", 0)),
        "topology_selected_sizes": topology.get("selected_sizes", []),
        "topology_selection_stage": topology.get("selection_stage", "unknown"),
        "topology_maintenance_count": int(
            topology.get("selected_maintenance_count", 0)
        ),
        "topology_maintenance_sizes": topology.get(
            "selected_maintenance_sizes", []
        ),
        "topology_discovery_count": int(
            topology.get("selected_discovery_count", 0)
        ),
        "topology_discovery_sizes": topology.get(
            "selected_discovery_sizes", []
        ),
        "topology_nested_expansion_count": int(
            topology.get("selected_nested_expansion_count", 0)
        ),
        "topology_nested_expansion_sizes": topology.get(
            "selected_nested_expansion_sizes", []
        ),
        "topology_purified_count": int(len(selected_purified)),
        "topology_bridge_release_count": int(sum(
            "bridge_release" in record.get("purification_modes", [])
            for record in selected_purified
        )),
        "graph_micro_sizes": selected.get("graph_micro_sizes", []),
        "local_refine_sizes": selected.get("local_refine_accepted_sizes", []),
        "cleanup_merge_count": int(cleanup.get("merge_count", 0)),
        "consensus_resolution": float(clustering.get("selected_resolution", np.nan)),
        "mean_adaptive_k_global": float(
            view_graphs.get("global", {}).get("mean_adaptive_k", np.nan)
        ),
        "mean_adaptive_k_discriminative": float(
            view_graphs.get("discriminative", {}).get("mean_adaptive_k", np.nan)
        ),
        "mean_adaptive_k_rare": float(
            view_graphs.get("rare", {}).get("mean_adaptive_k", np.nan)
        ),
    }


def resolve_full_datasets(dataset_source, requested):
    if requested:
        return requested
    if os.path.exists(dataset_source):
        frame = pd.read_csv(dataset_source)
        return frame["dataset"].drop_duplicates().astype(str).tolist()
    return sorted(
        name
        for name in os.listdir(BENCHMARK_DIR)
        if os.path.exists(os.path.join(BENCHMARK_DIR, name, "sc_cluster_fixed_result.h5ad"))
        and name != "Young"
    )


def run_full_benchmark(args, run_dir):
    datasets = resolve_full_datasets(args.datasets_from, args.dataset)
    prediction_dir = os.path.join(run_dir, "full_predictions")
    metric_dir = os.path.join(run_dir, "full_dataset_metrics")
    class_dir = os.path.join(run_dir, "full_class_metrics")
    diagnostic_dir = os.path.join(run_dir, "full_diagnostics")
    for path in [prediction_dir, metric_dir, class_dir, diagnostic_dir]:
        os.makedirs(path, exist_ok=True)

    for dataset_name in datasets:
        metric_path = os.path.join(metric_dir, dataset_name + ".json")
        prediction_path = os.path.join(prediction_dir, dataset_name + ".csv.gz")
        class_path = os.path.join(class_dir, dataset_name + ".csv")
        diagnostic_path = os.path.join(diagnostic_dir, dataset_name + ".json")
        if all(os.path.exists(path) for path in [metric_path, prediction_path, class_path, diagnostic_path]):
            print("[full] skip completed {}".format(dataset_name), flush=True)
            continue
        print("\n[full] {}".format(dataset_name), flush=True)
        counts, true_labels, cell_ids, already_log, source_path = load_full_dataset(
            dataset_name,
            source=args.full_source,
        )
        if isinstance(counts, ad.AnnData):
            model_input = strip_observation_metadata(counts)
        else:
            model_input = counts
        seed = stable_scenario_seed(args.seed, "full", dataset_name)
        random.seed(seed)
        np.random.seed(seed)
        started = time.time()
        pred, details = run_archer_v2_pipeline(
            model_input,
            true_labels=None,
            already_log=already_log,
            return_details=True,
            random_seed=seed,
        )
        runtime = float(time.time() - started)
        metrics, class_rows = compute_metrics(true_labels, pred)
        metrics.update({
            "dataset": dataset_name,
            "method": "ArCHeR_v2",
            "seed": seed,
            "runtime_seconds": runtime,
            "source_path": source_path,
            "already_log": bool(already_log),
        })
        diagnostics = extract_model_diagnostics(details)
        metrics.update({
            "topology_selected_count": diagnostics["topology_selected_count"],
            "topology_maintenance_count": diagnostics["topology_maintenance_count"],
            "topology_discovery_count": diagnostics["topology_discovery_count"],
            "topology_nested_expansion_count": diagnostics[
                "topology_nested_expansion_count"
            ],
            "topology_purified_count": diagnostics["topology_purified_count"],
            "cleanup_merge_count": diagnostics["cleanup_merge_count"],
        })
        pd.DataFrame({
            "dataset": dataset_name,
            "cell_id": np.asarray(cell_ids).astype(str),
            "true_label": np.asarray(true_labels),
            "pred_label": np.asarray(pred, dtype=int),
        }).to_csv(prediction_path, index=False, compression="gzip")
        class_frame = pd.DataFrame(class_rows)
        class_frame.insert(0, "dataset", dataset_name)
        class_frame.to_csv(class_path, index=False)
        with open(metric_path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(metrics), handle, indent=2, ensure_ascii=False)
        with open(diagnostic_path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(diagnostics), handle, indent=2, ensure_ascii=False)
        print(
            "[full] {} | ARI={:.4f} Macro_F1={:.4f} smallest_F1={:.4f} | {:.1f}s".format(
                dataset_name,
                metrics["ARI"],
                metrics["Macro_F1"],
                metrics["Smallest_Class_F1"],
                runtime,
            ),
            flush=True,
        )

    metric_rows = []
    for dataset_name in datasets:
        path = os.path.join(metric_dir, dataset_name + ".json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                metric_rows.append(json.load(handle))
    if metric_rows:
        frame = pd.DataFrame(metric_rows)
        frame.to_csv(os.path.join(run_dir, "full_metrics_by_dataset.csv"), index=False)
        numeric = frame.select_dtypes(include=[np.number]).columns
        summary = frame[numeric].agg(["mean", "std", "median"]).T.reset_index()
        summary.columns = ["metric", "mean", "std", "median"]
        summary.to_csv(os.path.join(run_dir, "full_metrics_summary.csv"), index=False)


def pick_rare_scenario_labels(labels):
    labels = np.asarray(labels)
    values, counts = np.unique(labels, return_counts=True)
    if len(values) < 4:
        return None
    descending = np.argsort(-counts)
    abundant = [values[descending[0]], values[descending[1]]]
    rare = []
    for index in np.argsort(counts):
        value = values[index]
        if value not in abundant:
            rare.append(value)
        if len(rare) == 2:
            break
    if len(rare) < 2:
        return None
    return abundant[0], abundant[1], rare[0], rare[1]


def sample_rare_scenario(
    labels,
    abundant_a,
    abundant_b,
    rare_label,
    abundant_n,
    rare_n,
    seed,
):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    groups = [
        np.where(labels == abundant_a)[0],
        np.where(labels == abundant_b)[0],
        np.where(labels == rare_label)[0],
    ]
    requested = [int(abundant_n), int(abundant_n), int(rare_n)]
    for indices, count in zip(groups, requested):
        if len(indices) < count:
            raise ValueError(
                "Requested {} cells from a class containing only {}".format(count, len(indices))
            )
    selected = np.concatenate([
        rng.choice(indices, size=count, replace=False)
        for indices, count in zip(groups, requested)
    ])
    return selected[rng.permutation(len(selected))]


def _append_csv_row(path, row):
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
    )


def run_rare_benchmark(args, run_dir):
    output_path = os.path.join(run_dir, "rare_runs.csv")
    diagnostics_path = os.path.join(run_dir, "rare_diagnostics.jsonl")
    existing = set()
    if os.path.exists(output_path):
        frame = pd.read_csv(output_path)
        existing = set(
            zip(
                frame["dataset"].astype(str),
                frame["setting"].astype(str),
                frame["rare_count"].astype(int),
                frame["rep"].astype(int),
            )
        )
    datasets = args.dataset or DEFAULT_RARE_DATASETS
    rare_counts = args.rare_counts or list(range(2, 21, 2))
    for dataset_name in datasets:
        adata, labels, _, already_log, _ = load_h5ad_dataset(dataset_name, backed=True)
        scenario_labels = pick_rare_scenario_labels(labels)
        if scenario_labels is None:
            print("[rare] skip {}: fewer than four usable classes".format(dataset_name), flush=True)
            continue
        abundant_a, abundant_b, rare_challenging, rare_easier = scenario_labels
        settings = [
            ("Setting1_Challenging", rare_challenging),
            ("Setting2_Easier", rare_easier),
        ]
        for setting_name, rare_label in settings:
            for rare_n in rare_counts:
                for rep in range(1, int(args.reps) + 1):
                    key = (dataset_name, setting_name, int(rare_n), int(rep))
                    if key in existing:
                        continue
                    sample_seed = stable_scenario_seed(
                        args.seed,
                        "rare",
                        dataset_name,
                        setting_name,
                        int(rare_n),
                        int(rep),
                    )
                    selected = sample_rare_scenario(
                        labels,
                        abundant_a,
                        abundant_b,
                        rare_label,
                        args.abundant_n,
                        rare_n,
                        sample_seed,
                    )
                    sub = adata[selected].to_memory()
                    true_labels = np.asarray(labels)[selected]
                    model_input = strip_observation_metadata(sub)
                    if model_input.X.dtype != np.float32:
                        model_input.X = model_input.X.astype(np.float32)
                    print(
                        "[rare] {} | {} | rare={:2d} | rep={}".format(
                            dataset_name,
                            setting_name,
                            int(rare_n),
                            rep,
                        ),
                        flush=True,
                    )
                    random.seed(sample_seed)
                    np.random.seed(sample_seed)
                    started = time.time()
                    pred, details = run_archer_v2_pipeline(
                        model_input,
                        true_labels=None,
                        already_log=already_log,
                        return_details=True,
                        random_seed=sample_seed,
                    )
                    runtime = float(time.time() - started)
                    metrics, _ = compute_metrics(
                        true_labels,
                        pred,
                        target_rare_label=rare_label,
                    )
                    diagnostics = extract_model_diagnostics(details)
                    row = {
                        "dataset": dataset_name,
                        "setting": setting_name,
                        "rare_label": str(rare_label),
                        "rare_count": int(rare_n),
                        "rep": int(rep),
                        "seed": int(sample_seed),
                        "method": "ArCHeR_v2",
                        "runtime_seconds": runtime,
                        "already_log": bool(already_log),
                    }
                    row.update(metrics)
                    row.update({
                        "topology_selected_count": diagnostics["topology_selected_count"],
                        "topology_selected_sizes": json.dumps(
                            diagnostics["topology_selected_sizes"]
                        ),
                        "topology_maintenance_count": diagnostics[
                            "topology_maintenance_count"
                        ],
                        "topology_maintenance_sizes": json.dumps(
                            diagnostics["topology_maintenance_sizes"]
                        ),
                        "topology_discovery_count": diagnostics[
                            "topology_discovery_count"
                        ],
                        "topology_discovery_sizes": json.dumps(
                            diagnostics["topology_discovery_sizes"]
                        ),
                        "topology_nested_expansion_count": diagnostics[
                            "topology_nested_expansion_count"
                        ],
                        "topology_nested_expansion_sizes": json.dumps(
                            diagnostics["topology_nested_expansion_sizes"]
                        ),
                        "topology_purified_count": diagnostics[
                            "topology_purified_count"
                        ],
                        "topology_bridge_release_count": diagnostics[
                            "topology_bridge_release_count"
                        ],
                        "local_refine_sizes": json.dumps(
                            diagnostics["local_refine_sizes"]
                        ),
                        "cleanup_merge_count": diagnostics["cleanup_merge_count"],
                        "selected_round_index": diagnostics["selected_round_index"],
                    })
                    _append_csv_row(output_path, row)
                    with open(diagnostics_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(_json_safe({
                            "key": key,
                            "diagnostics": diagnostics,
                        }), ensure_ascii=False) + "\n")
                    existing.add(key)
                    print(
                        "[rare] Rare_P={:.3f} Rare_R={:.3f} Rare_F1={:.3f} "
                        "Best_F1={:.3f} | {:.1f}s".format(
                            metrics["Rare_Precision"],
                            metrics["Rare_Recall"],
                            metrics["Rare_F1"],
                            metrics["Best_Rare_F1"],
                            runtime,
                        ),
                        flush=True,
                    )
        if getattr(adata, "isbacked", False):
            adata.file.close()

    if os.path.exists(output_path):
        frame = pd.read_csv(output_path)
        metric_columns = [
            "ARI",
            "Macro_F1",
            "Rare_Precision",
            "Rare_Recall",
            "Rare_F1",
            "Best_Rare_Precision",
            "Best_Rare_Recall",
            "Best_Rare_F1",
            "runtime_seconds",
        ]
        summary = (
            frame.groupby(["dataset", "setting", "rare_count"], as_index=False)[metric_columns]
            .agg(["mean", "std"])
        )
        summary.columns = [
            "_".join([item for item in column if item])
            for column in summary.columns.to_flat_index()
        ]
        summary.to_csv(os.path.join(run_dir, "rare_summary_by_scenario.csv"), index=False)
        overall = frame[metric_columns].agg(["mean", "std", "median"]).T.reset_index()
        overall.columns = ["metric", "mean", "std", "median"]
        overall.to_csv(os.path.join(run_dir, "rare_summary_overall.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["full", "rare", "all"])
    parser.add_argument("--run-name", default="frozen_candidate")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--datasets-from", default=DEFAULT_DATASET_SOURCE)
    parser.add_argument("--full-source", choices=["auto", "csv", "h5ad"], default="h5ad")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--abundant-n", type=int, default=200)
    parser.add_argument("--rare-counts", nargs="+", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = prepare_run_directory(args.output_root, args.run_name, args.seed)
    if args.mode in ["full", "all"]:
        run_full_benchmark(args, run_dir)
    if args.mode in ["rare", "all"]:
        run_rare_benchmark(args, run_dir)
    print("\nSaved evaluation outputs to {}".format(run_dir), flush=True)


if __name__ == "__main__":
    main()
