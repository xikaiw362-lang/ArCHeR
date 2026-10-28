"""Fair, paired ablation benchmark for the original ArCHeR method.

This runner does not modify ``method_archer.py``.  Every method is evaluated on
the same sampled cells with the same seed and the same configuration within a
dataset.  The ``paper_tuned`` profile reproduces the two-dataset ablation
design used in the original paper, including its per-dataset tuned settings.
Ground-truth labels are withheld from the model call and are used only after
predictions have been frozen.

The historical ``Rare_ACC`` quantity is rare-class recall, not accuracy.  This
runner therefore reports rare precision, recall, and F1 explicitly and keeps
``Rare_ACC_legacy`` only as an audited alias of ``Rare_Recall``.

The so-called "w/o DPC" branch in the legacy implementation needs pseudo-labels
for RF-HIG and consequently replaces DPC with a fixed kNN-Louvain initializer.
It is named accordingly here so the ablation semantics are unambiguous.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
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

from method_archer import Config, run_archer_pipeline  # noqa: E402
from method_archer_configurable import (  # noqa: E402
    run_archer_pipeline_with_params,
)
import method_archer_final as final  # noqa: E402


BASE_DIR = r"D:\sc_dataset"
DEFAULT_OUT_DIR = os.path.join(
    BASE_DIR,
    "experiments",
    "metrics",
    "archer_ablation_fair_20260715",
)
DEFAULT_DATASETS = [
    "Adam",
    "Bach",
    "Klein",
    "Muraro",
    "Plasschaert",
    "Pollen",
    "Quake_10x_Bladder",
    "Quake_10x_Limb_Muscle",
    "Quake_10x_Spleen",
    "Quake_Smart-seq2_Diaphragm",
    "Quake_Smart-seq2_Heart",
    "Quake_Smart-seq2_Limb_Muscle",
    "Quake_Smart-seq2_Lung",
    "Quake_Smart-seq2_Trachea",
    "Romanov",
    "Tosches_turtle",
]
DEFAULT_RARE_COUNTS = list(range(2, 21, 2))
PAPER_TUNED_DATASETS = [
    "Quake_10x_Limb_Muscle",
    "Quake_Smart-seq2_Heart",
]
PAPER_TUNED_CONFIGS = {
    # Exact per-dataset overrides used by the historical tuned ablation
    # runner.  All four variants share the selected dataset's overrides.
    "Quake_10x_Limb_Muscle": {
        "AKNN_KMAX": 15,
        "AKNN_PRUNE_SNN": 1 / 15,
        "DPC_GAMMA_TOPK": 12,
    },
    "Quake_Smart-seq2_Heart": {
        "AKNN_KMAX": 10,
        "AKNN_PRUNE_SNN": 1 / 24,
        "RESOLUTION_BASE": 0.22,
    },
}
KEY_COLUMNS = ["dataset", "setting", "rare_count", "rep"]
METHODS = [
    {
        "method": "ArCHeR-full",
        "display": "Full ArCHeR",
        "use_dpc": True,
        "use_rf_hig": True,
        "use_aknno": True,
        "semantics": "Frozen original ArCHeR",
    },
    {
        "method": "ArCHeR-dpc_to_knn_louvain",
        "display": "DPC -> fixed kNN-Louvain init",
        "use_dpc": False,
        "use_rf_hig": True,
        "use_aknno": True,
        "semantics": (
            "Replace DPC pseudo-label initialization with the fixed default "
            "kNN-Louvain initializer required by RF-HIG"
        ),
    },
    {
        "method": "ArCHeR-no_if_hig",
        "display": "w/o IF-HIG",
        "use_dpc": True,
        "use_rf_hig": False,
        "use_aknno": True,
        "semantics": "Remove RF-derived HIGs; retain the standard HVG view",
    },
    {
        "method": "ArCHeR-no_aknno",
        "display": "w/o aKNNO",
        "use_dpc": True,
        "use_rf_hig": True,
        "use_aknno": False,
        "semantics": "Replace aKNNO with the fixed default kNN-Louvain post-clustering",
    },
    {
        "method": "ArCHeR-knn_no_if_hig",
        "display": "kNN-Louvain init + w/o IF-HIG",
        "use_dpc": False,
        "use_rf_hig": False,
        "use_aknno": True,
        "semantics": (
            "Use the fixed kNN-Louvain initializer and remove RF-derived HIGs; "
            "the final aKNNO graph uses the standard HVG view"
        ),
    },
    {
        "method": "ArCHeR-consensus_hig",
        "display": "Consensus HIG from DPC + kNN views",
        "use_dpc": True,
        "use_rf_hig": True,
        "use_aknno": True,
        "hig_mode": "dpc_knn_consensus",
        "semantics": (
            "Run DPC and kNN-Louvain pseudo-label views, train RF-HIG on both, "
            "retain the intersection of their top-HIG sets, then run aKNNO"
        ),
    },
    {
        "method": "ArCHeR-union_hig",
        "display": "Union HIG from DPC + kNN views",
        "use_dpc": True,
        "use_rf_hig": True,
        "use_aknno": True,
        "hig_mode": "dpc_knn_capped_union",
        "semantics": (
            "Run DPC and kNN-Louvain pseudo-label views, retain the union of "
            "their top-HIG sets, then run aKNNO"
        ),
    },
    {
        "method": "ArCHeR-capped_union_hig",
        "display": "Capped union HIG from DPC + kNN views",
        "use_dpc": True,
        "use_rf_hig": True,
        "use_aknno": True,
        "hig_mode": "dpc_knn_capped_union",
        "semantics": (
            "Fuse the union of DPC and kNN-Louvain top-HIG candidates by "
            "best per-view rank, cap the final HIG set at RF_HIG_TOPN, then run aKNNO"
        ),
    },
    {
        "method": "ArCHeR-v3-full",
        "display": "Full ArCHeR v3.2 scale-adaptive capped-HIG + aKNNO",
        "use_dpc": True,
        "use_rf_hig": True,
        "use_aknno": True,
        "hig_mode": "dpc_knn_capped_union",
        "semantics": (
            "Scale-adaptive candidate: DPC and kNN-Louvain capped-union HIG "
            "up to 10,000 cells; kNN-only HIG above the fixed threshold; then aKNNO"
        ),
    },
    {
        "method": "ArCHeR-v3-no_aknno",
        "display": "ArCHeR v3.2 w/o aKNNO",
        "use_dpc": True,
        "use_rf_hig": True,
        "use_aknno": False,
        "hig_mode": "dpc_knn_capped_union",
        "semantics": (
            "Retain the identical dual-view capped-union HIG feature set "
            "and replace only the final aKNNO stage with fixed kNN-Louvain"
        ),
    },
    {
        "method": "ArCHeR-v3-no_hig",
        "display": "ArCHeR v3.2 w/o HIG",
        "use_dpc": False,
        "use_rf_hig": False,
        "use_aknno": True,
        "semantics": (
            "Remove the complete dual-view HIG module and run the identical "
            "aKNNO stage using standard HVGs only"
        ),
    },
]
METHOD_ORDER = [item["method"] for item in METHODS]
METRICS = [
    "ACC",
    "ARI",
    "NMI",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Rare_Precision",
    "Rare_Recall",
    "Rare_F1",
    "Rare_ACC_legacy",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_already_log(adata):
    if "log1p" in adata.uns:
        return True
    if "total_counts" in adata.obs:
        values = pd.to_numeric(adata.obs["total_counts"], errors="coerce")
        if values.notna().any():
            if float(values.max()) > 30000 or float(values.median()) > 2000:
                return False
    n_rows = min(1000, adata.n_obs)
    n_cols = min(200, adata.n_vars)
    block = adata.X[:n_rows, :n_cols]
    if hasattr(block, "toarray"):
        block = block.toarray()
    return float(np.nanmax(block)) <= 30.0


def resolve_dataset_path(dataset):
    candidates = [
        os.path.join(BASE_DIR, "data", "benchmarks", dataset, "sc_cluster_fixed_result.h5ad"),
        os.path.join(BASE_DIR, dataset, "sc_cluster_fixed_result.h5ad"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Missing benchmark h5ad for {}".format(dataset))


def pick_labels(labels):
    values, counts = np.unique(labels, return_counts=True)
    if len(values) < 3:
        return None
    descending = np.argsort(counts)[::-1]
    abundant_a, abundant_b = values[descending[:2]]
    eligible = [value for value in values[np.argsort(counts)] if value not in (abundant_a, abundant_b)]
    if not eligible:
        return None
    rare_challenging = eligible[0]
    rare_easier = eligible[1] if len(eligible) > 1 else eligible[0]
    return abundant_a, abundant_b, rare_challenging, rare_easier


def sample_cells(labels, abundant_a, abundant_b, rare_label, abundant_n, rare_n, seed):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    selected = []
    for label in (abundant_a, abundant_b):
        indices = np.where(labels == label)[0]
        size = min(int(abundant_n), len(indices))
        selected.append(rng.choice(indices, size=size, replace=False))
    rare_indices = np.where(labels == rare_label)[0]
    selected.append(
        rng.choice(rare_indices, size=int(rare_n), replace=(rare_n > len(rare_indices)))
    )
    return np.concatenate(selected)


def hungarian_match(true_labels, pred_labels):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    true_ids = np.unique(true_labels)
    pred_ids = np.unique(pred_labels)
    overlap = np.zeros((len(pred_ids), len(true_ids)), dtype=np.int64)
    for row, pred_id in enumerate(pred_ids):
        for col, true_id in enumerate(true_ids):
            overlap[row, col] = int(np.sum((pred_labels == pred_id) & (true_labels == true_id)))
    row_ind, col_ind = linear_sum_assignment(-overlap)
    mapping = {pred_ids[row]: true_ids[col] for row, col in zip(row_ind, col_ind)}
    for pred_id in pred_ids:
        if pred_id not in mapping:
            mask = pred_labels == pred_id
            mapping[pred_id] = pd.Series(true_labels[mask]).value_counts().idxmax()
    return np.asarray([mapping[value] for value in pred_labels])


def compute_metrics(true_labels, pred_labels, rare_label):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    mapped = hungarian_match(true_labels, pred_labels)
    label_order = np.unique(true_labels)
    rare_true = true_labels == rare_label
    rare_pred = mapped == rare_label
    tp = int(np.sum(rare_true & rare_pred))
    fp = int(np.sum(~rare_true & rare_pred))
    fn = int(np.sum(rare_true & ~rare_pred))
    rare_precision = tp / float(tp + fp) if tp + fp else 0.0
    rare_recall = tp / float(tp + fn) if tp + fn else 0.0
    rare_f1 = (
        2.0 * rare_precision * rare_recall / (rare_precision + rare_recall)
        if rare_precision + rare_recall
        else 0.0
    )
    return {
        "ACC": float(np.mean(mapped == true_labels)),
        "ARI": float(adjusted_rand_score(true_labels, pred_labels)),
        "NMI": float(normalized_mutual_info_score(true_labels, pred_labels)),
        "Macro_Precision": float(
            precision_score(true_labels, mapped, labels=label_order, average="macro", zero_division=0)
        ),
        "Macro_Recall": float(
            recall_score(true_labels, mapped, labels=label_order, average="macro", zero_division=0)
        ),
        "Macro_F1": float(
            f1_score(true_labels, mapped, labels=label_order, average="macro", zero_division=0)
        ),
        "Rare_Precision": float(rare_precision),
        "Rare_Recall": float(rare_recall),
        "Rare_F1": float(rare_f1),
        "Rare_ACC_legacy": float(rare_recall),
        "rare_tp": tp,
        "rare_fp": fp,
        "rare_fn": fn,
        "predicted_rare_cells": int(np.sum(rare_pred)),
        "n_pred_clusters": int(np.unique(pred_labels).size),
    }


def set_case_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import igraph as ig

        ig.set_random_number_generator(random.Random(int(seed)))
    except Exception:
        pass


def append_checkpoint(row, output_csv):
    pd.DataFrame([row]).to_csv(
        output_csv,
        mode="a",
        index=False,
        header=not os.path.exists(output_csv),
    )


def cluster_bootstrap_interval(dataset_deltas, seed, n_bootstrap=10000):
    values = np.asarray(dataset_deltas, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.RandomState(int(seed))
    samples = rng.choice(values, size=(int(n_bootstrap), len(values)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_results(
    frame,
    output_dir,
    bootstrap_seed,
    method_order=None,
    reference_method="ArCHeR-full",
):
    method_order = list(method_order or METHOD_ORDER)
    expected = set(method_order)
    observed = set(frame["method"].unique())
    if expected != observed:
        raise ValueError("Incomplete method coverage: expected {}, observed {}".format(expected, observed))

    counts = frame.groupby(KEY_COLUMNS)["method"].nunique()
    if not (counts == len(method_order)).all():
        raise ValueError("Some paired cases do not contain all methods")

    summary_rows = []
    for method in method_order:
        subset = frame[frame["method"].eq(method)]
        row = {"method": method, "n_runs": int(len(subset))}
        for metric in METRICS:
            row[metric + "_mean"] = float(subset[metric].mean())
            row[metric + "_sd"] = float(subset[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    dataset_means = (
        frame.groupby(["dataset", "method"], as_index=False)[METRICS]
        .mean()
        .sort_values(["dataset", "method"])
    )
    dataset_means.to_csv(os.path.join(output_dir, "by_dataset.csv"), index=False)

    scenario_means = (
        frame.groupby(["dataset", "setting", "method"], as_index=False)[METRICS]
        .mean()
        .sort_values(["dataset", "setting", "method"])
    )
    scenario_means.to_csv(os.path.join(output_dir, "by_scenario.csv"), index=False)

    paired_rows = []
    # The paper's ablation has four paired units: dataset x difficulty setting.
    # Keep the dataset-level summary above, but estimate component effects at
    # the scenario level so the two difficulty settings are not collapsed.
    reference = scenario_means[
        scenario_means["method"].eq(reference_method)
    ].set_index(["dataset", "setting"])
    for method in [item for item in method_order if item != reference_method]:
        ablated = scenario_means[scenario_means["method"].eq(method)].set_index(
            ["dataset", "setting"]
        )
        common = reference.index.intersection(ablated.index)
        for metric in METRICS:
            # Positive delta means the reference method improved the metric.
            deltas = reference.loc[common, metric] - ablated.loc[common, metric]
            ci_low, ci_high = cluster_bootstrap_interval(
                deltas.values,
                seed=int(bootstrap_seed) + METRICS.index(metric) * 101 + method_order.index(method),
            )
            if np.allclose(deltas.values, 0.0):
                p_value = 1.0
            else:
                try:
                    p_value = float(
                        wilcoxon(deltas.values, zero_method="pratt", alternative="two-sided").pvalue
                    )
                except ValueError:
                    p_value = np.nan
            paired_rows.append(
                {
                    "reference_method": reference_method,
                    "ablation": method,
                    "metric": metric,
                    "unit_level": "dataset_setting",
                    "n_scenarios": int(len(common)),
                    "n_datasets": int(len(set(common.get_level_values("dataset")))),
                    "full_minus_ablation_mean": float(deltas.mean()),
                    "reference_minus_method_mean": float(deltas.mean()),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "wilcoxon_p": p_value,
                    "full_wins": int(np.sum(deltas > 1e-12)),
                    "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
                    "full_losses": int(np.sum(deltas < -1e-12)),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(os.path.join(output_dir, "paired_effects.csv"), index=False)
    return summary, dataset_means, paired


def config_snapshot():
    keys = [
        "HVG_MIN_MEAN",
        "HVG_MAX_MEAN",
        "HVG_MIN_DISP",
        "HVG_TOPN",
        "DPC_PCA_N_COMPONENTS",
        "DPC_KNN_K",
        "DPC_DC_PERCENT",
        "DPC_GAMMA_TOPK",
        "RF_HIG_TOPN",
        "RF_PCA_DIMS",
        "RF_N_ESTIMATORS",
        "INIT_KNN_K",
        "INIT_LOUVAIN_RES",
        "AKNN_KMAX",
        "AKNN_PRUNE_SNN",
        "AKNN_CUTOFF",
        "USE_TARGET_CLUSTER_COUNT",
    ]
    return {key: getattr(Config, key) for key in keys}


def effective_params(dataset, profile, overrides=None):
    overrides = dict(overrides or {})
    if profile == "frozen_default":
        return overrides or None
    if dataset not in PAPER_TUNED_CONFIGS:
        raise ValueError(
            "paper_tuned profile is only defined for {}".format(
                ", ".join(PAPER_TUNED_DATASETS)
            )
        )
    params = dict(PAPER_TUNED_CONFIGS[dataset])
    params.update(overrides)
    return params


def main():
    parser = argparse.ArgumentParser(description="Run a paired, label-withheld ArCHeR ablation benchmark")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset", action="append", choices=DEFAULT_DATASETS)
    parser.add_argument("--method", action="append", choices=METHOD_ORDER)
    parser.add_argument(
        "--reference-method",
        choices=METHOD_ORDER,
        default="ArCHeR-full",
        help="Reference method used for paired effect estimation",
    )
    parser.add_argument("--rare-count", action="append", type=int)
    parser.add_argument("--setting", action="append", choices=["Setting1_Challenging", "Setting2_Easier"])
    parser.add_argument(
        "--profile",
        choices=["frozen_default", "paper_tuned"],
        default="frozen_default",
        help="Configuration profile; paper_tuned reproduces the historical two-dataset ablation settings",
    )
    parser.add_argument(
        "--backend",
        choices=["native", "configurable"],
        default="native",
        help="Execution backend; configurable gives a matched backend for consensus-HIG comparisons",
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--abundant-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--rf-hig-topn",
        type=int,
        default=None,
        help="Optional audited override for RF_HIG_TOPN during HIG sensitivity runs",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = args.dataset or (PAPER_TUNED_DATASETS if args.profile == "paper_tuned" else DEFAULT_DATASETS)
    active_method_order = [
        method for method in METHOD_ORDER
        if not args.method or method in set(args.method)
    ]
    if args.reference_method not in active_method_order:
        raise ValueError(
            "{} must be included for paired effect estimation".format(
                args.reference_method
            )
        )
    active_methods = [
        item for item in METHODS if item["method"] in active_method_order
    ]
    if args.profile == "paper_tuned":
        invalid = [dataset for dataset in datasets if dataset not in PAPER_TUNED_DATASETS]
        if invalid:
            raise ValueError(
                "paper_tuned profile does not define: {}".format(", ".join(invalid))
            )
    rare_counts = sorted(set(args.rare_count or DEFAULT_RARE_COUNTS))
    settings_filter = set(args.setting or [])
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, "per_run.csv")
    if args.overwrite and os.path.exists(output_csv):
        os.remove(output_csv)

    existing = set()
    if os.path.exists(output_csv):
        old = pd.read_csv(output_csv)
        existing = set(
            zip(old["dataset"], old["setting"], old["rare_count"], old["rep"], old["method"])
        )

    input_paths = {}
    for dataset in datasets:
        path = resolve_dataset_path(dataset)
        input_paths[dataset] = path
        print("\n=== {} ===".format(dataset), flush=True)
        adata = sc.read_h5ad(path, backed="r")
        if "true_label" not in adata.obs:
            raise ValueError("Missing true_label in {}".format(path))
        labels = pd.to_numeric(adata.obs["true_label"], errors="coerce").astype(int).values
        picked = pick_labels(labels)
        if picked is None:
            print("Skip {}: fewer than three eligible classes".format(dataset), flush=True)
            continue
        abundant_a, abundant_b, rare_challenging, rare_easier = picked
        settings = [
            ("Setting1_Challenging", rare_challenging),
            ("Setting2_Easier", rare_easier),
        ]
        if settings_filter:
            settings = [item for item in settings if item[0] in settings_filter]
        already_log = infer_already_log(adata)
        parameter_overrides = {}
        if args.rf_hig_topn is not None:
            if int(args.rf_hig_topn) <= 0:
                raise ValueError("--rf-hig-topn must be positive")
            parameter_overrides["RF_HIG_TOPN"] = int(args.rf_hig_topn)
        params = effective_params(dataset, args.profile, parameter_overrides)

        for setting, rare_label in settings:
            for rare_count in rare_counts:
                for rep in range(1, int(args.reps) + 1):
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
                    subset = adata[selected].to_memory()
                    if subset.X.dtype != np.float32:
                        subset.X = subset.X.astype(np.float32)
                    true_labels = pd.to_numeric(
                        subset.obs["true_label"], errors="coerce"
                    ).astype(int).values
                    # Enforce the label firewall even though the frozen default
                    # currently has USE_TARGET_CLUSTER_COUNT=False.
                    if "true_label" in subset.obs:
                        del subset.obs["true_label"]

                    for method_spec in active_methods:
                        key = (dataset, setting, rare_count, rep, method_spec["method"])
                        if key in existing:
                            continue
                        set_case_seed(sample_seed)
                        print(
                            "{} | {} | rare={} | rep={} | {}".format(
                                dataset, setting, rare_count, rep, method_spec["method"]
                            ),
                            flush=True,
                        )
                        if (
                            args.profile == "paper_tuned"
                            or args.backend == "configurable"
                            or str(method_spec.get("hig_mode", "")).startswith("dpc_knn_")
                        ):
                            method_params = params
                            if method_spec["method"].startswith("ArCHeR-v3"):
                                method_params = final.get_default_params(params)
                            prediction = run_archer_pipeline_with_params(
                                subset.copy(),
                                true_labels=None,
                                already_log=already_log,
                                params=method_params,
                                use_dpc=method_spec["use_dpc"],
                                use_rf_hig=method_spec["use_rf_hig"],
                                use_aknno=method_spec["use_aknno"],
                                random_seed=sample_seed,
                                hig_mode=method_spec.get("hig_mode", "single"),
                            )
                        else:
                            prediction = run_archer_pipeline(
                                subset.copy(),
                                true_labels=None,
                                already_log=already_log,
                                use_dpc=method_spec["use_dpc"],
                                use_rf_hig=method_spec["use_rf_hig"],
                                use_aknno=method_spec["use_aknno"],
                            )
                        row = {
                            "dataset": dataset,
                            "setting": setting,
                            "rare_count": int(rare_count),
                            "rep": int(rep),
                            "sample_seed": int(sample_seed),
                            "method": method_spec["method"],
                            "rare_label": int(rare_label),
                            "n_cells": int(len(true_labels)),
                        }
                        row.update(compute_metrics(true_labels, prediction, rare_label))
                        append_checkpoint(row, output_csv)
                        existing.add(key)

    frame = pd.read_csv(output_csv)
    frame = frame[
        frame["dataset"].isin(datasets)
        & frame["rare_count"].isin(rare_counts)
        & frame["method"].isin(active_method_order)
    ].copy()
    if settings_filter:
        frame = frame[frame["setting"].isin(settings_filter)].copy()
    summary, _, paired = summarize_results(
        frame,
        args.output_dir,
        args.seed,
        method_order=active_method_order,
        reference_method=args.reference_method,
    )

    uses_configurable = (
        args.profile == "paper_tuned"
        or args.backend == "configurable"
        or any(
            str(item.get("hig_mode", "")).startswith("dpc_knn_")
            for item in active_methods
        )
    )
    method_paths = [os.path.join(SCRIPT_DIR, "method_archer.py")]
    if uses_configurable:
        method_paths.append(os.path.join(SCRIPT_DIR, "method_archer_configurable.py"))
    uses_final_candidate = any(
        item["method"].startswith("ArCHeR-v3") for item in active_methods
    )
    if uses_final_candidate:
        method_paths.append(os.path.join(SCRIPT_DIR, "method_archer_final.py"))
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runner": os.path.abspath(__file__),
        "runner_sha256": sha256_file(os.path.abspath(__file__)),
        "method_source": method_paths if len(method_paths) > 1 else method_paths[0],
        "method_sha256": {
            path: sha256_file(path) for path in method_paths
        },
        "execution_backend": (
            "method_archer_configurable.run_archer_pipeline_with_params"
            if uses_configurable and (
                args.profile == "paper_tuned" or args.backend == "configurable"
            )
            else (
                "mixed original/configurable backends"
                if uses_configurable
                else "method_archer.run_archer_pipeline"
            )
        ),
        "datasets": datasets,
        "methods": active_method_order,
        "reference_method": args.reference_method,
        "rare_counts": rare_counts,
        "settings": sorted(settings_filter) if settings_filter else [
            "Setting1_Challenging",
            "Setting2_Easier",
        ],
        "reps": int(args.reps),
        "abundant_n": int(args.abundant_n),
        "base_seed": int(args.seed),
        "profile": args.profile,
        "backend": args.backend,
        "parameter_overrides": parameter_overrides,
        "final_method_version": (
            final.METHOD_VERSION if uses_final_candidate else None
        ),
        "label_firewall": "true_labels=None and true_label removed from model input obs",
        "legacy_metric_note": "Rare_ACC_legacy is exactly Rare_Recall",
        "paired_effect_unit": "dataset x setting",
        "ablation_semantics": {
            item["method"]: item["semantics"] for item in active_methods
        },
        "frozen_default_config": config_snapshot(),
        "paper_tuned_config": PAPER_TUNED_CONFIGS if args.profile == "paper_tuned" else None,
        "input_sources": input_paths,
        "input_sha256": {name: sha256_file(path) for name, path in input_paths.items()},
        "primary_metrics": ["Rare_F1", "Macro_F1", "ACC"],
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print("\nSummary", flush=True)
    print(summary[["method", "Rare_Precision_mean", "Rare_Recall_mean", "Rare_F1_mean"]].to_string(index=False), flush=True)
    print("\nPaired Rare_F1 effects (positive = full model better)", flush=True)
    print(paired[paired["metric"].eq("Rare_F1")].to_string(index=False), flush=True)
    print("\nSaved outputs to {}".format(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
