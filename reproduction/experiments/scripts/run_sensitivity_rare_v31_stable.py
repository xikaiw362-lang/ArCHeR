"""Dependency-safe ArCHeR v3.1 sensitivity analysis in rare-cell stress tests."""

import argparse
import csv
import os
import random
import sys

import igraph as ig
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import method_archer_final as final  # noqa: E402
import run_archer_v2_evaluation as evaluation  # noqa: E402


BASE = r"D:\sc_dataset"
RARE_DATASETS = ["Quake_10x_Limb_Muscle", "Quake_Smart-seq2_Heart"]
PARAM_GRID = {
    "DPC_GAMMA_TOPK": [3, 8, 15, 24],
    "DPC_DC_PERCENT": [2, 4, 6, 8, 10],
    "HVG_TOPN": [100, 500, 800, 1500, 2000, 3000],
    "RF_HIG_TOPN": [100, 500, 800, 1500, 2000, 3000],
}
OUTPUT_COLUMNS = [
    "dataset",
    "setting",
    "rare_count",
    "rep",
    "sample_seed",
    "param_name",
    "param_value",
    "rare_label",
    "n_cells",
    "ACC",
    "F1",
    "Precision",
    "Recall",
    "Rare_ACC",
    "Rare_F1",
    "Rare_Precision",
    "Rare_Recall",
    "n_pred",
]


def pick_labels(labels):
    values, counts = np.unique(labels, return_counts=True)
    if len(values) < 4:
        raise ValueError("Rare sensitivity requires at least four classes")
    descending = np.argsort(counts)[::-1]
    abundant_a = values[descending[0]]
    abundant_b = values[descending[1]]
    rare_candidates = [value for value in values[np.argsort(counts)] if value not in (abundant_a, abundant_b)]
    return abundant_a, abundant_b, rare_candidates[0], rare_candidates[1]


def sample_indices(labels, abundant_a, abundant_b, rare_label, abundant_n, rare_n, seed):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    idx_a = np.where(labels == abundant_a)[0]
    idx_b = np.where(labels == abundant_b)[0]
    idx_r = np.where(labels == rare_label)[0]
    selected_a = rng.choice(idx_a, size=min(abundant_n, len(idx_a)), replace=False)
    selected_b = rng.choice(idx_b, size=min(abundant_n, len(idx_b)), replace=False)
    selected_r = rng.choice(idx_r, size=rare_n, replace=rare_n > len(idx_r))
    return np.concatenate([selected_a, selected_b, selected_r])


def compute_metrics(true_labels, pred_labels, rare_label):
    true_labels = np.asarray(true_labels)
    mapped = evaluation.hungarian_map(true_labels, pred_labels)
    labels = np.unique(true_labels)
    rare_mask = true_labels == rare_label
    return {
        "ACC": float(np.mean(mapped == true_labels)),
        "F1": float(f1_score(true_labels, mapped, labels=labels, average="macro", zero_division=0)),
        "Precision": float(precision_score(true_labels, mapped, labels=labels, average="macro", zero_division=0)),
        "Recall": float(recall_score(true_labels, mapped, labels=labels, average="macro", zero_division=0)),
        "Rare_ACC": float(np.mean(mapped[rare_mask] == rare_label)),
        "Rare_F1": float(f1_score(true_labels, mapped, labels=[rare_label], average="macro", zero_division=0)),
        "Rare_Precision": float(precision_score(true_labels, mapped, labels=[rare_label], average="macro", zero_division=0)),
        "Rare_Recall": float(recall_score(true_labels, mapped, labels=[rare_label], average="macro", zero_division=0)),
        "n_pred": int(len(np.unique(pred_labels))),
    }


def load_existing(path):
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            keys.add(
                (
                    row["dataset"],
                    row["setting"],
                    str(row["rare_count"]),
                    str(row["rep"]),
                    row["param_name"],
                    str(row["param_value"]),
                )
            )
    return keys


def append_row(path, row):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in OUTPUT_COLUMNS})


def slice_counts(counts, indices):
    try:
        subset = counts[indices]
    except Exception:
        subset = counts.iloc[indices]
    return subset.copy() if hasattr(subset, "copy") else subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=os.path.join(BASE, "experiments", "metrics", "sensitivity_rare_v31_stable_20260717.csv"),
    )
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--abundant-n", type=int, default=200)
    parser.add_argument("--rare-min", type=int, default=2)
    parser.add_argument("--rare-max", type=int, default=20)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-new-runs", type=int, default=0)
    args = parser.parse_args()

    datasets = args.dataset if args.dataset else RARE_DATASETS
    unknown = sorted(set(datasets) - set(RARE_DATASETS))
    if unknown:
        raise ValueError("Unknown rare-sensitivity datasets: {}".format(unknown))
    rare_counts = list(range(args.rare_min, args.rare_max + 1, 2))
    candidates_per_dataset = sum(len(values) for values in PARAM_GRID.values()) * 2 * len(rare_counts) * args.reps
    total = candidates_per_dataset * len(datasets)
    existing = load_existing(args.out)
    position = 0
    new_runs = 0
    print("Rare sensitivity plan: {} candidates across {} datasets".format(total, len(datasets)), flush=True)

    for dataset_index, dataset in enumerate(datasets):
        counts, labels, _, already_log, _ = evaluation.load_full_dataset(dataset, source="h5ad")
        counts = evaluation.strip_observation_metadata(counts)
        labels = np.asarray(labels)
        abundant_a, abundant_b, rare_challenging, rare_easier = pick_labels(labels)
        settings = [
            ("Setting1_Challenging", rare_challenging),
            ("Setting2_Easier", rare_easier),
        ]
        print(
            "Dataset {} | abundant=({}, {}) | rare=({}, {})".format(
                dataset, abundant_a, abundant_b, rare_challenging, rare_easier
            ),
            flush=True,
        )
        for param_name, values in PARAM_GRID.items():
            for param_value in values:
                for setting, rare_label in settings:
                    for rare_count in rare_counts:
                        for rep in range(1, args.reps + 1):
                            position += 1
                            key = (
                                dataset,
                                setting,
                                str(rare_count),
                                str(rep),
                                param_name,
                                str(param_value),
                            )
                            if key in existing:
                                continue
                            sample_seed = args.seed + dataset_index * 100000 + rare_count * 100 + rep
                            indices = sample_indices(
                                labels,
                                abundant_a,
                                abundant_b,
                                rare_label,
                                args.abundant_n,
                                rare_count,
                                sample_seed,
                            )
                            subset = slice_counts(counts, indices)
                            true_subset = labels[indices]
                            params = final.get_default_params(
                                {param_name: param_value, "NEIGHBOR_N_JOBS": 1}
                            )
                            random.seed(sample_seed)
                            np.random.seed(sample_seed)
                            ig.set_random_number_generator(random.Random(sample_seed))
                            pred = final.run_archer_final_pipeline(
                                subset,
                                true_labels=None,
                                already_log=already_log,
                                params=params,
                                random_seed=sample_seed,
                            )
                            row = {
                                "dataset": dataset,
                                "setting": setting,
                                "rare_count": rare_count,
                                "rep": rep,
                                "sample_seed": sample_seed,
                                "param_name": param_name,
                                "param_value": param_value,
                                "rare_label": rare_label,
                                "n_cells": len(indices),
                            }
                            row.update(compute_metrics(true_subset, pred, rare_label))
                            append_row(args.out, row)
                            existing.add(key)
                            new_runs += 1
                            if new_runs == 1 or new_runs % 25 == 0:
                                print(
                                    "  [{}/{}] {} {} rare={} {}={} | Rare_F1={:.4f}".format(
                                        position,
                                        total,
                                        dataset,
                                        setting,
                                        rare_count,
                                        param_name,
                                        param_value,
                                        row["Rare_F1"],
                                    ),
                                    flush=True,
                                )
                            if args.max_new_runs and new_runs >= args.max_new_runs:
                                print("Reached max_new_runs={}".format(args.max_new_runs), flush=True)
                                return

    print("Rare sensitivity complete: {} new / {} planned".format(new_runs, total), flush=True)


if __name__ == "__main__":
    main()
