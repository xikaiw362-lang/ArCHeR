"""Dependency-safe v3.1 parameter sensitivity runner.

This is intentionally independent of the legacy sensitivity runner.  It uses
the same h5ad loading and observation-metadata firewall as the formal v3.1
benchmark, and appends checkpoints with the standard-library csv module so the
HNSW process is not mixed with legacy pandas metric side effects.
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter

import igraph as ig
import numpy as np
import pandas as pd
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

import method_archer_final as final  # noqa: E402
import run_archer_v2_evaluation as evaluation  # noqa: E402


BASE_DIR = r"D:\sc_dataset"
PARAM_GRID = {
    "DPC_GAMMA_TOPK": [3, 8, 15, 24],
    "DPC_DC_PERCENT": [2, 4, 6, 8, 10],
    "HVG_TOPN": [100, 500, 800, 1500, 2000, 3000],
    "RF_HIG_TOPN": [100, 500, 800, 1500, 2000, 3000],
}
OUTPUT_COLUMNS = [
    "dataset", "n_cells", "n_true_clusters", "param_name", "param_value",
    "ACC", "NMI", "ARI", "F1", "Precision", "Recall", "Purity", "n_pred",
]


def purity_score(true_labels, pred_labels):
    contingency = {}
    for true_label, pred_label in zip(true_labels, pred_labels):
        contingency.setdefault(pred_label, Counter())[true_label] += 1
    return sum(v.most_common(1)[0][1] for v in contingency.values()) / float(len(true_labels))


def metrics(true_labels, pred_labels):
    mapped = evaluation.hungarian_map(true_labels, pred_labels)
    labels = np.unique(true_labels)
    return {
        "ACC": float(np.mean(mapped == true_labels)),
        "NMI": float(normalized_mutual_info_score(true_labels, pred_labels)),
        "ARI": float(adjusted_rand_score(true_labels, pred_labels)),
        "F1": float(f1_score(true_labels, mapped, labels=labels, average="macro", zero_division=0)),
        "Precision": float(precision_score(true_labels, mapped, labels=labels, average="macro", zero_division=0)),
        "Recall": float(recall_score(true_labels, mapped, labels=labels, average="macro", zero_division=0)),
        "Purity": float(purity_score(true_labels, pred_labels)),
        "n_pred": int(len(np.unique(pred_labels))),
    }


def resolve_datasets(path, requested):
    names = sorted(pd.read_csv(path)["dataset"].dropna().astype(str).unique())
    if requested:
        missing = [name for name in requested if name not in names]
        if missing:
            raise ValueError("Unknown datasets: {}".format(missing))
        return list(dict.fromkeys(requested))
    return names


def load_existing(path):
    rows = []
    keys = set()
    if not os.path.exists(path):
        return rows, keys
    with open(path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            keys.add((row["dataset"], row["param_name"], str(row["param_value"])))
    return rows, keys


def append_row(path, row):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in OUTPUT_COLUMNS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-from", required=True)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefer-h5ad", action="store_true")
    parser.add_argument("--max-new-runs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    datasets = resolve_datasets(args.datasets_from, args.dataset)
    _, existing = load_existing(args.out)
    total = sum(len(values) for values in PARAM_GRID.values()) * len(datasets)
    completed = 0
    new_runs = 0
    print("Resolved {} datasets".format(len(datasets)), flush=True)

    for dataset in datasets:
        counts, true_labels, _, already_log, _ = evaluation.load_full_dataset(
            dataset, source="h5ad" if args.prefer_h5ad else "auto"
        )
        counts = evaluation.strip_observation_metadata(counts)
        print("Dataset {} | cells={} | clusters={}".format(
            dataset, len(true_labels), len(np.unique(true_labels))), flush=True)
        for param_name, values in PARAM_GRID.items():
            for param_value in values:
                completed += 1
                key = (dataset, param_name, str(param_value))
                if key in existing:
                    continue
                print("  [{}/{}] {} | {}={}".format(
                    completed, total, dataset, param_name, param_value
                ), flush=True)
                params = final.get_default_params({
                    param_name: param_value,
                    "NEIGHBOR_N_JOBS": 1,
                })
                random.seed(args.seed)
                np.random.seed(args.seed)
                ig.set_random_number_generator(random.Random(args.seed))
                pred = final.run_archer_final_pipeline(
                    counts,
                    true_labels=None,
                    already_log=already_log,
                    params=params,
                    random_seed=args.seed,
                )
                row = {
                    "dataset": dataset,
                    "n_cells": len(true_labels),
                    "n_true_clusters": len(np.unique(true_labels)),
                    "param_name": param_name,
                    "param_value": param_value,
                }
                row.update(metrics(true_labels, pred))
                append_row(args.out, row)
                existing.add(key)
                new_runs += 1
                print("    ACC={:.4f} F1={:.4f} ARI={:.4f}".format(
                    row["ACC"], row["F1"], row["ARI"]
                ), flush=True)
                if args.max_new_runs and new_runs >= args.max_new_runs:
                    print("Reached max_new_runs={}".format(args.max_new_runs), flush=True)
                    return

    print("Completed sensitivity runs: {} new / {} planned".format(new_runs, total), flush=True)


if __name__ == "__main__":
    main()
