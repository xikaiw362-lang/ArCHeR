"""Per-dataset tuning run used only for the Figure 2A descriptive panel.

This intentionally separate workflow selects one frozen candidate from a
predeclared parameter pool for each full benchmark dataset.  It is not used
for the primary no-tuning generalization claim.
"""

import argparse
import gc
import hashlib
import itertools
import json
import os
import random
import sys
import time
from datetime import datetime

import igraph as ig
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

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
    "archer_v31_fig2a_per_dataset_tuned_20260717",
)

SCORE_WEIGHTS = {
    "ACC": 1.0,
    "Macro_Precision": 1.0,
    "Macro_Recall": 1.0,
    "Macro_F1": 2.0,
    "ARI": 4.0,
    "NMI": 1.5,
    "Purity": 1.0,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    ig.set_random_number_generator(random.Random(int(seed)))


def stable_seed(base_seed, dataset):
    return evaluation.stable_scenario_seed(base_seed, "fig2a_per_dataset_tuning", dataset)


def candidate_pool():
    """Predeclared, compact grid for the Figure 2A descriptive comparison."""
    return [
        {},
        {"DPC_GAMMA_TOPK": 6},
        {"DPC_GAMMA_TOPK": 10},
        {"DPC_GAMMA_TOPK": 12},
        {"AKNN_KMAX": 15},
        {"AKNN_KMAX": 18},
        {"AKNN_PRUNE_SNN": 1 / 30},
        {"AKNN_KMAX": 15, "AKNN_PRUNE_SNN": 1 / 30},
        {"AKNN_KMAX": 18, "AKNN_PRUNE_SNN": 1 / 30},
        {
            "DPC_GAMMA_TOPK": 10,
            "AKNN_KMAX": 15,
            "AKNN_PRUNE_SNN": 1 / 30,
        },
        {
            "DPC_GAMMA_TOPK": 12,
            "AKNN_KMAX": 18,
            "AKNN_PRUNE_SNN": 1 / 30,
        },
        {"INIT_KNN_K": 15},
        {"INIT_KNN_K": 25},
        {"RF_HIG_TOPN": 1500},
        {"RF_HIG_TOPN": 2500},
    ]


def config_name(config):
    if not config:
        return "default"
    parts = []
    for key in sorted(config):
        value = config[key]
        if isinstance(value, float):
            value = "{:.6f}".format(value).rstrip("0").rstrip(".")
        parts.append("{}={}".format(key, value))
    return "|".join(parts)


def score_metrics(metrics):
    return float(
        sum(float(metrics[key]) * weight for key, weight in SCORE_WEIGHTS.items())
    )


def compute_row(dataset, config_index, config, true_labels, prediction, runtime):
    metrics, _ = evaluation.compute_metrics(true_labels, prediction)
    row = {
        "dataset": dataset,
        "config_index": int(config_index),
        "config_name": config_name(config),
        "runtime_seconds": float(runtime),
        "n_cells": int(len(true_labels)),
        "n_params_overridden": int(len(config)),
    }
    for key in [
        "ACC",
        "ARI",
        "NMI",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "Purity",
        "Smallest_Class_F1",
    ]:
        row[key] = float(metrics[key])
    row["score"] = score_metrics(metrics)
    for key, value in config.items():
        row[key] = value
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Tune final ArCHeR separately on each dataset for Figure 2A only"
    )
    parser.add_argument("--datasets-from", default=DEFAULT_DATASET_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = pd.read_csv(args.datasets_from)["dataset"].drop_duplicates().tolist()
    if len(datasets) != 16:
        raise ValueError("Expected 16 datasets, observed {}".format(len(datasets)))
    os.makedirs(args.output_dir, exist_ok=True)
    candidate_path = os.path.join(args.output_dir, "candidate_metrics.csv")
    best_path = os.path.join(args.output_dir, "best_config.csv")
    summary_path = os.path.join(args.output_dir, "tuned_summary.csv")
    if args.overwrite:
        for path in [candidate_path, best_path, summary_path]:
            if os.path.exists(path):
                os.remove(path)

    configs = candidate_pool()
    all_rows = []
    best_rows = []
    if not args.overwrite and os.path.exists(candidate_path):
        existing = pd.read_csv(candidate_path)
        versions = set(existing.get("method_version", pd.Series(dtype=str)).dropna().astype(str))
        if len(existing) and not versions:
            raise RuntimeError(
                "Existing tuning checkpoint has no method_version and cannot "
                "be mixed with ArCHeR {}. Use a new output directory or "
                "--overwrite.".format(final.METHOD_VERSION)
            )
        if versions and versions != {final.METHOD_VERSION}:
            raise RuntimeError(
                "Existing tuning checkpoint belongs to method version {}; "
                "use a new output directory or --overwrite for {}".format(
                    sorted(versions), final.METHOD_VERSION
                )
            )
        all_rows = existing.to_dict("records")
    if not args.overwrite and os.path.exists(best_path):
        best_rows = pd.read_csv(best_path).to_dict("records")

    completed_datasets = {
        str(row["dataset"])
        for row in best_rows
        if "dataset" in row and pd.notna(row["dataset"])
    }

    for dataset_index, dataset in enumerate(datasets, 1):
        if dataset in completed_datasets:
            print(
                "\n[{}/{}] {} | checkpoint complete, skipping".format(
                    dataset_index, len(datasets), dataset
                ),
                flush=True,
            )
            continue

        # A hard stop can leave candidate-level checkpoints for an unfinished
        # dataset. Resume those configurations and recompute only missing rows.
        prior_rows = [row for row in all_rows if str(row.get("dataset")) == dataset]
        completed_configs = {
            int(row["config_index"])
            for row in prior_rows
            if pd.notna(row.get("config_index"))
        }
        print(
            "\n[{}/{}] {} | {} candidates".format(
                dataset_index, len(datasets), dataset, len(configs)
            ),
            flush=True,
        )
        counts, true_labels, _, already_log, source_path = evaluation.load_full_dataset(
            dataset, source="h5ad"
        )
        model_input = evaluation.strip_observation_metadata(counts)
        del counts
        gc.collect()
        seed = stable_seed(args.seed, dataset)
        dataset_rows = list(prior_rows)

        for config_index, config in enumerate(configs):
            if config_index in completed_configs:
                print(
                    "  {:02d} checkpoint complete, skipping".format(config_index),
                    flush=True,
                )
                continue
            set_seed(seed)
            started = time.time()
            prediction = final.run_archer_final_pipeline(
                model_input,
                true_labels=None,
                already_log=already_log,
                params=config,
                random_seed=seed,
            )
            runtime = time.time() - started
            row = compute_row(
                dataset,
                config_index,
                config,
                true_labels,
                prediction,
                runtime,
            )
            row["seed"] = int(seed)
            row["source_path"] = source_path
            row["method_version"] = final.METHOD_VERSION
            dataset_rows.append(row)
            all_rows.append(row)
            pd.DataFrame(all_rows).to_csv(candidate_path, index=False)
            print(
                "  {:02d} {:<72} score={:.4f} ARI={:.4f} F1={:.4f}".format(
                    config_index,
                    row["config_name"],
                    row["score"],
                    row["ARI"],
                    row["Macro_F1"],
                ),
                flush=True,
            )
            del prediction
            gc.collect()

        dataset_frame = pd.DataFrame(dataset_rows).sort_values(
            ["score", "ARI", "Macro_F1", "NMI", "ACC"],
            ascending=[False, False, False, False, False],
        )
        best = dataset_frame.iloc[0].to_dict()
        best["dataset_rank"] = 1
        best_rows.append(best)
        print(
            "  BEST {} | score={:.4f} ARI={:.4f} F1={:.4f}".format(
                best["config_name"],
                best["score"],
                best["ARI"],
                best["Macro_F1"],
            ),
            flush=True,
        )
        pd.DataFrame(all_rows).to_csv(candidate_path, index=False)
        pd.DataFrame(best_rows).to_csv(best_path, index=False)
        del model_input, true_labels
        gc.collect()

    candidates = pd.DataFrame(all_rows)
    best = pd.DataFrame(best_rows).sort_values("dataset").reset_index(drop=True)
    tuned_summary = pd.DataFrame(
        [
            {
                "method": "ArCHeR v3.1 (per-dataset tuned)",
                "n_datasets": int(len(best)),
                "ACC": float(best["ACC"].mean()),
                "Precision": float(best["Macro_Precision"].mean()),
                "Recall": float(best["Macro_Recall"].mean()),
                "F1": float(best["Macro_F1"].mean()),
                "ARI": float(best["ARI"].mean()),
                "NMI": float(best["NMI"].mean()),
                "Purity": float(best["Purity"].mean()),
                "Smallest_Class_F1": float(best["Smallest_Class_F1"].mean()),
                "score": float(best["score"].mean()),
            }
        ]
    )
    candidates.to_csv(candidate_path, index=False)
    best.to_csv(best_path, index=False)
    tuned_summary.to_csv(summary_path, index=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis": "Figure 2A per-dataset tuned descriptive benchmark",
        "datasets": datasets,
        "method": final.METHOD_NAME,
        "method_version": final.METHOD_VERSION,
        "method_entrypoint": os.path.join(SCRIPT_DIR, "method_archer_final.py"),
        "method_entrypoint_sha256": sha256_file(
            os.path.join(SCRIPT_DIR, "method_archer_final.py")
        ),
        "implementation_sha256": sha256_file(
            os.path.join(SCRIPT_DIR, "method_archer_configurable.py")
        ),
        "runner": os.path.abspath(__file__),
        "runner_sha256": sha256_file(os.path.abspath(__file__)),
        "seed": int(args.seed),
        "seed_scheme": "sha256_dataset_fig2a_per_dataset_tuning_v1",
        "candidate_count": int(len(configs)),
        "candidate_pool": configs,
        "score_weights": SCORE_WEIGHTS,
        "selection_order": ["score", "ARI", "Macro_F1", "NMI", "ACC"],
        "label_firewall": (
            "true_labels=None and true_label metadata removed before every model call; "
            "labels used only after prediction for candidate selection"
        ),
        "interpretation_warning": (
            "This table is descriptive per-dataset tuning on the full benchmark "
            "datasets and must not be used as an unbiased generalization estimate."
        ),
        "outputs": {
            "candidate_metrics": candidate_path,
            "best_config": best_path,
            "tuned_summary": summary_path,
        },
    }
    with open(
        os.path.join(args.output_dir, "manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print("\nTuned summary", flush=True)
    print(tuned_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
