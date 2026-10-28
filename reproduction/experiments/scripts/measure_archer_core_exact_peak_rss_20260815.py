"""Measure peak RSS for the frozen exact-neighbour ArCHeR core.

Each dataset is executed in an isolated Python process.  The parent process
uses the dataset-specific seed recorded in the frozen 16-dataset benchmark and
writes only to a caller-supplied new output directory.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time

import igraph as ig
import numpy as np
import pandas as pd
import psutil


BASE = r"D:\sc_dataset"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.join(
    BASE,
    "experiments",
    "metrics",
    "archer_core_20260724_k20_p1of15_exact_hig500_16benchmark",
)
FROZEN_CSV = os.path.join(FROZEN_DIR, "per_dataset.csv")
DEFAULT_OUTPUT = os.path.join(
    BASE,
    "experiments",
    "metrics",
    "archer_core_20260724_k20_p1of15_exact_hig500_peakrss_20260815_v1",
)
PYTHON = r"D:\sccad\python3.7\python.exe"

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import method_archer_final as final  # noqa: E402
import run_archer_core_20260724 as frozen  # noqa: E402
import run_archer_v2_evaluation as evaluation  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_tree_rss_mb(process):
    total = 0
    try:
        total += process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return total / (1024.0 * 1024.0)


class MemoryMonitor(threading.Thread):
    def __init__(self, interval=0.02):
        super(MemoryMonitor, self).__init__()
        self.daemon = True
        self.interval = float(interval)
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.peak_mb = process_tree_rss_mb(self.process)

    def run(self):
        while not self.stop_event.is_set():
            self.peak_mb = max(self.peak_mb, process_tree_rss_mb(self.process))
            self.stop_event.wait(self.interval)
        self.peak_mb = max(self.peak_mb, process_tree_rss_mb(self.process))

    def stop(self):
        self.stop_event.set()


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    ig.set_random_number_generator(random.Random(int(seed)))


def worker(dataset, seed, expected_n_pred):
    counts, _, _, already_log, source_path = evaluation.load_full_dataset(
        dataset, source="h5ad"
    )
    model_input = evaluation.strip_observation_metadata(counts)
    n_cells, n_genes = model_input.shape

    params = frozen.effective_params()
    assertions = {
        "NEIGHBOR_BACKEND": params["NEIGHBOR_BACKEND"],
        "HNSW_ENABLED": bool(params["HNSW_ENABLED"]),
        "NEIGHBOR_N_JOBS": int(params["NEIGHBOR_N_JOBS"]),
        "AKNN_KMAX": int(params["AKNN_KMAX"]),
        "AKNN_PRUNE_SNN": float(params["AKNN_PRUNE_SNN"]),
        "RF_HIG_TOPN": int(params["RF_HIG_TOPN"]),
    }
    if assertions != {
        "NEIGHBOR_BACKEND": "exact",
        "HNSW_ENABLED": False,
        "NEIGHBOR_N_JOBS": 1,
        "AKNN_KMAX": 20,
        "AKNN_PRUNE_SNN": 1.0 / 15.0,
        "RF_HIG_TOPN": 500,
    }:
        raise RuntimeError("Frozen exact configuration assertion failed: {}".format(assertions))

    set_seed(seed)
    process = psutil.Process(os.getpid())
    baseline_mb = process_tree_rss_mb(process)
    monitor = MemoryMonitor(interval=0.02)
    monitor.start()
    started = time.perf_counter()
    prediction, diagnostics = final.run_archer_final_pipeline(
        model_input,
        true_labels=None,
        already_log=already_log,
        params=params,
        random_seed=int(seed),
        return_details=True,
    )
    elapsed = time.perf_counter() - started
    monitor.stop()
    monitor.join(timeout=5.0)

    prediction = np.asarray(prediction)
    n_pred = int(len(np.unique(prediction)))
    row = {
        "dataset": dataset,
        "source_path": source_path,
        "seed": int(seed),
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "method_version": final.METHOD_VERSION,
        "scale_route": diagnostics.get("scale_route"),
        "init_neighbor_backend": diagnostics.get("init_neighbor_backend"),
        "aknno_neighbor_backend": diagnostics.get("aknno_neighbor_backend"),
        "hnsw_enabled": bool(params["HNSW_ENABLED"]),
        "neighbor_jobs": int(params["NEIGHBOR_N_JOBS"]),
        "elapsed_sec_monitored": float(elapsed),
        "baseline_rss_mb": float(baseline_mb),
        "peak_rss_mb": float(monitor.peak_mb),
        "delta_peak_mb": float(max(0.0, monitor.peak_mb - baseline_mb)),
        "n_pred_clusters": n_pred,
        "expected_n_pred_clusters": int(expected_n_pred),
        "prediction_cluster_count_matches_frozen": bool(n_pred == int(expected_n_pred)),
    }
    print(json.dumps(row, ensure_ascii=False, sort_keys=True))


def append_row(path, row):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parent(output_dir, timeout_sec):
    if os.path.exists(output_dir):
        raise RuntimeError("Refusing to reuse existing output directory: {}".format(output_dir))
    os.makedirs(output_dir)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir)
    output_csv = os.path.join(output_dir, "per_dataset_peak_rss.csv")

    frozen_rows = pd.read_csv(FROZEN_CSV)
    frozen_rows = frozen_rows[frozen_rows["method"].eq("archer_final")].copy()
    if len(frozen_rows) != 16 or frozen_rows["dataset"].nunique() != 16:
        raise RuntimeError("Expected exactly 16 frozen archer_final rows")

    for index, row in enumerate(frozen_rows.sort_values("dataset").itertuples(), 1):
        command = [
            PYTHON,
            "-u",
            os.path.abspath(__file__),
            "--worker",
            "--dataset",
            str(row.dataset),
            "--seed",
            str(int(row.seed)),
            "--expected-n-pred",
            str(int(row.n_pred_clusters)),
        ]
        print("[{}/16] {}".format(index, row.dataset), flush=True)
        result = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=int(timeout_sec),
        )
        with open(os.path.join(log_dir, row.dataset + ".stdout.log"), "w", encoding="utf-8") as handle:
            handle.write(result.stdout)
        with open(os.path.join(log_dir, row.dataset + ".stderr.log"), "w", encoding="utf-8") as handle:
            handle.write(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                "Peak-RSS run failed for {} (exit {}); see logs".format(
                    row.dataset, result.returncode
                )
            )
        payload = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
        if not payload:
            raise RuntimeError("No JSON result for {}".format(row.dataset))
        measured = json.loads(payload[-1])
        measured["frozen_runtime_seconds"] = float(row.runtime_seconds)
        append_row(output_csv, measured)
        print(
            "  peak={:.1f} MB; delta={:.1f} MB; monitored={:.2f} s".format(
                measured["peak_rss_mb"],
                measured["delta_peak_mb"],
                measured["elapsed_sec_monitored"],
            ),
            flush=True,
        )

    result_frame = pd.read_csv(output_csv)
    if not result_frame["prediction_cluster_count_matches_frozen"].astype(bool).all():
        raise RuntimeError("One or more reruns did not reproduce the frozen cluster count")
    if not result_frame["init_neighbor_backend"].eq("exact").all():
        raise RuntimeError("Initial neighbour backend was not exact in every run")
    if not result_frame["aknno_neighbor_backend"].eq("exact").all():
        raise RuntimeError("aKNNO neighbour backend was not exact in every run")
    if result_frame["hnsw_enabled"].astype(bool).any():
        raise RuntimeError("HNSW was enabled in at least one run")

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": "isolated peak-RSS rerun for frozen exact-neighbour Figure 4 panel d",
        "n_datasets": int(len(result_frame)),
        "source_frozen_benchmark": FROZEN_DIR,
        "source_frozen_csv_sha256": sha256_file(FROZEN_CSV),
        "worker_source": os.path.abspath(__file__),
        "worker_sha256": sha256_file(os.path.abspath(__file__)),
        "python": platform.python_version(),
        "backend_assertion": "exact in initial and aKNNO stages; HNSW disabled",
        "memory_definition": (
            "maximum sampled resident-set size of the isolated Python process and any children; "
            "20-ms sampling interval; dataset loaded before baseline; peak measured during ArCHeR call"
        ),
        "runtime_policy": (
            "Figure 4 runtime panel retains frozen benchmark runtime_seconds; monitored rerun time is audit-only"
        ),
        "all_cluster_counts_match_frozen": True,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print("complete: {}".format(output_dir), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--expected-n-pred", type=int)
    args = parser.parse_args()
    if args.worker:
        if args.dataset is None or args.seed is None or args.expected_n_pred is None:
            raise SystemExit("Worker requires dataset, seed and expected cluster count")
        worker(args.dataset, args.seed, args.expected_n_pred)
    else:
        parent(os.path.abspath(args.output_dir), args.timeout_sec)


if __name__ == "__main__":
    main()
