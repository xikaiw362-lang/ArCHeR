"""Measure one isolated ArCHeR v3.1 run for efficiency analysis."""

import argparse
import os
import random
import sys
import threading
import time

import igraph as ig
import numpy as np
import psutil


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import method_archer_final as final  # noqa: E402
import run_archer_v2_evaluation as evaluation  # noqa: E402


class MemoryMonitor(threading.Thread):
    def __init__(self, interval=0.05):
        super(MemoryMonitor, self).__init__()
        self.daemon = True
        self.interval = interval
        self.stop_event = threading.Event()
        self.process = psutil.Process(os.getpid())
        self.peak_mb = self.process.memory_info().rss / (1024.0 * 1024.0)

    def run(self):
        while not self.stop_event.is_set():
            try:
                rss_mb = self.process.memory_info().rss / (1024.0 * 1024.0)
                self.peak_mb = max(self.peak_mb, rss_mb)
            except psutil.Error:
                pass
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prefer-h5ad", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--neighbor-jobs", type=int, default=1)
    args = parser.parse_args()

    counts, true_labels, _, already_log, _ = evaluation.load_full_dataset(
        args.dataset,
        source="h5ad" if args.prefer_h5ad else "auto",
    )
    counts = evaluation.strip_observation_metadata(counts)
    n_cells, n_genes = counts.shape
    baseline_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)

    params = final.get_default_params({"NEIGHBOR_N_JOBS": args.neighbor_jobs})
    route = "knn_only_large" if n_cells > int(params["LARGE_DATA_CELL_THRESHOLD"]) else "dpc_knn_dual_view"
    backend = "hnsw" if route == "knn_only_large" else "exact"

    random.seed(args.seed)
    np.random.seed(args.seed)
    ig.set_random_number_generator(random.Random(args.seed))
    monitor = MemoryMonitor()
    monitor.start()
    start = time.perf_counter()
    pred = final.run_archer_final_pipeline(
        counts,
        true_labels=None,
        already_log=already_log,
        params=params,
        random_seed=args.seed,
    )
    elapsed = time.perf_counter() - start
    monitor.stop()
    monitor.join(timeout=2.0)

    if len(pred) != n_cells:
        raise ValueError("Prediction length mismatch: {} vs {}".format(len(pred), n_cells))

    print("method_version={}".format(final.METHOD_VERSION))
    print("dataset={}".format(args.dataset))
    print("n_cells={}".format(n_cells))
    print("n_genes={}".format(n_genes))
    print("route={}".format(route))
    print("backend={}".format(backend))
    print("neighbor_jobs={}".format(args.neighbor_jobs))
    print("elapsed_sec={:.6f}".format(elapsed))
    print("baseline_rss_mb={:.3f}".format(baseline_mb))
    print("peak_rss_mb={:.3f}".format(monitor.peak_mb))
    print("delta_peak_mb={:.3f}".format(max(0.0, monitor.peak_mb - baseline_mb)))
    print("n_pred={}".format(len(np.unique(pred))))


if __name__ == "__main__":
    main()
