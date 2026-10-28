"""Run isolated ArCHeR v3.1 efficiency measurements with checkpoints."""

import argparse
import csv
import os
import subprocess

import pandas as pd


BASE = r"D:\sc_dataset"
PYTHON = r"D:\sccad\python3.7\python.exe"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(SCRIPT_DIR, "measure_archer_v31_efficiency.py")
DEFAULT_DATASETS = os.path.join(
    BASE, "experiments", "metrics", "full_metrics_16datasets_by_dataset.csv"
)
DEFAULT_OUTPUT = os.path.join(
    BASE, "experiments", "metrics", "archer_v31_efficiency_results_20260717.csv"
)
OUTPUT_COLUMNS = [
    "method_version",
    "dataset",
    "n_cells",
    "n_genes",
    "route",
    "backend",
    "neighbor_jobs",
    "elapsed_sec",
    "baseline_rss_mb",
    "peak_rss_mb",
    "delta_peak_mb",
    "n_pred",
]


def parse_key_values(text):
    values = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def append_row(path, row):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in OUTPUT_COLUMNS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-from", default=DEFAULT_DATASETS)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--prefer-h5ad", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--neighbor-jobs", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=1200)
    args = parser.parse_args()

    all_datasets = sorted(pd.read_csv(args.datasets_from)["dataset"].dropna().astype(str).unique())
    datasets = args.dataset if args.dataset else all_datasets
    missing = sorted(set(datasets) - set(all_datasets))
    if missing:
        raise ValueError("Unknown datasets: {}".format(missing))

    existing = set()
    if os.path.exists(args.out):
        with open(args.out, "r", newline="", encoding="utf-8") as handle:
            existing = {row["dataset"] for row in csv.DictReader(handle)}

    log_dir = os.path.splitext(args.out)[0] + "_logs"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    for index, dataset in enumerate(datasets, 1):
        if dataset in existing:
            print("[{}/{}] SKIP {}".format(index, len(datasets), dataset), flush=True)
            continue
        command = [
            PYTHON,
            "-u",
            WORKER,
            "--dataset",
            dataset,
            "--seed",
            str(args.seed),
            "--neighbor-jobs",
            str(args.neighbor_jobs),
        ]
        if args.prefer_h5ad:
            command.append("--prefer-h5ad")
        print("[{}/{}] RUN {}".format(index, len(datasets), dataset), flush=True)
        try:
            result = subprocess.run(
                command,
                cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=args.timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Efficiency run timed out for {}: {}".format(dataset, exc))

        stdout_path = os.path.join(log_dir, dataset + ".stdout.log")
        stderr_path = os.path.join(log_dir, dataset + ".stderr.log")
        with open(stdout_path, "w", encoding="utf-8") as handle:
            handle.write(result.stdout)
        with open(stderr_path, "w", encoding="utf-8") as handle:
            handle.write(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                "Efficiency run failed for {} (exit {}); see {}".format(
                    dataset, result.returncode, stderr_path
                )
            )
        row = parse_key_values(result.stdout)
        append_row(args.out, row)
        existing.add(dataset)
        print(
            "    time={}s peak={}MB delta={}MB".format(
                row.get("elapsed_sec"), row.get("peak_rss_mb"), row.get("delta_peak_mb")
            ),
            flush=True,
        )

    print("Efficiency measurements complete: {} datasets".format(len(existing)), flush=True)


if __name__ == "__main__":
    main()
