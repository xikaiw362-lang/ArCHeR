"""Run the fixed label-free HIG500 ArCHeR workflow on zebrafish AGM.

Reference annotations are deliberately loaded nowhere in this execution script.
They are used only later for post-hoc case-study visualisation and evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmread


ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = ROOT / "case_study_datasets" / "D_R_00002" / "exported"
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

import method_archer_final as archer  # noqa: E402


def json_safe(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return str(value)


def main(tag: str, resolution_base: float | None = None) -> None:
    counts = mmread(EXPORT_DIR / "counts.mtx").tocsr().T.astype(np.float32)
    genes = pd.read_csv(EXPORT_DIR / "genes.csv")["gene"].astype(str).to_numpy()
    barcodes = pd.read_csv(EXPORT_DIR / "meta.csv")["barcode"].astype(str).to_numpy()
    if counts.shape != (len(barcodes), len(genes)):
        raise ValueError("Matrix shape does not match barcode/gene metadata")

    adata = ad.AnnData(X=counts)
    adata.obs_names = pd.Index(barcodes, dtype="object")
    adata.var_names = pd.Index(genes, dtype="object")
    started = time.time()
    overrides = {
        "AKNN_RECORD_GRAPH_DIAGNOSTICS": True,
        "NEIGHBOR_TRACE": True,
        "HNSW_ENABLED": False,
    }
    if resolution_base is not None:
        overrides["RESOLUTION_BASE"] = float(resolution_base)
    labels, details = archer.run_archer_final_pipeline(
        adata,
        true_labels=None,
        already_log=False,
        params=overrides,
        return_details=True,
        random_seed=2026,
    )
    elapsed = time.time() - started
    labels = np.asarray(labels, dtype=int)
    rare = np.asarray(details["rare_cell_mask"], dtype=bool)
    sizes = pd.Series(labels).value_counts().sort_index()
    pd.DataFrame(
        {
            "barcode": barcodes,
            "archer_cluster": labels,
            "is_rare_cluster_by_size_rule": rare,
            "rare_cell_score": np.asarray(details["rare_cell_score"], dtype=float),
        }
    ).to_csv(EXPORT_DIR / f"archer_predictions_{tag}.csv", index=False)
    sizes.rename_axis("archer_cluster").rename("n_cells").reset_index().assign(
        fraction=lambda table: table["n_cells"] / len(labels),
        is_rare_cluster_by_size_rule=lambda table: table["archer_cluster"].astype(str).isin(details["selected_rare_clusters"]),
    ).to_csv(EXPORT_DIR / f"archer_cluster_sizes_{tag}.csv", index=False)
    np.save(EXPORT_DIR / f"archer_predictions_{tag}.npy", labels)
    diagnostic = {
        "dataset": "D_R_00002 zebrafish AGM",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "runtime_seconds": elapsed,
        "n_predicted_clusters": int(len(sizes)),
        "n_rare_cells_by_size_rule": int(rare.sum()),
        "fixed_params": archer.DEFAULT_PARAMS,
        "runtime_details": details,
        "labels_used_in_method": False,
    }
    (EXPORT_DIR / f"archer_{tag}_diagnostics.json").write_text(
        json.dumps(json_safe(diagnostic), indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "tag": tag,
        "n_predicted_clusters": int(len(sizes)),
        "n_rare_cells_by_size_rule": int(rare.sum()),
        "runtime_seconds": elapsed,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="hig500_repaired")
    parser.add_argument(
        "--resolution-base",
        type=float,
        default=None,
        help="Optional fixed Louvain base resolution for a predeclared display run.",
    )
    arguments = parser.parse_args()
    main(arguments.tag, arguments.resolution_base)
