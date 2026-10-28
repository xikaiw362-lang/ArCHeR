"""Run the frozen, label-isolated ArCHeR case configuration on S_S_00005.

This wrapper consumes only the sparse count matrix, barcodes and gene names
exported by ``prepare_s_s_00005_case.R``.  The Seurat metadata file is not
opened by this program.  It therefore cannot influence feature selection,
graph construction, rare-cluster selection or cluster IDs.
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
from scipy.sparse import csc_matrix


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "case_study_datasets" / "S_S_00005" / "case_inputs" / "label_isolated_allcells_20260730"
DEFAULT_OUTPUT = ROOT / "case_study_datasets" / "S_S_00005" / "case_reruns" / "archer_hig500_adopted_exact_20260730"
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

import method_archer_final as archer  # noqa: E402


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def load_sparse_input(input_dir: Path):
    with (input_dir / "shape.i32").open("rb") as handle:
        shape = tuple(np.fromfile(handle, dtype="<i4", count=2).tolist())
    with (input_dir / "indptr.i32").open("rb") as handle:
        indptr = np.fromfile(handle, dtype="<i4")
    with (input_dir / "indices.i32").open("rb") as handle:
        indices = np.fromfile(handle, dtype="<i4")
    with (input_dir / "data.f32").open("rb") as handle:
        data = np.fromfile(handle, dtype="<f4")
    counts = csc_matrix((data, indices, indptr), shape=shape, dtype=np.float32)
    cells = pd.read_csv(input_dir / "cells.tsv", header=None)[0].astype(str).to_numpy()
    genes = pd.read_csv(input_dir / "genes.tsv", header=None)[0].astype(str).to_numpy()
    if counts.shape != (len(cells), len(genes)):
        raise ValueError("Sparse matrix shape does not agree with barcode/gene files")
    if len(set(cells)) != len(cells):
        raise ValueError("Input barcode file contains duplicate barcodes")
    return counts, cells, genes


def main(input_dir: Path, output_dir: Path) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to reuse tagged output directory: %s" % output_dir)
    counts, cells, genes = load_sparse_input(input_dir)
    adata = ad.AnnData(X=counts)
    adata.obs_names = pd.Index(cells, dtype="object")
    adata.var_names = pd.Index(genes, dtype="object")

    # These are the adopted HIG500 numerical settings.  Exact neighbours and
    # disabled HNSW make the execution path explicit and reproducible.
    overrides = {
        "RF_HIG_TOPN": 500,
        "AKNN_KMAX": 12,
        "AKNN_PRUNE_SNN": 0.05,
        "NEIGHBOR_BACKEND": "exact",
        "NEIGHBOR_N_JOBS": 1,
        "HNSW_ENABLED": False,
        "AKNN_RECORD_GRAPH_DIAGNOSTICS": True,
        "NEIGHBOR_TRACE": True,
    }
    started = time.time()
    labels, details = archer.run_archer_final_pipeline(
        adata,
        true_labels=None,
        already_log=False,
        params=overrides,
        return_details=True,
        random_seed=20260730,
    )
    elapsed = time.time() - started
    labels = np.asarray(labels, dtype=int)
    rare = np.asarray(details["rare_cell_mask"], dtype=bool)
    scores = np.asarray(details["rare_cell_score"], dtype=float)
    output_dir.mkdir(parents=True, exist_ok=False)

    prediction = pd.DataFrame({
        "barcode": cells,
        "archer_cluster": labels,
        "is_rare_cluster_by_size_rule": rare,
        "rare_cell_score": scores,
    })
    prediction.to_csv(output_dir / "predictions_hig500_adopted_exact.csv", index=False)
    np.save(output_dir / "predictions_hig500_adopted_exact.npy", labels)
    sizes = pd.Series(labels).value_counts().sort_index()
    sizes.rename_axis("archer_cluster").rename("n_cells").reset_index().assign(
        fraction=lambda table: table["n_cells"] / len(labels),
        is_rare_cluster_by_size_rule=lambda table: table["archer_cluster"].astype(str).isin(
            details["selected_rare_clusters"]
        ),
    ).to_csv(output_dir / "predicted_cluster_sizes.csv", index=False)

    manifest = {
        "analysis_tag": "archer_hig500_adopted_exact_20260730",
        "dataset": "S_S_00005",
        "input_dir": str(input_dir),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "runtime_seconds": float(elapsed),
        "n_predicted_clusters": int(len(sizes)),
        "n_rare_cells_by_size_rule": int(rare.sum()),
        "fixed_params": json_safe(archer.get_default_params(overrides)),
        "runtime_details": json_safe(details),
        "labels_used_in_method": False,
        "label_firewall": [
            "The runner reads only counts, barcodes and genes from the prepared input.",
            "The source metadata file is not opened before or during ArCHeR.",
            "No label, marker or ontology result is used to select parameters, clusters or IDs.",
        ],
    }
    (output_dir / "case_study_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "n_cells": manifest["n_cells"],
        "n_genes": manifest["n_genes"],
        "n_predicted_clusters": manifest["n_predicted_clusters"],
        "n_rare_cells_by_size_rule": manifest["n_rare_cells_by_size_rule"],
        "runtime_seconds": manifest["runtime_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
