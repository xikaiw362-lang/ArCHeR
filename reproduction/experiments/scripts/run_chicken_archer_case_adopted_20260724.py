"""Run a traceable chicken case-study audit with the adopted HIG500 parameters.

This is intentionally written to a new case_reruns directory. It uses the
legacy HIG500 numerical settings (RF-HIG=500, Kmax=12, prune=0.05) with the
current engine's explicit exact-neighbor backend, so the manifest records that
it is not a byte-for-byte replay of the archived PyNNDescent run.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmread


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "case_study_datasets" / "G_G_00018"
DEFAULT_OUT_DIR = DATA_DIR / "case_reruns" / "archer_hig500_adopted_exact_20260724"
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

import method_archer_final as archer  # noqa: E402


def read_gzip_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


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
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return str(value)


def main(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to reuse non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    matrix_path = DATA_DIR / "GSE224329_final_matrix.mtx.gz.new"
    if not matrix_path.exists():
        matrix_path = DATA_DIR / "GSE224329_final_matrix.mtx.gz"
    with gzip.open(matrix_path, "rb") as handle:
        counts = mmread(handle).tocsr().T.astype(np.float32)
    barcodes = read_gzip_lines(DATA_DIR / "GSE224329_final_barcodes.tsv.gz")
    genes = read_gzip_lines(DATA_DIR / "GSE224329_final_feats.tsv.gz")
    if counts.shape != (len(barcodes), len(genes)):
        raise ValueError(f"Matrix shape {counts.shape} does not agree with barcode/gene files")

    adata = ad.AnnData(X=counts)
    adata.obs_names = pd.Index(barcodes, dtype="object")
    adata.var_names = pd.Index(genes, dtype="object")
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
        random_seed=2026,
    )
    elapsed = time.time() - started
    labels = np.asarray(labels, dtype=int)
    rare = np.asarray(details["rare_cell_mask"], dtype=bool)
    sizes = pd.Series(labels).value_counts().sort_index()
    prediction = pd.DataFrame(
        {
            "barcode": barcodes,
            "archer_cluster": labels,
            "is_rare_cluster_by_size_rule": rare,
            "rare_cell_score": np.asarray(details["rare_cell_score"], dtype=float),
        }
    )
    prediction.to_csv(output_dir / "predictions_hig500_legacy_params_exact.csv", index=False)
    sizes.rename_axis("archer_cluster").rename("n_cells").reset_index().assign(
        fraction=lambda table: table["n_cells"] / len(labels),
        is_rare_cluster_by_size_rule=lambda table: table["archer_cluster"].astype(str).isin(
            details["selected_rare_clusters"]
        ),
    ).to_csv(output_dir / "cluster_sizes_hig500_legacy_params_exact.csv", index=False)
    np.save(output_dir / "predictions_hig500_legacy_params_exact.npy", labels)

    archived_diag_path = DATA_DIR / "exported" / "archer_repaired_final_diagnostics.json"
    archived_prediction_path = DATA_DIR / "exported" / "archer_predictions_repaired_final.csv"
    archived_diag = {}
    if archived_diag_path.exists():
        archived_diag = json.loads(archived_diag_path.read_text(encoding="utf-8"))
    archived_params = archived_diag.get("fixed_params", {})
    archived_runtime = archived_diag.get("runtime_details", {})
    diagnostic = {
        "dataset": "GSE224329 chicken leukocytes",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "runtime_seconds": elapsed,
        "n_predicted_clusters": int(len(sizes)),
        "n_rare_cells_by_size_rule": int(rare.sum()),
        "fixed_params": json_safe(archer.get_default_params(overrides)),
        "runtime_details": json_safe(details),
        "labels_used_in_method": False,
        "case_study_status": "adopted legacy numerical settings on current engine with exact neighbors",
        "archived_reference": {
            "prediction_path": str(archived_prediction_path),
            "diagnostics_path": str(archived_diag_path),
            "effective_neighbor_backend": archived_runtime.get("aknno_neighbor_backend"),
            "requested_neighbor_backend": archived_params.get("NEIGHBOR_BACKEND"),
            "HNSW_ENABLED": archived_params.get("HNSW_ENABLED"),
            "RF_HIG_TOPN": archived_params.get("RF_HIG_TOPN"),
            "AKNN_KMAX": archived_params.get("AKNN_KMAX"),
            "AKNN_PRUNE_SNN": archived_params.get("AKNN_PRUNE_SNN"),
        },
        "interpretation_guard": [
            "No reference labels are loaded or passed to ArCHeR.",
            "Cluster IDs are retained exactly as emitted.",
            "The archived historical result used PyNNDescent effectively; it is not silently treated as an exact-neighbor control.",
            "No marker-driven parameter selection or cluster-ID remapping is applied.",
        ],
    }
    (output_dir / "case_study_manifest.json").write_text(
        json.dumps(json_safe(diagnostic), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_predicted_clusters": int(len(sizes)),
                "n_rare_cells_by_size_rule": int(rare.sum()),
                "runtime_seconds": float(elapsed),
                "effective_neighbor_backend": details.get("aknno_neighbor_backend"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    main(args.output_dir)
