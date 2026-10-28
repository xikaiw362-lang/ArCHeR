"""Candidate delivery entry point for scale-adaptive ArCHeR v3.2.

Small and medium datasets use independent DPC and kNN-Louvain pseudo-label
views and fuse their RF-HIG candidates with a fixed budget. Above a fixed,
label-free cell-count threshold, DPC is skipped and a fast kNN view supplies
HIGs. The same scale-aware neighbor engine is reused by the final aKNNO graph.
"""

import numpy as np

from method_archer_configurable import run_archer_pipeline_with_params


METHOD_NAME = "ArCHeR scale-adaptive capped-union HIG"
METHOD_VERSION = "3.2.0"
BASE_METHOD = "ArCHeR original"
FINAL_MODEL_STATUS = "CANDIDATE"
FINAL_MODEL_DATE = "2026-07-17"
FINAL_MODEL_RATIONALE = (
    "Fuse DPC/kNN HIG candidates by best per-view rank under a fixed HIG budget "
    "where DPC is tractable, "
    "but use a fixed label-free kNN-only route and approximate-neighbor graph "
    "above 10,000 cells to control runtime and memory without outcome-based "
    "dataset selection. Formal benchmark validation is still pending."
)
RARE_CLUSTER_MAX_FRACTION = 0.02

# Candidate configuration. Numerical overrides remain available for explicit
# sensitivity experiments; these defaults are fixed and never selected using
# ground-truth labels inside the method.
DEFAULT_PARAMS = {
    "HVG_MIN_MEAN": 0.0125,
    "HVG_MAX_MEAN": 3.0,
    "HVG_MIN_DISP": 0.5,
    "HVG_TOPN": 2000,
    "DPC_DC_PERCENT": 8,
    "DPC_GAMMA_TOPK": 8,
    "DPC_PCA_N_COMPONENTS": 40,
    "DPC_KNN_K": 200,
    "SCALE_ADAPTIVE_ENABLED": True,
    "LARGE_DATA_CELL_THRESHOLD": 10000,
    "LARGE_DATA_SPARSE_EMBEDDING": True,
    "NEIGHBOR_BACKEND": "auto",
    "NEIGHBOR_RANDOM_SEED": 42,
    "NEIGHBOR_N_JOBS": -1,
    "KDTREE_MAX_DIMS": 20,
    "APPROX_RERANK_MULTIPLIER": 3.0,
    "APPROX_RERANK_BATCH_SIZE": 256,
    "HNSW_M": 16,
    "HNSW_EF_CONSTRUCTION": 200,
    "HNSW_EF_SEARCH": 100,
    "HNSW_ENABLED": False,
    "RF_N_ESTIMATORS": 100,
    # Fixed final feature budget used by the HIG500 method and paper figures.
    "RF_HIG_TOPN": 500,
    "RF_PCA_DIMS": 50,
    "INIT_KNN_K": 20,
    "INIT_LOUVAIN_RES": 1.0,
    # Retain the published aKNNO neighbourhood defaults. The configurable
    # pipeline preserves its breakpoint rule with label-free connectivity
    # guards; the 30-PC cap below is the ArCHeR-specific regulariser.
    "AKNN_KMAX": 20,
    "AKNN_PRUNE_SNN": 1 / 15,
    # HIG augmentation adds a second feature view. Capping the graph
    # embedding at 30 PCs avoids low-variance PCs dominating SNN overlap,
    # while reducing graph construction cost.
    "AKNN_PCA_DIMS": list(range(1, 31)),
    "AKNN_EMBEDDING_DIMS": 30,
    "AKNN_CUTOFF": 5,
    "AKNN_MAX_ISOLATE_FRACTION": 0.01,
    "AKNN_MIN_LARGEST_COMPONENT_FRACTION": 0.90,
    "RESOLUTION_BASE": 0.28,
    "RESOLUTION_SCALE": 0.22,
    "RESOLUTION_MIN": 0.18,
    "RESOLUTION_MAX": 1.2,
}


def get_default_params(overrides=None):
    params = dict(DEFAULT_PARAMS)
    if overrides:
        params.update(overrides)
    return params


def calibrate_rare_partition(labels, threshold=RARE_CLUSTER_MAX_FRACTION):
    """Derive a label-free rare-cell mask from predicted cluster prevalence."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if len(labels) == 0:
        return {
            "rare_cell_mask": np.zeros(0, dtype=bool),
            "high_confidence_rare_cell_mask": np.zeros(0, dtype=bool),
            "rare_cell_score": np.zeros(0, dtype=float),
            "selected_rare_clusters": [],
            "operating_fraction": float(threshold),
            "crowded_candidate_regime": False,
            "rare_partition_rule": "predicted_cluster_fraction<=threshold",
        }

    values, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(values.tolist(), counts.astype(int).tolist()))
    fractions = {
        value: size / float(len(labels)) for value, size in cluster_sizes.items()
    }
    selected = [value for value in values if fractions[value] <= float(threshold)]
    rare_mask = np.isin(labels, selected)

    inverse_sizes = np.asarray(
        [1.0 / max(cluster_sizes[value], 1) for value in labels],
        dtype=float,
    )
    score = inverse_sizes / max(float(inverse_sizes.max()), 1e-12)
    return {
        "rare_cell_mask": rare_mask,
        "high_confidence_rare_cell_mask": rare_mask.copy(),
        "rare_cell_score": score,
        "selected_rare_clusters": [str(value) for value in selected],
        "selected_sizes": [cluster_sizes[value] for value in selected],
        "operating_fraction": float(threshold),
        "crowded_candidate_regime": False,
        "rare_partition_rule": "predicted_cluster_fraction<=threshold",
    }


def run_archer_final_pipeline(
    counts_df,
    true_labels=None,
    already_log=False,
    params=None,
    return_details=False,
    random_seed=None,
):
    """Run the scale-adaptive v3.1 candidate method.

    ``true_labels`` is accepted for API compatibility but deliberately ignored
    so ground truth cannot enter pseudo-label or HIG construction.
    """
    overrides = dict(params or {})
    if random_seed is None and "RANDOM_SEED" in overrides:
        random_seed = int(overrides.pop("RANDOM_SEED"))

    effective_params = get_default_params(overrides)
    result = run_archer_pipeline_with_params(
        counts_df,
        true_labels=None,
        already_log=already_log,
        params=effective_params,
        use_dpc=True,
        use_rf_hig=True,
        use_aknno=True,
        return_details=return_details,
        random_seed=random_seed,
        hig_mode="dpc_knn_capped_union",
    )
    if not return_details:
        return result

    labels, details = result
    labels = np.asarray(labels)
    rare = calibrate_rare_partition(labels)
    details = dict(details)
    details.update(rare)
    details.update(
        {
            "method_name": METHOD_NAME,
            "method_version": METHOD_VERSION,
            "base_method": BASE_METHOD,
            "hig_mode": "dpc_knn_capped_union",
            "effective_hig_mode": details.get(
                "effective_hig_mode", "dpc_knn_capped_union"
            ),
            "hig_fusion_rule": "best_per_view_rank_capped_at_RF_HIG_TOPN",
            "scale_adaptive_rule": "DPC+kNN if n_cells<=10000; kNN-only otherwise",
            "final_graph": "aKNNO",
            "label_firewall": "true_labels argument ignored",
        }
    )
    return labels, details


__all__ = [
    "METHOD_NAME",
    "METHOD_VERSION",
    "BASE_METHOD",
    "DEFAULT_PARAMS",
    "FINAL_MODEL_STATUS",
    "FINAL_MODEL_DATE",
    "FINAL_MODEL_RATIONALE",
    "RARE_CLUSTER_MAX_FRACTION",
    "get_default_params",
    "calibrate_rare_partition",
    "run_archer_final_pipeline",
]
