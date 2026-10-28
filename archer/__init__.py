"""ArCHeR: scale-adaptive hybrid-gene extraction for rare-cell-aware single-cell clustering.

The modules in this package are the frozen v3.2.0 implementation used for the manuscript;
the only difference from the archived scripts in ``reproduction/`` is that their two
intra-package imports are relative.
"""
from .method_archer_final import (
    METHOD_VERSION as __version__,
    RARE_CLUSTER_MAX_FRACTION,
    calibrate_rare_partition,
    get_default_params,
    run_archer_final_pipeline,
)

__all__ = ["run_archer", "run_archer_final_pipeline", "get_default_params", "MANUSCRIPT_NEIGHBORS",
           "calibrate_rare_partition", "RARE_CLUSTER_MAX_FRACTION", "__version__"]

# Neighbour settings of every reported run (reproduction/experiments/scripts/run_archer_core_20260724.py).
# The module default NEIGHBOR_BACKEND="auto" may switch to approximate neighbours above 10,000 cells.
MANUSCRIPT_NEIGHBORS = {"NEIGHBOR_BACKEND": "exact", "HNSW_ENABLED": False, "NEIGHBOR_N_JOBS": 1}


def run_archer(adata, random_seed=None, already_log=False, params=None, return_details=False):
    """Cluster an AnnData object of raw counts (cells x genes) with ArCHeR.

    Uses the manuscript configuration (exact neighbours, HNSW off) unless ``params``
    overrides it, e.g. ``params={"NEIGHBOR_BACKEND": "auto"}`` for faster large runs.
    Reference labels are never read; pass an object whose ``obs`` holds no labels.
    Returns an integer label array, or ``(labels, details)`` when ``return_details`` is True.
    For results identical to the manuscript, seed Python, NumPy and igraph as in
    ``reproduction/experiments/scripts/run_archer_dual_view_16_benchmark.py`` (``set_seed``).
    """
    merged = dict(MANUSCRIPT_NEIGHBORS)
    merged.update(params or {})
    return run_archer_final_pipeline(adata, true_labels=None, already_log=already_log, params=merged,
                                     return_details=return_details, random_seed=random_seed)
