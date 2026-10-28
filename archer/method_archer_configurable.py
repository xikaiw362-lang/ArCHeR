import os
import sys
import time
import logging
import random
import importlib.util

# On the Windows/Python 3.7 runtime used for this project, loading hnswlib
# only after numpy/sklearn/igraph can trigger a native access violation.  Load
# it before the numerical stack so an available HNSW backend is usable rather
# than silently depending on an import-order accident at large scale.
try:
    import hnswlib as _hnswlib
except ImportError:
    _hnswlib = None

import numpy as np
import pandas as pd
import anndata as ad
import igraph as ig
from scipy.sparse import issparse, csr_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

logger = logging.getLogger("ArCHeR")

# Import all preprocessing and utility functions from original
from .method_archer import (
    normalize_total_inplace,
    log1p_inplace,
    mean_var,
    highly_variable_genes_inplace,
    scale_dense_matrix,
    compute_pca_embedding,
    compute_min_distance,
    remap_labels
)


def is_large_dataset(n_cells, p):
    """Return the fixed, label-free scale route used by adaptive ArCHeR."""
    return bool(
        p.get("SCALE_ADAPTIVE_ENABLED", False)
        and int(n_cells) > int(p.get("LARGE_DATA_CELL_THRESHOLD", 10000))
    )


def resolve_neighbor_backend(n_cells, n_dims, p, available_backends=None):
    """Choose a neighbor engine using only matrix shape and installed libraries.

    ``KDTree`` remains exact and is used only in genuinely low-dimensional
    spaces. HNSW is preferred for large PCA spaces; PyNNDescent is the
    dependency-light approximate fallback used by the current Python 3.7
    environment.
    """
    requested = str(p.get("NEIGHBOR_BACKEND", "exact")).strip().lower()
    aliases = {
        "sklearn": "exact",
        "brute": "exact",
        "kdtree": "kd_tree",
        "pynn": "pynndescent",
    }
    requested = aliases.get(requested, requested)
    valid = {"auto", "exact", "kd_tree", "hnsw", "pynndescent"}
    if requested not in valid:
        raise ValueError(
            "Unknown NEIGHBOR_BACKEND: {}. Expected one of {}".format(
                requested, sorted(valid)
            )
        )

    if available_backends is None:
        available_backends = {
            # The bundled Windows hnswlib binary must be explicitly validated
            # on the target runtime before it is allowed to run in-process.
            # A native access violation cannot be caught and recovered from.
            "hnsw": bool(p.get("HNSW_ENABLED", False)) and _hnswlib is not None,
            "pynndescent": importlib.util.find_spec("pynndescent") is not None,
        }
    else:
        available_backends = dict(available_backends)

    kd_tree_allowed = int(n_dims) <= int(p.get("KDTREE_MAX_DIMS", 20))
    large = is_large_dataset(n_cells, p)

    if requested == "auto":
        if large:
            if available_backends.get("hnsw", False):
                return "hnsw", requested, None
            if available_backends.get("pynndescent", False):
                reason = (
                    "hnsw_disabled" if _hnswlib is not None
                    else "hnsw_unavailable"
                )
                return "pynndescent", requested, reason
        return ("kd_tree" if kd_tree_allowed else "exact"), requested, (
            None if not large else "approximate_backend_unavailable"
        )

    if requested == "hnsw" and not available_backends.get("hnsw", False):
        if available_backends.get("pynndescent", False):
            reason = "hnsw_disabled" if _hnswlib is not None else "hnsw_unavailable"
            return "pynndescent", requested, reason
        return ("kd_tree" if kd_tree_allowed else "exact"), requested, (
            "hnsw_and_pynndescent_unavailable"
        )
    if requested == "pynndescent" and not available_backends.get(
        "pynndescent", False
    ):
        return ("kd_tree" if kd_tree_allowed else "exact"), requested, (
            "pynndescent_unavailable"
        )
    if requested == "kd_tree" and not kd_tree_allowed:
        return "exact", requested, "dimension_above_kdtree_limit"
    return requested, requested, None


def _exact_rerank_candidates(X, candidate_indices, k, batch_size=256):
    """Put self first and exactly rerank approximate candidates in PCA space."""
    X = np.asarray(X, dtype=np.float32)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    n_cells = X.shape[0]
    k = min(max(int(k), 1), max(n_cells - 1, 1))
    out_indices = np.empty((n_cells, k + 1), dtype=np.int64)
    out_distances = np.empty((n_cells, k + 1), dtype=np.float32)
    out_indices[:, 0] = np.arange(n_cells, dtype=np.int64)
    out_distances[:, 0] = 0.0

    for start in range(0, n_cells, max(int(batch_size), 1)):
        stop = min(start + max(int(batch_size), 1), n_cells)
        for local_row, cell_index in enumerate(range(start, stop)):
            candidates = candidate_indices[cell_index]
            candidates = candidates[
                (candidates >= 0)
                & (candidates < n_cells)
                & (candidates != cell_index)
            ]
            if len(candidates):
                _, first = np.unique(candidates, return_index=True)
                candidates = candidates[np.sort(first)]

            # Approximate engines should supply at least k candidates. The
            # exact fallback below keeps the contract safe if they do not.
            if len(candidates) < k:
                candidates = np.arange(n_cells, dtype=np.int64)
                candidates = candidates[candidates != cell_index]

            diff = X[candidates] - X[cell_index]
            dist_sq = np.einsum("ij,ij->i", diff, diff)
            if len(candidates) > k:
                chosen = np.argpartition(dist_sq, k - 1)[:k]
                chosen = chosen[np.argsort(dist_sq[chosen], kind="mergesort")]
            else:
                chosen = np.argsort(dist_sq, kind="mergesort")[:k]
            selected = candidates[chosen]
            out_indices[cell_index, 1:] = selected
            out_distances[cell_index, 1:] = np.sqrt(
                np.maximum(dist_sq[chosen], 0.0)
            ).astype(np.float32, copy=False)
    return out_distances, out_indices


def query_neighbors_with_params(X, k, p, purpose="neighbors"):
    """Return self + k neighbors through a reproducible backend abstraction."""
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    n_cells, n_dims = X.shape
    if n_cells < 2:
        raise ValueError("At least two cells are required for neighbor search")
    k = min(max(int(k), 1), n_cells - 1)
    backend, requested, fallback_reason = resolve_neighbor_backend(
        n_cells, n_dims, p
    )
    seed = int(p.get("_RUNTIME_RANDOM_SEED", p.get("NEIGHBOR_RANDOM_SEED", 42)))
    rerank_multiplier = max(float(p.get("APPROX_RERANK_MULTIPLIER", 3.0)), 1.0)
    candidate_k = min(n_cells - 1, max(k, int(np.ceil(k * rerank_multiplier))))
    start = time.time()
    if p.get("NEIGHBOR_TRACE", False):
        print(
            "[ArCHeR] neighbor search start: purpose={}, backend={}, n={}, dims={}, k={}".format(
                purpose, backend, n_cells, n_dims, k
            ),
            flush=True,
        )

    if backend in {"exact", "kd_tree"}:
        algorithm = "brute" if backend == "exact" else "kd_tree"
        nn = NearestNeighbors(
            n_neighbors=min(n_cells, k + 1),
            metric="euclidean",
            algorithm=algorithm,
            n_jobs=-1,
        )
        nn.fit(X)
        raw_indices = nn.kneighbors(X, return_distance=False)
        distances, indices = _exact_rerank_candidates(
            X,
            raw_indices,
            k,
            batch_size=p.get("APPROX_RERANK_BATCH_SIZE", 256),
        )
        candidate_k = k
    elif backend == "hnsw":
        if _hnswlib is None:
            raise RuntimeError("HNSW was selected but hnswlib is unavailable")
        index = _hnswlib.Index(space="l2", dim=n_dims)
        index.init_index(
            max_elements=n_cells,
            ef_construction=int(p.get("HNSW_EF_CONSTRUCTION", 200)),
            M=int(p.get("HNSW_M", 16)),
            random_seed=seed,
        )
        index.add_items(
            X,
            np.arange(n_cells, dtype=np.int64),
            num_threads=int(p.get("NEIGHBOR_N_JOBS", -1)),
        )
        index.set_ef(
            max(int(p.get("HNSW_EF_SEARCH", 100)), candidate_k + 1)
        )
        raw_indices, _ = index.knn_query(
            X,
            k=min(n_cells, candidate_k + 1),
            num_threads=int(p.get("NEIGHBOR_N_JOBS", -1)),
        )
        distances, indices = _exact_rerank_candidates(
            X,
            raw_indices,
            k,
            batch_size=p.get("APPROX_RERANK_BATCH_SIZE", 256),
        )
    else:
        from pynndescent import NNDescent

        index = NNDescent(
            X,
            n_neighbors=min(n_cells - 1, candidate_k + 1),
            metric="euclidean",
            random_state=seed,
            n_jobs=int(p.get("NEIGHBOR_N_JOBS", -1)),
            low_memory=True,
        )
        raw_indices, _ = index.neighbor_graph
        distances, indices = _exact_rerank_candidates(
            X,
            raw_indices,
            k,
            batch_size=p.get("APPROX_RERANK_BATCH_SIZE", 256),
        )

    diagnostics = {
        "neighbor_purpose": str(purpose),
        "neighbor_backend_requested": requested,
        "neighbor_backend": backend,
        "neighbor_backend_fallback": fallback_reason,
        "neighbor_n_cells": int(n_cells),
        "neighbor_n_dims": int(n_dims),
        "neighbor_k": int(k),
        "neighbor_candidate_k": int(candidate_k),
        "neighbor_exact_rerank": bool(backend in {"hnsw", "pynndescent"}),
        "neighbor_runtime_seconds": float(time.time() - start),
    }
    if p.get("NEIGHBOR_TRACE", False):
        print(
            "[ArCHeR] neighbor search complete: purpose={}, backend={}, seconds={:.2f}".format(
                purpose, backend, diagnostics["neighbor_runtime_seconds"]
            ),
            flush=True,
        )
    return distances, indices, diagnostics


def compute_graph_embedding_with_params(
    adata, feature_mask, n_components, p, purpose="graph"
):
    """Compute a dense PCA or memory-safe sparse SVD graph representation."""
    selected = adata[:, np.asarray(feature_mask, dtype=bool)].copy()
    n_components = min(
        int(n_components), selected.n_obs - 1, selected.n_vars - 1
    )
    if n_components < 1:
        raise ValueError("At least two cells and two selected genes are required")

    use_sparse = bool(
        is_large_dataset(selected.n_obs, p)
        and p.get("LARGE_DATA_SPARSE_EMBEDDING", True)
        and issparse(selected.X)
    )
    if use_sparse:
        X = selected.X.astype(np.float32)
        X = StandardScaler(with_mean=False, copy=True).fit_transform(X)
        embedding = TruncatedSVD(
            n_components=n_components,
            random_state=int(
                p.get("_RUNTIME_RANDOM_SEED", p.get("NEIGHBOR_RANDOM_SEED", 42))
            ),
        ).fit_transform(X)
        mode = "sparse_standardized_svd"
    else:
        X = selected.X
        if issparse(X):
            X = X.toarray()
        X = scale_dense_matrix(X, max_value=10)
        embedding = compute_pca_embedding(X, n_components)
        mode = "dense_scaled_pca"

    diagnostics = {
        "{}_embedding_mode".format(purpose): mode,
        "{}_embedding_n_components".format(purpose): int(n_components),
        "{}_embedding_n_features".format(purpose): int(selected.n_vars),
    }
    return np.asarray(embedding, dtype=np.float32), diagnostics


def run_archer_pipeline_with_params(
    counts_df,
    true_labels=None,
    already_log=False,
    params=None,
    use_dpc=True,
    use_rf_hig=True,
    use_aknno=True,
    init_mode=None,
    target_init_clusters=None,
    return_details=False,
    random_seed=None,
    hig_mode="single",
):
    """
    Configurable ArCHeR pipeline that accepts custom parameters.

    Args:
        counts_df: Input count matrix
        true_labels: Ground truth labels (optional)
        already_log: Whether data is already log-transformed
        params: Dictionary of parameters to override defaults
    """
    # Default parameters
    default_params = {
        # HVG parameters
        'HVG_MIN_MEAN': 0.0125,
        'HVG_MAX_MEAN': 3,
        'HVG_MIN_DISP': 0.5,
        'HVG_TOPN': 2000,

        # DPC parameters
        'DPC_DC_PERCENT': 8,
        'DPC_GAMMA_TOPK': 8,
        'DPC_PCA_N_COMPONENTS': 40,
        'DPC_KNN_K': 200,
        'DPC_FEATURE_MODE': 'hvg',
        'DPC_CENTER_MODE': 'global_gamma',
        'DPC_LOCAL_CENTER_TOPK': 3,
        'DPC_LOCAL_NEIGHBOR_K': 10,
        'DPC_LOCAL_SUPPORT_K': 5,
        'DPC_LOCAL_MUTUAL_MIN': 1,
        'DPC_LOCAL_RHO_MIN_RATIO': 1.0,
        'DPC_LOCAL_PEAK_QUANTILE': 0.75,
        'DPC_LOCAL_SCORE_QUANTILE': 0.90,
        'DPC_LOCAL_DEDUP_K': 5,
        'DPC_RARE_AWARE_TOPN': 500,
        'DPC_RARE_MIN_CELLS': 2,
        'DPC_RARE_MAX_PREVALENCE': 0.25,
        'COLLECT_QUALITY_DIAGNOSTICS': False,
        'QUALITY_SILHOUETTE_SAMPLE_SIZE': 2000,

        # Scale-adaptive execution. Disabled here for backward compatibility;
        # the final v3.1 entry point enables it explicitly.
        'SCALE_ADAPTIVE_ENABLED': False,
        'LARGE_DATA_CELL_THRESHOLD': 10000,
        'LARGE_DATA_SPARSE_EMBEDDING': True,

        # Unified neighbor search. ``auto`` selects HNSW on large datasets,
        # then PyNNDescent, with exact/KDTree as deterministic fallbacks.
        'NEIGHBOR_BACKEND': 'exact',
        'NEIGHBOR_RANDOM_SEED': 42,
        'NEIGHBOR_N_JOBS': -1,
        'KDTREE_MAX_DIMS': 20,
        'APPROX_RERANK_MULTIPLIER': 3.0,
        'APPROX_RERANK_BATCH_SIZE': 256,
        'HNSW_M': 16,
        'HNSW_EF_CONSTRUCTION': 200,
        'HNSW_EF_SEARCH': 100,
        # Enable only after the installed native wheel passes the target-data
        # smoke test; see resolve_neighbor_backend for the safety rationale.
        'HNSW_ENABLED': False,

        # RF parameters
        'RF_N_ESTIMATORS': 100,
        'RF_HIG_TOPN': 2000,
        'RF_PCA_DIMS': 50,
        'RF_OOB_DIAGNOSTICS': False,

        # KNN/Louvain fallback parameters
        'INIT_KNN_K': 20,
        'INIT_LOUVAIN_RES': 1.0,
        'POST_KNN_K': 20,
        'POST_LOUVAIN_RES': 1.0,

        # Alternative pseudo-label initializers
        'DBSCAN_MIN_SAMPLES_GRID': [2, 3, 5],
        'DBSCAN_EPS_QUANTILES': [80, 85, 90, 95],
        'DBSCAN_MAX_NOISE_FRAC': 0.50,
        'OPTICS_MIN_SAMPLES': 2,
        'OPTICS_XI': 0.05,
        'OPTICS_MIN_CLUSTER_SIZE': 2,

        # Retain the published aKNNO neighbourhood defaults. The optimiser
        # preserves the original breakpoint rule with label-free connectivity
        # guards; the 30-PC cap below is the ArCHeR-specific regulariser.
        'AKNN_KMAX': 20,
        'AKNN_PRUNE_SNN': 1/15,
        'AKNN_GRAPH_MODE': 'default',
        # HIG augmentation adds a second feature view. Capping the graph
        # embedding at 30 PCs avoids low-variance PCs dominating SNN overlap,
        # while reducing graph construction cost.
        'AKNN_PCA_DIMS': list(range(1, 31)),
        'AKNN_EMBEDDING_DIMS': 30,
        'AKNN_DELTA_RANGE': np.arange(-1.0, 0.1, 0.1),
        'AKNN_CUTOFF': 5,
        # A candidate aKNNO graph must not be dominated by isolated cells.
        # These are graph-quality safeguards, not label-derived tuning knobs.
        'AKNN_MAX_ISOLATE_FRACTION': 0.01,
        'AKNN_MIN_LARGEST_COMPONENT_FRACTION': 0.90,

        # Resolution parameters
        'RESOLUTION_BASE': 0.28,
        'RESOLUTION_SCALE': 0.22,
        'RESOLUTION_MIN': 0.18,
        'RESOLUTION_MAX': 1.2,
    }

    # Merge with provided params
    if params:
        default_params.update(params)
    p = default_params
    p['_RUNTIME_RANDOM_SEED'] = int(
        random_seed if random_seed is not None else p.get('NEIGHBOR_RANDOM_SEED', 42)
    )

    # igraph community detection is stochastic. New diagnostic experiments
    # pass a per-subset seed so paired variants share reproducible randomness.
    if random_seed is not None:
        ig.set_random_number_generator(random.Random(int(random_seed)))

    # Prepare AnnData
    if isinstance(counts_df, ad.AnnData):
        adata = counts_df
        if not issparse(adata.X):
            adata.X = csr_matrix(adata.X.astype(np.float32))
    elif issparse(counts_df):
        adata = ad.AnnData(X=counts_df)
    elif isinstance(counts_df, tuple) and len(counts_df) == 3:
        X, obs_names, var_names = counts_df
        adata = ad.AnnData(X=X)
        if obs_names is not None:
            adata.obs_names = pd.Index(obs_names).astype(str)
        if var_names is not None:
            adata.var_names = pd.Index(var_names).astype(str)
    else:
        adata = ad.AnnData(X=csr_matrix(counts_df.values, dtype=np.float32))
        adata.obs_names = counts_df.index.astype(str)
        adata.var_names = counts_df.columns.astype(str)

    if true_labels is not None:
        adata.obs['true_label'] = true_labels

    # Preprocessing
    if not already_log:
        normalize_total_inplace(adata, target_sum=1e4)
        log1p_inplace(adata)

    highly_variable_genes_inplace(adata,
                                  min_mean=p['HVG_MIN_MEAN'],
                                  max_mean=p['HVG_MAX_MEAN'],
                                  min_disp=p['HVG_MIN_DISP'],
                                  n_top_genes=p['HVG_TOPN'])

    hig_mode = str(hig_mode or "single").lower()
    if hig_mode not in {
        "single",
        "dpc_knn_consensus",
        "dpc_knn_union",
        "dpc_knn_capped_union",
    }:
        raise ValueError(f"Unknown hig_mode: {hig_mode}")
    if hig_mode.startswith("dpc_knn_") and not use_rf_hig:
        raise ValueError(f"{hig_mode} requires use_rf_hig=True")

    if hig_mode.startswith("dpc_knn_"):
        large_route = is_large_dataset(adata.n_obs, p)
        scale_diagnostics = {
            "scale_adaptive_enabled": bool(p.get("SCALE_ADAPTIVE_ENABLED", False)),
            "scale_route": (
                "knn_only_large" if large_route else "dpc_knn_dual_view"
            ),
            "scale_route_n_cells": int(adata.n_obs),
            "scale_route_threshold": int(
                p.get("LARGE_DATA_CELL_THRESHOLD", 10000)
            ),
            "requested_hig_mode": hig_mode,
            "effective_hig_mode": "knn_only" if large_route else hig_mode,
            "scale_route_uses_labels": False,
        }

        if large_route:
            # DPC is deliberately skipped above the fixed cell-count threshold.
            # The kNN partition supplies the single RF-HIG view, avoiding the
            # density-order fallback whose worst case approaches O(n^2).
            knn_view = run_knn_louvain_init_with_params(
                adata.copy(), p, target_n_clusters=target_init_clusters
            )
            adata.obs["initial_cluster"] = knn_view.obs[
                "initial_cluster"
            ].values
            adata = run_rf_hig_with_params(adata, p)
            adata.uns["initializer_diagnostics"] = dict(
                knn_view.uns.get("initializer_diagnostics", {})
            )
            adata.uns["hig_consensus_diagnostics"] = {
                "hig_consensus_rule": "knn_only_scale_adaptive",
                "hig_view_topn": int(min(p['RF_HIG_TOPN'], adata.n_vars)),
                "hig_consensus_count": int(
                    np.asarray(adata.var["is_HIG"], dtype=bool).sum()
                ),
                "hig_dpc_skipped": True,
            }
            init_mode = "knn_louvain_scale_adaptive"
        else:
            # Two independent label-free pseudo-label views are retained for
            # small and medium datasets, followed by the requested HIG fusion.
            dpc_view = run_dpc_with_params(adata.copy(), p)
            knn_view = run_knn_louvain_init_with_params(
                adata.copy(), p, target_n_clusters=target_init_clusters
            )
            if hig_mode == "dpc_knn_consensus":
                adata = run_consensus_rf_hig_with_params(
                    adata,
                    p,
                    dpc_view.obs["initial_cluster"].values,
                    knn_view.obs["initial_cluster"].values,
                )
            elif hig_mode == "dpc_knn_union":
                adata = run_union_rf_hig_with_params(
                    adata,
                    p,
                    dpc_view.obs["initial_cluster"].values,
                    knn_view.obs["initial_cluster"].values,
                )
            else:
                adata = run_capped_union_rf_hig_with_params(
                    adata,
                    p,
                    dpc_view.obs["initial_cluster"].values,
                    knn_view.obs["initial_cluster"].values,
                )
            # The final aKNNO step does not consume initial_cluster, but the
            # kNN partition keeps return_details diagnostics well defined.
            adata.obs["initial_cluster"] = knn_view.obs[
                "initial_cluster"
            ].values
            adata.uns["dpc_feature_diagnostics"] = dict(
                dpc_view.uns.get("dpc_feature_diagnostics", {})
            )
            adata.uns["initializer_diagnostics"] = dict(
                knn_view.uns.get("initializer_diagnostics", {})
            )
            init_mode = hig_mode
        adata.uns["scale_adaptive_diagnostics"] = scale_diagnostics
    else:
        if init_mode is None:
            if use_dpc:
                init_mode = "dpc"
            elif use_rf_hig:
                init_mode = "knn_louvain"
            else:
                init_mode = "single"

        if init_mode == "dpc":
            adata = run_dpc_with_params(adata, p)
        elif init_mode == "knn_louvain":
            adata = run_knn_louvain_init_with_params(
                adata, p, target_n_clusters=target_init_clusters
            )
        elif init_mode == "dbscan":
            adata = run_dbscan_init_with_params(adata, p)
        elif init_mode == "optics":
            adata = run_optics_init_with_params(adata, p)
        elif init_mode == "kmeans":
            adata = run_kmeans_init_with_params(adata, p)
        elif init_mode == "single":
            adata.obs["initial_cluster"] = np.zeros(adata.n_obs, dtype=int).astype(str)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        if use_rf_hig:
            adata = run_rf_hig_with_params(adata, p)
        else:
            adata.var["is_HIG"] = False

    if use_aknno:
        adata = run_aknno_with_params(adata, p, use_hig=use_rf_hig)
    else:
        adata = run_knn_louvain_post_with_params(adata, p, use_hig=use_rf_hig)

    pred_labels = adata.obs["D_cluster"].values
    if not return_details:
        return pred_labels

    initial_labels = adata.obs["initial_cluster"].astype(str).values
    initial_sizes = pd.Series(initial_labels).value_counts()
    diagnostics = {
        "init_mode": init_mode,
        "initial_labels": initial_labels,
        "initial_n_clusters": int(initial_sizes.size),
        "initial_n_singletons": int((initial_sizes == 1).sum()),
        "initial_singleton_cells": int(initial_sizes[initial_sizes == 1].sum()),
        "initial_singleton_fraction": float(
            initial_sizes[initial_sizes == 1].sum() / max(adata.n_obs, 1)
        ),
        "target_init_clusters": None if target_init_clusters is None else int(target_init_clusters),
        "final_n_clusters": int(pd.Series(pred_labels).nunique()),
    }
    dpc_feature_diagnostics = adata.uns.get("dpc_feature_diagnostics", {})
    diagnostics.update(dpc_feature_diagnostics)
    initializer_diagnostics = adata.uns.get("initializer_diagnostics", {})
    diagnostics.update(initializer_diagnostics)
    hig_consensus_diagnostics = adata.uns.get("hig_consensus_diagnostics", {})
    diagnostics.update(hig_consensus_diagnostics)
    scale_adaptive_diagnostics = adata.uns.get(
        "scale_adaptive_diagnostics", {}
    )
    diagnostics.update(scale_adaptive_diagnostics)
    aknno_diagnostics = adata.uns.get("aknno_diagnostics", {})
    diagnostics.update(aknno_diagnostics)
    post_diagnostics = adata.uns.get("post_diagnostics", {})
    diagnostics.update(post_diagnostics)
    if p.get('COLLECT_QUALITY_DIAGNOSTICS', False):
        quality_embedding = compute_standard_hvg_embedding(adata, p)
        diagnostics.update(
            compute_partition_quality(
                quality_embedding,
                initial_labels,
                prefix="initial",
                sample_size=p.get('QUALITY_SILHOUETTE_SAMPLE_SIZE', 2000),
            )
        )
        diagnostics.update(
            compute_partition_quality(
                quality_embedding,
                pred_labels,
                prefix="final",
                sample_size=p.get('QUALITY_SILHOUETTE_SAMPLE_SIZE', 2000),
            )
        )
    if "rf_oob_score" in adata.uns:
        diagnostics["rf_oob_score"] = float(adata.uns["rf_oob_score"])
    return pred_labels, diagnostics


def run_archer_recommended_pipeline_with_params(
    counts_df,
    true_labels=None,
    already_log=False,
    params=None,
    initializer="dpc",
    use_rf_hig=True,
    use_aknno=True,
    random_seed=None,
    return_details=False,
):
    """Recommended initializer-agnostic ArCHeR entry point.

    DPC is retained as the primary benchmark initializer. KNN-Louvain is the
    recommended rare-cell-focused alternative, while KMeans, DBSCAN, and
    OPTICS remain available for sensitivity analyses without changing the
    hybrid-gene or adaptive-graph stages.
    """
    valid_initializers = {"knn_louvain", "dpc", "kmeans", "dbscan", "optics"}
    if initializer not in valid_initializers:
        raise ValueError(
            f"Unknown initializer: {initializer}. "
            f"Expected one of {sorted(valid_initializers)}"
        )
    return run_archer_pipeline_with_params(
        counts_df,
        true_labels=true_labels,
        already_log=already_log,
        params=params,
        use_dpc=(initializer == "dpc"),
        use_rf_hig=use_rf_hig,
        use_aknno=use_aknno,
        init_mode=initializer,
        return_details=return_details,
        random_seed=random_seed,
    )


def run_archer_adaptive_dpc_with_params(
    counts_df,
    true_labels=None,
    already_log=False,
    params=None,
    silhouette_margin=0.10,
    rare_aware_topn=250,
    use_rf_hig=True,
    use_aknno=True,
    random_seed=None,
    return_details=False,
):
    """Run standard and two-view DPC, then select without using labels.

    Both candidates are evaluated on the same standard-HVG embedding. The
    two-view candidate is accepted only when its final silhouette exceeds the
    standard candidate by ``silhouette_margin``; otherwise the conservative
    standard DPC result is retained.
    """
    standard_params = dict(params or {})
    standard_params.update({
        "DPC_FEATURE_MODE": "hvg",
        "COLLECT_QUALITY_DIAGNOSTICS": True,
    })
    two_view_params = dict(params or {})
    two_view_params.update({
        "DPC_FEATURE_MODE": "hvg_plus_specificity_center_only",
        "DPC_RARE_AWARE_TOPN": int(rare_aware_topn),
        "COLLECT_QUALITY_DIAGNOSTICS": True,
    })

    def copy_input(value):
        if hasattr(value, "copy"):
            return value.copy()
        if isinstance(value, tuple) and len(value) == 3:
            X, obs_names, var_names = value
            return (
                X.copy() if hasattr(X, "copy") else X,
                obs_names,
                var_names,
            )
        return value

    standard_pred, standard_diag = run_archer_pipeline_with_params(
        copy_input(counts_df),
        true_labels=true_labels,
        already_log=already_log,
        params=standard_params,
        use_dpc=True,
        use_rf_hig=use_rf_hig,
        use_aknno=use_aknno,
        init_mode="dpc",
        return_details=True,
        random_seed=random_seed,
    )
    two_view_pred, two_view_diag = run_archer_pipeline_with_params(
        copy_input(counts_df),
        true_labels=true_labels,
        already_log=already_log,
        params=two_view_params,
        use_dpc=True,
        use_rf_hig=use_rf_hig,
        use_aknno=use_aknno,
        init_mode="dpc",
        return_details=True,
        random_seed=random_seed,
    )

    standard_silhouette = float(standard_diag["final_silhouette_hvg"])
    two_view_silhouette = float(two_view_diag["final_silhouette_hvg"])
    silhouette_delta = two_view_silhouette - standard_silhouette
    choose_two_view = (
        np.isfinite(silhouette_delta)
        and silhouette_delta >= float(silhouette_margin)
    )
    if choose_two_view:
        selected_pred = two_view_pred
        selected_diag = two_view_diag
        selected_initializer = "two_view_250"
    else:
        selected_pred = standard_pred
        selected_diag = standard_diag
        selected_initializer = "standard_hvg"

    if not return_details:
        return selected_pred
    details = dict(selected_diag)
    details.update({
        "selected_initializer": selected_initializer,
        "selection_metric": "final_silhouette_hvg",
        "selection_margin": float(silhouette_margin),
        "standard_final_silhouette_hvg": standard_silhouette,
        "two_view_final_silhouette_hvg": two_view_silhouette,
        "selection_silhouette_delta": float(silhouette_delta),
        "standard_final_n_clusters": standard_diag["final_n_clusters"],
        "two_view_final_n_clusters": two_view_diag["final_n_clusters"],
        "standard_pred_labels": standard_pred,
        "two_view_pred_labels": two_view_pred,
    })
    return selected_pred, details


def compute_standard_hvg_embedding(adata, p):
    """Compute a fixed, label-free HVG embedding for quality comparison."""
    base_mask = np.asarray(adata.var["highly_variable"], dtype=bool)
    adata_hvg = adata[:, base_mask].copy()
    X = adata_hvg.X
    if issparse(X):
        X = X.toarray()
    X = scale_dense_matrix(X, max_value=10)
    n_pcs = min(
        p['DPC_PCA_N_COMPONENTS'],
        adata_hvg.shape[0] - 1,
        adata_hvg.shape[1] - 1,
    )
    return compute_pca_embedding(X, n_pcs)


def compute_partition_quality(X, labels, prefix, sample_size=2000):
    """Return label-free partition quality metrics on a fixed embedding."""
    labels = np.asarray(labels).astype(str)
    sizes = pd.Series(labels).value_counts().to_numpy(dtype=float)
    n_clusters = len(sizes)
    n_cells = len(labels)
    proportions = sizes / max(float(n_cells), 1.0)
    if n_clusters > 1:
        entropy = float(
            -np.sum(proportions * np.log(np.maximum(proportions, 1e-12)))
            / np.log(float(n_clusters))
        )
    else:
        entropy = 0.0
    result = {
        f"{prefix}_size_entropy": entropy,
        f"{prefix}_min_cluster_fraction": float(proportions.min()),
        f"{prefix}_max_cluster_fraction": float(proportions.max()),
    }
    if n_clusters <= 1 or n_clusters >= n_cells:
        result.update({
            f"{prefix}_silhouette_hvg": np.nan,
            f"{prefix}_calinski_harabasz_hvg": np.nan,
            f"{prefix}_davies_bouldin_hvg": np.nan,
        })
        return result
    result.update({
        f"{prefix}_silhouette_hvg": float(
            silhouette_score(
                X,
                labels,
                sample_size=(
                    int(sample_size)
                    if sample_size and len(labels) > int(sample_size)
                    else None
                ),
                random_state=42,
            )
        ),
        f"{prefix}_calinski_harabasz_hvg": float(
            calinski_harabasz_score(X, labels)
        ),
        f"{prefix}_davies_bouldin_hvg": float(
            davies_bouldin_score(X, labels)
        ),
    })
    return result


def build_dpc_feature_mask(adata, p):
    """Build a DPC-only feature mask without changing downstream HVG/HIG masks.

    The rare-aware augmentation is label-free. It favors genes with at least a
    few expressing cells, low-to-moderate prevalence, and concentrated
    expression among positive cells. This is intended as an input-space
    diagnostic, not as a replacement for the main HVG selection.
    """
    base_mask = np.asarray(adata.var["highly_variable"], dtype=bool)
    mode = str(p.get("DPC_FEATURE_MODE", "hvg")).lower()
    if mode in {"hvg", "standard"}:
        return base_mask, {
            "dpc_feature_mode": "hvg",
            "dpc_n_features": int(base_mask.sum()),
            "dpc_n_added_features": 0,
        }
    if mode not in {
        "hvg_plus_specificity",
        "hvg_plus_rare",
        "hvg_plus_specificity_center_only",
    }:
        raise ValueError(f"Unknown DPC_FEATURE_MODE: {mode}")

    X = adata.X
    if issparse(X):
        n_cells = X.shape[0]
        nonzero = np.asarray(X.getnnz(axis=0)).ravel().astype(np.int64)
        sum_expr = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
        mean_expr = sum_expr / max(float(n_cells), 1.0)
        mean_square = (
            np.asarray(X.power(2).sum(axis=0)).ravel().astype(np.float64)
            / max(float(n_cells), 1.0)
        )
        variance = np.maximum(mean_square - np.square(mean_expr), 0.0)
        mean_positive = np.divide(sum_expr, np.maximum(nonzero, 1))
    else:
        X = np.asarray(X, dtype=np.float32)
        n_cells = X.shape[0]
        nonzero = (X > 0).sum(axis=0).astype(np.int64)
        mean_expr = X.mean(axis=0)
        variance = X.var(axis=0)
        mean_positive = np.divide(
            X.sum(axis=0),
            np.maximum(nonzero, 1),
        )
    prevalence = nonzero / max(float(n_cells), 1.0)
    dispersion = variance / np.maximum(mean_expr, 1e-8)

    min_cells = max(int(p.get("DPC_RARE_MIN_CELLS", 2)), 1)
    max_prevalence = float(p.get("DPC_RARE_MAX_PREVALENCE", 0.25))
    candidate = (
        (~base_mask)
        & (nonzero >= min_cells)
        & (prevalence <= max_prevalence)
        & (mean_positive > 0)
    )
    specificity_score = (
        np.log1p(mean_positive)
        * (1.0 - prevalence)
        * np.log1p(np.maximum(dispersion, 0.0))
    )
    specificity_score[~candidate] = -np.inf
    topn = max(int(p.get("DPC_RARE_AWARE_TOPN", 500)), 0)
    ranked = np.argsort(-specificity_score)
    added = ranked[np.isfinite(specificity_score[ranked])][:topn]
    dpc_mask = base_mask.copy()
    dpc_mask[added] = True
    feature_mode = (
        "hvg_plus_specificity_center_only"
        if mode == "hvg_plus_specificity_center_only"
        else "hvg_plus_specificity"
    )
    return dpc_mask, {
        "dpc_feature_mode": feature_mode,
        "dpc_n_features": int(dpc_mask.sum()),
        "dpc_n_added_features": int(len(added)),
        "dpc_candidate_features": int(candidate.sum()),
    }


def select_dpc_centers(X, rho, delta, nn_dists, nn_indices, p):
    """Select standard DPC peaks and optional locally supported peaks.

    Standard DPC ranks cells by the absolute-density score ``rho * delta``.
    That score can miss a small but internally coherent population because its
    absolute density is lower than that of abundant populations.  The
    ``global_plus_local`` mode therefore supplements the standard peaks with a
    small number of cells that are density peaks relative to their own local
    neighborhood.

    The local score is deliberately dimensionless: relative density and the
    distance to a higher-density cell are each normalized by a local reference.
    Mutual-neighbor support and graph-based center de-duplication reduce the
    chance that isolated cells or duplicate peaks from an abundant population
    are promoted.  Ground-truth labels are never consulted.
    """
    rho = np.asarray(rho, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    n_cells = len(rho)
    if n_cells == 0:
        raise ValueError("DPC center selection requires at least one cell.")

    gamma = rho * delta
    base_topk = min(max(int(p['DPC_GAMMA_TOPK']), 1), n_cells)
    base_centers = np.argsort(-gamma)[:base_topk].astype(int)
    mode = str(p.get('DPC_CENTER_MODE', 'global_gamma')).lower()

    diagnostics = {
        "dpc_center_mode": mode,
        "dpc_base_center_count": int(len(base_centers)),
        "dpc_local_candidate_count": 0,
        "dpc_local_center_count": 0,
        "dpc_total_center_count": int(len(base_centers)),
        "dpc_base_center_indices": base_centers.tolist(),
        "dpc_local_center_indices": [],
        "dpc_center_indices": base_centers.tolist(),
    }
    if mode in {'global_gamma', 'standard', 'global'}:
        return base_centers, diagnostics
    if mode not in {'global_plus_local', 'relative_density'}:
        raise ValueError("Unknown DPC_CENTER_MODE: {}".format(mode))

    available_k = max(int(nn_indices.shape[1]) - 1, 0)
    if available_k <= 0 or n_cells <= 1:
        return base_centers, diagnostics

    local_k = min(max(int(p.get('DPC_LOCAL_NEIGHBOR_K', 10)), 1), available_k)
    support_k = min(max(int(p.get('DPC_LOCAL_SUPPORT_K', 5)), 1), available_k)
    dedup_k = min(max(int(p.get('DPC_LOCAL_DEDUP_K', 5)), 0), available_k)

    local_neighbors = np.asarray(nn_indices[:, 1:local_k + 1], dtype=int)
    local_distances = np.asarray(nn_dists[:, 1:local_k + 1], dtype=np.float64)
    support_neighbors = np.asarray(nn_indices[:, 1:support_k + 1], dtype=int)

    eps = 1e-12
    local_rho_reference = np.median(rho[local_neighbors], axis=1)
    relative_rho = rho / np.maximum(local_rho_reference, eps)
    local_scale = np.median(local_distances, axis=1)
    relative_delta = delta / np.maximum(local_scale, eps)

    support_sets = [set(row.tolist()) for row in support_neighbors]
    mutual_support = np.asarray([
        sum(1 for neighbor in support_sets[i] if i in support_sets[neighbor])
        for i in range(n_cells)
    ], dtype=int)

    peak_quantile = float(p.get('DPC_LOCAL_PEAK_QUANTILE', 0.75))
    peak_quantile = float(np.clip(peak_quantile, 0.0, 1.0))
    local_peak_floor = np.quantile(rho[local_neighbors], peak_quantile, axis=1)
    support_factor = np.sqrt(
        (mutual_support.astype(np.float64) + 1.0) / (float(support_k) + 1.0)
    )
    local_score = (
        np.log1p(np.maximum(relative_rho, 0.0))
        * np.log1p(np.maximum(relative_delta, 0.0))
        * support_factor
    )

    eligible = (
        np.isfinite(local_score)
        & (rho >= local_peak_floor)
        & (relative_rho >= float(p.get('DPC_LOCAL_RHO_MIN_RATIO', 1.0)))
        & (mutual_support >= int(p.get('DPC_LOCAL_MUTUAL_MIN', 1)))
    )
    eligible[base_centers] = False

    score_quantile = float(p.get('DPC_LOCAL_SCORE_QUANTILE', 0.90))
    score_quantile = float(np.clip(score_quantile, 0.0, 1.0))
    if np.any(eligible):
        score_floor = float(np.quantile(local_score[eligible], score_quantile))
        eligible &= local_score >= score_floor
    else:
        score_floor = np.nan

    candidate_order = np.where(eligible)[0]
    candidate_order = candidate_order[np.argsort(-local_score[candidate_order])]
    diagnostics.update({
        "dpc_local_neighbor_k": int(local_k),
        "dpc_local_support_k": int(support_k),
        "dpc_local_score_floor": score_floor,
        "dpc_local_candidate_count": int(len(candidate_order)),
    })

    max_local_centers = max(int(p.get('DPC_LOCAL_CENTER_TOPK', 3)), 0)
    dedup_neighbors = (
        np.asarray(nn_indices[:, 1:dedup_k + 1], dtype=int)
        if dedup_k > 0
        else np.empty((n_cells, 0), dtype=int)
    )
    selected = base_centers.tolist()
    local_centers = []
    for candidate in candidate_order.tolist():
        duplicate = False
        if dedup_k > 0:
            candidate_neighbors = set(dedup_neighbors[candidate].tolist())
            for center in selected:
                if center in candidate_neighbors or candidate in dedup_neighbors[center]:
                    duplicate = True
                    break
        if duplicate:
            continue
        selected.append(int(candidate))
        local_centers.append(int(candidate))
        if len(local_centers) >= max_local_centers:
            break

    centers = np.asarray(selected, dtype=int)
    diagnostics.update({
        "dpc_local_center_count": int(len(local_centers)),
        "dpc_total_center_count": int(len(centers)),
        "dpc_local_center_indices": local_centers,
        "dpc_center_indices": centers.tolist(),
        "dpc_local_center_relative_rho": [
            float(relative_rho[index]) for index in local_centers
        ],
        "dpc_local_center_relative_delta": [
            float(relative_delta[index]) for index in local_centers
        ],
        "dpc_local_center_mutual_support": [
            int(mutual_support[index]) for index in local_centers
        ],
        "dpc_local_center_scores": [
            float(local_score[index]) for index in local_centers
        ],
    })
    return centers, diagnostics


def run_dpc_with_params(adata, p):
    dpc_mask, feature_diagnostics = build_dpc_feature_mask(adata, p)
    adata.uns["dpc_feature_diagnostics"] = feature_diagnostics
    X, embedding_diagnostics = compute_graph_embedding_with_params(
        adata,
        dpc_mask,
        p['DPC_PCA_N_COMPONENTS'],
        p,
        purpose="dpc",
    )
    adata.uns["dpc_feature_diagnostics"].update(embedding_diagnostics)

    k = min(p['DPC_KNN_K'], X.shape[0] - 1)
    nn_dists, nn_indices, neighbor_diagnostics = query_neighbors_with_params(
        X, k, p, purpose="dpc"
    )
    adata.uns["dpc_feature_diagnostics"].update(
        {"dpc_{}".format(key): value for key, value in neighbor_diagnostics.items()}
    )
    dc = max(
        float(np.percentile(nn_dists[:, 1:].ravel(), p['DPC_DC_PERCENT'])),
        float(np.finfo(np.float32).eps),
    )

    # Reuse the already-computed neighborhood. The legacy implementation
    # rebuilt a KDTree solely to query the same local distances a second time.
    # Including self plus k-1 neighbors preserves its density convention.
    density_dists = nn_dists[:, :k]
    rho = np.sum(np.exp(-(density_dists / dc) ** 2), axis=1).astype(np.float32)
    delta, nneigh = compute_min_distance(X, rho, nn_dists, nn_indices)

    # In the two-view variant, augmented features identify candidate density
    # peaks, while standard HVGs determine nearest-higher-density propagation.
    # This isolates peak detection from noisy assignment geometry.
    assignment_nneigh = nneigh
    if feature_diagnostics.get("dpc_feature_mode") == "hvg_plus_specificity_center_only":
        base_mask = np.asarray(adata.var["highly_variable"], dtype=bool)
        X_base, base_embedding_diagnostics = compute_graph_embedding_with_params(
            adata,
            base_mask,
            p['DPC_PCA_N_COMPONENTS'],
            p,
            purpose="dpc_assignment",
        )
        adata.uns["dpc_feature_diagnostics"].update(
            base_embedding_diagnostics
        )
        base_k = min(p['DPC_KNN_K'], X_base.shape[0] - 1)
        base_nn_dists, base_nn_indices, base_neighbor_diagnostics = (
            query_neighbors_with_params(
                X_base, base_k, p, purpose="dpc_assignment"
            )
        )
        adata.uns["dpc_feature_diagnostics"].update(
            {
                "dpc_assignment_{}".format(key): value
                for key, value in base_neighbor_diagnostics.items()
            }
        )
        _, assignment_nneigh = compute_min_distance(
            X_base, rho, base_nn_dists, base_nn_indices
        )

    gamma = rho * delta
    center_idx, center_diagnostics = select_dpc_centers(
        X, rho, delta, nn_dists, nn_indices, p
    )
    adata.uns["dpc_feature_diagnostics"].update(center_diagnostics)
    gamma_sorted = np.sort(gamma)[::-1]
    topk = min(int(p['DPC_GAMMA_TOPK']), len(gamma_sorted))
    if topk < len(gamma_sorted):
        gamma_margin = float(
            gamma_sorted[topk - 1] / max(float(gamma_sorted[topk]), 1e-12)
        )
    else:
        gamma_margin = np.nan
    adata.uns["dpc_feature_diagnostics"].update({
        "dpc_gamma_margin": gamma_margin,
        "dpc_gamma_cv": float(np.std(gamma) / max(float(np.mean(gamma)), 1e-12)),
        "dpc_rho_cv": float(np.std(rho) / max(float(np.mean(rho)), 1e-12)),
    })

    labels = -np.ones(adata.n_obs, dtype=int)
    for i, idx in enumerate(center_idx):
        labels[idx] = i
    for i in np.argsort(-rho):
        if labels[i] == -1:
            labels[i] = labels[assignment_nneigh[i]]

    adata.obs["initial_cluster"] = labels.astype(str)
    return adata


def compute_rf_hig_importance_with_params(adata, p, pseudo_labels):
    X_raw = adata.X
    y = np.asarray(pseudo_labels).astype(str)

    n_pca = min(p['RF_PCA_DIMS'], X_raw.shape[1] - 1, X_raw.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_pca, random_state=42)
    X_pca = svd.fit_transform(X_raw)

    rf = RandomForestClassifier(
        n_estimators=p['RF_N_ESTIMATORS'],
        random_state=42,
        n_jobs=-1,
        max_features='sqrt',
        oob_score=bool(p.get('RF_OOB_DIAGNOSTICS', False)),
    )
    rf.fit(X_pca, y)
    gene_importance = np.abs(svd.components_).T @ rf.feature_importances_
    oob_score = (
        float(rf.oob_score_)
        if p.get('RF_OOB_DIAGNOSTICS', False)
        else None
    )
    return gene_importance, oob_score


def run_rf_hig_with_params(adata, p):
    gene_importance, oob_score = compute_rf_hig_importance_with_params(
        adata, p, adata.obs["initial_cluster"].values
    )
    if oob_score is not None:
        adata.uns["rf_oob_score"] = oob_score
    top_idx = np.argsort(-gene_importance)[:p['RF_HIG_TOPN']]
    hig_names = adata.var_names[top_idx]
    adata.var["is_HIG"] = False
    adata.var.loc[hig_names, "is_HIG"] = True

    return adata


def run_consensus_rf_hig_with_params(adata, p, dpc_labels, knn_labels):
    """Keep RF-HIGs supported by both DPC and kNN pseudo-label views."""
    dpc_importance, dpc_oob = compute_rf_hig_importance_with_params(
        adata, p, dpc_labels
    )
    knn_importance, knn_oob = compute_rf_hig_importance_with_params(
        adata, p, knn_labels
    )
    topn = min(int(p['RF_HIG_TOPN']), adata.n_vars)
    dpc_top = np.argsort(-dpc_importance)[:topn]
    knn_top = np.argsort(-knn_importance)[:topn]
    consensus_idx = np.intersect1d(dpc_top, knn_top, assume_unique=True)

    adata.var["is_HIG"] = False
    if len(consensus_idx):
        adata.var.loc[adata.var_names[consensus_idx], "is_HIG"] = True

    union_count = len(np.union1d(dpc_top, knn_top))
    diagnostics = {
        "hig_consensus_rule": "intersection_of_topn",
        "hig_view_topn": int(topn),
        "hig_consensus_count": int(len(consensus_idx)),
        "hig_consensus_jaccard": float(
            len(consensus_idx) / max(union_count, 1)
        ),
    }
    if dpc_oob is not None:
        diagnostics["hig_dpc_rf_oob_score"] = dpc_oob
    if knn_oob is not None:
        diagnostics["hig_knn_rf_oob_score"] = knn_oob
    adata.uns["hig_consensus_diagnostics"] = diagnostics
    return adata


def run_union_rf_hig_with_params(adata, p, dpc_labels, knn_labels):
    """Keep the union of the two independent top-HIG pseudo-label views."""
    dpc_importance, dpc_oob = compute_rf_hig_importance_with_params(
        adata, p, dpc_labels
    )
    knn_importance, knn_oob = compute_rf_hig_importance_with_params(
        adata, p, knn_labels
    )
    topn = min(int(p['RF_HIG_TOPN']), adata.n_vars)
    dpc_top = np.argsort(-dpc_importance)[:topn]
    knn_top = np.argsort(-knn_importance)[:topn]
    union_idx = np.union1d(dpc_top, knn_top)

    adata.var["is_HIG"] = False
    if len(union_idx):
        adata.var.loc[adata.var_names[union_idx], "is_HIG"] = True

    intersection_count = len(np.intersect1d(dpc_top, knn_top, assume_unique=True))
    diagnostics = {
        "hig_consensus_rule": "union_of_topn",
        "hig_view_topn": int(topn),
        "hig_consensus_count": int(len(union_idx)),
        "hig_view_intersection_count": int(intersection_count),
        "hig_union_jaccard": float(
            intersection_count / max(len(union_idx), 1)
        ),
    }
    if dpc_oob is not None:
        diagnostics["hig_dpc_rf_oob_score"] = dpc_oob
    if knn_oob is not None:
        diagnostics["hig_knn_rf_oob_score"] = knn_oob
    adata.uns["hig_consensus_diagnostics"] = diagnostics
    return adata


def run_capped_union_rf_hig_with_params(adata, p, dpc_labels, knn_labels):
    """Fuse both top-HIG views while preserving the original HIG budget.

    Candidate genes are the union of the two per-view top-HIG sets. Genes are
    ordered by their best rank in either view, with mean rank as a deterministic
    agreement-aware tie breaker. The final set is capped at RF_HIG_TOPN, so the
    downstream feature count does not exceed the original ArCHeR HIG budget.
    """
    dpc_importance, dpc_oob = compute_rf_hig_importance_with_params(
        adata, p, dpc_labels
    )
    knn_importance, knn_oob = compute_rf_hig_importance_with_params(
        adata, p, knn_labels
    )
    n_genes = adata.n_vars
    topn = min(int(p['RF_HIG_TOPN']), n_genes)
    dpc_order = np.argsort(-dpc_importance)
    knn_order = np.argsort(-knn_importance)
    dpc_top = dpc_order[:topn]
    knn_top = knn_order[:topn]
    union_idx = np.union1d(dpc_top, knn_top)

    dpc_rank = np.empty(n_genes, dtype=np.int64)
    knn_rank = np.empty(n_genes, dtype=np.int64)
    dpc_rank[dpc_order] = np.arange(n_genes, dtype=np.int64)
    knn_rank[knn_order] = np.arange(n_genes, dtype=np.int64)
    best_rank = np.minimum(dpc_rank[union_idx], knn_rank[union_idx])
    mean_rank = (dpc_rank[union_idx] + knn_rank[union_idx]) / 2.0
    fused_order = np.lexsort((mean_rank, best_rank))
    selected_idx = union_idx[fused_order[:topn]]

    adata.var["is_HIG"] = False
    if len(selected_idx):
        adata.var.loc[adata.var_names[selected_idx], "is_HIG"] = True

    intersection_count = len(np.intersect1d(dpc_top, knn_top, assume_unique=True))
    diagnostics = {
        "hig_consensus_rule": "capped_union_best_rank",
        "hig_view_topn": int(topn),
        "hig_union_candidate_count": int(len(union_idx)),
        "hig_view_intersection_count": int(intersection_count),
        "hig_consensus_count": int(len(selected_idx)),
        "hig_union_jaccard": float(
            intersection_count / max(len(union_idx), 1)
        ),
    }
    if dpc_oob is not None:
        diagnostics["hig_dpc_rf_oob_score"] = dpc_oob
    if knn_oob is not None:
        diagnostics["hig_knn_rf_oob_score"] = knn_oob
    adata.uns["hig_consensus_diagnostics"] = diagnostics
    return adata


def _assign_density_noise(X, labels):
    """Assign density-clustering noise cells to the nearest cluster centroid."""
    labels = np.asarray(labels, dtype=int).copy()
    cluster_ids = sorted(int(x) for x in np.unique(labels) if x >= 0)
    if not cluster_ids:
        return labels
    noise_mask = labels < 0
    if not noise_mask.any():
        return remap_labels(labels)
    centroids = np.vstack([X[labels == cluster_id].mean(axis=0) for cluster_id in cluster_ids])
    nearest = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nearest.fit(centroids)
    assigned = nearest.kneighbors(X[noise_mask], return_distance=False).ravel()
    labels[noise_mask] = np.asarray(cluster_ids, dtype=int)[assigned]
    return remap_labels(labels)


def run_kmeans_init_with_params(adata, p):
    X = compute_standard_hvg_embedding(adata, p)
    n_clusters = min(max(int(p['DPC_GAMMA_TOPK']), 2), X.shape[0] - 1)
    labels = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,
    ).fit_predict(X)
    adata.obs["initial_cluster"] = remap_labels(labels).astype(str)
    adata.uns["initializer_diagnostics"] = {
        "initializer_name": "kmeans",
        "initializer_fallback": False,
        "initializer_pre_noise_clusters": int(n_clusters),
        "initializer_noise_fraction": 0.0,
    }
    return adata


def run_dbscan_init_with_params(adata, p):
    """Run label-free DBSCAN parameter selection on the standard HVG space."""
    X = compute_standard_hvg_embedding(adata, p)
    target_clusters = min(max(int(p['DPC_GAMMA_TOPK']), 2), X.shape[0] - 1)
    candidates = []
    for min_samples in p.get('DBSCAN_MIN_SAMPLES_GRID', [2, 3, 5]):
        min_samples = min(max(int(min_samples), 2), X.shape[0] - 1)
        nn = NearestNeighbors(n_neighbors=min_samples, metric="euclidean", n_jobs=-1)
        nn.fit(X)
        k_distances = nn.kneighbors(X, return_distance=True)[0][:, -1]
        for quantile in p.get('DBSCAN_EPS_QUANTILES', [80, 85, 90, 95]):
            eps = float(np.percentile(k_distances, float(quantile)))
            if not np.isfinite(eps) or eps <= 0:
                continue
            labels = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric="euclidean",
                n_jobs=-1,
            ).fit_predict(X)
            non_noise = labels >= 0
            cluster_ids = np.unique(labels[non_noise])
            n_clusters = len(cluster_ids)
            noise_fraction = float((~non_noise).mean())
            if n_clusters < 2 or non_noise.sum() <= n_clusters:
                continue
            sil = float(
                silhouette_score(
                    X[non_noise],
                    labels[non_noise],
                    sample_size=(2000 if non_noise.sum() > 2000 else None),
                    random_state=42,
                )
            )
            capacity_penalty = abs(n_clusters - target_clusters) / max(target_clusters, 1)
            score = sil - 0.35 * noise_fraction - 0.05 * capacity_penalty
            candidates.append({
                "score": score,
                "labels": labels,
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "noise_fraction": noise_fraction,
                "silhouette": sil,
            })

    if not candidates:
        adata = run_kmeans_init_with_params(adata, p)
        adata.uns["initializer_diagnostics"].update({
            "initializer_name": "dbscan_auto",
            "initializer_fallback": True,
            "dbscan_candidate_count": 0,
        })
        return adata

    best = max(candidates, key=lambda item: item["score"])
    labels = _assign_density_noise(X, best["labels"])
    adata.obs["initial_cluster"] = labels.astype(str)
    adata.uns["initializer_diagnostics"] = {
        "initializer_name": "dbscan_auto",
        "initializer_fallback": False,
        "initializer_pre_noise_clusters": int(best["n_clusters"]),
        "initializer_noise_fraction": float(best["noise_fraction"]),
        "initializer_internal_silhouette": float(best["silhouette"]),
        "dbscan_eps": float(best["eps"]),
        "dbscan_min_samples": int(best["min_samples"]),
        "dbscan_candidate_count": int(len(candidates)),
    }
    return adata


def run_optics_init_with_params(adata, p):
    X = compute_standard_hvg_embedding(adata, p)
    min_samples = min(
        max(int(p.get('OPTICS_MIN_SAMPLES', 2)), 2),
        X.shape[0] - 1,
    )
    min_cluster_size = min(
        max(int(p.get('OPTICS_MIN_CLUSTER_SIZE', 2)), 2),
        X.shape[0] - 1,
    )
    labels = OPTICS(
        min_samples=min_samples,
        xi=float(p.get('OPTICS_XI', 0.05)),
        min_cluster_size=min_cluster_size,
        cluster_method="xi",
        n_jobs=-1,
    ).fit_predict(X)
    non_noise = labels >= 0
    n_clusters = len(np.unique(labels[non_noise]))
    noise_fraction = float((~non_noise).mean())
    if n_clusters < 2:
        adata = run_kmeans_init_with_params(adata, p)
        adata.uns["initializer_diagnostics"].update({
            "initializer_name": "optics",
            "initializer_fallback": True,
            "initializer_noise_fraction": noise_fraction,
            "initializer_pre_noise_clusters": int(n_clusters),
        })
        return adata
    labels = _assign_density_noise(X, labels)
    adata.obs["initial_cluster"] = labels.astype(str)
    adata.uns["initializer_diagnostics"] = {
        "initializer_name": "optics",
        "initializer_fallback": False,
        "initializer_pre_noise_clusters": int(n_clusters),
        "initializer_noise_fraction": noise_fraction,
    }
    return adata


def run_knn_louvain_init_with_params(adata, p, target_n_clusters=None):
    hvg_mask = np.asarray(adata.var.highly_variable, dtype=bool)
    X, embedding_diagnostics = compute_graph_embedding_with_params(
        adata,
        hvg_mask,
        p['DPC_PCA_N_COMPONENTS'],
        p,
        purpose="init",
    )

    k = min(p['INIT_KNN_K'], max(2, X.shape[0] - 1))
    k = min(k, X.shape[0] - 1)
    nn_dists, nn_indices, neighbor_diagnostics = query_neighbors_with_params(
        X, k, p, purpose="init"
    )

    rows, cols, weights = [], [], []
    for i in range(X.shape[0]):
        neigh_ids = nn_indices[i, 1:]
        neigh_dists = nn_dists[i, 1:]
        for j, dist in zip(neigh_ids, neigh_dists):
            rows.append(i)
            cols.append(int(j))
            weights.append(float(1.0 / (1.0 + dist)))

    g = ig.Graph(n=X.shape[0], directed=False)
    g.add_edges(list(zip(rows, cols)))
    g.es["weight"] = weights
    g.simplify(combine_edges="max")
    def run_community(resolution):
        return g.community_multilevel(weights=g.es["weight"], resolution=float(resolution))

    if target_n_clusters is None:
        comm = run_community(p['INIT_LOUVAIN_RES'])
    else:
        # Match the capacity of the alternative initializer by selecting the
        # Louvain resolution whose community count is closest to DPC's count.
        # This is an unsupervised capacity match; labels are never used here.
        grid = p.get('INIT_MATCH_RES_GRID')
        if grid is None:
            grid = np.unique(np.concatenate([
                np.linspace(0.05, 3.0, 16),
                np.asarray([p['INIT_LOUVAIN_RES'] / 2.0,
                            p['INIT_LOUVAIN_RES'],
                            p['INIT_LOUVAIN_RES'] * 2.0])
            ]))
        candidates = []
        for resolution in sorted(float(x) for x in grid if float(x) > 0):
            candidate = run_community(resolution)
            n_clusters = len(candidate)
            distance = abs(n_clusters - int(target_n_clusters))
            base_distance = abs(resolution - float(p['INIT_LOUVAIN_RES']))
            candidates.append((distance, base_distance, resolution, candidate))
        _, _, _, comm = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    adata.obs["initial_cluster"] = np.asarray(comm.membership, dtype=int).astype(str)
    adata.uns["initializer_diagnostics"] = {
        "initializer_name": "knn_louvain",
        **embedding_diagnostics,
        **{
            "init_{}".format(key): value
            for key, value in neighbor_diagnostics.items()
        },
    }
    return adata


def find_neighbors_aknn_with_params(adata, delta, p, reduction="X_pca", precomputed_nn=None):
    pca_dims = [d - 1 for d in p['AKNN_PCA_DIMS'] if d - 1 < adata.obsm[reduction].shape[1]]
    X = adata.obsm[reduction][:, pca_dims]
    n_cells = X.shape[0]
    kmax = p['AKNN_KMAX']

    if precomputed_nn is not None:
        nn_dists, nn_indices = precomputed_nn
    else:
        nn_dists, nn_indices, neighbor_diagnostics = query_neighbors_with_params(
            X, kmax, p, purpose="aknno"
        )
        adata.uns.setdefault("aknno_diagnostics", {}).update(
            {
                "aknno_{}".format(key): value
                for key, value in neighbor_diagnostics.items()
            }
        )

    rows, cols = [], []
    for i in range(n_cells):
        dists = nn_dists[i, 1:]
        threshold = (np.sum(np.sqrt(dists)) / (kmax - 1 - delta)) ** 2
        valid = nn_indices[i, 1:][dists < threshold]
        rows.extend([i] * len(valid))
        cols.extend(valid.tolist())

    adj = csr_matrix((np.ones(len(rows), dtype=np.float32),
                      (rows, cols)), shape=(n_cells, n_cells))
    if p['AKNN_GRAPH_MODE'] == "symmetric":
        adj = adj.maximum(adj.T)
    elif p['AKNN_GRAPH_MODE'] == "mutual":
        adj = adj.minimum(adj.T)

    snn = adj.dot(adj.T).tocsr()
    row_sums = np.array(adj.sum(axis=1)).flatten()
    cx = snn.tocoo()
    union_vals = row_sums[cx.row] + row_sums[cx.col] - cx.data
    union_vals = np.where(union_vals == 0, 1.0, union_vals)
    snn_jaccard = csr_matrix((cx.data / union_vals, (cx.row, cx.col)), shape=snn.shape)
    snn_jaccard.data[snn_jaccard.data <= p['AKNN_PRUNE_SNN']] = 0.0
    snn_jaccard.eliminate_zeros()

    adata.obsp["aknn_connectivities"] = snn_jaccard
    return adata


def find_optimal_delta_with_params(adata, p, reduction="X_pca"):
    pca_dims = [d - 1 for d in p['AKNN_PCA_DIMS'] if d - 1 < adata.obsm[reduction].shape[1]]
    X = adata.obsm[reduction][:, pca_dims]

    nn_dists, nn_indices, neighbor_diagnostics = query_neighbors_with_params(
        X, p['AKNN_KMAX'], p, purpose="aknno"
    )
    precomputed_nn = (nn_dists, nn_indices)

    delta_metrics = {}
    for raw_delta in p['AKNN_DELTA_RANGE']:
        # np.arange can represent the intended terminal value 0 as a tiny
        # negative number.  Canonicalise scan values for deterministic logs
        # and transparent parameter reporting.
        delta = float(np.round(float(raw_delta), 8))
        if abs(delta) < 1e-12:
            delta = 0.0
        adata = find_neighbors_aknn_with_params(adata, delta=delta, p=p,
                                               reduction=reduction,
                                               precomputed_nn=precomputed_nn)
        snn = adata.obsp["aknn_connectivities"]
        src, tgt = snn.nonzero()
        g = ig.Graph(n=snn.shape[0], directed=False)
        g.add_edges(list(zip(src.tolist(), tgt.tolist())))
        g.es["weight"] = snn.data.tolist()
        comm = g.community_multilevel(weights=g.es["weight"], resolution=1.0)
        sizes = pd.Series(comm.membership).value_counts()
        components = g.components(mode="weak")
        component_sizes = np.asarray(components.sizes(), dtype=int)
        entry = {
            "n_communities": len(sizes),
            "n_singletons": int((sizes == 1).sum()),
            "n_edges": int(g.ecount()),
            "n_connected_components": int(len(component_sizes)),
            "n_isolated_vertices": int((component_sizes == 1).sum()),
            "largest_component_cells": (
                int(component_sizes.max()) if len(component_sizes) else 0
            ),
        }
        if p.get("AKNN_RECORD_GRAPH_DIAGNOSTICS", False):
            entry["isolate_fraction"] = float(
                entry["n_isolated_vertices"] / max(1, snn.shape[0])
            )
            entry["largest_component_fraction"] = float(
                entry["largest_component_cells"] / max(1, snn.shape[0])
            )
        delta_metrics[float(delta)] = entry

    delta_list = sorted(delta_metrics.keys())
    comm_list = [delta_metrics[d]["n_communities"] for d in delta_list]
    n_cells = max(1, adata.n_obs)
    max_isolates = int(np.floor(
        float(p.get("AKNN_MAX_ISOLATE_FRACTION", 0.01)) * n_cells
    ))
    min_largest_component = int(np.ceil(
        float(p.get("AKNN_MIN_LARGEST_COMPONENT_FRACTION", 0.90)) * n_cells
    ))
    eligible_idx = [
        idx for idx, delta in enumerate(delta_list)
        if delta_metrics[delta]["n_isolated_vertices"] <= max_isolates
        and delta_metrics[delta]["largest_component_cells"] >= min_largest_component
    ]

    # The original aKNNO optimizer selects the delta immediately after the
    # *last* rapid community-count change while scanning from -1 to 0.  Taking
    # the first plateau instead changes the intended operating point and can
    # produce an overly sparse, over-sensitive graph.
    rapid_idx = [
        idx for idx in range(1, len(delta_list))
        if abs(comm_list[idx] - comm_list[idx - 1]) > p['AKNN_CUTOFF']
    ]
    if rapid_idx:
        proposed_idx = max(rapid_idx)
        base_rule = "aKNNO breakpoint after last rapid community change"
    else:
        zero_singleton_idx = [
            idx for idx, delta in enumerate(delta_list)
            if delta_metrics[delta]["n_singletons"] == 0
        ]
        proposed_idx = zero_singleton_idx[0] if zero_singleton_idx else 0
        base_rule = "aKNNO first zero-singleton graph"

    if proposed_idx in eligible_idx:
        opt_idx = proposed_idx
        selection_rule = base_rule
    elif eligible_idx:
        # Keep the original over-clustering interpretation: if the proposed
        # sparse graph fails connectivity checks, move toward the next denser
        # eligible graph before considering any earlier delta.
        later_eligible = [idx for idx in eligible_idx if idx >= proposed_idx]
        opt_idx = later_eligible[0] if later_eligible else min(
            eligible_idx, key=lambda idx: abs(idx - proposed_idx)
        )
        selection_rule = base_rule + "; connectivity-safe fallback"
    else:
        opt_idx = min(
            range(len(delta_list)),
            key=lambda idx: (
                delta_metrics[delta_list[idx]]["n_isolated_vertices"],
                -delta_metrics[delta_list[idx]]["largest_component_cells"],
                idx,
            ),
        )
        selection_rule = base_rule + "; no graph met connectivity safeguards"
    optimal_delta = delta_list[opt_idx]

    if p.get("AKNN_RECORD_GRAPH_DIAGNOSTICS", False):
        adata.uns.setdefault("aknno_diagnostics", {})["aknno_delta_metrics"] = delta_metrics
        adata.uns["aknno_diagnostics"]["aknno_delta_selection_rule"] = selection_rule
        adata.uns["aknno_diagnostics"]["aknno_max_isolated_vertices"] = max_isolates
        adata.uns["aknno_diagnostics"]["aknno_min_largest_component_cells"] = (
            min_largest_component
        )

    return optimal_delta, precomputed_nn, neighbor_diagnostics


def run_multilevel_clustering(snn, resolution):
    src, tgt = snn.nonzero()
    g = ig.Graph(n=snn.shape[0], directed=False)
    g.add_edges(list(zip(src.tolist(), tgt.tolist())))
    g.es["weight"] = snn.data.tolist()
    comm = g.community_multilevel(weights=g.es["weight"], resolution=resolution)
    return np.array(comm.membership)


def run_aknno_with_params(adata, p, use_hig=True):
    if "is_HIG" not in adata.var.columns:
        adata.var["is_HIG"] = False
    merge_mask = adata.var.highly_variable | adata.var["is_HIG"] if use_hig else adata.var.highly_variable
    merge_num = int(np.asarray(merge_mask, dtype=bool).sum())
    if merge_num < 50:
        return adata

    n_pcs = min(
        int(p.get('AKNN_EMBEDDING_DIMS', 50)), merge_num - 1, adata.n_obs - 1
    )
    X_pca, embedding_diagnostics = compute_graph_embedding_with_params(
        adata,
        merge_mask,
        n_pcs,
        p,
        purpose="aknno",
    )
    adata_merge = ad.AnnData(X=csr_matrix((adata.n_obs, 1), dtype=np.float32))
    adata_merge.obsm["X_pca"] = X_pca
    adata_merge.uns["aknno_diagnostics"] = dict(embedding_diagnostics)

    optimal_delta, precomputed_nn, neighbor_diagnostics = (
        find_optimal_delta_with_params(adata_merge, p, reduction="X_pca")
    )
    adata_merge.uns["aknno_diagnostics"].update(
        {
            "aknno_{}".format(key): value
            for key, value in neighbor_diagnostics.items()
        }
    )
    adata_merge = find_neighbors_aknn_with_params(
        adata_merge, delta=optimal_delta, p=p,
        reduction="X_pca", precomputed_nn=precomputed_nn
    )
    snn = adata_merge.obsp["aknn_connectivities"]

    # Calculate resolution
    res = np.clip(
        p['RESOLUTION_BASE'] + np.log10(max(adata.n_obs / 5000, 1)) * p['RESOLUTION_SCALE'],
        p['RESOLUTION_MIN'],
        p['RESOLUTION_MAX']
    )
    best_labels = run_multilevel_clustering(snn, resolution=res)
    best_labels = remap_labels(best_labels)

    adata.obs["D_cluster"] = best_labels
    adata.uns["aknno_diagnostics"] = dict(
        adata_merge.uns.get("aknno_diagnostics", {})
    )
    adata.uns["aknno_diagnostics"]["aknno_optimal_delta"] = float(
        optimal_delta
    )
    return adata


def run_knn_louvain_post_with_params(adata, p, use_hig=True):
    if "is_HIG" not in adata.var.columns:
        adata.var["is_HIG"] = False
    merge_mask = adata.var.highly_variable | adata.var["is_HIG"] if use_hig else adata.var.highly_variable
    merge_num = int(np.asarray(merge_mask, dtype=bool).sum())
    if merge_num < 50:
        if "initial_cluster" not in adata.obs:
            raise ValueError("No initial clusters available for fallback post-clustering.")
        adata.obs["D_cluster"] = pd.to_numeric(
            adata.obs["initial_cluster"], errors="coerce"
        ).fillna(0).astype(int).values
        return adata

    n_pcs = min(
        int(p.get('AKNN_EMBEDDING_DIMS', 50)), merge_num - 1, adata.n_obs - 1
    )
    X_pca, embedding_diagnostics = compute_graph_embedding_with_params(
        adata,
        merge_mask,
        n_pcs,
        p,
        purpose="post",
    )

    k = min(p['POST_KNN_K'], max(2, X_pca.shape[0] - 1))
    k = min(k, X_pca.shape[0] - 1)
    nn_dists, nn_indices, neighbor_diagnostics = query_neighbors_with_params(
        X_pca, k, p, purpose="post"
    )

    rows, cols, weights = [], [], []
    for i in range(X_pca.shape[0]):
        neigh_ids = nn_indices[i, 1:]
        neigh_dists = nn_dists[i, 1:]
        for j, dist in zip(neigh_ids, neigh_dists):
            rows.append(i)
            cols.append(int(j))
            weights.append(float(1.0 / (1.0 + dist)))

    g = ig.Graph(n=X_pca.shape[0], directed=False)
    g.add_edges(list(zip(rows, cols)))
    g.es["weight"] = weights
    g.simplify(combine_edges="max")
    comm = g.community_multilevel(weights=g.es["weight"], resolution=p['POST_LOUVAIN_RES'])
    adata.obs["D_cluster"] = remap_labels(np.asarray(comm.membership, dtype=int))
    adata.uns["post_diagnostics"] = {
        **embedding_diagnostics,
        **{
            "post_{}".format(key): value
            for key, value in neighbor_diagnostics.items()
        },
    }
    return adata
