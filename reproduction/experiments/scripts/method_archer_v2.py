"""ArCHeR v2 research pipeline for rare-cell-aware unsupervised clustering.

The original ArCHeR pipeline is kept intact.  This module is a coherent research
variant that addresses six failure modes of rare-cell clustering:

1. absolute-density DPC can miss a small local density peak;
2. pseudo-label feature importance can be dominated by abundant groups;
3. count magnitude alone can hide shared low-frequency marker detection;
4. one global distance/neighborhood scale can blur a tiny population boundary;
5. modularity clustering can absorb a well-supported micro-population;
6. stochastic graph partitioning can create unsupported tiny fragments.

No ground-truth labels are used by the model.  ``true_labels`` is accepted only
for API compatibility and its length is validated, but it is not stored in the
AnnData object or consulted by any selection rule.
"""

import random
from copy import deepcopy

import anndata as ad
import igraph as ig
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse, triu
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from method_archer import (
    compute_min_distance,
    highly_variable_genes_inplace,
    log1p_inplace,
    normalize_total_inplace,
    remap_labels,
    scale_dense_matrix,
)
from method_archer_configurable import select_dpc_centers


METHOD_NAME = "ArCHeR v2"
METHOD_VERSION = "2.0.0-rc1"


DEFAULT_PARAMS = {
    # Preprocessing
    "HVG_MIN_MEAN": 0.0125,
    "HVG_MAX_MEAN": 3.0,
    "HVG_MIN_DISP": 0.5,
    "HVG_TOPN": 2000,
    "RANDOM_SEED": 2026,

    # Rare-aware feature view used only for DPC initialization
    "DPC_PCA_N_COMPONENTS": 40,
    "DPC_KNN_K": 200,
    "DPC_DC_PERCENT": 8.0,
    "DPC_DENSITY_MODE": "hybrid_local",
    "DPC_LOCAL_DENSITY_WEIGHT": 0.45,
    "DPC_LOCAL_DENSITY_K": 20,
    "DPC_LOCAL_DENSITY_SCALE_K": 7,
    "DPC_RARE_FEATURE_TOPN": 500,
    "DPC_BASE_CENTER_MODE": "adaptive_knee",
    "DPC_GAMMA_TOPK": 8,
    "DPC_MIN_BASE_CENTERS": 3,
    "DPC_MAX_BASE_CENTERS": 18,
    "DPC_CENTER_MODE": "global_plus_local",
    "DPC_LOCAL_CENTER_TOPK": 3,
    "DPC_LOCAL_NEIGHBOR_K": 10,
    "DPC_LOCAL_SUPPORT_K": 5,
    "DPC_LOCAL_MUTUAL_MIN": 1,
    "DPC_LOCAL_RHO_MIN_RATIO": 1.0,
    "DPC_LOCAL_PEAK_QUANTILE": 0.75,
    "DPC_LOCAL_SCORE_QUANTILE": 0.90,
    "DPC_LOCAL_DEDUP_K": 5,

    # Balanced hybrid-gene extraction
    "FEATURE_MIN_CLUSTER_SIZE": 2,
    "FEATURE_RARE_REFERENCE_FRAC": 0.10,
    "FEATURE_RARE_WEIGHT_CAP": 2.5,
    "RF_PCA_DIMS": 50,
    "RF_N_ESTIMATORS": 160,
    "ET_N_ESTIMATORS": 160,
    "HIG_TOPN": 1800,
    "RARE_GENE_TOPN": 600,
    "RARE_GENE_MAX_PREVALENCE": 0.35,
    "RARE_GENE_MIN_CELLS": 2,
    "HYBRID_RF_WEIGHT": 0.40,
    "HYBRID_CONTRAST_WEIGHT": 0.35,
    "HYBRID_SPECIFICITY_WEIGHT": 0.25,
    "REFINE_PREVIOUS_WEIGHT": 0.45,

    # Three partially whitened views
    "GLOBAL_PCA_DIMS": 30,
    "DISCRIM_PCA_DIMS": 30,
    "RARE_PCA_DIMS": 20,
    "EMBED_WHITENING_POWER": 0.50,
    "MIN_VIEW_FEATURES": 30,
    "RARE_IDF_POWER": 0.65,
    "RARE_SCORE_POWER": 0.75,
    "RARE_BINARY_WEIGHT": 0.40,

    # Gap-adaptive, locally scaled reciprocal-neighbor graph
    "GRAPH_KMAX": 30,
    "GRAPH_DEFAULT_K": 12,
    "GRAPH_MIN_K": 3,
    "GRAPH_SCALE_K": 7,
    "GRAPH_GAP_SEARCH_K": 12,
    "GRAPH_GAP_THRESHOLD": 1.40,
    "GRAPH_PAIR_GAP_THRESHOLD": 1.85,
    "GRAPH_MICRO_GAP_THRESHOLD": 1.70,
    "GRAPH_MICRO_MAX_K": 5,
    "GRAPH_RANK_POWER": 0.35,
    "GRAPH_ONE_WAY_PENALTY": 0.15,
    "GRAPH_SNN_WEIGHT": 0.40,
    "GRAPH_SNN_PRUNE": 0.04,
    "GRAPH_FUSED_TOPK": 24,
    "GRAPH_FUSED_MIN_TOPK": 12,
    "GRAPH_MAX_VIEW_BONUS": 0.20,
    "GLOBAL_VIEW_WEIGHT": 0.30,
    "DISCRIM_VIEW_WEIGHT": 0.40,
    "RARE_VIEW_WEIGHT": 0.30,

    # Stable micro-population protection
    "MICRO_CORE_K": 3,
    "MICRO_CORE_MIN_WEIGHT": 0.42,
    "MICRO_MAX_SIZE": 30,
    "MICRO_MAX_FRAC": 0.05,
    "MICRO_MIN_INTERNAL_MEAN": 0.40,
    "MICRO_MAX_CONDUCTANCE": 0.58,
    "MICRO_MIN_SEPARATION": 1.05,
    "MICRO_MAX_LOCAL_CONDUCTANCE": 0.62,
    "MICRO_MIN_EDGE_SEPARATION": 1.10,
    "MICRO_INTERNAL_BOOST": 2.25,
    "MICRO_EXTERNAL_SHRINK": 0.22,
    "MICRO_FORCE_SPLIT": True,

    # Gene-aware multi-view topology anchors.  Graph-only micro-components are
    # softly protected; exact forced splits additionally require local marker
    # coherence and separation in the rare/discriminative representations.
    "TOPOLOGY_ANCHOR_ENABLED": True,
    "TOPOLOGY_CORE_KS": [1, 2, 3],
    "TOPOLOGY_EDGE_MIN_WEIGHT": 0.28,
    "TOPOLOGY_MAX_SIZE": 30,
    "TOPOLOGY_MAX_FRAC": 0.05,
    "TOPOLOGY_MAX_CANDIDATES": 2000,
    "TOPOLOGY_MAX_ACCEPTED": 50,
    "TOPOLOGY_MIN_ACCEPTED": 1,
    "TOPOLOGY_MAX_ACCEPTED_RATIO": 0.25,
    "TOPOLOGY_REFINE_MAX_ACCEPTED_RATIO": 1.00,
    "TOPOLOGY_PARENT_MIN_PURITY": 0.80,
    "TOPOLOGY_MAINTENANCE_MIN_COVERAGE": 0.75,
    "TOPOLOGY_MAX_DISCOVERIES_PER_PARENT": 1,
    "TOPOLOGY_NESTED_EXPANSION_ENABLED": False,
    "TOPOLOGY_NESTED_MAX_ADDED": 1,
    "TOPOLOGY_NESTED_MIN_SOURCE_COUNT": 2,
    "TOPOLOGY_NESTED_MARKER_RETENTION": 0.80,
    "TOPOLOGY_DIAGNOSTIC_TOPN": 100,
    "TOPOLOGY_BACKGROUND_MIN": 12,
    "TOPOLOGY_BACKGROUND_MAX": 80,
    "TOPOLOGY_MARKER_TOPK": 5,
    "TOPOLOGY_MARKER_MIN_PREVALENCE": 0.50,
    "TOPOLOGY_MIN_INTERNAL_MEAN": 0.28,
    "TOPOLOGY_MAX_CONDUCTANCE": 0.68,
    "TOPOLOGY_MIN_SEPARATION": 0.90,
    "TOPOLOGY_MAX_LOCAL_CONDUCTANCE": 0.62,
    "TOPOLOGY_MIN_EDGE_SEPARATION": 1.10,
    "TOPOLOGY_MIN_ANCHOR_SCORE": 0.45,
    "TOPOLOGY_MIN_EMBED_SCORE": 0.03,
    "TOPOLOGY_MIN_MARKER_SCORE": 0.16,
    "TOPOLOGY_MIN_COMPOSITE": 0.48,
    "TOPOLOGY_PURIFY_ENABLED": True,
    "TOPOLOGY_PURIFY_MAX_SIZE": 6,
    "TOPOLOGY_PURIFY_MIN_COMPOSITE_GAIN": 0.04,
    "TOPOLOGY_PURIFY_MIN_EMBED_GAIN": 0.12,
    "TOPOLOGY_PURIFY_MIN_ANCHOR_GAIN": 0.08,
    "TOPOLOGY_PURIFY_MARKER_RETENTION": 0.85,
    "TOPOLOGY_PURIFY_BRIDGE_MAX_PARENT_EMBED": 0.05,
    "TOPOLOGY_PURIFY_BRIDGE_MIN_CHILD_EMBED": 0.00,
    "TOPOLOGY_PURIFY_BRIDGE_MIN_EMBED_GAIN": 0.05,
    "TOPOLOGY_PURIFY_BRIDGE_MIN_EMBED_MARGIN": 0.08,
    "TOPOLOGY_PURIFY_BRIDGE_MIN_ANCHOR": 0.45,
    "TOPOLOGY_PRIORITY_RARE_EMBED_WEIGHT": 0.08,
    "TOPOLOGY_PRIORITY_MARKER_PANEL_WEIGHT": 0.04,
    "TOPOLOGY_PRIORITY_MARKER_PANEL_SCALE": 30.0,

    # Heterogeneity-driven local decomposition for hidden rare populations
    "LOCAL_REFINE_ENABLED": True,
    "LOCAL_REFINE_MIN_PARENT_SIZE": 20,
    "LOCAL_REFINE_MAX_CHILD_SIZE": 30,
    "LOCAL_REFINE_MAX_CHILD_FRAC": 0.15,
    "LOCAL_REFINE_RESOLUTION_MULTIPLIERS": [1.50, 2.00, 3.00],
    "LOCAL_REFINE_SEEDS": 2,
    "LOCAL_REFINE_MIN_SUPPORT": 2,
    "LOCAL_REFINE_MATCH_JACCARD": 0.80,
    "LOCAL_REFINE_MIN_INTERNAL_MEAN": 0.32,
    "LOCAL_REFINE_MAX_CONDUCTANCE": 0.66,
    "LOCAL_REFINE_MIN_SEPARATION": 0.80,
    "LOCAL_REFINE_MAX_LOCAL_CONDUCTANCE": 0.64,
    "LOCAL_REFINE_MIN_EDGE_SEPARATION": 1.05,
    "LOCAL_REFINE_MIN_EMBED_SCORE": 0.10,
    "LOCAL_REFINE_MIN_MARKER_SCORE": 0.12,
    "LOCAL_REFINE_MAX_CHILDREN_PER_PARENT": 3,
    "LOCAL_REFINE_MAX_TOTAL_RATIO": 0.25,
    "LOCAL_REFINE_MIN_TOTAL": 1,

    # Conservative cleanup of unsupported stochastic fragments
    "CLEANUP_ENABLED": True,
    "CLEANUP_MAX_SIZE": 20,
    "CLEANUP_MAX_FRAC": 0.02,
    "CLEANUP_MIN_CONDUCTANCE": 0.80,
    "CLEANUP_MAX_MARKER_SCORE": 0.10,
    "CLEANUP_MAX_ITERATIONS": 2,

    # Multi-resolution consensus Leiden clustering
    "RESOLUTION_BASE": 0.28,
    "RESOLUTION_SCALE": 0.15,
    "RESOLUTION_MIN": 0.16,
    "RESOLUTION_MAX": 1.50,
    "CONSENSUS_RESOLUTION_MULTIPLIERS": [0.50, 0.60, 0.75, 1.00, 1.25, 1.75],
    "CONSENSUS_SEEDS": 2,
    "CONSENSUS_WEIGHT_FLOOR": 0.20,
    "CONSENSUS_SELECT_RESOLUTION": True,
    "CONSENSUS_TARGET_COUNT_WEIGHT": 0.35,
    "CONSENSUS_DEGENERATE_PENALTY": 0.50,
    "CONSENSUS_RESOLUTION_WINDOW_RATIO": 1.20,
    "LEIDEN_ITERATIONS": 4,

    # One conservative self-refinement round
    "REFINE_ITERATIONS": 1,
    "REFINE_MIN_PARTITION_ARI": 0.55,
    "REFINE_MIN_CLUSTER_COUNT_RATIO": 0.60,
    "REFINE_MAX_CLUSTER_COUNT_RATIO": 1.00,
}


def get_default_params(overrides=None):
    params = deepcopy(DEFAULT_PARAMS)
    if overrides:
        params.update(overrides)
    return params


def _prepare_adata(counts, already_log=False):
    if isinstance(counts, ad.AnnData):
        adata = counts.to_memory() if getattr(counts, "isbacked", False) else counts.copy()
    elif issparse(counts):
        adata = ad.AnnData(X=counts.copy())
    elif isinstance(counts, tuple) and len(counts) == 3:
        matrix, obs_names, var_names = counts
        adata = ad.AnnData(X=matrix.copy() if hasattr(matrix, "copy") else matrix)
        if obs_names is not None:
            adata.obs_names = pd.Index(obs_names).astype(str)
        if var_names is not None:
            adata.var_names = pd.Index(var_names).astype(str)
    else:
        values = np.asarray(counts.values, dtype=np.float32)
        adata = ad.AnnData(X=csr_matrix(values))
        adata.obs_names = pd.Index(counts.index).astype(str)
        adata.var_names = pd.Index(counts.columns).astype(str)

    if issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    if not already_log:
        normalize_total_inplace(adata, target_sum=1e4)
        log1p_inplace(adata)
    return adata


def _column_statistics(X):
    n_cells = int(X.shape[0])
    denominator = max(float(n_cells), 1.0)
    if issparse(X):
        X = X.tocsr()
        nonzero = np.asarray(X.getnnz(axis=0)).ravel().astype(np.int64)
        sums = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
        square_sums = np.asarray(X.power(2).sum(axis=0)).ravel().astype(np.float64)
    else:
        dense = np.asarray(X, dtype=np.float64)
        nonzero = np.sum(dense > 0, axis=0).astype(np.int64)
        sums = dense.sum(axis=0)
        square_sums = np.square(dense).sum(axis=0)
    mean = sums / denominator
    variance = np.maximum(square_sums / denominator - np.square(mean), 0.0)
    prevalence = nonzero / denominator
    positive_mean = np.divide(sums, np.maximum(nonzero, 1))
    return {
        "nonzero": nonzero,
        "sum": sums,
        "square_sum": square_sums,
        "mean": mean,
        "variance": variance,
        "prevalence": prevalence,
        "positive_mean": positive_mean,
    }


def _rank01(values, eligible=None):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if eligible is not None:
        finite &= np.asarray(eligible, dtype=bool)
    output = np.zeros(len(values), dtype=np.float64)
    indices = np.where(finite)[0]
    if len(indices) == 0:
        return output
    if len(indices) == 1:
        output[indices] = 1.0
        return output
    ranks = rankdata(values[indices], method="average")
    output[indices] = (ranks - 1.0) / float(len(indices) - 1)
    return output


def compute_rare_gene_specificity(adata, p):
    stats = _column_statistics(adata.X)
    dispersion = stats["variance"] / np.maximum(stats["mean"], 1e-8)
    eligible = (
        (stats["nonzero"] >= int(p["RARE_GENE_MIN_CELLS"]))
        & (stats["prevalence"] <= float(p["RARE_GENE_MAX_PREVALENCE"]))
        & (stats["positive_mean"] > 0)
    )
    raw_score = (
        np.log1p(stats["positive_mean"])
        * np.power(np.maximum(1.0 - stats["prevalence"], 0.0), 1.5)
        * np.log1p(np.maximum(dispersion, 0.0))
    )
    raw_score[~eligible] = 0.0
    return raw_score, eligible, stats


def _scaled_svd(X, n_components, whitening_power=0.5, random_seed=42):
    if issparse(X):
        X = X.toarray()
    dense = scale_dense_matrix(np.asarray(X, dtype=np.float32), max_value=10)
    n_components = min(
        int(n_components),
        max(int(dense.shape[0]) - 1, 1),
        max(int(dense.shape[1]) - 1, 1),
    )
    if n_components < 1:
        return np.zeros((dense.shape[0], 1), dtype=np.float32)
    embedding = TruncatedSVD(
        n_components=n_components,
        random_state=int(random_seed),
    ).fit_transform(dense)
    embedding = np.asarray(embedding, dtype=np.float32)
    embedding -= embedding.mean(axis=0, keepdims=True)
    component_scale = embedding.std(axis=0, keepdims=True)
    component_scale[component_scale <= 1e-8] = 1.0
    power = float(np.clip(whitening_power, 0.0, 1.0))
    embedding /= np.power(component_scale, power)
    return embedding.astype(np.float32, copy=False)


def _postprocess_svd_embedding(embedding, whitening_power):
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.ndim == 1:
        embedding = embedding[:, None]
    embedding -= embedding.mean(axis=0, keepdims=True)
    component_scale = embedding.std(axis=0, keepdims=True)
    component_scale[component_scale <= 1e-8] = 1.0
    power = float(np.clip(whitening_power, 0.0, 1.0))
    embedding /= np.power(component_scale, power)
    return embedding.astype(np.float32, copy=False)


def _direct_svd_embedding(X, n_components, whitening_power, random_seed):
    """Project a pre-weighted matrix without undoing its column weights."""
    n_cells, n_features = X.shape
    if n_cells == 0 or n_features == 0:
        return np.zeros((n_cells, 1), dtype=np.float32)
    if issparse(X):
        X = X.tocsr().astype(np.float32)
        has_signal = X.nnz > 0 and np.any(np.abs(X.data) > 0)
    else:
        X = np.asarray(X, dtype=np.float32)
        has_signal = np.any(np.abs(X) > 0)
    if not has_signal:
        return np.zeros((n_cells, 1), dtype=np.float32)
    components = min(
        max(int(n_components), 1),
        max(int(n_cells) - 1, 1),
        max(int(n_features) - 1, 1),
    )
    embedding = TruncatedSVD(
        n_components=components,
        random_state=int(random_seed),
    ).fit_transform(X)
    return _postprocess_svd_embedding(embedding, whitening_power)


def _weighted_rare_embedding(adata, feature_views, p, random_seed):
    """Build a rare view from expression magnitude and marker co-detection.

    Inverse-prevalence weighting keeps a shared low-frequency marker visible,
    while a binary detection block is less sensitive to count-depth variation.
    The two blocks are projected separately so expression scaling cannot erase
    the inverse-frequency prior.
    """
    rare_mask = np.asarray(feature_views["rare_mask"], dtype=bool)
    if not np.any(rare_mask):
        return np.zeros((adata.n_obs, 1), dtype=np.float32)
    X = adata[:, rare_mask].X
    if issparse(X):
        X = X.tocsr().astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    stats = _column_statistics(X)
    idf = np.log(
        (1.0 + float(adata.n_obs)) / (1.0 + stats["nonzero"].astype(np.float64))
    ) + 1.0
    idf_reference = float(np.percentile(idf, 75)) if len(idf) else 1.0
    idf = np.clip(idf / max(idf_reference, 1e-8), 0.25, 4.0)
    score_rank = _rank01(np.asarray(feature_views["rare_score"])[rare_mask])
    score_weight = np.power(
        0.25 + 0.75 * score_rank,
        max(float(p["RARE_SCORE_POWER"]), 0.0),
    )
    column_weight = (
        np.power(idf, max(float(p["RARE_IDF_POWER"]), 0.0))
        * score_weight
    ).astype(np.float32)

    if issparse(X):
        expression = X.multiply(column_weight[None, :]).tocsr()
        detection = X.copy().tocsr()
        detection.data = np.ones_like(detection.data, dtype=np.float32)
        detection = detection.multiply(column_weight[None, :]).tocsr()
    else:
        expression = X * column_weight[None, :]
        detection = (X > 0).astype(np.float32) * column_weight[None, :]
    expression = normalize(expression, norm="l2", axis=1, copy=False)
    detection = normalize(detection, norm="l2", axis=1, copy=False)

    total_dims = max(int(p["RARE_PCA_DIMS"]), 1)
    binary_weight = float(np.clip(p["RARE_BINARY_WEIGHT"], 0.0, 1.0))
    if binary_weight <= 0:
        return _direct_svd_embedding(
            expression,
            total_dims,
            p["EMBED_WHITENING_POWER"],
            random_seed,
        )
    if binary_weight >= 1:
        return _direct_svd_embedding(
            detection,
            total_dims,
            p["EMBED_WHITENING_POWER"],
            random_seed + 1,
        )

    binary_dims = max(int(round(total_dims * binary_weight)), 1)
    expression_dims = max(total_dims - binary_dims, 1)
    expression_embedding = _direct_svd_embedding(
        expression,
        expression_dims,
        p["EMBED_WHITENING_POWER"],
        random_seed,
    )
    detection_embedding = _direct_svd_embedding(
        detection,
        binary_dims,
        p["EMBED_WHITENING_POWER"],
        random_seed + 1,
    )
    combined = np.hstack([
        expression_embedding * np.sqrt(1.0 - binary_weight),
        detection_embedding * np.sqrt(binary_weight),
    ]).astype(np.float32, copy=False)
    return normalize(combined, norm="l2", axis=1, copy=False).astype(
        np.float32,
        copy=False,
    )


def _adaptive_gamma_center_count(gamma, p):
    n_cells = len(gamma)
    minimum = min(max(int(p["DPC_MIN_BASE_CENTERS"]), 1), n_cells)
    maximum = min(max(int(p["DPC_MAX_BASE_CENTERS"]), minimum), n_cells)
    fallback = min(max(int(p["DPC_GAMMA_TOPK"]), minimum), maximum)
    if str(p.get("DPC_BASE_CENTER_MODE", "adaptive_knee")).lower() != "adaptive_knee":
        return fallback
    if maximum <= minimum:
        return minimum

    sorted_gamma = np.sort(np.maximum(np.asarray(gamma, dtype=np.float64), 1e-12))[::-1]
    y = np.log(sorted_gamma[:maximum])
    if not np.all(np.isfinite(y)) or np.ptp(y) <= 1e-12:
        return fallback
    x = np.linspace(0.0, 1.0, len(y))
    y_norm = (y - y[-1]) / max(float(y[0] - y[-1]), 1e-12)
    chord = 1.0 - x
    curvature = chord - y_norm
    candidate_positions = np.arange(minimum - 1, max(maximum - 1, minimum))
    if len(candidate_positions) == 0:
        return fallback
    best_position = candidate_positions[np.argmax(curvature[candidate_positions])]
    return int(np.clip(best_position + 1, minimum, maximum))


def _dpc_feature_mask(adata, specificity, p):
    hvg = np.asarray(adata.var["highly_variable"], dtype=bool)
    topn = max(int(p["DPC_RARE_FEATURE_TOPN"]), 0)
    order = np.argsort(-specificity)
    added = order[specificity[order] > 0][:topn]
    mask = hvg.copy()
    mask[added] = True
    return mask, added


def run_rare_aware_dpc(adata, specificity, p, random_seed):
    n_cells = adata.n_obs
    if n_cells < 2:
        labels = np.zeros(n_cells, dtype=int)
        return labels, {
            "dpc_total_center_count": int(n_cells),
            "dpc_local_center_count": 0,
            "dpc_center_indices": list(range(n_cells)),
        }
    dpc_mask, added_features = _dpc_feature_mask(adata, specificity, p)
    embedding = _scaled_svd(
        adata[:, dpc_mask].X,
        p["DPC_PCA_N_COMPONENTS"],
        p["EMBED_WHITENING_POWER"],
        random_seed,
    )
    k = min(max(int(p["DPC_KNN_K"]), 2), n_cells - 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    neighbors.fit(embedding)
    nn_dists, nn_indices = neighbors.kneighbors(embedding)
    nonself_distances = nn_dists[:, 1:].ravel()
    positive_distances = nonself_distances[nonself_distances > 0]
    if len(positive_distances):
        dc = float(np.percentile(positive_distances, p["DPC_DC_PERCENT"]))
    else:
        dc = 1.0
    dc = max(dc, 1e-8)
    rho_global = np.sum(
        np.exp(-np.square(nn_dists[:, 1:] / dc)), axis=1
    ).astype(np.float64)
    density_mode = str(p.get("DPC_DENSITY_MODE", "hybrid_local")).lower()
    rho_local = np.zeros_like(rho_global)
    if density_mode == "hybrid_local":
        local_k = min(max(int(p["DPC_LOCAL_DENSITY_K"]), 1), k)
        scale_k = min(max(int(p["DPC_LOCAL_DENSITY_SCALE_K"]), 1), k)
        local_indices = nn_indices[:, 1:local_k + 1]
        local_distances = nn_dists[:, 1:local_k + 1]
        local_scale = np.maximum(nn_dists[:, scale_k], 1e-8)
        local_denominator = np.maximum(
            local_scale[:, None] * local_scale[local_indices],
            1e-12,
        )
        rho_local = np.sum(
            np.exp(-np.square(local_distances) / local_denominator),
            axis=1,
        )
        global_scale = np.median(rho_global[rho_global > 0]) if np.any(rho_global > 0) else 1.0
        local_rho_scale = np.median(rho_local[rho_local > 0]) if np.any(rho_local > 0) else 1.0
        global_normalized = rho_global / max(float(global_scale), 1e-12)
        local_normalized = rho_local / max(float(local_rho_scale), 1e-12)
        local_weight = float(np.clip(p["DPC_LOCAL_DENSITY_WEIGHT"], 0.0, 1.0))
        rho = (
            (1.0 - local_weight) * global_normalized
            + local_weight * local_normalized
        ).astype(np.float32)
    elif density_mode in {"global", "global_kernel"}:
        rho = rho_global.astype(np.float32)
    else:
        raise ValueError("Unknown DPC_DENSITY_MODE: {}".format(density_mode))
    delta, nneigh = compute_min_distance(embedding, rho, nn_dists, nn_indices)

    # Candidate centers use the rare-aware view. Assignment uses the standard
    # HVG geometry so that a few specificity genes cannot distort propagation.
    hvg_mask = np.asarray(adata.var["highly_variable"], dtype=bool)
    assignment_embedding = _scaled_svd(
        adata[:, hvg_mask].X,
        p["DPC_PCA_N_COMPONENTS"],
        p["EMBED_WHITENING_POWER"],
        random_seed,
    )
    assignment_neighbors = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        n_jobs=-1,
    )
    assignment_neighbors.fit(assignment_embedding)
    assignment_dists, assignment_indices = assignment_neighbors.kneighbors(
        assignment_embedding
    )
    _, assignment_nneigh = compute_min_distance(
        assignment_embedding,
        rho,
        assignment_dists,
        assignment_indices,
    )

    gamma = rho * delta
    center_params = dict(p)
    center_params["DPC_GAMMA_TOPK"] = _adaptive_gamma_center_count(gamma, p)
    center_indices, center_diagnostics = select_dpc_centers(
        embedding,
        rho,
        delta,
        nn_dists,
        nn_indices,
        center_params,
    )

    labels = -np.ones(n_cells, dtype=int)
    for cluster_id, center_index in enumerate(center_indices):
        labels[int(center_index)] = int(cluster_id)
    for cell_index in np.argsort(-rho):
        if labels[cell_index] < 0:
            labels[cell_index] = labels[assignment_nneigh[cell_index]]
    if np.any(labels < 0):
        labels[labels < 0] = 0

    center_diagnostics.update({
        "dpc_dc": float(dc),
        "dpc_density_mode": density_mode,
        "dpc_global_rho_cv": float(
            np.std(rho_global) / max(float(np.mean(rho_global)), 1e-12)
        ),
        "dpc_local_rho_cv": float(
            np.std(rho_local) / max(float(np.mean(rho_local)), 1e-12)
        ) if np.any(rho_local) else 0.0,
        "dpc_adaptive_base_topk": int(center_params["DPC_GAMMA_TOPK"]),
        "dpc_added_feature_count": int(len(added_features)),
        "dpc_gamma_cv": float(np.std(gamma) / max(float(np.mean(gamma)), 1e-12)),
    })
    return remap_labels(labels), center_diagnostics


def _cluster_gene_contrast(adata, labels, p):
    labels = np.asarray(labels)
    n_cells, n_genes = adata.shape
    stats = _column_statistics(adata.X)
    global_std = np.sqrt(np.maximum(stats["variance"], 1e-8))
    best_absolute = np.zeros(n_genes, dtype=np.float64)
    best_positive = np.zeros(n_genes, dtype=np.float64)
    best_detection = np.zeros(n_genes, dtype=np.float64)
    min_size = max(int(p["FEATURE_MIN_CLUSTER_SIZE"]), 1)
    reference_frac = max(float(p["FEATURE_RARE_REFERENCE_FRAC"]), 1.0 / max(n_cells, 1))
    weight_cap = max(float(p["FEATURE_RARE_WEIGHT_CAP"]), 1.0)

    for cluster_id in np.unique(labels):
        member = labels == cluster_id
        cluster_size = int(np.sum(member))
        rest_size = n_cells - cluster_size
        if cluster_size < min_size or rest_size < 1:
            continue
        X_cluster = adata.X[member]
        cluster_stats = _column_statistics(X_cluster)
        cluster_mean = cluster_stats["sum"] / float(cluster_size)
        rest_mean = (stats["sum"] - cluster_stats["sum"]) / float(rest_size)
        effect = (cluster_mean - rest_mean) / global_std
        cluster_prevalence = cluster_stats["nonzero"] / float(cluster_size)
        rest_prevalence = (
            stats["nonzero"] - cluster_stats["nonzero"]
        ) / float(rest_size)

        fraction = cluster_size / float(n_cells)
        rarity_weight = min(np.sqrt(reference_frac / max(fraction, 1.0 / n_cells)), weight_cap)
        rarity_weight = max(rarity_weight, 1.0)
        reliability = min(1.0, np.sqrt(cluster_size / float(min_size)))
        weight = rarity_weight * reliability
        best_absolute = np.maximum(best_absolute, np.abs(effect) * weight)
        best_positive = np.maximum(best_positive, np.maximum(effect, 0.0) * weight)
        best_detection = np.maximum(
            best_detection,
            np.maximum(cluster_prevalence - rest_prevalence, 0.0) * weight,
        )

    absolute_rank = _rank01(best_absolute)
    positive_rank = _rank01(best_positive)
    detection_rank = _rank01(best_detection)
    contrast = 0.50 * absolute_rank + 0.35 * positive_rank + 0.15 * detection_rank
    rare_contrast = 0.65 * positive_rank + 0.35 * detection_rank
    return contrast, rare_contrast


def _ensemble_tree_gene_importance(adata, labels, p, random_seed):
    labels = np.asarray(labels).astype(str)
    n_genes = adata.n_vars
    if len(np.unique(labels)) < 2 or adata.n_obs < 3 or n_genes < 2:
        return np.zeros(n_genes, dtype=np.float64)

    n_components = min(
        int(p["RF_PCA_DIMS"]),
        adata.n_obs - 1,
        adata.n_vars - 1,
    )
    if n_components < 1:
        return np.zeros(n_genes, dtype=np.float64)
    reducer = TruncatedSVD(n_components=n_components, random_state=int(random_seed))
    latent = reducer.fit_transform(adata.X)

    forest = RandomForestClassifier(
        n_estimators=int(p["RF_N_ESTIMATORS"]),
        random_state=int(random_seed),
        n_jobs=-1,
        max_features="sqrt",
        class_weight="balanced_subsample",
    )
    extra = ExtraTreesClassifier(
        n_estimators=int(p["ET_N_ESTIMATORS"]),
        random_state=int(random_seed) + 17,
        n_jobs=-1,
        max_features="sqrt",
        class_weight="balanced",
    )
    forest.fit(latent, labels)
    extra.fit(latent, labels)
    latent_importance = 0.5 * forest.feature_importances_ + 0.5 * extra.feature_importances_
    gene_importance = np.abs(reducer.components_).T.dot(latent_importance)
    return np.asarray(gene_importance, dtype=np.float64)


def select_hybrid_gene_views(
    adata,
    pseudo_labels,
    specificity,
    specificity_eligible,
    p,
    random_seed,
    previous_hybrid_score=None,
):
    tree_score = _ensemble_tree_gene_importance(
        adata, pseudo_labels, p, random_seed
    )
    contrast_score, rare_contrast = _cluster_gene_contrast(adata, pseudo_labels, p)
    tree_rank = _rank01(tree_score)
    contrast_rank = _rank01(contrast_score)
    specificity_rank = _rank01(specificity, specificity_eligible)

    hybrid_score = (
        float(p["HYBRID_RF_WEIGHT"]) * tree_rank
        + float(p["HYBRID_CONTRAST_WEIGHT"]) * contrast_rank
        + float(p["HYBRID_SPECIFICITY_WEIGHT"]) * specificity_rank
    )
    if previous_hybrid_score is not None:
        previous_weight = float(np.clip(p["REFINE_PREVIOUS_WEIGHT"], 0.0, 1.0))
        hybrid_score = (
            previous_weight * np.asarray(previous_hybrid_score, dtype=np.float64)
            + (1.0 - previous_weight) * hybrid_score
        )

    n_hig = min(max(int(p["HIG_TOPN"]), 1), adata.n_vars)
    hig_indices = np.argsort(-hybrid_score)[:n_hig]
    hig_mask = np.zeros(adata.n_vars, dtype=bool)
    hig_mask[hig_indices] = True

    rare_score = 0.50 * specificity_rank + 0.40 * rare_contrast + 0.10 * tree_rank
    rare_score[~specificity_eligible] *= 0.25
    n_rare = min(max(int(p["RARE_GENE_TOPN"]), 1), adata.n_vars)
    rare_indices = np.argsort(-rare_score)[:n_rare]
    rare_mask = np.zeros(adata.n_vars, dtype=bool)
    rare_mask[rare_indices] = True

    hvg_mask = np.asarray(adata.var["highly_variable"], dtype=bool)
    discriminative_mask = hvg_mask | hig_mask
    min_features = min(max(int(p["MIN_VIEW_FEATURES"]), 1), adata.n_vars)
    if np.sum(rare_mask) < min_features:
        rare_mask[np.argsort(-hybrid_score)[:min_features]] = True

    adata.var["archer_v2_hig"] = hig_mask
    adata.var["archer_v2_rare_gene"] = rare_mask
    adata.var["archer_v2_hybrid_score"] = hybrid_score.astype(np.float32)
    adata.var["archer_v2_rare_score"] = rare_score.astype(np.float32)
    return {
        "global_mask": hvg_mask,
        "hig_mask": hig_mask,
        "discriminative_mask": discriminative_mask,
        "rare_mask": rare_mask,
        "hybrid_score": hybrid_score,
        "rare_score": rare_score,
        "n_hvg": int(np.sum(hvg_mask)),
        "n_hig": int(np.sum(hig_mask)),
        "n_discriminative": int(np.sum(discriminative_mask)),
        "n_rare_genes": int(np.sum(rare_mask)),
    }


def build_three_embeddings(adata, feature_views, p, random_seed):
    whitening = float(p["EMBED_WHITENING_POWER"])
    global_embedding = _scaled_svd(
        adata[:, feature_views["global_mask"]].X,
        p["GLOBAL_PCA_DIMS"],
        whitening,
        random_seed,
    )
    discriminative_embedding = _scaled_svd(
        adata[:, feature_views["discriminative_mask"]].X,
        p["DISCRIM_PCA_DIMS"],
        whitening,
        random_seed + 11,
    )
    rare_embedding = _weighted_rare_embedding(
        adata,
        feature_views,
        p,
        random_seed + 23,
    )
    return {
        "global": global_embedding,
        "discriminative": discriminative_embedding,
        "rare": rare_embedding,
    }


def _adaptive_neighbor_counts(distances, p):
    # ``distances`` excludes self and is ordered by increasing distance.
    n_cells, available_k = distances.shape
    default_k = min(max(int(p["GRAPH_DEFAULT_K"]), 1), available_k)
    minimum_k = min(max(int(p["GRAPH_MIN_K"]), 1), available_k)
    search_k = min(max(int(p["GRAPH_GAP_SEARCH_K"]), 2), available_k)
    gap_threshold = float(p["GRAPH_GAP_THRESHOLD"])
    pair_threshold = float(p["GRAPH_PAIR_GAP_THRESHOLD"])
    micro_gap_threshold = float(p["GRAPH_MICRO_GAP_THRESHOLD"])
    micro_max_k = max(int(p["GRAPH_MICRO_MAX_K"]), 1)
    counts = np.full(n_cells, default_k, dtype=int)
    gap_strength = np.ones(n_cells, dtype=np.float32)

    if search_k < 2:
        return counts, gap_strength
    eps = 1e-8
    ratios = distances[:, 1:search_k] / np.maximum(distances[:, :search_k - 1], eps)
    best_positions = np.argmax(ratios, axis=1)
    best_ratios = ratios[np.arange(n_cells), best_positions]
    proposed = best_positions + 1
    for i in range(n_cells):
        ratio = float(best_ratios[i])
        candidate_k = int(proposed[i])
        if candidate_k == 1:
            if ratio >= pair_threshold:
                counts[i] = 1
        elif candidate_k <= micro_max_k and ratio >= micro_gap_threshold:
            # A sharp early gap is direct evidence for a micro-neighborhood;
            # it may legitimately be smaller than the global minimum k.
            counts[i] = candidate_k
        elif ratio >= gap_threshold:
            counts[i] = max(candidate_k, minimum_k)
        gap_strength[i] = ratio
    counts = np.clip(counts, 1, available_k)
    return counts, gap_strength


def _clear_sparse_diagonal(matrix):
    matrix = matrix.tocsr()
    if np.any(np.abs(matrix.diagonal()) > 0):
        matrix = matrix.tolil(copy=True)
        matrix.setdiag(0)
        matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    return matrix


def _normalize_sparse_weights(matrix):
    matrix = matrix.tocsr().astype(np.float32)
    matrix = _clear_sparse_diagonal(matrix)
    if matrix.nnz == 0:
        return matrix
    positive = matrix.data[matrix.data > 0]
    if len(positive) == 0:
        return matrix
    scale = float(np.percentile(positive, 95))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.max(positive))
    if scale > 0:
        matrix.data = np.clip(matrix.data / scale, 0.0, 1.0)
    return matrix


def build_adaptive_view_graph(embedding, metric, p, return_state=False):
    n_cells = embedding.shape[0]
    if n_cells <= 1:
        empty = csr_matrix((n_cells, n_cells), dtype=np.float32)
        diagnostics = {"mean_adaptive_k": 0.0, "pair_adaptive_fraction": 0.0}
        if return_state:
            state = {
                "indices": np.zeros((n_cells, 0), dtype=int),
                "distances": np.zeros((n_cells, 0), dtype=np.float32),
                "adaptive_k": np.zeros(n_cells, dtype=int),
                "gap_strength": np.ones(n_cells, dtype=np.float32),
            }
            return empty, empty, diagnostics, state
        return empty, empty, diagnostics
    kmax = min(max(int(p["GRAPH_KMAX"]), 2), n_cells - 1)
    nearest = NearestNeighbors(n_neighbors=kmax + 1, metric=metric, n_jobs=-1)
    nearest.fit(embedding)
    nn_dists, nn_indices = nearest.kneighbors(embedding)
    distances = np.maximum(np.asarray(nn_dists[:, 1:], dtype=np.float64), 0.0)
    indices = np.asarray(nn_indices[:, 1:], dtype=int)
    adaptive_k, gap_strength = _adaptive_neighbor_counts(distances, p)

    scale_k = min(max(int(p["GRAPH_SCALE_K"]), 1), kmax)
    sigma = np.maximum(distances[:, scale_k - 1], 1e-8)
    rank_power = max(float(p["GRAPH_RANK_POWER"]), 0.0)
    rows = []
    cols = []
    values = []
    for cell_index in range(n_cells):
        count = int(adaptive_k[cell_index])
        selected_indices = indices[cell_index, :count]
        selected_distances = distances[cell_index, :count]
        ranks = np.arange(1, count + 1, dtype=np.float64)
        local_denominator = np.maximum(
            sigma[cell_index] * sigma[selected_indices],
            1e-12,
        )
        affinity = np.exp(-np.square(selected_distances) / local_denominator)
        affinity /= np.power(ranks, rank_power)
        rows.extend([cell_index] * count)
        cols.extend(selected_indices.tolist())
        values.extend(affinity.tolist())

    directed = csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, cols)),
        shape=(n_cells, n_cells),
    )
    directed = _clear_sparse_diagonal(directed)
    product = directed.multiply(directed.T).tocsr()
    mutual = product.copy()
    mutual.data = np.sqrt(np.maximum(mutual.data, 0.0))
    union = directed.maximum(directed.T).tocsr()
    one_way_penalty = float(np.clip(p["GRAPH_ONE_WAY_PENALTY"], 0.0, 1.0))
    graph = union.multiply(one_way_penalty) + mutual.multiply(1.0 - one_way_penalty)

    binary = directed.copy()
    binary.data = np.ones_like(binary.data, dtype=np.float32)
    shared = binary.dot(binary.T).tocsr()
    degrees = np.asarray(binary.sum(axis=1)).ravel()
    coo = shared.tocoo()
    unions = degrees[coo.row] + degrees[coo.col] - coo.data
    jaccard = np.divide(
        coo.data,
        np.maximum(unions, 1.0),
    )
    keep = jaccard > float(p["GRAPH_SNN_PRUNE"])
    snn = csr_matrix(
        (
            np.asarray(jaccard[keep], dtype=np.float32),
            (coo.row[keep], coo.col[keep]),
        ),
        shape=(n_cells, n_cells),
    )
    graph = graph + snn.multiply(float(p["GRAPH_SNN_WEIGHT"]))
    graph = _normalize_sparse_weights(graph.maximum(graph.T))
    mutual = _normalize_sparse_weights(mutual.maximum(mutual.T))
    diagnostics = {
        "mean_adaptive_k": float(np.mean(adaptive_k)),
        "median_adaptive_k": float(np.median(adaptive_k)),
        "pair_adaptive_fraction": float(np.mean(adaptive_k == 1)),
        "micro_adaptive_fraction": float(
            np.mean(adaptive_k < max(int(p["GRAPH_MIN_K"]), 1))
        ),
        "median_gap_strength": float(np.median(gap_strength)),
        "view_edge_count": int(graph.nnz // 2),
    }
    if return_state:
        state = {
            "indices": indices,
            "distances": distances.astype(np.float32, copy=False),
            "adaptive_k": adaptive_k,
            "gap_strength": gap_strength,
        }
        return graph, mutual, diagnostics, state
    return graph, mutual, diagnostics


def _topk_row_prune(matrix, k):
    matrix = matrix.tocsr()
    n_cells = matrix.shape[0]
    k = max(int(k), 1)
    rows = []
    cols = []
    values = []
    for row in range(n_cells):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        row_cols = matrix.indices[start:end]
        row_values = matrix.data[start:end]
        if len(row_values) > k:
            selected = np.argpartition(row_values, -k)[-k:]
            row_cols = row_cols[selected]
            row_values = row_values[selected]
        rows.extend([row] * len(row_cols))
        cols.extend(row_cols.tolist())
        values.extend(row_values.tolist())
    directed = csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, cols)),
        shape=matrix.shape,
    )
    pruned = directed.maximum(directed.T).tocsr()
    pruned = _clear_sparse_diagonal(pruned)
    return pruned


def _ensure_no_isolates(pruned, source):
    pruned = pruned.tolil(copy=True)
    source = source.tocsr()
    degrees = np.asarray(pruned.getnnz(axis=1)).ravel()
    for row in np.where(degrees == 0)[0]:
        start, end = source.indptr[row], source.indptr[row + 1]
        if end <= start:
            continue
        local = int(np.argmax(source.data[start:end]))
        col = int(source.indices[start + local])
        value = float(source.data[start + local])
        pruned[row, col] = value
        pruned[col, row] = value
    output = pruned.tocsr()
    output = _clear_sparse_diagonal(output)
    return output


def fuse_view_graphs(view_graphs, core_graphs, p):
    weights = {
        "global": float(p["GLOBAL_VIEW_WEIGHT"]),
        "discriminative": float(p["DISCRIM_VIEW_WEIGHT"]),
        "rare": float(p["RARE_VIEW_WEIGHT"]),
    }
    weight_sum = max(sum(weights.values()), 1e-12)
    fused = None
    max_view = None
    core_fused = None
    view_support = None
    for name in ["global", "discriminative", "rare"]:
        weight = weights[name] / weight_sum
        weighted = view_graphs[name].multiply(weight)
        core_weighted = core_graphs[name].multiply(weight)
        fused = weighted if fused is None else fused + weighted
        core_fused = core_weighted if core_fused is None else core_fused + core_weighted
        max_view = view_graphs[name] if max_view is None else max_view.maximum(view_graphs[name])
        binary = view_graphs[name].tocsr(copy=True)
        binary.data = np.ones_like(binary.data, dtype=np.float32)
        view_support = binary if view_support is None else view_support + binary
    support_fraction = view_support.multiply(1.0 / max(float(len(view_graphs)), 1.0))
    supported_max = max_view.multiply(support_fraction)
    fused = fused + supported_max.multiply(float(p["GRAPH_MAX_VIEW_BONUS"]))
    fused = _normalize_sparse_weights(fused.maximum(fused.T))
    unpruned = fused.copy()
    dynamic_topk = max(
        int(p["GRAPH_FUSED_MIN_TOPK"]),
        int(round(np.sqrt(max(fused.shape[0], 1)))),
    )
    dynamic_topk = min(dynamic_topk, int(p["GRAPH_FUSED_TOPK"]))
    fused = _topk_row_prune(fused, dynamic_topk)
    fused = _ensure_no_isolates(fused, unpruned)
    core_fused = _normalize_sparse_weights(core_fused.maximum(core_fused.T))
    return fused, core_fused


def _mutual_topk_row_prune(matrix, k):
    """Keep an edge only when both endpoints rank it among their top-k."""
    matrix = matrix.tocsr()
    k = max(int(k), 1)
    rows = []
    cols = []
    values = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        row_cols = matrix.indices[start:end]
        row_values = matrix.data[start:end]
        if len(row_values) > k:
            selected = np.argpartition(row_values, -k)[-k:]
            row_cols = row_cols[selected]
            row_values = row_values[selected]
        rows.extend([row] * len(row_cols))
        cols.extend(row_cols.tolist())
        values.extend(row_values.tolist())
    directed = csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, cols)),
        shape=matrix.shape,
    )
    mutual = directed.minimum(directed.T).tocsr()
    mutual = _clear_sparse_diagonal(mutual)
    return mutual


def _view_support_fraction(core_graphs):
    support = None
    for graph in core_graphs.values():
        binary = graph.tocsr(copy=True)
        binary.data = np.ones_like(binary.data, dtype=np.float32)
        support = binary if support is None else support + binary
    if support is None:
        return csr_matrix((0, 0), dtype=np.float32)
    support = support.multiply(1.0 / max(float(len(core_graphs)), 1.0)).tocsr()
    support = _clear_sparse_diagonal(support)
    return support


def _generate_topology_candidates(core_graphs, core_fused, p):
    n_cells = core_fused.shape[0]
    if n_cells < 2:
        return []
    max_size = min(
        max(int(p["TOPOLOGY_MAX_SIZE"]), 2),
        max(int(np.floor(n_cells * float(p["TOPOLOGY_MAX_FRAC"]))), 2),
    )
    edge_threshold = float(p["TOPOLOGY_EDGE_MIN_WEIGHT"])
    sources = list(core_graphs.items()) + [("fused", core_fused)]
    candidate_sources = {}
    for view_name, source_graph in sources:
        if source_graph.nnz == 0:
            continue
        for core_k in p["TOPOLOGY_CORE_KS"]:
            pruned = _mutual_topk_row_prune(source_graph, core_k)
            if pruned.nnz == 0:
                continue
            pruned = pruned.tocsr(copy=True)
            pruned.data[pruned.data < edge_threshold] = 0.0
            pruned.eliminate_zeros()
            if pruned.nnz == 0:
                continue
            component_count, membership = connected_components(
                pruned,
                directed=False,
            )
            source_id = "{}:k{}".format(view_name, int(core_k))
            for component_id in range(component_count):
                cells = np.where(membership == component_id)[0]
                if 2 <= len(cells) <= max_size:
                    key = tuple(cells.astype(int).tolist())
                    candidate_sources.setdefault(key, set()).add(source_id)
    return [
        {
            "cells": np.asarray(cells, dtype=int),
            "sources": sorted(source_ids),
        }
        for cells, source_ids in candidate_sources.items()
    ]


def _candidate_topology_evidence(
    cells,
    support_fraction,
    core_graphs,
    view_states,
    p,
):
    cells = np.asarray(cells, dtype=int)
    cell_set = set(cells.tolist())
    size = len(cells)
    possible_edges = max(size * (size - 1) / 2.0, 1.0)
    support_mean = float(support_fraction[cells][:, cells].sum())
    support_mean = support_mean / (2.0 * possible_edges)
    core_internal_means = {}
    view_anchor_scores = {}
    view_nearest_retention = {}
    view_adaptive_retention = {}
    view_gap_evidence = {}
    gap_reference = (
        float(p["GRAPH_PAIR_GAP_THRESHOLD"])
        if size == 2
        else float(p["GRAPH_GAP_THRESHOLD"])
    )
    gap_denominator = max(gap_reference - 1.0, 1e-8)

    for view_name, core_graph in core_graphs.items():
        internal_sum = float(core_graph[cells][:, cells].sum()) / 2.0
        core_internal_means[view_name] = internal_sum / possible_edges
        state = view_states[view_name]
        indices = np.asarray(state["indices"], dtype=int)
        adaptive_k = np.asarray(state["adaptive_k"], dtype=int)
        gap_strength = np.asarray(state["gap_strength"], dtype=np.float64)
        nearest_values = []
        adaptive_values = []
        gap_values = []
        for cell in cells:
            available = indices.shape[1]
            nearest_count = min(max(size - 1, 1), available)
            nearest = indices[cell, :nearest_count]
            nearest_values.append(
                float(np.mean([int(item) in cell_set for item in nearest]))
                if nearest_count
                else 0.0
            )
            local_k = min(max(int(adaptive_k[cell]), 1), available)
            local_neighbors = indices[cell, :local_k]
            adaptive_values.append(
                float(np.mean([int(item) in cell_set for item in local_neighbors]))
                if local_k
                else 0.0
            )
            gap_values.append(
                float(np.clip((gap_strength[cell] - 1.0) / gap_denominator, 0.0, 1.0))
            )
        nearest_retention = float(np.mean(nearest_values))
        adaptive_retention = float(np.mean(adaptive_values))
        gap_evidence = float(np.mean(gap_values))
        view_nearest_retention[view_name] = nearest_retention
        view_adaptive_retention[view_name] = adaptive_retention
        view_gap_evidence[view_name] = gap_evidence
        view_anchor_scores[view_name] = (
            0.55 * nearest_retention
            + 0.25 * adaptive_retention
            + 0.20 * gap_evidence
        )

    anchor_values = list(view_anchor_scores.values())
    best_anchor = max(anchor_values) if anchor_values else 0.0
    mean_anchor = float(np.mean(anchor_values)) if anchor_values else 0.0
    anchor_score = 0.70 * best_anchor + 0.30 * mean_anchor
    return {
        "support_mean": float(support_mean),
        "best_core_internal_mean": float(
            max(core_internal_means.values()) if core_internal_means else 0.0
        ),
        "anchor_score": float(anchor_score),
        "best_view_anchor_score": float(best_anchor),
        "view_anchor_scores": view_anchor_scores,
        "view_nearest_retention": view_nearest_retention,
        "view_adaptive_retention": view_adaptive_retention,
        "view_gap_evidence": view_gap_evidence,
    }


def _candidate_background_cells(cells, fused_graph, view_states, p):
    cells = np.asarray(cells, dtype=int)
    cell_set = set(cells.tolist())
    minimum = max(int(p["TOPOLOGY_BACKGROUND_MIN"]), 1)
    maximum = max(int(p["TOPOLOGY_BACKGROUND_MAX"]), minimum)
    external_strength = {}
    graph = fused_graph.tocsr()
    for cell in cells:
        start, end = graph.indptr[cell], graph.indptr[cell + 1]
        for neighbor, weight in zip(
            graph.indices[start:end],
            graph.data[start:end],
        ):
            neighbor = int(neighbor)
            if neighbor not in cell_set:
                external_strength[neighbor] = (
                    external_strength.get(neighbor, 0.0) + float(weight)
                )
    ordered = [
        cell
        for cell, _ in sorted(
            external_strength.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    seen = set(ordered)
    for view_name in ["rare", "discriminative", "global"]:
        indices = np.asarray(view_states[view_name]["indices"], dtype=int)
        for cell in cells:
            for neighbor in indices[cell, :minimum]:
                neighbor = int(neighbor)
                if neighbor not in cell_set and neighbor not in seen:
                    ordered.append(neighbor)
                    seen.add(neighbor)
    if len(ordered) < minimum:
        for cell in range(fused_graph.shape[0]):
            if cell not in cell_set and cell not in seen:
                ordered.append(cell)
                seen.add(cell)
                if len(ordered) >= minimum:
                    break
    return np.asarray(ordered[:maximum], dtype=int)


def _normalize_embedding_rows(embedding):
    values = np.asarray(embedding, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _candidate_embedding_separation(normalized_embedding, cells, background_cells):
    cells = np.asarray(cells, dtype=int)
    background_cells = np.asarray(background_cells, dtype=int)
    if len(cells) < 2 or len(background_cells) == 0:
        return 0.0
    candidate = normalized_embedding[cells]
    background = normalized_embedding[background_cells]
    within = np.linalg.norm(candidate[:, None, :] - candidate[None, :, :], axis=2)
    within_mean = within.sum(axis=1) / max(len(cells) - 1, 1)
    between = np.linalg.norm(candidate[:, None, :] - background[None, :, :], axis=2)
    nearest_count = min(10, between.shape[1])
    nearest_background = np.partition(
        between,
        nearest_count - 1,
        axis=1,
    )[:, :nearest_count].mean(axis=1)
    denominator = np.maximum(np.maximum(within_mean, nearest_background), 1e-12)
    return float(np.mean((nearest_background - within_mean) / denominator))


def _prepare_marker_context(adata, feature_views):
    rare_mask = np.asarray(feature_views["rare_mask"], dtype=bool)
    gene_indices = np.where(rare_mask)[0]
    X = adata[:, gene_indices].X
    if issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X)
    stats = _column_statistics(X)
    global_std = np.sqrt(np.maximum(stats["variance"], 1e-8))
    rare_score = np.asarray(feature_views["rare_score"], dtype=np.float64)
    prior = 0.50 + 0.50 * _rank01(rare_score[gene_indices])
    gene_names = np.asarray(adata.var_names.astype(str))[gene_indices]
    return {
        "X": X,
        "global_std": global_std,
        "prior": prior,
        "gene_names": gene_names,
    }


def _candidate_marker_coherence(cells, background_cells, marker_context, p):
    cells = np.asarray(cells, dtype=int)
    background_cells = np.asarray(background_cells, dtype=int)
    if len(cells) < 2 or len(background_cells) == 0:
        return 0.0, 0, None, 0.0
    candidate_stats = _column_statistics(marker_context["X"][cells])
    background_stats = _column_statistics(marker_context["X"][background_cells])
    effect = np.maximum(
        (candidate_stats["mean"] - background_stats["mean"])
        / np.maximum(marker_context["global_std"], 1e-8),
        0.0,
    )
    detection_gain = np.maximum(
        candidate_stats["prevalence"] - background_stats["prevalence"],
        0.0,
    )
    eligible = (
        candidate_stats["prevalence"]
        >= float(p["TOPOLOGY_MARKER_MIN_PREVALENCE"])
    ) & (effect > 0)
    gene_evidence = (
        (1.0 - np.exp(-effect))
        * np.sqrt(detection_gain)
        * candidate_stats["prevalence"]
        * marker_context["prior"]
    )
    gene_evidence[~eligible] = 0.0
    positive = np.where(gene_evidence > 0)[0]
    if len(positive) == 0:
        return 0.0, 0, None, 0.0
    order = positive[np.argsort(-gene_evidence[positive])]
    top = order[:max(int(p["TOPOLOGY_MARKER_TOPK"]), 1)]
    best_index = int(top[0])
    marker_score = 0.65 * float(gene_evidence[best_index]) + 0.35 * float(
        np.mean(gene_evidence[top])
    )
    marker_count = int(np.sum(gene_evidence >= 0.20))
    return (
        float(marker_score),
        marker_count,
        str(marker_context["gene_names"][best_index]),
        float(gene_evidence[best_index]),
    )


def _score_topology_cells(
    cells,
    sources,
    fused_graph,
    support_fraction,
    core_graphs,
    view_states,
    rare_normalized,
    discriminative_normalized,
    marker_context,
    p,
):
    """Score an arbitrary small set with the same evidence as an anchor."""
    cells = np.asarray(cells, dtype=int)
    graph_quality = _component_graph_quality(fused_graph, cells)
    topology = _candidate_topology_evidence(
        cells,
        support_fraction,
        core_graphs,
        view_states,
        p,
    )
    background = _candidate_background_cells(
        cells,
        fused_graph,
        view_states,
        p,
    )
    rare_embed_score = _candidate_embedding_separation(
        rare_normalized,
        cells,
        background,
    )
    discriminative_embed_score = _candidate_embedding_separation(
        discriminative_normalized,
        cells,
        background,
    )
    embed_score = max(
        rare_embed_score,
        0.60 * rare_embed_score + 0.40 * discriminative_embed_score,
    )
    marker_score, marker_count, best_marker, best_marker_score = (
        _candidate_marker_coherence(
            cells,
            background,
            marker_context,
            p,
        )
    )
    separation_normalized = graph_quality["edge_separation"] / (
        1.0 + graph_quality["edge_separation"]
    )
    support_normalized = 0.25 + 0.75 * np.clip(
        topology["support_mean"],
        0.0,
        1.0,
    )
    embed_normalized = np.clip((embed_score + 0.20) / 0.80, 0.0, 1.0)
    composite = (
        0.15 * np.clip(graph_quality["internal_mean"], 0.0, 1.0)
        + 0.15 * np.clip(1.0 - graph_quality["local_conductance"], 0.0, 1.0)
        + 0.10 * np.clip(separation_normalized, 0.0, 1.0)
        + 0.20 * np.clip(topology["anchor_score"], 0.0, 1.0)
        + 0.10 * support_normalized
        + 0.15 * embed_normalized
        + 0.15 * np.clip(marker_score, 0.0, 1.0)
    )
    return {
        "cells": cells.astype(int).tolist(),
        "size": int(len(cells)),
        "source_count": int(len(sources)),
        "sources": list(sources),
        "internal_mean": float(graph_quality["internal_mean"]),
        "conductance": float(graph_quality["conductance"]),
        "separation": float(graph_quality["separation"]),
        "boundary_mean": float(graph_quality["boundary_mean"]),
        "local_conductance": float(graph_quality["local_conductance"]),
        "edge_separation": float(graph_quality["edge_separation"]),
        "view_support": float(topology["support_mean"]),
        "anchor_score": float(topology["anchor_score"]),
        "rare_embedding_score": float(rare_embed_score),
        "discriminative_embedding_score": float(discriminative_embed_score),
        "embedding_score": float(embed_score),
        "marker_score": float(marker_score),
        "marker_count": int(marker_count),
        "best_marker": best_marker,
        "best_marker_score": float(best_marker_score),
        "composite_score": float(composite),
        "marker_panel_score": float(
            1.0 - np.exp(
                -marker_count
                / max(float(p["TOPOLOGY_PRIORITY_MARKER_PANEL_SCALE"]), 1e-8)
            )
        ),
        "priority_score": float(
            composite
            + float(p["TOPOLOGY_PRIORITY_RARE_EMBED_WEIGHT"])
            * np.clip(rare_embed_score, 0.0, 1.0)
            + float(p["TOPOLOGY_PRIORITY_MARKER_PANEL_WEIGHT"])
            * (
                1.0 - np.exp(
                    -marker_count
                    / max(
                        float(p["TOPOLOGY_PRIORITY_MARKER_PANEL_SCALE"]),
                        1e-8,
                    )
                )
            )
        ),
        "accepted": True,
        "selected": False,
        "_cells": cells,
    }


def _purify_topology_record(
    record,
    fused_graph,
    support_fraction,
    core_graphs,
    view_states,
    rare_normalized,
    discriminative_normalized,
    marker_context,
    p,
):
    """Remove a boundary cell under strict improvement or a narrow bridge rule."""
    original_cells = np.asarray(record["_cells"], dtype=int)
    original_priority = float(record.get("priority_score", 0.0))
    original_composite = float(record.get("composite_score", 0.0))
    if (
        not bool(p.get("TOPOLOGY_PURIFY_ENABLED", True))
        or len(original_cells) <= 2
        or len(original_cells) > int(p["TOPOLOGY_PURIFY_MAX_SIZE"])
    ):
        record["purified"] = False
        return record
    current = record
    removed = []
    purification_modes = []
    while len(current["_cells"]) > 2:
        children = []
        current_cells = np.asarray(current["_cells"], dtype=int)
        for removed_cell in current_cells:
            child_cells = current_cells[current_cells != removed_cell]
            child = _score_topology_cells(
                child_cells,
                current["sources"],
                fused_graph,
                support_fraction,
                core_graphs,
                view_states,
                rare_normalized,
                discriminative_normalized,
                marker_context,
                p,
            )
            child["_removed_cell"] = int(removed_cell)
            children.append(child)
        selected_child = max(
            children,
            key=lambda item: item["composite_score"],
        )
        composite_gain = (
            selected_child["composite_score"] - current["composite_score"]
        )
        embed_gain = (
            selected_child["embedding_score"] - current["embedding_score"]
        )
        anchor_gain = selected_child["anchor_score"] - current["anchor_score"]
        marker_retained = selected_child["marker_score"] >= (
            float(p["TOPOLOGY_PURIFY_MARKER_RETENTION"])
            * current["marker_score"]
        )
        strict_improvement = (
            composite_gain >= float(p["TOPOLOGY_PURIFY_MIN_COMPOSITE_GAIN"])
            and embed_gain >= float(p["TOPOLOGY_PURIFY_MIN_EMBED_GAIN"])
            and anchor_gain >= float(p["TOPOLOGY_PURIFY_MIN_ANCHOR_GAIN"])
            and marker_retained
        )
        mode = "strict_improvement"

        # A three-cell topology component can be a coherent rare pair joined
        # by one graph bridge.  In that specific case the full component has
        # poor embedding separation, while exactly one leave-one-out pair is
        # decisively better and keeps the same marker program.  This narrow
        # rule avoids general-purpose pruning of genuinely heterogeneous rare
        # groups.
        if not strict_improvement and len(current_cells) == 3:
            embedding_order = sorted(
                children,
                key=lambda item: -item["embedding_score"],
            )
            bridge_child = embedding_order[0]
            runner_up = embedding_order[1]
            bridge_marker_retained = bridge_child["marker_score"] >= (
                float(p["TOPOLOGY_PURIFY_MARKER_RETENTION"])
                * current["marker_score"]
            )
            bridge_release = (
                current["embedding_score"]
                <= float(p["TOPOLOGY_PURIFY_BRIDGE_MAX_PARENT_EMBED"])
                and bridge_child["embedding_score"]
                >= float(p["TOPOLOGY_PURIFY_BRIDGE_MIN_CHILD_EMBED"])
                and bridge_child["embedding_score"] - current["embedding_score"]
                >= float(p["TOPOLOGY_PURIFY_BRIDGE_MIN_EMBED_GAIN"])
                and bridge_child["embedding_score"] - runner_up["embedding_score"]
                >= float(p["TOPOLOGY_PURIFY_BRIDGE_MIN_EMBED_MARGIN"])
                and bridge_child["anchor_score"]
                >= float(p["TOPOLOGY_PURIFY_BRIDGE_MIN_ANCHOR"])
                and bridge_child["marker_score"]
                >= float(p["TOPOLOGY_MIN_MARKER_SCORE"])
                and bridge_child["composite_score"]
                >= float(p["TOPOLOGY_MIN_COMPOSITE"])
                and bridge_marker_retained
            )
            if bridge_release:
                selected_child = bridge_child
                mode = "bridge_release"
            else:
                break
        elif not strict_improvement:
            break

        removed.append(int(selected_child.pop("_removed_cell")))
        purification_modes.append(mode)
        current = selected_child
    current.pop("_removed_cell", None)
    current["purified"] = bool(removed)
    current["purified_from_size"] = int(len(original_cells))
    current["purified_removed_cells"] = removed
    current["purification_modes"] = purification_modes
    current["original_cells"] = original_cells.astype(int).tolist()
    current["prepurification_composite_score"] = original_composite
    current["prepurification_priority_score"] = original_priority
    current["selection_priority_score"] = max(
        float(current.get("priority_score", 0.0)),
        original_priority,
    )
    return current


def _topology_acceptance_limit(
    p,
    reference_cluster_count=None,
    selection_stage="initial",
):
    limit = max(int(p["TOPOLOGY_MAX_ACCEPTED"]), 1)
    if reference_cluster_count is not None:
        ratio_key = (
            "TOPOLOGY_REFINE_MAX_ACCEPTED_RATIO"
            if selection_stage == "refine"
            else "TOPOLOGY_MAX_ACCEPTED_RATIO"
        )
        adaptive = max(
            int(p["TOPOLOGY_MIN_ACCEPTED"]),
            int(np.ceil(
                max(int(reference_cluster_count), 1)
                * float(p[ratio_key])
            )),
        )
        limit = min(limit, adaptive)
    return max(int(limit), 1)


def _annotate_topology_parent(record, parent_labels, p):
    """Describe whether an anchor maintains or discovers a parent subcluster."""
    record["parent_label"] = None
    record["parent_size"] = None
    record["parent_purity"] = 0.0
    record["parent_coverage"] = 0.0
    record["maintenance_candidate"] = False
    if parent_labels is None:
        return record
    labels = np.asarray(parent_labels)
    cells = np.asarray(record["_cells"], dtype=int)
    if len(labels) == 0 or len(cells) == 0:
        return record
    values, counts = np.unique(labels[cells], return_counts=True)
    dominant_index = int(np.argmax(counts))
    dominant = values[dominant_index]
    dominant_count = int(counts[dominant_index])
    parent_size = int(np.sum(labels == dominant))
    purity = dominant_count / max(float(len(cells)), 1.0)
    coverage = dominant_count / max(float(parent_size), 1.0)
    if isinstance(dominant, np.generic):
        dominant = dominant.item()
    record["parent_label"] = dominant
    record["parent_size"] = int(parent_size)
    record["parent_purity"] = float(purity)
    record["parent_coverage"] = float(coverage)
    record["maintenance_candidate"] = bool(
        purity >= float(p["TOPOLOGY_PARENT_MIN_PURITY"])
        and coverage >= float(p["TOPOLOGY_MAINTENANCE_MIN_COVERAGE"])
    )
    return record


def _select_nonoverlapping_components(
    records,
    p,
    reference_cluster_count=None,
    parent_labels=None,
    selection_stage="initial",
):
    """Select stable anchors without letting old anchors starve new parents.

    A refinement candidate that covers most of an existing parent is a
    maintenance anchor.  It is retained first, while the remaining capacity is
    assigned parent-wise to genuinely new subclusters.  The initial round stays
    deliberately conservative because it has no graph partition to maintain.
    """
    selected = []
    selected_records = []
    selected_sets = []
    limit = _topology_acceptance_limit(
        p,
        reference_cluster_count,
        selection_stage=selection_stage,
    )
    ordered = sorted(
        records,
        key=lambda item: -item.get(
            "selection_priority_score",
            item.get("priority_score", item["composite_score"]),
        ),
    )
    for record in ordered:
        record["selected"] = False
        record["selection_role"] = (
            "maintenance_candidate"
            if record.get("maintenance_candidate", False)
            else "discovery_candidate"
        )

    maintenance = (
        [item for item in ordered if item.get("maintenance_candidate", False)]
        if selection_stage == "refine" and parent_labels is not None
        else []
    )
    maintenance_ids = {id(item) for item in maintenance}
    discovery = [item for item in ordered if id(item) not in maintenance_ids]

    def try_select(record, role):
        if len(selected) >= limit:
            return False
        cells = np.asarray(record["_cells"], dtype=int)
        cell_set = set(cells.tolist())
        if any(cell_set & existing for existing in selected_sets):
            record["selected"] = False
            return False
        record["selected"] = True
        record["selection_role"] = role
        selected.append(cells)
        selected_records.append(record)
        selected_sets.append(cell_set)
        return True

    maintained_parents = set()
    for record in maintenance:
        parent = record.get("parent_label")
        if parent in maintained_parents:
            continue
        if try_select(record, "maintenance"):
            maintained_parents.add(parent)
        if len(selected) >= limit:
            break

    discoveries_per_parent = {}
    max_per_parent = max(int(p["TOPOLOGY_MAX_DISCOVERIES_PER_PARENT"]), 1)
    # Round-robin parent allocation: every parent receives its first discovery
    # before any parent receives a second.  This preserves breadth while using
    # otherwise idle capacity for non-overlapping substructure.
    for parent_quota in range(1, max_per_parent + 1):
        for record in discovery:
            if len(selected) >= limit:
                break
            parent = record.get("parent_label")
            if selection_stage == "refine" and parent in maintained_parents:
                continue
            parent_key = parent if parent is not None else "__unassigned__"
            current_count = discoveries_per_parent.get(parent_key, 0)
            if current_count >= parent_quota:
                continue
            if try_select(record, "discovery"):
                discoveries_per_parent[parent_key] = current_count + 1
        if len(selected) >= limit:
            break

    if bool(p.get("TOPOLOGY_NESTED_EXPANSION_ENABLED", True)):
        max_added = max(int(p["TOPOLOGY_NESTED_MAX_ADDED"]), 0)
        minimum_sources = max(int(p["TOPOLOGY_NESTED_MIN_SOURCE_COUNT"]), 1)
        marker_retention = float(p["TOPOLOGY_NESTED_MARKER_RETENTION"])
        for selected_index, initial_record in enumerate(list(selected_records)):
            current = initial_record
            current_set = set(np.asarray(current["_cells"], dtype=int).tolist())
            forbidden = set(current.get("purified_removed_cells", []))
            added = 0
            while added < max_added:
                other_selected = set()
                for other_index, other_set in enumerate(selected_sets):
                    if other_index != selected_index:
                        other_selected.update(other_set)
                immediate_supersets = []
                for candidate in ordered:
                    candidate_set = set(
                        np.asarray(candidate["_cells"], dtype=int).tolist()
                    )
                    if len(candidate_set) != len(current_set) + 1:
                        continue
                    if not current_set.issubset(candidate_set):
                        continue
                    if candidate_set & forbidden:
                        continue
                    if candidate_set & other_selected:
                        continue
                    if candidate.get("parent_label") != current.get("parent_label"):
                        continue
                    if int(candidate.get("source_count", 0)) < minimum_sources:
                        continue
                    if candidate.get("marker_score", 0.0) < (
                        marker_retention * current.get("marker_score", 0.0)
                    ):
                        continue
                    if candidate.get("anchor_score", 0.0) < float(
                        p["TOPOLOGY_MIN_ANCHOR_SCORE"]
                    ):
                        continue
                    if candidate.get("composite_score", 0.0) < float(
                        p["TOPOLOGY_MIN_COMPOSITE"]
                    ):
                        continue
                    immediate_supersets.append(candidate)
                if not immediate_supersets:
                    break
                expanded = max(
                    immediate_supersets,
                    key=lambda item: item.get(
                        "selection_priority_score",
                        item.get("priority_score", item["composite_score"]),
                    ),
                )
                previous_cells = sorted(current_set)
                previous_role = current.get("selection_role", "discovery")
                current["selected"] = False
                current["selection_role"] = previous_role + "_core"
                expanded["selected"] = True
                expanded["selection_role"] = previous_role
                expanded["nested_expansion"] = True
                expanded["nested_expanded_from_cells"] = previous_cells
                current = expanded
                current_set = set(
                    np.asarray(current["_cells"], dtype=int).tolist()
                )
                forbidden.update(current.get("purified_removed_cells", []))
                selected[selected_index] = np.asarray(current["_cells"], dtype=int)
                selected_records[selected_index] = current
                selected_sets[selected_index] = current_set
                added += 1
    return selected


def discover_topology_anchors(
    adata,
    fused_graph,
    core_fused,
    core_graphs,
    view_states,
    embeddings,
    feature_views,
    p,
    marker_context=None,
    reference_cluster_count=None,
    parent_labels=None,
    selection_stage="initial",
):
    """Find tiny cross-view cliques whose boundary has gene-level support."""
    if not bool(p.get("TOPOLOGY_ANCHOR_ENABLED", True)):
        return [], {"enabled": False, "candidate_count": 0}
    candidates = _generate_topology_candidates(core_graphs, core_fused, p)
    if not candidates:
        return [], {
            "enabled": True,
            "candidate_count": 0,
            "prequalified_count": 0,
            "accepted_count": 0,
        }
    support_fraction = _view_support_fraction(core_graphs)
    rare_normalized = _normalize_embedding_rows(embeddings["rare"])
    discriminative_normalized = _normalize_embedding_rows(
        embeddings["discriminative"]
    )

    prequalified = []
    for candidate in candidates:
        cells = candidate["cells"]
        graph_quality = _component_graph_quality(fused_graph, cells)
        topology = _candidate_topology_evidence(
            cells,
            support_fraction,
            core_graphs,
            view_states,
            p,
        )
        global_boundary_gate = (
            graph_quality["conductance"]
            <= float(p["TOPOLOGY_MAX_CONDUCTANCE"])
            and graph_quality["separation"]
            >= float(p["TOPOLOGY_MIN_SEPARATION"])
        )
        local_boundary_gate = (
            graph_quality["local_conductance"]
            <= float(p["TOPOLOGY_MAX_LOCAL_CONDUCTANCE"])
            and graph_quality["edge_separation"]
            >= float(p["TOPOLOGY_MIN_EDGE_SEPARATION"])
        )
        graph_gate = (
            graph_quality["internal_mean"]
            >= float(p["TOPOLOGY_MIN_INTERNAL_MEAN"])
            and (global_boundary_gate or local_boundary_gate)
        )
        anchor_gate = (
            topology["anchor_score"]
            >= float(p["TOPOLOGY_MIN_ANCHOR_SCORE"])
        )
        if not (graph_gate and anchor_gate):
            continue
        preliminary_score = (
            topology["anchor_score"]
            + graph_quality["internal_mean"]
            + (1.0 - graph_quality["local_conductance"])
            + 0.10 * len(candidate["sources"])
        )
        prequalified.append({
            "cells": cells,
            "sources": candidate["sources"],
            "graph_quality": graph_quality,
            "topology": topology,
            "preliminary_score": float(preliminary_score),
        })

    prequalified.sort(key=lambda item: -item["preliminary_score"])
    retained = prequalified[:max(int(p["TOPOLOGY_MAX_CANDIDATES"]), 1)]
    if marker_context is None:
        marker_context = _prepare_marker_context(adata, feature_views)
    evaluated_records = []
    accepted_records = []
    for candidate in retained:
        cells = candidate["cells"]
        record = _score_topology_cells(
            cells,
            candidate["sources"],
            fused_graph,
            support_fraction,
            core_graphs,
            view_states,
            rare_normalized,
            discriminative_normalized,
            marker_context,
            p,
        )
        biology_gate = (
            record["marker_score"] >= float(p["TOPOLOGY_MIN_MARKER_SCORE"])
            and (
                record["embedding_score"]
                >= float(p["TOPOLOGY_MIN_EMBED_SCORE"])
                or record["marker_score"]
                >= 2.0 * float(p["TOPOLOGY_MIN_MARKER_SCORE"])
            )
        )
        accepted = (
            biology_gate
            and record["composite_score"]
            >= float(p["TOPOLOGY_MIN_COMPOSITE"])
        )
        record["accepted"] = bool(accepted)
        if accepted:
            record = _purify_topology_record(
                record,
                fused_graph,
                support_fraction,
                core_graphs,
                view_states,
                rare_normalized,
                discriminative_normalized,
                marker_context,
                p,
            )
        record = _annotate_topology_parent(record, parent_labels, p)
        evaluated_records.append(record)
        if accepted:
            accepted_records.append(record)

    selected = _select_nonoverlapping_components(
        accepted_records,
        p,
        reference_cluster_count=reference_cluster_count,
        parent_labels=parent_labels,
        selection_stage=selection_stage,
    )
    diagnostic_topn = max(int(p["TOPOLOGY_DIAGNOSTIC_TOPN"]), 1)
    sorted_records = sorted(
        evaluated_records,
        key=lambda item: -item["composite_score"],
    )[:diagnostic_topn]
    diagnostics_records = []
    for record in sorted_records:
        clean_record = dict(record)
        clean_record.pop("_cells", None)
        diagnostics_records.append(clean_record)
    diagnostics = {
        "enabled": True,
        "candidate_count": int(len(candidates)),
        "prequalified_count": int(len(prequalified)),
        "evaluated_count": int(len(retained)),
        "accepted_count": int(len(accepted_records)),
        "acceptance_limit": int(
            _topology_acceptance_limit(
                p,
                reference_cluster_count,
                selection_stage=selection_stage,
            )
        ),
        "selected_count": int(len(selected)),
        "selected_sizes": [int(len(cells)) for cells in selected],
        "selection_stage": str(selection_stage),
        "selected_maintenance_count": int(sum(
            record.get("selected", False)
            and record.get("selection_role") == "maintenance"
            for record in accepted_records
        )),
        "selected_discovery_count": int(sum(
            record.get("selected", False)
            and record.get("selection_role") == "discovery"
            for record in accepted_records
        )),
        "selected_maintenance_sizes": [
            int(record["size"])
            for record in accepted_records
            if record.get("selected", False)
            and record.get("selection_role") == "maintenance"
        ],
        "selected_discovery_sizes": [
            int(record["size"])
            for record in accepted_records
            if record.get("selected", False)
            and record.get("selection_role") == "discovery"
        ],
        "selected_nested_expansion_count": int(sum(
            record.get("selected", False)
            and record.get("nested_expansion", False)
            for record in accepted_records
        )),
        "selected_nested_expansion_sizes": [
            int(record["size"])
            for record in accepted_records
            if record.get("selected", False)
            and record.get("nested_expansion", False)
        ],
        "top_candidates": diagnostics_records,
    }
    return selected, diagnostics


def _combine_nonoverlapping_components(primary, secondary):
    combined = [np.asarray(cells, dtype=int) for cells in primary]
    occupied = set()
    for cells in combined:
        occupied.update(cells.tolist())
    for cells in secondary:
        cells = np.asarray(cells, dtype=int)
        if not (set(cells.tolist()) & occupied):
            combined.append(cells)
            occupied.update(cells.tolist())
    return combined


def detect_micro_populations(fused_graph, core_graph, p):
    n_cells = fused_graph.shape[0]
    if n_cells < 2 or core_graph.nnz == 0:
        return [], []
    core = _topk_row_prune(core_graph, p["MICRO_CORE_K"])
    core = core.tocsr(copy=True)
    core.data[core.data < float(p["MICRO_CORE_MIN_WEIGHT"])] = 0.0
    core.eliminate_zeros()
    if core.nnz == 0:
        return [], []
    n_components, membership = connected_components(core, directed=False)
    max_size = min(
        max(int(p["MICRO_MAX_SIZE"]), 2),
        max(int(np.floor(n_cells * float(p["MICRO_MAX_FRAC"]))), 2),
    )
    protected = []
    diagnostics = []
    for component_id in range(n_components):
        cells = np.where(membership == component_id)[0]
        size = len(cells)
        if size < 2 or size > max_size:
            continue
        quality = _component_graph_quality(fused_graph, cells)
        global_boundary_gate = (
            quality["conductance"] <= float(p["MICRO_MAX_CONDUCTANCE"])
            and quality["separation"] >= float(p["MICRO_MIN_SEPARATION"])
        )
        local_boundary_gate = (
            quality["local_conductance"]
            <= float(p["MICRO_MAX_LOCAL_CONDUCTANCE"])
            and quality["edge_separation"]
            >= float(p["MICRO_MIN_EDGE_SEPARATION"])
        )
        accepted = (
            quality["internal_mean"] >= float(p["MICRO_MIN_INTERNAL_MEAN"])
            and (global_boundary_gate or local_boundary_gate)
        )
        record = {
            "size": int(size),
            "accepted": bool(accepted),
        }
        record.update(quality)
        diagnostics.append(record)
        if accepted:
            protected.append(cells.astype(int))
    return protected, diagnostics


def protect_micro_populations(graph, components, p):
    if not components:
        return graph
    component_membership = -np.ones(graph.shape[0], dtype=int)
    for component_id, cells in enumerate(components):
        component_membership[np.asarray(cells, dtype=int)] = component_id
    coo = graph.tocoo()
    left = component_membership[coo.row]
    right = component_membership[coo.col]
    same_protected = (left >= 0) & (left == right)
    boundary = ((left >= 0) | (right >= 0)) & ~same_protected
    factors = np.ones(len(coo.data), dtype=np.float32)
    factors[same_protected] *= float(p["MICRO_INTERNAL_BOOST"])
    factors[boundary] *= float(p["MICRO_EXTERNAL_SHRINK"])
    protected = csr_matrix(
        (coo.data * factors, (coo.row, coo.col)),
        shape=graph.shape,
    )
    return _normalize_sparse_weights(protected.maximum(protected.T))


def _graph_to_igraph(graph):
    upper = triu(graph, k=1).tocoo()
    result = ig.Graph(n=graph.shape[0], directed=False)
    if upper.nnz:
        result.add_edges(list(zip(upper.row.tolist(), upper.col.tolist())))
        result.es["weight"] = upper.data.astype(float).tolist()
    return result


def _run_leiden(graph, resolution, seed, p):
    if graph.vcount() == 0:
        return np.zeros(0, dtype=int)
    if graph.ecount() == 0:
        return np.arange(graph.vcount(), dtype=int)
    ig.set_random_number_generator(random.Random(int(seed)))
    partition = graph.community_leiden(
        objective_function="modularity",
        weights=graph.es["weight"],
        resolution=float(resolution),
        n_iterations=int(p["LEIDEN_ITERATIONS"]),
    )
    return np.asarray(partition.membership, dtype=int)


def _base_resolution(n_cells, p):
    resolution = (
        float(p["RESOLUTION_BASE"])
        + np.log10(max(n_cells / 5000.0, 1.0)) * float(p["RESOLUTION_SCALE"])
    )
    return float(np.clip(resolution, p["RESOLUTION_MIN"], p["RESOLUTION_MAX"]))


def _consensus_reweight(graph, partitions, p):
    coo = graph.tocoo()
    if not partitions or coo.nnz == 0:
        return graph
    agreement = np.zeros(coo.nnz, dtype=np.float32)
    for labels in partitions:
        agreement += (labels[coo.row] == labels[coo.col]).astype(np.float32)
    agreement /= float(len(partitions))
    floor = float(np.clip(p["CONSENSUS_WEIGHT_FLOOR"], 0.0, 1.0))
    data = coo.data * (floor + (1.0 - floor) * agreement)
    consensus = csr_matrix((data, (coo.row, coo.col)), shape=graph.shape)
    return _normalize_sparse_weights(consensus.maximum(consensus.T))


def _force_protected_splits(labels, components):
    labels = np.asarray(labels, dtype=int).copy()
    next_label = int(labels.max()) + 1 if len(labels) else 0
    for cells in components:
        cells = np.asarray(cells, dtype=int)
        if len(cells) == 0:
            continue
        current = labels[cells]
        already_exact = (
            np.all(current == current[0])
            and int(np.sum(labels == current[0])) == len(cells)
        )
        if not already_exact:
            # A validated anchor is an atomic unit.  This also reunites an
            # anchor that a stochastic partition happened to split in two.
            labels[cells] = next_label
            next_label += 1
    return remap_labels(labels)


def _set_jaccard(left, right):
    left = set(left)
    right = set(right)
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / float(len(union))


def _component_graph_quality(graph, cells):
    cells = np.asarray(cells, dtype=int)
    size = len(cells)
    if size < 2:
        return {
            "internal_mean": 0.0,
            "conductance": 1.0,
            "separation": 0.0,
            "boundary_mean": 0.0,
            "local_conductance": 1.0,
            "edge_separation": 0.0,
        }
    subgraph = graph[cells][:, cells]
    internal_sum = float(subgraph.sum()) / 2.0
    possible_edges = size * (size - 1) / 2.0
    internal_mean = internal_sum / max(possible_edges, 1.0)
    total_strength = np.asarray(graph[cells].sum(axis=1)).ravel()
    external_sum = max(float(np.sum(total_strength)) - 2.0 * internal_sum, 0.0)
    conductance = external_sum / max(2.0 * internal_sum + external_sum, 1e-12)
    internal_per_cell = 2.0 * internal_sum / float(size)
    external_per_cell = external_sum / float(size)
    separation = internal_per_cell / max(external_per_cell, 1e-12)
    cell_set = set(cells.tolist())
    boundary_means = []
    graph_csr = graph.tocsr()
    boundary_k = max(size - 1, 1)
    for cell in cells:
        start, end = graph_csr.indptr[cell], graph_csr.indptr[cell + 1]
        external_values = np.asarray([
            float(weight)
            for neighbor, weight in zip(
                graph_csr.indices[start:end],
                graph_csr.data[start:end],
            )
            if int(neighbor) not in cell_set
        ], dtype=np.float64)
        if len(external_values):
            keep = min(boundary_k, len(external_values))
            strongest = np.partition(external_values, len(external_values) - keep)[-keep:]
            boundary_means.append(float(np.mean(strongest)))
        else:
            boundary_means.append(0.0)
    boundary_mean = float(np.mean(boundary_means)) if boundary_means else 0.0
    local_conductance = boundary_mean / max(
        internal_mean + boundary_mean,
        1e-12,
    )
    edge_separation = internal_mean / max(boundary_mean, 1e-12)
    return {
        "internal_mean": float(internal_mean),
        "conductance": float(conductance),
        "separation": float(separation),
        "boundary_mean": float(boundary_mean),
        "local_conductance": float(local_conductance),
        "edge_separation": float(edge_separation),
    }


def _local_embedding_separation(embedding, parent_cells, candidate_cells):
    parent_cells = np.asarray(parent_cells, dtype=int)
    candidate_cells = np.asarray(candidate_cells, dtype=int)
    candidate_set = set(candidate_cells.tolist())
    background_cells = np.asarray(
        [cell for cell in parent_cells if int(cell) not in candidate_set],
        dtype=int,
    )
    if len(candidate_cells) < 2 or len(background_cells) == 0:
        return 0.0

    values = np.asarray(embedding, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    candidate = normalized[candidate_cells]
    background = normalized[background_cells]

    within = np.linalg.norm(candidate[:, None, :] - candidate[None, :, :], axis=2)
    within_sum = within.sum(axis=1)
    within_mean = within_sum / max(len(candidate_cells) - 1, 1)
    between = np.linalg.norm(candidate[:, None, :] - background[None, :, :], axis=2)
    nearest_count = min(10, between.shape[1])
    nearest_background = np.partition(
        between,
        nearest_count - 1,
        axis=1,
    )[:, :nearest_count].mean(axis=1)
    denominator = np.maximum(np.maximum(within_mean, nearest_background), 1e-12)
    return float(np.mean((nearest_background - within_mean) / denominator))


def discover_hidden_micro_populations(
    labels,
    graph,
    rare_embedding,
    base_resolution,
    p,
    random_seed,
    marker_context=None,
    view_states=None,
    reference_cluster_count=None,
):
    """Search inside broad clusters for stable, marker-coherent small children."""
    if not bool(p.get("LOCAL_REFINE_ENABLED", True)):
        return [], []
    labels = np.asarray(labels, dtype=int)
    accepted_scored = []
    diagnostics = []
    min_parent_size = max(int(p["LOCAL_REFINE_MIN_PARENT_SIZE"]), 4)
    match_threshold = float(p["LOCAL_REFINE_MATCH_JACCARD"])
    minimum_support = max(int(p["LOCAL_REFINE_MIN_SUPPORT"]), 1)
    max_children = max(int(p["LOCAL_REFINE_MAX_CHILDREN_PER_PARENT"]), 1)

    for parent_label in np.unique(labels):
        parent_cells = np.where(labels == parent_label)[0]
        parent_size = len(parent_cells)
        if parent_size < min_parent_size:
            continue
        max_child_size = min(
            max(int(p["LOCAL_REFINE_MAX_CHILD_SIZE"]), 2),
            max(int(np.floor(parent_size * float(p["LOCAL_REFINE_MAX_CHILD_FRAC"]))), 2),
        )
        if max_child_size >= parent_size:
            max_child_size = parent_size - 1
        if max_child_size < 2:
            continue

        parent_graph = graph[parent_cells][:, parent_cells]
        local_igraph = _graph_to_igraph(parent_graph)
        if local_igraph.ecount() == 0:
            continue
        occurrences = []
        partition_id = 0
        for multiplier_index, multiplier in enumerate(
            p["LOCAL_REFINE_RESOLUTION_MULTIPLIERS"]
        ):
            for seed_offset in range(max(int(p["LOCAL_REFINE_SEEDS"]), 1)):
                local_labels = _run_leiden(
                    local_igraph,
                    float(base_resolution) * float(multiplier),
                    int(random_seed) + 3001 + multiplier_index * 107 + seed_offset * 19,
                    p,
                )
                for child_label in np.unique(local_labels):
                    local_indices = np.where(local_labels == child_label)[0]
                    if 2 <= len(local_indices) <= max_child_size:
                        occurrences.append({
                            "cells": set(parent_cells[local_indices].tolist()),
                            "partition_id": partition_id,
                        })
                partition_id += 1

        groups = []
        for occurrence in occurrences:
            best_group = None
            best_similarity = 0.0
            for group_index, group in enumerate(groups):
                similarity = max(
                    _set_jaccard(occurrence["cells"], existing["cells"])
                    for existing in group["occurrences"]
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_group = group_index
            if best_group is not None and best_similarity >= match_threshold:
                groups[best_group]["occurrences"].append(occurrence)
            else:
                groups.append({"occurrences": [occurrence]})

        parent_candidates = []
        for group in groups:
            occurrence_group = group["occurrences"]
            support = len(set(item["partition_id"] for item in occurrence_group))
            if support < minimum_support:
                continue
            representatives = [item["cells"] for item in occurrence_group]
            average_jaccard = []
            for left_index, left in enumerate(representatives):
                similarities = [
                    _set_jaccard(left, right)
                    for right_index, right in enumerate(representatives)
                    if right_index != left_index
                ]
                average_jaccard.append(
                    float(np.mean(similarities)) if similarities else 1.0
                )
            representative = representatives[int(np.argmax(average_jaccard))]
            cells = np.asarray(sorted(representative), dtype=int)
            graph_quality = _component_graph_quality(graph, cells)
            embed_score = _local_embedding_separation(
                rare_embedding,
                parent_cells,
                cells,
            )
            marker_score = 1.0
            marker_count = 0
            best_marker = None
            best_marker_score = 0.0
            if marker_context is not None:
                candidate_set = set(cells.tolist())
                parent_set = set(parent_cells.tolist())
                if view_states is not None:
                    background = _candidate_background_cells(
                        cells,
                        graph,
                        view_states,
                        p,
                    )
                    background = np.asarray(
                        [
                            cell
                            for cell in background
                            if int(cell) in parent_set
                            and int(cell) not in candidate_set
                        ],
                        dtype=int,
                    )
                else:
                    background = np.zeros(0, dtype=int)
                minimum_background = max(int(p["TOPOLOGY_BACKGROUND_MIN"]), 1)
                if len(background) < minimum_background:
                    existing = set(background.tolist())
                    additions = [
                        int(cell)
                        for cell in parent_cells
                        if int(cell) not in candidate_set
                        and int(cell) not in existing
                    ]
                    background = np.concatenate([
                        background,
                        np.asarray(additions, dtype=int),
                    ])
                background = background[:max(
                    int(p["TOPOLOGY_BACKGROUND_MAX"]),
                    minimum_background,
                )]
                (
                    marker_score,
                    marker_count,
                    best_marker,
                    best_marker_score,
                ) = _candidate_marker_coherence(
                    cells,
                    background,
                    marker_context,
                    p,
                )
            global_boundary_gate = (
                graph_quality["conductance"]
                <= float(p["LOCAL_REFINE_MAX_CONDUCTANCE"])
                and graph_quality["separation"]
                >= float(p["LOCAL_REFINE_MIN_SEPARATION"])
            )
            local_boundary_gate = (
                graph_quality["local_conductance"]
                <= float(p["LOCAL_REFINE_MAX_LOCAL_CONDUCTANCE"])
                and graph_quality["edge_separation"]
                >= float(p["LOCAL_REFINE_MIN_EDGE_SEPARATION"])
            )
            accepted = (
                graph_quality["internal_mean"]
                >= float(p["LOCAL_REFINE_MIN_INTERNAL_MEAN"])
                and (global_boundary_gate or local_boundary_gate)
                and embed_score >= float(p["LOCAL_REFINE_MIN_EMBED_SCORE"])
                and marker_score >= float(p["LOCAL_REFINE_MIN_MARKER_SCORE"])
            )
            record = {
                "parent_label": int(parent_label),
                "parent_size": int(parent_size),
                "child_size": int(len(cells)),
                "support": int(support),
                "mean_match_jaccard": float(np.max(average_jaccard)),
                "embedding_score": float(embed_score),
                "marker_score": float(marker_score),
                "marker_count": int(marker_count),
                "best_marker": best_marker,
                "best_marker_score": float(best_marker_score),
                "accepted": bool(accepted),
            }
            record.update(graph_quality)
            diagnostics.append(record)
            if accepted:
                composite = (
                    support
                    + graph_quality["edge_separation"]
                    + max(embed_score, 0.0)
                    + marker_score
                    - graph_quality["local_conductance"]
                )
                parent_candidates.append((float(composite), cells))

        selected_for_parent = []
        for score, cells in sorted(parent_candidates, key=lambda item: -item[0]):
            candidate_set = set(cells.tolist())
            if any(
                _set_jaccard(candidate_set, set(existing[1].tolist())) > 0.25
                for existing in selected_for_parent
            ):
                continue
            selected_for_parent.append((float(score), cells))
            if len(selected_for_parent) >= max_children:
                break
        accepted_scored.extend(selected_for_parent)
    accepted_scored.sort(key=lambda item: -item[0])
    if reference_cluster_count is not None:
        total_limit = max(
            int(p["LOCAL_REFINE_MIN_TOTAL"]),
            int(np.ceil(
                max(int(reference_cluster_count), 1)
                * float(p["LOCAL_REFINE_MAX_TOTAL_RATIO"])
            )),
        )
        accepted_scored = accepted_scored[:total_limit]
    accepted_components = [cells for _, cells in accepted_scored]
    return accepted_components, diagnostics


def merge_unsupported_microclusters(
    labels,
    graph,
    marker_context,
    view_states,
    protected_components,
    p,
):
    """Merge only tiny fragments lacking both graph closure and markers."""
    labels = np.asarray(labels, dtype=int).copy()
    if not bool(p.get("CLEANUP_ENABLED", True)) or len(labels) < 2:
        return remap_labels(labels), {"enabled": False, "merges": []}
    protected_cells = set()
    for cells in protected_components:
        protected_cells.update(np.asarray(cells, dtype=int).tolist())
    max_size = min(
        max(int(p["CLEANUP_MAX_SIZE"]), 1),
        max(int(np.floor(len(labels) * float(p["CLEANUP_MAX_FRAC"]))), 1),
    )
    graph = graph.tocsr()
    merge_records = []
    for iteration in range(max(int(p["CLEANUP_MAX_ITERATIONS"]), 0)):
        changed = False
        cluster_sizes = {
            int(cluster): int(np.sum(labels == cluster))
            for cluster in np.unique(labels)
        }
        candidate_labels = sorted(
            [
                cluster
                for cluster, size in cluster_sizes.items()
                if size <= max_size
            ],
            key=lambda cluster: (cluster_sizes[cluster], cluster),
        )
        for cluster in candidate_labels:
            cells = np.where(labels == cluster)[0]
            if (
                len(cells) == 0
                or len(cells) > max_size
                or any(int(cell) in protected_cells for cell in cells)
            ):
                continue
            quality = _component_graph_quality(graph, cells)
            if quality["conductance"] < float(p["CLEANUP_MIN_CONDUCTANCE"]):
                continue
            background = _candidate_background_cells(
                cells,
                graph,
                view_states,
                p,
            )
            marker_score, marker_count, best_marker, _ = _candidate_marker_coherence(
                cells,
                background,
                marker_context,
                p,
            )
            if marker_score > float(p["CLEANUP_MAX_MARKER_SCORE"]):
                continue

            target_strength = {}
            for cell in cells:
                start, end = graph.indptr[cell], graph.indptr[cell + 1]
                for neighbor, weight in zip(
                    graph.indices[start:end],
                    graph.data[start:end],
                ):
                    target = int(labels[int(neighbor)])
                    if target != cluster:
                        target_strength[target] = (
                            target_strength.get(target, 0.0) + float(weight)
                        )
            if not target_strength:
                continue
            target = max(
                target_strength,
                key=lambda item: (
                    target_strength[item]
                    / np.sqrt(max(float(cluster_sizes.get(item, 1)), 1.0)),
                    target_strength[item],
                    -item,
                ),
            )
            merge_records.append({
                "iteration": int(iteration),
                "source_label": int(cluster),
                "source_size": int(len(cells)),
                "target_label": int(target),
                "conductance": float(quality["conductance"]),
                "marker_score": float(marker_score),
                "marker_count": int(marker_count),
                "best_marker": best_marker,
                "boundary_strength": float(target_strength[target]),
            })
            labels[cells] = target
            cluster_sizes[target] = cluster_sizes.get(target, 0) + len(cells)
            cluster_sizes[cluster] = 0
            changed = True
        if not changed:
            break
        labels = remap_labels(labels)
    return remap_labels(labels), {
        "enabled": True,
        "max_fragment_size": int(max_size),
        "merge_count": int(len(merge_records)),
        "merges": merge_records,
    }


def consensus_cluster(
    graph,
    protected_components,
    p,
    random_seed,
    reference_cluster_count=None,
):
    base_resolution = _base_resolution(graph.shape[0], p)
    igraph_graph = _graph_to_igraph(graph)
    partitions = []
    partition_resolutions = []
    seed_count = max(int(p["CONSENSUS_SEEDS"]), 1)
    for multiplier_index, multiplier in enumerate(p["CONSENSUS_RESOLUTION_MULTIPLIERS"]):
        for seed_offset in range(seed_count):
            seed = int(random_seed) + multiplier_index * 101 + seed_offset * 17
            resolution = base_resolution * float(multiplier)
            partitions.append(
                _run_leiden(
                    igraph_graph,
                    resolution,
                    seed,
                    p,
                )
            )
            partition_resolutions.append(float(resolution))

    selected_resolution = float(base_resolution)
    partition_centrality = []
    if partitions:
        for left_index, left in enumerate(partitions):
            similarities = [
                adjusted_rand_score(left, right)
                for right_index, right in enumerate(partitions)
                if right_index != left_index
            ]
            partition_centrality.append(
                float(np.mean(similarities)) if similarities else 1.0
            )
    expected_partition_count = None
    partition_selection_scores = list(partition_centrality)
    if reference_cluster_count is not None:
        expected_partition_count = max(
            int(reference_cluster_count) - len(protected_components),
            1,
        )
        target_weight = float(p["CONSENSUS_TARGET_COUNT_WEIGHT"])
        degenerate_penalty = float(p["CONSENSUS_DEGENERATE_PENALTY"])
        partition_selection_scores = []
        for centrality, labels in zip(partition_centrality, partitions):
            cluster_count = max(int(len(np.unique(labels))), 1)
            count_distance = abs(np.log(
                cluster_count / max(float(expected_partition_count), 1.0)
            ))
            score = centrality - target_weight * count_distance
            if expected_partition_count > 1 and cluster_count <= 1:
                score -= degenerate_penalty
            partition_selection_scores.append(float(score))
    if bool(p.get("CONSENSUS_SELECT_RESOLUTION", True)) and partition_selection_scores:
        best_index = min(
            range(len(partitions)),
            key=lambda index: (
                -partition_selection_scores[index],
                abs(partition_resolutions[index] - base_resolution),
                partition_resolutions[index],
            ),
        )
        selected_resolution = partition_resolutions[best_index]

    window_ratio = max(float(p["CONSENSUS_RESOLUTION_WINDOW_RATIO"]), 1.0)
    consensus_indices = [
        index
        for index, resolution in enumerate(partition_resolutions)
        if max(
            resolution / max(selected_resolution, 1e-12),
            selected_resolution / max(resolution, 1e-12),
        ) <= window_ratio
    ]
    if not consensus_indices and partitions:
        consensus_indices = [
            int(np.argmin(np.abs(
                np.asarray(partition_resolutions) - selected_resolution
            )))
        ]
    consensus_partitions = [partitions[index] for index in consensus_indices]
    consensus_graph = _consensus_reweight(graph, consensus_partitions, p)
    labels = _run_leiden(
        _graph_to_igraph(consensus_graph),
        selected_resolution,
        int(random_seed) + 1009,
        p,
    )
    labels = remap_labels(labels)
    if bool(p["MICRO_FORCE_SPLIT"]):
        labels = _force_protected_splits(labels, protected_components)
    diagnostics = {
        "base_resolution": float(base_resolution),
        "selected_resolution": float(selected_resolution),
        "ensemble_partition_count": int(len(partitions)),
        "ensemble_resolutions": partition_resolutions,
        "ensemble_partition_centrality": partition_centrality,
        "ensemble_partition_selection_scores": partition_selection_scores,
        "ensemble_cluster_counts": [int(len(np.unique(item))) for item in partitions],
        "expected_partition_count": expected_partition_count,
        "consensus_partition_indices": consensus_indices,
        "consensus_partition_resolutions": [
            partition_resolutions[index] for index in consensus_indices
        ],
        "consensus_n_clusters": int(len(np.unique(labels))),
    }
    return labels, consensus_graph, diagnostics


def _graph_round(
    adata,
    pseudo_labels,
    specificity,
    specificity_eligible,
    p,
    random_seed,
    previous_hybrid_score=None,
    reference_cluster_count=None,
    topology_selection_stage="initial",
):
    feature_views = select_hybrid_gene_views(
        adata,
        pseudo_labels,
        specificity,
        specificity_eligible,
        p,
        random_seed,
        previous_hybrid_score=previous_hybrid_score,
    )
    embeddings = build_three_embeddings(adata, feature_views, p, random_seed)
    marker_context = _prepare_marker_context(adata, feature_views)
    view_metrics = {
        "global": "euclidean",
        "discriminative": "cosine",
        "rare": "cosine",
    }
    view_graphs = {}
    core_graphs = {}
    view_diagnostics = {}
    view_states = {}
    for name in ["global", "discriminative", "rare"]:
        (
            view_graphs[name],
            core_graphs[name],
            view_diagnostics[name],
            view_states[name],
        ) = (
            build_adaptive_view_graph(
                embeddings[name],
                view_metrics[name],
                p,
                return_state=True,
            )
        )
    fused_graph, core_graph = fuse_view_graphs(view_graphs, core_graphs, p)
    graph_components, component_diagnostics = detect_micro_populations(
        fused_graph, core_graph, p
    )
    topology_components, topology_diagnostics = discover_topology_anchors(
        adata,
        fused_graph,
        core_graph,
        core_graphs,
        view_states,
        embeddings,
        feature_views,
        p,
        marker_context=marker_context,
        reference_cluster_count=reference_cluster_count,
        parent_labels=pseudo_labels,
        selection_stage=topology_selection_stage,
    )
    protected_components = _combine_nonoverlapping_components(
        topology_components,
        graph_components,
    )
    protected_graph = protect_micro_populations(
        fused_graph,
        protected_components,
        p,
    )
    forced_components = (
        topology_components
        if bool(p.get("TOPOLOGY_ANCHOR_ENABLED", True))
        else graph_components
    )
    labels, consensus_graph, clustering_diagnostics = consensus_cluster(
        protected_graph,
        forced_components,
        p,
        random_seed,
        reference_cluster_count=reference_cluster_count,
    )
    hidden_components, local_refine_diagnostics = discover_hidden_micro_populations(
        labels,
        consensus_graph,
        embeddings["rare"],
        clustering_diagnostics["selected_resolution"],
        p,
        random_seed,
        marker_context=marker_context,
        view_states=view_states,
        reference_cluster_count=reference_cluster_count,
    )
    labels = _force_protected_splits(labels, hidden_components)
    clustering_diagnostics["post_local_refine_n_clusters"] = int(
        len(np.unique(labels))
    )
    cleanup_protected = _combine_nonoverlapping_components(
        topology_components,
        hidden_components,
    )
    labels, cleanup_diagnostics = merge_unsupported_microclusters(
        labels,
        consensus_graph,
        marker_context,
        view_states,
        cleanup_protected,
        p,
    )
    clustering_diagnostics["post_cleanup_n_clusters"] = int(
        len(np.unique(labels))
    )
    diagnostics = {
        "feature_counts": {
            key: feature_views[key]
            for key in ["n_hvg", "n_hig", "n_discriminative", "n_rare_genes"]
        },
        "view_graphs": view_diagnostics,
        "micro_candidates": component_diagnostics,
        "graph_micro_sizes": [int(len(cells)) for cells in graph_components],
        "topology_anchors": topology_diagnostics,
        "forced_topology_sizes": [
            int(len(cells)) for cells in forced_components
        ],
        "protected_micro_sizes": [
            int(len(cells)) for cells in protected_components
        ],
        "local_refine_candidates": local_refine_diagnostics,
        "local_refine_accepted_sizes": [
            int(len(cells)) for cells in hidden_components
        ],
        "cleanup": cleanup_diagnostics,
        "clustering": clustering_diagnostics,
        "fused_edge_count": int(fused_graph.nnz // 2),
        "protected_edge_count": int(protected_graph.nnz // 2),
    }
    return {
        "labels": labels,
        "hybrid_score": feature_views["hybrid_score"],
        "rare_score": feature_views["rare_score"],
        "hig_mask": np.asarray(feature_views["hig_mask"], dtype=bool),
        "rare_gene_mask": np.asarray(feature_views["rare_mask"], dtype=bool),
        "embeddings": embeddings,
        "graph": consensus_graph,
        "diagnostics": diagnostics,
    }


def run_archer_v2_pipeline(
    counts,
    true_labels=None,
    already_log=False,
    params=None,
    return_details=False,
    random_seed=None,
):
    """Run the label-free ArCHeR v2 research pipeline."""
    p = get_default_params(params)
    seed = int(p["RANDOM_SEED"] if random_seed is None else random_seed)
    adata = _prepare_adata(counts, already_log=already_log)
    if true_labels is not None and len(true_labels) != adata.n_obs:
        raise ValueError(
            "true_labels length mismatch: {} labels for {} cells".format(
                len(true_labels), adata.n_obs
            )
        )
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("ArCHeR v2 requires a non-empty cell-by-gene matrix.")

    highly_variable_genes_inplace(
        adata,
        min_mean=p["HVG_MIN_MEAN"],
        max_mean=p["HVG_MAX_MEAN"],
        min_disp=p["HVG_MIN_DISP"],
        n_top_genes=p["HVG_TOPN"],
    )
    specificity, specificity_eligible, _ = compute_rare_gene_specificity(adata, p)
    initial_labels, dpc_diagnostics = run_rare_aware_dpc(
        adata,
        specificity,
        p,
        seed,
    )

    rounds = []
    first_round = _graph_round(
        adata,
        initial_labels,
        specificity,
        specificity_eligible,
        p,
        seed + 101,
        reference_cluster_count=int(len(np.unique(initial_labels))),
        topology_selection_stage="initial",
    )
    rounds.append(first_round)
    current = first_round
    round_acceptance = [True]
    round_rejection_reasons = [None]
    round_supported_new_splits = [0]
    round_maximum_allowed_cluster_count = [
        int(len(np.unique(first_round["labels"])))
    ]
    for iteration in range(max(int(p["REFINE_ITERATIONS"]), 0)):
        candidate = _graph_round(
            adata,
            current["labels"],
            specificity,
            specificity_eligible,
            p,
            seed + 1001 + iteration * 313,
            previous_hybrid_score=current["hybrid_score"],
            reference_cluster_count=int(len(np.unique(initial_labels))),
            topology_selection_stage="refine",
        )
        rounds.append(candidate)
        partition_ari = float(
            adjusted_rand_score(current["labels"], candidate["labels"])
        )
        previous_count = max(int(len(np.unique(current["labels"]))), 1)
        candidate_count = max(int(len(np.unique(candidate["labels"]))), 1)
        count_ratio = candidate_count / float(previous_count)
        topology_diagnostics = candidate["diagnostics"]["topology_anchors"]
        supported_new_splits = int(
            topology_diagnostics.get("selected_discovery_count", 0)
        ) + int(len(
            candidate["diagnostics"].get("local_refine_accepted_sizes", [])
        ))
        ratio_allowed_count = int(np.floor(
            previous_count * float(p["REFINE_MAX_CLUSTER_COUNT_RATIO"])
            + 1e-8
        ))
        evidence_allowed_count = previous_count + supported_new_splits
        maximum_allowed_count = max(
            ratio_allowed_count,
            evidence_allowed_count,
        )
        round_supported_new_splits.append(int(supported_new_splits))
        round_maximum_allowed_cluster_count.append(int(maximum_allowed_count))
        rejection_reasons = []
        if partition_ari < float(p["REFINE_MIN_PARTITION_ARI"]):
            rejection_reasons.append("partition_instability")
        if count_ratio < float(p["REFINE_MIN_CLUSTER_COUNT_RATIO"]):
            rejection_reasons.append("cluster_collapse")
        if candidate_count > maximum_allowed_count:
            rejection_reasons.append("cluster_explosion")
        accepted = len(rejection_reasons) == 0
        round_acceptance.append(bool(accepted))
        round_rejection_reasons.append(
            rejection_reasons if rejection_reasons else None
        )
        if not accepted:
            break
        current = candidate

    labels = remap_labels(current["labels"])
    adata.obs["archer_v2_cluster"] = labels.astype(str)
    if not return_details:
        return labels

    details = {
        "method_name": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "random_seed": int(seed),
        "params": deepcopy(p),
        "initial_labels": initial_labels,
        "initial_n_clusters": int(len(np.unique(initial_labels))),
        "final_n_clusters": int(len(np.unique(labels))),
        "dpc": dpc_diagnostics,
        "rounds": [item["diagnostics"] for item in rounds],
        "round_cluster_counts": [
            int(len(np.unique(item["labels"]))) for item in rounds
        ],
        "round_partition_ari": [
            float(adjusted_rand_score(rounds[index - 1]["labels"], rounds[index]["labels"]))
            for index in range(1, len(rounds))
        ],
        "round_acceptance": round_acceptance,
        "round_rejection_reasons": round_rejection_reasons,
        "round_supported_new_splits": round_supported_new_splits,
        "round_maximum_allowed_cluster_count": (
            round_maximum_allowed_cluster_count
        ),
        "selected_round_index": int(
            max(index for index, accepted in enumerate(round_acceptance) if accepted)
        ),
        "hig_mask": np.asarray(current["hig_mask"], dtype=bool),
        "rare_gene_mask": np.asarray(current["rare_gene_mask"], dtype=bool),
        "hybrid_score": np.asarray(current["hybrid_score"], dtype=np.float32),
        "rare_score": np.asarray(current["rare_score"], dtype=np.float32),
    }
    return labels, details


# Familiar alias for downstream scripts while keeping the module name explicit.
run_archer_pipeline = run_archer_v2_pipeline


__all__ = [
    "METHOD_NAME",
    "METHOD_VERSION",
    "DEFAULT_PARAMS",
    "get_default_params",
    "run_archer_v2_pipeline",
    "run_archer_pipeline",
]
