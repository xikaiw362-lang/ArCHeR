import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import anndata as ad
import igraph as ig
from scipy.sparse import issparse, csr_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree

logger = logging.getLogger("ArCHeR")

class Config:
    MAX_ITER      = 1
    HVG_MIN_MEAN  = 0.0125
    HVG_MAX_MEAN  = 3
    HVG_MIN_DISP  = 0.5
    HVG_TOPN      = 2000

    DPC_DC_PERCENT       = 8
    DPC_GAMMA_TOPK       = 8
    DPC_PCA_N_COMPONENTS = 40
    DPC_KNN_K            = 200

    RF_N_ESTIMATORS = 100
    RF_HIG_TOPN     = 2000
    RF_PCA_DIMS     = 50

    INIT_KNN_K           = 20
    INIT_LOUVAIN_RES     = 1.0

    AKNN_KMAX        = 12
    AKNN_PRUNE_SNN   = 1 / 20
    AKNN_GRAPH_MODE  = "default"
    AKNN_PCA_DIMS    = list(range(1, 31))
    AKNN_DELTA_RANGE = np.arange(-1.0, 0.1, 0.1)
    AKNN_CUTOFF      = 5
    POST_KNN_K       = 20
    POST_LOUVAIN_RES = 1.0
    USE_TARGET_CLUSTER_COUNT = False
    POST_MERGE_SMALL_CLUSTERS = False
    POST_MERGE_MIN_CLUSTER_SIZE = 25
    POST_MERGE_MIN_CLUSTER_FRAC = 0.01
    POST_MERGE_MAX_CLUSTER_FRAC = 0.35
    POST_MERGE_SIM_THRESHOLD = 0.12
    POST_MERGE_SHARED_EDGE_THRESHOLD = 0.08


def normalize_total_inplace(adata, target_sum=1e4):
    X = adata.X
    if issparse(X):
        totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        scale = np.divide(target_sum, totals, out=np.ones_like(totals), where=totals > 0)
        adata.X = X.multiply(scale[:, None]).tocsr()
        return

    totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    scale = np.divide(target_sum, totals, out=np.ones_like(totals), where=totals > 0)
    adata.X = (np.asarray(X, dtype=np.float32) * scale[:, None]).astype(np.float32, copy=False)


def log1p_inplace(adata):
    X = adata.X
    if issparse(X):
        X = X.tocsr(copy=True)
        X.data = np.log1p(X.data)
        adata.X = X
        return
    adata.X = np.log1p(np.asarray(X, dtype=np.float32)).astype(np.float32, copy=False)


def mean_var(X):
    if issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = np.maximum(mean_sq - np.square(mean), 0.0)
        return mean.astype(np.float32), var.astype(np.float32)

    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    return mean.astype(np.float32), var.astype(np.float32)


def highly_variable_genes_inplace(adata, min_mean, max_mean, min_disp, n_top_genes):
    means, vars_ = mean_var(adata.X)
    disp = vars_ / np.maximum(means, 1e-12)
    disp = np.log1p(disp)
    mean_log = np.log1p(means)

    valid = np.isfinite(mean_log) & np.isfinite(disp)
    disp_norm = np.full(adata.n_vars, -np.inf, dtype=np.float32)

    if valid.any():
        n_bins = min(20, max(int(valid.sum()), 1))
        try:
            bins = pd.qcut(mean_log[valid], q=n_bins, duplicates="drop")
        except ValueError:
            bins = pd.cut(mean_log[valid], bins=max(min(n_bins, 20), 1), duplicates="drop")

        disp_series = pd.Series(disp[valid])
        bin_series = pd.Series(bins)
        bin_mean = disp_series.groupby(bin_series).transform("mean").to_numpy()
        bin_std = disp_series.groupby(bin_series).transform("std").fillna(0.0).to_numpy()
        disp_norm_valid = (disp[valid] - bin_mean) / np.where(bin_std > 0, bin_std, 1.0)
        disp_norm[valid] = disp_norm_valid.astype(np.float32, copy=False)

    candidate = valid & (means >= min_mean) & (means <= max_mean)
    if not candidate.any():
        candidate = valid

    hv_mask = candidate & (disp_norm >= min_disp)
    target_n = min(n_top_genes, int(candidate.sum()))
    if target_n <= 0:
        hv_mask = np.zeros(adata.n_vars, dtype=bool)
    elif hv_mask.sum() != target_n:
        scores = np.where(candidate, disp_norm, -np.inf)
        top_idx = np.argsort(-scores)[:target_n]
        hv_mask = np.zeros(adata.n_vars, dtype=bool)
        hv_mask[top_idx] = True

    adata.var["means"] = means
    adata.var["dispersions"] = disp
    adata.var["dispersions_norm"] = disp_norm
    adata.var["highly_variable"] = hv_mask


def scale_dense_matrix(X, max_value=10):
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X = (X - mean) / std
    if max_value is not None:
        X = np.clip(X, -max_value, max_value)
    return X.astype(np.float32, copy=False)


def compute_pca_embedding(X, n_comps):
    if issparse(X):
        X = X.astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)
    svd = TruncatedSVD(n_components=n_comps, random_state=42)
    return svd.fit_transform(X).astype(np.float32, copy=False)


def preprocess_hvg(adata, already_log=False):
    if not already_log:
        normalize_total_inplace(adata, target_sum=1e4)
        log1p_inplace(adata)
    highly_variable_genes_inplace(adata, min_mean=Config.HVG_MIN_MEAN,
                                  max_mean=Config.HVG_MAX_MEAN,
                                  min_disp=Config.HVG_MIN_DISP,
                                  n_top_genes=Config.HVG_TOPN)
    return adata

def compute_local_density_fast(X, dc, k=200):
    tree = KDTree(X)
    distances, _ = tree.query(X, k=k)
    rho = np.sum(np.exp(-(distances / dc) ** 2), axis=1).astype(np.float32)
    return rho

def compute_min_distance(X, rho, nn_dists, nn_indices):
    n = len(rho)
    delta = np.zeros(n, dtype=np.float32)
    nneigh = np.zeros(n, dtype=int)
    idx_rho = np.argsort(-rho)
    rank = np.empty(n, dtype=int)
    rank[idx_rho] = np.arange(n)

    densest = idx_rho[0]
    delta[densest] = np.linalg.norm(X - X[densest], axis=1).max()
    nneigh[densest] = densest

    for i in range(1, n):
        ci = idx_rho[i]
        neighbor_ids = nn_indices[ci, 1:]
        neighbor_dists = nn_dists[ci, 1:]
        valid_mask = rank[neighbor_ids] < i

        if np.any(valid_mask):
            valid_ids = neighbor_ids[valid_mask]
            valid_dists = neighbor_dists[valid_mask]
            best = np.argmin(valid_dists)
            delta[ci] = valid_dists[best]
            nneigh[ci] = valid_ids[best]
            continue

        higher_list = idx_rho[:i]
        dists = np.linalg.norm(X[higher_list] - X[ci], axis=1)
        best = np.argmin(dists)
        delta[ci] = dists[best]
        nneigh[ci] = higher_list[best]

    return delta, nneigh


def run_dpc(adata):
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    X_scaled = adata_hvg.X
    if issparse(X_scaled):
        X_scaled = X_scaled.toarray()
    X_scaled = scale_dense_matrix(X_scaled, max_value=10)
    
    n_pcs = min(Config.DPC_PCA_N_COMPONENTS, adata_hvg.shape[0]-1, adata_hvg.shape[1]-1)
    X = compute_pca_embedding(X_scaled, n_pcs)
    adata_hvg.obsm["X_pca"] = X

    k = min(Config.DPC_KNN_K, X.shape[0] - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
    nbrs.fit(X)
    nn_dists, nn_indices = nbrs.kneighbors(X)
    dc = np.percentile(nn_dists[:, 1:].ravel(), Config.DPC_DC_PERCENT)

    rho = compute_local_density_fast(X, dc, k=k)
    delta, nneigh = compute_min_distance(X, rho, nn_dists, nn_indices)

    gamma = rho * delta
    center_idx = np.argsort(-gamma)[:Config.DPC_GAMMA_TOPK]

    labels = -np.ones(adata.n_obs, dtype=int)
    for i, idx in enumerate(center_idx):
        labels[idx] = i
    for i in np.argsort(-rho):
        if labels[i] == -1:
            labels[i] = labels[nneigh[i]]

    adata.obs["initial_cluster"] = labels.astype(str)
    return adata


def run_knn_louvain_init(adata):
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    X = adata_hvg.X
    if issparse(X):
        X = X.toarray()
    X = scale_dense_matrix(X, max_value=10)

    n_pcs = min(Config.DPC_PCA_N_COMPONENTS, adata_hvg.shape[0] - 1, adata_hvg.shape[1] - 1)
    X = compute_pca_embedding(X, n_pcs)

    k = min(Config.INIT_KNN_K, max(2, X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    nn_dists, nn_indices = nn.kneighbors(X)

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
    comm = g.community_multilevel(weights=g.es["weight"], resolution=Config.INIT_LOUVAIN_RES)
    adata.obs["initial_cluster"] = np.asarray(comm.membership, dtype=int).astype(str)
    return adata

def run_rf_hig(adata):
    X_raw = adata.X
    y = adata.obs["initial_cluster"].values

    n_pca = min(Config.RF_PCA_DIMS, X_raw.shape[1] - 1, X_raw.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_pca, random_state=42)
    X_pca = svd.fit_transform(X_raw)

    rf = RandomForestClassifier(n_estimators=Config.RF_N_ESTIMATORS,
                                random_state=42, n_jobs=-1, max_features='sqrt')
    rf.fit(X_pca, y)

    gene_importance = np.abs(svd.components_).T @ rf.feature_importances_
    top_idx = np.argsort(-gene_importance)[:Config.RF_HIG_TOPN]
    hig_names = adata.var_names[top_idx]
    adata.var["is_HIG"] = False
    adata.var.loc[hig_names, "is_HIG"] = True

    hvg = set(adata.var[adata.var.highly_variable].index)
    hig = set(hig_names)
    overlap = len(hvg & hig) / min(len(hvg), len(hig)) if min(len(hvg), len(hig)) > 0 else 0
    return adata, overlap

def find_neighbors_aknn(adata, delta=-0.8, kmax=10, dims=None,
                        prune_snn=1/15, reduction="X_pca",
                        precomputed_nn=None):
    if dims is None:
        dims = Config.AKNN_PCA_DIMS

    pca_dims = [d - 1 for d in dims if d - 1 < adata.obsm[reduction].shape[1]]
    X = adata.obsm[reduction][:, pca_dims]
    n_cells = X.shape[0]

    if precomputed_nn is not None:
        nn_dists, nn_indices = precomputed_nn
    else:
        nn = NearestNeighbors(n_neighbors=kmax + 1, metric="euclidean", n_jobs=-1)
        nn.fit(X)
        nn_dists, nn_indices = nn.kneighbors(X)

    rows, cols = [], []
    for i in range(n_cells):
        dists = nn_dists[i, 1:]
        threshold = (np.sum(np.sqrt(dists)) / (kmax - 1 - delta)) ** 2
        valid = nn_indices[i, 1:][dists < threshold]
        rows.extend([i] * len(valid))
        cols.extend(valid.tolist())

    adj = csr_matrix((np.ones(len(rows), dtype=np.float32),
                      (rows, cols)), shape=(n_cells, n_cells))
    if Config.AKNN_GRAPH_MODE == "symmetric":
        adj = adj.maximum(adj.T)
    elif Config.AKNN_GRAPH_MODE == "mutual":
        adj = adj.minimum(adj.T)

    snn = adj.dot(adj.T).tocsr()
    row_sums = np.array(adj.sum(axis=1)).flatten()
    cx = snn.tocoo()
    union_vals = row_sums[cx.row] + row_sums[cx.col] - cx.data
    union_vals = np.where(union_vals == 0, 1.0, union_vals)
    snn_jaccard = csr_matrix((cx.data / union_vals, (cx.row, cx.col)), shape=snn.shape)
    snn_jaccard.data[snn_jaccard.data <= prune_snn] = 0.0
    snn_jaccard.eliminate_zeros()

    adata.obsp["aknn_connectivities"] = snn_jaccard
    return adata

def find_optimal_delta(adata, delta_range=None, cutoff=5, kmax=10, dims=None,
                       prune_snn=1/15, reduction="X_pca"):
    if delta_range is None:
        delta_range = Config.AKNN_DELTA_RANGE
    if dims is None:
        dims = Config.AKNN_PCA_DIMS

    pca_dims = [d - 1 for d in dims if d - 1 < adata.obsm[reduction].shape[1]]
    X = adata.obsm[reduction][:, pca_dims]

    nn = NearestNeighbors(n_neighbors=kmax + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    precomputed_nn = nn.kneighbors(X)

    delta_metrics = {}
    for delta in delta_range:
        adata = find_neighbors_aknn(adata, delta=delta, kmax=kmax, dims=dims,
                                    prune_snn=prune_snn, reduction=reduction,
                                    precomputed_nn=precomputed_nn)
        snn = adata.obsp["aknn_connectivities"]
        src, tgt = snn.nonzero()
        g = ig.Graph(n=snn.shape[0], directed=False)
        g.add_edges(list(zip(src.tolist(), tgt.tolist())))
        g.es["weight"] = snn.data.tolist()
        comm = g.community_multilevel(weights=g.es["weight"], resolution=1.0)
        sizes = pd.Series(comm.membership).value_counts()
        delta_metrics[delta] = {"n_communities": len(sizes),
                                 "n_singletons": int((sizes == 1).sum())}

    delta_list = sorted(delta_metrics.keys())
    comm_list = [delta_metrics[d]["n_communities"] for d in delta_list]
    diffs = np.diff(comm_list)
    opt_idx = next((i + 1 for i, d in enumerate(diffs) if d <= cutoff), None)
    if opt_idx is None:
        opt_idx = int(np.argmin([delta_metrics[d]["n_singletons"] for d in delta_list]))
    optimal_delta = delta_list[opt_idx]

    return optimal_delta, precomputed_nn

def merge_excess_clusters(labels, X_pca, target_n):
    labels = np.array(labels, dtype=int)
    current_n = len(np.unique(labels))
    if current_n <= target_n:
        return labels

    while len(np.unique(labels)) > target_n:
        unique, counts = np.unique(labels, return_counts=True)
        smallest = unique[np.argmin(counts)]
        mask_small = labels == smallest
        center_small = X_pca[mask_small].mean(axis=0)
        best_label, best_dist = None, np.inf
        for u in unique:
            if u == smallest:
                continue
            center_u = X_pca[labels == u].mean(axis=0)
            d = np.linalg.norm(center_small - center_u)
            if d < best_dist:
                best_dist, best_label = d, u
        labels[mask_small] = best_label

    unique_final = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique_final)}
    labels = np.array([remap[l] for l in labels])
    return labels

def find_optimal_resolution(adata_merge, target_n_clusters, res_range=(0.1, 2.0), adjacency_key="aknn_connectivities"):
    adj = adata_merge.obsp[adjacency_key]
    src, tgt = adj.nonzero()
    g = ig.Graph(n=adj.shape[0], directed=False)
    g.add_edges(list(zip(src.tolist(), tgt.tolist())))
    g.es["weight"] = adj.data.tolist()

    lo, hi = res_range
    best_res, best_diff, best_labels = 1.0, np.inf, None

    for _ in range(15):
        mid = (lo + hi) / 2.0
        comm = g.community_multilevel(weights=g.es["weight"], resolution=mid)
        n_clusters = len(set(comm.membership))
        diff = abs(n_clusters - target_n_clusters)
        if diff < best_diff:
            best_diff, best_res = diff, mid
            best_labels = np.array(comm.membership)
        if diff <= 1:
            break
        elif n_clusters > target_n_clusters:
            hi = mid
        else:
            lo = mid

    return best_res, best_labels


def run_multilevel_clustering(snn, resolution):
    src, tgt = snn.nonzero()
    g = ig.Graph(n=snn.shape[0], directed=False)
    g.add_edges(list(zip(src.tolist(), tgt.tolist())))
    g.es["weight"] = snn.data.tolist()
    comm = g.community_multilevel(weights=g.es["weight"], resolution=resolution)
    return np.array(comm.membership)


def remap_labels(labels):
    unique_final = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique_final)}
    return np.array([remap[l] for l in labels], dtype=int)


def merge_small_clusters(labels, X_pca, snn, min_cluster_size, max_cluster_frac,
                         sim_threshold, shared_edge_threshold):
    labels = np.array(labels, dtype=int).copy()
    n_cells = len(labels)
    if n_cells == 0:
        return labels

    max_cluster_size = max(int(n_cells * max_cluster_frac), min_cluster_size + 1)
    snn_coo = snn.tocoo()

    while True:
        unique, counts = np.unique(labels, return_counts=True)
        count_map = {u: int(c) for u, c in zip(unique, counts)}
        center_map = {u: X_pca[labels == u].mean(axis=0) for u in unique}

        edge_weights = {}
        for src, dst, val in zip(snn_coo.row, snn_coo.col, snn_coo.data):
            ls = labels[src]
            ld = labels[dst]
            if ls == ld:
                continue
            key = (ls, ld) if ls < ld else (ld, ls)
            edge_weights[key] = edge_weights.get(key, 0.0) + float(val)

        small_labels = [u for u, c in zip(unique, counts) if c < min_cluster_size]
        if not small_labels:
            break

        merged = False
        current_unique = list(unique)
        for small in sorted(small_labels, key=lambda x: count_map[x]):
            small_size = count_map[small]
            if small_size == 0:
                continue

            center_small = center_map[small]
            best_label = None
            best_score = -np.inf

            for other in current_unique:
                if other == small:
                    continue
                other_size = count_map[other]
                if other_size == 0 or other_size > max_cluster_size:
                    continue

                center_other = center_map[other]
                dist = float(np.linalg.norm(center_small - center_other))
                key = (small, other) if small < other else (other, small)
                edge_weight = edge_weights.get(key, 0.0)
                shared_ratio = edge_weight / max(small_size, 1)
                score = shared_ratio - dist

                if shared_ratio >= shared_edge_threshold and dist <= sim_threshold and score > best_score:
                    best_score = score
                    best_label = other

            if best_label is not None:
                labels[labels == small] = best_label
                labels = remap_labels(labels)
                merged = True
                break

        if not merged:
            break

    return remap_labels(labels)

def run_aknno(adata):
    merge_mask = adata.var.highly_variable | adata.var["is_HIG"]
    adata_merge = adata[:, merge_mask].copy()
    merge_num = adata_merge.n_vars
    if merge_num < 50:
        return adata

    n_pcs = min(30, merge_num - 1, adata_merge.n_obs - 1)
    X_scaled = adata_merge.X
    if issparse(X_scaled):
        X_scaled = X_scaled.toarray()
    X_scaled = scale_dense_matrix(X_scaled, max_value=10)
    adata_merge.obsm["X_pca"] = compute_pca_embedding(X_scaled, n_pcs)

    optimal_delta, precomputed_nn = find_optimal_delta(
        adata_merge,
        delta_range=Config.AKNN_DELTA_RANGE,
        cutoff=Config.AKNN_CUTOFF,
        kmax=Config.AKNN_KMAX,
        dims=Config.AKNN_PCA_DIMS,
        prune_snn=Config.AKNN_PRUNE_SNN,
        reduction="X_pca"
    )
    adata_merge = find_neighbors_aknn(
        adata_merge, delta=optimal_delta, kmax=Config.AKNN_KMAX,
        dims=Config.AKNN_PCA_DIMS, prune_snn=Config.AKNN_PRUNE_SNN,
        reduction="X_pca", precomputed_nn=precomputed_nn
    )
    snn = adata_merge.obsp["aknn_connectivities"]
    use_target_n = Config.USE_TARGET_CLUSTER_COUNT and "true_label" in adata.obs.columns
    if use_target_n:
        target_n = int(adata.obs["true_label"].nunique())
        _, best_labels = find_optimal_resolution(adata_merge, target_n)
        X_pca = adata_merge.obsm["X_pca"]
        best_labels = merge_excess_clusters(best_labels, X_pca, target_n)
    else:
        res = np.clip(0.28 + np.log10(max(adata.n_obs / 5000, 1)) * 0.22, 0.18, 1.2)
        best_labels = run_multilevel_clustering(snn, resolution=res)
        if Config.POST_MERGE_SMALL_CLUSTERS:
            min_cluster_size = max(
                Config.POST_MERGE_MIN_CLUSTER_SIZE,
                int(np.ceil(adata.n_obs * Config.POST_MERGE_MIN_CLUSTER_FRAC))
            )
            X_pca = adata_merge.obsm["X_pca"]
            best_labels = merge_small_clusters(
                best_labels,
                X_pca,
                snn,
                min_cluster_size=min_cluster_size,
                max_cluster_frac=Config.POST_MERGE_MAX_CLUSTER_FRAC,
                sim_threshold=Config.POST_MERGE_SIM_THRESHOLD,
                shared_edge_threshold=Config.POST_MERGE_SHARED_EDGE_THRESHOLD
            )

    best_labels = remap_labels(best_labels)
    adata.obs["D_cluster"] = best_labels
    return adata


def run_knn_louvain_post(adata, use_hig=True):
    if "is_HIG" not in adata.var.columns:
        adata.var["is_HIG"] = False
    merge_mask = adata.var.highly_variable | adata.var["is_HIG"] if use_hig else adata.var.highly_variable
    adata_merge = adata[:, merge_mask].copy()
    merge_num = adata_merge.n_vars
    if merge_num < 50:
        if "initial_cluster" not in adata.obs:
            raise ValueError("No initial clusters available for fallback post-clustering.")
        adata.obs["D_cluster"] = pd.to_numeric(
            adata.obs["initial_cluster"], errors="coerce"
        ).fillna(0).astype(int).values
        return adata

    n_pcs = min(30, merge_num - 1, adata_merge.n_obs - 1)
    X = adata_merge.X
    if issparse(X):
        X = X.toarray()
    X = scale_dense_matrix(X, max_value=10)
    X_pca = compute_pca_embedding(X, n_pcs)

    k = min(Config.POST_KNN_K, max(2, X_pca.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X_pca)
    nn_dists, nn_indices = nn.kneighbors(X_pca)

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
    comm = g.community_multilevel(weights=g.es["weight"], resolution=Config.POST_LOUVAIN_RES)
    adata.obs["D_cluster"] = remap_labels(np.asarray(comm.membership, dtype=int))
    return adata


def run_archer_pipeline(counts_df, true_labels=None, already_log=False,
                        use_dpc=True, use_rf_hig=True, use_aknno=True):
    if isinstance(counts_df, ad.AnnData):
        adata = counts_df
        # Ensure X is sparse to save memory
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

    adata = preprocess_hvg(adata, already_log=already_log)

    if use_dpc:
        adata = run_dpc(adata)
    else:
        if use_rf_hig:
            adata = run_knn_louvain_init(adata)
        else:
            adata.obs["initial_cluster"] = np.zeros(adata.n_obs, dtype=int).astype(str)

    if use_rf_hig:
        adata, _ = run_rf_hig(adata)
    else:
        adata.var["is_HIG"] = False

    if use_aknno:
        adata = run_aknno(adata)
    else:
        adata = run_knn_louvain_post(adata, use_hig=use_rf_hig)

    return adata.obs["D_cluster"].values
