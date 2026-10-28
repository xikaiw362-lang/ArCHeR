"""Run label-free GO/pathway enrichment for small ArCHeR chicken communities.

Candidate genes are selected without author labels: genes are ranked by the
difference in mean log-normalized expression between one ArCHeR community and
all remaining cells.  g:Profiler uses Gallus gallus annotations and the genes
detected in this dataset as the custom background.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.io import mmread


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "case_study_datasets" / "G_G_00018"
OUT_DIR = DATA_DIR / "exported"
SOURCE_DIR = ROOT / "paper" / "fig6_chicken_case" / "source_data"
API_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"

CLUSTERS = {
    4: "SOX5+ B cell",
    10: "ATP11A+ B-cell-associated",
    12: "Cycling lymphoid",
    13: "HDC+GATA2+ cell",
    14: "HBBA+ erythroid-like",
}
NON_INFORMATIVE_PREFIXES = ("RPL", "RPS", "MRPL", "MRPS", "MT-", "MTRNR")
NON_INFORMATIVE_GENES = {"ACTB", "GAPDH", "PPIA", "TPT1", "B2M", "FTH1", "FTL"}


def load_data():
    prediction = pd.read_csv(OUT_DIR / "archer_predictions_repaired_final.csv")
    with gzip.open(DATA_DIR / "GSE224329_final_matrix.mtx.gz.new", "rb") as handle:
        counts = mmread(handle).tocsr().T.astype(np.float32)
    with gzip.open(DATA_DIR / "GSE224329_final_feats.tsv.gz", "rt", encoding="utf-8") as handle:
        genes = [line.rstrip("\n").split("\t")[0] for line in handle]
    if counts.shape != (len(prediction), len(genes)):
        raise ValueError("Matrix dimensions do not match prediction or feature files")
    return prediction["archer_cluster"].to_numpy(dtype=int), counts, genes


def log_normalize(counts):
    library_size = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    library_size[library_size == 0] = 1.0
    normalized = counts.multiply((1e4 / library_size)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized


def select_marker_genes(norm, labels, genes, cluster, max_genes=150):
    inside = labels == cluster
    n_inside = int(inside.sum())
    mean_inside = np.asarray(norm[inside].mean(axis=0)).ravel()
    mean_outside = np.asarray(norm[~inside].mean(axis=0)).ravel()
    fraction_inside = np.asarray(norm[inside].getnnz(axis=0)).ravel() / n_inside
    delta = mean_inside - mean_outside
    candidate = np.flatnonzero(
        (delta >= 0.25) & (mean_inside >= 0.10) & (fraction_inside >= 0.10)
    )
    candidate = candidate[np.argsort(delta[candidate])[::-1]]
    candidate = np.asarray(
        [
            index for index in candidate
            if genes[index] not in NON_INFORMATIVE_GENES
            and not genes[index].startswith(NON_INFORMATIVE_PREFIXES)
        ],
        dtype=int,
    )[:max_genes]
    frame = pd.DataFrame(
        {
            "archer_cluster": cluster,
            "identity": CLUSTERS[cluster],
            "gene": [genes[index] for index in candidate],
            "mean_log_expression_in_cluster": mean_inside[candidate],
            "mean_log_expression_outside_cluster": mean_outside[candidate],
            "mean_expression_difference": delta[candidate],
            "fraction_expressing_in_cluster": fraction_inside[candidate],
            "selection_rule": "mean_difference>=0.25, mean>=0.10, detection>=0.10; ribosomal/mitochondrial/housekeeping genes excluded",
        }
    )
    return frame


def gprofiler_profile(query_genes, background_genes):
    payload = {
        "organism": "ggallus",
        "query": list(query_genes),
        "background": list(background_genes),
        "domain_scope": "custom",
        "sources": ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"],
        "user_threshold": 0.05,
        "significance_threshold_method": "g_SCS",
        "no_evidences": True,
    }
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "ArCHeR-case-study/1.0"},
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    labels, counts, genes = load_data()
    norm = log_normalize(counts)
    detected = np.asarray(counts.getnnz(axis=0)).ravel() > 0
    background_genes = [gene for gene, present in zip(genes, detected) if present]

    marker_frames = []
    term_frames = []
    diagnostics = []
    for cluster in CLUSTERS:
        markers = select_marker_genes(norm, labels, genes, cluster)
        marker_frames.append(markers)
        response = gprofiler_profile(markers["gene"].tolist(), background_genes)
        metadata = response["meta"]["genes_metadata"]["query"]["query_1"]
        mapped = len(metadata.get("mapping", {}))
        failed = metadata.get("failed", [])
        terms = pd.DataFrame(response.get("result", []))
        if not terms.empty:
            terms = terms.loc[terms["significant"].astype(bool)].copy()
            terms["archer_cluster"] = cluster
            terms["identity"] = CLUSTERS[cluster]
            terms["minus_log10_p_value"] = -np.log10(terms["p_value"].astype(float))
            term_frames.append(terms)
        diagnostics.append(
            {
                "archer_cluster": cluster,
                "identity": CLUSTERS[cluster],
                "n_cells": int((labels == cluster).sum()),
                "n_input_genes": int(len(markers)),
                "marker_selection_rule": "mean_difference>=0.25, mean>=0.10, detection>=0.10; ribosomal/mitochondrial/housekeeping genes excluded",
                "n_mapped_input_genes": int(mapped),
                "input_mapping_fraction": float(mapped / max(1, len(markers))),
                "n_unmapped_input_genes": int(len(failed)),
                "n_significant_terms": int(0 if terms.empty else len(terms)),
                "background_gene_count": int(len(background_genes)),
                "gprofiler_database_version": response["meta"].get("version", "unknown"),
            }
        )

    pd.concat(marker_frames, ignore_index=True).to_csv(
        SOURCE_DIR / "chicken_small_community_upregulated_genes.csv", index=False
    )
    pd.DataFrame(diagnostics).to_csv(
        SOURCE_DIR / "chicken_small_community_enrichment_diagnostics.csv", index=False
    )
    if term_frames:
        pd.concat(term_frames, ignore_index=True).to_csv(
            SOURCE_DIR / "chicken_small_community_gprofiler_terms.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(
            SOURCE_DIR / "chicken_small_community_gprofiler_terms.csv", index=False
        )


if __name__ == "__main__":
    main()
