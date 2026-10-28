"""Generate label-free marker and functional evidence for the zebrafish case.

The HIG500 partition is fixed before this script is run.  Reference labels are
not used to construct its clusters or select marker genes; they are used only
to give post-hoc display names to the four reported communities.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.io import mmread


ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = ROOT / "case_study_datasets" / "D_R_00002" / "exported"
SOURCE_DIR = ROOT / "paper" / "fig5" / "source_data"
API_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"

# Names state the observed post-hoc enrichment, rather than claiming identity.
COMMUNITIES = {
    5: "HE-enriched community",
    13: "Lymphoid-HSC-enriched community",
    18: "Neural-cell-3 community",
    26: "Pigment-cell community",
}
NON_INFORMATIVE_PREFIXES = ("rpl", "rps", "mrpl", "mrps", "mt-")
NON_INFORMATIVE_GENES = {"actb1", "actb2", "gapdh", "ef1a1", "tuba1b", "b2m"}


def log_normalize(counts):
    library = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    library[library == 0] = 1.0
    norm = counts.multiply((1e4 / library)[:, None]).tocsr()
    norm.data = np.log1p(norm.data)
    return norm


def select_markers(norm, labels, genes, cluster, max_genes=150):
    inside = labels == cluster
    mean_in = np.asarray(norm[inside].mean(axis=0)).ravel()
    mean_out = np.asarray(norm[~inside].mean(axis=0)).ravel()
    detected = np.asarray(norm[inside].getnnz(axis=0)).ravel() / int(inside.sum())
    difference = mean_in - mean_out
    candidates = np.flatnonzero((difference >= 0.25) & (mean_in >= 0.10) & (detected >= 0.10))
    candidates = candidates[np.argsort(difference[candidates])[::-1]]
    candidates = [
        index for index in candidates
        if genes[index].lower() not in NON_INFORMATIVE_GENES
        and not genes[index].lower().startswith(NON_INFORMATIVE_PREFIXES)
    ][:max_genes]
    return pd.DataFrame({
        "archer_cluster": cluster,
        "posthoc_display_name": COMMUNITIES[cluster],
        "gene": [genes[index] for index in candidates],
        "mean_log_expression_in_cluster": mean_in[candidates],
        "mean_log_expression_outside_cluster": mean_out[candidates],
        "mean_expression_difference": difference[candidates],
        "fraction_expressing_in_cluster": detected[candidates],
        "selection_rule": "mean_difference>=0.25, mean>=0.10, detection>=0.10; ribosomal/mitochondrial/housekeeping genes excluded",
    })


def profile(query, background):
    payload = {
        "organism": "drerio",
        "query": list(query),
        "background": list(background),
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
    counts = mmread(EXPORT_DIR / "counts.mtx").tocsr().T.astype(np.float32)
    genes = pd.read_csv(EXPORT_DIR / "genes.csv")["gene"].astype(str).tolist()
    labels = np.load(EXPORT_DIR / "archer_predictions_hig500_res04_repaired.npy")
    if counts.shape != (len(labels), len(genes)):
        raise ValueError("Matrix, genes, and HIG500 prediction sizes disagree")
    norm = log_normalize(counts)
    background = [gene for gene, present in zip(genes, np.asarray(counts.getnnz(axis=0)).ravel() > 0) if present]
    marker_frames, term_frames, diagnostics = [], [], []
    for cluster, display_name in COMMUNITIES.items():
        markers = select_markers(norm, labels, genes, cluster)
        marker_frames.append(markers)
        response = profile(markers["gene"].tolist(), background)
        query_meta = response["meta"]["genes_metadata"]["query"]["query_1"]
        terms = pd.DataFrame(response.get("result", []))
        if not terms.empty:
            terms = terms.loc[terms["significant"].astype(bool)].copy()
            terms["archer_cluster"] = cluster
            terms["posthoc_display_name"] = display_name
            terms["minus_log10_p_value"] = -np.log10(terms["p_value"].astype(float))
            term_frames.append(terms)
        diagnostics.append({
            "archer_cluster": cluster,
            "posthoc_display_name": display_name,
            "n_cells": int((labels == cluster).sum()),
            "n_input_genes": len(markers),
            "n_mapped_input_genes": len(query_meta.get("mapping", {})),
            "mapping_fraction": len(query_meta.get("mapping", {})) / max(1, len(markers)),
            "n_significant_terms": 0 if terms.empty else len(terms),
            "background_gene_count": len(background),
            "gprofiler_database_version": response["meta"].get("version", "unknown"),
        })
    pd.concat(marker_frames, ignore_index=True).to_csv(SOURCE_DIR / "zebrafish_hig500_community_upregulated_genes.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(SOURCE_DIR / "zebrafish_hig500_enrichment_diagnostics.csv", index=False)
    pd.concat(term_frames, ignore_index=True).to_csv(SOURCE_DIR / "zebrafish_hig500_gprofiler_terms.csv", index=False)


if __name__ == "__main__":
    main()
