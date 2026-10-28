"""Post-hoc GO:BP annotation for the fixed S_S_00005 C25 programme.

The four-comparator consensus list is created before this script runs.  GO
results are therefore annotation only and cannot feed back into clustering,
candidate selection, comparator selection or marker ranking.
"""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread


ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "case_study_datasets" / "S_S_00005"
RUN = CASE / "case_reruns" / "archer_hig500_adopted_exact_20260730"
COMPARATOR = RUN / "C25_large_comparator_locked_20260730" / "four_comparator_consensus_genewise.csv"
INPUT = CASE / "case_inputs" / "comparator_protocol_mtx_20260730"
OUT = RUN / "C25_gobp_posthoc_20260730"
API = "https://biit.cs.ut.ee/gprofiler/api"


def post(endpoint: str, payload: dict) -> dict:
    request = urllib.request.Request(
        API + endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    if OUT.exists():
        raise FileExistsError("Refusing to overwrite tagged output: %s" % OUT)
    consensus = pd.read_csv(COMPARATOR)
    query = consensus.loc[consensus["strict_consensus_all_four"], "gene"].astype(str).tolist()
    if not query:
        raise ValueError("The fixed four-comparator consensus list is empty")
    with gzip.open(INPUT / "counts.mtx.gz", "rb") as handle:
        counts = mmread(handle).tocsr()
    genes = pd.read_csv(INPUT / "genes.tsv", sep="\t", header=None).iloc[:, 0].astype(str).to_numpy()
    if len(genes) != counts.shape[0]:
        raise ValueError("Gene file and count matrix differ in gene count")
    background = pd.Series(genes[np.asarray((counts > 0).sum(axis=1)).ravel() > 0]).drop_duplicates().tolist()
    response = post("/gost/profile/", {
        "organism": "sscrofa",
        "query": query,
        "sources": ["GO:BP"],
        "user_threshold": 0.05,
        "significance_threshold_method": "g_SCS",
        "domain_scope": "custom",
        "background": background,
        "no_evidences": False,
    })
    rows = [{
        "term_id": item.get("native"),
        "term_name": item.get("name"),
        "source": item.get("source"),
        "adjusted_p_value": item.get("p_value"),
        "significant": item.get("significant"),
        "term_size": item.get("term_size"),
        "query_size": item.get("query_size"),
        "intersection_size": item.get("intersection_size"),
        "effective_domain_size": item.get("effective_domain_size"),
    } for item in response.get("result", [])]
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["adjusted_p_value", "term_id"]).reset_index(drop=True)
    OUT.mkdir(parents=True)
    pd.DataFrame({"strict_consensus_gene": query}).to_csv(
        OUT / "C25_four_comparator_strict_consensus_input.csv", index=False
    )
    pd.DataFrame({"observed_background_gene": background}).to_csv(
        OUT / "C25_observed_gene_background.csv", index=False
    )
    table.to_csv(OUT / "C25_gobp_gprofiler_results.csv", index=False)
    (OUT / "C25_gprofiler_raw_response.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta = response.get("meta", {})
    manifest = {
        "analysis_tag": "s_s_00005_C25_gobp_posthoc_20260730",
        "status": "COMPLETED",
        "scope": "Post-hoc GO:BP annotation only; GO did not select clusters, candidates, comparators, parameters or genes.",
        "species": "Sus scrofa (sscrofa)",
        "query_definition": "Genes strict in pooled-large and each of the three nearest-large comparisons.",
        "query_input_count": len(query),
        "custom_background_definition": "Unique observed genes in the supplied S_S_00005 count matrix.",
        "custom_background_input_count": len(background),
        "source": "GO:BP",
        "multiple_testing": "g:SCS, user threshold 0.05",
        "gprofiler_data_version": meta.get("version"),
        "significant_term_count": int(table["significant"].fillna(False).sum()) if not table.empty else 0,
        "interpretation_guardrail": "Enrichment describes a fixed programme and cannot establish a cell type, lineage or novel subtype.",
    }
    (OUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(table.loc[table["significant"].fillna(False)].head(15).to_string(index=False) if not table.empty else "No GO:BP terms returned.")


if __name__ == "__main__":
    main()
