# Input data

No data are distributed with this repository. Download the public datasets and place them at the paths below, relative to the `reproduction/` folder. `INPUT_CHECKSUMS.csv` gives the SHA-256 of every input file used in the reported runs, so local copies can be checked against them.

## Benchmark datasets (16)

Expected file: `data/benchmarks/<dataset>/sc_cluster_fixed_result.h5ad`. Each file is an AnnData object of raw counts with the reference label in `obs`.

| Dataset | Source |
|---|---|
| Adam | GSE94333 |
| Bach | GSE106273 |
| Klein | GSE65525 |
| Muraro | GSE85241 |
| Plasschaert | GSE102580 |
| Pollen | SRP041736 |
| Quake_10x_Bladder, Quake_10x_Limb_Muscle, Quake_10x_Spleen | GSE109774 |
| Quake_Smart-seq2_Diaphragm, _Heart, _Limb_Muscle, _Lung, _Trachea | GSE109774 |
| Romanov | GSE74672 |
| Tosches_turtle | Tosches et al., Science 2018 (aar4237), author resource |

Cell, gene and class counts are listed in Supplementary Table S1. Matching the public records does not guarantee that a fresh download yields the identical local subset, so check each file against `INPUT_CHECKSUMS.csv`.

## Case-study datasets (3)

| Dataset | Source | Expected location |
|---|---|---|
| Chicken PBL, G_G_00018 | GEO GSE224329; SCAN G_G_00018 | `case_study_datasets/G_G_00018/` (`GSE224329_final_matrix.mtx.gz`, `_barcodes.tsv.gz`, `_feats.tsv.gz`) |
| Zebrafish, D_R_00002 | GEO GSE186423 (superseries GSE186427); SCAN D_R_00002 | `case_study_datasets/D_R_00002/exported/` (`counts.mtx`, `genes.csv`, `meta.csv`) |
| Pig tracheal glands, S_S_00005 | GEO GSE185849; SCAN S_S_00005 | `case_study_datasets/S_S_00005/case_inputs/label_isolated_allcells_20260730/` (written by `prepare_s_s_00005_case.R`) |

Notes:
- **Chicken.** The run script first looks for `GSE224329_final_matrix.mtx.gz.new` (a re-downloaded copy) and falls back to `GSE224329_final_matrix.mtx.gz`. Manual labels and UMAP come from the SCAN source Seurat object, via `extract_scan_g_g_00018_metadata_20260810.R`. The raw counts for the C14 QC (Supplementary Table S14) are exported from the same object by `chicken_export_counts_20261027.R`.
- **Zebrafish and pig.** The exported matrices are derived from the SCAN source Seurat objects.
