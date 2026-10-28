# Reproduction scripts

This folder contains the scripts behind the manuscript analyses. They are byte-identical to the archived runs; `PROVENANCE.csv` lists each file's original path and SHA-256. The layout mirrors the original workspace. The scripts import each other from their own directory, and they locate data relative to this folder (`Path(__file__).parents[2]`), so keep the structure unchanged.

## Main entry points

| Analysis | Command (run from `experiments/scripts/`) |
|---|---|
| 16-dataset benchmark, frozen core (exact neighbours) | `python run_archer_core_20260724.py full ../metrics/<new_dir>` |
| Controlled rare-cell stress tests (HIG500) | `python run_archer_core_20260724.py rare ../metrics/<new_dir>` |
| Per-dataset settings (Fig. 2a–d, Supplementary Table S5) | `run_archer_v3_per_dataset_tuning_16.py` |
| Paired HIG/graph replacement (Supplementary Figs. S3, S4) | `rerun_hig_paired_20260915.py`, then `audit_hig_rerun_20260915.py` |
| Parameter sensitivity (Fig. 3a,b) | `run_sensitivity_v31_stable.py`, `run_sensitivity_rare_v31_stable.py` |
| Peak memory (Fig. 3d) | `measure_archer_core_exact_peak_rss_20260815.py` |
| Chicken / pig / zebrafish case studies | `run_chicken_archer_case_adopted_20260724.py`, `run_s_s_00005_archer_case_adopted_20260730.py`, `run_zebrafish_archer_case.py`, then `run_case_study_archer_adopted_20260724.py` |
| Chicken C14 QC (Supplementary Table S14) | `chicken_export_counts_20261027.R`, then `chicken_C14_qc_20261027.py` |
| Zebrafish CellChat summary (Fig. 6h,i) | `fish_native_communication_20261020.R` |

**Important.** Run the benchmark through `run_archer_core_20260724.py`. It wraps `run_archer_dual_view_16_benchmark.py` and enforces the exact-neighbour settings used for all reported results. Calling the inner runner directly uses the module default `NEIGHBOR_BACKEND="auto"`. That default gives the same result for datasets of 10,000 cells or fewer, but different partitions for larger ones such as Bach and Tosches turtle.

## Portability limits

- **Hard-coded paths.** Several scripts contain absolute Windows paths to the original workspace (`D:\sc_dataset`); they were left unchanged to keep the files byte-identical. Edit the `BASE_DIR`/`ROOT` constant at the top of each script, or place the data at that path. Affected files:
  - `analyze_hig_overlap.py`, `audit_hig_rerun_20260915.py`, `compare_archer_core_to_hig500_20260724.py`
  - `measure_archer_core_exact_peak_rss_20260815.py`, `rerun_hig_paired_20260915.py`, `run_all_efficiency_v31.py`
  - `run_archer_ablation_multimetric_fair.py`, `run_archer_dual_view_16_benchmark.py`, `run_archer_v2_evaluation.py`
  - `run_archer_v3_per_dataset_tuning_16.py`, `run_sensitivity_rare_v31_stable.py`, `run_sensitivity_v31_stable.py`
  - `chicken_C14_qc_20261027.py`, `chicken_export_counts_20261027.R`, `fish_native_communication_20261020.R`, `prepare_s_s_00005_case.R`
- **g:Profiler.** The enrichment scripts query the public g:Profiler API. The database versions used are listed in Supplementary Table S6; results from a later database release may differ.
- **Not included.** Comparator method wrappers (CHOIR, Seurat, SC3, GiniClust3, dropClust, scSHC, CIDR) and figure-assembly scripts are not part of this release.

## Data

See `data/README.md`. Input SHA-256 values are listed in `data/INPUT_CHECKSUMS.csv`.
