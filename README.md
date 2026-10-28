# ArCHeR

**ArCHeR** (Adaptive and Robust Clustering via Hybrid-gene Extraction for Rare-cells) is an unsupervised scRNA-seq clustering method. It combines highly variable genes (HVGs) with high-importance genes (HIGs) learned from provisional unsupervised partitions, and builds a cell-specific adaptive shared-nearest-neighbour graph.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23724937.svg)](https://doi.org/10.5281/zenodo.23724937)

Release v3.2.0 (the version used in the manuscript) is archived on Zenodo: https://doi.org/10.5281/zenodo.23724937

Licence: GNU General Public License v3.0 (GPL-3.0); see `NOTICE.md`. The adaptive-graph step follows aKNNO (GPL-3.0).

## Repository layout

```
archer/                  importable package (method implementation, v3.2.0)
reproduction/            archived scripts, byte-identical to the runs reported in the manuscript
results/frozen_benchmark frozen 16-dataset ArCHeR predictions and metrics
tests/                   check that the package reproduces the frozen predictions
environment/             pinned Python and R versions
pyproject.toml           installs the archer package
```

## Install

```bash
conda env create -f environment/environment.yml
conda activate archer
pip install -e .
```

## Usage

```python
import anndata as ad
import archer

adata = ad.read_h5ad("my_counts.h5ad")        # raw counts, cells x genes
labels = archer.run_archer(adata, random_seed=2026)
rare = archer.calibrate_rare_partition(labels)  # size-defined rare clusters (≤ 2% of cells)
```

By default, `run_archer` uses the manuscript configuration:
- exact nearest neighbours, with HNSW disabled;
- HIG budget 500;
- adaptive graph with kmax = 20 and SNN pruning threshold 1/15.

The three case studies used kmax = 12 and pruning 0.05 (Supplementary Table S2), for example `archer.run_archer(adata, params={"AKNN_KMAX": 12, "AKNN_PRUNE_SNN": 0.05})`. For faster neighbour search on large datasets, pass `params={"NEIGHBOR_BACKEND": "auto"}`. This mode was not used for any reported result.

## Relation between `archer/` and `reproduction/`

`archer/` contains the three frozen core modules: `method_archer.py`, `method_archer_configurable.py` and `method_archer_final.py`. The only change from the archived copies in `reproduction/experiments/scripts/` is that two import statements became relative so the code can be imported as a package. `method_archer.py` is byte-identical to its archived copy.

To confirm that the package reproduces the manuscript benchmark cell for cell, place the benchmark data as described in `reproduction/data/README.md` and run:

```bash
python tests/check_package_vs_frozen.py Pollen Klein Tosches_turtle
```

It compares the package output with `results/frozen_benchmark/archer_final/<dataset>.csv.gz`.

## Reproducing the manuscript analyses

See `reproduction/README.md`.

## Licence

GPL-3.0. The full licence text is in `LICENSE`; see `NOTICE.md` for the copyright notice and third-party credit.

## Citation

Until the article is published, please cite the software release:

> ArCHeR v3.2.0. Zenodo (2026). https://doi.org/10.5281/zenodo.23724937

The article reference will be added here after publication.
