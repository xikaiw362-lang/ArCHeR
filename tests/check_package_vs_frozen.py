"""Check that the archer package reproduces the frozen benchmark predictions.

usage: python tests/check_package_vs_frozen.py Pollen [Klein ...]

Input loading, label stripping and seeding are imported unchanged from the archived
runner in reproduction/, so only the clustering code differs between the two runs: the
importable package here versus the frozen scripts that produced
results/frozen_benchmark/archer_final/<dataset>.csv.gz. The benchmark .h5ad files must be
placed as described in reproduction/data/README.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reproduction" / "experiments" / "scripts"))

import archer  # noqa: E402
import run_archer_dual_view_16_benchmark as runner  # noqa: E402

failed = []
for dataset in sys.argv[1:] or ["Pollen"]:
    counts, true_labels, cell_ids, already_log, _ = runner.evaluation.load_full_dataset(dataset, source="h5ad")
    model_input = runner.evaluation.strip_observation_metadata(counts)
    seed = runner.stable_seed(2026, dataset)
    runner.set_seed(seed)
    labels = np.asarray(archer.run_archer(model_input, random_seed=seed, already_log=already_log), dtype=int)
    frozen = pd.read_csv(ROOT / "results" / "frozen_benchmark" / "archer_final" / f"{dataset}.csv.gz")
    same_cells = (frozen.cell_id.astype(str).to_numpy() == np.asarray(cell_ids, dtype=str)).all()
    same = same_cells and (frozen.pred_label.to_numpy() == labels).all()
    print(f"{dataset}: {len(labels)} cells, {len(np.unique(labels))} clusters, identical to frozen: {same}")
    if not same:
        failed.append(dataset)
sys.exit(1 if failed else 0)
