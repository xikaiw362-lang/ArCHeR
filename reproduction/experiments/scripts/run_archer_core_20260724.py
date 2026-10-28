"""Audited 2026-07-24 runner for the frozen ArCHeR core configuration.

This wrapper reuses the established benchmark runners but injects the explicit
exact-neighbor requirement for this rerun. It writes only to caller-supplied
new output directories and repairs the runner manifest with the effective
configuration actually used.
"""

import hashlib
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except ImportError:
        pass
    return value


def effective_params(overrides=None):
    import method_archer_final as final

    params = final.get_default_params(overrides or {})
    params.update(
        {
            "NEIGHBOR_BACKEND": "exact",
            "HNSW_ENABLED": False,
            "NEIGHBOR_N_JOBS": 1,
        }
    )
    return params


def patch_manifest(path, runner_name, output_tag, overrides=None):
    import method_archer_final as final

    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    params = effective_params(overrides)
    manifest.update(
        {
            "rerun_tag": output_tag,
            "audited_wrapper": os.path.abspath(__file__),
            "audited_wrapper_sha256": sha256_file(os.path.abspath(__file__)),
            "runner_mode": runner_name,
            "final_method_version": final.METHOD_VERSION,
            "effective_core_config": json_safe(params),
            "frozen_default_config": json_safe(params),
            "dual_view_hig_topn_per_view": int(params["RF_HIG_TOPN"]),
            "core_config_assertions": {
                "AKNN_KMAX": 20,
                "AKNN_PRUNE_SNN": 1.0 / 15.0,
                "RF_HIG_TOPN": 500,
                "AKNN_EMBEDDING_DIMS": 30,
                "HNSW_ENABLED": False,
                "NEIGHBOR_BACKEND": "exact",
                "label_free_selection": True,
            },
            "neighbor_backend_rule": "fixed exact sklearn brute; HNSW disabled",
            "delta_rule": (
                "author aKNNO point immediately after the last rapid community-count "
                "change, with connectivity-safe graph fallback"
            ),
            "hig_rule": "HVG union HIG; capped DPC+kNN view fusion at RF_HIG_TOPN",
            "snn_rule": "SNN-Jaccard after each cell's adaptive neighbor set",
            "graph_embedding_rule": "at most 30 PCs on HVG union HIG graph view",
        }
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(manifest), handle, indent=2, ensure_ascii=False)


def run_full(output_dir, datasets_from=None, seed=2026):
    import method_archer_final as final

    original = final.run_archer_final_pipeline

    def exact_final(*args, **kwargs):
        params = dict(kwargs.pop("params", {}) or {})
        params.update(effective_params(params))
        kwargs["params"] = params
        return original(*args, **kwargs)

    final.run_archer_final_pipeline = exact_final
    import run_archer_dual_view_16_benchmark as runner

    argv = [
        runner.__file__,
        "--output-dir",
        output_dir,
        "--seed",
        str(int(seed)),
    ]
    if datasets_from:
        argv.extend(["--datasets-from", datasets_from])
    old_argv = sys.argv
    try:
        sys.argv = argv
        runner.main()
    finally:
        sys.argv = old_argv
    patch_manifest(
        os.path.join(output_dir, "manifest.json"),
        "full16_dual_runner",
        os.path.basename(output_dir),
    )


def run_rare(output_dir, datasets, seed=2026, rare_min=2, rare_max=20, reps=3):
    import run_archer_ablation_multimetric_fair as ablation

    original = ablation.run_archer_pipeline_with_params

    def exact_configurable(*args, **kwargs):
        params = dict(kwargs.pop("params", {}) or {})
        params.update(effective_params(params))
        kwargs["params"] = params
        return original(*args, **kwargs)

    ablation.run_archer_pipeline_with_params = exact_configurable
    argv = [
        ablation.__file__,
        "--output-dir",
        output_dir,
        "--backend",
        "configurable",
        "--method",
        "ArCHeR-v3-full",
        "--reference-method",
        "ArCHeR-v3-full",
        "--reps",
        str(int(reps)),
        "--seed",
        str(int(seed)),
        "--rf-hig-topn",
        "500",
        "--overwrite",
    ]
    for dataset in datasets:
        argv.extend(["--dataset", dataset])
    for count in range(int(rare_min), int(rare_max) + 1, 2):
        argv.extend(["--rare-count", str(count)])
    old_argv = sys.argv
    try:
        sys.argv = argv
        ablation.main()
    finally:
        sys.argv = old_argv
    patch_manifest(
        os.path.join(output_dir, "manifest.json"),
        "rare2ds_2settings_10counts_3reps",
        os.path.basename(output_dir),
        {"RF_HIG_TOPN": 500},
    )


def main():
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: run_archer_core_20260724.py full|rare OUTPUT_DIR [args...]"
        )
    mode = sys.argv[1]
    output_dir = os.path.abspath(sys.argv[2])
    if os.path.exists(output_dir):
        raise RuntimeError("Refusing to reuse existing output directory: {}".format(output_dir))
    os.makedirs(output_dir, exist_ok=False)
    if mode == "full":
        run_full(output_dir)
    elif mode == "rare":
        run_rare(
            output_dir,
            ["Quake_10x_Limb_Muscle", "Quake_Smart-seq2_Heart"],
        )
    else:
        raise SystemExit("mode must be full or rare")


if __name__ == "__main__":
    main()
