#!/usr/bin/env python3
"""Finish the Hermes-vs-existing-baselines comparison without retraining.

The two-GPU launcher already merged the Hermes workers into
<hermes-root>/merged_10_seeds before the earlier run failed while resolving a
baseline path. This script reads that merged result and runs only the paired
baseline comparison stage.

Default baseline roots are the paths supplied in the intended launch command:
  connection_operator_extensions_v5_additional/merged_10_seeds
  pde_results_v7/extensions

It also repairs the common accidental `_v2` suffix mismatch when the sibling
path without that suffix exists. It does not alter the source result folders.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hermes-root", type=Path, default=Path("hermes_adapted_10seeds_v2"),
                   help="Output root created by run_hermes_pde_comparison_two_gpu.sh")
    p.add_argument("--merged-dir", type=Path, default=None,
                   help="Merged Hermes folder (default: HERMES_ROOT/merged_10_seeds)")
    p.add_argument("--runner", type=Path, default=here / "run_hermes_pde_comparison.py",
                   help="Existing Hermes runner, used for its validated paired-comparison code")
    p.add_argument("--baseline-results", nargs="+", type=Path, default=[
        Path("connection_operator_extensions_v5_additional/merged_10_seeds"),
        Path("pde_results_v7/extensions"),
    ], help="Existing matched baseline result roots")
    p.add_argument("--bootstrap-samples", type=int, default=None,
                   help="Override bootstrap count; by default use the saved Hermes manifest")
    p.add_argument("--bootstrap-seed", type=int, default=20261116)
    p.add_argument("--cpu-threads", type=int, default=4)
    return p.parse_args()


def resolve_existing(path: Path, label: str) -> Path:
    """Resolve a path and handle a mistaken merged_10_seeds_v2 suffix."""
    path = path.expanduser()
    candidates = [path]
    name = path.name
    if name.endswith("_v2"):
        candidates.append(path.with_name(name[:-3]))
    elif name == "merged_10_seeds":
        candidates.append(path.with_name(name + "_v2"))

    # Also consider paths relative to the directory containing this script.
    expanded = []
    for candidate in candidates:
        expanded.append(candidate)
        if not candidate.is_absolute():
            expanded.append(Path(__file__).resolve().parent / candidate)
    for candidate in expanded:
        if candidate.exists():
            resolved = candidate.resolve()
            print(f"[{label}] {path} -> {resolved}")
            return resolved

    attempted = "\n  ".join(str(x) for x in expanded)
    raise FileNotFoundError(f"Could not find {label} path. Tried:\n  {attempted}")


def load_runner(path: Path):
    path = resolve_existing(path, "Hermes runner")
    runner_dir = str(path.parent)
    if runner_dir not in sys.path:
        sys.path.insert(0, runner_dir)
    # Importing through the module name ensures the runner's companion
    # run_connection_operator_extensions_v5.py is found beside it.
    import importlib.util
    spec = importlib.util.spec_from_file_location("hermes_pde_runner_for_merge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Hermes runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    cli = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(cli.cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cli.cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cli.cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cli.cpu_threads))

    hermes_root = cli.hermes_root.expanduser()
    if not hermes_root.is_absolute():
        hermes_root = (Path.cwd() / hermes_root).resolve()
    merged_dir = cli.merged_dir or (hermes_root / "merged_10_seeds")
    if not merged_dir.is_absolute():
        merged_dir = (Path.cwd() / merged_dir).resolve()
    if not merged_dir.is_dir():
        raise FileNotFoundError(
            f"Merged Hermes directory not found: {merged_dir}\n"
            "The worker merge must have completed before running this script."
        )

    manifest_path = merged_dir / "hermes_run_manifest.json"
    if not manifest_path.exists():
        manifest_path = hermes_root / "hermes_run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Could not find hermes_run_manifest.json under {merged_dir} or {hermes_root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") not in ("MERGED_COMPLETE", "COMPLETED"):
        raise RuntimeError(f"Hermes manifest is not complete: status={manifest.get('status')!r}")

    metrics_path = merged_dir / "all_trajectory_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing merged Hermes metrics: {metrics_path}")
    hermes_metrics = pd.read_csv(metrics_path)
    if hermes_metrics.empty:
        raise ValueError(f"Merged Hermes metrics are empty: {metrics_path}")

    cases = list(manifest.get("cases", []))
    seeds = sorted(int(x) for x in manifest.get("seeds", []))
    if not cases or not seeds:
        raise ValueError("Hermes manifest must contain non-empty 'cases' and 'seeds'.")
    present_cases = set(hermes_metrics["case"].astype(str).unique())
    missing_cases = sorted(set(cases) - present_cases)
    if missing_cases:
        raise ValueError(f"Hermes metrics are missing cases: {missing_cases}")

    numeric_seeds = pd.to_numeric(hermes_metrics["seed"], errors="coerce")
    present_seeds = sorted(int(x) for x in numeric_seeds.dropna().unique())
    if present_seeds != seeds:
        raise ValueError(f"Hermes metrics seeds {present_seeds} do not match manifest seeds {seeds}")

    baseline_roots = [resolve_existing(x, f"baseline[{i}]")
                      for i, x in enumerate(cli.baseline_results, start=1)]

    runner = load_runner(cli.runner)
    # Reconstruct the namespace expected by the runner's paired comparison
    # routine from the saved manifest, so we don't silently fall back to a
    # different training/data protocol during the merge.
    defaults = {
        "train_trajectories": 16,
        "val_trajectories": 4,
        "test_trajectories": 6,
        "steps": 101,
        "train_steps": 26,
        "history": 3,
        "dt": 0.02,
        "reference_substeps": 4,
        "reaction_substeps": 16,
        "epochs": 45,
        "batch_size": 16,
        "lr": 8e-4,
        "weight_decay": 1e-6,
        "patience": 10,
        "model_width": 32,
        "grad_clip_norm": 1.0,
        "warmup_epochs": 0,
        "ema_decay": 0.0,
        "batch_sampling": "trajectory-balanced",
        "residual_bound_multiplier": 3.0,
        "bootstrap_samples": 3000,
        "bootstrap_horizons": [1, 5, 10, 25, 50, 75],
    }
    value_map = {
        "lr": "learning_rate",
    }
    values = {}
    for key, default in defaults.items():
        manifest_key = value_map.get(key, key)
        values[key] = manifest.get(manifest_key, default)
    if cli.bootstrap_samples is not None:
        values["bootstrap_samples"] = cli.bootstrap_samples

    args = SimpleNamespace(
        cases=cases,
        seeds=seeds,
        out=merged_dir,
        baseline_results=baseline_roots,
        bootstrap_seed=cli.bootstrap_seed,
        cpu_threads=cli.cpu_threads,
        **values,
    )

    print(f"[HERMES] merged metrics: {metrics_path}")
    print(f"[HERMES] seeds: {seeds}")
    print(f"[HERMES] cases: {len(cases)}")
    print("[BASELINES]")
    for root in baseline_roots:
        print(f"  {root}")

    # This is the same validated comparison function called at the end of the
    # original runner. It checks that prior and Hermes runs share reference
    # trajectories before writing paired metrics/bootstrap tables.
    runner._merge_prior_baselines(args, hermes_metrics)

    comparison_dir = merged_dir / "comparison"
    expected_outputs = [
        comparison_dir / "all_trajectory_metrics.csv",
        comparison_dir / "summary.csv",
        comparison_dir / "horizon_75_summary.csv",
        comparison_dir / "hermes_paired_bootstrap.csv",
        comparison_dir / "comparison_manifest.json",
    ]
    missing = [str(p) for p in expected_outputs if not p.exists()]
    if missing:
        raise RuntimeError("Comparison stage returned but expected outputs are missing:\n" + "\n".join(missing))
    print("\n[DONE] Paired comparison files written to:", comparison_dir)
    print("[DONE] Horizon-75 table:", comparison_dir / "horizon_75_summary.csv")
    print("[DONE] Paired bootstrap:", comparison_dir / "hermes_paired_bootstrap.csv")


if __name__ == "__main__":
    main()
