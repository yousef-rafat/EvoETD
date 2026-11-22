#!/usr/bin/env python3
"""Merge two seed-sharded EvoETD high-resolution transfer workers."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ["case", "model", "seed", "test_trajectory", "horizon"]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("worker0", type=Path)
    ap.add_argument("worker1", type=Path)
    args = ap.parse_args()
    w0, w1, out = args.worker0, args.worker1, args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Merge output already exists and is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    m0, m1 = json.loads((w0 / "run_manifest.json").read_text()), json.loads((w1 / "run_manifest.json").read_text())
    for key in ("runner_version", "training_grid", "target_resolutions", "models", "test_trajectories",
                "steps", "train_steps", "history", "dt", "reference_substeps", "chebyshev_order"):
        if m0.get(key) != m1.get(key):
            raise ValueError(f"Worker manifests disagree on {key}: {m0.get(key)!r} vs {m1.get(key)!r}")
    seeds = sorted(set(m0.get("seeds", [])) | set(m1.get("seeds", [])))
    overlap = set(m0.get("seeds", [])) & set(m1.get("seeds", []))
    if overlap:
        raise ValueError(f"Workers contain overlapping training seeds: {sorted(overlap)}")

    metrics = pd.concat([read_csv(w0 / "trajectory_metrics.csv"), read_csv(w1 / "trajectory_metrics.csv")], ignore_index=True)
    if not metrics.empty:
        if not set(KEY).issubset(metrics.columns):
            raise ValueError(f"Trajectory table is missing required keys {KEY}")
        dup = metrics.duplicated(KEY, keep=False)
        if dup.any():
            raise ValueError("Duplicate model/seed/trajectory/horizon rows found across workers.")
        metrics.to_csv(out / "trajectory_metrics.csv", index=False)
        seed_level = (metrics.groupby(["case", "pde", "test_height", "test_width", "model", "seed", "horizon"],
                                      as_index=False).rel_l2.mean()
                      .rename(columns={"rel_l2": "seed_mean_rel_l2"}))
        summary = (seed_level.groupby(["case", "pde", "test_height", "test_width", "model", "horizon"])
                   .seed_mean_rel_l2.agg(["mean", "std", "count"]).reset_index()
                   .rename(columns={"std": "seed_std", "count": "n_seeds"}))
        seed_level.to_csv(out / "seed_level_metrics.csv", index=False)
        summary.to_csv(out / "resolution_summary.csv", index=False)

    profiles = pd.concat([read_csv(w0 / "runtime_memory.csv"), read_csv(w1 / "runtime_memory.csv")], ignore_index=True)
    if not profiles.empty:
        pkey = ["case", "model", "seed"]
        if profiles.duplicated(pkey).any():
            raise ValueError("Duplicate runtime profile rows found across workers.")
        profiles.to_csv(out / "runtime_memory.csv", index=False)

    failures = pd.concat([read_csv(w0 / "failures.csv"), read_csv(w1 / "failures.csv")], ignore_index=True)
    if not failures.empty:
        failures.to_csv(out / "failures.csv", index=False)

    refs0, refs1 = w0 / "references", w1 / "references"
    if refs0.exists():
        shutil.copytree(refs0, out / "references", dirs_exist_ok=True)
        if refs1.exists():
            for p0 in refs0.glob("*.npz"):
                p1 = refs1 / p0.name
                if not p1.exists():
                    raise FileNotFoundError(f"Worker 1 is missing reference archive {p1}")
                with np.load(p0, allow_pickle=False) as a, np.load(p1, allow_pickle=False) as b:
                    if not np.array_equal(a["trajectories"], b["trajectories"]):
                        raise ValueError(f"Reference trajectories differ between workers: {p0.name}")

    ref0, ref1 = read_csv(w0 / "reference_refinement.csv"), read_csv(w1 / "reference_refinement.csv")
    if not ref0.empty or not ref1.empty:
        ref = pd.concat([ref0, ref1], ignore_index=True)
        if "case" in ref.columns:
            ref = ref.drop_duplicates("case", keep="first")
        ref.to_csv(out / "reference_refinement.csv", index=False)

    merged = dict(m0)
    merged.update({
        "seeds": seeds,
        "workers": [str(w0), str(w1)],
        "device": "two-worker merge (see each worker run_manifest.json)",
        "successful_metric_rows": int(len(metrics)),
        "failed_model_runs": int(len(failures)),
        "mode": "zero-shot transfer; no model retraining; disjoint seed shards merged",
    })
    (out / "run_manifest.json").write_text(json.dumps(merged, indent=2))
    print(f"Merged {len(metrics):,} trajectory rows, {len(profiles):,} profiles, {len(failures):,} failures.")
    print(f"Seeds: {seeds}")
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()
