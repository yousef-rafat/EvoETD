#!/usr/bin/env python3
"""Validate and merge the two-GPU outputs produced by run_pde1_all10_two_gpu.sh.

Expected input layout (under --root):
  base_gpu0/       base_gpu1/
  extensions_gpu0/ extensions_gpu1/

The output is --root/merged by default. The original spherical-patch PDE1 base
and the constant-curvature extension suite remain separate; they are not pooled
as if they were the same PDE/data protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

EXPECTED_SEEDS = {17, 29, 41, 53, 65, 71, 83, 97, 109, 127}
HORIZONS = set(range(1, 76))
TEST_TRAJECTORIES = set(range(6))
REPORT_HORIZONS = [1, 5, 10, 25, 50, 75]
BASE_MODELS = {
    "FNO2D", "ConvGRU", "Scalar-Laplacian-ETD",
    "Connection-ETD-FNOResidual", "Connection-ETD-FibreEquivariant",
    "Scalar-Laplacian-FibreEquivariant",
}
BASE_CANDIDATE = "Connection-ETD-FibreEquivariant"


def load_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{path}: empty file")
    return df


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def seed_set(series: pd.Series) -> set[int]:
    out = set()
    for value in series.dropna().unique():
        s = str(value).strip().lower()
        if s in {"oracle", "nan", "none", ""}:
            continue
        out.add(int(float(s)))
    return out


def canonical_without_keys(obj: dict, excluded: set[str]) -> str:
    return json.dumps({k: v for k, v in obj.items() if k not in excluded}, sort_keys=True, default=str)


def normalize_oracle(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    q = df[(df.model == "Oracle-Connection-Strang") | (df.seed.astype(str).str.lower() == "oracle")].copy()
    if q.empty:
        raise ValueError(f"No Oracle-Connection-Strang rows in {root}")
    keys = ["test_trajectory", "horizon"]
    if q.duplicated(keys).any():
        # The oracle is reference-only: duplicate entries are accepted only when identical.
        for _, g in q.groupby(keys):
            if float(g.rel_l2.max()) - float(g.rel_l2.min()) > 1e-12:
                raise ValueError(f"Inconsistent duplicated oracle rows in {root}")
        q = q.drop_duplicates(keys)
    if set(q.test_trajectory.astype(int)) != TEST_TRAJECTORIES or set(q.horizon.astype(int)) != HORIZONS:
        raise ValueError(f"Oracle coverage incomplete in {root}")
    if not np.isfinite(q.rel_l2.to_numpy(float)).all():
        raise ValueError(f"Non-finite oracle metrics in {root}")
    return q.sort_values(keys).reset_index(drop=True)


def validate_base_worker(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = read_json(root / "pde_manifest.json")
    df = load_csv(root / "pde_rollout_metrics.csv", {"seed", "test_trajectory", "model", "horizon", "rel_l2"})
    oracle = normalize_oracle(df, root)
    learned = df[(df.model != "Oracle-Connection-Strang") & (df.seed.astype(str).str.lower() != "oracle")].copy()
    seeds = seed_set(learned.seed)
    if not seeds:
        raise ValueError(f"No learned PDE1 seeds in {root}")
    if not set(learned.model.unique()).issubset(BASE_MODELS):
        raise ValueError(f"Unknown base model(s) in {root}: {sorted(set(learned.model.unique())-BASE_MODELS)}")
    if set(learned.horizon.astype(int)) != HORIZONS or set(learned.test_trajectory.astype(int)) != TEST_TRAJECTORIES:
        raise ValueError(f"Base horizons/test-trajectory coverage incomplete in {root}")
    if not np.isfinite(learned.rel_l2.to_numpy(float)).all():
        raise ValueError(f"Non-finite base metrics in {root}")
    if learned.duplicated(["seed", "test_trajectory", "model", "horizon"]).any():
        raise ValueError(f"Duplicate base metrics in {root}")
    observed_models = set(learned.model.unique())
    for seed in seeds:
        for model in observed_models:
            g = learned[(learned.seed.astype(str) == str(seed)) & (learned.model == model)]
            if len(g) != 6 * 75:
                raise ValueError(f"Incomplete base result: worker={root.name} seed={seed} model={model} rows={len(g)}")
    return learned, oracle, manifest


def base_paired_bootstrap(learned: pd.DataFrame, out_path: Path, nboot: int = 3000) -> pd.DataFrame:
    candidate = BASE_CANDIDATE
    models = sorted(set(learned.model.unique()) - {candidate})
    rng = np.random.default_rng(20261106)
    records = []
    for horizon in REPORT_HORIZONS:
        at = learned[learned.horizon.astype(int) == horizon]
        pivot = at.pivot(index=["seed", "test_trajectory"], columns="model", values="rel_l2")
        seeds = sorted(seed_set(at.seed))
        tids = sorted(set(at.test_trajectory.astype(int)))
        for comparator in models:
            if candidate not in pivot or comparator not in pivot:
                continue
            mat = pivot[[candidate, comparator]].reindex(
                pd.MultiIndex.from_product([seeds, tids], names=["seed", "test_trajectory"]))
            if mat.isna().any().any():
                continue
            vals = mat.to_numpy(float).reshape(len(seeds), len(tids), 2)
            observed_c, observed_b = float(vals[:, :, 0].mean()), float(vals[:, :, 1].mean())
            boot_diff = np.empty(nboot); boot_red = np.empty(nboot)
            for b in range(nboot):
                si = rng.integers(0, len(seeds), size=len(seeds))
                ti = rng.integers(0, len(tids), size=len(tids))
                smp = vals[si[:, None], ti[None, :], :]
                cm, bm = float(smp[:, :, 0].mean()), float(smp[:, :, 1].mean())
                boot_diff[b] = cm - bm
                boot_red[b] = 1.0 - cm / max(bm, 1e-15)
            records.append({
                "horizon": horizon, "candidate": candidate, "comparator": comparator,
                "candidate_mean": observed_c, "comparator_mean": observed_b,
                "paired_difference": observed_c - observed_b,
                "relative_error_reduction": 1.0 - observed_c / max(observed_b, 1e-15),
                "difference_ci_low": float(np.quantile(boot_diff, .025)),
                "difference_ci_high": float(np.quantile(boot_diff, .975)),
                "reduction_ci_low": float(np.quantile(boot_red, .025)),
                "reduction_ci_high": float(np.quantile(boot_red, .975)),
                "n_seeds": len(seeds), "n_test_trajectories": len(tids), "bootstrap_samples": nboot,
            })
    result = pd.DataFrame(records)
    result.to_csv(out_path, index=False)
    return result


def merge_base(root: Path, out: Path) -> pd.DataFrame:
    inputs = [root / "base_gpu0", root / "base_gpu1"]
    data, oracles, manifests = [], [], []
    used_seeds: set[int] = set()
    config_keys = ["equation", "geometry", "nu", "mu", "eta", "dt", "steps", "train_steps",
                   "train_trajectories", "validation_trajectories", "test_trajectories", "scale", "reference_substeps"]
    base_config = None
    for worker in inputs:
        learned, oracle, manifest = validate_base_worker(worker)
        s = seed_set(learned.seed)
        overlap = used_seeds & s
        if overlap:
            raise ValueError(f"Base worker seed overlap {sorted(overlap)} in {worker}")
        used_seeds |= s
        cfg = {k: manifest.get(k) for k in config_keys}
        if base_config is None:
            base_config = cfg
        elif cfg != base_config:
            raise ValueError(f"Base worker configs differ: {worker} has {cfg}, expected {base_config}")
        data.append(learned); oracles.append(oracle); manifests.append(manifest)
    if used_seeds != EXPECTED_SEEDS:
        raise ValueError(f"Base merged seed set mismatch: observed={sorted(used_seeds)} expected={sorted(EXPECTED_SEEDS)}")
    oracle_ref = oracles[0]
    for other in oracles[1:]:
        a = oracle_ref[["test_trajectory", "horizon", "rel_l2"]].reset_index(drop=True)
        b = other[["test_trajectory", "horizon", "rel_l2"]].reset_index(drop=True)
        if not a[["test_trajectory", "horizon"]].equals(b[["test_trajectory", "horizon"]]) or not np.allclose(a.rel_l2, b.rel_l2, rtol=0, atol=1e-12):
            raise ValueError("Base GPU workers have different oracle/reference trajectories.")
    learned = pd.concat(data, ignore_index=True)
    if learned.duplicated(["seed", "test_trajectory", "model", "horizon"]).any():
        raise ValueError("Duplicate rows after base merge.")
    models = sorted(learned.model.unique())
    if set(models) != BASE_MODELS:
        raise ValueError(f"Expected six base models {sorted(BASE_MODELS)}; found {models}")
    for (seed, model), g in learned.groupby(["seed", "model"]):
        if len(g) != 450:
            raise ValueError(f"Base seed/model coverage is not 6x75: {seed}/{model}: {len(g)} rows")

    dest = out / "base"
    dest.mkdir(parents=True, exist_ok=True)
    all_metrics = pd.concat([learned, oracle_ref], ignore_index=True)
    all_metrics.to_csv(dest / "pde_rollout_metrics.csv", index=False)
    all_metrics.to_csv(dest / "pde_test_trajectory_metrics.csv", index=False)
    per_seed = (learned.groupby(["model", "seed", "horizon"]).rel_l2
                .agg(seed_mean="mean", seed_std="std", n_test_trajectories="count").reset_index())
    per_seed.to_csv(dest / "pde_by_seed.csv", index=False)
    across = (per_seed.groupby(["model", "horizon"]).seed_mean
              .agg(mean="mean", seed_std="std", n_seeds="count").reset_index())
    ntest = per_seed.groupby(["model", "horizon"]).n_test_trajectories.mean().reset_index(name="n_test_trajectories_per_seed")
    across = across.merge(ntest, on=["model", "horizon"], how="left")
    across["cv"] = across.seed_std / across["mean"].clip(lower=1e-15)
    oracle_summary = (oracle_ref.groupby(["model", "horizon"]).rel_l2
                      .agg(mean="mean", seed_std="std", n_test_trajectories="count").reset_index())
    oracle_summary["n_seeds"] = 0
    oracle_summary["n_test_trajectories_per_seed"] = oracle_summary.pop("n_test_trajectories")
    oracle_summary["cv"] = np.nan
    summary = pd.concat([across, oracle_summary], ignore_index=True).sort_values(["model", "horizon"])
    summary.to_csv(dest / "pde_rollout_summary.csv", index=False)
    all_metrics.groupby(["model", "horizon"]).rel_l2.agg(["mean", "std", "count"]).reset_index().to_csv(dest / "pde_rollout_summary_raw.csv", index=False)
    base_paired_bootstrap(learned, dest / "paired_bootstrap.csv")
    man_rows = []
    for worker, man in zip(inputs, manifests):
        man_rows.append({"worker": worker.name, "seeds": sorted(seed_set(pd.Series([x for x in learned[learned.seed.astype(int).isin(seed_set(learned.seed))].seed.unique()]))),
                         "manifest": man})
    (dest / "merged_manifest.json").write_text(json.dumps({
        "status": "MERGED_COMPLETE", "seeds": sorted(used_seeds), "models": models,
        "input_workers": [str(x.resolve()) for x in inputs], "base_config": base_config,
        "worker_manifests": manifests,
        "uncertainty": "seed_std is computed across ten per-seed means; each per-seed mean averages six shared test trajectories.",
    }, indent=2), encoding="utf-8")
    return summary


def npz_hashes(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        need = {"train_sha256", "validation_sha256", "test_sha256", "metadata"}
        missing = need - set(z.files)
        if missing:
            raise ValueError(f"{path}: missing cache-integrity fields {sorted(missing)}")
        return {k: str(z[k].item()) for k in need}


def import_extension_runner(path: Path):
    name = "_pde1_extensions_v6_for_merge"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import extension runner {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def merge_extensions(root: Path, out: Path, ext_runner: Path) -> pd.DataFrame:
    inputs = [root / "extensions_gpu0", root / "extensions_gpu1"]
    top_manifests = [read_json(x / "run_manifest.json") for x in inputs]
    cases = list(top_manifests[0].get("cases", []))
    if not cases or set(cases) != set(top_manifests[1].get("cases", [])):
        raise ValueError("PDE1 extension worker case lists differ or are empty.")
    excluded = {"seeds", "device", "elapsed_seconds", "status"}
    if canonical_without_keys(top_manifests[0], excluded) != canonical_without_keys(top_manifests[1], excluded):
        raise ValueError("PDE1 extension worker configurations differ beyond device/seed/time/status fields.")
    seed_sets = []
    frames, profiles, case_manifests = [], [], {}
    for worker in inputs:
        man = read_json(worker / "run_manifest.json")
        d = load_csv(worker / "all_trajectory_metrics.csv", {"case", "pde", "seed", "test_trajectory", "model", "horizon", "rel_l2"})
        if not np.isfinite(d.rel_l2.to_numpy(float)).all():
            raise ValueError(f"Non-finite PDE1 extension metric in {worker}")
        s = seed_set(d.seed)
        seed_sets.append(s)
        if d.duplicated(["case", "seed", "test_trajectory", "model", "horizon"]).any():
            raise ValueError(f"Duplicate extension metric keys inside {worker}")
        expected_models = man.get("models_by_case", {})
        if set(expected_models) != set(cases):
            raise ValueError(f"models_by_case mismatch in {worker}")
        for case in cases:
            case_dir = worker / case
            if not case_dir.is_dir():
                raise FileNotFoundError(case_dir)
            cm = read_json(case_dir / "case_manifest.json")
            case_manifests.setdefault(case, []).append(cm)
            case_df = d[d.case == case]
            if case_df.empty:
                raise ValueError(f"No metrics for {case} in {worker}")
            for seed in s:
                for model in expected_models[case]:
                    g = case_df[(case_df.seed.astype(int) == seed) & (case_df.model == model)]
                    if len(g) != 450 or set(g.test_trajectory.astype(int)) != TEST_TRAJECTORIES or set(g.horizon.astype(int)) != HORIZONS:
                        raise ValueError(f"Incomplete extension result {case}/{model}/seed={seed} in {worker}: {len(g)} rows")
        frames.append(d)
        rp = worker / "runtime_memory.csv"
        if rp.exists() and rp.stat().st_size > 1:
            profiles.append(pd.read_csv(rp))
    if seed_sets[0] & seed_sets[1]:
        raise ValueError(f"Extension worker seed overlap: {sorted(seed_sets[0]&seed_sets[1])}")
    seen = seed_sets[0] | seed_sets[1]
    if seen != EXPECTED_SEEDS:
        raise ValueError(f"Extension merged seed set mismatch: observed={sorted(seen)} expected={sorted(EXPECTED_SEEDS)}")

    # Confirm the same fixed reference trajectories were used by both GPUs for every case.
    for case in cases:
        a = npz_hashes(inputs[0] / case / "reference_trajectories.npz")
        b = npz_hashes(inputs[1] / case / "reference_trajectories.npz")
        if a != b:
            raise ValueError(f"Reference trajectory cache hashes/config differ for case {case} across GPUs.")
        cm0, cm1 = case_manifests[case]
        ignored_case_keys = {"seeds", "resume_signature"}
        if canonical_without_keys(cm0, ignored_case_keys) != canonical_without_keys(cm1, ignored_case_keys):
            raise ValueError(f"Case manifests differ for {case} across GPUs.")

    d = pd.concat(frames, ignore_index=True)
    if d.duplicated(["case", "seed", "test_trajectory", "model", "horizon"]).any():
        raise ValueError("Duplicate extension rows after merging GPUs.")
    seed_level = (d.groupby(["case", "pde", "model", "seed", "horizon"], as_index=False)
                  .rel_l2.mean().rename(columns={"rel_l2": "seed_mean_rel_l2"}))
    summary = (seed_level.groupby(["case", "pde", "model", "horizon"])
               .seed_mean_rel_l2.agg(mean="mean", seed_std="std", n_seeds="count").reset_index())
    summary["cv"] = summary.seed_std / summary["mean"].clip(lower=1e-15)
    dest = out / "extensions"
    dest.mkdir(parents=True, exist_ok=True)
    d.to_csv(dest / "all_trajectory_metrics.csv", index=False)
    seed_level.to_csv(dest / "all_seed_level_metrics.csv", index=False)
    summary.to_csv(dest / "all_summary.csv", index=False)
    profile = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()
    if not profile.empty:
        profile.to_csv(dest / "runtime_memory.csv", index=False)
    summary[summary.horizon.isin(REPORT_HORIZONS)].assign(suite="PDE1_constant_curvature_extensions").to_csv(
        dest / "summary_all_summary.csv", index=False)

    # Recompute the runner's two-way seed x test-trajectory paired bootstrap.
    mod = import_extension_runner(ext_runner)
    mod._BOOTSTRAP_HORIZONS = REPORT_HORIZONS
    mod.paired_bootstrap(d, dest / "all_paired_bootstrap.csv", 3000, 20261106)

    # Preserve per-case reference data, manifests, metrics and checkpoint/history files.
    case_rows = []
    for case in cases:
        case_out = dest / case
        case_out.mkdir(parents=True, exist_ok=True)
        source0, source1 = inputs[0] / case, inputs[1] / case
        shutil.copy2(source0 / "reference_trajectories.npz", case_out / "reference_trajectories.npz")
        for nm in ("reference_refinement.csv", "reference_refinement_train_by_trajectory.csv",
                   "reference_refinement_test_by_trajectory.csv", "reference_refinement_train_and_test_by_trajectory.csv"):
            if (source0 / nm).exists():
                shutil.copy2(source0 / nm, case_out / nm)
        part = d[d.case == case].copy()
        part.to_csv(case_out / "trajectory_metrics.csv", index=False)
        sl = seed_level[seed_level.case == case]
        sl.to_csv(case_out / "seed_level_metrics.csv", index=False)
        summary[summary.case == case].to_csv(case_out / "summary.csv", index=False)
        case_profile = profile[profile.case == case] if not profile.empty and "case" in profile else pd.DataFrame()
        case_profile.to_csv(case_out / "runtime_memory.csv", index=False)
        rms = []
        for worker in inputs:
            p = worker / case / "run_manifest.csv"
            if p.exists() and p.stat().st_size > 1:
                rms.append(pd.read_csv(p))
        if rms:
            pd.concat(rms, ignore_index=True).to_csv(case_out / "run_manifest.csv", index=False)
        cm = json.loads((source0 / "case_manifest.json").read_text(encoding="utf-8"))
        cm["seeds"] = sorted(seen)
        cm["merged_workers"] = [str(x.resolve()) for x in inputs]
        (case_out / "case_manifest.json").write_text(json.dumps(cm, indent=2), encoding="utf-8")
        ck_out = case_out / "checkpoints"
        ck_out.mkdir(exist_ok=True)
        for worker in inputs:
            for hist in (worker / case).glob("*_history.json"):
                shutil.copy2(hist, case_out / hist.name)
            src_ck = worker / case / "checkpoints"
            if src_ck.exists():
                for ck in src_ck.glob("*.pt"):
                    shutil.copy2(ck, ck_out / ck.name)
        case_rows.append({"case": case, "models": top_manifests[0]["models_by_case"][case], "n_seeds": len(seen)})

    (dest / "merged_manifest.json").write_text(json.dumps({
        "status": "MERGED_COMPLETE", "runner_version": top_manifests[0].get("runner_version"),
        "cases": cases, "seeds": sorted(seen), "models_by_case": top_manifests[0]["models_by_case"],
        "input_workers": [str(x.resolve()) for x in inputs], "case_integrity": case_rows,
        "uncertainty": "seed_std is across ten per-seed means; each per-seed mean averages six shared test trajectories.",
    }, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("pde1_all10"), help="Folder produced by run_pde1_all10_two_gpu.sh")
    ap.add_argument("--out", type=Path, default=None, help="Defaults to ROOT/merged")
    ap.add_argument("--extension-runner", type=Path, default=None,
                    help="Defaults to run_bundle_pde_validation_v6.py beside this script")
    ap.add_argument("--force", action="store_true", help="Replace only the existing merged output folder")
    a = ap.parse_args()
    root = a.root.resolve()
    out = (a.out.resolve() if a.out else root / "merged")
    if not root.exists():
        raise FileNotFoundError(root)
    if out.exists():
        if not a.force:
            raise FileExistsError(f"Output already exists: {out}; use --force to replace the merged folder only.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    ext_runner = a.extension_runner or Path(__file__).resolve().with_name("run_bundle_pde_validation_v6.py")
    base_summary = merge_base(root, out)
    ext_summary = merge_extensions(root, out, ext_runner.resolve())
    # Concise, report-horizon comparison. Keep the two protocols visibly separated.
    b = base_summary[base_summary.horizon.isin(REPORT_HORIZONS)].copy()
    b.insert(0, "case", "pde1_spherical_patch_base")
    b.insert(0, "suite", "PDE1_spherical_patch_base")
    e = ext_summary[ext_summary.horizon.isin(REPORT_HORIZONS)].copy()
    e.insert(0, "suite", "PDE1_constant_curvature_extensions")
    combined = pd.concat([b, e], ignore_index=True, sort=False)
    combined.to_csv(out / "summary_all_summary.csv", index=False)
    combined.to_csv(out / "all_summary.csv", index=False)
    manifest = {
        "status": "MERGED_COMPLETE", "expected_seeds": sorted(EXPECTED_SEEDS),
        "base_workers": [str((root / "base_gpu0").resolve()), str((root / "base_gpu1").resolve())],
        "extension_workers": [str((root / "extensions_gpu0").resolve()), str((root / "extensions_gpu1").resolve())],
        "output": str(out.resolve()),
        "important_note": "Original spherical-patch base and constant-curvature-chart extensions are kept separate; they use different geometry/data protocols.",
    }
    (out / "merged_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"PDE1 all-10 merge completed: {out}")
    print("\nBase summary at horizon 75:")
    print(base_summary[base_summary.horizon.eq(75)].to_string(index=False))
    print("\nExtension summary at horizon 75 (first 20 rows):")
    print(ext_summary[ext_summary.horizon.eq(75)].head(20).to_string(index=False))
    print("\nCompact table: ", out / "summary_all_summary.csv")


if __name__ == "__main__":
    main()
