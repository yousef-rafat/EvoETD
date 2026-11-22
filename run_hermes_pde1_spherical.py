#!/usr/bin/env python3
"""Hermes-Adapted baseline on the original spherical-patch PDE1 benchmark.

This runner deliberately reuses the original PDE1 spherical-patch reference
solver, trajectory generation, train/validation/test split, normalization,
training loop, and rollout metric from run_bundle_pde_validation_v5.py.

Required companion files in the same directory:
  run_bundle_pde_validation_v5.py
  run_ocean_connection_benchmark_v5.py
  run_hermes_pde_comparison.py
  run_connection_operator_extensions_v5.py

The Hermes message-passing graph is constructed from the *same spherical
finite-volume edges, weights, and Levi-Civita transports* used to assemble the
PDE1 connection Laplacian. A sparse-matrix audit checks that the graph edges
reconstruct that exact stiffness matrix before training.

For a two-T4 run use run_hermes_pde1_spherical_two_gpu.sh. The launcher prepares
reference trajectories once using four CPU threads, then trains disjoint seed
groups on the two GPUs with two CPU threads per worker.
"""
from __future__ import annotations

import os
# These are defaults for direct single-process use. The two-GPU launcher sets
# the per-worker values to two before starting either process.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import run_bundle_pde_validation_v5 as pde1
    import run_hermes_pde_comparison as hermes_runner
except Exception as exc:
    raise ImportError(
        "Could not import the PDE1/Hermes benchmark sources. Keep this file beside "
        "run_bundle_pde_validation_v5.py, run_ocean_connection_benchmark_v5.py, "
        "run_hermes_pde_comparison.py, and run_connection_operator_extensions_v5.py."
    ) from exc

core = pde1.core
MODEL_NAME = "Hermes-Adapted"
RUNNER_VERSION = "hermes_adapted_original_pde1_spherical_patch_v1_20261119"
DEFAULT_SEEDS = [17, 29, 41, 53, 65, 71, 83, 97, 109, 127]
REPORT_HORIZONS = [1, 5, 10, 25, 50, 75]


class HermesSphericalPredictor(nn.Module):
    """Adapt Hermes' residual MPNN to the original PDE1 predictor interface."""

    def __init__(self, history: int, width: int, layers: int):
        super().__init__()
        self.mpnn = hermes_runner.HermesAdaptedMPNN(
            history=history, width=width, layers=layers
        )

    def set_graph(self, graph: Dict[str, torch.Tensor]) -> None:
        self.mpnn.set_graph(graph)

    def set_residual_bound(self, value: float) -> None:
        self.mpnn.set_residual_bound(float(value))

    def forward(self, x: torch.Tensor, last: torch.Tensor,
                dt: torch.Tensor, geom: Dict[str, torch.Tensor], nu: float) -> torch.Tensor:
        # Match the Hermes comparison protocol: learned one-step residual,
        # added to the latest observed/predicted state.
        return last + self.mpnn(x)


def set_cpu_threads(n: int) -> None:
    n = max(1, int(n))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n)
    except Exception:
        pass


def array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(arr).cast("B")).hexdigest()


def spherical_edge_records(lat: np.ndarray, lon: np.ndarray) -> List[Tuple[int, int, np.ndarray, float]]:
    """Recreate the exact undirected edge table used by the spherical PDE1 K."""
    phi = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lam = np.deg2rad(np.asarray(lon, dtype=np.float64))
    H, W = len(phi), len(lam)
    if H < 2 or W < 2 or np.any(np.diff(phi) <= 0) or np.any(np.diff(lam) <= 0):
        raise ValueError("Spherical-patch coordinates must be ascending 1D grids.")
    phi_width = core._cell_widths(phi)
    lam_width = core._cell_widths(lam)
    records: List[Tuple[int, int, np.ndarray, float]] = []
    eye = np.eye(2, dtype=np.float64)

    # Same east-west conductance and Levi-Civita rotation as
    # core.build_bochner_eigenbasis(..., connection=True).
    for i in range(H):
        c = max(math.cos(float(phi[i])), 1e-6)
        for j in range(W - 1):
            dl = float(lam[j + 1] - lam[j])
            weight = float(phi_width[i] / (c * dl))
            theta = float(math.sin(float(phi[i])) * dl)
            P = np.array([[math.cos(theta), -math.sin(theta)],
                          [math.sin(theta),  math.cos(theta)]], dtype=np.float64)
            records.append((i * W + j, i * W + j + 1, P, weight))

    # Same north-south conductance and identity transport as the PDE1 core.
    for i in range(H - 1):
        dp = float(phi[i + 1] - phi[i])
        cm = max(math.cos(0.5 * float(phi[i] + phi[i + 1])), 1e-6)
        for j in range(W):
            weight = float(cm * lam_width[j] / dp)
            records.append((i * W + j, (i + 1) * W + j, eye.copy(), weight))
    return records


def attach_spherical_graph(geom: Dict[str, object], lat: np.ndarray,
                           lon: np.ndarray) -> float:
    """Attach Hermes graph metadata and audit exact stiffness reconstruction."""
    H, W = len(lat), len(lon)
    records = spherical_edge_records(lat, lon)
    geom["height"] = int(H)
    geom["width"] = int(W)
    geom["edge_records"] = records

    # Reassemble K using precisely the graph records passed to Hermes.
    K_graph = sp.lil_matrix((2 * H * W, 2 * H * W), dtype=np.float64)
    eye = np.eye(2, dtype=np.float64)
    for i, j, P, weight in records:
        ii = slice(2 * i, 2 * i + 2)
        jj = slice(2 * j, 2 * j + 2)
        K_graph[ii, ii] += weight * eye
        K_graph[jj, jj] += weight * (P.T @ P)
        K_graph[ii, jj] += -weight * P
        K_graph[jj, ii] += -weight * P.T
    K_graph = ((K_graph.tocsr() + K_graph.T.tocsr()) * 0.5).tocsr()
    K_reference = geom["laplacian"].tocsr()
    numerator = float(sp.linalg.norm(K_graph - K_reference))
    denominator = max(float(sp.linalg.norm(K_reference)), 1e-15)
    relative_error = numerator / denominator
    if not np.isfinite(relative_error) or relative_error > 2e-12:
        raise RuntimeError(
            "Hermes spherical message graph does not reconstruct the PDE1 "
            f"connection stiffness matrix: relative error={relative_error:.3e}."
        )
    return relative_error


def build_spherical_setup(args):
    H, W = int(args.height), int(args.width_grid)
    lat = np.linspace(20.0, 60.0, H, dtype=np.float32)
    lon = np.linspace(260.0, 320.0, W, dtype=np.float32)
    mask = np.ones((H, W), dtype=bool)
    gc = core.build_bochner_eigenbasis(lat, lon, mask, True)
    gs = core.build_bochner_eigenbasis(lat, lon, mask, False)
    graph_audit_error = attach_spherical_graph(gc, lat, lon)
    tile = {
        "mask": mask, "lat": lat, "lon": lon,
        "geom_conn": gc, "geom_scalar": gs,
        "spatial_weight": np.ones_like(mask, dtype=np.float32),
    }
    tile["u"] = np.zeros((args.steps + 1, H, W), dtype=np.float32)
    tile["v"] = np.zeros_like(tile["u"])
    tile["valid"] = np.ones((args.steps + 1, H, W), dtype=bool)
    tile["time_days"] = np.arange(args.steps + 1, dtype=np.float64) * args.dt
    return tile, graph_audit_error


def generate_reference_data(args, tile) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    H, W = tile["mask"].shape
    print(
        f"[PDE1-SPHERICAL][DATA] Generating train={args.train_trajectories}, "
        f"validation={args.val_trajectories}, test={args.test_trajectories}; "
        f"{args.steps} saved steps per trajectory",
        flush=True,
    )
    train = pde1.generate_trajectories(
        args.train_trajectories, args.steps, H, W, tile["lat"], tile["lon"],
        tile["mask"], tile["geom_conn"], args.dt, args.nu, args.mu, args.eta,
        seed=202601, substeps=args.reference_substeps, progress_label="train",
    )
    val = pde1.generate_trajectories(
        args.val_trajectories, args.steps, H, W, tile["lat"], tile["lon"],
        tile["mask"], tile["geom_conn"], args.dt, args.nu, args.mu, args.eta,
        seed=202602, substeps=args.reference_substeps, progress_label="validation",
    )
    test = pde1.generate_trajectories(
        args.test_trajectories, args.steps, H, W, tile["lat"], tile["lon"],
        tile["mask"], tile["geom_conn"], args.dt, args.nu, args.mu, args.eta,
        seed=202603, substeps=args.reference_substeps, progress_label="test",
    )
    # Deliberately retain the original PDE1 runner's unweighted training-only RMS.
    scale = float(np.sqrt(np.mean(train[:, :args.train_steps + 1] ** 2)))
    if not np.isfinite(scale) or scale <= 0:
        raise FloatingPointError(f"Invalid training normalization scale: {scale}")
    return train, val, test, scale


def data_hashes(train, val, test, scale, tile) -> Dict[str, str]:
    return {
        "train": array_sha256(train),
        "validation": array_sha256(val),
        "test": array_sha256(test),
        "lat": array_sha256(tile["lat"]),
        "lon": array_sha256(tile["lon"]),
        "mask": array_sha256(tile["mask"].astype(np.uint8)),
        "scale": array_sha256(np.asarray([scale], dtype=np.float64)),
    }


def save_reference_cache(path: Path, train, val, test, scale, tile, args,
                         graph_audit_error: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    npz_path = path / "reference_trajectories.npz"
    np.savez_compressed(
        npz_path, train=train, validation=val, test=test,
        scale=np.asarray(scale, dtype=np.float64),
        lat=tile["lat"], lon=tile["lon"], mask=tile["mask"].astype(np.uint8),
    )
    hashes = data_hashes(train, val, test, scale, tile)
    manifest = {
        "runner_version": RUNNER_VERSION,
        "geometry": "original positive-curvature spherical patch; unit sphere metric; full rectangular patch",
        "grid": [int(tile["mask"].shape[0]), int(tile["mask"].shape[1])],
        "lat_range_degrees": [float(tile["lat"].min()), float(tile["lat"].max())],
        "lon_range_degrees": [float(tile["lon"].min()), float(tile["lon"].max())],
        "equation": "u_t = -nu Delta_B u + (mu - eta |u|^2)u",
        "nu": float(args.nu), "mu": float(args.mu), "eta": float(args.eta),
        "dt": float(args.dt), "steps": int(args.steps),
        "train_steps": int(args.train_steps), "history": int(args.history),
        "train_trajectories": int(args.train_trajectories),
        "val_trajectories": int(args.val_trajectories),
        "test_trajectories": int(args.test_trajectories),
        "reference_substeps": int(args.reference_substeps),
        "trajectory_generation_seeds": {"train": 202601, "validation": 202602, "test": 202603},
        "normalization": "unweighted RMS on the training trajectories through the training prefix, matching run_bundle_pde_validation_v5.py",
        "reference_hashes": hashes,
        "hermes_graph_stiffness_relative_error": float(graph_audit_error),
        "status": "REFERENCE_DATA_READY",
    }
    (path / "reference_data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[PDE1-SPHERICAL][DATA] Cache written: {npz_path}", flush=True)


def load_reference_cache(path: Path, args, tile):
    if not path.exists():
        raise FileNotFoundError(f"Reference cache not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        train = data["train"]
        val = data["validation"]
        test = data["test"]
        scale = float(data["scale"])
        lat = data["lat"]
        lon = data["lon"]
        mask = data["mask"].astype(bool)
    expected = (tile["lat"], tile["lon"], tile["mask"])
    actual = (lat, lon, mask)
    if any(not np.array_equal(a, b) for a, b in zip(expected, actual)):
        raise ValueError("Cached spherical geometry does not match the requested grid.")
    if train.shape != (args.train_trajectories, args.steps + 1, 2, args.height, args.width_grid):
        raise ValueError(f"Cached train trajectory shape mismatch: {train.shape}")
    if val.shape != (args.val_trajectories, args.steps + 1, 2, args.height, args.width_grid):
        raise ValueError(f"Cached validation trajectory shape mismatch: {val.shape}")
    if test.shape != (args.test_trajectories, args.steps + 1, 2, args.height, args.width_grid):
        raise ValueError(f"Cached test trajectory shape mismatch: {test.shape}")
    # Validate the physics/training settings recorded by the cache, not only array shapes.
    manifest_path = path.parent / "reference_data_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    for key, expected_value in {
        "nu": float(args.nu), "mu": float(args.mu), "eta": float(args.eta),
        "dt": float(args.dt), "steps": int(args.steps), "train_steps": int(args.train_steps),
        "history": int(args.history), "reference_substeps": int(args.reference_substeps),
        "train_trajectories": int(args.train_trajectories),
        "val_trajectories": int(args.val_trajectories),
        "test_trajectories": int(args.test_trajectories),
    }.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"Reference cache setting mismatch for {key}: "
                f"{manifest.get(key)!r} != {expected_value!r}."
            )
    hashes = data_hashes(train, val, test, scale, tile)
    if hashes != manifest.get("reference_hashes"):
        raise ValueError("Reference cache hashes do not match its manifest.")
    return train, val, test, scale, hashes, manifest


def make_dataset(train, val, args, tile, scale):
    train_ds = pde1.PDEDataset(
        train, args.history, scale, args.train_steps, tile["lat"], tile["lon"], tile["mask"], args.dt
    )
    val_ds = pde1.PDEDataset(
        val, args.history, scale, args.train_steps, tile["lat"], tile["lon"], tile["mask"], args.dt
    )
    if not len(train_ds) or not len(val_ds):
        raise RuntimeError("Empty training or validation dataset.")
    return train_ds, val_ds


def oracle_metrics(test, args, tile) -> pd.DataFrame:
    rows = []
    mask = tile["mask"]
    for ti, trajectory in enumerate(test):
        print(f"[PDE1-SPHERICAL][ORACLE] trajectory {ti + 1}/{len(test)}", flush=True)
        state = trajectory[args.train_steps].copy()
        for horizon in range(1, args.steps - args.train_steps + 1):
            for _ in range(args.reference_substeps):
                state = pde1.strang_step(
                    state, args.dt / args.reference_substeps, tile["geom_conn"],
                    args.nu, args.mu, args.eta, mask,
                )
            truth = trajectory[args.train_steps + horizon]
            rel = float(
                np.linalg.norm((state - truth)[:, mask]) /
                max(np.linalg.norm(truth[:, mask]), 1e-12)
            )
            rows.append({
                "seed": "oracle", "test_trajectory": ti, "model": "Oracle-Connection-Strang",
                "horizon": horizon, "elapsed_time": horizon * args.dt, "rel_l2": rel,
            })
    return pd.DataFrame(rows)


def run_seed(seed: int, train, val, test, scale, tile, args,
             output: Path, device: torch.device, graph_t: Dict[str, torch.Tensor]) -> pd.DataFrame:
    started = time.monotonic()
    train_ds, val_ds = make_dataset(train, val, args, tile, scale)

    # Match the Hermes runner's deterministic model-init offset.
    core.set_seed(int(seed + 500))
    model = HermesSphericalPredictor(
        history=args.history, width=args.model_width, layers=args.hermes_layers
    )
    residual_bound = hermes_runner.base.estimate_residual_bound(
        train, args.train_steps, scale, args.dt, args.residual_bound_multiplier
    )
    model.set_residual_bound(residual_bound)
    model.set_graph(graph_t)

    print(
        f"\n[HERMES-PDE1-SPHERICAL][seed={seed}] start; "
        f"residual_bound={residual_bound:.6g}; params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    model, history, best = pde1.train_pde_model(
        model, "fibre", train_ds, val_ds, tile, device,
        args.epochs, args.batch, args.lr, args.nu, seed,
        patience=args.patience, progress_every=args.progress_every,
        progress_label=f"seed={seed} model={MODEL_NAME}",
    )
    history_path = output / "seed_histories" / f"hermes_seed{seed}_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))
    ckpt_path = output / "checkpoints" / f"hermes_seed{seed}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "seed": int(seed),
        "model_init_seed": int(seed + 500), "best_val_loss": float(best),
        "residual_bound": float(residual_bound),
        "runner_version": RUNNER_VERSION,
    }, ckpt_path)

    rows = []
    mask = tile["mask"]
    for ti, trajectory in enumerate(test):
        print(
            f"[HERMES-PDE1-SPHERICAL][seed={seed}] rollout "
            f"trajectory {ti + 1}/{len(test)}",
            flush=True,
        )
        hist_states = [trajectory[j].copy() for j in range(args.train_steps - args.history + 1,
                                                           args.train_steps + 1)]
        hist_valid = [mask.copy() for _ in hist_states]
        hist_dt = [1.0] * args.history
        for horizon in range(1, args.steps - args.train_steps + 1):
            hh = np.stack(hist_states) / scale
            hv = np.stack(hist_valid)
            feat = core.make_features(hh, hv, hist_dt, 1.0, mask, tile["lat"], tile["lon"])
            x = torch.from_numpy(feat[None]).to(device)
            last = torch.from_numpy(hh[-1][None]).to(device)
            dt_tensor = torch.tensor([[[[args.dt]]]], device=device, dtype=last.dtype)
            with torch.no_grad():
                pred = model(
                    x, last, dt_tensor,
                    {k: torch.as_tensor(tile["geom_conn"][k], device=device)
                     for k in ("V", "eig", "mass_sqrt", "wet_idx")},
                    args.nu,
                )[0].cpu().numpy() * scale
            pred[:, ~mask] = 0.0
            truth = trajectory[args.train_steps + horizon]
            rel = float(
                np.linalg.norm((pred - truth)[:, mask]) /
                max(np.linalg.norm(truth[:, mask]), 1e-12)
            )
            rows.append({
                "seed": int(seed), "test_trajectory": int(ti), "model": MODEL_NAME,
                "horizon": int(horizon), "elapsed_time": float(horizon * args.dt),
                "rel_l2": rel,
            })
            hist_states = hist_states[1:] + [pred]
            hist_valid = hist_valid[1:] + [mask.copy()]
            hist_dt = hist_dt[1:] + [1.0]

    result = pd.DataFrame(rows)
    expected = args.test_trajectories * (args.steps - args.train_steps)
    if len(result) != expected or not np.isfinite(result["rel_l2"].to_numpy()).all():
        raise RuntimeError(
            f"Seed {seed} produced invalid metric coverage: rows={len(result)}, expected={expected}."
        )
    per_seed_path = output / "seed_metrics" / f"seed_{seed}.csv"
    per_seed_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(per_seed_path, index=False)
    print(
        f"[HERMES-PDE1-SPHERICAL][seed={seed}] done; best_val={best:.6g}; "
        f"elapsed={(time.monotonic() - started) / 60:.1f} min",
        flush=True,
    )
    return result


def validate_seed_metrics(path: Path, seed: int, args) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = args.test_trajectories * (args.steps - args.train_steps)
    if len(df) != expected or set(df["model"].astype(str)) != {MODEL_NAME}:
        raise ValueError(f"Existing seed metrics are incomplete or malformed: {path}")
    if set(df["test_trajectory"].astype(int)) != set(range(args.test_trajectories)):
        raise ValueError(f"Missing test trajectories in {path}")
    if set(df["horizon"].astype(int)) != set(range(1, args.steps - args.train_steps + 1)):
        raise ValueError(f"Missing horizons in {path}")
    if set(df["seed"].astype(int)) != {int(seed)} or not np.isfinite(df["rel_l2"]).all():
        raise ValueError(f"Invalid seed or non-finite errors in {path}")
    return df


def summarize_metrics(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    learned = metrics[metrics["model"] == MODEL_NAME].copy()
    per_seed = (
        learned.groupby(["model", "seed", "horizon"], as_index=False)["rel_l2"]
        .mean().rename(columns={"rel_l2": "seed_mean"})
    )
    across = (
        per_seed.groupby(["model", "horizon"])["seed_mean"]
        .agg(mean="mean", std="std", n_seeds="count").reset_index()
    )
    ntest = (
        learned.groupby(["model", "seed", "horizon"])["test_trajectory"]
        .nunique().groupby(["model", "horizon"]).mean()
        .reset_index(name="n_test_trajectories_per_seed")
    )
    summary = across.merge(ntest, on=["model", "horizon"], how="left")
    oracle = (
        metrics[metrics["model"] == "Oracle-Connection-Strang"]
        .groupby(["model", "horizon"])["rel_l2"]
        .agg(mean="mean", std="std", n_test_trajectories="count").reset_index()
    )
    if not oracle.empty:
        oracle["n_seeds"] = 0
        oracle["n_test_trajectories_per_seed"] = oracle.pop("n_test_trajectories")
        summary = pd.concat([summary, oracle[summary.columns]], ignore_index=True)
    summary = summary.sort_values(["model", "horizon"]).reset_index(drop=True)
    return per_seed, summary


def worker_configuration(args, hashes: Dict[str, str]) -> Dict[str, object]:
    return {
        "runner_version": RUNNER_VERSION,
        "model": MODEL_NAME,
        "seeds": [int(s) for s in args.seeds],
        "reference_hashes": hashes,
        "pde1_protocol": "original positive-curvature spherical patch; unit-sphere finite-volume Bochner Laplacian",
        "height": int(args.height), "width_grid": int(args.width_grid),
        "steps": int(args.steps), "train_steps": int(args.train_steps),
        "history": int(args.history), "dt": float(args.dt),
        "nu": float(args.nu), "mu": float(args.mu), "eta": float(args.eta),
        "reference_substeps": int(args.reference_substeps),
        "train_trajectories": int(args.train_trajectories),
        "val_trajectories": int(args.val_trajectories),
        "test_trajectories": int(args.test_trajectories),
        "epochs": int(args.epochs), "batch": int(args.batch),
        "learning_rate": float(args.lr), "patience": int(args.patience),
        "model_width": int(args.model_width), "hermes_layers": int(args.hermes_layers),
        "residual_bound_multiplier": float(args.residual_bound_multiplier),
    }


def write_worker_outputs(output: Path, args, metrics: pd.DataFrame,
                         hashes: Dict[str, str], graph_error: float,
                         device: torch.device) -> None:
    metrics.to_csv(output / "pde_rollout_metrics.csv", index=False)
    metrics.to_csv(output / "all_trajectory_metrics.csv", index=False)
    seed_frame, summary = summarize_metrics(metrics)
    seed_frame.to_csv(output / "all_seed_level_metrics.csv", index=False)
    summary.to_csv(output / "pde_rollout_summary.csv", index=False)
    summary.to_csv(output / "all_summary.csv", index=False)
    summary[summary.horizon.isin(REPORT_HORIZONS)].to_csv(
        output / "summary_all_summary.csv", index=False
    )
    manifest = {
        "runner_version": RUNNER_VERSION,
        "model": MODEL_NAME,
        "architecture": "parallel-transport edge message passing + invariant scalar node updates",
        "official_hermes_reproduction": False,
        "pde1_protocol": "original positive-curvature spherical patch; unit-sphere finite-volume Bochner Laplacian",
        "geometry_graph_audit_relative_error": float(graph_error),
        "reference_hashes": hashes,
        "seeds": [int(s) for s in args.seeds],
        "device": str(device), "cpu_threads": int(args.cpu_threads),
        "hermes_layers": int(args.hermes_layers), "model_width": int(args.model_width),
        "train_trajectories": int(args.train_trajectories),
        "val_trajectories": int(args.val_trajectories),
        "test_trajectories": int(args.test_trajectories),
        "height": int(args.height), "width_grid": int(args.width_grid),
        "steps": int(args.steps), "train_steps": int(args.train_steps),
        "history": int(args.history), "dt": float(args.dt),
        "nu": float(args.nu), "mu": float(args.mu), "eta": float(args.eta),
        "reference_substeps": int(args.reference_substeps),
        "epochs": int(args.epochs), "batch": int(args.batch),
        "learning_rate": float(args.lr), "patience": int(args.patience),
        "residual_bound_multiplier": float(args.residual_bound_multiplier),
        "residual_bound_rule": "3x 99.5th percentile of normalized observed training increments; training data only",
        "normalization": "same unweighted training-prefix RMS as original PDE1 runner",
        "uncertainty": f"mean and sample standard deviation across {len(args.seeds)} training-seed means; six held-out trajectories reused across seeds",
        "status": "COMPLETED",
    }
    (output / "hermes_spherical_run_manifest.json").write_text(json.dumps(manifest, indent=2))


def merge_workers(worker_dirs: Sequence[Path], output: Path,
                  expected_seeds: Sequence[int]) -> None:
    if len(worker_dirs) < 2:
        raise ValueError("Pass both GPU worker directories to --merge-workers.")
    manifests = []
    frames = []
    oracle_frames = []
    seen: List[int] = []
    for root in worker_dirs:
        manifest_path = root / "hermes_spherical_run_manifest.json"
        metrics_path = root / "pde_rollout_metrics.csv"
        if not manifest_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(f"Worker is incomplete: {root}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "COMPLETED":
            raise ValueError(f"Worker manifest is not complete: {root}")
        manifests.append(manifest)
        df = pd.read_csv(metrics_path)
        model_rows = df[df.model.astype(str) == MODEL_NAME].copy()
        oracle_rows = df[df.model.astype(str) == "Oracle-Connection-Strang"].copy()
        seeds = sorted(model_rows.seed.astype(int).unique().tolist())
        if set(seeds) != set(int(s) for s in manifest.get("seeds", [])):
            raise ValueError(f"Worker seed manifest and metrics disagree: {root}")
        for seed in seeds:
            if seed in seen:
                raise ValueError(f"Overlapping training seed {seed} across GPU workers.")
            seen.append(seed)
            validate_seed_metrics(
                root / "seed_metrics" / f"seed_{seed}.csv", seed,
                _args_from_manifest(manifest),
            )
        frames.append(model_rows)
        oracle_frames.append(oracle_rows)

    if sorted(seen) != sorted(int(s) for s in expected_seeds):
        raise ValueError(
            f"Seed coverage mismatch: expected {sorted(expected_seeds)}, got {sorted(seen)}."
        )
    reference_hashes = manifests[0].get("reference_hashes")
    if not reference_hashes or any(m.get("reference_hashes") != reference_hashes for m in manifests[1:]):
        raise ValueError("GPU workers did not use byte-identical train/validation/test references.")
    comparable_keys = (
        "pde1_protocol", "height", "width_grid", "steps", "train_steps", "history", "dt",
        "nu", "mu", "eta", "reference_substeps", "epochs", "batch", "learning_rate",
        "patience", "model_width", "hermes_layers", "train_trajectories", "val_trajectories",
        "test_trajectories", "residual_bound_multiplier",
    )
    for key in comparable_keys:
        values = [m.get(key) for m in manifests]
        if any(v != values[0] for v in values[1:]):
            raise ValueError(f"Worker configuration mismatch for {key}: {values}")

    learned = pd.concat(frames, ignore_index=True)
    all_oracle = pd.concat(oracle_frames, ignore_index=True)
    if not all_oracle.empty:
        canonical = oracle_frames[0].sort_values(["test_trajectory", "horizon"]).reset_index(drop=True)
        for idx, candidate in enumerate(oracle_frames[1:], start=1):
            candidate = candidate.sort_values(["test_trajectory", "horizon"]).reset_index(drop=True)
            cols = ["test_trajectory", "horizon", "elapsed_time", "rel_l2"]
            if not canonical[cols].equals(candidate[cols]):
                raise ValueError(f"Oracle reference trajectories differ between worker {idx} and worker 0.")
        oracle = canonical.copy()
    else:
        oracle = pd.DataFrame(columns=learned.columns)

    # Validate full per-seed rollout coverage (6 held-out trajectories x 75 steps).
    expected_horizons = set(range(1, int(manifests[0]["steps"]) - int(manifests[0]["train_steps"]) + 1))
    expected_trajectories = set(range(int(manifests[0]["test_trajectories"])))
    for seed, group in learned.groupby("seed"):
        if set(group.horizon.astype(int)) != expected_horizons or set(group.test_trajectory.astype(int)) != expected_trajectories:
            raise ValueError(f"Incomplete rollout coverage for seed {seed}.")
        if len(group) != len(expected_horizons) * len(expected_trajectories):
            raise ValueError(f"Duplicate metric rows detected for seed {seed}.")

    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat([learned, oracle], ignore_index=True)
    metrics.to_csv(output / "pde_rollout_metrics.csv", index=False)
    metrics.to_csv(output / "all_trajectory_metrics.csv", index=False)
    seed_frame, summary = summarize_metrics(metrics)
    seed_frame.to_csv(output / "all_seed_level_metrics.csv", index=False)
    summary.to_csv(output / "pde_rollout_summary.csv", index=False)
    summary.to_csv(output / "all_summary.csv", index=False)
    summary[summary.horizon.isin(REPORT_HORIZONS)].to_csv(output / "summary_all_summary.csv", index=False)
    manifest = dict(
        runner_version=RUNNER_VERSION, model=MODEL_NAME,
        pde1_protocol=manifests[0]["pde1_protocol"],
        seeds=sorted(seen), n_seeds=len(seen), worker_directories=[str(p.resolve()) for p in worker_dirs],
        reference_hashes=reference_hashes,
        geometry_graph_audit_relative_error=max(float(m["geometry_graph_audit_relative_error"]) for m in manifests),
        status="MERGED_COMPLETE",
        uncertainty="mean and sample standard deviation across ten training-seed means; six held-out trajectories reused across seeds",
    )
    (output / "hermes_spherical_merged_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[MERGE] Complete ten-seed spherical PDE1 Hermes result: {output.resolve()}", flush=True)
    print(summary[summary.horizon.isin(REPORT_HORIZONS)].to_string(index=False), flush=True)


def _args_from_manifest(manifest: dict):
    """Minimal namespace used to validate per-seed CSV coverage during merge."""
    class A:
        pass
    a = A()
    a.test_trajectories = int(manifest["test_trajectories"])
    a.steps = int(manifest["steps"])
    a.train_steps = int(manifest["train_steps"])
    return a


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run Hermes-Adapted on the original spherical-patch PDE1 benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--out", type=Path, default=Path("hermes_pde1_spherical"))
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda:0, cuda:1")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--hermes-layers", type=int, default=3)
    ap.add_argument("--model-width", type=int, default=32)
    ap.add_argument("--train-trajectories", type=int, default=16)
    ap.add_argument("--val-trajectories", type=int, default=4)
    ap.add_argument("--test-trajectories", type=int, default=6)
    ap.add_argument("--height", type=int, default=16)
    ap.add_argument("--width-grid", type=int, default=20)
    ap.add_argument("--steps", type=int, default=101)
    ap.add_argument("--train-steps", type=int, default=26)
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--nu", type=float, default=0.02)
    ap.add_argument("--mu", type=float, default=0.7)
    ap.add_argument("--eta", type=float, default=3.0)
    ap.add_argument("--reference-substeps", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--progress-every", type=int, default=1)
    ap.add_argument("--residual-bound-multiplier", type=float, default=3.0)
    ap.add_argument("--data-cache", type=Path, default=None,
                    help="Path to reference_trajectories.npz generated by --prepare-data-only.")
    ap.add_argument("--prepare-data-only", action="store_true",
                    help="Generate the deterministic spherical PDE1 references and exit.")
    ap.add_argument("--merge-workers", nargs="+", type=Path, default=None,
                    help="Merge completed GPU worker directories and validate exact seed coverage.")
    ap.add_argument("--expected-seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                    help="Expected disjoint seed union when --merge-workers is used.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip per-seed metric files that already have complete rollout coverage.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Tiny CPU smoke test; not scientific results.")
    return ap.parse_args()


def validate_args(args) -> None:
    if args.cpu_threads < 1 or args.hermes_layers < 1 or args.model_width < 1:
        raise ValueError("cpu_threads, hermes_layers, and model_width must be positive.")
    if args.train_steps <= args.history or args.steps <= args.train_steps + 1:
        raise ValueError("Require steps > train_steps > history.")
    if args.height < 5 or args.width_grid < 5:
        raise ValueError("Spherical grid dimensions must be at least 5x5.")
    if args.dt <= 0 or args.nu < 0 or args.mu <= 0 or args.eta <= 0:
        raise ValueError("Require dt>0, nu>=0, mu>0, eta>0.")
    if args.reference_substeps < 1 or args.epochs < 1 or args.batch < 1:
        raise ValueError("reference_substeps, epochs, and batch must be positive.")
    if args.patience < 1 or args.progress_every < 0 or args.residual_bound_multiplier <= 0:
        raise ValueError("Invalid patience, progress interval, or residual-bound multiplier.")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Training seeds must be non-empty and unique.")


def main() -> None:
    args = parse_args()
    set_cpu_threads(args.cpu_threads)

    if args.self_test:
        # First verify the local O(2) covariance of the existing Hermes model.
        hermes_test = hermes_runner.self_test()
        # Then verify that our actual spherical graph reconstructs the exact
        # spherical-patch stiffness matrix used by the reference PDE solver.
        test_args = argparse.Namespace(height=8, width_grid=9)
        lat = np.linspace(20.0, 60.0, test_args.height, dtype=np.float32)
        lon = np.linspace(260.0, 320.0, test_args.width_grid, dtype=np.float32)
        mask = np.ones((test_args.height, test_args.width_grid), dtype=bool)
        geom = core.build_bochner_eigenbasis(lat, lon, mask, True)
        graph_error = attach_spherical_graph(geom, lat, lon)
        print(json.dumps({
            "status": "PASS", "hermes_local_O2_covariance": hermes_test,
            "spherical_graph_stiffness_relative_error": graph_error,
        }, indent=2), flush=True)
        return

    if args.smoke:
        args.seeds = [17]
        args.train_trajectories = 3
        args.val_trajectories = 2
        args.test_trajectories = 2
        args.height = 8
        args.width_grid = 10
        args.steps = 14
        args.train_steps = 6
        args.history = 3
        args.reference_substeps = 2
        args.epochs = 2
        args.batch = 3
        args.model_width = 12
        args.hermes_layers = 2
        args.progress_every = 1
        args.out = Path(args.out)
        args.data_cache = None
        args.prepare_data_only = False
        args.merge_workers = None

    validate_args(args)

    if args.merge_workers:
        merge_workers(args.merge_workers, args.out, args.expected_seeds)
        return

    tile, graph_error = build_spherical_setup(args)

    if args.prepare_data_only:
        if (args.out / "reference_trajectories.npz").exists():
            raise FileExistsError(
                f"Reference cache already exists: {args.out / 'reference_trajectories.npz'}"
            )
        args.out.mkdir(parents=True, exist_ok=True)
        train, val, test, scale = generate_reference_data(args, tile)
        save_reference_cache(args.out, train, val, test, scale, tile, args, graph_error)
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable.")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible.")
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args.out.mkdir(parents=True, exist_ok=True)
    seed_dir = args.out / "seed_metrics"
    if not args.resume and seed_dir.exists() and any(seed_dir.glob("seed_*.csv")):
        raise FileExistsError(f"Existing per-seed metrics found under {seed_dir}; use a fresh output or --resume.")

    if args.data_cache:
        train, val, test, scale, hashes, cache_manifest = load_reference_cache(
            args.data_cache, args, tile
        )
        graph_error = max(graph_error, float(cache_manifest.get("hermes_graph_stiffness_relative_error", 0.0)))
    else:
        train, val, test, scale = generate_reference_data(args, tile)
        hashes = data_hashes(train, val, test, scale, tile)

    train_ds, _ = make_dataset(train, val, args, tile, scale)
    residual_bound_preview = hermes_runner.base.estimate_residual_bound(
        train, args.train_steps, scale, args.dt, args.residual_bound_multiplier
    )
    print(
        f"[HERMES-PDE1-SPHERICAL] device={device}; cpu_threads={args.cpu_threads}; "
        f"seeds={args.seeds}; grid={args.height}x{args.width_grid}; "
        f"train_samples={len(train_ds)}; residual_bound={residual_bound_preview:.6g}; "
        f"graph_stiffness_relerr={graph_error:.3e}",
        flush=True,
    )

    graph_t = hermes_runner._make_graph_tensors(tile["geom_conn"], device)
    config = worker_configuration(args, hashes)
    config_path = args.out / "worker_config.json"
    if args.resume:
        if config_path.exists():
            existing_config = json.loads(config_path.read_text())
            if existing_config != config:
                raise ValueError(
                    "Cannot resume this worker: its existing worker_config.json "
                    "does not match the current seeds, reference trajectories, "
                    "geometry, or training configuration. Use a fresh output directory."
                )
        elif seed_dir.exists() and any(seed_dir.glob("seed_*.csv")):
            raise ValueError(
                "Found per-seed metrics but no worker_config.json to verify them; "
                "refusing to resume unverifiable results. Use a fresh output directory."
            )
        else:
            config_path.write_text(json.dumps(config, indent=2))
    else:
        if config_path.exists() or (seed_dir.exists() and any(seed_dir.glob("seed_*.csv"))):
            raise FileExistsError(
                f"Existing worker results found under {args.out}; use --resume or a fresh directory."
            )
        config_path.write_text(json.dumps(config, indent=2))
    all_frames = []
    for seed in args.seeds:
        per_seed_path = seed_dir / f"seed_{seed}.csv"
        if args.resume and per_seed_path.exists():
            try:
                completed = validate_seed_metrics(per_seed_path, seed, args)
            except (ValueError, KeyError, pd.errors.ParserError) as exc:
                print(f"[RESUME] seed {seed} metrics are incomplete ({exc}); rerunning this seed", flush=True)
                per_seed_path.unlink(missing_ok=True)
            else:
                print(f"[RESUME] validating and skipping completed seed {seed}", flush=True)
                all_frames.append(completed)
                continue
        result = run_seed(seed, train, val, test, scale, tile, args,
                          args.out, device, graph_t)
        all_frames.append(result)

    oracle_path = args.out / "oracle_metrics.csv"
    if args.resume and oracle_path.exists():
        oracle = pd.read_csv(oracle_path)
        if len(oracle) != args.test_trajectories * (args.steps - args.train_steps):
            raise ValueError(f"Existing oracle metrics are incomplete: {oracle_path}")
    else:
        oracle = oracle_metrics(test, args, tile)
        oracle.to_csv(oracle_path, index=False)
    metrics = pd.concat(all_frames + [oracle], ignore_index=True)
    write_worker_outputs(args.out, args, metrics, hashes, graph_error, device)
    print(
        f"[HERMES-PDE1-SPHERICAL] worker complete; output={args.out.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
