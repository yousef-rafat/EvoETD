#!/usr/bin/env python3
"""Parameter, inference-speed, training-step and memory benchmark for EvoETD.

Put beside these project files:
  run_connection_operator_extensions_v5.py
  run_hermes_pde_comparison.py
  run_ocean_connection_benchmark_v5.py

The script profiles the actual model classes and predictor paths used by the
project. It does NOT train full PDE experiments or recompute accuracy metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Set limits before importing numpy / torch / project modules.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import run_connection_operator_extensions_v5 as base
    import run_hermes_pde_comparison as hermes
    import run_ocean_connection_benchmark_v5 as legacy_core
except Exception as exc:
    raise ImportError(
        "Could not import EvoETD source files. Keep this benchmark beside "
        "run_connection_operator_extensions_v5.py, run_hermes_pde_comparison.py, "
        "and run_ocean_connection_benchmark_v5.py."
    ) from exc

HISTORY = 3
WIDTH = 32
MODES = 6
LAYERS_HERMES = 3
NU = 0.02
DT = 0.02

# method_key, displayed label, family, heat mode, geometry selector
METHODS = [
    ("conn_fibre", "Connection-ETD-Fibre", "new", "connection", "connection"),
    ("scalar_fibre", "Scalar-Laplacian-ETD-Fibre", "new", "connection", "scalar"),
    ("conn_fno", "Connection-ETD-FNOResidual", "new", "connection", "connection"),
    ("scalar_fno", "Scalar-Laplacian-ETD-FNOResidual", "new", "connection", "scalar"),
    ("fno2d", "FNO2D", "new", "connection", "none"),
    ("convgru", "ConvGRU", "new", "connection", "none"),
    ("hermes", "Hermes-Adapted", "hermes", "connection", "connection"),
    ("noheat_fibre", "NoHeat-Ablation / Fibre", "new", "none", "connection"),
    ("noheat_fno", "NoHeat-Ablation / FNOResidual", "new", "none", "connection"),
    ("legacy_conn_fibre", "Legacy-PDE1 / Connection-ETD-FibreEquivariant", "legacy", "connection", "connection"),
    ("legacy_scalar_fibre", "Legacy-PDE1 / Scalar-Laplacian-FibreEquivariant", "legacy", "connection", "scalar"),
    ("persistence", "Persistence (supplemental baseline)", "baseline", "none", "none"),
]

COLUMNS = [
    "device", "device_name", "worker", "spin", "height", "width", "batch_size_train",
    "model", "family", "operator_mode", "geometry_kind", "status", "error",
    "parameters_total", "parameters_trainable", "parameters_frozen",
    "parameter_storage_mb", "model_init_seconds", "geometry_build_seconds", "graph_edges",
    "inference_b1_median_ms", "inference_b1_mean_ms", "inference_b1_samples_per_s",
    "inference_b16_median_ms", "inference_b16_mean_ms", "inference_b16_samples_per_s",
    "train_step_b16_median_ms", "train_step_b16_mean_ms", "train_samples_per_s",
    "device_total_memory_mb", "vram_baseline_allocated_mb",
    "vram_infer_b1_peak_allocated_mb", "vram_infer_b1_peak_reserved_mb",
    "vram_infer_b16_peak_allocated_mb", "vram_infer_b16_peak_reserved_mb",
    "vram_train_peak_allocated_mb", "vram_train_peak_reserved_mb", "vram_train_incremental_mb",
    "cpu_peak_rss_process_mb", "torch_version", "threads",
]


def _set_threads(n: int) -> None:
    n = max(1, int(n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
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


def _rss_peak_mb() -> float:
    try:
        import resource
        # Linux reports KiB; macOS reports bytes.
        v = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return v / (1024.0 if sys.platform != "darwin" else 1024.0**2)
    except Exception:
        return float("nan")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(fn: Callable[[], object], device: torch.device, warmup: int,
             repeats: int) -> Tuple[float, float]:
    for _ in range(max(0, warmup)):
        fn()
    _sync(device)
    samples = []
    for _ in range(max(1, repeats)):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(samples)), float(statistics.mean(samples))


def _parameter_stats(model: Optional[nn.Module]) -> Tuple[int, int, int, float]:
    if model is None:
        return 0, 0, 0, 0.0
    params = list(model.parameters())
    total = sum(int(p.numel()) for p in params)
    trainable = sum(int(p.numel()) for p in params if p.requires_grad)
    frozen = total - trainable
    bytes_total = sum(int(p.numel() * p.element_size()) for p in params)
    return total, trainable, frozen, bytes_total / (1024.0**2)


def _new_model(key: str, device: torch.device, geom_conn: dict, geom_scalar: dict):
    in_channels = 4 * HISTORY + 6
    if key == "conn_fibre":
        model, kind = base.build_model("Connection-ETD-Fibre", in_channels, HISTORY, WIDTH, MODES, 20261118)
    elif key == "scalar_fibre":
        model, kind = base.build_model("Scalar-Laplacian-ETD-Fibre", in_channels, HISTORY, WIDTH, MODES, 20261118)
    elif key == "conn_fno":
        model, kind = base.build_model("Connection-ETD-FNOResidual", in_channels, HISTORY, WIDTH, MODES, 20261118)
    elif key == "scalar_fno":
        model, kind = base.build_model("Scalar-Laplacian-ETD-FNOResidual", in_channels, HISTORY, WIDTH, MODES, 20261118)
    elif key == "fno2d":
        model, kind = base.build_model("FNO2D", in_channels, HISTORY, WIDTH, MODES, 20261118)
    elif key == "convgru":
        model, kind = base.build_model("ConvGRU", in_channels, HISTORY, WIDTH, MODES, 20261118)
    elif key == "hermes":
        base.set_seed(20261618)
        model = hermes.HermesAdaptedMPNN(history=HISTORY, width=WIDTH, layers=LAYERS_HERMES)
        kind = "hermes_message"
        graph = hermes._make_graph_tensors(geom_conn, device)
        model.set_graph(graph)
        return model, kind, None, graph
    elif key in ("noheat_fibre", "noheat_fno"):
        name = "Connection-ETD-Fibre" if key == "noheat_fibre" else "Connection-ETD-FNOResidual"
        model, kind = base.build_model(name, in_channels, HISTORY, WIDTH, MODES, 20261118)
    elif key == "legacy_conn_fibre":
        legacy_core.set_seed(20261118)
        model = legacy_core.FibreConnectionETDModel(hist=HISTORY, width=WIDTH)
        kind = "legacy_connection_fibre"
    elif key == "legacy_scalar_fibre":
        legacy_core.set_seed(20261118)
        model = legacy_core.FibreConnectionETDModel(hist=HISTORY, width=WIDTH)
        kind = "legacy_scalar_fibre"
    else:
        raise KeyError(key)
    return model, kind, None, None


def _make_prediction(model, kind: str, x: torch.Tensor, last: torch.Tensor,
                     dt: torch.Tensor, geom_t, legacy_geom, graph) -> torch.Tensor:
    if kind == "hermes_message":
        return last + model(x)
    if kind.startswith("legacy_"):
        geom = legacy_geom["connection"] if kind == "legacy_connection_fibre" else legacy_geom["scalar"]
        return model(x, last, dt, geom, NU)
    return base.predict(model, kind, x, last, dt, geom_t, NU)


def _profile_task(key: str, label: str, family: str, heat_mode: str, geometry_kind: str,
                  spin: int, height: int, width: int, device: torch.device,
                  threads: int, warmup: int, repeats: int, worker: int) -> dict:
    row = {c: np.nan for c in COLUMNS}
    row.update(device=str(device), device_name="", worker=worker, spin=spin,
               height=height, width=width, batch_size_train=16, model=label,
               family=family, operator_mode=heat_mode, geometry_kind=geometry_kind,
               status="ERROR", error="", torch_version=str(torch.__version__), threads=threads)
    model = None
    try:
        device_total_memory = float("nan")
        if device.type == "cuda":
            row["device_name"] = torch.cuda.get_device_name(device)
            device_total_memory = torch.cuda.get_device_properties(device).total_memory / (1024.0**2)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        geometry_t0 = time.perf_counter()
        geom_conn = base.build_geometry(height, width, 0.8, spin, True)
        geom_scalar = base.build_geometry(height, width, 0.8, spin, False)
        # The historical PDE1 implementation's heat kernel consumes the same
        # V/eig/mass_sqrt/wet_idx tensor contract. Reuse the chart eigentensors
        # at the matching resolution here: the benchmark is profiling model
        # arithmetic, not comparing the two geometry-construction algorithms.
        # This avoids repeating a dense spherical-patch eigendecomposition for
        # every legacy architecture.
        legacy_geom = {"connection": geom_conn, "scalar": geom_scalar} if family == "legacy" else None
        geometry_build_seconds = time.perf_counter() - geometry_t0
        geom_cpu = geom_conn if geometry_kind == "connection" else geom_scalar
        geom_t = base.geometry_tensors(geom_cpu, device) if geometry_kind in ("connection", "scalar") else None

        init_t0 = time.perf_counter()
        if key == "persistence":
            model, kind, graph = None, "persistence", None
        else:
            model, kind, _, graph = _new_model(key, device, geom_conn, geom_scalar)
            model = model.to(device).eval()
        model_init_seconds = time.perf_counter() - init_t0

        # A batch of 16 is the experiment's normal training batch; batch 1 is its
        # autoregressive deployment/rollout path.
        def make_inputs(batch: int):
            gen = torch.Generator(device="cpu").manual_seed(7331 + batch + spin + height)
            xx = torch.randn((batch, 4 * HISTORY + 6, height, width), generator=gen, dtype=torch.float32).to(device)
            last = xx[:, 2 * (HISTORY - 1):2 * HISTORY]
            dd = torch.full((batch,), DT, dtype=torch.float32, device=device)
            return xx, last, dd

        x1, last1, dt1 = make_inputs(1)
        x16, last16, dt16 = make_inputs(16)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline_alloc = torch.cuda.memory_allocated(device)
        else:
            baseline_alloc = 0

        old_heat_mode = base.HEAT_ABLATION_MODE
        base.HEAT_ABLATION_MODE = "none" if heat_mode == "none" else "connection"

        def infer1():
            if kind == "persistence":
                return last1
            return _make_prediction(model, kind, x1, last1, dt1, geom_t, legacy_geom, graph)

        def infer16():
            if kind == "persistence":
                return last16
            return _make_prediction(model, kind, x16, last16, dt16, geom_t, legacy_geom, graph)

        with torch.inference_mode():
            b1_med, b1_mean = _measure(infer1, device, warmup, repeats)
            if device.type == "cuda":
                _sync(device)
                infer_b1_peak_alloc = torch.cuda.max_memory_allocated(device)
                infer_b1_peak_res = torch.cuda.max_memory_reserved(device)
                torch.cuda.reset_peak_memory_stats(device)
            else:
                infer_b1_peak_alloc = infer_b1_peak_res = float("nan")
            b16_med, b16_mean = _measure(infer16, device, warmup, repeats)
            if device.type == "cuda":
                _sync(device)
                infer_b16_peak_alloc = torch.cuda.max_memory_allocated(device)
                infer_b16_peak_res = torch.cuda.max_memory_reserved(device)
                torch.cuda.reset_peak_memory_stats(device)
            else:
                infer_b16_peak_alloc = infer_b16_peak_res = float("nan")

        total, trainable, frozen, param_mb = _parameter_stats(model)
        train_med = train_mean = float("nan")
        train_peak_alloc = train_peak_res = train_increment = float("nan")
        if model is not None:
            model.train()
            opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=8e-4)

            def train_step():
                opt.zero_grad(set_to_none=True)
                out = _make_prediction(model, kind, x16, last16, dt16, geom_t, legacy_geom, graph)
                loss = out.square().mean()
                loss.backward()
                opt.step()
                return loss

            # Allocate Adam state during warmup, then reset peak stats so the
            # recorded peak includes persistent model/optimizer + step activations.
            for _ in range(max(1, min(2, warmup))):
                train_step()
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                train_base = torch.cuda.memory_allocated(device)
            else:
                train_base = 0
            samples = []
            for _ in range(max(1, repeats)):
                _sync(device)
                t0 = time.perf_counter()
                train_step()
                _sync(device)
                samples.append((time.perf_counter() - t0) * 1000.0)
            train_med = float(statistics.median(samples))
            train_mean = float(statistics.mean(samples))
            if device.type == "cuda":
                train_peak_alloc = torch.cuda.max_memory_allocated(device)
                train_peak_res = torch.cuda.max_memory_reserved(device)
                train_increment = max(0.0, train_peak_alloc - train_base) / (1024.0**2)

        # Leave process-global ablation state clean for the next model.
        base.HEAT_ABLATION_MODE = old_heat_mode
        graph_edges = int(graph["edge_src"].numel()) if graph is not None else 0
        row.update(
            status="OK", parameters_total=total, parameters_trainable=trainable,
            parameters_frozen=frozen, parameter_storage_mb=param_mb,
            model_init_seconds=model_init_seconds, geometry_build_seconds=geometry_build_seconds,
            graph_edges=graph_edges, inference_b1_median_ms=b1_med,
            inference_b1_mean_ms=b1_mean,
            inference_b1_samples_per_s=(1000.0 / b1_med if b1_med > 0 else float("nan")),
            inference_b16_median_ms=b16_med, inference_b16_mean_ms=b16_mean,
            inference_b16_samples_per_s=(16000.0 / b16_med if b16_med > 0 else float("nan")),
            train_step_b16_median_ms=train_med, train_step_b16_mean_ms=train_mean,
            train_samples_per_s=(16000.0 / train_med if train_med > 0 and math.isfinite(train_med) else float("nan")),
            device_total_memory_mb=device_total_memory,
            vram_baseline_allocated_mb=(baseline_alloc / (1024.0**2) if device.type == "cuda" else float("nan")),
            vram_infer_b1_peak_allocated_mb=(infer_b1_peak_alloc / (1024.0**2) if device.type == "cuda" else float("nan")),
            vram_infer_b1_peak_reserved_mb=(infer_b1_peak_res / (1024.0**2) if device.type == "cuda" else float("nan")),
            vram_infer_b16_peak_allocated_mb=(infer_b16_peak_alloc / (1024.0**2) if device.type == "cuda" else float("nan")),
            vram_infer_b16_peak_reserved_mb=(infer_b16_peak_res / (1024.0**2) if device.type == "cuda" else float("nan")),
            vram_train_peak_allocated_mb=(train_peak_alloc / (1024.0**2) if device.type == "cuda" else float("nan")),
            vram_train_peak_reserved_mb=(train_peak_res / (1024.0**2) if device.type == "cuda" else float("nan")),
            vram_train_incremental_mb=train_increment,
            cpu_peak_rss_process_mb=_rss_peak_mb(),
        )
        print(f"[OK][{device}][spin={spin} {height}x{width}] {label}: params={total:,} "
              f"infer1={b1_med:.3f} ms train16={train_med:.3f} ms", flush=True)
        return row
    except Exception as exc:
        row.update(status="ERROR", error=f"{exc!r}\n{traceback.format_exc()}", cpu_peak_rss_process_mb=_rss_peak_mb())
        print(f"[ERROR][{device}][spin={spin} {height}x{width}] {label}: {exc!r}", flush=True)
        return row
    finally:
        try:
            base.HEAT_ABLATION_MODE = "connection"
        except Exception:
            pass
        if model is not None:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _parse_resolution(value: str) -> Tuple[int, int]:
    try:
        h, w = value.lower().split("x", 1)
        h, w = int(h), int(w)
        if h < 4 or w < 4:
            raise ValueError
        return h, w
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid resolution {value!r}; use HxW such as 12x16") from exc


def _run_worker(args, task_indices: Optional[List[int]] = None) -> pd.DataFrame:
    device = torch.device(args.device)
    _set_threads(args.cpu_threads)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.quick:
        warmup, repeats = 1, 3
    else:
        warmup, repeats = (5, 20) if device.type == "cuda" else (2, 8)
    tasks = []
    for spin in args.spins:
        for h, w in args.resolutions:
            for method in METHODS:
                key, label, family, heat_mode, geom_kind = method
                # The older spherical-patch FibreEquivariant implementation
                # belongs to the PDE1 spin-1 suite only.
                if family == "legacy" and spin != 1:
                    continue
                tasks.append((method, spin, h, w))
    if task_indices is not None:
        tasks = [tasks[i] for i in task_indices]
    rows = []
    for (key, label, family, heat_mode, geom_kind), spin, h, w in tasks:
        row = _profile_task(key, label, family, heat_mode, geom_kind, spin, h, w,
                            device, args.cpu_threads, warmup, repeats, args.worker_index)
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def _build_tasks(args):
    tasks = []
    for spin in args.spins:
        for h, w in args.resolutions:
            for method in METHODS:
                key, label, family, heat_mode, geom_kind = method
                if family == "legacy" and spin != 1:
                    continue
                tasks.append((method, spin, h, w))
    return tasks


def _parent_two_gpu(args) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    n_gpu = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
    tasks = _build_tasks(args)
    workers = min(2, n_gpu)
    procs = []
    for wi in range(workers):
        assigned = list(range(wi, len(tasks), workers))
        task_file = out / f"worker_{wi}_tasks.json"
        task_file.write_text(json.dumps(assigned))
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--worker", "--device", "cuda:0", "--cpu-threads", str(max(1, args.cpu_threads // workers)),
               "--out", str(out), "--worker-index", str(wi), "--task-file", str(task_file),
               "--resolutions", *[f"{h}x{w}" for h, w in args.resolutions],
               "--spins", *[str(x) for x in args.spins]]
        if args.quick:
            cmd.append("--quick")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(wi)
        env["OMP_NUM_THREADS"] = str(max(1, args.cpu_threads // workers))
        env["MKL_NUM_THREADS"] = str(max(1, args.cpu_threads // workers))
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = str(max(1, args.cpu_threads // workers))
        log = (out / f"gpu{wi}.log").open("w", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, wi, assigned))
        print(f"[LAUNCH] GPU {wi} ({gpu_names[wi]}): {len(assigned)} tasks", flush=True)
    failures = []
    for proc, log, wi, assigned in procs:
        code = proc.wait()
        log.close()
        if code != 0:
            failures.append((wi, code))
    worker_dfs = []
    for wi in range(workers):
        path = out / f"worker_{wi}_results.csv"
        if path.exists():
            worker_dfs.append(pd.read_csv(path))
    merged = pd.concat(worker_dfs, ignore_index=True) if worker_dfs else pd.DataFrame(columns=COLUMNS)
    merged.to_csv(out / "efficiency_results.csv", index=False)
    _write_summary(merged, out)
    manifest = _manifest(args, "two_t4_parallel", gpu_names)
    manifest["worker_failures"] = failures
    manifest["task_count"] = len(tasks)
    manifest["tasks_completed"] = int((merged.status == "OK").sum()) if not merged.empty else 0
    (out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2))
    if failures:
        raise RuntimeError(f"GPU worker failures: {failures}; inspect {out}/gpu*.log")
    print(f"[DONE] Results: {out / 'efficiency_results.csv'}", flush=True)


def _write_summary(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    ok = df[df.status == "OK"].copy()
    if ok.empty:
        return
    group_cols = ["model", "family", "operator_mode", "spin", "height", "width", "device"]
    cols = [c for c in ["parameters_total", "parameters_trainable", "parameter_storage_mb",
                        "inference_b1_median_ms", "inference_b16_median_ms",
                        "train_step_b16_median_ms", "device_total_memory_mb",
                        "vram_baseline_allocated_mb", "vram_infer_b1_peak_allocated_mb",
                        "vram_infer_b1_peak_reserved_mb", "vram_infer_b16_peak_allocated_mb",
                        "vram_infer_b16_peak_reserved_mb", "vram_train_peak_allocated_mb",
                        "vram_train_peak_reserved_mb", "vram_train_incremental_mb"] if c in ok]
    summary = ok.groupby(group_cols, dropna=False)[cols].median().reset_index()
    summary.to_csv(out / "efficiency_summary.csv", index=False)


def _manifest(args, mode: str, gpu_names: Optional[List[str]] = None) -> dict:
    return {
        "benchmark": "EvoETD model parameter/speed/memory efficiency",
        "benchmark_version": "1.1",
        "created_unix": time.time(),
        "mode": mode,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_names": gpu_names or ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                                    if torch.cuda.is_available() else []),
        "model_width": WIDTH,
        "fno_modes": MODES,
        "history": HISTORY,
        "hermes_layers": LAYERS_HERMES,
        "batch_sizes": [1, 16],
        "spins": list(args.spins),
        "resolutions": [[h, w] for h, w in args.resolutions],
        "methods": [{"name": m[1], "family": m[2], "operator_mode": m[3], "geometry": m[4]} for m in METHODS],
        "measurement": {
            "inference": "median and mean synchronized wall-clock forward latency; eval/inference_mode",
            "training": "single synthetic Adam update including forward, backward and optimizer step; batch=16",
            "vram": "PyTorch peak allocated/reserved CUDA memory for batch-1 inference, batch-16 inference, and batch-16 train step; includes resident model/geometry, excludes other processes and CUDA context overhead",
            "parameters": "total/trainable/frozen parameter counts and actual tensor storage in MiB",
            "limitations": "No full training, accuracy or PDE solver cost is included; this is network/operator-step profiling only. Persistence is supplemental. Oracle-Connection-Strang is a numerical reference solver rather than a learned architecture and is not mixed into neural throughput comparisons."
        },
        "cpu_threads": args.cpu_threads,
        "quick": bool(args.quick),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("evoetd_efficiency"))
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda:0, cuda:1")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--resolutions", nargs="+", type=_parse_resolution, default=[(12, 16), (18, 24)])
    ap.add_argument("--spins", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    ap.add_argument("--quick", action="store_true", help="Short diagnostic timing run; use normal mode for final measurements")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--task-file", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.out = args.out.resolve()

    if args.device == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2 and not args.worker:
            _parent_two_gpu(args)
            return
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.worker:
        if args.task_file is None or not args.task_file.exists():
            raise FileNotFoundError("GPU worker requires --task-file")
        tasks = _build_tasks(args)
        indices = json.loads(args.task_file.read_text())
        df = _run_worker(args, indices)
        path = args.out / f"worker_{args.worker_index}_results.csv"
        df.to_csv(path, index=False)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    df = _run_worker(args)
    df.to_csv(args.out / "efficiency_results.csv", index=False)
    _write_summary(df, args.out)
    (args.out / "benchmark_manifest.json").write_text(json.dumps(_manifest(args, "single_device"), indent=2))
    print(f"[DONE] Results: {args.out / 'efficiency_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
