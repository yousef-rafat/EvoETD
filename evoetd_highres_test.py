#!/usr/bin/env python3
"""EvoETD high-resolution scaling stress test v2.

Core methods (compact four-row comparison):
  * FNO2D
  * ConvGRU
  * Hermes-Adapted + Chebyshev ETD
  * Connection-ETD-Fibre (the intended EvoETD method)

Optional diagnostic control:
  * Connection-ETD-FNOResidual (--include-fnoresidual)

Both Connection-ETD-Fibre and Hermes-Adapted use the same matrix-free,
area-normalized connection Bochner operator and the same Chebyshev actions for
exp(-nu*dt*A) and phi_1(-nu*dt*A). Hermes keeps its transported-neighbour
message equations but computes the four grid directions as local stencil
slices (no materialized edge index/transport graph). Its message blocks use
activation checkpointing during training to limit edge-message activation
memory. This is a benchmark adapter, not an official Hermes/ESCNN reproduction.

This is a synthetic-input compute/memory stress test. It does not measure
converged predictive accuracy or zero-shot resolution transfer.
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
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from scipy.fft import dct

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import run_connection_operator_extensions_v5 as base
    import run_hermes_pde_comparison as hermes
except Exception as exc:
    raise ImportError(
        "Keep run_connection_operator_extensions_v5.py and "
        "run_hermes_pde_comparison.py beside this script."
    ) from exc

HISTORY = 3
WIDTH = 32
MODES = 6
HERMES_LAYERS = 3
NU = 0.02
DT = 0.02
CURVATURE = 0.8
X_EXTENT = 1.6
Y_EXTENT = 1.2
IN_CHANNELS = 4 * HISTORY + 6
SEED = 20261118

CORE_METHODS = (
    "FNO2D",
    "ConvGRU",
    "Hermes-Adapted + Cheb-ETD",
    "Connection-ETD-Fibre",
)
OPTIONAL_METHODS = ("Connection-ETD-FNOResidual",)
HERMES_METHOD = "Hermes-Adapted + Cheb-ETD"

RESULT_COLUMNS = [
    "model", "spin", "height", "width", "nodes", "device", "device_name", "worker",
    "parameters_total", "parameters_trainable", "parameter_storage_mb",
    "batch_inference", "inference_median_ms", "inference_mean_ms", "inference_samples_per_s",
    "batch_train", "train_step_median_ms", "train_step_mean_ms", "train_samples_per_s",
    "device_total_memory_mb", "baseline_allocated_mb",
    "infer_peak_allocated_mb", "infer_peak_reserved_mb", "infer_incremental_mb",
    "train_peak_allocated_mb", "train_peak_reserved_mb", "train_incremental_mb",
    "geometry_build_seconds", "operator_build_seconds", "model_init_seconds",
    "graph_directed_edges", "graph_storage_mode", "hermes_block_checkpointing",
    "cheb_degree_exp", "cheb_degree_phi1", "lambda_max_bound",
    "operator_apply_ms", "etd_apply_ms", "inference_status", "training_status", "status", "error",
]


def set_threads(n: int) -> None:
    torch.set_num_threads(max(1, int(n)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=max(1, int(n)))
    except Exception:
        pass


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cell_widths(x: np.ndarray) -> np.ndarray:
    edges = np.empty(len(x) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return np.diff(edges)


def grid_metadata(height: int, width: int, spin: int = 1) -> Dict[str, np.ndarray | float | int]:
    """Local finite-volume geometry; no matrix, eigenbasis, or per-edge Python objects."""
    if height < 5 or width < 5:
        raise ValueError("Resolution must be at least 5x5.")
    if spin not in (1, 2):
        raise ValueError("spin must be 1 or 2")
    x = np.linspace(-X_EXTENT / 2, X_EXTENT / 2, width, dtype=np.float64)
    y = np.linspace(-Y_EXTENT / 2, Y_EXTENT / 2, height, dtype=np.float64)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dxw = cell_widths(x)
    dyw = cell_widths(y)
    root_k = math.sqrt(CURVATURE)
    f = np.cos(root_k * y)
    fp = -root_k * np.sin(root_k * y)
    if np.any(f <= 0):
        raise ValueError("Metric factor became non-positive on requested chart.")

    # Edge conductances in the same finite-volume convention as build_geometry.
    wh = dyw / (f * dx)  # one horizontal conductance per row
    fmid = 0.5 * (f[:-1] + f[1:])
    wv = fmid[:, None] * dxw[None, :] / dy
    area = f[:, None] * dyw[:, None] * dxw[None, :]
    mass_sqrt = np.sqrt(area)
    angle = float(spin) * fp * dx
    hc, hs = np.cos(angle), np.sin(angle)

    # Gershgorin upper bound for the symmetric area-normalized operator
    # A = M^{-1/2} K_A M^{-1/2}. Each 2x2 rotation block has row abs-sum |c|+|s|.
    rowsum = np.zeros((height, width), dtype=np.float64)
    inv_area = 1.0 / area
    h_off = wh[:, None] * (np.abs(hc) + np.abs(hs))[:, None] / np.sqrt(area[:, :-1] * area[:, 1:])
    rowsum[:, :-1] += wh[:, None] * inv_area[:, :-1] + h_off
    rowsum[:, 1:] += wh[:, None] * inv_area[:, 1:] + h_off
    v_off = wv / np.sqrt(area[:-1, :] * area[1:, :])
    rowsum[:-1, :] += wv * inv_area[:-1, :] + v_off
    rowsum[1:, :] += wv * inv_area[1:, :] + v_off
    lambda_max = float(rowsum.max()) * (1.0 + 1e-5)
    if not math.isfinite(lambda_max) or lambda_max <= 0:
        raise FloatingPointError("Invalid Gershgorin spectral bound.")
    return {
        "height": height, "width": width, "spin": spin,
        "mass_sqrt": mass_sqrt.astype(np.float32),
        "inv_mass_sqrt": (1.0 / mass_sqrt).astype(np.float32),
        "h_cos": hc.astype(np.float32), "h_sin": hs.astype(np.float32),
        "h_weight": wh.astype(np.float32), "v_weight": wv.astype(np.float32),
        "lambda_max_bound": lambda_max,
        "area": area.astype(np.float64),
    }


def chebyshev_coefficients(lambda_max: float, tau: float, function: str,
                           tolerance: float = 1e-6, samples: int = 4096,
                           max_degree: int = 1024) -> np.ndarray:
    """DCT-generated Chebyshev coefficients on [0, lambda_max].

    function is exp: exp(-tau*lambda), or phi1: expm1(-tau*lambda)/(-tau*lambda).
    Truncation uses the L1 tail of the coefficient sequence as a uniform bound.
    """
    if tau < 0 or lambda_max <= 0:
        raise ValueError("tau must be nonnegative and lambda_max positive")
    theta = np.pi * (np.arange(samples, dtype=np.float64) + 0.5) / samples
    x = np.cos(theta)
    lam = 0.5 * lambda_max * (x + 1.0)
    z = -tau * lam
    if function == "exp":
        values = np.exp(z)
    elif function == "phi1":
        values = np.ones_like(z)
        nz = np.abs(z) > 1e-8
        values[nz] = np.expm1(z[nz]) / z[nz]
        values[~nz] = 1.0 + 0.5 * z[~nz] + z[~nz] ** 2 / 6.0
    else:
        raise ValueError(f"unknown function {function}")
    coeff = dct(values, type=2) / samples
    coeff[0] *= 0.5
    tail = np.cumsum(np.abs(coeff[::-1]))[::-1]
    candidates = np.flatnonzero(tail <= tolerance)
    degree = int(candidates[0]) if len(candidates) else min(max_degree, samples - 1)
    degree = min(degree, max_degree)
    # Fail rather than silently claiming the requested approximation tolerance.
    actual_tail = float(np.sum(np.abs(coeff[degree + 1:])))
    if actual_tail > tolerance * 1.05 and degree >= max_degree:
        raise RuntimeError(
            f"Chebyshev degree cap {max_degree} insufficient for {function}: "
            f"coefficient tail={actual_tail:.3e}, tolerance={tolerance:.3e}."
        )
    return coeff[:degree + 1].astype(np.float32)


class MatrixFreeConnectionOperator:
    """Apply the area-normalized connection Laplacian using local edge actions."""
    def __init__(self, meta: Dict[str, object], device: torch.device):
        self.height = int(meta["height"])
        self.width = int(meta["width"])
        self.lambda_max = float(meta["lambda_max_bound"])
        self.inv_mass_sqrt = torch.as_tensor(meta["inv_mass_sqrt"], device=device)
        self.h_cos = torch.as_tensor(meta["h_cos"], device=device).view(1, self.height, 1)
        self.h_sin = torch.as_tensor(meta["h_sin"], device=device).view(1, self.height, 1)
        self.h_weight = torch.as_tensor(meta["h_weight"], device=device).view(1, self.height, 1)
        self.v_weight = torch.as_tensor(meta["v_weight"], device=device).view(1, self.height - 1, self.width, 1)

    @torch.no_grad()
    def apply_A(self, z: torch.Tensor) -> torch.Tensor:
        """z is [B,H,W,2], in the symmetric M^{-1/2} coordinates."""
        q = z * self.inv_mass_sqrt[None, :, :, None]
        kq = torch.zeros_like(q)

        # Horizontal transported differences: q_i - P_(j->i) q_j.
        ql = q[:, :, :-1, :]
        qr = q[:, :, 1:, :]
        c, s = self.h_cos, self.h_sin
        pu = c * qr[..., 0] - s * qr[..., 1]
        pv = s * qr[..., 0] + c * qr[..., 1]
        du = ql[..., 0] - pu
        dv = ql[..., 1] - pv
        wh = self.h_weight
        kq[:, :, :-1, 0] += wh * du
        kq[:, :, :-1, 1] += wh * dv
        # Reverse edge uses P^T.
        ptu = c * du + s * dv
        ptv = -s * du + c * dv
        kq[:, :, 1:, 0] -= wh * ptu
        kq[:, :, 1:, 1] -= wh * ptv

        # Vertical transport is identity in the chosen chart frame.
        qtop = q[:, :-1, :, :]
        qbottom = q[:, 1:, :, :]
        dvv = qtop - qbottom
        wv = self.v_weight
        kq[:, :-1, :, :] += wv * dvv
        kq[:, 1:, :, :] -= wv * dvv
        return kq * self.inv_mass_sqrt[None, :, :, None]

    @torch.no_grad()
    def apply_scaled(self, z: torch.Tensor) -> torch.Tensor:
        return (2.0 / self.lambda_max) * self.apply_A(z) - z


@torch.no_grad()
def chebyshev_apply(x: torch.Tensor, op: MatrixFreeConnectionOperator,
                    coeff: torch.Tensor) -> torch.Tensor:
    """Apply sum_k c_k T_k(2A/lambda_max-I) to one or more fields."""
    if coeff.numel() == 1:
        return coeff[0] * x
    t0 = x
    t1 = op.apply_scaled(x)
    out = coeff[0] * t0 + coeff[1] * t1
    for k in range(1, coeff.numel() - 1):
        t2 = 2.0 * op.apply_scaled(t1) - t0
        out.add_(coeff[k + 1] * t2)
        t0, t1 = t1, t2
    return out


class _SymmetricChebyshevAction(torch.autograd.Function):
    """Memory-saving autograd: f(A) is symmetric, so backward applies f(A) again."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, op: MatrixFreeConnectionOperator,
                coeff: torch.Tensor) -> torch.Tensor:
        ctx.op = op
        ctx.coeff = coeff
        return chebyshev_apply(x, op, coeff)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = chebyshev_apply(grad_output, ctx.op, ctx.coeff)
        return grad_x, None, None


def make_cheb_etd_operator(height: int, width: int, spin: int,
                           device: torch.device, tolerance: float = 1e-6,
                           meta: Optional[Dict[str, object]] = None):
    t0 = time.perf_counter()
    if meta is None:
        meta = grid_metadata(height, width, spin)
    op = MatrixFreeConnectionOperator(meta, device)
    tau = NU * DT
    ce = chebyshev_coefficients(op.lambda_max, tau, "exp", tolerance=tolerance)
    cp = chebyshev_coefficients(op.lambda_max, tau, "phi1", tolerance=tolerance)
    ce_t = torch.as_tensor(ce, device=device)
    cp_t = torch.as_tensor(cp, device=device)
    return op, ce_t, cp_t, len(ce) - 1, len(cp) - 1, time.perf_counter() - t0


def sparse_cheb_etd(last: torch.Tensor, residual: torch.Tensor, dt: torch.Tensor,
                    op: MatrixFreeConnectionOperator, coeff_exp: torch.Tensor,
                    coeff_phi: torch.Tensor) -> torch.Tensor:
    # Match the original heat_etd's interleaved fibre ordering and mass transform.
    ms = 1.0 / op.inv_mass_sqrt
    x = last.permute(0, 2, 3, 1).contiguous() * ms[None, :, :, None]
    r = residual.permute(0, 2, 3, 1).contiguous() * ms[None, :, :, None]
    ex = _SymmetricChebyshevAction.apply(x, op, coeff_exp)
    ph = _SymmetricChebyshevAction.apply(r, op, coeff_phi)
    y = (ex + dt.view(-1, 1, 1, 1) * ph) * op.inv_mass_sqrt[None, :, :, None]
    return y.permute(0, 3, 1, 2).contiguous()


def make_local_hermes_geometry(meta: Dict[str, object],
                               device: torch.device) -> Dict[str, torch.Tensor]:
    """Geometry tensors for exact four-neighbour Hermes messages, without edges.

    The original graph has two directed edges per undirected horizontal or
    vertical neighbour pair. The edge attributes are log(weight / global
    mean edge weight); the node factor is inverse square-root degree.
    """
    h = int(meta["height"]); w = int(meta["width"])
    wh = np.asarray(meta["h_weight"], dtype=np.float64)  # [H]
    wv = np.asarray(meta["v_weight"], dtype=np.float64)  # [H-1,W]
    n_h = h * (w - 1)
    n_v = (h - 1) * w
    wmean = max((float(wh.sum()) * (w - 1) + float(wv.sum())) / max(n_h + n_v, 1),
                1e-12)
    edge_attr_h = np.log(np.maximum(wh[:, None] / wmean, 1e-8))
    edge_attr_h = np.broadcast_to(edge_attr_h, (h, w - 1)).copy()
    edge_attr_v = np.log(np.maximum(wv / wmean, 1e-8))

    degree = np.zeros((h, w), dtype=np.float32)
    degree[:, :-1] += 1.0
    degree[:, 1:] += 1.0
    degree[:-1, :] += 1.0
    degree[1:, :] += 1.0
    inv_degree = 1.0 / np.sqrt(np.maximum(degree, 1.0))

    return {
        "h_cos": torch.as_tensor(np.asarray(meta["h_cos"], dtype=np.float32), device=device),
        "h_sin": torch.as_tensor(np.asarray(meta["h_sin"], dtype=np.float32), device=device),
        "edge_attr_h": torch.as_tensor(edge_attr_h[None, :, :, None], device=device, dtype=torch.float32),
        "edge_attr_v": torch.as_tensor(edge_attr_v[None, :, :, None], device=device, dtype=torch.float32),
        "inv_sqrt_degree": torch.as_tensor(inv_degree[None, :, :, None], device=device),
        "height": h,
        "width": w,
    }


def _hermes_direction(block, scalar_src: torch.Tensor, scalar_dst: torch.Tensor,
                      fibre_src: torch.Tensor, fibre_dst: torch.Tensor,
                      transported: torch.Tensor, edge_attr: torch.Tensor):
    """Run one orientation of edge messages; edge tensors are [B,H,Wedge,...]."""
    b, eh, ew, p, _ = transported.shape
    cross_gram = torch.einsum("bhwpc,bhwqc->bhwpq", fibre_dst, transported)
    edge_scalar = torch.cat([
        scalar_dst,
        scalar_src,
        cross_gram.reshape(b, eh, ew, p * p),
        edge_attr.expand(b, -1, -1, -1),
    ], dim=-1)
    edge_out = block.edge_net(edge_scalar)
    scalar_message = edge_out[..., :block.width]
    beta = torch.tanh(edge_out[..., block.width:].reshape(b, eh, ew, p, p)) / math.sqrt(max(p, 1))
    fibre_message = torch.einsum("bhwaq,bhwqc->bhwac", beta, transported)
    return scalar_message, fibre_message


def local_hermes_message_block(block, scalar: torch.Tensor, fibre: torch.Tensor,
                               geom: Dict[str, torch.Tensor]):
    """Exact local-stencil form of HermesMessageLayer.forward.

    It preserves the four directed transport/message/aggregation operations
    and their ordering, but never creates src/dst/transport arrays of size O(N).
    """
    b, h, w, _ = scalar.shape
    p = fibre.shape[-2]
    hc = geom["h_cos"].view(1, h, 1, 1)
    hs = geom["h_sin"].view(1, h, 1, 1)

    agg_scalar = torch.zeros_like(scalar)
    agg_fibre = torch.zeros_like(fibre)

    # Direction 1: source right -> target left, transport by P.
    fs = fibre[:, :, 1:, :, :]
    transported = torch.stack((hc * fs[..., 0] - hs * fs[..., 1],
                               hs * fs[..., 0] + hc * fs[..., 1]), dim=-1)
    sm, fm = _hermes_direction(block, scalar[:, :, 1:, :], scalar[:, :, :-1, :],
                               fs, fibre[:, :, :-1, :, :], transported,
                               geom["edge_attr_h"])
    agg_scalar[:, :, :-1, :] += sm
    agg_fibre[:, :, :-1, :, :] += fm

    # Direction 2: source left -> target right, transport by P^T.
    fs = fibre[:, :, :-1, :, :]
    transported = torch.stack((hc * fs[..., 0] + hs * fs[..., 1],
                               -hs * fs[..., 0] + hc * fs[..., 1]), dim=-1)
    sm, fm = _hermes_direction(block, scalar[:, :, :-1, :], scalar[:, :, 1:, :],
                               fs, fibre[:, :, 1:, :, :], transported,
                               geom["edge_attr_h"])
    agg_scalar[:, :, 1:, :] += sm
    agg_fibre[:, :, 1:, :, :] += fm

    # Direction 3: source bottom -> target top, identity transport.
    fs = fibre[:, 1:, :, :, :]
    sm, fm = _hermes_direction(block, scalar[:, 1:, :, :], scalar[:, :-1, :, :],
                               fs, fibre[:, :-1, :, :, :], fs,
                               geom["edge_attr_v"])
    agg_scalar[:, :-1, :, :] += sm
    agg_fibre[:, :-1, :, :, :] += fm

    # Direction 4: source top -> target bottom, identity transport.
    fs = fibre[:, :-1, :, :, :]
    sm, fm = _hermes_direction(block, scalar[:, :-1, :, :], scalar[:, 1:, :, :],
                               fs, fibre[:, 1:, :, :, :], fs,
                               geom["edge_attr_v"])
    agg_scalar[:, 1:, :, :] += sm
    agg_fibre[:, 1:, :, :, :] += fm

    inv_degree = geom["inv_sqrt_degree"]
    agg_scalar = agg_scalar * inv_degree
    agg_fibre = agg_fibre * inv_degree.unsqueeze(-1)

    flat_fibre = fibre.reshape(b, h * w, p, 2)
    flat_agg_fibre = agg_fibre.reshape(b, h * w, p, 2)
    node_scalar = torch.cat([
        scalar,
        agg_scalar,
        hermes._upper_gram(flat_fibre).reshape(b, h, w, -1),
        hermes._upper_gram(flat_agg_fibre).reshape(b, h, w, -1),
    ], dim=-1)
    scalar_new = block.node_norm(scalar + block.node_net(node_scalar))

    mix = 0.10 * torch.tanh(block.fibre_mix(scalar_new)).reshape(b, h, w, 2, p, p)
    self_mix, neighbour_mix = mix[:, :, :, 0], mix[:, :, :, 1]
    fibre_new = (
        fibre
        + torch.einsum("bhwaq,bhwqc->bhwac", self_mix, fibre)
        + torch.einsum("bhwaq,bhwqc->bhwac", neighbour_mix, agg_fibre)
    )
    return scalar_new, fibre_new


class MatrixFreeHermesAdaptedMPNN(hermes.HermesAdaptedMPNN):
    """Hermes-Adapted with implicit stencil geometry and checkpointed blocks."""
    def __init__(self, history: int, width: int = 32, layers: int = 3,
                 checkpoint_blocks: bool = True):
        super().__init__(history=history, width=width, layers=layers)
        self._local_geometry = None
        self.checkpoint_blocks = bool(checkpoint_blocks)

    def set_local_geometry(self, geom: Dict[str, torch.Tensor]) -> None:
        self._local_geometry = geom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._local_geometry is None:
            raise RuntimeError("Hermes local stencil geometry has not been assigned.")
        b, c, h, w = x.shape
        p = self.history
        if c != 4 * p + 6:
            raise ValueError(f"Expected {4*p+6} input channels, got {c}.")
        history = x[:, :2 * p].reshape(b, p, 2, h, w)
        fibre = history.permute(0, 3, 4, 1, 2).reshape(b, h * w, p, 2)
        gram = hermes._upper_gram(fibre)
        mag = torch.sqrt(torch.diagonal(
            torch.einsum("bnac,bndc->bnad", fibre, fibre), dim1=-2, dim2=-1
        ).clamp_min(0.0) + 1e-12)
        logmag = torch.log1p(mag)
        extra = x[:, 2 * p:].permute(0, 2, 3, 1).reshape(b, h * w, 2 * p + 6)
        scalar = self.scalar_encoder(torch.cat([gram, logmag, extra], dim=-1))
        scalar = scalar.reshape(b, h, w, self.width)
        fibre = fibre.reshape(b, h, w, p, 2)

        for block in self.blocks:
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                scalar, fibre = checkpoint(
                    lambda sc, fi, block=block: local_hermes_message_block(
                        block, sc, fi, self._local_geometry),
                    scalar, fibre, use_reentrant=False)
            else:
                scalar, fibre = local_hermes_message_block(
                    block, scalar, fibre, self._local_geometry)

        alpha = torch.tanh(self.readout(scalar.reshape(b, h * w, self.width))) / float(max(p, 1))
        residual = torch.einsum("bnp,bnpc->bnc", alpha, fibre.reshape(b, h * w, p, 2))
        residual = residual.reshape(b, h, w, 2).permute(0, 3, 1, 2).contiguous()
        return base.bound_vector_residual(residual, self.residual_bound)


def make_model(name: str, device: torch.device):
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    if name == HERMES_METHOD:
        model = MatrixFreeHermesAdaptedMPNN(
            history=HISTORY, width=WIDTH, layers=HERMES_LAYERS,
            checkpoint_blocks=True)
        return model.to(device), "hermes_cheb"
    if name == "Connection-ETD-Fibre":
        model, kind = base.build_model(name, IN_CHANNELS, HISTORY, WIDTH, MODES, SEED)
        return model.to(device), kind
    if name == "Connection-ETD-FNOResidual":
        model, kind = base.build_model(name, IN_CHANNELS, HISTORY, WIDTH, MODES, SEED)
        return model.to(device), kind
    model, kind = base.build_model(name, IN_CHANNELS, HISTORY, WIDTH, MODES, SEED)
    return model.to(device), kind


def predict(name: str, model: nn.Module, kind: str, x: torch.Tensor,
            op_bundle, dt: torch.Tensor) -> torch.Tensor:
    last = x[:, 2 * (HISTORY - 1):2 * HISTORY]
    if name in (HERMES_METHOD, "Connection-ETD-Fibre", "Connection-ETD-FNOResidual"):
        if op_bundle is None:
            raise RuntimeError(f"{name} requires the shared matrix-free Chebyshev ETD operator.")
        op, ce, cp = op_bundle
        if name == HERMES_METHOD:
            residual = model(x)
        elif name == "Connection-ETD-Fibre":
            residual = base.bound_vector_residual(model.net(x), model.residual_bound)
        else:
            residual = base.bound_vector_residual(model.net(x), model.residual_bound)
        return sparse_cheb_etd(last, residual, dt, op, ce, cp)
    return last + model(x)

def parameter_stats(model: nn.Module) -> Tuple[int, int, float]:
    params = list(model.parameters())
    total = sum(int(p.numel()) for p in params)
    trainable = sum(int(p.numel()) for p in params if p.requires_grad)
    nbytes = sum(int(p.numel() * p.element_size()) for p in params)
    return total, trainable, nbytes / (1024.0 ** 2)


def measure(fn, device: torch.device, warmup: int, repeats: int) -> Tuple[float, float]:
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(samples)), float(statistics.mean(samples))


def mb(x: int) -> float:
    return float(x) / (1024.0 ** 2)


def run_task(name: str, height: int, width: int, spin: int, device: torch.device,
             threads: int, warmup: int, repeats: int, train_repeats: int,
             worker: int, cheb_tolerance: float) -> dict:
    row = {k: np.nan for k in RESULT_COLUMNS}
    row.update(model=name, spin=spin, height=height, width=width, nodes=height*width,
               device=str(device), device_name=(torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"),
               worker=worker, batch_inference=1, batch_train=1,
               inference_status="NOT_RUN", training_status="NOT_RUN", status="ERROR", error="",
               graph_storage_mode="none", hermes_block_checkpointing=False)
    model = None; op_bundle = None; local_hermes_geom = None; x = None; opt = None
    try:
        set_threads(threads)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            row["device_total_memory_mb"] = mb(torch.cuda.get_device_properties(device).total_memory)

        needs_geometry = name in (HERMES_METHOD, "Connection-ETD-Fibre", "Connection-ETD-FNOResidual")
        meta = None
        t0 = time.perf_counter()
        if needs_geometry:
            meta = grid_metadata(height, width, spin)
        if name == HERMES_METHOD:
            local_hermes_geom = make_local_hermes_geometry(meta, device)
            row["graph_directed_edges"] = int(2 * (height * (width - 1) + (height - 1) * width))
            row["graph_storage_mode"] = "implicit_four_direction_stencil"
        row["geometry_build_seconds"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if needs_geometry:
            op, ce, cp, de, dp, op_time = make_cheb_etd_operator(
                height, width, spin, device, tolerance=cheb_tolerance, meta=meta)
            op_bundle = (op, ce, cp)
            row["operator_build_seconds"] = op_time
            row["cheb_degree_exp"] = de
            row["cheb_degree_phi1"] = dp
            row["lambda_max_bound"] = op.lambda_max
        else:
            row["operator_build_seconds"] = 0.0

        t0 = time.perf_counter()
        model, kind = make_model(name, device)
        if name == HERMES_METHOD:
            model.set_local_geometry(local_hermes_geom)
            row["hermes_block_checkpointing"] = True
        model.eval()
        row["model_init_seconds"] = time.perf_counter() - t0
        total, trainable, param_mb = parameter_stats(model)
        row.update(parameters_total=total, parameters_trainable=trainable,
                   parameter_storage_mb=param_mb)

        g = torch.Generator(device="cpu").manual_seed(SEED + height * 7 + spin)
        x_cpu = torch.randn((1, IN_CHANNELS, height, width), generator=g, dtype=torch.float32)
        x = x_cpu.to(device, non_blocking=False)
        dt = torch.full((1,), DT, dtype=torch.float32, device=device)
        if device.type == "cuda":
            sync(device)
            baseline = torch.cuda.memory_allocated(device)
            row["baseline_allocated_mb"] = mb(baseline)
        else:
            baseline = 0

        def forward():
            return predict(name, model, kind, x, op_bundle, dt)

        try:
            model.eval()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode():
                inf_med, inf_mean = measure(forward, device, warmup, repeats)
            row["inference_median_ms"] = inf_med
            row["inference_mean_ms"] = inf_mean
            row["inference_samples_per_s"] = 1000.0 / inf_med if inf_med > 0 else np.nan
            row["inference_status"] = "OK"
            if device.type == "cuda":
                sync(device)
                palloc = torch.cuda.max_memory_allocated(device)
                presid = torch.cuda.max_memory_reserved(device)
                row["infer_peak_allocated_mb"] = mb(palloc)
                row["infer_peak_reserved_mb"] = mb(presid)
                row["infer_incremental_mb"] = max(0.0, mb(palloc - baseline))
        except torch.cuda.OutOfMemoryError as exc:
            row["inference_status"] = "OOM"
            row["error"] += f"inference OOM: {exc}; "
            if device.type == "cuda": torch.cuda.empty_cache()
        except Exception as exc:
            row["inference_status"] = "ERROR"
            row["error"] += f"inference: {exc!r}; "
            if device.type == "cuda": torch.cuda.empty_cache()

        if row["inference_status"] == "OK":
            try:
                model.train()
                opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-6)

                def train_step():
                    opt.zero_grad(set_to_none=True)
                    pred = forward()
                    loss = pred.square().mean()
                    loss.backward()
                    opt.step()
                    return loss

                train_step()  # initialize Adam state before steady-state timings
                sync(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    train_base = torch.cuda.memory_allocated(device)
                else:
                    train_base = 0
                samples = []
                for _ in range(max(1, train_repeats)):
                    sync(device)
                    t1 = time.perf_counter()
                    train_step()
                    sync(device)
                    samples.append((time.perf_counter() - t1) * 1000.0)
                tmed = float(statistics.median(samples))
                row["train_step_median_ms"] = tmed
                row["train_step_mean_ms"] = float(statistics.mean(samples))
                row["train_samples_per_s"] = 1000.0 / tmed if tmed > 0 else np.nan
                row["training_status"] = "OK"
                if device.type == "cuda":
                    sync(device)
                    palloc = torch.cuda.max_memory_allocated(device)
                    presid = torch.cuda.max_memory_reserved(device)
                    row["train_peak_allocated_mb"] = mb(palloc)
                    row["train_peak_reserved_mb"] = mb(presid)
                    row["train_incremental_mb"] = max(0.0, mb(palloc - train_base))
            except torch.cuda.OutOfMemoryError as exc:
                row["training_status"] = "OOM"
                row["error"] += f"training OOM: {exc}; "
                if device.type == "cuda": torch.cuda.empty_cache()
            except Exception as exc:
                row["training_status"] = "ERROR"
                row["error"] += f"training: {exc!r}; "
                if device.type == "cuda": torch.cuda.empty_cache()
        else:
            row["training_status"] = "NOT_RUN_INFERENCE_FAILED"

        if needs_geometry and op_bundle is not None:
            op = op_bundle[0]
            q = torch.randn((1, height, width, 2), device=device)
            probe_last = torch.randn((1, 2, height, width), device=device)
            probe_residual = torch.randn_like(probe_last)
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
            op_med, _ = measure(lambda: (op.apply_A(q),), device, warmup, repeats)
            row["operator_apply_ms"] = op_med
            etd_med, _ = measure(
                lambda: sparse_cheb_etd(probe_last, probe_residual, dt, op_bundle[0],
                                        op_bundle[1], op_bundle[2]),
                device, warmup, repeats)
            row["etd_apply_ms"] = etd_med

        if row["inference_status"] == "OK" and row["training_status"] == "OK":
            row["status"] = "OK"
        elif row["inference_status"] == "OK":
            row["status"] = "INFERENCE_OK_TRAIN_" + str(row["training_status"])
        else:
            row["status"] = str(row["inference_status"])
        print(
            f"[{row['status']}][GPU{worker}][{height}x{width}] {name}: "
            f"params={total:,} infer={row['inference_median_ms']:.3f} ms "
            f"train={row['train_step_median_ms']:.3f} ms "
            f"VRAM+={row['train_incremental_mb']:.1f} MB",
            flush=True,
        )
    except torch.cuda.OutOfMemoryError as exc:
        row["status"] = "OOM_SETUP"
        row["inference_status"] = "OOM_SETUP"
        row["error"] += f"setup OOM: {exc}"
        if device.type == "cuda": torch.cuda.empty_cache()
    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] += f"{exc!r}\n{traceback.format_exc()}"
        if device.type == "cuda": torch.cuda.empty_cache()
    finally:
        del model, local_hermes_geom, op_bundle, x, opt
        if device.type == "cuda": torch.cuda.empty_cache()
    return row

def write_compact_summary(df: pd.DataFrame, out: Path) -> None:
    """Four paper rows, with only reference, 512^2 and max-tested columns."""
    if df.empty:
        return
    rows = []
    method_order = [m for m in (*CORE_METHODS, *OPTIONAL_METHODS) if m in set(df.model)]
    for method in method_order:
        part = df[df.model == method].copy()
        if part.empty:
            continue
        ok = part[part.inference_status == "OK"]
        max_res = int(ok.height.max()) if not ok.empty else 0
        p = part.sort_values(["height", "width"])
        row = {
            "model": method,
            "parameters": int(p["parameters_total"].dropna().iloc[0]) if p["parameters_total"].notna().any() else np.nan,
            "infer_128_ms": np.nan, "train_128_ms": np.nan,
            "infer_512_ms": np.nan, "train_512_ms": np.nan,
            "max_successful_inference_resolution": (f"{max_res}x{max_res}" if max_res else "none"),
            "max_successful_training_resolution": "none",
            "peak_train_vram_at_max_mb": np.nan,
        }
        for n in (128, 512):
            q = part[(part.height == n) & (part.width == n)]
            if not q.empty:
                row[f"infer_{n}_ms"] = q.inference_median_ms.iloc[0]
                row[f"train_{n}_ms"] = q.train_step_median_ms.iloc[0]
        tok = part[part.training_status == "OK"]
        if not tok.empty:
            tmax = int(tok.height.max())
            row["max_successful_training_resolution"] = f"{tmax}x{tmax}"
            q = tok[tok.height == tmax]
            if not q.empty:
                row["peak_train_vram_at_max_mb"] = q.train_peak_allocated_mb.max()
        rows.append(row)
    compact = pd.DataFrame(rows)
    compact.to_csv(out / "compact_efficiency_table.csv", index=False)
    md = ["# EvoETD high-resolution efficiency v2 (compact table)", "",
          "Latency in ms; VRAM is peak allocated memory in MiB. Batch size is 1.", "",
          compact.to_markdown(index=False), "",
          "Hermes-Adapted uses implicit four-direction stencil messages and the same Chebyshev ETD propagation as Connection-ETD-Fibre.",
          "This is a compute/memory stress test, not a predictive-accuracy experiment."]
    (out / "compact_efficiency_table.md").write_text("\n".join(md), encoding="utf-8")


def parse_resolution(value: str) -> Tuple[int, int]:
    try:
        h, w = value.lower().split("x", 1)
        h, w = int(h), int(w)
        if h < 5 or w < 5:
            raise ValueError
        return h, w
    except Exception as exc:
        raise argparse.ArgumentTypeError("resolution must be e.g. 128x128") from exc


def selected_methods(args) -> Tuple[str, ...]:
    return CORE_METHODS + (OPTIONAL_METHODS if args.include_fnoresidual else ())


def build_tasks(args) -> List[Tuple[str, int, int]]:
    return [(m, h, w) for h, w in args.resolutions for m in selected_methods(args)]


def worker_run(args, task_indices: Optional[List[int]] = None) -> pd.DataFrame:
    device = torch.device(args.device)
    tasks = build_tasks(args)
    if task_indices is None:
        task_indices = list(range(len(tasks)))
    rows = []
    for i in task_indices:
        model, h, w = tasks[i]
        row = run_task(model, h, w, args.spin, device, args.cpu_threads,
                       args.warmup, args.repeats, args.train_repeats,
                       args.worker_index, args.cheb_tolerance)
        rows.append(row)
        pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
            args.out / f"worker_{args.worker_index}_partial.csv", index=False)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def parent_two_gpu(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.out}")
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError("Two-GPU launcher called without two visible CUDA devices.")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    tasks = build_tasks(args)
    procs = []
    for wi in range(2):
        assigned = list(range(wi, len(tasks), 2))
        task_file = args.out / f"worker_{wi}_tasks.json"
        task_file.write_text(json.dumps(assigned), encoding="utf-8")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--device", "cuda:0", "--worker-index", str(wi),
               "--task-file", str(task_file), "--out", str(args.out),
               "--cpu-threads", "2", "--spin", str(args.spin),
               "--warmup", str(args.warmup), "--repeats", str(args.repeats),
               "--train-repeats", str(args.train_repeats),
               "--cheb-tolerance", str(args.cheb_tolerance),
               "--resolutions", *[f"{h}x{w}" for h, w in args.resolutions]]
        if args.include_fnoresidual:
            cmd.append("--include-fnoresidual")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(wi)
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "2"
        log = (args.out / f"gpu{wi}.log").open("w", encoding="utf-8")
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, log, wi))
        print(f"[LAUNCH] GPU {wi} {gpu_names[wi]}: {len(assigned)} model-resolution tasks", flush=True)
    failures = []
    for p, log, wi in procs:
        code = p.wait()
        log.close()
        if code != 0:
            failures.append((wi, code))
    frames = []
    for wi in range(2):
        f = args.out / f"worker_{wi}_results.csv"
        partial = args.out / f"worker_{wi}_partial.csv"
        use = f if f.exists() else partial
        if use.exists():
            frames.append(pd.read_csv(use))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RESULT_COLUMNS)
    df.to_csv(args.out / "highres_efficiency.csv", index=False)
    write_compact_summary(df, args.out)
    manifest = make_manifest(args, gpu_names, failures)
    (args.out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"GPU worker process failures: {failures}; inspect gpu*.log")
    print(f"[DONE] {args.out / 'highres_efficiency.csv'}", flush=True)


def make_manifest(args, gpu_names=None, failures=None) -> dict:
    return {
        "benchmark": "EvoETD high-resolution efficiency stress test",
        "version": "2.0-fibre-hermes-matrix-free-chebyshev",
        "timestamp_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": gpu_names or ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                                    if torch.cuda.is_available() else []),
        "methods": list(selected_methods(args)),
        "primary_method": "Connection-ETD-Fibre",
        "optional_fnoresidual_control": bool(args.include_fnoresidual),
        "resolutions": [[h, w] for h, w in args.resolutions],
        "spin": args.spin,
        "batch_size_inference": 1,
        "batch_size_training_step": 1,
        "history": HISTORY,
        "width": WIDTH,
        "fno_modes": MODES,
        "hermes_layers": HERMES_LAYERS,
        "geometry": {"curvature": CURVATURE, "x_extent": X_EXTENT, "y_extent": Y_EXTENT,
                     "connection": "spin-s horizontal Levi-Civita transport; vertical identity"},
        "connection_etd_operator": {
            "representation": "matrix-free local edge action of the area-normalized finite-volume connection stiffness",
            "spectral_bound": "Gershgorin absolute-row-sum upper bound",
            "matrix_functions": "Chebyshev approximations to exp(-nu*dt*A) and phi_1(-nu*dt*A)",
            "nu": NU, "dt": DT, "tolerance": args.cheb_tolerance,
            "autograd": "symmetric polynomial is recomputed in backward to avoid retaining recurrence activations",
            "shared_by": ["Connection-ETD-Fibre", "Hermes-Adapted + Cheb-ETD"] + (["Connection-ETD-FNOResidual"] if args.include_fnoresidual else []),
        },
        "hermes_implementation": {
            "architecture": "existing HermesAdaptedMPNN edge/node networks with unchanged local message equations",
            "edge_processing": "four directed neighbour orientations evaluated as local tensor slices; no explicit src/dst/transport edge arrays",
            "geometric_propagation": "same matrix-free Chebyshev ETD operator as Connection-ETD-Fibre",
            "training_memory": "non-reentrant activation checkpointing around each Hermes message block",
            "label": "Hermes-Adapted + Cheb-ETD; not official Hermes/ESCNN",
        },
        "timing": {"warmup": args.warmup, "inference_repeats": args.repeats,
                   "training_step_repeats": args.train_repeats, "training_batch_size": 1},
        "measurement_scope": "synthetic input model/operator forward and one optimizer step; excludes PDE data generation, full training, and accuracy",
        "limitations": [
            "Hermes-Adapted + Cheb-ETD is the project's Hermes-style adaptation using an ETD wrapper, not official Hermes/ESCNN.",
            "This benchmark tests computational scaling, not zero-shot predictive accuracy.",
            "At the largest resolutions a training step may OOM while batch-1 inference still succeeds; both statuses are recorded separately.",
        ],
        "worker_failures": failures or [],
    }


def self_test() -> dict:
    """Audit sparse operator, Chebyshev ETD, and matrix-free Hermes messages."""
    device = torch.device("cpu")
    h, w, spin = 9, 11, 1
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    geom = base.build_geometry(h, w, CURVATURE, spin, True)
    meta = grid_metadata(h, w, spin)
    op = MatrixFreeConnectionOperator(meta, device)
    rng = np.random.default_rng(20261118)
    z_np = rng.normal(size=(1, h, w, 2)).astype(np.float32)
    z = torch.from_numpy(z_np)
    out = op.apply_A(z).numpy()
    pair = z_np[0].reshape(-1, 2).reshape(-1).astype(np.float64)
    ms = geom["mass_sqrt64"]
    q = pair / ms
    rhs = geom["laplacian"] @ q
    ref = (rhs / ms).reshape(h, w, 2)
    op_err = float(np.linalg.norm(out[0] - ref) / max(np.linalg.norm(ref), 1e-12))

    op2, ce, cp, de, dp, _ = make_cheb_etd_operator(h, w, spin, device, 1e-8, meta=meta)
    last = torch.randn((1, 2, h, w), generator=torch.Generator().manual_seed(1))
    residual = torch.randn((1, 2, h, w), generator=torch.Generator().manual_seed(2))
    dt = torch.tensor([DT])
    approx = sparse_cheb_etd(last, residual, dt, op2, ce, cp)
    geom_t = base.geometry_tensors(geom, device)
    exact = base.heat_etd(last, residual, dt, geom_t, NU)
    etd_err = float((torch.linalg.norm(approx - exact) / torch.linalg.norm(exact)).item())

    a = torch.randn((1, h, w, 2), generator=torch.Generator().manual_seed(3))
    b = torch.randn((1, h, w, 2), generator=torch.Generator().manual_seed(4))
    Aa, Ab = op2.apply_A(a), op2.apply_A(b)
    sym_err = float((torch.sum(a * Ab) - torch.sum(Aa * b)).abs().item() /
                    max(torch.sum(a * Ab).abs().item(), 1.0))
    rayleigh = float((torch.sum(a * Aa) / torch.sum(a * a)).item())

    # Compare local stencil Hermes implementation against the project's explicit
    # edge-graph implementation, including parameter gradients.
    old_graph = hermes._make_graph_tensors(geom, device)
    local_geom = make_local_hermes_geometry(meta, device)
    torch.manual_seed(77)
    old_model = hermes.HermesAdaptedMPNN(history=HISTORY, width=12, layers=2).eval()
    new_model = MatrixFreeHermesAdaptedMPNN(history=HISTORY, width=12, layers=2,
                                             checkpoint_blocks=False).eval()
    new_model.load_state_dict(old_model.state_dict())
    old_model.set_graph(old_graph)
    new_model.set_local_geometry(local_geom)
    xx = torch.randn((1, IN_CHANNELS, h, w), generator=torch.Generator().manual_seed(5))
    x_old = xx.clone().requires_grad_(True)
    x_new = xx.clone().requires_grad_(True)
    h_old = old_model(x_old)
    h_new = new_model(x_new)
    hermes_out_err = float((torch.linalg.norm(h_old - h_new) /
                            torch.linalg.norm(h_old).clamp_min(1e-12)).item())
    loss_old = h_old.square().mean()
    loss_new = h_new.square().mean()
    loss_old.backward(); loss_new.backward()
    input_grad_err = float((torch.linalg.norm(x_old.grad - x_new.grad) /
                             torch.linalg.norm(x_old.grad).clamp_min(1e-12)).item())
    param_grad_errs = []
    for p_old, p_new in zip(old_model.parameters(), new_model.parameters()):
        if p_old.grad is not None and p_new.grad is not None:
            denom = torch.linalg.norm(p_old.grad).clamp_min(1e-12)
            param_grad_errs.append(float((torch.linalg.norm(p_old.grad-p_new.grad)/denom).item()))
    param_grad_err = max(param_grad_errs, default=0.0)

    # Ensure non-reentrant checkpointed Hermes blocks support parameter gradients.
    checkpoint_model = MatrixFreeHermesAdaptedMPNN(
        history=HISTORY, width=12, layers=2, checkpoint_blocks=True).train()
    checkpoint_model.load_state_dict(old_model.state_dict())
    checkpoint_model.set_local_geometry(local_geom)
    y = checkpoint_model(xx)
    y.square().mean().backward()
    checkpoint_grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                             for p in checkpoint_model.parameters() if p.requires_grad)

    torch.set_num_threads(old_threads)
    result = {
        "matrix_free_vs_original_operator_relative_error": op_err,
        "chebyshev_etd_vs_dense_eigen_relative_error": etd_err,
        "matrix_free_symmetry_relative_error": sym_err,
        "random_rayleigh_quotient": rayleigh,
        "hermes_local_stencil_output_relative_error": hermes_out_err,
        "hermes_input_gradient_relative_error": input_grad_err,
        "hermes_max_parameter_gradient_relative_error": param_grad_err,
        "hermes_checkpointed_backward_finite": bool(checkpoint_grad_ok),
        "chebyshev_degrees": {"exp": de, "phi1": dp},
    }
    if (op_err > 2e-5 or etd_err > 2e-5 or sym_err > 2e-5 or
            hermes_out_err > 2e-5 or input_grad_err > 2e-4 or
            param_grad_err > 2e-4 or not checkpoint_grad_ok or rayleigh < -1e-5):
        raise RuntimeError(f"Self-test failed: {result}")
    return {"status": "PASS", **result}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("evoetd_highres_stress_v2"))
    ap.add_argument("--resolutions", nargs="+", type=parse_resolution,
                    default=[(12, 16), (128, 128), (256, 256), (512, 512), (1024, 1024)])
    ap.add_argument("--spin", type=int, default=1, choices=(1, 2),
                    help="one spin is used to keep the high-resolution benchmark compact")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda:0, cuda:1")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--train-repeats", type=int, default=2)
    ap.add_argument("--cheb-tolerance", type=float, default=1e-6)
    ap.add_argument("--include-fnoresidual", action="store_true",
                    help="Add Connection-ETD-FNOResidual as an optional fifth diagnostic row.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--task-file", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.out = args.out.resolve()
    if args.self_test:
        print(json.dumps(self_test(), indent=2), flush=True)
        return
    if args.device == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2 and not args.worker:
            args.out.mkdir(parents=True, exist_ok=True)
            parent_two_gpu(args)
            return
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.worker:
        if args.task_file is None or not args.task_file.exists():
            raise FileNotFoundError("GPU worker requires a task file")
        args.out.mkdir(parents=True, exist_ok=True)
        indices = json.loads(args.task_file.read_text(encoding="utf-8"))
        df = worker_run(args, indices)
        df.to_csv(args.out / f"worker_{args.worker_index}_results.csv", index=False)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.out}")
    df = worker_run(args)
    df.to_csv(args.out / "highres_efficiency.csv", index=False)
    write_compact_summary(df, args.out)
    (args.out / "benchmark_manifest.json").write_text(
        json.dumps(make_manifest(args), indent=2), encoding="utf-8")
    print(f"[DONE] Results: {args.out / 'highres_efficiency.csv'}", flush=True)


if __name__ == "__main__":
    main()
