#!/usr/bin/env python3
"""Train a Hermes-style gauge-equivariant message-passing baseline on EvoETD PDEs.

This runner imports the existing ``run_connection_operator_extensions_v5.py``
experiment code and reuses its reference PDE solver, data splits, normalization,
training loop, rollout evaluator, metrics, checkpointing, and resume logic.

Important naming note
---------------------
The added model is called ``Hermes-Adapted``. It is a self-contained
parallel-transport nonlinear message-passing adaptation designed for the
spin-1 and spin-2 fibre outputs in this benchmark. It is NOT a bit-for-bit run
of the official Hermes/ESCNN implementation. The official Hermes code uses
its own gauge-equivariant convolutional feature machinery and dependencies.
This adaptation preserves the central Hermes design pattern (transport
neighbour features, compute learned edge messages, aggregate at vertices,
then apply learned node updates) while using invariant scalar coefficient
networks to support our two-component spin-s representations exactly.

Default cases
-------------
Runs the three chart-base PDEs plus their existing sensitivity, curvature,
and resolution-transfer cases. PDE1 uses the v6 chart suite; PDE2/PDE3 use
the complete v5 extension catalogues. Use --cases to select a subset.

PDE1 deliberately uses the constant-curvature chart-base protocol so all
three runs use the same 12x16 chart family. It is NOT the original PDE1
spherical-patch benchmark. Keep these result rows separate in the paper.

The default ten seeds and numerical/training settings match the v5 extension
suite. The model is trained on one-step targets and evaluated autoregressively
using the imported evaluator.

Example:
    python run_hermes_pde_comparison.py \\
      --out hermes_adapted_pde_comparison \\
      --device auto --cpu-threads 4

Optional comparison merge:
    python run_hermes_pde_comparison.py \\
      --out hermes_adapted_pde_comparison \\
      --baseline-results connection_operator_extensions_v5_additional/merged_10_seeds \\
                         pde_results_v6/extensions \\
      --device auto --cpu-threads 4

The merge writes combined metrics and paired Hermes-vs-baseline bootstrap
comparisons when compatible prior result folders are supplied.
"""
from __future__ import annotations

# Set common CPU thread limits before importing the benchmark module (and its
# NumPy/SciPy/PyTorch stack). The CLI applies the exact user-selected value too.
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import math
import sys
import time
import traceback
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    import run_connection_operator_extensions_v5 as base
except Exception as exc:  # pragma: no cover - gives a useful Kaggle-side message
    raise ImportError(
        "Could not import run_connection_operator_extensions_v5.py. Put this file "
        "in the same directory as that existing runner, then run it from that directory."
    ) from exc


RUNNER_VERSION = "hermes_adapted_mpnn_for_connection_pdes_v1_20261116"
MODEL_NAME = "Hermes-Adapted"
MODEL_KIND = "hermes_message"
DEFAULT_SEEDS = [17, 29, 41, 53, 65, 71, 83, 97, 109, 127]
PDE1_CASE_NAMES = (
    "pde1_chart_base", "pde1_nu_low", "pde1_nu_high", "pde1_mu_low",
    "pde1_mu_high", "pde1_eta_high", "pde1_curvature_negative",
    "pde1_curvature_flat", "pde1_resolution_transfer", "pde1_curvature_transfer",
)
CASE_NAMES = tuple(PDE1_CASE_NAMES) + tuple(base.case_catalogue().keys())


def _upper_gram(vectors: torch.Tensor) -> torch.Tensor:
    """Upper-triangular entries of the channel Gram matrix.

    vectors: [batch, nodes, fibre_channels, 2]
    returns: [batch, nodes, fibre_channels*(fibre_channels+1)//2]
    """
    gram = torch.einsum("bnac,bndc->bnad", vectors, vectors)
    p = vectors.shape[2]
    ii, jj = torch.triu_indices(p, p, device=vectors.device)
    return gram[:, :, ii, jj]


class HermesMessageLayer(nn.Module):
    """One parallel-transport edge-message + vertex-update block.

    The message coefficients are predicted only from gauge-invariant scalars.
    Fibre-valued messages are linear combinations of transported neighbour
    vectors, so the block commutes with independent orthogonal frame changes.
    """
    def __init__(self, width: int, fibre_channels: int):
        super().__init__()
        self.width = int(width)
        self.p = int(fibre_channels)
        gram_dim = self.p * (self.p + 1) // 2
        edge_in = 2 * self.width + self.p * self.p + 1
        self.edge_net = nn.Sequential(
            nn.Linear(edge_in, self.width), nn.GELU(),
            nn.Linear(self.width, self.width + self.p * self.p),
        )
        node_in = 2 * self.width + 2 * gram_dim
        self.node_net = nn.Sequential(
            nn.Linear(node_in, self.width), nn.GELU(),
            nn.Linear(self.width, self.width),
        )
        self.node_norm = nn.LayerNorm(self.width)
        self.fibre_mix = nn.Linear(self.width, 2 * self.p * self.p)
        # Small initial fibre updates make the residual message-passing stack
        # start close to the identity and avoid a large random vector update.
        nn.init.normal_(self.fibre_mix.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fibre_mix.bias)

    def forward(self, scalar: torch.Tensor, fibre: torch.Tensor,
                graph: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # fibre: [B,N,p,2]; P_e maps source fibre coordinates to target frame.
        src = graph["edge_src"]
        dst = graph["edge_dst"]
        P = graph["edge_P"]
        edge_attr = graph["edge_attr"]
        B, N, p, _ = fibre.shape
        E = int(src.numel())

        source_fibre = fibre.index_select(1, src)              # [B,E,p,2]
        transported = torch.einsum("eij,bepj->bepi", P, source_fibre)
        target_fibre = fibre.index_select(1, dst)

        # Cross-Gram terms compare target vectors with transported source
        # vectors in one common fibre. These are invariant under local O(2).
        cross_gram = torch.einsum("bepi,beqi->bepq", target_fibre, transported)
        edge_scalar = torch.cat([
            scalar.index_select(1, dst),
            scalar.index_select(1, src),
            cross_gram.reshape(B, E, p * p),
            edge_attr.view(1, E, 1).expand(B, -1, -1),
        ], dim=-1)
        edge_out = self.edge_net(edge_scalar)
        scalar_message = edge_out[..., :self.width]
        beta = torch.tanh(edge_out[..., self.width:].reshape(B, E, p, p)) / math.sqrt(max(p, 1))
        fibre_message = torch.einsum("beac,becd->bead", beta, transported)

        # Sum aggregation follows the message-passing construction. A fixed
        # inverse-sqrt degree factor keeps magnitudes comparable across nodes.
        agg_scalar = scalar.new_zeros((B, N, self.width))
        agg_fibre = fibre.new_zeros((B, N, p, 2))
        agg_scalar.index_add_(1, dst, scalar_message)
        agg_fibre.index_add_(1, dst, fibre_message)
        inv_sqrt_degree = graph["inv_sqrt_degree"].view(1, N, 1)
        agg_scalar = agg_scalar * inv_sqrt_degree
        agg_fibre = agg_fibre * inv_sqrt_degree.unsqueeze(-1)

        node_scalar = torch.cat([
            scalar,
            agg_scalar,
            _upper_gram(fibre),
            _upper_gram(agg_fibre),
        ], dim=-1)
        scalar_new = self.node_norm(scalar + self.node_net(node_scalar))

        mix = 0.10 * torch.tanh(self.fibre_mix(scalar_new)).reshape(B, N, 2, p, p)
        self_mix, neighbour_mix = mix[:, :, 0], mix[:, :, 1]
        fibre_new = (
            fibre
            + torch.einsum("bnaq,bnqc->bnac", self_mix, fibre)
            + torch.einsum("bnaq,bnqc->bnac", neighbour_mix, agg_fibre)
        )
        return scalar_new, fibre_new


class HermesAdaptedMPNN(nn.Module):
    """Spin-1/spin-2 parallel-transport message-passing predictor.

    It accepts the exact feature tensor built by the existing v5 data loader.
    The first 2p channels are p two-component fibre vectors; all remaining
    channels are scalar metadata. The model outputs a two-component one-step
    residual, which the shared evaluator adds to the latest state.
    """
    def __init__(self, history: int, width: int = 32, layers: int = 3):
        super().__init__()
        self.history = int(history)
        self.width = int(width)
        self.p = self.history
        self.layers_count = int(layers)
        self.gram_dim = self.p * (self.p + 1) // 2
        self.scalar_in = self.gram_dim + self.p + 2 * self.p + 6
        self.scalar_encoder = nn.Sequential(
            nn.Linear(self.scalar_in, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width),
        )
        self.blocks = nn.ModuleList([
            HermesMessageLayer(width, self.p) for _ in range(self.layers_count)
        ])
        self.readout = nn.Linear(width, self.p)
        nn.init.normal_(self.readout.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.readout.bias)
        self.register_buffer("residual_bound", torch.tensor(1.0, dtype=torch.float32))
        self._graph: Optional[Dict[str, torch.Tensor]] = None
        self._graph_key: Optional[Tuple[object, ...]] = None

    def set_residual_bound(self, value: float) -> None:
        self.residual_bound.fill_(float(value))

    def set_graph(self, graph: Dict[str, torch.Tensor]) -> None:
        # Graph tensors are created once per train/evaluation split and already
        # live on the requested device. Avoid registering them in state_dict:
        # they are deterministic geometry, not learned checkpoint parameters.
        key = (graph["edge_src"].device, graph["edge_src"].numel(),
               graph["edge_src"].data_ptr(), graph["edge_P"].data_ptr())
        if key != self._graph_key:
            self._graph = graph
            self._graph_key = key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._graph is None:
            raise RuntimeError("Hermes-Adapted graph has not been assigned by the benchmark adapter.")
        B, C, H, W = x.shape
        p = self.history
        if C != 4 * p + 6:
            raise ValueError(f"Expected {4*p+6} input channels, got {C}.")

        # Recover history vectors in [B,N,p,2] ordering.
        history = x[:, :2 * p].reshape(B, p, 2, H, W)
        fibre = history.permute(0, 3, 4, 1, 2).reshape(B, H * W, p, 2)
        gram = _upper_gram(fibre)
        mag = torch.sqrt(torch.diagonal(
            torch.einsum("bnac,bndc->bnad", fibre, fibre), dim1=-2, dim2=-1
        ).clamp_min(0.0) + 1e-12)
        logmag = torch.log1p(mag)
        extra = x[:, 2 * p:].permute(0, 2, 3, 1).reshape(B, H * W, 2 * p + 6)
        scalar = self.scalar_encoder(torch.cat([gram, logmag, extra], dim=-1))

        for block in self.blocks:
            scalar, fibre = block(scalar, fibre, self._graph)

        alpha = torch.tanh(self.readout(scalar)) / float(max(p, 1))
        residual = torch.einsum("bnp,bnpc->bnc", alpha, fibre)
        residual = residual.reshape(B, H, W, 2).permute(0, 3, 1, 2).contiguous()
        return base.bound_vector_residual(residual, self.residual_bound)


def _directed_graph_arrays(geom: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build both directed transports from the runner's undirected edge table."""
    records = geom.get("edge_records")
    if records is None:
        raise ValueError("The supplied geometry has no edge_records for message passing.")
    src: List[int] = []
    dst: List[int] = []
    transports: List[np.ndarray] = []
    weights: List[float] = []
    n = int(geom["height"]) * int(geom["width"])
    for i, j, P, weight in records:
        # Existing runner stores P in u_i - P u_j, so P transports j -> i.
        src.extend([int(j), int(i)])
        dst.extend([int(i), int(j)])
        transports.extend([np.asarray(P, dtype=np.float32), np.asarray(P, dtype=np.float32).T])
        weights.extend([float(weight), float(weight)])
    weights_arr = np.asarray(weights, dtype=np.float32)
    wmean = max(float(weights_arr.mean()), 1e-12)
    edge_attr = np.log(np.maximum(weights_arr / wmean, 1e-8)).astype(np.float32)
    return (
        np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64),
        np.asarray(transports, dtype=np.float32), edge_attr,
    )


def _make_graph_tensors(geom: Dict[str, object], device: torch.device) -> Dict[str, torch.Tensor]:
    src, dst, transports, edge_attr = _directed_graph_arrays(geom)
    n = int(geom["height"]) * int(geom["width"])
    degree = np.zeros(n, dtype=np.float32)
    np.add.at(degree, dst, 1.0)
    return {
        "edge_src": torch.as_tensor(src, device=device, dtype=torch.long),
        "edge_dst": torch.as_tensor(dst, device=device, dtype=torch.long),
        "edge_P": torch.as_tensor(transports, device=device, dtype=torch.float32),
        "edge_attr": torch.as_tensor(edge_attr.reshape(-1, 1), device=device, dtype=torch.float32),
        "inv_sqrt_degree": torch.as_tensor(1.0 / np.sqrt(np.maximum(degree, 1.0)),
                                             device=device, dtype=torch.float32),
    }


def _install_adapter() -> None:
    """Extend the imported experiment module without copying its pipeline."""
    base.FULL_MODELS = tuple(base.FULL_MODELS) + (MODEL_NAME,)
    base._MODEL_OVERRIDE = (MODEL_NAME,)

    original_build_model = base.build_model
    original_geometry_for_kind = base.geometry_for_kind
    original_geometry_tensors = base.geometry_tensors
    original_predict = base.predict

    def build_model(name: str, in_channels: int, hist: int, width: int,
                    modes: int, seed: int):
        if name == MODEL_NAME:
            base.set_seed(int(seed + 500))
            return HermesAdaptedMPNN(history=hist, width=width,
                                     layers=int(_ACTIVE_HERMES_LAYERS[0])), MODEL_KIND
        return original_build_model(name, in_channels, hist, width, modes, seed)

    def geometry_for_kind(kind: str, connection_geom, scalar_geom):
        if kind == MODEL_KIND:
            return connection_geom
        return original_geometry_for_kind(kind, connection_geom, scalar_geom)

    def geometry_tensors(geom, device):
        tensors = original_geometry_tensors(geom, device)
        if geom is not None and "edge_records" in geom:
            tensors.update(_make_graph_tensors(geom, device))
        return tensors

    def predict(model, kind, x, last, dt, geom_t, nu):
        if kind == MODEL_KIND:
            if geom_t is None:
                raise RuntimeError("Hermes-Adapted prediction requires the mesh graph.")
            model.set_graph(geom_t)
            return last + model(x)
        return original_predict(model, kind, x, last, dt, geom_t, nu)

    base.build_model = build_model
    base.geometry_for_kind = geometry_for_kind
    base.geometry_tensors = geometry_tensors
    base.predict = predict


_ACTIVE_HERMES_LAYERS = [3]


def make_cases() -> Dict[str, base.Case]:
    """Build the three matched base cases and their existing extensions.

    PDE1 uses the v6 constant-curvature chart protocol. PDE2/PDE3 use the
    complete v5 extension catalogue. Keeping the same Case type and solver
    means Hermes sees exactly the same generated trajectories as the other
    models when the numerical settings match.
    """
    p1 = dict(pde="vector_fisher", train_curvature=0.8,
              nu=0.02, mu=0.7, alpha=0.0, eta=3.0)
    cases: Dict[str, base.Case] = {
        "pde1_chart_base": base.Case(
            "pde1_chart_base", **p1, model_group="full",
            description=("PDE1 spin-1 radial-GL chart-base case; K=+0.8, "
                         "nu=0.02, mu=0.7, eta=3.0. Chart protocol, not "
                         "the original spherical-patch PDE1 benchmark.")),
        "pde1_nu_low": base.Case(
            "pde1_nu_low", **{**p1, "nu": 0.01}, model_group="sensitivity",
            description="PDE1 diffusion sensitivity: nu=0.01."),
        "pde1_nu_high": base.Case(
            "pde1_nu_high", **{**p1, "nu": 0.04}, model_group="sensitivity",
            description="PDE1 diffusion sensitivity: nu=0.04."),
        "pde1_mu_low": base.Case(
            "pde1_mu_low", **{**p1, "mu": 0.35}, model_group="sensitivity",
            description="PDE1 radial-growth sensitivity: mu=0.35."),
        "pde1_mu_high": base.Case(
            "pde1_mu_high", **{**p1, "mu": 0.95}, model_group="sensitivity",
            description="PDE1 radial-growth sensitivity: mu=0.95."),
        "pde1_eta_high": base.Case(
            "pde1_eta_high", **{**p1, "eta": 4.0}, model_group="sensitivity",
            description="PDE1 saturation sensitivity: eta=4.0."),
        "pde1_curvature_negative": base.Case(
            "pde1_curvature_negative", **{**p1, "train_curvature": -0.8},
            model_group="curvature", description="PDE1 independently trained/tested on K=-0.8."),
        "pde1_curvature_flat": base.Case(
            "pde1_curvature_flat", **{**p1, "train_curvature": 0.0},
            model_group="curvature", description="PDE1 independently trained/tested on K=0."),
        "pde1_resolution_transfer": base.Case(
            "pde1_resolution_transfer", **p1, test_height=18, test_width=24,
            model_group="transfer", description="PDE1 train at 12x16 and test at 18x24."),
        "pde1_curvature_transfer": base.Case(
            "pde1_curvature_transfer", **p1, test_curvature=-0.8,
            model_group="transfer", description="PDE1 train at K=+0.8 and test at K=-0.8."),
    }
    cases.update(base.case_catalogue())
    return cases


def _set_cpu_threads(n: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    try:
        # Keep the inter-op pool small; the intra-op and BLAS pools use the
        # requested core budget, avoiding two competing full-size thread pools.
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows setting inter-op threads only before parallel work;
        # a previously initialized runtime can make this a one-time setting.
        pass
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n)
    except Exception:
        pass


def self_test() -> Dict[str, object]:
    """Check shapes, finite outputs, and independent O(2) gauge covariance."""
    results = {}
    torch.manual_seed(1234)
    for spin in (1, 2):
        H, W, p = 5, 6, 3
        geom = base.build_geometry(H, W, 0.8, spin, True)
        graph = _make_graph_tensors(geom, torch.device("cpu"))
        model = HermesAdaptedMPNN(history=p, width=12, layers=2).eval()
        model.set_residual_bound(10.0)
        model.set_graph(graph)
        x = torch.randn(2, 4 * p + 6, H, W)
        with torch.no_grad():
            y = model(x)

        # Independent random orthogonal frame at each vertex, including
        # reflections. The edge transports are transformed accordingly.
        n = H * W
        theta = 2.0 * math.pi * torch.rand(n)
        det_sign = torch.where(torch.rand(n) > 0.5, 1.0, -1.0)
        c, s = torch.cos(theta), torch.sin(theta)
        Q = torch.zeros(n, 2, 2)
        Q[:, 0, 0] = c
        Q[:, 0, 1] = -s * det_sign
        Q[:, 1, 0] = s
        Q[:, 1, 1] = c * det_sign
        xg = x.clone()
        f = x[:, :2 * p].reshape(2, p, 2, H, W).permute(0, 3, 4, 1, 2).reshape(2, n, p, 2)
        fg = torch.einsum("nij,bnpj->bnpi", Q, f)
        xg[:, :2 * p] = fg.reshape(2, H, W, p, 2).permute(0, 3, 4, 1, 2).reshape(2, 2 * p, H, W)
        gg = {k: v.clone() for k, v in graph.items()}
        src, dst = gg["edge_src"], gg["edge_dst"]
        gg["edge_P"] = torch.einsum("eia,eab,ebj->eij", Q[dst], graph["edge_P"], Q[src].transpose(1, 2))
        with torch.no_grad():
            model.set_graph(gg)
            yg = model(xg)
        y_nodes = y.permute(0, 2, 3, 1).reshape(2, n, 2)
        expected = torch.einsum("nij,bnj->bni", Q, y_nodes)
        expected = expected.reshape(2, H, W, 2).permute(0, 3, 1, 2)
        err = float((yg - expected).abs().max())
        if not torch.isfinite(y).all() or not torch.isfinite(yg).all() or err > 3e-5:
            raise AssertionError(f"spin-{spin} O(2) covariance test failed: max_abs_error={err:.3e}")
        results[f"spin_{spin}"] = {"max_abs_gauge_error": err, "shape": list(y.shape)}
    return {"status": "PASS", "checks": results}


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Run a Hermes-style parallel-transport message-passing baseline on PDE1/PDE2/PDE3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--out", type=Path, default=Path("hermes_adapted_pde_comparison"))
    ap.add_argument("--cases", nargs="+", choices=CASE_NAMES, default=list(CASE_NAMES))
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda:0, cuda:1")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--hermes-layers", type=int, default=3)
    ap.add_argument("--model-width", type=int, default=32)
    ap.add_argument("--train-trajectories", type=int, default=16)
    ap.add_argument("--val-trajectories", type=int, default=4)
    ap.add_argument("--test-trajectories", type=int, default=6)
    ap.add_argument("--steps", type=int, default=101)
    ap.add_argument("--train-steps", type=int, default=26)
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--reference-substeps", type=int, default=4)
    ap.add_argument("--reaction-substeps", type=int, default=16)
    ap.add_argument("--reference-audit-steps", type=int, default=None)
    ap.add_argument("--reference-audit-trajectories", type=int, default=3)
    ap.add_argument("--max-reference-refinement-error", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument("--ema-decay", type=float, default=0.0)
    ap.add_argument("--batch-sampling", choices=("trajectory-balanced", "random"),
                    default="trajectory-balanced")
    ap.add_argument("--residual-bound-multiplier", type=float, default=3.0)
    ap.add_argument("--bootstrap-samples", type=int, default=3000)
    ap.add_argument("--bootstrap-horizons", nargs="+", type=int,
                    default=[1, 5, 10, 25, 50, 75])
    ap.add_argument("--bootstrap-seed", type=int, default=20261116)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reuse-data", action="store_true")
    ap.add_argument("--baseline-results", nargs="*", type=Path, default=[],
                    help="Prior result roots to merge for paired Hermes-vs-baseline comparison.")
    ap.add_argument("--merge-workers", nargs="+", type=Path, default=None,
                    help="Merge completed disjoint-seed worker directories; skip training.")
    ap.add_argument("--smoke", action="store_true", help="Tiny CPU/GPU code-path test; not scientific evidence.")
    ap.add_argument("--self-test", action="store_true", help="Run shape and O(2)-covariance checks, then exit.")
    return ap.parse_args()


def _build_args_for_base(args, device: torch.device):
    """The namespace expected by the imported v5 run_case implementation."""
    class Namespace:
        pass
    a = Namespace()
    for key in (
        "out", "seeds", "train_trajectories", "val_trajectories", "test_trajectories",
        "steps", "train_steps", "history", "dt", "reference_substeps", "reaction_substeps",
        "reference_audit_steps", "reference_audit_trajectories", "max_reference_refinement_error",
        "epochs", "batch_size", "lr", "weight_decay", "patience", "model_width",
        "grad_clip_norm", "warmup_epochs", "ema_decay", "batch_sampling",
        "residual_bound_multiplier", "bootstrap_samples", "bootstrap_horizons", "bootstrap_seed",
        "resume", "reuse_data", "cpu_threads", "device",
    ):
        setattr(a, key, getattr(args, key))
    a.modes = 6
    a.fibre_final_init_std = 0.0
    a.heat_ablation = "connection"
    a.resolved_device = str(device)
    return a


def _read_existing_metrics(root: Path, wanted_cases: Sequence[str]) -> List[pd.DataFrame]:
    if not root.exists():
        raise FileNotFoundError(f"Baseline result path does not exist: {root}")
    paths: List[Path] = []
    direct = root / "all_trajectory_metrics.csv"
    if direct.exists():
        paths.append(direct)
    else:
        # Supports both a suite root and a subfolder containing case folders.
        for case in wanted_cases:
            paths.extend(root.rglob(f"{case}/trajectory_metrics.csv"))
            candidate = root / case / "trajectory_metrics.csv"
            if candidate.exists():
                paths.append(candidate)
    unique = list(dict.fromkeys(paths))
    frames = []
    for path in unique:
        df = pd.read_csv(path)
        if "case" not in df.columns:
            continue
        df = df[df["case"].isin(wanted_cases)].copy()
        if not df.empty:
            frames.append(df)
    return frames


def _deduplicate_identical_metrics(df: pd.DataFrame) -> pd.DataFrame:
    key = ["case", "seed", "test_trajectory", "model", "horizon"]
    if not set(key).issubset(df.columns):
        raise ValueError(f"Metrics are missing pairing columns {key}.")
    duplicate_rows = df[df.duplicated(key, keep=False)]
    if not duplicate_rows.empty:
        compare_cols = [c for c in ("rel_l2", "elapsed_time", "pde") if c in df.columns]
        for _, group in duplicate_rows.groupby(key, dropna=False):
            for c in compare_cols:
                vals = group[c].to_numpy()
                if c == "rel_l2" or c == "elapsed_time":
                    if not np.allclose(vals.astype(float), float(vals[0]), rtol=1e-10, atol=1e-12):
                        raise ValueError(f"Conflicting duplicate baseline rows for key {tuple(group.iloc[0][key])}.")
                elif not (vals == vals[0]).all():
                    raise ValueError(f"Conflicting duplicate baseline rows for key {tuple(group.iloc[0][key])}.")
        df = df.drop_duplicates(key, keep="first")
    return df


def _summary_frame(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    seed = (metrics.groupby(["case", "pde", "model", "seed", "horizon"], as_index=False)
            .rel_l2.mean().rename(columns={"rel_l2": "seed_mean_rel_l2"}))
    summary = (seed.groupby(["case", "pde", "model", "horizon"])
               .seed_mean_rel_l2.agg(["mean", "std", "count"]).reset_index()
               .rename(columns={"std": "seed_std", "count": "n_seeds"}))
    return seed, summary


def _paired_bootstrap_hermes(metrics: pd.DataFrame, output: Path,
                             horizons: Sequence[int], nboot: int, seed: int) -> pd.DataFrame:
    """Two-way paired bootstrap: training seeds x shared test trajectories."""
    rng = np.random.default_rng(seed)
    rows = []
    numeric_seed_mask = pd.to_numeric(metrics["seed"], errors="coerce").notna()
    metrics = metrics.loc[numeric_seed_mask].copy()
    metrics["seed"] = pd.to_numeric(metrics["seed"], errors="raise").astype(int)
    for (case, horizon), part in metrics.groupby(["case", "horizon"]):
        if int(horizon) not in set(int(x) for x in horizons):
            continue
        if part.duplicated(["seed", "test_trajectory", "model"]).any():
            raise ValueError(f"Duplicate model/seed/test-trajectory rows in {case}, H={horizon}.")
        pivot = part.pivot(index=["seed", "test_trajectory"], columns="model", values="rel_l2")
        if MODEL_NAME not in pivot.columns:
            continue
        for comparator in [c for c in pivot.columns if c != MODEL_NAME]:
            paired = pivot[[MODEL_NAME, comparator]].dropna()
            seeds = np.asarray(sorted(part["seed"].unique()), dtype=int)
            tids = np.asarray(sorted(part["test_trajectory"].unique()), dtype=int)
            expected = pd.MultiIndex.from_product([seeds, tids], names=paired.index.names)
            paired = paired.reindex(expected)
            if len(seeds) < 2 or len(tids) < 2 or paired.isna().any().any():
                continue
            vals = paired.to_numpy().reshape(len(seeds), len(tids), 2)
            hermes_mean = float(vals[:, :, 0].mean())
            baseline_mean = float(vals[:, :, 1].mean())
            boot_diff = np.empty(nboot, dtype=np.float64)
            boot_reduction = np.empty(nboot, dtype=np.float64)
            for b in range(nboot):
                si = rng.integers(0, len(seeds), size=len(seeds))
                ti = rng.integers(0, len(tids), size=len(tids))
                sample = vals[si[:, None], ti[None, :], :]
                hm = float(sample[:, :, 0].mean())
                bm = float(sample[:, :, 1].mean())
                boot_diff[b] = hm - bm
                boot_reduction[b] = 1.0 - hm / max(bm, 1e-15)
            rows.append({
                "case": case, "horizon": int(horizon), "candidate": MODEL_NAME,
                "comparator": comparator, "candidate_mean": hermes_mean,
                "comparator_mean": baseline_mean,
                "paired_difference": hermes_mean - baseline_mean,
                "relative_error_reduction": 1.0 - hermes_mean / max(baseline_mean, 1e-15),
                "difference_ci_low": float(np.quantile(boot_diff, 0.025)),
                "difference_ci_high": float(np.quantile(boot_diff, 0.975)),
                "reduction_ci_low": float(np.quantile(boot_reduction, 0.025)),
                "reduction_ci_high": float(np.quantile(boot_reduction, 0.975)),
                "n_seeds": len(seeds), "n_test_trajectories": len(tids),
                "bootstrap_samples": nboot,
            })
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    return result



def _find_case_directory(root: Path, case: str) -> Optional[Path]:
    direct = root / case
    if (direct / "reference_trajectories.npz").exists():
        return direct
    matches = list(root.rglob(f"{case}/reference_trajectories.npz"))
    return matches[0].parent if matches else None


def _verify_paired_reference_data(args, root: Path, prior: pd.DataFrame) -> None:
    """Ensure prior and Hermes metrics use identical train/val/test trajectories."""
    cases = make_cases()
    for case_name in args.cases:
        part = prior[prior.case == case_name]
        if part.empty:
            continue
        old_dir = _find_case_directory(root, case_name)
        new_dir = Path(args.out) / case_name
        if old_dir is None or not (new_dir / "reference_trajectories.npz").exists():
            raise ValueError(
                f"Cannot verify paired data for {root}/{case_name}. The prior result "
                "must retain its case/reference_trajectories.npz file. Refusing a paired comparison."
            )
        old_manifest_path = old_dir / "case_manifest.json"
        if old_manifest_path.exists():
            old_manifest = json.loads(old_manifest_path.read_text())
            expected_case = base.asdict(cases[case_name])
            previous_case = old_manifest.get("case", {})
            scientific_case_keys = (
                "name", "pde", "train_curvature", "test_curvature", "nu", "mu",
                "eta", "alpha", "train_height", "train_width", "test_height", "test_width",
            )
            mismatches = {
                key: (previous_case.get(key), expected_case.get(key))
                for key in scientific_case_keys
                if previous_case.get(key) != expected_case.get(key)
            }
            if mismatches:
                raise ValueError(
                    f"PDE configuration differs for baseline {root}/{case_name}: {mismatches}"
                )
            expected_protocol = {
                "train_trajectories": args.train_trajectories,
                "validation_trajectories": args.val_trajectories,
                "test_trajectories": args.test_trajectories,
                "steps": args.steps,
                "train_steps": args.train_steps,
                "dt": args.dt,
                "reference_split_substeps": args.reference_substeps,
                "reference_reaction_substeps": args.reaction_substeps,
            }
            for key, value in expected_protocol.items():
                if old_manifest.get(key) != value:
                    raise ValueError(
                        f"Protocol mismatch for {root}/{case_name}: {key}="
                        f"{old_manifest.get(key)!r}, expected {value!r}."
                    )

        # When the result root retains its original run manifest, verify the
        # optimizer and training setup as well as the shared trajectory data.
        manifest_candidates = [root / "run_manifest.json", old_dir.parent / "run_manifest.json"]
        root_manifest = None
        for candidate in manifest_candidates:
            if candidate.exists():
                candidate_json = json.loads(candidate.read_text())
                if "epochs" in candidate_json or "learning_rate" in candidate_json or "lr" in candidate_json:
                    root_manifest = candidate_json
                    break
                for source_root in candidate_json.get("input_roots", []):
                    source_manifest = Path(source_root) / "run_manifest.json"
                    if source_manifest.exists():
                        source_json = json.loads(source_manifest.read_text())
                        if "epochs" in source_json or "lr" in source_json:
                            root_manifest = source_json
                            break
                if root_manifest is not None:
                    break
        if root_manifest is not None:
            expected_training = {
                "train_trajectories": args.train_trajectories,
                "val_trajectories": args.val_trajectories,
                "test_trajectories": args.test_trajectories,
                "steps": args.steps, "train_steps": args.train_steps,
                "history": args.history, "dt": args.dt,
                "reference_substeps": args.reference_substeps,
                "reaction_substeps": args.reaction_substeps,
                "epochs": args.epochs, "batch_size": args.batch_size,
                "weight_decay": args.weight_decay, "patience": args.patience,
                "model_width": args.model_width,
                "grad_clip_norm": args.grad_clip_norm,
                "warmup_epochs": args.warmup_epochs,
                "batch_sampling": args.batch_sampling,
                "ema_decay": args.ema_decay,
                "residual_bound_multiplier": args.residual_bound_multiplier,
            }
            if "lr" in root_manifest:
                expected_training["lr"] = args.lr
            elif "learning_rate" in root_manifest:
                expected_training["learning_rate"] = args.lr
            for key, value in expected_training.items():
                if key in root_manifest and root_manifest[key] != value:
                    raise ValueError(
                        f"Training configuration mismatch for {root}/{case_name}: "
                        f"{key}={root_manifest[key]!r}, expected {value!r}."
                    )
        old_meta, old_arrays = _load_npz_reference(old_dir / "reference_trajectories.npz")
        new_meta, new_arrays = _load_npz_reference(new_dir / "reference_trajectories.npz")
        # Compare arrays, not runner-version strings: PDE1's chart data were
        # generated by the v6 wrapper, while PDE2/PDE3 use the v5 extension runner.
        for split in ("train", "validation", "test"):
            if old_arrays[split].shape != new_arrays[split].shape or not np.array_equal(
                    old_arrays[split], new_arrays[split]):
                raise ValueError(
                    f"Baseline data are not paired with the Hermes run: {root}/{case_name}/{split}. "
                    "Use the matching trajectory-generation protocol before comparing models."
                )

def _merge_prior_baselines(args, hermes_metrics: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not args.baseline_results:
        return None
    frames = []
    for root in args.baseline_results:
        got = _read_existing_metrics(root, args.cases)
        if not got:
            print(f"[COMPARE][WARN] No compatible case metrics found under {root}", flush=True)
            continue
        root_metrics = pd.concat(got, ignore_index=True)
        _verify_paired_reference_data(args, Path(root), root_metrics)
        frames.extend(got)
    if not frames:
        print("[COMPARE][WARN] No prior baseline metrics were loaded; Hermes outputs remain separate.", flush=True)
        return None
    prior = pd.concat(frames, ignore_index=True)
    prior = _deduplicate_identical_metrics(prior)
    # Do not accidentally import an earlier experimental version of the same
    # label into the new paired comparison.
    prior = prior[prior["model"].astype(str) != MODEL_NAME]
    combined = pd.concat([prior, hermes_metrics], ignore_index=True, sort=False)
    combined = _deduplicate_identical_metrics(combined)

    expected_seeds = sorted(int(x) for x in args.seeds)
    for case in args.cases:
        part = combined[combined.case == case]
        if part.empty:
            continue
        for model_name, mpart in part.groupby("model"):
            got_seeds = sorted(int(float(x)) for x in mpart.seed.unique())
            if got_seeds != expected_seeds:
                print(f"[COMPARE][WARN] {case}/{model_name}: seeds={got_seeds}; expected={expected_seeds}. "
                      "This model will appear in the summary but may be excluded from paired bootstrap.", flush=True)

    comparison_dir = args.out / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(comparison_dir / "all_trajectory_metrics.csv", index=False)
    seed_frame, summary_frame = _summary_frame(combined)
    seed_frame.to_csv(comparison_dir / "seed_level_metrics.csv", index=False)
    summary_frame.to_csv(comparison_dir / "summary.csv", index=False)
    h75 = summary_frame[summary_frame.horizon == 75].copy()
    h75.to_csv(comparison_dir / "horizon_75_summary.csv", index=False)
    boot = _paired_bootstrap_hermes(
        combined, comparison_dir / "hermes_paired_bootstrap.csv",
        args.bootstrap_horizons, args.bootstrap_samples, args.bootstrap_seed,
    )
    report = {
        "model_added": MODEL_NAME,
        "cases": list(args.cases),
        "seeds_expected": expected_seeds,
        "baseline_roots": [str(p.resolve()) for p in args.baseline_results],
        "baseline_models": sorted(str(x) for x in prior.model.unique()),
        "paired_bootstrap_rows": int(len(boot)),
        "pde1_note": "PDE1 uses pde1_chart_base (constant-curvature chart), not the original spherical-patch base.",
        "interpretation": "Paired bootstrap resamples training seeds and shared test trajectories; incomplete pairs are omitted.",
    }
    (comparison_dir / "comparison_manifest.json").write_text(json.dumps(report, indent=2))
    print("\n[COMPARE] Horizon-75 summary:")
    print(h75.to_string(index=False))
    print(f"[COMPARE] paired comparisons: {len(boot)} rows -> {comparison_dir/'hermes_paired_bootstrap.csv'}", flush=True)
    return combined



def _load_npz_reference(path: Path) -> Tuple[dict, Dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"].item()))
        arrays = {name: z[name].copy() for name in ("train", "validation", "test")}
        hashes = {name: str(z[f"{name}_sha256"].item()) for name in arrays}
    for name, array in arrays.items():
        digest = __import__("hashlib").sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()
        if digest != hashes[name]:
            raise ValueError(f"Reference-data checksum failed: {path} ({name}).")
    return metadata, arrays


def _merge_worker_outputs(args) -> pd.DataFrame:
    """Merge disjoint seed partitions produced by the two-GPU launcher."""
    roots = [Path(p) for p in args.merge_workers]
    if len(roots) < 2:
        raise ValueError("--merge-workers needs at least two worker directories.")
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Worker directory does not exist: {root}")
        if not (root / "hermes_run_manifest.json").exists():
            raise FileNotFoundError(f"Missing hermes_run_manifest.json in {root}")
        if not (root / "all_trajectory_metrics.csv").exists():
            raise FileNotFoundError(f"Missing all_trajectory_metrics.csv in {root}")

    manifests = [json.loads((root / "hermes_run_manifest.json").read_text()) for root in roots]
    comparable_keys = [
        "runner_version", "model", "architecture", "official_hermes_reproduction",
        "cases", "hermes_layers", "model_width", "train_trajectories",
        "val_trajectories", "test_trajectories", "steps", "train_steps",
        "history", "dt", "epochs", "batch_size", "learning_rate", "weight_decay",
        "reference_substeps", "reaction_substeps", "pde1_protocol",
    ]
    reference = manifests[0]
    for root, manifest in zip(roots[1:], manifests[1:]):
        for key in comparable_keys:
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"Worker configuration mismatch for {key}: {root}")

    worker_seeds = []
    metric_frames = []
    profile_frames = []
    for root in roots:
        manifest = json.loads((root / "hermes_run_manifest.json").read_text())
        if manifest.get("status") != "COMPLETED":
            raise ValueError(f"Worker is not marked COMPLETED: {root} ({manifest.get('status')})")
        seeds = sorted(int(x) for x in manifest.get("seeds", []))
        if not seeds:
            raise ValueError(f"Worker has no recorded seeds: {root}")
        worker_seeds.extend(seeds)
        frame = pd.read_csv(root / "all_trajectory_metrics.csv")
        if frame.empty:
            raise ValueError(f"Worker metric file is empty: {root}")
        metric_frames.append(frame)
        profile_path = root / "runtime_memory.csv"
        if profile_path.exists() and profile_path.stat().st_size > 1:
            profile_frames.append(pd.read_csv(profile_path))

    if len(worker_seeds) != len(set(worker_seeds)):
        raise ValueError(f"Worker seed partitions overlap: {worker_seeds}")
    expected_seeds = sorted(int(x) for x in args.seeds)
    if sorted(worker_seeds) != expected_seeds:
        raise ValueError(f"Worker seeds {sorted(worker_seeds)} do not equal expected {expected_seeds}.")

    combined = _deduplicate_identical_metrics(pd.concat(metric_frames, ignore_index=True))
    if set(combined["case"].unique()) != set(args.cases):
        raise ValueError("Merged metrics do not contain exactly the requested cases.")
    for case in args.cases:
        part = combined[combined.case == case]
        if set(int(x) for x in part.seed.unique()) != set(expected_seeds):
            raise ValueError(f"Incomplete merged seed coverage for {case}.")
        for seed in expected_seeds:
            run = part[part.seed == seed]
            if set(int(x) for x in run.test_trajectory.unique()) != set(range(args.test_trajectories)):
                raise ValueError(f"Incomplete test-trajectory coverage: {case}, seed {seed}.")
            expected_horizons = list(range(1, args.steps - args.train_steps + 1))
            for tid in range(args.test_trajectories):
                tid_rows = run[run.test_trajectory == tid]
                horizons = sorted(int(x) for x in tid_rows.horizon.unique())
                if horizons != expected_horizons or len(tid_rows) != len(expected_horizons):
                    raise ValueError(
                        f"Incomplete rollout coverage: {case}, seed {seed}, trajectory {tid}."
                    )

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty merge directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / "all_trajectory_metrics.csv", index=False)
    profiles = pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame()
    profiles.to_csv(out / "runtime_memory.csv", index=False)
    seed_frame, summary_frame = _summary_frame(combined)
    seed_frame.to_csv(out / "all_seed_level_metrics.csv", index=False)
    summary_frame.to_csv(out / "all_summary.csv", index=False)
    summary_frame[summary_frame.horizon.isin([1, 5, 10, 25, 50, 75])].to_csv(
        out / "summary_all_summary.csv", index=False)

    # Rebuild per-case folders and retain checkpoints, histories, reference data,
    # and audit files. Reference arrays must be exactly the same across workers.
    for case in args.cases:
        case_out = out / case
        case_out.mkdir(parents=True, exist_ok=True)
        case_frames = []
        case_profiles = []
        case_manifests = []
        reference_meta = None
        reference_arrays = None
        reference_checks = None
        case_manifest = None
        for root in roots:
            case_dir = root / case
            if not case_dir.is_dir():
                raise FileNotFoundError(f"Missing case directory {case_dir}")
            case_metrics_path = case_dir / "trajectory_metrics.csv"
            if not case_metrics_path.exists():
                raise FileNotFoundError(f"Missing {case_metrics_path}")
            case_frames.append(pd.read_csv(case_metrics_path))
            for source_name, dest_name, bucket in (
                ("runtime_memory.csv", None, case_profiles),
                ("run_manifest.csv", None, case_manifests),
            ):
                source = case_dir / source_name
                if source.exists() and source.stat().st_size > 1:
                    bucket.append(pd.read_csv(source))
            ref_path = case_dir / "reference_trajectories.npz"
            meta, arrays = _load_npz_reference(ref_path)
            if reference_meta is None:
                reference_meta, reference_arrays = meta, arrays
                shutil.copy2(ref_path, case_out / ref_path.name)
            else:
                if meta != reference_meta:
                    raise ValueError(f"Reference trajectory metadata differs between workers for {case}.")
                for split in arrays:
                    if not np.array_equal(arrays[split], reference_arrays[split]):
                        raise ValueError(f"Reference {split} arrays differ between workers for {case}.")
            audit = case_dir / "reference_refinement.csv"
            if audit.exists():
                current = pd.read_csv(audit)
                if reference_checks is None:
                    reference_checks = current
                    current.to_csv(case_out / audit.name, index=False)
                elif not current.equals(reference_checks):
                    raise ValueError(f"Reference refinement audits differ between workers for {case}.")
            case_manifest_path = case_dir / "case_manifest.json"
            if case_manifest_path.exists():
                cmanifest = json.loads(case_manifest_path.read_text())
                # Seed lists differ by worker; all other protocol settings must match.
                cmanifest.pop("seeds", None)
                cmanifest.pop("resume_signature", None)
                if case_manifest is None:
                    case_manifest = cmanifest
                elif cmanifest != case_manifest:
                    raise ValueError(f"Case manifest differs between workers for {case}.")
            # Copy seed-specific artifacts; workers must own disjoint seeds.
            for subdir in ("checkpoints",):
                srcdir = case_dir / subdir
                if srcdir.exists():
                    dstdir = case_out / subdir
                    dstdir.mkdir(exist_ok=True)
                    for item in srcdir.iterdir():
                        target = dstdir / item.name
                        if target.exists():
                            raise FileExistsError(f"Duplicate seed artifact during merge: {target}")
                        shutil.copy2(item, target)
            for item in case_dir.glob("*_seed*_history.json"):
                target = case_out / item.name
                if target.exists():
                    raise FileExistsError(f"Duplicate history artifact during merge: {target}")
                shutil.copy2(item, target)
        case_metric = _deduplicate_identical_metrics(pd.concat(case_frames, ignore_index=True))
        case_metric.to_csv(case_out / "trajectory_metrics.csv", index=False)
        case_seed_frame, case_summary_frame = _summary_frame(case_metric)
        case_seed_frame.to_csv(case_out / "seed_level_metrics.csv", index=False)
        case_summary_frame.to_csv(case_out / "summary.csv", index=False)
        if case_profiles:
            pd.concat(case_profiles, ignore_index=True).to_csv(case_out / "runtime_memory.csv", index=False)
        if case_manifests:
            merged_case_manifest = pd.concat(case_manifests, ignore_index=True)
            if "checkpoint" in merged_case_manifest.columns:
                merged_case_manifest["checkpoint"] = merged_case_manifest["checkpoint"].map(
                    lambda value: str(case_out / "checkpoints" / Path(str(value)).name)
                    if pd.notna(value) else value)
            merged_case_manifest.to_csv(case_out / "run_manifest.csv", index=False)
        if case_manifest is not None:
            case_manifest["seeds"] = expected_seeds
            case_manifest["merged_for_reporting_only"] = True
            case_manifest.pop("resume_signature", None)
            (case_out / "case_manifest.json").write_text(json.dumps(case_manifest, indent=2))
        # Merge failure reports too; a completed successful run should have none.
        fails = []
        for root in roots:
            fp = root / case / "failures.csv"
            if fp.exists() and fp.stat().st_size > 1:
                fails.append(pd.read_csv(fp))
        if fails:
            pd.concat(fails, ignore_index=True).to_csv(case_out / "failures.csv", index=False)

    merged_manifest = dict(reference)
    merged_manifest.update({
        "seeds": expected_seeds,
        "worker_directories": [str(p.resolve()) for p in roots],
        "device": "multi-GPU worker merge",
        "cpu_threads_per_worker": [int(m.get("cpu_threads", args.cpu_threads)) for m in manifests],
        "elapsed_seconds_sum_workers": float(sum(m.get("elapsed_seconds", 0.0) for m in manifests)),
        "status": "MERGED_COMPLETE",
    })
    (out / "hermes_run_manifest.json").write_text(json.dumps(merged_manifest, indent=2))
    print(f"[MERGE] merged {len(expected_seeds)} seeds across {len(args.cases)} cases -> {out}", flush=True)
    if args.baseline_results:
        _merge_prior_baselines(args, combined)
    print("[MERGE] Horizon-75 Hermes summary:")
    print(summary_frame[summary_frame.horizon == 75].to_string(index=False))
    return combined


def _metric_coverage_errors(metrics: pd.DataFrame, args) -> List[str]:
    errors: List[str] = []
    expected_horizons = list(range(1, int(args.steps) - int(args.train_steps) + 1))
    for case in args.cases:
        part = metrics[metrics.case == case] if not metrics.empty else pd.DataFrame()
        for seed in args.seeds:
            run = (part[pd.to_numeric(part.seed, errors="coerce") == int(seed)]
                   if not part.empty else pd.DataFrame())
            if run.empty:
                errors.append(f"{case}: seed {seed} has no metrics")
                continue
            if set(int(x) for x in run.test_trajectory.unique()) != set(range(args.test_trajectories)):
                errors.append(f"{case}: seed {seed} has incomplete test-trajectory coverage")
                continue
            for tid in range(args.test_trajectories):
                rows = run[run.test_trajectory == tid]
                horizons = sorted(int(x) for x in rows.horizon.unique())
                if horizons != expected_horizons or len(rows) != len(expected_horizons):
                    errors.append(f"{case}: seed {seed}, trajectory {tid} has incomplete rollout coverage")
                    break
    return errors

def _validate_args(args) -> None:
    if args.cpu_threads < 1 or args.hermes_layers < 1 or args.model_width < 1:
        raise ValueError("CPU threads, Hermes layers, and model width must be positive.")
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        raise ValueError("Use at least two unique training seeds.")
    if args.history < 1 or args.train_steps <= args.history or args.steps <= args.train_steps + 1:
        raise ValueError("Require history >= 1 and steps > train_steps > history.")
    if args.batch_sampling == "trajectory-balanced" and args.batch_size != args.train_trajectories:
        raise ValueError("trajectory-balanced sampling requires --batch-size == --train-trajectories.")
    if min(args.train_trajectories, args.val_trajectories, args.test_trajectories) < 2:
        raise ValueError("Use at least two trajectories in each split.")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch size, and patience must be positive.")
    if args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples.")
    if args.dt <= 0 or args.reference_substeps < 1 or args.reaction_substeps < 1:
        raise ValueError("Invalid time-integration settings.")



def _annotate_case_manifest(case_dir: Path) -> None:
    path = case_dir / "case_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    contract = manifest.setdefault("covariance_contract", {})
    contract[MODEL_NAME] = (
        "parallel-transport nonlinear message passing with invariant scalar edge/node updates "
        "and locally O(2)-equivariant two-component fibre updates; Hermes-style adaptation, "
        "not the official ESCNN implementation"
    )
    manifest["comparison_model"] = MODEL_NAME
    manifest["official_hermes_reproduction"] = False
    path.write_text(json.dumps(manifest, indent=2))

def main() -> None:
    global _ACTIVE_HERMES_LAYERS
    args = _parse_args()
    _ACTIVE_HERMES_LAYERS[0] = int(args.hermes_layers)
    _validate_args(args)
    _set_cpu_threads(args.cpu_threads)

    if args.merge_workers:
        _merge_worker_outputs(args)
        return

    if args.self_test:
        print(json.dumps(self_test(), indent=2), flush=True)
        return

    if args.smoke:
        args.cases = ["pde1_chart_base", "pde2_base", "pde3_base"]
        args.seeds = [17]
        args.train_trajectories = 3
        args.val_trajectories = 2
        args.test_trajectories = 2
        args.steps = 14
        args.train_steps = 6
        args.history = 3
        args.reference_audit_steps = 14
        args.reference_audit_trajectories = 2
        args.epochs = 2
        args.batch_size = 3
        args.model_width = 12
        args.hermes_layers = 2
        _ACTIVE_HERMES_LAYERS[0] = 2
        args.bootstrap_samples = 200
        args.bootstrap_horizons = [1, 3, 5]
        args.resume = False
        args.reuse_data = False

    if args.reference_audit_steps is None:
        args.reference_audit_steps = int(args.steps)
    if args.resume:
        args.reuse_data = True

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable.")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible.")

    base.RUNNER_VERSION = f"{RUNNER_VERSION}_L{args.hermes_layers}_W{args.model_width}"
    _install_adapter()
    base._BOOTSTRAP_HORIZONS = [int(x) for x in args.bootstrap_horizons]
    base.HEAT_ABLATION_MODE = "connection"
    base.FIBRE_FINAL_INIT_STD = 0.0
    base._MODEL_OVERRIDE = (MODEL_NAME,)

    cases = make_cases()
    args.out.mkdir(parents=True, exist_ok=True)
    run_args = _build_args_for_base(args, device)
    run_start = time.monotonic()
    all_rows = []
    all_profiles = []
    for index, case_name in enumerate(args.cases, start=1):
        case = cases[case_name]
        print(f"\n[HERMES-ADAPTED][{index}/{len(args.cases)}] {case_name}: {case.description}", flush=True)
        rows, profiles = base.run_case(case, run_args, args.out, device)
        _annotate_case_manifest(args.out / case.name)
        if not rows.empty:
            all_rows.append(rows)
        if not profiles.empty:
            all_profiles.append(profiles)

    metrics = (pd.concat(all_rows, ignore_index=True) if all_rows
               else pd.DataFrame(columns=base.METRIC_COLUMNS))
    profiles = (pd.concat(all_profiles, ignore_index=True) if all_profiles
                else pd.DataFrame(columns=base.PROFILE_COLUMNS))
    metrics.to_csv(args.out / "all_trajectory_metrics.csv", index=False)
    profiles.to_csv(args.out / "runtime_memory.csv", index=False)
    seed_frame, summary_frame = _summary_frame(metrics)
    seed_frame.to_csv(args.out / "all_seed_level_metrics.csv", index=False)
    summary_frame.to_csv(args.out / "all_summary.csv", index=False)
    compact_summary = summary_frame[summary_frame.horizon.isin([1, 5, 10, 25, 50, 75])].copy()
    compact_summary.to_csv(args.out / "summary_all_summary.csv", index=False)
    coverage_errors = _metric_coverage_errors(metrics, args)
    if coverage_errors:
        (args.out / "incomplete_runs.txt").write_text("\n".join(coverage_errors) + "\n")

    # Parameter count is recorded by the imported runner in runtime_memory.csv.
    manifest = {
        "runner_version": base.RUNNER_VERSION,
        "model": MODEL_NAME,
        "architecture": "parallel-transport edge message passing + invariant scalar node updates",
        "official_hermes_reproduction": False,
        "adaptation_note": (
            "Self-contained Hermes-style adaptation for two-component spin-1/spin-2 fibre outputs. "
            "It is not a bit-for-bit reproduction of the official ESCNN Hermes implementation."
        ),
        "equivariance": "exact local O(2) covariance for the implemented two-component representations; checked by self-test",
        "cases": list(args.cases),
        "seeds": [int(x) for x in args.seeds],
        "device": str(device),
        "cpu_threads": int(args.cpu_threads),
        "hermes_layers": int(args.hermes_layers),
        "model_width": int(args.model_width),
        "train_trajectories": int(args.train_trajectories),
        "val_trajectories": int(args.val_trajectories),
        "test_trajectories": int(args.test_trajectories),
        "steps": int(args.steps), "train_steps": int(args.train_steps),
        "history": int(args.history), "dt": float(args.dt),
        "epochs": int(args.epochs), "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr), "weight_decay": float(args.weight_decay),
        "patience": int(args.patience), "grad_clip_norm": float(args.grad_clip_norm),
        "warmup_epochs": int(args.warmup_epochs), "ema_decay": float(args.ema_decay),
        "batch_sampling": args.batch_sampling,
        "residual_bound_multiplier": float(args.residual_bound_multiplier),
        "reference_substeps": int(args.reference_substeps),
        "reaction_substeps": int(args.reaction_substeps),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_horizons": [int(x) for x in args.bootstrap_horizons],
        "elapsed_seconds": float(time.monotonic() - run_start),
        "pde1_protocol": "pde1_chart_base; constant-curvature K=+0.8 chart, not spherical-patch PDE1 base",
        "status": "COMPLETED" if not coverage_errors else "INCOMPLETE",
        "coverage_errors": coverage_errors,
    }
    (args.out / "hermes_run_manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.baseline_results and not coverage_errors:
        _merge_prior_baselines(args, metrics)
    elif args.baseline_results and coverage_errors:
        print("[COMPARE][SKIP] Hermes metrics are incomplete; no paired comparison written.", flush=True)

    print("\n[HERMES-ADAPTED] Horizon-75 summary:")
    if not summary_frame.empty:
        print(summary_frame[summary_frame.horizon == 75].to_string(index=False))
    print(f"\nRun status: {'INCOMPLETE' if coverage_errors else 'COMPLETED'}", flush=True)
    print(f"Elapsed: {(time.monotonic()-run_start)/60:.1f} min. Output: {args.out.resolve()}", flush=True)
    if coverage_errors:
        raise RuntimeError(f"Incomplete Hermes benchmark: {len(coverage_errors)} case/seed rollout records are missing. See {args.out/'incomplete_runs.txt'}.")


if __name__ == "__main__":
    main()
