"""
B-spline dense-layer contract for *fair* float baselines.

This module provides a packed representation of a single dense KAN activation
layer as per-edge B-splines plus per-edge scaling (PyKAN semantics):

    phi_ij(x) = m_ij * ( sb_ij * base(x) + ss_ij * spline_ij(x) )
    y_j(x)    = sum_i phi_ij(x_i)

It mirrors the LUT dense backends interface so we can benchmark fairly:
  - B-spline evaluation (NumPy/Numba) vs LUT evaluation (NumPy/Numba)
under the same vectorization/JIT conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

BoundaryMode = Literal["half_open", "closed"]


@dataclass(frozen=True)
class PackedBSplineDenseLayer:
    """
    Packed representation of one dense activation layer.

    Shapes:
      - grid:      [in_dim, G]            interior grid points
      - knots_aug: [in_dim, T]            augmented knot vector (used for eval)
      - coef:      [in_dim, out_dim, C]   B-spline coefficients
      - sb, ss, m: [in_dim, out_dim]      scalars

    Contract:
      - T == C + degree + 1
      - G == T - 2*degree
    """
    in_dim: int
    out_dim: int
    degree: int

    boundary_mode: BoundaryMode

    grid: np.ndarray
    knots_aug: np.ndarray
    coef: np.ndarray

    sb: np.ndarray
    ss: np.ndarray
    m: np.ndarray

    base_kind: str = "none"  # "none" | "silu" (must match src/models/kan_wrapper.py)

    @property
    def x_min(self) -> float:
        return float(self.grid[0, 0])

    @property
    def x_max(self) -> float:
        return float(self.grid[0, -1])


def pack_bspline_dense_layer_from_pykankan_adapter(
    adapter: Any,
    *,
    boundary_mode: BoundaryMode = "half_open",
) -> PackedBSplineDenseLayer:
    """
    Extract knots/coeffs/scales from PyKANSingleLayerAdapter.

    PyKAN stores an *augmented* knot vector in layer.grid, typically of length:
        T = grid + 2*k + 1

    We expose:
      - knots_aug := layer.grid
      - grid      := knots_aug[k:-k]
    which matches the "interior domain" you use for LUT building / OOB checks.
    """
    layer = adapter.model.act_fun[int(adapter.layer_idx)]

    grid = getattr(layer, "grid")
    coef = getattr(layer, "coef")
    scale_base = getattr(layer, "scale_base")
    scale_sp = getattr(layer, "scale_sp")
    mask = getattr(layer, "mask")
    base_fun = getattr(layer, "base_fun", None)

    k = int(getattr(layer, "k", 3))

    def _to_np(a) -> np.ndarray:
        try:
            import torch  # type: ignore
            if isinstance(a, torch.Tensor):
                return a.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(a)

    grid_np = _to_np(grid).astype(np.float32, copy=False)
    coef_np = _to_np(coef).astype(np.float32, copy=False)
    sb_np = _to_np(scale_base).astype(np.float32, copy=False)
    ss_np = _to_np(scale_sp).astype(np.float32, copy=False)
    m_np = _to_np(mask).astype(np.float32, copy=False)

    # PyKAN shapes:
    #   grid: [in_dim, 1, T] or [in_dim, T]
    #   coef: [in_dim, out_dim, 1, C] or [in_dim, out_dim, C]
    if grid_np.ndim == 3:
        grid_np = grid_np[:, 0, :]
    if coef_np.ndim == 4:
        coef_np = coef_np[:, :, 0, :]

    in_dim = int(grid_np.shape[0])
    out_dim = int(coef_np.shape[1])

    # Enforce shared knots across inputs (same assumption as adapter.extract_edges()).
    ref = grid_np[0]
    for i in range(1, in_dim):
        if not np.allclose(grid_np[i], ref, rtol=0.0, atol=1e-7):
            raise ValueError(
                "PyKAN grid is not shared across inputs; packed baseline assumes shared knots."
            )

    knots_aug = grid_np

    if knots_aug.shape[1] < 2 * k + 2:
        raise ValueError(f"Unexpected knot vector length {knots_aug.shape[1]} for degree k={k}.")

    grid_interior = knots_aug[:, k:-k]

    knot_len = int(knots_aug.shape[1])
    coef_len = int(coef_np.shape[2])
    expected_knot_len = coef_len + k + 1
    if knot_len != expected_knot_len:
        raise ValueError(
            f"Inconsistent shapes: knot_len={knot_len}, coef_len={coef_len}, degree={k} "
            f"(expected {expected_knot_len})."
        )

    base_kind = "none"
    try:
        import torch.nn as nn  # type: ignore
        if isinstance(base_fun, nn.SiLU):
            base_kind = "silu"
    except Exception:
        pass
    name = type(base_fun).__name__.lower() if base_fun is not None else ""
    if "silu" in name:
        base_kind = "silu"
    elif "identity" in name:
        base_kind = "none"

    if boundary_mode not in ("half_open", "closed"):
        boundary_mode = "half_open"

    return PackedBSplineDenseLayer(
        in_dim=in_dim,
        out_dim=out_dim,
        degree=k,
        boundary_mode=boundary_mode,
        grid=grid_interior.astype(np.float32, copy=False),
        knots_aug=knots_aug.astype(np.float32, copy=False),
        coef=coef_np.astype(np.float32, copy=False),
        sb=sb_np.astype(np.float32, copy=False),
        ss=ss_np.astype(np.float32, copy=False),
        m=m_np.astype(np.float32, copy=False),
        base_kind=base_kind,
    )
