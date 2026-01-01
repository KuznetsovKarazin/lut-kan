"""
Dense forward for B-spline activation layer (NumPy).

This is the fair float baseline (vectorized stack = NumPy) for comparing against
LUT NumPy.
"""

from __future__ import annotations

import numpy as np

from src.kernels.bspline_contract import PackedBSplineDenseLayer
from src.kernels.bspline_math import apply_base, coef2curve_numpy


def forward_bspline_dense_numpy(x: np.ndarray, packed: PackedBSplineDenseLayer) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != packed.in_dim:
        raise ValueError(f"Expected x shape [N,{packed.in_dim}], got {x.shape}")

    N = int(x.shape[0])
    y = np.zeros((N, packed.out_dim), dtype=np.float32)

    base_x = apply_base(x, packed.base_kind)

    for i in range(packed.in_dim):
        xi = x[:, i]
        base_i = base_x[:, i]
        knots_i = packed.knots_aug[i, :]
        grid_i = packed.grid[i, :]
        for j in range(packed.out_dim):
            m = float(packed.m[i, j])
            if m == 0.0:
                continue
            sb = float(packed.sb[i, j])
            ss = float(packed.ss[i, j])

            spl = coef2curve_numpy(
                xi,
                grid=grid_i,
                coef=packed.coef[i, j, :],
                degree=packed.degree,
                boundary_mode=packed.boundary_mode,
                knots_aug=knots_i,
            )
            y[:, j] += m * (sb * base_i + ss * spl)

    return y
