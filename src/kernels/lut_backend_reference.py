# src/kernels/lut_backend_reference.py
from __future__ import annotations

import numpy as np

from src.kernels.lut_contract import PackedLUT
from src.kernels.lut_math import (
    base_fn,
    clip_for_indexing,
    dequant,
    in_domain_mask,
    lut_interp_indices,
    segment_params_nonuniform,
    segment_params_uniform,
)


def _edge_lut_value(x: np.ndarray, packed: PackedLUT, i: int, j: int) -> np.ndarray:
    """
    Returns the LUT component value for edge (i,j), after OOB policy is applied,
    but before reconstruction (base + scaling).
    """
    x = np.asarray(x, dtype=np.float32)
    x_raw = x

    x_clip = clip_for_indexing(x_raw, packed.x_min, packed.x_max, packed.boundary_mode)

    if packed.uniform_dx is not None:
        k, u = segment_params_uniform(x_clip, packed.x_min, packed.uniform_dx, packed.K)
    else:
        k, u = segment_params_nonuniform(x_clip, packed.knots)

    r0, r1, w = lut_interp_indices(u, packed.L)

    # Gather q for this edge:
    q_flat = packed.q_flat[i, j]  # [K*L]
    base_idx0 = k * packed.L + r0
    base_idx1 = k * packed.L + r1
    q0 = q_flat[base_idx0]
    q1 = q_flat[base_idx1]

    y0 = dequant(q0, packed.y_min[i, j, :][k], packed.scale[i, j, :][k])
    y1 = dequant(q1, packed.y_min[i, j, :][k], packed.scale[i, j, :][k])

    if packed.interp == "nearest":
        lut_val = np.where(w < 0.5, y0, y1)
    else:
        lut_val = (1.0 - w) * y0 + w * y1

    if packed.oob_behavior == "zero":
        lut_val = lut_val * in_domain_mask(x_raw, packed.x_min, packed.x_max, packed.boundary_mode).astype(np.float32)

    return lut_val.astype(np.float32, copy=False)


def forward_reference(x: np.ndarray, packed: PackedLUT) -> np.ndarray:
    """
    Reference implementation: correct, simple, not optimized.

    x: [N, in_dim] float32
    returns y: [N, out_dim] float32
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected x shape [N,in_dim], got {x.shape}")

    N, in_dim = x.shape
    out_dim = packed.q_flat.shape[1]
    if in_dim != packed.q_flat.shape[0]:
        raise ValueError(f"in_dim mismatch: x has {in_dim}, packed has {packed.q_flat.shape[0]}")

    y = np.zeros((N, out_dim), dtype=np.float32)

    for i in range(in_dim):
        xi = x[:, i]
        if packed.value_representation == "phi":
            for j in range(out_dim):
                y[:, j] += _edge_lut_value(xi, packed, i, j)
        else:
            base = base_fn(xi, packed.base_kind)
            for j in range(out_dim):
                lut_val = _edge_lut_value(xi, packed, i, j)
                y[:, j] += packed.coef_out[i, j] * (packed.coef_base[i, j] * base + packed.coef_lut[i, j] * lut_val)

    return y
