# src/kernels/bspline_backend_dense_numba.py
from __future__ import annotations

import numpy as np

from src.kernels.bspline_contract import PackedBSplineDenseLayer
from src.kernels.bspline_math import numba_available

if numba_available():
    import numba as nb
    from src.kernels.bspline_math import coef2curve_numba_1d

    @nb.njit(cache=True)
    def _silu(x: float) -> float:
        return x / (1.0 + np.exp(-x))

    @nb.njit(cache=True)
    def _forward_bspline_dense_numba(
        x: np.ndarray,
        coef: np.ndarray,
        knots: np.ndarray,
        sb: np.ndarray,
        ss: np.ndarray,
        m: np.ndarray,
        degree: int,
        closed: bool,
        base_is_silu: bool,
    ) -> np.ndarray:
        N = x.shape[0]
        in_dim = x.shape[1]
        out_dim = coef.shape[1]
        y = np.zeros((N, out_dim), dtype=np.float32)

        tmp = np.empty(N, dtype=np.float32)

        for i in range(in_dim):
            for j in range(out_dim):
                if m[i, j] == 0.0:
                    continue

                tmp[:] = coef2curve_numba_1d(x[:, i], coef[i, j, :], knots[i, :], degree, closed)

                for n in range(N):
                    xv = float(x[n, i])
                    base_v = _silu(xv) if base_is_silu else np.float32(xv)
                    y[n, j] += m[i, j] * (sb[i, j] * base_v + ss[i, j] * tmp[n])

        return y


def forward_bspline_dense_numba(x: np.ndarray, packed: PackedBSplineDenseLayer) -> np.ndarray:
    if not numba_available():
        raise RuntimeError("numba not available")

    x = np.asarray(x, dtype=np.float32)
    closed = bool(packed.boundary_mode == "closed")
    base_is_silu = bool((packed.base_kind or "none").lower().strip() == "silu")

    return _forward_bspline_dense_numba(
        x,
        np.asarray(packed.coef, dtype=np.float32),
        np.asarray(packed.knots_aug, dtype=np.float32),
        np.asarray(packed.sb, dtype=np.float32),
        np.asarray(packed.ss, dtype=np.float32),
        np.asarray(packed.m, dtype=np.float32),
        int(packed.degree),
        bool(closed),
        bool(base_is_silu),
    )

def warmup_bspline_numba(packed):
    import numpy as np
    x = np.zeros((1, packed.in_dim), dtype=np.float32)
    forward_bspline_dense_numba(x, packed)
