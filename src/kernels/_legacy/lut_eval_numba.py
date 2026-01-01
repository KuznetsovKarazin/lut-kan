# src/kernels/lut_eval_numba.py
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import numba as nb
    _NUMBA_AVAILABLE = True
except Exception:
    nb = None
    _NUMBA_AVAILABLE = False


def numba_available() -> bool:
    return _NUMBA_AVAILABLE


if _NUMBA_AVAILABLE:

    @nb.njit(cache=True, fastmath=False)
    def _silu(x: float) -> float:
        return x / (1.0 + np.exp(-x))

    @nb.njit(cache=True, fastmath=False)
    def _binary_search_segment(knots: np.ndarray, x: float) -> int:
        lo = 0
        hi = knots.shape[0] - 1
        if x <= knots[0]:
            return 0
        if x >= knots[hi]:
            return hi - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if knots[mid] <= x:
                lo = mid
            else:
                hi = mid
        return lo

    @nb.njit(cache=True, fastmath=False)
    def _segment_params_uniform(x: float, x_min: float, dx: float, K: int) -> (int, float):
        k = int(np.floor((x - x_min) / dx))
        if k < 0:
            k = 0
        elif k > K - 1:
            k = K - 1
        t0 = x_min + k * dx
        u = (x - t0) / dx
        return k, u

    @nb.njit(cache=True, fastmath=False)
    def _segment_params_general(x: float, knots: np.ndarray, K: int) -> (int, float):
        k = _binary_search_segment(knots, x)
        if k < 0:
            k = 0
        elif k > K - 1:
            k = K - 1
        t0 = knots[k]
        t1 = knots[k + 1]
        denom = t1 - t0
        if denom == 0.0:
            denom = 1.0
        u = (x - t0) / denom
        return k, u

    @nb.njit(cache=True, fastmath=False)
    def _index_r_and_w(u: float, L: int) -> (int, float):
        p = u * (L - 1)
        r0 = int(np.floor(p))
        if r0 < 0:
            r0 = 0
        elif r0 > L - 2:
            r0 = L - 2
        w = p - r0
        return r0, w

    @nb.njit(parallel=True, cache=True, fastmath=False)
    def _forward_numba_uniform(
        x: np.ndarray,              # [N, in_dim] float32
        q_flat: np.ndarray,         # [in_dim, out_dim, K*L] uint8/int8
        scale: np.ndarray,          # [in_dim, out_dim, K] float32
        y_min: np.ndarray,          # [in_dim, out_dim, K] float32
        edge_sb: np.ndarray,        # [in_dim, out_dim] float32
        edge_ss: np.ndarray,        # [in_dim, out_dim] float32
        edge_m: np.ndarray,         # [in_dim, out_dim] float32
        knots: np.ndarray,          # [K+1] float32
        L: int,
        K: int,
        interp_mode: int,           # 0=linear, 1=nearest
        oob_clip: int,              # 0/1 (clip for indexing)
        zero_spline: int,           # 0/1 (if 1: spline=0 outside [x_min,x_max))
        x_min: float,
        x_max: float,
        dx: float,
    ) -> np.ndarray:
        N = x.shape[0]
        in_dim = x.shape[1]
        out_dim = q_flat.shape[1]
        y = np.zeros((N, out_dim), dtype=np.float32)

        for n in nb.prange(N):
            for j in range(out_dim):
                acc = 0.0
                for i in range(in_dim):
                    x_raw = x[n, i]
                    xv = x_raw

                    if oob_clip == 1:
                        if xv < x_min:
                            xv = x_min
                        elif xv > x_max:
                            xv = x_max

                    k, u = _segment_params_uniform(xv, x_min, dx, K)
                    r0, w = _index_r_and_w(u, L)
                    idx0 = k * L + r0
                    idx1 = idx0 + 1

                    q0 = float(q_flat[i, j, idx0])
                    y0 = y_min[i, j, k] + scale[i, j, k] * q0

                    if interp_mode == 1:
                        lut_val = y0
                    else:
                        q1 = float(q_flat[i, j, idx1])
                        y1 = y_min[i, j, k] + scale[i, j, k] * q1
                        lut_val = (1.0 - w) * y0 + w * y1

                    if zero_spline == 1:
                        # pykan semantics: spline(x)=0 for x outside [x_min, x_max)
                        if not (x_raw >= x_min and x_raw < x_max):
                            lut_val = 0.0

                    base = _silu(x_raw)  # base uses raw x
                    acc += edge_m[i, j] * (edge_sb[i, j] * base + edge_ss[i, j] * lut_val)

                y[n, j] = acc

        return y

    @nb.njit(parallel=True, cache=True, fastmath=False)
    def _forward_numba_general(
        x: np.ndarray,
        q_flat: np.ndarray,
        scale: np.ndarray,
        y_min: np.ndarray,
        edge_sb: np.ndarray,
        edge_ss: np.ndarray,
        edge_m: np.ndarray,
        knots: np.ndarray,
        L: int,
        K: int,
        interp_mode: int,
        oob_clip: int,
        zero_spline: int,
        x_min: float,
        x_max: float,
    ) -> np.ndarray:
        N = x.shape[0]
        in_dim = x.shape[1]
        out_dim = q_flat.shape[1]
        y = np.zeros((N, out_dim), dtype=np.float32)

        for n in nb.prange(N):
            for j in range(out_dim):
                acc = 0.0
                for i in range(in_dim):
                    x_raw = x[n, i]
                    xv = x_raw

                    if oob_clip == 1:
                        if xv < x_min:
                            xv = x_min
                        elif xv > x_max:
                            xv = x_max

                    k, u = _segment_params_general(xv, knots, K)
                    r0, w = _index_r_and_w(u, L)
                    idx0 = k * L + r0
                    idx1 = idx0 + 1

                    q0 = float(q_flat[i, j, idx0])
                    y0 = y_min[i, j, k] + scale[i, j, k] * q0

                    if interp_mode == 1:
                        lut_val = y0
                    else:
                        q1 = float(q_flat[i, j, idx1])
                        y1 = y_min[i, j, k] + scale[i, j, k] * q1
                        lut_val = (1.0 - w) * y0 + w * y1

                    if zero_spline == 1:
                        if not (x_raw >= x_min and x_raw < x_max):
                            lut_val = 0.0

                    base = _silu(x_raw)
                    acc += edge_m[i, j] * (edge_sb[i, j] * base + edge_ss[i, j] * lut_val)

                y[n, j] = acc

        return y


def forward_lut_layer_numba(x: np.ndarray, packed) -> np.ndarray:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not available. Install it with: pip install numba")

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected x shape [N,in_dim], got {x.shape}")
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)

    q_flat = np.ascontiguousarray(packed.q_flat)
    scale = np.ascontiguousarray(packed.scale, dtype=np.float32)
    y_min = np.ascontiguousarray(packed.y_min, dtype=np.float32)
    knots = np.ascontiguousarray(packed.knots, dtype=np.float32)

    edge_sb = np.ascontiguousarray(packed.edge_sb, dtype=np.float32)
    edge_ss = np.ascontiguousarray(packed.edge_ss, dtype=np.float32)
    edge_m = np.ascontiguousarray(packed.edge_m, dtype=np.float32)

    L = int(packed.L)
    K = int(packed.K)

    interp_mode = 1 if str(packed.interp) == "nearest" else 0

    oob_mode = str(packed.oob_mode).lower().strip()
    # clip-for-indexing is needed for clip_x, saturate_y, zero_spline
    oob_clip = 1 if oob_mode in ("clip_x", "saturate_y", "zero_spline") else 0

    value_kind = str(getattr(packed, "value_kind", "phi")).lower().strip()
    zero_spline = 1 if (oob_mode == "zero_spline" and value_kind == "spline") else 0

    x_min = float(packed.x_min)
    x_max = float(packed.x_max)

    dx: Optional[float] = packed.uniform_dx
    if dx is not None:
        return _forward_numba_uniform(
            x, q_flat, scale, y_min,
            edge_sb, edge_ss, edge_m,
            knots, L, K, interp_mode, oob_clip, zero_spline,
            x_min, x_max, float(dx)
        )
    return _forward_numba_general(
        x, q_flat, scale, y_min,
        edge_sb, edge_ss, edge_m,
        knots, L, K, interp_mode, oob_clip, zero_spline,
        x_min, x_max
    )


def warmup_numba(packed, in_dim: int, out_dim: int) -> None:
    if not _NUMBA_AVAILABLE:
        return
    x = np.zeros((8, in_dim), dtype=np.float32)
    _ = forward_lut_layer_numba(x, packed)
