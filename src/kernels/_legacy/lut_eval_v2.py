# src/kernels/lut_eval_v2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.quant.lut_builder import LUTArtifact


def _silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def _base_fn(x: np.ndarray, base_kind: str) -> np.ndarray:
    bk = (base_kind or "none").strip().lower()
    if bk in ("none", "identity"):
        return np.asarray(x, dtype=np.float32)
    if bk == "silu":
        return _silu(x)
    raise ValueError(f"Unsupported base_kind='{base_kind}'")


def _nextafter_less(a: float) -> float:
    # largest float32 strictly smaller than a
    return float(np.nextafter(np.float32(a), np.float32(-np.inf), dtype=np.float32))


def _in_domain_mask(x: np.ndarray, x_min: float, x_max: float, boundary: str) -> np.ndarray:
    if boundary == "closed":
        return (x >= x_min) & (x <= x_max)
    # half_open
    return (x >= x_min) & (x < x_max)


def _segment_index_uniform(
    x_raw: np.ndarray,
    x_min: float,
    x_max: float,
    dx: float,
    K: int,
    L: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (k, r0, r1, w) for each x in vectorized manner.

    Indexing is always safe:
      - x is clipped into [x_min, x_max) (half-open) for index computation.
      - this is NOT semantic OOB behavior.
    """
    x = np.asarray(x_raw, dtype=np.float32)
    x_hi = _nextafter_less(x_max)
    x_clip = np.minimum(np.maximum(x, x_min), x_hi)

    t = (x_clip - x_min) / np.float32(dx)  # in [0, K)
    k = np.floor(t).astype(np.int32)
    k = np.clip(k, 0, K - 1)

    x0 = x_min + k.astype(np.float32) * np.float32(dx)
    u = (x_clip - x0) / np.float32(dx)  # in [0,1)
    pos = u * np.float32(L)

    r0 = np.floor(pos).astype(np.int32)
    r0 = np.clip(r0, 0, L - 1)

    if L > 1:
        r1 = np.minimum(r0 + 1, L - 1).astype(np.int32)
    else:
        r1 = r0

    w = (pos - r0.astype(np.float32)).astype(np.float32)
    return k, r0, r1, w


def _segment_index_general(
    x_raw: np.ndarray,
    knots: np.ndarray,
    K: int,
    L: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_raw, dtype=np.float32)
    x_min = float(knots[0])
    x_max = float(knots[-1])
    x_hi = _nextafter_less(x_max)
    x_clip = np.minimum(np.maximum(x, x_min), x_hi)

    # k via searchsorted: find rightmost knot <= x
    k = np.searchsorted(knots, x_clip, side="right").astype(np.int32) - 1
    k = np.clip(k, 0, K - 1)

    x0 = knots[k].astype(np.float32, copy=False)
    x1 = knots[k + 1].astype(np.float32, copy=False)
    dx = (x1 - x0)
    dx = np.where(dx > 0, dx, np.float32(1.0))
    u = (x_clip - x0) / dx
    u = np.clip(u, 0.0, 0.99999994).astype(np.float32)

    pos = u * np.float32(L)
    r0 = np.floor(pos).astype(np.int32)
    r0 = np.clip(r0, 0, L - 1)
    r1 = np.minimum(r0 + 1, L - 1).astype(np.int32)
    w = (pos - r0.astype(np.float32)).astype(np.float32)
    return k, r0, r1, w


@dataclass
class PackedLUT:
    """
    Dense packed layout for a single dense KAN layer.

    Shapes:
      q_flat: [in_dim, out_dim, K*L]
      scale:  [in_dim, out_dim, K]
      y_min:  [in_dim, out_dim, K]
      coef_*: [in_dim, out_dim]
    """
    q_flat: np.ndarray
    scale: np.ndarray
    y_min: np.ndarray
    knots: np.ndarray
    L: int
    K: int
    interp: str

    # Domain semantics
    x_min: float
    x_max: float
    boundary_mode: str
    oob_behavior: str  # 'clip' or 'zero' (semantic)

    # Representation
    value_representation: str  # 'phi' or 'spline_component'
    base_kind: str
    coef_base: np.ndarray   # [in_dim, out_dim]
    coef_spline: np.ndarray # [in_dim, out_dim]
    coef_out: np.ndarray    # [in_dim, out_dim]

    uniform_dx: Optional[float]


def pack_lut_dense(art: LUTArtifact, edges, in_dim: int, out_dim: int) -> PackedLUT:
    """
    Pack edge-major artifact into dense [in_dim, out_dim, ...] layout.
    Requires a dense layer (all i->j edges exist).
    """
    K = int(np.asarray(art.knots).size - 1)
    L = int(art.L)

    edge_ids = np.full((in_dim, out_dim), -1, dtype=np.int32)
    for e in edges:
        edge_ids[int(e.src_idx), int(e.dst_idx)] = int(e.edge_id)
    if np.any(edge_ids < 0):
        raise ValueError("pack_lut_dense expects a dense layer (all (i,j) edges exist).")

    q = art.q_table[edge_ids.reshape(-1)].reshape(in_dim, out_dim, K, L)
    q_flat = q.reshape(in_dim, out_dim, K * L)

    scale = np.asarray(art.scale)[edge_ids.reshape(-1)].reshape(in_dim, out_dim, K)
    y_min = np.asarray(art.y_min)[edge_ids.reshape(-1)].reshape(in_dim, out_dim, K)

    # coefficients
    rep = str(art.value_representation).strip().lower()
    if rep == "spline_component":
        if art.edge_base_scale is None or art.edge_spline_scale is None or art.edge_out_scale is None:
            raise ValueError("spline_component artifact missing reconstruction coefficients")
        coef_base = np.asarray(art.edge_base_scale, dtype=np.float32)[edge_ids.reshape(-1)].reshape(in_dim, out_dim)
        coef_spline = np.asarray(art.edge_spline_scale, dtype=np.float32)[edge_ids.reshape(-1)].reshape(in_dim, out_dim)
        coef_out = np.asarray(art.edge_out_scale, dtype=np.float32)[edge_ids.reshape(-1)].reshape(in_dim, out_dim)
        base_kind = str(art.base_kind or "unknown")
    else:
        coef_base = np.zeros((in_dim, out_dim), dtype=np.float32)
        coef_spline = np.ones((in_dim, out_dim), dtype=np.float32)
        coef_out = np.ones((in_dim, out_dim), dtype=np.float32)
        base_kind = "none"

    # uniform dx if possible
    knots = np.asarray(art.knots, dtype=np.float32)
    diffs = np.diff(knots)
    uniform_dx = None
    if np.allclose(diffs, diffs[0]):
        uniform_dx = float(diffs[0])

    return PackedLUT(
        q_flat=np.ascontiguousarray(q_flat),
        scale=np.ascontiguousarray(scale),
        y_min=np.ascontiguousarray(y_min),
        knots=knots,
        L=L,
        K=K,
        interp=str(art.interp),
        x_min=float(knots[0]),
        x_max=float(knots[-1]),
        boundary_mode=str(getattr(art, "boundary_mode", "half_open")),
        oob_behavior=str(getattr(art, "oob_behavior", "clip")),
        value_representation=rep,
        base_kind=base_kind,
        coef_base=np.ascontiguousarray(coef_base),
        coef_spline=np.ascontiguousarray(coef_spline),
        coef_out=np.ascontiguousarray(coef_out),
        uniform_dx=uniform_dx,
    )


def forward_lut_layer_v2(x: np.ndarray, packed: PackedLUT) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected x shape [N,in_dim], got {x.shape}")

    N, in_dim = x.shape
    if int(packed.q_flat.shape[0]) != in_dim:
        raise ValueError(f"PackedLUT in_dim mismatch: packed={packed.q_flat.shape[0]} vs x={in_dim}")

    out_dim = int(packed.q_flat.shape[1])
    y = np.zeros((N, out_dim), dtype=np.float32)

    boundary = str(packed.boundary_mode).strip().lower()
    oob = str(packed.oob_behavior).strip().lower()
    KL = int(packed.K * packed.L)

    for i in range(in_dim):
        x_raw = x[:, i]  # [N]

        # segment indices (safe)
        if packed.uniform_dx is not None:
            k, r0, r1, w = _segment_index_uniform(
                x_raw=x_raw,
                x_min=packed.x_min,
                x_max=packed.x_max,
                dx=float(packed.uniform_dx),
                K=packed.K,
                L=packed.L,
            )
        else:
            k, r0, r1, w = _segment_index_general(
                x_raw=x_raw,
                knots=packed.knots,
                K=packed.K,
                L=packed.L,
            )

        seg0 = (k * packed.L + r0).astype(np.int32)
        seg1 = (k * packed.L + r1).astype(np.int32)
        seg0 = np.clip(seg0, 0, KL - 1)
        seg1 = np.clip(seg1, 0, KL - 1)

        q_flat = packed.q_flat[i]  # [out_dim, K*L]
        q0 = np.take(q_flat, seg0, axis=1).astype(np.float32, copy=False)  # [out_dim, N]
        q1 = np.take(q_flat, seg1, axis=1).astype(np.float32, copy=False)  # [out_dim, N]

        y_min = packed.y_min[i]   # [out_dim, K]
        scale = packed.scale[i]   # [out_dim, K]

        y0 = np.take(y_min, k, axis=1).astype(np.float32, copy=False) + np.take(scale, k, axis=1).astype(np.float32, copy=False) * q0

        if str(packed.interp).strip().lower() == "nearest":
            lut_val = y0
        else:
            y1 = np.take(y_min, k, axis=1).astype(np.float32, copy=False) + np.take(scale, k, axis=1).astype(np.float32, copy=False) * q1
            lut_val = (1.0 - w[None, :]) * y0 + w[None, :] * y1  # [out_dim,N]

        # semantic OOB behavior
        if oob == "zero":
            in_mask = _in_domain_mask(x_raw, packed.x_min, packed.x_max, boundary).astype(np.float32)
            lut_val = lut_val * in_mask[None, :]

        # reconstruction
        base = _base_fn(x_raw, packed.base_kind).astype(np.float32, copy=False)  # [N]
        cb = packed.coef_base[i].astype(np.float32, copy=False)[:, None]        # [out_dim,1]
        cs = packed.coef_spline[i].astype(np.float32, copy=False)[:, None]
        co = packed.coef_out[i].astype(np.float32, copy=False)[:, None]

        phi = co * (cb * base[None, :] + cs * lut_val)  # [out_dim,N]
        y += phi.T

    return y
