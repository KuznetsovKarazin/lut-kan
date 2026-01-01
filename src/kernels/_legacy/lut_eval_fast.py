# src/kernels/lut_eval_fast.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from src.quant.lut_builder import LUTArtifact


def _silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def _base_fn(x: np.ndarray, base_kind: str) -> np.ndarray:
    bk = (base_kind or "silu").lower().strip()
    if bk in ("silu", "swish"):
        return _silu(x)
    if bk in ("identity", "none", "linear"):
        return np.asarray(x, dtype=np.float32)
    if bk == "relu":
        return np.maximum(x, 0.0).astype(np.float32)
    raise ValueError(f"Unsupported base_kind='{base_kind}'")


def _clip_for_indexing(x: np.ndarray, x_min: float, x_max: float, oob_mode: str) -> np.ndarray:
    if oob_mode in ("clip_x", "saturate_y", "zero_spline"):
        return np.clip(x, x_min, x_max).astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _precompute_indices(x: np.ndarray, knots: np.ndarray, L: int, oob_mode: str):
    x = np.asarray(x, dtype=np.float32)
    x_min = float(knots[0])
    x_max = float(knots[-1])
    x_idx = _clip_for_indexing(x, x_min, x_max, oob_mode)

    K = int(knots.size - 1)
    k = np.searchsorted(knots, x_idx, side="right") - 1
    k = np.clip(k, 0, K - 1).astype(np.int32)

    t0 = knots[k]
    t1 = knots[k + 1]
    denom = (t1 - t0)
    denom = np.where(denom == 0.0, 1.0, denom)
    u = (x_idx - t0) / denom
    p = u * (L - 1)

    if L < 2:
        raise ValueError("L must be >= 2")

    r0 = np.floor(p).astype(np.int32)
    r0 = np.clip(r0, 0, L - 2)
    w = (p - r0.astype(np.float32)).astype(np.float32)
    r1 = r0 + 1
    return k, r0, r1, w


def forward_lut_layer_fast(x: np.ndarray, art: LUTArtifact, edges) -> np.ndarray:
    """
    Fast path: precompute (k,r0,r1,w) per src_idx once; gather edge rows in batch.
    Works for:
      - value_kind='phi'  (legacy)
      - value_kind='spline' (Method 2B) with reconstruction and optional oob_mode='zero_spline'
    """
    x = np.asarray(x, dtype=np.float32)
    in_dim = int(max(e.src_idx for e in edges) + 1)
    out_dim = int(max(e.dst_idx for e in edges) + 1)
    if x.ndim != 2 or x.shape[1] != in_dim:
        raise ValueError(f"Expected x shape [N,{in_dim}], got {x.shape}")

    value_kind = str(getattr(art, "value_kind", "phi")).lower().strip()
    oob_mode = str(getattr(art, "oob_mode", "clip_x")).lower().strip()

    # Edge params (only needed for spline mode, but we keep fallback defaults for phi mode)
    sb_arr = getattr(art, "edge_sb", None)
    ss_arr = getattr(art, "edge_ss", None)
    m_arr = getattr(art, "edge_m", None)
    base_kind = str(getattr(art, "base_kind", "silu"))

    # Group edges by src_idx
    by_src: Dict[int, List[Tuple[int, int]]] = defaultdict(list)  # src -> [(edge_id,dst_idx)]
    for e in edges:
        by_src[int(e.src_idx)].append((int(e.edge_id), int(e.dst_idx)))

    N = int(x.shape[0])
    y = np.zeros((N, out_dim), dtype=np.float32)

    knots = np.asarray(art.knots, dtype=np.float32)
    L = int(art.L)
    x_min = float(knots[0])
    x_max = float(knots[-1])

    for src_idx, lst in by_src.items():
        x_src = x[:, src_idx].astype(np.float32, copy=False)
        k, r0, r1, w = _precompute_indices(x_src, knots, L, oob_mode)

        eids = np.array([t[0] for t in lst], dtype=np.int32)   # [M]
        dsts = np.array([t[1] for t in lst], dtype=np.int32)   # [M]
        M = int(eids.size)

        # Gather q0/q1 for all edges (M) and all samples (N): (M,N)
        q_table = art.q_table[eids]  # (M,K,L)
        q0 = q_table[np.arange(M)[:, None], k[None, :], r0[None, :]].astype(np.float32, copy=False)
        q1 = q_table[np.arange(M)[:, None], k[None, :], r1[None, :]].astype(np.float32, copy=False)

        y_min = art.y_min[eids, :][:, k].astype(np.float32, copy=False)   # (M,N)
        scale = art.scale[eids, :][:, k].astype(np.float32, copy=False)   # (M,N)
        y0 = y_min + scale * q0

        if str(getattr(art, "interp", "linear")) == "nearest":
            val = y0
        else:
            y1 = y_min + scale * q1
            val = (1.0 - w[None, :]) * y0 + w[None, :] * y1  # (M,N)

        if value_kind == "phi":
            # direct phi accumulation
            for dst in np.unique(dsts):
                mask = (dsts == dst)
                y[:, dst] += np.sum(val[mask, :], axis=0)
            continue

        # Method 2B: val is spline_hat; apply OOB policy for spline branch
        spline = val
        if oob_mode == "zero_spline":
            in_mask = ((x_src >= x_min) & (x_src < x_max)).astype(np.float32)  # [N]
            spline = spline * in_mask[None, :]

        if sb_arr is None or ss_arr is None or m_arr is None:
            raise ValueError("value_kind='spline' but edge_sb/edge_ss/edge_m are missing. Fix lut_io or rebuild LUT.")

        sb = sb_arr[eids].astype(np.float32, copy=False)[:, None]  # (M,1)
        ss = ss_arr[eids].astype(np.float32, copy=False)[:, None]  # (M,1)
        mm = m_arr[eids].astype(np.float32, copy=False)[:, None]   # (M,1)

        base = _base_fn(x_src, base_kind).astype(np.float32, copy=False)[None, :]  # (1,N)
        phi = mm * (sb * base + ss * spline)  # (M,N)

        for dst in np.unique(dsts):
            mask = (dsts == dst)
            y[:, dst] += np.sum(phi[mask, :], axis=0)

    return y
