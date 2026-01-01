# src/kernels/lut_eval.py
from __future__ import annotations

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
    # For "zero_spline" we still clip ONLY for safe indexing, but later zero-out spline by mask.
    if oob_mode in ("clip_x", "saturate_y", "zero_spline"):
        return np.clip(x, x_min, x_max).astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _eval_lut_value(
    x: np.ndarray,
    art: LUTArtifact,
    edge_id: int,
) -> np.ndarray:
    """
    Evaluate LUT-stored value (either phi or spline depending on how artifact was built).
    Returns float32 [N].
    """
    x = np.asarray(x, dtype=np.float32)
    knots = np.asarray(art.knots, dtype=np.float32)
    K = int(knots.size - 1)
    L = int(art.L)

    x_min = float(knots[0])
    x_max = float(knots[-1])
    x_idx = _clip_for_indexing(x, x_min, x_max, str(getattr(art, "oob_mode", "clip_x")))

    k = np.searchsorted(knots, x_idx, side="right") - 1
    k = np.clip(k, 0, K - 1).astype(np.int32)

    t0 = knots[k]
    t1 = knots[k + 1]
    denom = (t1 - t0)
    denom = np.where(denom == 0.0, 1.0, denom)
    u = (x_idx - t0) / denom
    p = u * (L - 1)

    if str(getattr(art, "interp", "linear")) == "nearest":
        r = np.rint(p).astype(np.int32)
        r = np.clip(r, 0, L - 1)
        qv = art.q_table[edge_id, k, r].astype(np.float32, copy=False)
        return (art.y_min[edge_id, k].astype(np.float32, copy=False) +
                art.scale[edge_id, k].astype(np.float32, copy=False) * qv).astype(np.float32)

    r0 = np.floor(p).astype(np.int32)
    r0 = np.clip(r0, 0, L - 2)
    w = (p - r0.astype(np.float32)).astype(np.float32)
    r1 = r0 + 1

    q0 = art.q_table[edge_id, k, r0].astype(np.float32, copy=False)
    q1 = art.q_table[edge_id, k, r1].astype(np.float32, copy=False)

    y0 = art.y_min[edge_id, k].astype(np.float32, copy=False) + art.scale[edge_id, k].astype(np.float32, copy=False) * q0
    y1 = art.y_min[edge_id, k].astype(np.float32, copy=False) + art.scale[edge_id, k].astype(np.float32, copy=False) * q1
    return ((1.0 - w) * y0 + w * y1).astype(np.float32)


def eval_phi_lut(x: np.ndarray, art: LUTArtifact, edge_id: int) -> np.ndarray:
    """
    Always returns reconstructed phi(x) for the edge:
      - if artifact stores phi: returns phi_hat directly
      - if artifact stores spline: returns m*(sb*base(x) + ss*spline_hat(x)),
        with optional OOB policy 'zero_spline' for the spline branch.
    """
    value_kind = str(getattr(art, "value_kind", "phi")).lower().strip()
    oob_mode = str(getattr(art, "oob_mode", "clip_x")).lower().strip()
    knots = np.asarray(art.knots, dtype=np.float32)
    x_min = float(knots[0])
    x_max = float(knots[-1])

    # Get LUT value (phi or spline)
    val = _eval_lut_value(x, art, edge_id)

    if value_kind == "phi":
        return val.astype(np.float32)

    # Method 2B: LUT stores spline values
    sb_arr = getattr(art, "edge_sb", None)
    ss_arr = getattr(art, "edge_ss", None)
    m_arr = getattr(art, "edge_m", None)
    if sb_arr is None or ss_arr is None or m_arr is None:
        raise ValueError("Artifact has value_kind='spline' but edge_sb/edge_ss/edge_m are missing. "
                         "Fix LUT saving/loading (lut_io.py) or rebuild LUT.")

    sb = float(sb_arr[edge_id])
    ss = float(ss_arr[edge_id])
    m = float(m_arr[edge_id])

    spline = val.astype(np.float32, copy=False)

    # pykan spline basis has zero support outside [x_min, x_max) -> spline(x)=0 there
    if oob_mode == "zero_spline":
        x = np.asarray(x, dtype=np.float32)
        in_mask = (x >= x_min) & (x < x_max)
        spline = spline * in_mask.astype(np.float32)

    base_kind = str(getattr(art, "base_kind", "silu"))
    base = _base_fn(np.asarray(x, dtype=np.float32), base_kind)
    phi = m * (sb * base + ss * spline)
    return phi.astype(np.float32)


def forward_lut_layer(x: np.ndarray, art: LUTArtifact, edges) -> np.ndarray:
    """
    x: [N, in_dim]
    y: [N, out_dim]
    """
    x = np.asarray(x, dtype=np.float32)
    in_dim = int(max(e.src_idx for e in edges) + 1)
    out_dim = int(max(e.dst_idx for e in edges) + 1)

    if x.ndim != 2 or x.shape[1] != in_dim:
        raise ValueError(f"Expected x shape [N,{in_dim}], got {x.shape}")

    y = np.zeros((x.shape[0], out_dim), dtype=np.float32)
    for e in edges:
        phi_hat = eval_phi_lut(x[:, int(e.src_idx)], art, int(e.edge_id))
        y[:, int(e.dst_idx)] += phi_hat
    return y
