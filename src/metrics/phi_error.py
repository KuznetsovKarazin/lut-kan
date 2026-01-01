# src/metrics/phi_error.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import inspect
import numpy as np

from src.quant.lut_builder import LUTArtifact, build_segment_grid


@dataclass(frozen=True)
class PhiErrorReport:
    summary: Dict[str, float]
    topk_by_max_abs: List[Dict[str, float]]


def _base_fn_np(x: np.ndarray, base_kind: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    bk = (base_kind or "none").lower().strip()
    if bk in ("none", "identity", "id"):
        return x
    if bk == "silu":
        return x / (1.0 + np.exp(-x, dtype=np.float32))
    raise ValueError(f"Unsupported base_kind='{base_kind}'")


def _nextafter_less(x: float) -> float:
    return float(np.nextafter(np.float32(x), np.float32(-np.inf), dtype=np.float32))


def _normalize_value_representation(art: LUTArtifact) -> str:
    vr = getattr(art, "value_representation", None)
    if vr is not None:
        s = str(vr).lower().strip()
        if s in ("phi",):
            return "phi"
        if s in ("spline_component", "spline", "component"):
            return "spline_component"
    # legacy
    vk = str(getattr(art, "value_kind", "phi") or "phi").lower().strip()
    return "spline_component" if vk in ("spline", "spline_component") else "phi"


def _normalize_oob(art: LUTArtifact) -> Tuple[str, str]:
    """
    Returns (oob_behavior, legacy_hint)
      oob_behavior: "clip" | "zero"
      legacy_hint:  "zero_phi" | "zero_spline" | ""
    """
    ob = getattr(art, "oob_behavior", None)
    if ob is not None:
        s = str(ob).lower().strip()
        if s in ("clip", "clip_x"):
            return "clip", ""
        if s in ("zero", "zero_spline", "zero_phi", "zero_lut"):
            return "zero", ""

    mode = str(getattr(art, "oob_mode", "clip_x") or "clip_x").lower().strip()
    if mode in ("zero_spline", "zero"):
        return "zero", "zero_spline"
    if mode in ("zero_phi",):
        return "zero", "zero_phi"
    return "clip", ""


def _read_coeffs(art: LUTArtifact, edge_id: int) -> Tuple[float, float, float]:
    # new
    ebs = getattr(art, "edge_base_scale", None)
    ess = getattr(art, "edge_spline_scale", None)
    eos = getattr(art, "edge_out_scale", None)
    if ebs is not None and ess is not None and eos is not None:
        return float(np.asarray(ebs)[edge_id]), float(np.asarray(ess)[edge_id]), float(np.asarray(eos)[edge_id])

    # legacy
    sb = getattr(art, "edge_sb", None)
    ss = getattr(art, "edge_ss", None)
    m = getattr(art, "edge_m", None)
    if sb is None or ss is None or m is None:
        raise ValueError("Missing reconstruction coefficients for spline_component.")
    return float(np.asarray(sb)[edge_id]), float(np.asarray(ss)[edge_id]), float(np.asarray(m)[edge_id])


def _eval_lut_component(
    art: LUTArtifact, edge_id: int, x: np.ndarray, *, boundary_mode: str = "half_open"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (lut_val, in_mask) where:
      - lut_val is the LUT component evaluated with safe indexing and interpolation
      - in_mask is the in-domain mask (float32 0/1) for x_raw (NOT clipped)
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    knots = np.asarray(art.knots, dtype=np.float32)
    K = int(knots.size - 1)
    L = int(art.L)

    x_min = float(knots[0])
    x_max = float(knots[-1])
    x_hi = x_max if boundary_mode == "closed" else _nextafter_less(x_max)

    # Safe indexing uses clipped x
    x_clip = np.clip(x, np.float32(x_min), np.float32(x_hi))

    k = np.searchsorted(knots, x_clip, side="right").astype(np.int32) - 1
    k = np.clip(k, 0, K - 1)

    x0 = knots[k]
    x1 = knots[k + 1]
    denom = np.maximum(x1 - x0, np.float32(1e-12))
    u = (x_clip - x0) / denom
    u = np.clip(u, np.float32(0.0), np.float32(1.0))

    pos = u * np.float32(L - 1)
    r0 = np.floor(pos).astype(np.int32)
    r0 = np.clip(r0, 0, L - 1)
    r1 = np.clip(r0 + 1, 0, L - 1)
    w = (pos - r0.astype(np.float32)).astype(np.float32)

    q = np.asarray(art.q_table[edge_id], dtype=np.float32)  # [K,L]
    q0 = q[k, r0]
    q1 = q[k, r1]

    y_min = np.asarray(art.y_min[edge_id], dtype=np.float32)[k]
    scale = np.asarray(art.scale[edge_id], dtype=np.float32)[k]

    y0 = y_min + scale * q0
    y1 = y_min + scale * q1

    interp = str(getattr(art, "interp", "linear") or "linear").lower().strip()
    if interp == "nearest":
        lut_val = np.where(w < 0.5, y0, y1)
    else:
        lut_val = (1.0 - w) * y0 + w * y1

    # in-domain mask uses raw x (not clipped)
    if boundary_mode == "closed":
        in_mask = ((x >= x_min) & (x <= x_max)).astype(np.float32)
    else:
        in_mask = ((x >= x_min) & (x < x_max)).astype(np.float32)

    oob_behavior, _hint = _normalize_oob(art)
    if oob_behavior == "zero":
        lut_val = lut_val * in_mask

    return lut_val.astype(np.float32, copy=False), in_mask


def _eval_phi_lut(art: LUTArtifact, edge, x: np.ndarray) -> np.ndarray:
    rep = _normalize_value_representation(art)
    boundary_mode = str(getattr(art, "boundary_mode", "half_open") or "half_open").lower().strip()
    if boundary_mode not in ("half_open", "closed"):
        boundary_mode = "half_open"

    lut_component, in_mask = _eval_lut_component(art, int(edge.edge_id), x, boundary_mode=boundary_mode)

    if rep == "phi":
        # Legacy hint: some old experiments used "zero_phi" (mask full phi)
        _oob_behavior, hint = _normalize_oob(art)
        if hint == "zero_phi":
            return (lut_component * in_mask).astype(np.float32, copy=False)
        return lut_component

    # spline_component: reconstruct phi = out*(base_scale*base(x_raw) + spline_scale*lut_component)
    base_kind = str(getattr(art, "base_kind", getattr(edge, "base_kind", "none")) or "none").lower().strip()
    sb, ss, m = _read_coeffs(art, int(edge.edge_id))

    base = _base_fn_np(x, base_kind)
    phi = (m * (sb * base + ss * lut_component)).astype(np.float32, copy=False)

    # Legacy hint: if a legacy artifact intended "zero_phi", apply it after reconstruction.
    _oob_behavior, hint = _normalize_oob(art)
    if hint == "zero_phi":
        phi = phi * in_mask.astype(np.float32)

    return phi


def evaluate_phi_error_on_grid(edges: List, art: LUTArtifact, num_points: int = 256, topk: int = 10) -> PhiErrorReport:
    knots = np.asarray(art.knots, dtype=np.float32)
    K = int(knots.size - 1)

    # build_segment_grid signature compatibility:
    # - new: build_segment_grid(knots, L)
    # - old: build_segment_grid(knots, K, num_points)
    _sig = inspect.signature(build_segment_grid)
    _nparams = len(_sig.parameters)
    if _nparams == 2:
        # Here we use num_points as "L per segment" for the evaluation grid.
        x_grid_2d = build_segment_grid(knots, int(num_points)).astype(np.float32, copy=False)  # [K,L]
        x_grid = x_grid_2d.reshape(-1)
    else:
        x_grid = build_segment_grid(knots, K, int(num_points)).astype(np.float32, copy=False)

    per_edge = []
    for e in edges:
        x = x_grid
        y_true = np.asarray(e.eval_phi(x), dtype=np.float32)
        y_hat = _eval_phi_lut(art, e, x)
        d = y_hat - y_true
        per_edge.append(
            {
                "edge_id": int(e.edge_id),
                "src": int(e.src_idx),
                "dst": int(e.dst_idx),
                "mae": float(np.mean(np.abs(d))),
                "rmse": float(np.sqrt(np.mean(d * d))),
                "max_abs": float(np.max(np.abs(d))),
            }
        )

    max_abs_all = np.array([p["max_abs"] for p in per_edge], dtype=np.float32)
    summary = {
        "edges": float(len(per_edge)),
        "max_abs_mean": float(np.mean(max_abs_all)) if len(per_edge) else 0.0,
        "max_abs_p95": float(np.percentile(max_abs_all, 95)) if len(per_edge) else 0.0,
        "max_abs_max": float(np.max(max_abs_all)) if len(per_edge) else 0.0,
    }

    top = sorted(per_edge, key=lambda r: r["max_abs"], reverse=True)[: int(topk)]
    return PhiErrorReport(summary=summary, topk_by_max_abs=top)
