# src/models/bspline_adapter.py
"""
Standalone B-spline KAN adapter without PyKAN/torch dependency.

This provides the same interface as JacobiKANSingleLayerAdapter for fair comparison.
Implements cubic B-splines (k=3) using Cox-de Boor recursion in pure NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class EdgeSpec:
    """Edge specification compatible with build_lut_for_edges()."""
    edge_id: int
    src_idx: int
    dst_idx: int
    knots: np.ndarray
    domain: Tuple[float, float]
    eval_phi: Callable[[np.ndarray], np.ndarray]
    eval_spline: Optional[Callable[[np.ndarray], np.ndarray]] = None
    base_kind: str = "none"
    sb: float = 1.0
    ss: float = 1.0
    m: float = 1.0


def _bspline_basis_all(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """
    Compute all B-spline basis functions N_{i,degree}(x) using Cox-de Boor recursion.
    
    Args:
        x: (N,) array of evaluation points
        knots: (M,) augmented knot vector (must include k repeated knots at ends)
        degree: B-spline degree (k)
    
    Returns:
        B: (M-k-1, N) array where B[i, n] = N_{i,k}(x[n])
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    t = np.asarray(knots, dtype=np.float32).ravel()
    k = int(degree)
    
    M = len(t)
    N = len(x)
    
    if M < k + 2:
        raise ValueError(f"Knot vector too short: M={M}, degree={k}")
    
    # Degree 0 basis: indicator functions for [t_i, t_{i+1})
    B = np.zeros((M - 1, N), dtype=np.float32)
    for i in range(M - 1):
        left, right = t[i], t[i + 1]
        # Last interval is closed [t_{M-2}, t_{M-1}]
        if i == M - 2:
            B[i, :] = ((x >= left) & (x <= right)).astype(np.float32)
        else:
            B[i, :] = ((x >= left) & (x < right)).astype(np.float32)
    
    # Elevate degree using recurrence
    for d in range(1, k + 1):
        B_new = np.zeros((M - 1 - d, N), dtype=np.float32)
        for i in range(M - 1 - d):
            denom1 = t[i + d] - t[i]
            denom2 = t[i + d + 1] - t[i + 1]
            
            term1 = 0.0
            term2 = 0.0
            
            if denom1 > 1e-10:
                term1 = (x - t[i]) / denom1 * B[i, :]
            if denom2 > 1e-10:
                term2 = (t[i + d + 1] - x) / denom2 * B[i + 1, :]
            
            B_new[i, :] = term1 + term2
        B = B_new
    
    return B  # shape: (M-k-1, N) = (num_basis, N)


def _bspline_eval(x: np.ndarray, coef: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """
    Evaluate B-spline curve: S(x) = sum_i c_i * N_{i,k}(x)
    
    Args:
        x: (N,) evaluation points
        coef: (num_basis,) control points
        knots: (M,) augmented knot vector
        degree: spline degree
    
    Returns:
        y: (N,) spline values
    """
    B = _bspline_basis_all(x, knots, degree)  # (num_basis, N)
    c = np.asarray(coef, dtype=np.float32).ravel()  # (num_basis,)
    
    if B.shape[0] != len(c):
        raise ValueError(f"Coefficient mismatch: basis={B.shape[0]}, coef={len(c)}")
    
    # y = c @ B = sum_i c_i * B_i(x)
    y = np.dot(c, B)  # (N,)
    return y.astype(np.float32)


def _silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x / (1 + exp(-x))"""
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


class BSplineKANSingleLayerAdapter:
    """
    Standalone B-spline KAN adapter (no PyKAN dependency).
    
    Implements: phi_{ij}(x) = m_{ij} * (sb_{ij} * base(x) + ss_{ij} * spline_{ij}(x))
    
    This matches PyKAN semantics for fair comparison with Jacobi adapter.
    """
    
    def __init__(
        self,
        coef: np.ndarray,
        *,
        degree: int,
        grid: np.ndarray,
        sb: np.ndarray,
        ss: np.ndarray,
        m: np.ndarray,
        base_kind: str = "silu",
    ) -> None:
        """
        Args:
            coef: (in_dim, out_dim, num_coef) spline coefficients
            degree: B-spline degree (typically 3 for cubic)
            grid: (num_interior_knots,) interior grid points
            sb: (in_dim, out_dim) base scale
            ss: (in_dim, out_dim) spline scale
            m: (in_dim, out_dim) mask/output scale
            base_kind: "silu" or "none"
        """
        self.coef = np.asarray(coef, dtype=np.float32)
        self.degree = int(degree)
        self.base_kind = str(base_kind).lower()
        
        if self.coef.ndim != 3:
            raise ValueError(f"coef must be 3D [in_dim, out_dim, num_coef], got {self.coef.shape}")
        
        self.in_dim = int(self.coef.shape[0])
        self.out_dim = int(self.coef.shape[1])
        self.num_coef = int(self.coef.shape[2])
        
        # Build augmented knot vector from interior grid
        grid = np.asarray(grid, dtype=np.float32).ravel()
        self.grid = grid
        self.x_min = float(grid[0])
        self.x_max = float(grid[-1])
        
        # Augment knots: repeat boundary knots k times
        k = self.degree
        self.knots_aug = np.concatenate([
            np.full(k, grid[0], dtype=np.float32),
            grid,
            np.full(k, grid[-1], dtype=np.float32),
        ])
        
        # Check consistency
        expected_coef = len(self.knots_aug) - self.degree - 1
        if self.num_coef != expected_coef:
            raise ValueError(
                f"Coefficient count mismatch: got {self.num_coef}, "
                f"expected {expected_coef} for knots_len={len(self.knots_aug)}, degree={self.degree}"
            )
        
        self.sb = np.asarray(sb, dtype=np.float32)
        self.ss = np.asarray(ss, dtype=np.float32)
        self.m = np.asarray(m, dtype=np.float32)
        
        if self.sb.shape != (self.in_dim, self.out_dim):
            raise ValueError(f"sb shape mismatch: {self.sb.shape}")
        if self.ss.shape != (self.in_dim, self.out_dim):
            raise ValueError(f"ss shape mismatch: {self.ss.shape}")
        if self.m.shape != (self.in_dim, self.out_dim):
            raise ValueError(f"m shape mismatch: {self.m.shape}")
    
    @staticmethod
    def from_arch(arch: dict, *, seed: int = 0) -> "BSplineKANSingleLayerAdapter":
        """
        Construct adapter from architecture config.
        
        Config keys:
            - in_dim, out_dim (required)
            - degree (default: 3)
            - grid_points (default: 5) - number of interior grid points
            - x_min, x_max (default: -3, 3)
            - base_kind (default: "silu")
        """
        in_dim = int(arch.get("in_dim", 0))
        out_dim = int(arch.get("out_dim", 0))
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be > 0")
        
        degree = int(arch.get("degree", 3))
        grid_points = int(arch.get("grid_points", 5))
        x_min = float(arch.get("x_min", -3.0))
        x_max = float(arch.get("x_max", 3.0))
        base_kind = str(arch.get("base_kind", "silu"))
        
        # Build interior grid
        grid = np.linspace(x_min, x_max, grid_points, dtype=np.float32)
        
        # Number of coefficients: len(knots_aug) - degree - 1 = (grid_points + 2*degree) - degree - 1 = grid_points + degree - 1
        num_coef = grid_points + degree - 1
        
        # Random initialization
        rng = np.random.default_rng(int(seed))
        
        # Initialize coefficients (small random values)
        coef = rng.normal(0, 0.1 / np.sqrt(in_dim * num_coef), 
                          size=(in_dim, out_dim, num_coef)).astype(np.float32)
        
        # PyKAN-style initialization for scales
        sb = np.ones((in_dim, out_dim), dtype=np.float32)  # base scale
        ss = np.ones((in_dim, out_dim), dtype=np.float32)  # spline scale (can be learned)
        m = np.ones((in_dim, out_dim), dtype=np.float32)   # mask (typically 1)
        
        return BSplineKANSingleLayerAdapter(
            coef=coef,
            degree=degree,
            grid=grid,
            sb=sb,
            ss=ss,
            m=m,
            base_kind=base_kind,
        )
    
    def _apply_base(self, x: np.ndarray) -> np.ndarray:
        """Apply base function."""
        if self.base_kind == "silu":
            return _silu(x)
        return x
    
    def extract_edges(self) -> List[EdgeSpec]:
        """Extract edge specifications for LUT building."""
        edges: List[EdgeSpec] = []
        edge_id = 0
        
        # For LUT builder, we use interior grid as knots
        knots_for_lut = self.grid.copy()
        domain = (float(self.x_min), float(self.x_max))
        
        for out_idx in range(self.out_dim):
            for in_idx in range(self.in_dim):
                c = self.coef[in_idx, out_idx, :].copy()
                sb_val = float(self.sb[in_idx, out_idx])
                ss_val = float(self.ss[in_idx, out_idx])
                m_val = float(self.m[in_idx, out_idx])
                
                def _eval_spline(
                    x: np.ndarray,
                    c=c,
                    knots=self.knots_aug.copy(),
                    degree=self.degree,
                ) -> np.ndarray:
                    x = np.asarray(x, dtype=np.float32).ravel()
                    return _bspline_eval(x, c, knots, degree)
                
                def _eval_phi(
                    x: np.ndarray,
                    c=c,
                    knots=self.knots_aug.copy(),
                    degree=self.degree,
                    sb=sb_val,
                    ss=ss_val,
                    m=m_val,
                    base_kind=self.base_kind,
                ) -> np.ndarray:
                    x = np.asarray(x, dtype=np.float32).ravel()
                    base = _silu(x) if base_kind == "silu" else x
                    spl = _bspline_eval(x, c, knots, degree)
                    return (m * (sb * base + ss * spl)).astype(np.float32)
                
                edges.append(EdgeSpec(
                    edge_id=edge_id,
                    src_idx=in_idx,
                    dst_idx=out_idx,
                    knots=knots_for_lut,
                    domain=domain,
                    eval_phi=_eval_phi,
                    eval_spline=_eval_spline,
                    base_kind=self.base_kind,
                    sb=sb_val,
                    ss=ss_val,
                    m=m_val,
                ))
                edge_id += 1
        
        return edges
    
    def forward_float(self, x: np.ndarray) -> np.ndarray:
        """
        Float forward pass: y = sum_i phi_{ij}(x_i)
        
        Args:
            x: (N, in_dim) input
        
        Returns:
            y: (N, out_dim) output
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected x shape [N, {self.in_dim}], got {x.shape}")
        
        N = x.shape[0]
        y = np.zeros((N, self.out_dim), dtype=np.float32)
        
        base_x = self._apply_base(x)
        
        for i in range(self.in_dim):
            xi = x[:, i]
            base_i = base_x[:, i]
            
            for j in range(self.out_dim):
                m_val = self.m[i, j]
                if m_val == 0.0:
                    continue
                
                sb_val = self.sb[i, j]
                ss_val = self.ss[i, j]
                c = self.coef[i, j, :]
                
                spl = _bspline_eval(xi, c, self.knots_aug, self.degree)
                y[:, j] += m_val * (sb_val * base_i + ss_val * spl)
        
        return y


# =============================================================================
# MCU Cycle Estimation for B-spline
# =============================================================================

def estimate_bspline_float_cycles(
    degree: int,
    num_coef: int,
    in_dim: int,
    out_dim: int,
    use_silu: bool = True,
    float_add: int = 50,
    float_mul: int = 80,
    float_div: int = 150,
    silu_cycles: int = 300,  # exp() call dominates
) -> int:
    """
    Estimate cycles for B-spline float forward on MCU without FPU.
    
    Cox-de Boor recursion costs:
    - Degree 0: num_coef comparisons
    - Each degree elevation: ~4 mul + 4 add + 2 div per basis function
    
    Total per input coordinate:
    - SiLU: 1 exp + 1 div + 1 add
    - Basis evaluation: O(degree * num_coef) ops
    - Output accumulation: O(out_dim * 2) ops
    """
    # SiLU cost
    silu_cost = silu_cycles if use_silu else 0
    
    # Cox-de Boor recursion (rough estimate)
    # Each degree step: ~4 mul + 4 add + 2 div per surviving basis
    # Basis count decreases each step
    basis_cost = 0
    num_basis = num_coef + degree
    for d in range(1, degree + 1):
        num_basis -= 1
        basis_cost += num_basis * (4 * float_mul + 4 * float_add + 2 * float_div)
    
    # Coefficient dot product
    dot_cost = num_coef * (float_mul + float_add)
    
    # Per-edge total
    cost_per_coord = silu_cost + basis_cost + dot_cost
    
    # Output accumulation
    cost_per_coord += out_dim * (2 * float_mul + 2 * float_add)
    
    total = in_dim * cost_per_coord
    return int(total)
