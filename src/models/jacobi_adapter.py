# src/models/jacobi_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class EdgeSpec:
    """Edge specification compatible with build_lut_for_edges().

    The LUT builder expects:
      - a shared 1D knot vector available as `edge.knots`
      - callable `edge.eval_phi(x: np.ndarray) -> np.ndarray`

    Other fields are carried for logging/debug and to mirror the adapter API.
    """

    edge_id: int
    src_idx: int
    dst_idx: int

    # We reuse the LUT artifact format assumption: one shared knot vector per layer.
    knots: np.ndarray
    domain: Tuple[float, float]

    eval_phi: Callable[[np.ndarray], np.ndarray]

    # Fields used only when value_representation='spline_component' (not used for Jacobi).
    eval_spline: Optional[Callable[[np.ndarray], np.ndarray]] = None
    base_kind: str = "none"
    sb: float = 1.0
    ss: float = 1.0
    m: float = 1.0


def _jacobi_polynomials(x: np.ndarray, degree: int, a: float, b: float) -> np.ndarray:
    """Compute Jacobi polynomials P_0..P_degree at x.

    This matches the recurrence used in Ali Kashefi's PointNet-KAN code.

    Args:
      x: float32 array of arbitrary shape
      degree: >= 0
      a, b: Jacobi parameters (alpha, beta)

    Returns:
      P: float32 array with shape x.shape + (degree+1,)
    """
    x = np.asarray(x, dtype=np.float32)
    if degree < 0:
        raise ValueError("degree must be >= 0")

    out = np.empty(x.shape + (degree + 1,), dtype=np.float32)
    out[..., 0] = 1.0

    if degree == 0:
        return out

    out[..., 1] = (((a - b) + (a + b + 2.0) * x) / 2.0).astype(np.float32)

    for i in range(2, degree + 1):
        ii = float(i)
        A = (2 * ii + a + b - 1) * (2 * ii + a + b) / ((2 * ii) * (ii + a + b))
        B = (2 * ii + a + b - 1) * (a * a - b * b) / ((2 * ii) * (ii + a + b) * (2 * ii + a + b - 2))
        C = -2.0 * (ii + a - 1) * (ii + b - 1) * (2 * ii + a + b) / (
            (2 * ii) * (ii + a + b) * (2 * ii + a + b - 2)
        )
        out[..., i] = (A * x + B) * out[..., i - 1] + C * out[..., i - 2]

    return out


class JacobiKANSingleLayerAdapter:
    """Adapter around a single JacobiKANLayer-like parameter tensor.

    It exposes the same interface as DummyKANAdapter / PyKANSingleLayerAdapter:
      - extract_edges() -> list[EdgeSpec]
      - forward_float(x) -> float forward for x [N,in_dim]

    This adapter is intentionally torch-free to keep the LUT compiler usable
    in minimal environments.
    """

    def __init__(
        self,
        coeffs: np.ndarray,
        *,
        degree: int,
        alpha: float,
        beta: float,
        knots: np.ndarray,
        use_tanh: bool = True,
    ) -> None:
        coeffs = np.asarray(coeffs, dtype=np.float32)
        if coeffs.ndim != 3:
            raise ValueError(f"coeffs must have shape [in_dim,out_dim,degree+1], got {coeffs.shape}")
        if coeffs.shape[2] != int(degree) + 1:
            raise ValueError(
                f"coeffs last dim must be degree+1={int(degree)+1}, got {coeffs.shape[2]}"
            )

        self.coeffs = coeffs
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.use_tanh = bool(use_tanh)

        knots = np.asarray(knots, dtype=np.float32)
        if knots.ndim != 1 or knots.size < 2:
            raise ValueError("knots must be 1D with len>=2")
        if not np.all(np.diff(knots) > 0):
            raise ValueError("knots must be strictly increasing")
        self.knots = knots

        self.in_dim = int(coeffs.shape[0])
        self.out_dim = int(coeffs.shape[1])

    @staticmethod
    def from_arch(arch: dict, *, seed: int = 0) -> "JacobiKANSingleLayerAdapter":
        """Construct adapter from a config dict.

        Supported keys:
          - in_dim, out_dim (required if coeffs_path absent)
          - degree (default 3)
          - alpha, beta (defaults -0.5,-0.5)
          - use_tanh (default True)
          - x_min, x_max, num_knots (defaults -3,3,9)
          - coeffs_path: .npy or .npz. If .npz, expects array under key 'jacobi_coeffs' or 'coeffs'.
        """
        degree = int(arch.get("degree", 3))
        alpha = float(arch.get("alpha", -0.5))
        beta = float(arch.get("beta", -0.5))
        use_tanh = bool(arch.get("use_tanh", True))

        x_min = float(arch.get("x_min", -3.0))
        x_max = float(arch.get("x_max", 3.0))
        num_knots = int(arch.get("num_knots", 9))
        if num_knots < 2:
            raise ValueError("arch.num_knots must be >= 2")
        knots = np.linspace(x_min, x_max, num_knots, dtype=np.float32)

        coeffs_path = arch.get("coeffs_path", None)
        if coeffs_path:
            p = Path(str(coeffs_path))
            if not p.exists():
                raise FileNotFoundError(f"coeffs_path not found: {p}")
            if p.suffix.lower() == ".npy":
                coeffs = np.load(str(p)).astype(np.float32)
            elif p.suffix.lower() == ".npz":
                z = np.load(str(p))
                if "jacobi_coeffs" in z:
                    coeffs = z["jacobi_coeffs"].astype(np.float32)
                elif "coeffs" in z:
                    coeffs = z["coeffs"].astype(np.float32)
                else:
                    raise ValueError(".npz coeffs_path must contain 'jacobi_coeffs' or 'coeffs'")
            else:
                raise ValueError("coeffs_path must be .npy or .npz")
        else:
            in_dim = int(arch.get("in_dim", 0))
            out_dim = int(arch.get("out_dim", 0))
            if in_dim <= 0 or out_dim <= 0:
                raise ValueError("arch.in_dim and arch.out_dim are required when coeffs_path is not set")
            rng = np.random.default_rng(int(seed))
            coeffs = rng.normal(loc=0.0, scale=1.0 / (in_dim * (degree + 1)), size=(in_dim, out_dim, degree + 1)).astype(
                np.float32
            )

        return JacobiKANSingleLayerAdapter(
            coeffs=coeffs,
            degree=degree,
            alpha=alpha,
            beta=beta,
            use_tanh=use_tanh,
            knots=knots,
        )

    def extract_edges(self) -> List[EdgeSpec]:
        edges: List[EdgeSpec] = []
        edge_id = 0
        dom = (float(self.knots[0]), float(self.knots[-1]))
        for out_idx in range(self.out_dim):
            for in_idx in range(self.in_dim):
                c = self.coeffs[in_idx, out_idx, :].copy()

                def _phi(x: np.ndarray, c=c, deg=self.degree, a=self.alpha, b=self.beta, use_tanh=self.use_tanh) -> np.ndarray:
                    x = np.asarray(x, dtype=np.float32)
                    if use_tanh:
                        x = np.tanh(x).astype(np.float32, copy=False)
                    P = _jacobi_polynomials(x, degree=deg, a=a, b=b)  # [...,deg+1]
                    # dot along last axis
                    y = np.tensordot(P, c, axes=([-1], [0])).astype(np.float32, copy=False)
                    return y

                edges.append(
                    EdgeSpec(
                        edge_id=edge_id,
                        src_idx=in_idx,
                        dst_idx=out_idx,
                        knots=self.knots,
                        domain=dom,
                        eval_phi=_phi,
                    )
                )
                edge_id += 1
        return edges

    def forward_float(self, x: np.ndarray) -> np.ndarray:
        """Float forward for x [N,in_dim] -> y [N,out_dim]."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected x shape [N,{self.in_dim}], got {x.shape}")

        if self.use_tanh:
            xt = np.tanh(x).astype(np.float32, copy=False)
        else:
            xt = x

        # Compute P for each input dim independently (avoid huge 4D tensors).
        y = np.zeros((x.shape[0], self.out_dim), dtype=np.float32)
        for i in range(self.in_dim):
            P = _jacobi_polynomials(xt[:, i], degree=self.degree, a=self.alpha, b=self.beta)  # [N,deg+1]
            # For each out channel: y += P @ coeffs[i,out,:]
            # (N,deg+1) @ (deg+1,out) -> (N,out)
            y += P @ self.coeffs[i, :, :].T
        return y.astype(np.float32, copy=False)
