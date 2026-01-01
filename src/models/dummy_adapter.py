# src/models/dummy_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class EdgeSpec:
    edge_id: int
    src_idx: int
    dst_idx: int
    domain: Tuple[float, float]
    knots: np.ndarray
    eval_phi: Callable[[np.ndarray], np.ndarray]


class DummyKANAdapter:
    """
    Simple KAN-like layer: y_j = sum_i phi_{ij}(x_i)
    """
    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 2,
        num_knots: int = 9,
        x_min: float = -3.0,
        x_max: float = 3.0,
        seed: int = 42,
    ) -> None:
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.rng = np.random.default_rng(int(seed))

        if self.in_dim <= 0 or self.out_dim <= 0:
            raise ValueError("in_dim/out_dim must be > 0")
        if int(num_knots) < 2:
            raise ValueError("num_knots must be >= 2")

        self.knots = np.linspace(float(x_min), float(x_max), int(num_knots), dtype=np.float32)

        self.params = {}
        edge_id = 0
        for i in range(self.in_dim):
            for j in range(self.out_dim):
                a = self.rng.uniform(0.6, 1.6)
                b = self.rng.uniform(-1.2, 1.2)
                c = self.rng.uniform(0.2, 1.5)
                d = self.rng.uniform(-0.5, 0.5)
                self.params[(i, j)] = (a, b, c, d)
                edge_id += 1
 
    def extract_edges(self) -> List[EdgeSpec]:
        edges: List[EdgeSpec] = []
        edge_id = 0
        for i in range(self.in_dim):
            for j in range(self.out_dim):
                a, b, c, d = self.params[(i, j)]

                def _phi(x: np.ndarray, a=a, b=b, c=c, d=d) -> np.ndarray:
                    # Smooth, non-polynomial, bounded-ish on domain
                    # phi(x) = a*tanh(c*(x-b)) + d*sin(1.7x)
                    return (a * np.tanh(c * (x - b)) + d * np.sin(1.7 * x)).astype(np.float32)

                edges.append(
                    EdgeSpec(
                        edge_id=edge_id,
                        src_idx=i,
                        dst_idx=j,
                        domain=(float(self.knots[0]), float(self.knots[-1])),
                        knots=self.knots.copy(),
                        eval_phi=_phi,
                    )
                )
                edge_id += 1
        return edges

    def forward_float(self, x: np.ndarray) -> np.ndarray:
        """
        x: [N, in_dim] float32
        y: [N, out_dim] float32
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected x shape [N,{self.in_dim}], got {x.shape}")

        y = np.zeros((x.shape[0], self.out_dim), dtype=np.float32)
        for e in self.extract_edges():
            y[:, e.dst_idx] += e.eval_phi(x[:, e.src_idx])
        return y
