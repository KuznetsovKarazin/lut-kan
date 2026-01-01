# tests/test_bspline_backend_pykankan.py
from __future__ import annotations

import numpy as np
import pytest

try:
    from kan import KAN  # noqa: F401
except Exception:
    pytest.skip(
        "PyKAN is not installed (or a non-PyKAN 'kan' package is installed). "
        "Install PyKAN via `pip install pykan` or "
        "`pip install git+https://github.com/KindXiaoming/pykan.git`.",
        allow_module_level=True,
    )


from src.kernels.bspline_contract import pack_bspline_dense_layer_from_pykankan_adapter
from src.kernels.bspline_backend_dense_numpy import forward_bspline_dense_numpy
from src.kernels.bspline_backend_dense_numba import forward_bspline_dense_numba, numba_available
from src.kernels.bspline_math import coef2curve_numpy


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.max(np.abs(d)))


def _make_adapter(width, grid, k, seed, layer_idx=0):
    pytest.importorskip("torch")
    #pytest.importorskip("kan", reason="PyKAN (kan) not installed")
    from src.models.kan_wrapper import PyKANSingleLayerAdapter

    arch = {"width": list(width), "grid": int(grid), "k": int(k), "seed": int(seed), "layer_idx": int(layer_idx)}
    return PyKANSingleLayerAdapter.from_arch(arch=arch, checkpoint=None, device="cpu")


def test_coef2curve_numpy_matches_pykan_for_one_edge():
    adapter = _make_adapter(width=[4, 4], grid=5, k=3, seed=0, layer_idx=0)
    edges = adapter.extract_edges()

    packed = pack_bspline_dense_layer_from_pykankan_adapter(adapter, boundary_mode="closed")

    i, j = 0, 0
    x_min = float(packed.grid[i, 0])
    x_max = float(packed.grid[i, -1])
    x = np.linspace(x_min - 0.7, x_max + 0.7, 257).astype(np.float32)

    # PyKAN reference spline via adapter edge
    # edge_id convention: out_idx * in_dim + in_idx
    edge_id = j * adapter.in_dim + i
    y_py = edges[edge_id].eval_spline(x)

    # NumPy baseline spline
    y_np = coef2curve_numpy(
        x,
        grid=packed.grid[i, :],
        coef=packed.coef[i, j, :],
        degree=packed.degree,
        boundary_mode=packed.boundary_mode,
        knots_aug=packed.knots_aug[i, :],
    )

    assert _max_abs(y_py, y_np) < 5e-5


def test_forward_bspline_dense_numpy_matches_adapter_eval_phi_sum():
    adapter = _make_adapter(width=[8, 8], grid=5, k=3, seed=0, layer_idx=0)
    edges = adapter.extract_edges()

    N = 256
    x = np.random.default_rng(0).normal(0.0, 1.0, size=(N, adapter.in_dim)).astype(np.float32)
    x = np.clip(x, -2.2, 2.2).astype(np.float32)

    # Reference: sum of per-edge phi from PyKAN
    y_ref = np.zeros((N, adapter.out_dim), dtype=np.float32)
    for e in edges:
        y_ref[:, e.dst_idx] += e.eval_phi(x[:, e.src_idx])

    packed = pack_bspline_dense_layer_from_pykankan_adapter(adapter, boundary_mode="closed")
    y_np = forward_bspline_dense_numpy(x, packed)

    assert _max_abs(y_ref, y_np) < 5e-4


def test_forward_bspline_dense_numba_matches_numpy():
    if not numba_available():
        pytest.skip("numba not available")

    adapter = _make_adapter(width=[8, 8], grid=5, k=3, seed=1, layer_idx=0)
    packed = pack_bspline_dense_layer_from_pykankan_adapter(adapter, boundary_mode="closed")

    N = 512
    x = np.random.default_rng(1).normal(0.0, 1.0, size=(N, adapter.in_dim)).astype(np.float32)
    x = np.clip(x, -2.2, 2.2).astype(np.float32)

    y_np = forward_bspline_dense_numpy(x, packed)
    y_nb = forward_bspline_dense_numba(x, packed)

    assert _max_abs(y_np, y_nb) < 5e-5
