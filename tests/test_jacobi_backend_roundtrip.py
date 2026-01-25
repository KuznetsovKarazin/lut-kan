# tests/test_jacobi_backend_roundtrip.py
from __future__ import annotations

import numpy as np

from src.kernels.lut_backend_reference import forward_reference
from src.kernels.lut_backend_dense_numpy import forward_dense_numpy
from src.kernels.lut_backend_dense_numba import numba_available, forward_dense_numba, warmup_numba
from src.kernels.lut_contract import pack_dense_layer
from src.models.jacobi_adapter import JacobiKANSingleLayerAdapter
from src.quant.lut_builder import build_lut_for_edges


def test_jacobi_lut_backends_agree() -> None:
    adapter = JacobiKANSingleLayerAdapter.from_arch(
        arch={
            "in_dim": 4,
            "out_dim": 3,
            "degree": 3,
            "alpha": -0.5,
            "beta": -0.5,
            "use_tanh": True,
            "x_min": -3.0,
            "x_max": 3.0,
            "num_knots": 9,
        },
        seed=0,
    )
    edges = adapter.extract_edges()

    art = build_lut_for_edges(
        edges=edges,
        L=32,
        interp="linear",
        oob_behavior="clip",
        boundary_mode="half_open",
        y_range_method="minmax",
        lower_pct=0.1,
        upper_pct=99.9,
        dtype="uint8",
        scheme="asymmetric",
        qmin=0,
        qmax=255,
        meta_dtype="float16",
        value_representation="phi",
    )

    packed = pack_dense_layer(
        art,
        edges=edges,
        in_dim=adapter.in_dim,
        out_dim=adapter.out_dim,
        boundary_mode="half_open",
    )

    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, adapter.in_dim)).astype(np.float32)

    y_ref = forward_reference(x, packed)
    y_np = forward_dense_numpy(x, packed)
    assert float(np.max(np.abs(y_ref - y_np))) <= 1e-5

    if numba_available():
        warmup_numba(packed, in_dim=adapter.in_dim, out_dim=adapter.out_dim)
        y_nb = forward_dense_numba(x, packed)
        assert float(np.max(np.abs(y_ref - y_nb))) <= 1e-5
