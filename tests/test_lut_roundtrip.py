# tests/test_lut_roundtrip.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.kernels.lut_backend_reference import forward_reference
from src.kernels.lut_backend_dense_numpy import forward_dense_numpy
from src.kernels.lut_backend_dense_numba import numba_available, forward_dense_numba, warmup_numba
from src.kernels.lut_contract import pack_dense_layer
from src.models.dummy_adapter import DummyKANAdapter
from src.quant.lut_builder import build_lut_for_edges
from src.quant.lut_io import load_lut_npz, save_lut_npz


def _call_build_lut_for_edges(edges, **kwargs):
    import inspect

    sig = inspect.signature(build_lut_for_edges)
    accepted = set(sig.parameters.keys())

    mapped = {}

    # REQUIRED in current repo version
    if "lower_pct" in accepted:
        mapped["lower_pct"] = float(kwargs.pop("lower_pct", 0.1))
    if "upper_pct" in accepted:
        mapped["upper_pct"] = float(kwargs.pop("upper_pct", 99.9))

    for k, v in kwargs.items():
        if k in accepted:
            mapped[k] = v
            continue

        if k == "value_kind" and "value_representation" in accepted:
            vv = str(v).lower().strip()
            mapped["value_representation"] = "spline_component" if vv == "spline" else "phi"
            continue

        if k == "oob_mode" and "oob_behavior" in accepted:
            mm = str(v).lower().strip()
            mapped["oob_behavior"] = "zero" if mm in ("zero", "zero_spline", "zero_phi") else "clip"
            continue

        # ignore unknown legacy keys
        continue

    return build_lut_for_edges(edges=edges, **mapped)




def test_lut_save_load_and_backends_agree(tmp_path: Path) -> None:
    adapter = DummyKANAdapter(in_dim=3, out_dim=2, num_knots=9, x_min=-2.0, x_max=2.0, seed=0)
    edges = adapter.extract_edges()

    art = _call_build_lut_for_edges(
        edges,
        L=32,
        interp="linear",
        oob_mode="clip_x",
        y_range_method="minmax",
        dtype="uint8",
        scheme="asymmetric",
        qmin=0,
        qmax=255,
        zero_point=0,
        meta_dtype="float16",
        value_kind="phi",
    )

    p = tmp_path / "lut.npz"
    save_lut_npz(p, art)
    loaded = load_lut_npz(p)


    packed = pack_dense_layer(
        loaded,
        edges=edges,
        in_dim=adapter.in_dim,
        out_dim=adapter.out_dim,
        boundary_mode="half_open",
    )

    rng = np.random.default_rng(0)
    x = rng.normal(size=(512, adapter.in_dim)).astype(np.float32)

    y_ref = forward_reference(x, packed)
    y_np = forward_dense_numpy(x, packed)

    assert float(np.max(np.abs(y_ref - y_np))) <= 1e-5

    if numba_available():
        warmup_numba(packed, in_dim=adapter.in_dim, out_dim=adapter.out_dim)
        y_nb = forward_dense_numba(x, packed)
        assert float(np.max(np.abs(y_ref - y_nb))) <= 1e-5
