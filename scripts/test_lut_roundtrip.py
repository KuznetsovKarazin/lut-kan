# scripts/test_lut_roundtrip.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.kernels.lut_eval import forward_lut_layer
from src.models.dummy_adapter import DummyKANAdapter
from src.quant.lut_builder import build_lut_for_edges
from src.quant.lut_io import load_lut_npz, save_lut_npz
from src.utils.parse_config import load_and_validate


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_lut_roundtrip.py <config.yaml>")
        raise SystemExit(2)

    cfg, _ = load_and_validate(Path(sys.argv[1]))

    adapter = DummyKANAdapter(seed=cfg.runtime.seed)
    edges = adapter.extract_edges()

    q = cfg.converter.quant
    if q.dtype == "uint8":
        qmin, qmax, scheme, zp = 0, 255, "asymmetric", None
    else:
        if q.scheme == "symmetric":
            qmin, qmax, scheme, zp = -127, 127, "symmetric", 0
        else:
            qmin, qmax, scheme, zp = -127, 127, "asymmetric", None

    art = build_lut_for_edges(
        edges=edges,
        L=cfg.converter.build_lut.L,
        interp=cfg.converter.interp.mode,
        oob_mode=cfg.converter.oob_policy.mode,
        y_range_method=cfg.converter.y_range.method,
        lower_pct=cfg.converter.y_range.lower_percentile,
        upper_pct=cfg.converter.y_range.upper_percentile,
        dtype=q.dtype,
        scheme=scheme,
        qmin=qmin,
        qmax=qmax,
        zero_point=zp if isinstance(q.zero_point, str) else q.zero_point,
        meta_dtype=cfg.converter.quant.meta_dtype,
    )

    N = 2048
    x = np.random.default_rng(cfg.runtime.seed).normal(size=(N, 2)).astype(np.float32)

    y0 = forward_lut_layer(x, art, edges)

    tmp = Path("outputs") / "tmp_lut_roundtrip.npz"
    save_lut_npz(tmp, art)
    art2 = load_lut_npz(tmp)

    y1 = forward_lut_layer(x, art2, edges)

    diff = np.max(np.abs(y1 - y0))
    print("Round-trip max abs diff:", float(diff))

    # npz should preserve exact arrays; tolerate only numerical dtype casts if any
    if diff != 0.0:
        raise SystemExit(1)

    print("OK")


if __name__ == "__main__":
    main()
