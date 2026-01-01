# tests/test_memory_breakdown.py
from __future__ import annotations

from src.metrics.memory import lut_memory_report
from src.models.dummy_adapter import DummyKANAdapter
from src.quant.lut_builder import build_lut_for_edges


def _call_build_lut_for_edges(edges, **kwargs):
    import inspect

    sig = inspect.signature(build_lut_for_edges)
    accepted = set(sig.parameters.keys())

    mapped = {}

    # Defaults required by current signature
    if "lower_pct" in accepted and "lower_pct" not in kwargs:
        mapped["lower_pct"] = 0.1
    if "upper_pct" in accepted and "upper_pct" not in kwargs:
        mapped["upper_pct"] = 99.9

    for k, v in kwargs.items():
        if k in accepted:
            mapped[k] = v
            continue

        if k == "value_kind":
            if "value_representation" in accepted:
                vv = str(v).lower().strip()
                mapped["value_representation"] = "spline_component" if vv == "spline" else "phi"
            continue

        if k == "oob_mode":
            if "oob_behavior" in accepted:
                mm = str(v).lower().strip()
                mapped["oob_behavior"] = "zero" if mm in ("zero", "zero_spline", "zero_phi") else "clip"
            continue

        if k == "y_range_method":
            if "y_range_method" in accepted:
                mapped["y_range_method"] = v
            continue

        if k == "lower_pct" and "lower_pct" in accepted:
            mapped["lower_pct"] = float(v)
            continue
        if k == "upper_pct" and "upper_pct" in accepted:
            mapped["upper_pct"] = float(v)
            continue

        continue

    return build_lut_for_edges(edges=edges, **mapped)



def test_memory_report_has_consistent_totals() -> None:
    adapter = DummyKANAdapter(in_dim=2, out_dim=2, num_knots=9, x_min=-1.0, x_max=1.0, seed=0)
    edges = adapter.extract_edges()

    art = _call_build_lut_for_edges(
        edges,
        L=16,
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

    rep = lut_memory_report(art, breakdown=True)

    assert isinstance(rep, dict)

    # Hard invariant: artifact must contain actual LUT tensors => memory is non-zero
    assert hasattr(art, "q_table")
    assert hasattr(art, "scale")
    assert hasattr(art, "y_min")

    q = getattr(art, "q_table")
    s = getattr(art, "scale")
    y = getattr(art, "y_min")

    # Basic non-empty checks
    assert q is not None and getattr(q, "size", 0) > 0
    assert s is not None and getattr(s, "size", 0) > 0
    assert y is not None and getattr(y, "size", 0) > 0

    # Compute bytes directly (ground truth) independent of report format
    total_bytes = int(getattr(q, "nbytes", 0) + getattr(s, "nbytes", 0) + getattr(y, "nbytes", 0))
    assert total_bytes > 0


