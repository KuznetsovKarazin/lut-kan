from __future__ import annotations

import numpy as np

from src.kernels.lut_math import lut_interp_indices
from src.models.dummy_adapter import DummyKANAdapter
from src.quant.lut_builder import build_lut_for_edges, build_segment_grid


def test_segment_grid_includes_both_endpoints_uniform() -> None:
    knots = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    L = 5

    grid = build_segment_grid(knots, L)

    assert grid.shape == (2, L)
    np.testing.assert_array_equal(grid[:, 0], knots[:-1])
    np.testing.assert_array_equal(grid[:, -1], knots[1:])
    np.testing.assert_allclose(
        grid[0],
        np.linspace(-2.0, 0.0, L, endpoint=True, dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        grid[1],
        np.linspace(0.0, 2.0, L, endpoint=True, dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_segment_grid_includes_both_endpoints_nonuniform() -> None:
    knots = np.array([-3.0, -0.5, 0.25, 4.0], dtype=np.float32)
    L = 7

    grid = build_segment_grid(knots, L)

    np.testing.assert_array_equal(grid[:, 0], knots[:-1])
    np.testing.assert_array_equal(grid[:, -1], knots[1:])
    for k in range(len(knots) - 1):
        expected = np.linspace(knots[k], knots[k + 1], L, endpoint=True, dtype=np.float32)
        np.testing.assert_allclose(grid[k], expected, rtol=1e-6, atol=1e-7)


def test_runtime_interpolation_coordinate_matches_sampling_grid() -> None:
    """The builder grid and pos=u*(L-1) runtime contract must be identical."""
    a = np.float32(-1.75)
    b = np.float32(2.25)
    L = 17
    grid = build_segment_grid(np.array([a, b], dtype=np.float32), L)[0]

    # Test nodes and midpoints, not only the exact LUT nodes.
    u_nodes = np.arange(L, dtype=np.float32) / np.float32(L - 1)
    u_mid = (np.arange(L - 1, dtype=np.float32) + np.float32(0.5)) / np.float32(L - 1)
    u = np.concatenate([u_nodes, u_mid])

    r0, r1, w = lut_interp_indices(u, L)
    x_from_lut_coordinate = (np.float32(1.0) - w) * grid[r0] + w * grid[r1]
    x_expected = a + u * (b - a)

    np.testing.assert_allclose(x_from_lut_coordinate, x_expected, rtol=1e-6, atol=2e-7)


def test_new_builder_marks_endpoint_inclusive_format_v2() -> None:
    adapter = DummyKANAdapter(in_dim=2, out_dim=2, num_knots=7, x_min=-2.0, x_max=2.0, seed=0)
    edges = adapter.extract_edges()

    art = build_lut_for_edges(
        edges=edges,
        L=16,
        interp="linear",
        y_range_method="minmax",
        lower_pct=0.1,
        upper_pct=99.9,
        dtype="uint8",
        scheme="asymmetric",
        qmin=0,
        qmax=255,
        meta_dtype="float32",
        value_representation="phi",
        oob_behavior="clip",
        boundary_mode="half_open",
    )

    assert art.format_version == 2
    assert art.sample_grid == "endpoint_inclusive"
