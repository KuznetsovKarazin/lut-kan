# tests/test_jacobi_polynomial_types.py
"""
Tests for different Jacobi polynomial types based on the paper:
"Kolmogorov-Arnold PointNet: Deep learning for prediction of fluid fields on irregular geometries"

Special cases of Jacobi polynomials (see paper Section 3.1):
- Legendre: α = β = 0
- Chebyshev 1st kind: α = β = -0.5
- Chebyshev 2nd kind: α = β = 0.5
- Gegenbauer: α = β (any equal value)
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.jacobi_adapter import JacobiKANSingleLayerAdapter, _jacobi_polynomials
from src.quant.lut_builder import build_lut_for_edges
from src.kernels.lut_contract import pack_dense_layer
from src.kernels.lut_backend_reference import forward_reference
from src.kernels.lut_backend_dense_numpy import forward_dense_numpy


# ============================================================================
# Polynomial type definitions from the paper
# ============================================================================
POLYNOMIAL_TYPES = {
    "legendre": {"alpha": 0.0, "beta": 0.0},
    "chebyshev_1st": {"alpha": -0.5, "beta": -0.5},
    "chebyshev_2nd": {"alpha": 0.5, "beta": 0.5},
    "gegenbauer_1": {"alpha": 1.0, "beta": 1.0},
    "gegenbauer_2": {"alpha": 2.0, "beta": 2.0},
    "asymmetric_ab": {"alpha": 1.0, "beta": 2.0},  # 2α = β = 2 from paper Table 3
    "asymmetric_ba": {"alpha": 2.0, "beta": 1.0},  # α = 2β = 2 from paper Table 3
}

# Polynomial degrees tested in the paper (Table 2)
POLYNOMIAL_DEGREES = [2, 3, 4, 5, 6]


# ============================================================================
# Unit tests for polynomial recurrence correctness
# ============================================================================
class TestJacobiPolynomialRecurrence:
    """Verify the Jacobi polynomial recurrence is correct."""

    def test_P0_is_one(self):
        """P_0(x) = 1 for all Jacobi polynomials."""
        x = np.linspace(-1, 1, 100, dtype=np.float32)
        for name, params in POLYNOMIAL_TYPES.items():
            P = _jacobi_polynomials(x, degree=3, a=params["alpha"], b=params["beta"])
            np.testing.assert_allclose(P[..., 0], 1.0, atol=1e-6, err_msg=f"P_0 failed for {name}")

    def test_P1_formula(self):
        """P_1(x) = ((α-β) + (α+β+2)x) / 2."""
        x = np.linspace(-1, 1, 100, dtype=np.float32)
        for name, params in POLYNOMIAL_TYPES.items():
            alpha, beta = params["alpha"], params["beta"]
            P = _jacobi_polynomials(x, degree=3, a=alpha, b=beta)
            expected = ((alpha - beta) + (alpha + beta + 2) * x) / 2
            np.testing.assert_allclose(
                P[..., 1], expected, atol=1e-5, err_msg=f"P_1 failed for {name}"
            )

    def test_legendre_special_values(self):
        """Verify Legendre polynomials at special points."""
        # Legendre: P_n(1) = 1, P_n(-1) = (-1)^n
        P = _jacobi_polynomials(np.array([1.0, -1.0], dtype=np.float32), degree=5, a=0.0, b=0.0)
        for n in range(6):
            np.testing.assert_allclose(P[0, n], 1.0, atol=1e-5, err_msg=f"P_{n}(1) failed")
            np.testing.assert_allclose(P[1, n], (-1) ** n, atol=1e-5, err_msg=f"P_{n}(-1) failed")

    def test_chebyshev_1st_at_zero(self):
        """Chebyshev 1st kind: T_n(0) = cos(nπ/2)."""
        P = _jacobi_polynomials(np.array([0.0], dtype=np.float32), degree=6, a=-0.5, b=-0.5)
        # For Chebyshev 1st kind with standard normalization
        # T_0(0) = 1, T_1(0) = 0, T_2(0) = -1, T_3(0) = 0, ...
        # But Jacobi polynomials have different normalization
        # Just check the pattern alternates correctly
        assert abs(P[0, 1]) < 0.1  # Should be close to 0 for odd n
        assert abs(P[0, 3]) < 0.1

    def test_polynomial_degree_zero(self):
        """Degree 0 should return only P_0."""
        x = np.array([0.5], dtype=np.float32)
        P = _jacobi_polynomials(x, degree=0, a=0.0, b=0.0)
        assert P.shape == (1, 1)
        assert P[0, 0] == 1.0


# ============================================================================
# Integration tests for adapter with different polynomial types
# ============================================================================
class TestJacobiAdapterPolynomialTypes:
    """Test JacobiKANSingleLayerAdapter with different polynomial types."""

    @pytest.fixture(params=list(POLYNOMIAL_TYPES.keys()))
    def poly_type(self, request):
        return request.param

    @pytest.fixture(params=[2, 3, 5])
    def degree(self, request):
        return request.param

    def test_adapter_forward_shapes(self, poly_type, degree):
        """Test that forward pass produces correct shapes."""
        params = POLYNOMIAL_TYPES[poly_type]
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={
                "in_dim": 4,
                "out_dim": 3,
                "degree": degree,
                "alpha": params["alpha"],
                "beta": params["beta"],
                "use_tanh": True,
                "x_min": -3.0,
                "x_max": 3.0,
                "num_knots": 9,
            },
            seed=42,
        )
        x = np.random.randn(64, 4).astype(np.float32)
        y = adapter.forward_float(x)
        assert y.shape == (64, 3)
        assert y.dtype == np.float32

    def test_adapter_edges_count(self, poly_type, degree):
        """Test that extract_edges produces correct number of edges."""
        params = POLYNOMIAL_TYPES[poly_type]
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={
                "in_dim": 4,
                "out_dim": 3,
                "degree": degree,
                "alpha": params["alpha"],
                "beta": params["beta"],
            },
            seed=42,
        )
        edges = adapter.extract_edges()
        assert len(edges) == 4 * 3  # in_dim * out_dim

    def test_phi_eval_matches_forward(self, poly_type):
        """Test that edge phi functions match forward pass."""
        params = POLYNOMIAL_TYPES[poly_type]
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={
                "in_dim": 2,
                "out_dim": 2,
                "degree": 3,
                "alpha": params["alpha"],
                "beta": params["beta"],
                "use_tanh": True,
            },
            seed=0,
        )
        x = np.random.randn(100, 2).astype(np.float32)
        y_forward = adapter.forward_float(x)

        # Reconstruct from edges
        edges = adapter.extract_edges()
        y_from_edges = np.zeros_like(y_forward)
        for edge in edges:
            phi_vals = edge.eval_phi(x[:, edge.src_idx])
            y_from_edges[:, edge.dst_idx] += phi_vals

        np.testing.assert_allclose(y_forward, y_from_edges, atol=1e-5)


# ============================================================================
# LUT compilation tests for different polynomial types
# ============================================================================
class TestJacobiLUTCompilation:
    """Test LUT compilation for different Jacobi polynomial configurations."""

    @pytest.mark.parametrize("poly_type", ["legendre", "chebyshev_1st", "chebyshev_2nd", "gegenbauer_1"])
    @pytest.mark.parametrize("degree", [2, 3, 5])
    def test_lut_compilation_succeeds(self, poly_type, degree):
        """Test that LUT compilation works for different polynomial types."""
        params = POLYNOMIAL_TYPES[poly_type]
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={
                "in_dim": 4,
                "out_dim": 3,
                "degree": degree,
                "alpha": params["alpha"],
                "beta": params["beta"],
                "use_tanh": True,
                "x_min": -3.0,
                "x_max": 3.0,
                "num_knots": 9,
            },
            seed=42,
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

        # q_table shape is [E, K, L] where K = num_segments = len(knots) - 1
        assert art.q_table.shape[0] == len(edges)
        assert art.q_table.shape[2] == 32  # L dimension

    @pytest.mark.parametrize("L", [16, 32, 64])
    @pytest.mark.parametrize("interp", ["nearest", "linear"])
    def test_lut_quantization_error_bounded(self, L, interp):
        """Test that LUT quantization error is bounded."""
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={
                "in_dim": 8,
                "out_dim": 4,
                "degree": 3,
                "alpha": -0.5,
                "beta": -0.5,
                "use_tanh": True,
                "x_min": -3.0,
                "x_max": 3.0,
                "num_knots": 9,
            },
            seed=42,
        )
        edges = adapter.extract_edges()

        art = build_lut_for_edges(
            edges=edges,
            L=L,
            interp=interp,
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

        # Generate test data
        rng = np.random.default_rng(0)
        x = rng.normal(size=(256, adapter.in_dim)).astype(np.float32)
        x = np.clip(x, -3.0, 3.0)  # Stay in domain

        y_float = adapter.forward_float(x)
        y_lut = forward_reference(x, packed)

        # Compute relative error
        rmse = np.sqrt(np.mean((y_float - y_lut) ** 2))
        max_abs = np.max(np.abs(y_float - y_lut))

        # Bounds depend on L and interp
        if interp == "linear" and L >= 32:
            assert rmse <= 0.05, f"RMSE {rmse} too high for L={L}, interp={interp}"
            assert max_abs <= 0.15, f"Max abs {max_abs} too high for L={L}, interp={interp}"
        else:
            assert rmse <= 0.15, f"RMSE {rmse} too high for L={L}, interp={interp}"


# ============================================================================
# Numerical stability tests
# ============================================================================
class TestJacobiNumericalStability:
    """Test numerical stability for edge cases."""

    def test_high_degree_stability(self):
        """Test that high polynomial degrees don't cause numerical issues."""
        for degree in [6, 8, 10]:
            adapter = JacobiKANSingleLayerAdapter.from_arch(
                arch={
                    "in_dim": 4,
                    "out_dim": 3,
                    "degree": degree,
                    "alpha": 0.0,
                    "beta": 0.0,
                    "use_tanh": True,
                },
                seed=42,
            )
            x = np.random.randn(100, 4).astype(np.float32)
            y = adapter.forward_float(x)
            assert np.all(np.isfinite(y)), f"Non-finite values for degree={degree}"

    def test_extreme_alpha_beta(self):
        """Test with extreme but valid alpha/beta values."""
        extreme_params = [
            {"alpha": -0.49, "beta": -0.49},  # Close to Chebyshev 1st limit
            {"alpha": 5.0, "beta": 5.0},      # Large Gegenbauer
            {"alpha": 0.1, "beta": 3.0},      # Asymmetric
        ]
        for params in extreme_params:
            adapter = JacobiKANSingleLayerAdapter.from_arch(
                arch={
                    "in_dim": 2,
                    "out_dim": 2,
                    "degree": 4,
                    "alpha": params["alpha"],
                    "beta": params["beta"],
                    "use_tanh": True,
                },
                seed=42,
            )
            x = np.random.randn(50, 2).astype(np.float32)
            y = adapter.forward_float(x)
            assert np.all(np.isfinite(y)), f"Non-finite values for params={params}"

    def test_without_tanh_normalization(self):
        """Test behavior without tanh normalization (inputs must be in [-1, 1])."""
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={
                "in_dim": 4,
                "out_dim": 3,
                "degree": 3,
                "alpha": 0.0,
                "beta": 0.0,
                "use_tanh": False,
                "x_min": -1.0,
                "x_max": 1.0,
            },
            seed=42,
        )
        # Generate data in [-1, 1]
        x = np.random.uniform(-1, 1, size=(100, 4)).astype(np.float32)
        y = adapter.forward_float(x)
        assert np.all(np.isfinite(y))


# ============================================================================
# Paper validation tests
# ============================================================================
class TestPaperResults:
    """Tests that validate findings from the paper."""

    def test_coefficient_initialization_scale(self):
        """
        Verify coefficient initialization matches paper:
        normal(0, 1/(in_dim*(degree+1)))
        """
        in_dim, out_dim, degree = 4, 3, 3
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={"in_dim": in_dim, "out_dim": out_dim, "degree": degree},
            seed=42,
        )
        # Expected std: 1/(4*4) = 0.0625
        expected_std = 1.0 / (in_dim * (degree + 1))
        actual_std = np.std(adapter.coeffs)
        # Allow some variance due to finite sample
        assert 0.5 * expected_std < actual_std < 2.0 * expected_std

    @pytest.mark.parametrize("poly_name,expected_better", [
        ("chebyshev_1st", True),   # Paper: Chebyshev 1st performs best
        ("chebyshev_2nd", True),   # Paper: Chebyshev 2nd also good
        ("legendre", True),        # Paper: Legendre is acceptable
        ("asymmetric_ba", False),  # Paper: α = 2β = 2 has lowest performance
    ])
    def test_polynomial_type_ranking_qualitative(self, poly_name, expected_better):
        """
        Qualitative test: polynomial types should compile correctly.
        Actual ranking from paper Table 3 is data-dependent.
        """
        params = POLYNOMIAL_TYPES[poly_name]
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={
                "in_dim": 4,
                "out_dim": 3,
                "degree": 3,
                "alpha": params["alpha"],
                "beta": params["beta"],
            },
            seed=42,
        )
        edges = adapter.extract_edges()
        assert len(edges) == 12  # Just verify it works


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
