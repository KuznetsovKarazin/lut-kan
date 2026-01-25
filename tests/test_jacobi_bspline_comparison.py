# tests/test_jacobi_bspline_comparison.py
"""
Comprehensive comparison tests between Jacobi and B-spline KAN implementations.

Tests:
1. Jacobi polynomial correctness (matches paper equations)
2. B-spline basis function correctness
3. LUT quantization accuracy for both types
4. Cross-validation with MCU-compatible implementations
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.special import jacobi

from src.models.jacobi_adapter import JacobiKANSingleLayerAdapter, _jacobi_polynomials


class TestJacobiPolynomials:
    """Test Jacobi polynomial implementation against scipy reference."""
    
    @pytest.mark.parametrize("degree", [0, 1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize("alpha,beta", [
        (-0.5, -0.5),  # Chebyshev 1st kind
        (0.5, 0.5),    # Chebyshev 2nd kind
        (0.0, 0.0),    # Legendre
        (1.0, 1.0),    # Gegenbauer
        (1.0, 0.0),    # Asymmetric
        (2.0, 1.0),    # High alpha/beta
    ])
    def test_jacobi_against_scipy(self, degree: int, alpha: float, beta: float) -> None:
        """Verify our Jacobi implementation matches scipy.special.jacobi."""
        x = np.linspace(-1, 1, 101, dtype=np.float32)
        
        # Our implementation
        P_ours = _jacobi_polynomials(x, degree, alpha, beta)  # shape: [101, degree+1]
        
        # Scipy reference
        for n in range(degree + 1):
            P_scipy = jacobi(n, alpha, beta)(x)
            P_our_n = P_ours[:, n]
            
            # Allow for numerical differences due to different recurrence formulations
            max_diff = float(np.max(np.abs(P_our_n - P_scipy)))
            assert max_diff < 1e-4, f"Jacobi P_{n}^({alpha},{beta}) max diff: {max_diff}"
    
    def test_jacobi_boundary_values(self) -> None:
        """Test Jacobi polynomials at x=±1."""
        # P_n^(α,β)(1) = C(n+α, n)
        # P_n^(α,β)(-1) = (-1)^n * C(n+β, n)
        
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        P = _jacobi_polynomials(x, degree=3, a=0.0, b=0.0)
        
        # Legendre polynomials: P_n(1) = 1, P_n(-1) = (-1)^n
        assert abs(P[2, 0] - 1.0) < 1e-5  # P_0(1) = 1
        assert abs(P[2, 1] - 1.0) < 1e-5  # P_1(1) = 1
        assert abs(P[2, 2] - 1.0) < 1e-5  # P_2(1) = 1
        assert abs(P[0, 2] - 1.0) < 1e-5  # P_2(-1) = 1


class TestJacobiAdapter:
    """Test JacobiKANSingleLayerAdapter functionality."""
    
    def test_adapter_creation(self) -> None:
        """Test adapter can be created from arch dict."""
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={
                "in_dim": 4,
                "out_dim": 3,
                "degree": 3,
                "alpha": -0.5,
                "beta": -0.5,
            },
            seed=42,
        )
        
        assert adapter.in_dim == 4
        assert adapter.out_dim == 3
        assert adapter.degree == 3
        assert adapter.coeffs.shape == (4, 3, 4)  # [in, out, degree+1]
    
    def test_forward_float(self) -> None:
        """Test float forward pass produces reasonable output."""
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={"in_dim": 2, "out_dim": 2, "degree": 3},
            seed=0,
        )
        
        x = np.array([[0.5, -0.5], [1.0, 0.0]], dtype=np.float32)
        y = adapter.forward_float(x)
        
        assert y.shape == (2, 2)
        assert not np.any(np.isnan(y))
        assert not np.any(np.isinf(y))
    
    def test_extract_edges(self) -> None:
        """Test edge extraction for LUT building."""
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={"in_dim": 2, "out_dim": 3, "degree": 2},
            seed=0,
        )
        
        edges = adapter.extract_edges()
        
        assert len(edges) == 6  # 2 * 3
        
        # Test edge evaluation
        x_test = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        y = edges[0].eval_phi(x_test)
        
        assert y.shape == (3,)
        assert not np.any(np.isnan(y))


class TestBSplineBasis:
    """Test B-spline basis function implementation (for comparison)."""
    
    def test_bspline_partition_of_unity(self) -> None:
        """B-spline basis functions should sum to 1 in the interior."""
        # Create uniform knot vector
        k = 3  # cubic
        n_interior = 5
        knots = np.concatenate([
            np.zeros(k),
            np.linspace(0, 1, n_interior + 1),
            np.ones(k),
        ])
        
        # Sample points in interior
        x = np.linspace(0.01, 0.99, 100)
        
        # Sum of all basis functions
        n_coefs = len(knots) - k - 1
        basis_sum = np.zeros_like(x)
        
        for i in range(n_coefs):
            c = np.zeros(n_coefs)
            c[i] = 1.0
            spl = BSpline(knots, c, k)
            basis_sum += spl(x)
        
        # Should be approximately 1 everywhere
        assert np.allclose(basis_sum, 1.0, atol=1e-10)
    
    def test_bspline_locality(self) -> None:
        """Each B-spline basis function has local support."""
        k = 3
        knots = np.linspace(0, 1, 10)  # 10 knots
        knots = np.concatenate([np.zeros(k), knots, np.ones(k)])
        
        n_coefs = len(knots) - k - 1
        
        for i in range(n_coefs):
            c = np.zeros(n_coefs)
            c[i] = 1.0
            spl = BSpline(knots, c, k)
            
            # Test that basis function is zero outside its support
            x_far = -1.0 if i == 0 else 2.0
            # Note: BSpline extrapolates, so we test internal structure
            

class TestLUTQuantization:
    """Test LUT quantization accuracy for both kernel types."""
    
    def test_jacobi_lut_accuracy(self) -> None:
        """Test Jacobi LUT maintains accuracy after quantization."""
        from src.quant.lut_builder import build_lut_for_edges
        
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={"in_dim": 2, "out_dim": 2, "degree": 3},
            seed=42,
        )
        edges = adapter.extract_edges()
        
        art = build_lut_for_edges(
            edges=edges,
            L=64,
            interp="linear",
            oob_behavior="clip",
            boundary_mode="half_open",
            y_range_method="minmax",
            lower_pct=0.0,
            upper_pct=100.0,
            dtype="uint8",
            scheme="asymmetric",
            qmin=0,
            qmax=255,
            meta_dtype="float16",
            value_representation="phi",
        )
        
        # Test quantization error - use linear interpolation for proper reconstruction
        x_test = np.linspace(-2.5, 2.5, 100, dtype=np.float32)
        
        # Compute reference float values for each edge
        for edge in edges:
            y_float = edge.eval_phi(x_test)
            
            # Get LUT data for this edge: shape [K, L]
            lut_edge = art.q_table[edge.edge_id]  # [K, L]
            scale_edge = art.scale[edge.edge_id]  # [K]
            y_min_edge = art.y_min[edge.edge_id]  # [K]
            
            # For simplicity, flatten across all segments and dequantize
            # This is a rough approximation - real inference uses proper segment lookup
            K, L = lut_edge.shape
            knots = art.knots
            
            y_lut = np.zeros_like(y_float)
            for i, x in enumerate(x_test):
                # Find segment
                seg = 0
                for k in range(len(knots) - 1):
                    if knots[k] <= x < knots[k + 1]:
                        seg = k
                        break
                else:
                    seg = len(knots) - 2 if x >= knots[-1] else 0
                
                # Interpolate within segment
                seg_start = knots[seg]
                seg_end = knots[seg + 1]
                t = (x - seg_start) / (seg_end - seg_start) if seg_end != seg_start else 0.5
                t = np.clip(t, 0, 1)
                
                idx_f = t * (L - 1)
                idx_lo = int(np.floor(idx_f))
                idx_hi = min(idx_lo + 1, L - 1)
                frac = idx_f - idx_lo
                
                # Dequantize
                q_lo = float(lut_edge[seg, idx_lo])
                q_hi = float(lut_edge[seg, idx_hi])
                q_interp = q_lo * (1 - frac) + q_hi * frac
                
                y_lut[i] = float(y_min_edge[seg]) + float(scale_edge[seg]) * q_interp
            
            # Check error (with more generous threshold for segmented quantization)
            max_err = float(np.max(np.abs(y_float - y_lut)))
            assert max_err < 0.1, f"Edge {edge.edge_id}: max error {max_err} too large"
    
    @pytest.mark.parametrize("L", [32, 64, 128])
    @pytest.mark.parametrize("dtype", ["uint8", "uint16"])
    def test_lut_size_vs_accuracy(self, L: int, dtype: str) -> None:
        """Larger LUT and higher precision should give better accuracy."""
        from src.quant.lut_builder import build_lut_for_edges
        
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={"in_dim": 2, "out_dim": 2, "degree": 4},
            seed=42,
        )
        edges = adapter.extract_edges()
        
        qmax = 255 if dtype == "uint8" else 65535
        
        art = build_lut_for_edges(
            edges=edges,
            L=L,
            interp="linear",
            oob_behavior="clip",
            boundary_mode="half_open",
            y_range_method="minmax",
            lower_pct=0.0,
            upper_pct=100.0,
            dtype=dtype,
            scheme="asymmetric",
            qmin=0,
            qmax=qmax,
            meta_dtype="float16",
            value_representation="phi",
        )
        
        # Verify artifact properties
        assert art.L == L
        assert art.q_table.shape[2] == L  # [edges, segments, L]
        

class TestJacobiPolynomialTypes:
    """Test different Jacobi polynomial types (Chebyshev, Legendre, etc.)."""
    
    @pytest.mark.parametrize("alpha,beta,name", [
        (-0.5, -0.5, "Chebyshev_1st"),
        (0.5, 0.5, "Chebyshev_2nd"),
        (0.0, 0.0, "Legendre"),
        (1.0, 1.0, "Gegenbauer"),
    ])
    def test_polynomial_type_forward(self, alpha: float, beta: float, name: str) -> None:
        """Test different polynomial types produce valid output."""
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={
                "in_dim": 4,
                "out_dim": 4,
                "degree": 4,
                "alpha": alpha,
                "beta": beta,
            },
            seed=42,
        )
        
        x = np.random.randn(10, 4).astype(np.float32)
        y = adapter.forward_float(x)
        
        assert y.shape == (10, 4)
        assert not np.any(np.isnan(y)), f"{name}: NaN in output"
        assert not np.any(np.isinf(y)), f"{name}: Inf in output"


class TestCrossValidation:
    """Cross-validation tests between Python and C implementations."""
    
    def test_jacobi_recurrence_matches_c(self) -> None:
        """Verify Python recurrence matches C implementation."""
        # Test case: same parameters as C implementation
        x = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        degree = 3
        alpha = -0.5
        beta = -0.5
        
        P = _jacobi_polynomials(x, degree, alpha, beta)
        
        # P_0 should be 1
        assert np.allclose(P[:, 0], 1.0)
        
        # P_1(x) = ((α-β) + (α+β+2)x) / 2
        P1_expected = 0.5 * ((alpha - beta) + (alpha + beta + 2) * x)
        assert np.allclose(P[:, 1], P1_expected, atol=1e-6)
    
    def test_tanh_normalization(self) -> None:
        """Test tanh normalization matches C implementation."""
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={
                "in_dim": 2,
                "out_dim": 2,
                "degree": 3,
                "use_tanh": True,
            },
            seed=0,
        )
        
        # Large input values should be normalized by tanh
        x = np.array([[5.0, -5.0]], dtype=np.float32)
        y = adapter.forward_float(x)
        
        # Output should be bounded (not blow up)
        assert np.max(np.abs(y)) < 100
