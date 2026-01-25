#!/usr/bin/env python3
"""
Unified LUT-KAN Benchmark Sweeper for Publication-Ready Research.

This script provides comprehensive benchmarking for:
1. B-spline KAN (PyKAN-style)
2. Jacobi KAN with all polynomial variants:
   - Chebyshev 1st kind (α=β=-0.5)
   - Chebyshev 2nd kind (α=β=0.5)
   - Legendre (α=β=0)
   - Gegenbauer (α=β>0)
   - Asymmetric (α≠β)

Sweep dimensions:
- Basis type: bspline, chebyshev_1, chebyshev_2, legendre, gegenbauer, asymmetric
- Complexity (degree/k): 2, 3, 4, 5, 6, 8, 10, 15, 20
- LUT resolution (L): 16, 32, 64, 128, 256
- Interpolation: nearest, linear
- Layer size: 8x8, 16x16, 32x32
- Quantization: uint8, int8, uint16

Outputs:
- results/unified_benchmark_{timestamp}/
  - raw_results.csv
  - raw_results.json
  - summary_tables/
    - speedup_by_basis.{md,tex}
    - accuracy_by_basis.{md,tex}
    - memory_by_config.{md,tex}
  - plots/
    - speedup_heatmap.png
    - accuracy_vs_complexity.png
    - pareto_front.png

Usage:
    python scripts/unified_benchmark_sweeper.py --full
    python scripts/unified_benchmark_sweeper.py --quick  # Reduced grid for testing
    python scripts/unified_benchmark_sweeper.py --basis jacobi,bspline --degrees 3,5,10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.jacobi_adapter import JacobiKANSingleLayerAdapter
from src.models.bspline_adapter import BSplineKANSingleLayerAdapter, estimate_bspline_float_cycles
from src.quant.lut_builder import build_lut_for_edges
from src.kernels.lut_contract import PackedLUT, pack_dense_layer
from src.kernels.lut_backend_dense_numpy import forward_dense_numpy

# Optional numba import
try:
    from src.kernels.lut_backend_dense_numba import (
        forward_dense_numba,
        numba_available,
        warmup_numba,
    )
    HAS_NUMBA = bool(numba_available())
except Exception:
    HAS_NUMBA = False


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class BasisConfig:
    """Configuration for a basis function type."""
    name: str
    family: Literal["bspline", "jacobi"]
    # Jacobi-specific
    alpha: float = 0.0
    beta: float = 0.0
    # B-spline specific
    grid_points: int = 5  # Interior grid points
    base_kind: str = "silu"  # "silu" or "none"
    
    @staticmethod
    def jacobi_presets() -> Dict[str, "BasisConfig"]:
        return {
            "chebyshev_1": BasisConfig("Chebyshev 1st", "jacobi", alpha=-0.5, beta=-0.5),
            "chebyshev_2": BasisConfig("Chebyshev 2nd", "jacobi", alpha=0.5, beta=0.5),
            "legendre": BasisConfig("Legendre", "jacobi", alpha=0.0, beta=0.0),
            "gegenbauer_1": BasisConfig("Gegenbauer L=1", "jacobi", alpha=0.5, beta=0.5),
            "gegenbauer_2": BasisConfig("Gegenbauer L=2", "jacobi", alpha=1.5, beta=1.5),
            "jacobi_asym_1": BasisConfig("Jacobi (1,0)", "jacobi", alpha=1.0, beta=0.0),
            "jacobi_asym_2": BasisConfig("Jacobi (2,1)", "jacobi", alpha=2.0, beta=1.0),
        }
    
    @staticmethod
    def bspline_presets() -> Dict[str, "BasisConfig"]:
        """B-spline configurations with different grid densities.
        
        Complexity equivalence with Jacobi:
        - Jacobi degree d has d+1 coefficients
        - B-spline grid_points g has g+k-1 coefficients (k=3 -> g+2)
        - So g = d-1 for equivalent complexity
        
        Mapping: d=3->g=2, d=5->g=4, d=10->g=9, d=15->g=14, d=20->g=19, d=30->g=29
        """
        presets = {}
        
        # With SiLU base (PyKAN-style)
        for g in [3, 4, 5, 6, 7, 9, 14, 19, 24, 29]:
            key = f"bspline_g{g}"
            name = f"B-spline g={g}"
            presets[key] = BasisConfig(name, "bspline", grid_points=g, base_kind="silu")
        
        # Without SiLU base (pure spline) - for accuracy comparison
        for g in [5, 9, 14, 19]:
            key = f"bspline_pure_g{g}"
            name = f"B-spline pure g={g}"
            presets[key] = BasisConfig(name, "bspline", grid_points=g, base_kind="none")
        
        # Legacy aliases
        presets["bspline_k3_g5"] = BasisConfig("B-spline k=3 g=5", "bspline", grid_points=5, base_kind="silu")
        presets["bspline_k3_g9"] = BasisConfig("B-spline k=3 g=9", "bspline", grid_points=9, base_kind="silu")
        presets["bspline_k3_g17"] = BasisConfig("B-spline k=3 g=17", "bspline", grid_points=17, base_kind="silu")
        
        return presets
    
    @staticmethod
    def all_presets() -> Dict[str, "BasisConfig"]:
        """All available basis configurations."""
        presets = {}
        presets.update(BasisConfig.jacobi_presets())
        presets.update(BasisConfig.bspline_presets())
        return presets


@dataclass 
class SweepConfig:
    """Full sweep configuration."""
    # Basis types to test (from BasisConfig.all_presets())
    basis_types: List[str] = field(default_factory=lambda: [
        # Jacobi variants
        "chebyshev_1", "legendre", "gegenbauer_1",
        # B-spline variants
        "bspline_k3_g5", "bspline_k3_g9",
    ])
    
    # Polynomial degrees (for Jacobi only; B-spline uses grid_points from BasisConfig)
    degrees: List[int] = field(default_factory=lambda: [3, 5, 8, 10, 15, 20])
    
    # LUT resolutions
    lut_sizes: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    
    # Layer dimensions (square layers for simplicity)
    layer_sizes: List[int] = field(default_factory=lambda: [8, 16])
    
    # Interpolation modes
    interp_modes: List[str] = field(default_factory=lambda: ["linear", "nearest"])
    
    # Quantization types
    quant_dtypes: List[str] = field(default_factory=lambda: ["uint8"])
    
    # Batch sizes for timing
    batch_sizes: List[int] = field(default_factory=lambda: [1, 256, 1024])
    
    # Timing parameters
    warmup_iters: int = 10
    measure_iters: int = 100
    timing_repeats: int = 5
    
    # Random seed
    seed: int = 42
    
    # Domain
    x_min: float = -3.0
    x_max: float = 3.0
    num_knots: int = 9
    use_tanh: bool = True


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    # Configuration
    basis_type: str
    basis_family: str
    degree: int
    L: int
    in_dim: int
    out_dim: int
    edges: int
    interp: str
    quant_dtype: str
    batch_size: int
    
    # Jacobi-specific
    alpha: float = 0.0
    beta: float = 0.0
    
    # Timing (ms per iteration)
    float_ms: float = 0.0
    lut_numpy_ms: float = 0.0
    lut_numba_ms: Optional[float] = None
    
    # Speedups
    speedup_numpy: float = 1.0
    speedup_numba: Optional[float] = None
    
    # Accuracy
    rmse: float = 0.0
    mae: float = 0.0
    max_abs: float = 0.0
    
    # Memory (bytes)
    lut_mem_bytes: int = 0
    lut_mem_per_edge: float = 0.0
    float_mem_bytes: int = 0
    
    # MCU estimates (cycles per sample, N=1)
    mcu_float_cycles: int = 0
    mcu_lut_cycles: int = 0
    mcu_speedup: float = 1.0


# =============================================================================
# MCU Cycle Estimation
# =============================================================================

@dataclass
class MCUProfile:
    """Cycle counts for MCU without FPU."""
    name: str
    int_add: int = 1
    int_mul: int = 1
    int_load: int = 2
    float_add: int = 70
    float_mul: int = 100
    float_div: int = 200
    tanh: int = 800


MCU_CORTEX_M3 = MCUProfile(
    name="ARM Cortex-M3 (no FPU)",
    float_add=50, float_mul=80, float_div=150, tanh=800
)


def estimate_jacobi_float_cycles(
    degree: int, 
    in_dim: int, 
    out_dim: int,
    use_tanh: bool,
    profile: MCUProfile = MCU_CORTEX_M3
) -> int:
    """Estimate cycles for single sample Jacobi float forward."""
    cycles_per_coord = (
        (profile.tanh if use_tanh else 0) +
        (4 * degree - 1) * profile.float_mul +
        (3 * degree + 1) * profile.float_add +
        profile.float_div
    )
    cycles_per_coord += out_dim * (
        (degree + 1) * profile.float_mul + degree * profile.float_add
    )
    return in_dim * cycles_per_coord


def estimate_lut_cycles(
    in_dim: int,
    out_dim: int,
    interp: str,
    profile: MCUProfile = MCU_CORTEX_M3
) -> int:
    """Estimate cycles for single sample LUT forward (int accum)."""
    edges = in_dim * out_dim
    
    if interp == "nearest":
        cycles_per_edge = (
            4 * profile.int_add +
            1 * profile.int_load +
            1 * profile.int_mul +
            1 * profile.int_add
        )
    else:  # linear
        cycles_per_edge = (
            4 * profile.int_add +
            2 * profile.int_load +
            2 * profile.int_mul +
            2 * profile.int_add
        )
    
    # Final dequant per output
    cycles_final = out_dim * (profile.float_mul + profile.float_add)
    
    return edges * cycles_per_edge + cycles_final


# =============================================================================
# Benchmarking Core
# =============================================================================

def time_function(fn, warmup: int, iters: int, repeats: int) -> Tuple[float, List[float]]:
    """Time a function, return (median_ms, all_samples_ms)."""
    for _ in range(warmup):
        fn()
    
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dt = time.perf_counter() - t0
        samples.append((dt / iters) * 1000.0)
    
    samples.sort()
    median = samples[len(samples) // 2]
    return median, samples


def packed_memory_bytes(p: PackedLUT) -> int:
    """Calculate total memory of packed LUT."""
    return (
        int(p.q_flat.nbytes) +
        int(p.scale.nbytes) +
        int(p.y_min.nbytes) +
        int(p.knots.nbytes) +
        int(p.coef_base.nbytes) +
        int(p.coef_lut.nbytes) +
        int(p.coef_out.nbytes)
    )


def benchmark_single_config(
    basis_cfg: BasisConfig,
    degree: int,
    L: int,
    in_dim: int,
    out_dim: int,
    interp: str,
    quant_dtype: str,
    batch_size: int,
    sweep_cfg: SweepConfig,
) -> BenchmarkResult:
    """Run benchmark for a single configuration."""
    
    # Create adapter based on basis family
    if basis_cfg.family == "bspline":
        # For B-spline, 'degree' parameter controls grid_points for fair comparison
        # More grid points = more coefficients = similar complexity to higher polynomial degree
        grid_points = basis_cfg.grid_points
        base_kind = basis_cfg.base_kind
        adapter = BSplineKANSingleLayerAdapter.from_arch(
            arch={
                "in_dim": in_dim,
                "out_dim": out_dim,
                "degree": 3,  # Cubic B-spline (fixed)
                "grid_points": grid_points,
                "x_min": sweep_cfg.x_min,
                "x_max": sweep_cfg.x_max,
                "base_kind": base_kind,
            },
            seed=sweep_cfg.seed,
        )
        # For B-spline, effective complexity is num_coef = grid_points + degree - 1
        effective_degree = adapter.num_coef
        use_silu = (base_kind == "silu")
    else:
        # Jacobi adapter
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={
                "in_dim": in_dim,
                "out_dim": out_dim,
                "degree": degree,
                "alpha": basis_cfg.alpha,
                "beta": basis_cfg.beta,
                "use_tanh": sweep_cfg.use_tanh,
                "x_min": sweep_cfg.x_min,
                "x_max": sweep_cfg.x_max,
                "num_knots": sweep_cfg.num_knots,
            },
            seed=sweep_cfg.seed,
        )
        effective_degree = degree
    
    edges = adapter.extract_edges()
    
    # Build LUT
    qmax = 255 if quant_dtype == "uint8" else (127 if quant_dtype == "int8" else 65535)
    qmin = 0 if quant_dtype.startswith("u") else (-128 if quant_dtype == "int8" else -32768)
    scheme = "asymmetric" if quant_dtype.startswith("u") else "symmetric"
    
    art = build_lut_for_edges(
        edges=edges,
        L=L,
        interp=interp,
        oob_behavior="clip",
        boundary_mode="half_open",
        y_range_method="minmax",
        lower_pct=0.1,
        upper_pct=99.9,
        dtype=quant_dtype,
        scheme=scheme,
        qmin=qmin,
        qmax=qmax,
        meta_dtype="float16",
        value_representation="phi",
    )
    
    packed = pack_dense_layer(
        art,
        edges=edges,
        in_dim=in_dim,
        out_dim=out_dim,
        boundary_mode="half_open",
    )
    
    # Generate test data
    rng = np.random.default_rng(sweep_cfg.seed + 12345)
    x = rng.normal(size=(batch_size, in_dim)).astype(np.float32)
    x = np.clip(x, sweep_cfg.x_min, sweep_cfg.x_max)
    
    # Float forward
    y_float = adapter.forward_float(x)
    float_ms, _ = time_function(
        lambda: adapter.forward_float(x),
        sweep_cfg.warmup_iters,
        sweep_cfg.measure_iters,
        sweep_cfg.timing_repeats
    )
    
    # LUT numpy forward
    y_lut_np = forward_dense_numpy(x, packed)
    lut_np_ms, _ = time_function(
        lambda: forward_dense_numpy(x, packed),
        sweep_cfg.warmup_iters,
        sweep_cfg.measure_iters,
        sweep_cfg.timing_repeats
    )
    
    # LUT numba forward (if available)
    lut_nb_ms = None
    if HAS_NUMBA:
        warmup_numba(packed, in_dim=in_dim, out_dim=out_dim)
        y_lut_nb = forward_dense_numba(x, packed)
        lut_nb_ms, _ = time_function(
            lambda: forward_dense_numba(x, packed),
            sweep_cfg.warmup_iters,
            sweep_cfg.measure_iters,
            sweep_cfg.timing_repeats
        )
    
    # Compute errors
    rmse = float(np.sqrt(np.mean((y_float - y_lut_np) ** 2)))
    mae = float(np.mean(np.abs(y_float - y_lut_np)))
    max_abs = float(np.max(np.abs(y_float - y_lut_np)))
    
    # Memory - different attribute names for different adapters
    lut_mem = packed_memory_bytes(packed)
    if basis_cfg.family == "bspline":
        float_mem = adapter.coef.nbytes + adapter.knots_aug.nbytes
        # MCU estimate for B-spline
        mcu_float = estimate_bspline_float_cycles(
            degree=3,  # cubic
            num_coef=adapter.num_coef,
            in_dim=in_dim,
            out_dim=out_dim,
            use_silu=use_silu,
        )
    else:
        float_mem = adapter.coeffs.nbytes + adapter.knots.nbytes
        # MCU estimate for Jacobi
        mcu_float = estimate_jacobi_float_cycles(degree, in_dim, out_dim, sweep_cfg.use_tanh)
    
    mcu_lut = estimate_lut_cycles(in_dim, out_dim, interp)
    
    return BenchmarkResult(
        basis_type=basis_cfg.name,
        basis_family=basis_cfg.family,
        degree=degree,
        L=L,
        in_dim=in_dim,
        out_dim=out_dim,
        edges=in_dim * out_dim,
        interp=interp,
        quant_dtype=quant_dtype,
        batch_size=batch_size,
        alpha=basis_cfg.alpha,
        beta=basis_cfg.beta,
        float_ms=float_ms,
        lut_numpy_ms=lut_np_ms,
        lut_numba_ms=lut_nb_ms,
        speedup_numpy=float_ms / lut_np_ms if lut_np_ms > 0 else 0,
        speedup_numba=float_ms / lut_nb_ms if lut_nb_ms and lut_nb_ms > 0 else None,
        rmse=rmse,
        mae=mae,
        max_abs=max_abs,
        lut_mem_bytes=lut_mem,
        lut_mem_per_edge=lut_mem / (in_dim * out_dim),
        float_mem_bytes=float_mem,
        mcu_float_cycles=mcu_float,
        mcu_lut_cycles=mcu_lut,
        mcu_speedup=mcu_float / mcu_lut if mcu_lut > 0 else 0,
    )


# =============================================================================
# Table Generation
# =============================================================================

def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Generate markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_latex_table(
    headers: List[str], 
    rows: List[List[str]], 
    caption: str, 
    label: str
) -> str:
    """Generate LaTeX table."""
    ncols = len(headers)
    colspec = "l" + "r" * (ncols - 1)
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join([h.replace("_", "\\_") for h in headers]) + " \\\\",
        "\\midrule",
    ]
    
    for row in rows:
        lines.append(" & ".join([c.replace("_", "\\_") for c in row]) + " \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    return "\n".join(lines)


def pivot_results(
    results: List[BenchmarkResult],
    row_key: str,
    col_key: str,
    val_key: str,
    fmt: str = "{:.2f}",
    filter_fn=None,
) -> Tuple[List[str], List[List[str]]]:
    """Create pivot table from results."""
    filtered = [r for r in results if filter_fn is None or filter_fn(r)]
    
    # Get unique values
    row_vals = sorted(set(getattr(r, row_key) for r in filtered))
    col_vals = sorted(set(getattr(r, col_key) for r in filtered))
    
    # Build lookup
    lookup = {}
    for r in filtered:
        key = (getattr(r, row_key), getattr(r, col_key))
        lookup[key] = getattr(r, val_key)
    
    headers = [row_key.replace("_", " ").title()] + [str(c) for c in col_vals]
    rows = []
    
    for rv in row_vals:
        row = [str(rv)]
        for cv in col_vals:
            val = lookup.get((rv, cv))
            if val is None:
                row.append("-")
            else:
                try:
                    row.append(fmt.format(float(val)))
                except:
                    row.append(str(val))
        rows.append(row)
    
    return headers, rows


# =============================================================================
# Main Sweep Runner
# =============================================================================

def run_full_sweep(sweep_cfg: SweepConfig, output_dir: Path) -> List[BenchmarkResult]:
    """Run complete benchmark sweep."""
    
    basis_presets = BasisConfig.all_presets()  # Include both Jacobi and B-spline
    results: List[BenchmarkResult] = []
    
    # Calculate total configurations (accounting for B-spline having fixed degree)
    jacobi_count = sum(1 for b in sweep_cfg.basis_types if b in BasisConfig.jacobi_presets())
    bspline_count = sum(1 for b in sweep_cfg.basis_types if b in BasisConfig.bspline_presets())
    
    base_count = (
        len(sweep_cfg.lut_sizes) *
        len(sweep_cfg.layer_sizes) *
        len(sweep_cfg.interp_modes) *
        len(sweep_cfg.quant_dtypes) *
        len(sweep_cfg.batch_sizes)
    )
    
    total = (
        jacobi_count * len(sweep_cfg.degrees) * base_count +
        bspline_count * 1 * base_count  # B-spline: degree is fixed
    )
    
    print(f"Total configurations: {total}")
    print(f"Basis types: {sweep_cfg.basis_types}")
    print(f"  Jacobi variants: {jacobi_count}")
    print(f"  B-spline variants: {bspline_count}")
    print(f"Degrees (Jacobi only): {sweep_cfg.degrees}")
    print(f"LUT sizes: {sweep_cfg.lut_sizes}")
    print(f"Layer sizes: {sweep_cfg.layer_sizes}")
    print()
    
    idx = 0
    for basis_name in sweep_cfg.basis_types:
        if basis_name not in basis_presets:
            print(f"[WARN] Unknown basis type: {basis_name}, skipping")
            continue
        basis_cfg = basis_presets[basis_name]
        
        # For B-spline, degree is fixed (cubic=3), complexity is in grid_points
        # For Jacobi, iterate over degrees as usual
        if basis_cfg.family == "bspline":
            degrees_to_test = [3]  # Cubic B-spline, but actual complexity is in grid_points
        else:
            degrees_to_test = sweep_cfg.degrees
        
        for layer_size in sweep_cfg.layer_sizes:
            in_dim = out_dim = layer_size
            
            for degree in degrees_to_test:
                for L in sweep_cfg.lut_sizes:
                    for interp in sweep_cfg.interp_modes:
                        for dtype in sweep_cfg.quant_dtypes:
                            for batch_size in sweep_cfg.batch_sizes:
                                idx += 1
                                if basis_cfg.family == "bspline":
                                    desc = f"{basis_name} g={basis_cfg.grid_points}"
                                else:
                                    desc = f"{basis_name} d={degree}"
                                print(f"[{idx}/{total}] {desc} L={L} {layer_size}x{layer_size} {interp} N={batch_size}")
                                
                                try:
                                    result = benchmark_single_config(
                                        basis_cfg=basis_cfg,
                                        degree=degree,
                                        L=L,
                                        in_dim=in_dim,
                                        out_dim=out_dim,
                                        interp=interp,
                                        quant_dtype=dtype,
                                        batch_size=batch_size,
                                        sweep_cfg=sweep_cfg,
                                    )
                                    results.append(result)
                                except Exception as e:
                                    print(f"  [ERROR] {e}")
    
    return results


def save_results(results: List[BenchmarkResult], output_dir: Path) -> None:
    """Save results to various formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    # Save raw CSV
    csv_path = output_dir / "raw_results.csv"
    if results:
        fieldnames = list(asdict(results[0]).keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
    print(f"Saved: {csv_path}")
    
    # Save JSON
    json_path = output_dir / "raw_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "platform": platform.platform(),
                "python": sys.version,
                "has_numba": HAS_NUMBA,
            },
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"Saved: {json_path}")
    
    # Generate pivot tables
    # 1. CPU Speedup by Basis x Degree (L=64, 16x16, linear, N=256)
    headers, rows = pivot_results(
        results,
        row_key="basis_type",
        col_key="degree",
        val_key="speedup_numba" if HAS_NUMBA else "speedup_numpy",
        fmt="{:.2f}",
        filter_fn=lambda r: r.L == 64 and r.in_dim == 16 and r.interp == "linear" and r.batch_size == 256,
    )
    if rows:
        (tables_dir / "cpu_speedup_by_basis_degree.md").write_text(format_markdown_table(headers, rows), encoding="utf-8")
        (tables_dir / "cpu_speedup_by_basis_degree.tex").write_text(
            format_latex_table(headers, rows, "CPU Speedup by Basis Type and Degree (16x16, L=64, linear)", "tab:cpu_speedup"),
            encoding="utf-8"
        )
    
    # 2. MCU Speedup by Basis x Degree
    headers, rows = pivot_results(
        results,
        row_key="basis_type",
        col_key="degree",
        val_key="mcu_speedup",
        fmt="{:.1f}",
        filter_fn=lambda r: r.L == 64 and r.in_dim == 16 and r.interp == "linear" and r.batch_size == 1,
    )
    if rows:
        (tables_dir / "mcu_speedup_by_basis_degree.md").write_text(format_markdown_table(headers, rows), encoding="utf-8")
        (tables_dir / "mcu_speedup_by_basis_degree.tex").write_text(
            format_latex_table(headers, rows, "MCU Speedup (Cortex-M3) by Basis Type and Degree", "tab:mcu_speedup"),
            encoding="utf-8"
        )
    
    # 3. RMSE by Degree x L
    headers, rows = pivot_results(
        results,
        row_key="degree",
        col_key="L",
        val_key="rmse",
        fmt="{:.6f}",
        filter_fn=lambda r: r.basis_type == "Chebyshev 1st" and r.in_dim == 16 and r.interp == "linear" and r.batch_size == 256,
    )
    if rows:
        (tables_dir / "rmse_chebyshev1_degree_L.md").write_text(format_markdown_table(headers, rows), encoding="utf-8")
        (tables_dir / "rmse_chebyshev1_degree_L.tex").write_text(
            format_latex_table(headers, rows, "RMSE by Degree and LUT Size (Chebyshev 1st, 16x16)", "tab:rmse_chebyshev"),
            encoding="utf-8"
        )
    
    # 4. Memory usage
    headers, rows = pivot_results(
        results,
        row_key="degree",
        col_key="L",
        val_key="lut_mem_bytes",
        fmt="{:.0f}",
        filter_fn=lambda r: r.basis_type == "Chebyshev 1st" and r.in_dim == 16 and r.batch_size == 256,
    )
    if rows:
        (tables_dir / "memory_degree_L.md").write_text(format_markdown_table(headers, rows), encoding="utf-8")
    
    print(f"Saved tables to: {tables_dir}")


def main():
    parser = argparse.ArgumentParser(description="Unified LUT-KAN Benchmark Sweeper")
    
    parser.add_argument("--full", action="store_true", help="Run full sweep (all configurations)")
    parser.add_argument("--quick", action="store_true", help="Quick test with reduced grid")
    
    parser.add_argument("--basis", type=str, default="chebyshev_1,legendre,gegenbauer_1",
                        help="Comma-separated basis types")
    parser.add_argument("--degrees", type=str, default="3,5,8,10,15,20",
                        help="Comma-separated degrees")
    parser.add_argument("--Ls", type=str, default="32,64,128",
                        help="Comma-separated LUT sizes")
    parser.add_argument("--layers", type=str, default="16",
                        help="Comma-separated layer sizes (square)")
    parser.add_argument("--batch_sizes", type=str, default="1,256",
                        help="Comma-separated batch sizes")
    
    parser.add_argument("--out_dir", type=str, default="",
                        help="Output directory")
    
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Build sweep config
    if args.full:
        sweep_cfg = SweepConfig(
            basis_types=["chebyshev_1", "chebyshev_2", "legendre", "gegenbauer_1", "jacobi_asym_1"],
            degrees=[2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30],
            lut_sizes=[16, 32, 64, 128, 256],
            layer_sizes=[8, 16, 32],
            interp_modes=["linear", "nearest"],
            quant_dtypes=["uint8"],
            batch_sizes=[1, 256, 1024, 2048],
        )
    elif args.quick:
        sweep_cfg = SweepConfig(
            basis_types=["chebyshev_1", "legendre"],
            degrees=[3, 5, 10],
            lut_sizes=[32, 64],
            layer_sizes=[8, 16],
            interp_modes=["linear"],
            quant_dtypes=["uint8"],
            batch_sizes=[1, 256],
        )
    else:
        sweep_cfg = SweepConfig(
            basis_types=[s.strip() for s in args.basis.split(",") if s.strip()],
            degrees=[int(x) for x in args.degrees.split(",") if x.strip()],
            lut_sizes=[int(x) for x in args.Ls.split(",") if x.strip()],
            layer_sizes=[int(x) for x in args.layers.split(",") if x.strip()],
            interp_modes=["linear", "nearest"],
            quant_dtypes=["uint8"],
            batch_sizes=[int(x) for x in args.batch_sizes.split(",") if x.strip()],
        )
    
    sweep_cfg.warmup_iters = args.warmup
    sweep_cfg.measure_iters = args.iters
    sweep_cfg.timing_repeats = args.repeats
    sweep_cfg.seed = args.seed
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) if args.out_dir else Path(f"outputs/unified_benchmark_{timestamp}")
    
    print("=" * 80)
    print("Unified LUT-KAN Benchmark Sweeper")
    print("=" * 80)
    print(f"Output: {output_dir}")
    print(f"Numba available: {HAS_NUMBA}")
    print()
    
    # Run sweep
    results = run_full_sweep(sweep_cfg, output_dir)
    
    # Save results
    save_results(results, output_dir)
    
    print()
    print("=" * 80)
    print(f"Benchmark complete! Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()