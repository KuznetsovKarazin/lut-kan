#!/usr/bin/env python3
"""
Benchmark Jacobi float vs LUT at different polynomial degrees.

Usage:
    python scripts/bench_jacobi_degrees.py
    python scripts/bench_jacobi_degrees.py --degrees 3 5 8 10 15 20
    python scripts/bench_jacobi_degrees.py --in_dim 64 --out_dim 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.jacobi_adapter import JacobiKANSingleLayerAdapter
from src.quant.lut_builder import build_lut_for_edges
from src.kernels.lut_contract import pack_dense_layer
from src.kernels.lut_backend_dense_numpy import forward_dense_numpy

try:
    from src.kernels.lut_backend_dense_numba import (
        forward_dense_numba, 
        numba_available, 
        warmup_numba
    )
    HAS_NUMBA = numba_available()
except ImportError:
    HAS_NUMBA = False


def benchmark_degree(
    degree: int,
    in_dim: int = 16,
    out_dim: int = 16,
    L: int = 64,
    N: int = 2048,
    warmup: int = 10,
    iters: int = 100,
) -> dict:
    """Benchmark float vs LUT for a given degree."""
    
    # Create adapter
    adapter = JacobiKANSingleLayerAdapter.from_arch(
        arch={
            "in_dim": in_dim,
            "out_dim": out_dim,
            "degree": degree,
            "alpha": -0.5,
            "beta": -0.5,
            "use_tanh": True,
            "x_min": -3.0,
            "x_max": 3.0,
            "num_knots": 9,
        },
        seed=42,
    )
    
    # Build LUT
    edges = adapter.extract_edges()
    art = build_lut_for_edges(
        edges=edges,
        L=L,
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
        in_dim=in_dim,
        out_dim=out_dim,
        boundary_mode="half_open",
    )
    
    # Generate test data
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N, in_dim)).astype(np.float32)
    x = np.clip(x, -3.0, 3.0)
    
    # Warmup float
    for _ in range(warmup):
        _ = adapter.forward_float(x)
    
    # Benchmark float
    t0 = time.perf_counter()
    for _ in range(iters):
        y_float = adapter.forward_float(x)
    float_time = (time.perf_counter() - t0) / iters * 1000  # ms
    
    # Warmup LUT numpy
    for _ in range(warmup):
        _ = forward_dense_numpy(x, packed)
    
    # Benchmark LUT numpy
    t0 = time.perf_counter()
    for _ in range(iters):
        y_lut = forward_dense_numpy(x, packed)
    lut_numpy_time = (time.perf_counter() - t0) / iters * 1000  # ms
    
    # Benchmark LUT numba if available
    lut_numba_time = None
    if HAS_NUMBA:
        warmup_numba(packed, in_dim=in_dim, out_dim=out_dim)
        for _ in range(warmup):
            _ = forward_dense_numba(x, packed)
        
        t0 = time.perf_counter()
        for _ in range(iters):
            y_lut_numba = forward_dense_numba(x, packed)
        lut_numba_time = (time.perf_counter() - t0) / iters * 1000  # ms
    
    # Compute error
    rmse = float(np.sqrt(np.mean((y_float - y_lut) ** 2)))
    max_abs = float(np.max(np.abs(y_float - y_lut)))
    
    return {
        "degree": degree,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "edges": len(edges),
        "L": L,
        "float_ms": float_time,
        "lut_numpy_ms": lut_numpy_time,
        "lut_numba_ms": lut_numba_time,
        "speedup_numpy": float_time / lut_numpy_time if lut_numpy_time else None,
        "speedup_numba": float_time / lut_numba_time if lut_numba_time else None,
        "rmse": rmse,
        "max_abs": max_abs,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Jacobi degrees")
    parser.add_argument("--degrees", type=int, nargs="+", 
                       default=[2, 3, 5, 8, 10, 15, 20, 30],
                       help="Degrees to test")
    parser.add_argument("--in_dim", type=int, default=16)
    parser.add_argument("--out_dim", type=int, default=16)
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--N", type=int, default=2048, help="Batch size")
    parser.add_argument("--iters", type=int, default=100)
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Jacobi Float vs LUT Benchmark")
    print(f"Layer: {args.in_dim}x{args.out_dim} = {args.in_dim * args.out_dim} edges")
    print(f"Batch: N={args.N}, LUT size: L={args.L}")
    print(f"{'='*80}\n")
    
    # Header
    print(f"{'Degree':<8} {'Float(ms)':<12} {'LUT np(ms)':<12} {'LUT nb(ms)':<12} "
          f"{'Speedup np':<12} {'Speedup nb':<12} {'RMSE':<12}")
    print("-" * 80)
    
    results = []
    for degree in args.degrees:
        try:
            r = benchmark_degree(
                degree=degree,
                in_dim=args.in_dim,
                out_dim=args.out_dim,
                L=args.L,
                N=args.N,
                iters=args.iters,
            )
            results.append(r)
            
            speedup_np = f"{r['speedup_numpy']:.2f}x" if r['speedup_numpy'] else "N/A"
            speedup_nb = f"{r['speedup_numba']:.2f}x" if r['speedup_numba'] else "N/A"
            lut_nb = f"{r['lut_numba_ms']:.3f}" if r['lut_numba_ms'] else "N/A"
            
            # Color code speedup
            if r['speedup_numba'] and r['speedup_numba'] > 1:
                speedup_nb = f"✅ {speedup_nb}"
            elif r['speedup_numba']:
                speedup_nb = f"❌ {speedup_nb}"
            
            print(f"{degree:<8} {r['float_ms']:<12.3f} {r['lut_numpy_ms']:<12.3f} "
                  f"{lut_nb:<12} {speedup_np:<12} {speedup_nb:<12} {r['rmse']:<12.6f}")
            
        except Exception as e:
            print(f"{degree:<8} ERROR: {e}")
    
    # Find crossover point
    print("\n" + "=" * 80)
    crossover = None
    for r in results:
        if r['speedup_numba'] and r['speedup_numba'] > 1:
            crossover = r['degree']
            break
    
    if crossover:
        print(f"✅ LUT becomes faster at degree >= {crossover}")
    else:
        print("❌ LUT is slower than float for all tested degrees")
        print("   Consider: larger layer size, or target platform without fast FPU")


if __name__ == "__main__":
    main()
