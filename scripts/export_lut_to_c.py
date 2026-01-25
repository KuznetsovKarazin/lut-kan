#!/usr/bin/env python3
"""
Export Jacobi KAN LUT to C header file for MCU deployment.

Usage:
    python export_lut_to_c.py --in_dim 16 --out_dim 16 --degree 3 --L 64
    python export_lut_to_c.py --in_dim 8 --out_dim 8 --degree 5 --L 32 --output my_layer.h

Output:
    - generated_layer.h (LUT tables + float coefficients)
"""

import argparse
import struct
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add lut-kan to path for imports
LUT_KAN_PATH = Path(__file__).resolve().parents[2] / "lut-kan"
if LUT_KAN_PATH.exists():
    sys.path.insert(0, str(LUT_KAN_PATH))


def float32_to_float16_bits(val: float) -> int:
    """Convert float32 to IEEE 754 float16 bit representation."""
    try:
        packed = struct.pack('>e', float(val))
        return struct.unpack('>H', packed)[0]
    except (OverflowError, struct.error):
        # Handle overflow - return inf
        return 0x7C00 if val > 0 else 0xFC00


def generate_demo_lut_data(in_dim: int, out_dim: int, L: int, degree: int, seed: int = 42):
    """Generate demo LUT data when lut-kan is not available."""
    rng = np.random.default_rng(seed)
    edges = in_dim * out_dim
    
    # Generate smooth LUT curves
    q_table = np.zeros((edges, L), dtype=np.uint8)
    for e in range(edges):
        # Random smooth curve
        t = np.linspace(0, 1, L)
        curve = np.sin(t * np.pi * (1 + rng.random())) * 0.5 + 0.5
        curve += rng.normal(0, 0.02, L)  # Add noise
        curve = np.clip(curve, 0, 1)
        q_table[e] = (curve * 255).astype(np.uint8)
    
    # Scale and y_min
    scale = rng.uniform(0.05, 0.2, edges).astype(np.float32)
    y_min = rng.uniform(-1.0, -0.5, edges).astype(np.float32)
    
    # Float coefficients
    coeffs = rng.normal(0, 0.1, (in_dim, out_dim, degree + 1)).astype(np.float32)
    
    return q_table, scale, y_min, coeffs


def generate_lut_from_adapter(in_dim: int, out_dim: int, L: int, degree: int, 
                               alpha: float, beta: float, seed: int):
    """Generate LUT from actual Jacobi adapter."""
    try:
        from src.models.jacobi_adapter import JacobiKANSingleLayerAdapter
        from src.quant.lut_builder import build_lut_for_edges
    except ImportError:
        print("[WARN] lut-kan not found, using demo data")
        return generate_demo_lut_data(in_dim, out_dim, L, degree, seed)
    
    # Create adapter
    adapter = JacobiKANSingleLayerAdapter.from_arch(
        arch={
            "in_dim": in_dim,
            "out_dim": out_dim,
            "degree": degree,
            "alpha": alpha,
            "beta": beta,
            "use_tanh": True,
            "x_min": -3.0,
            "x_max": 3.0,
            "num_knots": 9,
        },
        seed=seed,
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
    
    # Extract arrays
    q_table = art.q_table[:, 0, :]  # [E, K, L] -> [E, L]
    scale = art.scale[:, 0]
    y_min = art.y_min[:, 0]
    coeffs = adapter.coeffs
    
    return q_table, scale, y_min, coeffs


def write_c_header(
    output_path: str,
    layer_name: str,
    in_dim: int,
    out_dim: int,
    L: int,
    degree: int,
    alpha: float,
    beta: float,
    q_table: np.ndarray,
    scale: np.ndarray,
    y_min: np.ndarray,
    coeffs: np.ndarray,
):
    """Write C header file with LUT data."""
    edges = in_dim * out_dim
    
    with open(output_path, 'w') as f:
        # Header comment
        f.write(f"""/**
 * @file {Path(output_path).name}
 * @brief Generated Jacobi KAN LUT layer
 * 
 * Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 * 
 * Configuration:
 *   Layer: {in_dim}x{out_dim} = {edges} edges
 *   LUT size: L={L}
 *   Degree: {degree}
 *   Alpha: {alpha}, Beta: {beta}
 * 
 * Memory:
 *   LUT:   {edges * L + edges * 4:,} bytes
 *   Float: {edges * (degree + 1) * 4:,} bytes
 */

#ifndef {layer_name.upper()}_H
#define {layer_name.upper()}_H

#include <stdint.h>
#include "jacobi_lut.h"
#include "jacobi_float.h"

/* Layer dimensions */
#define {layer_name.upper()}_IN_DIM   {in_dim}
#define {layer_name.upper()}_OUT_DIM  {out_dim}
#define {layer_name.upper()}_L        {L}
#define {layer_name.upper()}_EDGES    {edges}
#define {layer_name.upper()}_DEGREE   {degree}

/* ============================================================================
 * LUT Data
 * ============================================================================ */

// Quantized LUT values [{edges}][{L}]
static const uint8_t {layer_name}_q_table[{edges} * {L}] = {{
""")
        
        # Write q_table
        for e in range(edges):
            f.write(f"    // Edge {e} (in={e // out_dim}, out={e % out_dim})\n    ")
            row = q_table[e]
            for i, val in enumerate(row):
                f.write(f"{int(val):3d}")
                if i < len(row) - 1:
                    f.write(",")
                if (i + 1) % 16 == 0 and i < len(row) - 1:
                    f.write("\n    ")
            if e < edges - 1:
                f.write(",")
            f.write("\n")
        
        f.write("};\n\n")
        
        # Write scale
        f.write(f"// Scale as float16 bits [{edges}]\n")
        f.write(f"static const uint16_t {layer_name}_scale[{edges}] = {{\n    ")
        for e in range(edges):
            bits = float32_to_float16_bits(float(scale[e]))
            f.write(f"0x{bits:04X}")
            if e < edges - 1:
                f.write(", ")
            if (e + 1) % 8 == 0 and e < edges - 1:
                f.write("\n    ")
        f.write("\n};\n\n")
        
        # Write y_min
        f.write(f"// Y_min as float16 bits [{edges}]\n")
        f.write(f"static const uint16_t {layer_name}_y_min[{edges}] = {{\n    ")
        for e in range(edges):
            bits = float32_to_float16_bits(float(y_min[e]))
            f.write(f"0x{bits:04X}")
            if e < edges - 1:
                f.write(", ")
            if (e + 1) % 8 == 0 and e < edges - 1:
                f.write("\n    ")
        f.write("\n};\n\n")
        
        # Write float coefficients
        f.write(f"""/* ============================================================================
 * Float Coefficients (for baseline comparison)
 * ============================================================================ */

static const float {layer_name}_float_coeffs[{edges} * {degree + 1}] = {{
""")
        
        for i in range(in_dim):
            for j in range(out_dim):
                e = i * out_dim + j
                f.write(f"    // Edge {e} (in={i}, out={j})\n    ")
                c = coeffs[i, j, :]
                for k, val in enumerate(c):
                    f.write(f"{float(val):.8f}f")
                    if k < len(c) - 1:
                        f.write(", ")
                if e < edges - 1:
                    f.write(",")
                f.write("\n")
        
        f.write("};\n\n")
        
        # Write layer structs
        f.write(f"""/* ============================================================================
 * Layer Configurations
 * ============================================================================ */

static const JacobiLUT {layer_name}_lut = {{
    .in_dim = {in_dim},
    .out_dim = {out_dim},
    .L = {L},
    .x_min = -3.0f,
    .x_max = 3.0f,
    .q_table = {layer_name}_q_table,
    .scale = {layer_name}_scale,
    .y_min = {layer_name}_y_min,
}};

static const JacobiFloat {layer_name}_float = {{
    .in_dim = {in_dim},
    .out_dim = {out_dim},
    .degree = {degree},
    .alpha = {alpha}f,
    .beta = {beta}f,
    .use_tanh = 1,
    .coeffs = {layer_name}_float_coeffs,
}};

#endif // {layer_name.upper()}_H
""")
    
    print(f"[OK] Generated: {output_path}")
    print(f"     Edges: {edges}")
    print(f"     LUT memory: {edges * L + edges * 4:,} bytes")
    print(f"     Float memory: {edges * (degree + 1) * 4:,} bytes")


def main():
    parser = argparse.ArgumentParser(description="Export Jacobi LUT to C header")
    parser.add_argument("--in_dim", type=int, default=4, help="Input dimension")
    parser.add_argument("--out_dim", type=int, default=4, help="Output dimension")
    parser.add_argument("--degree", type=int, default=3, help="Polynomial degree")
    parser.add_argument("--L", type=int, default=32, help="LUT size")
    parser.add_argument("--alpha", type=float, default=-0.5, help="Jacobi alpha")
    parser.add_argument("--beta", type=float, default=-0.5, help="Jacobi beta")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="include/generated_layer.h")
    parser.add_argument("--name", type=str, default="layer0", help="Layer name in C")
    parser.add_argument("--demo", action="store_true", help="Use demo data (no lut-kan needed)")
    
    args = parser.parse_args()
    
    print(f"\nGenerating Jacobi LUT C header:")
    print(f"  Layer: {args.in_dim}x{args.out_dim} = {args.in_dim * args.out_dim} edges")
    print(f"  LUT: L={args.L}, Degree: {args.degree}")
    print(f"  Jacobi: α={args.alpha}, β={args.beta}")
    print()
    
    if args.demo:
        q_table, scale, y_min, coeffs = generate_demo_lut_data(
            args.in_dim, args.out_dim, args.L, args.degree, args.seed
        )
    else:
        q_table, scale, y_min, coeffs = generate_lut_from_adapter(
            args.in_dim, args.out_dim, args.L, args.degree,
            args.alpha, args.beta, args.seed
        )
    
    write_c_header(
        output_path=args.output,
        layer_name=args.name,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        L=args.L,
        degree=args.degree,
        alpha=args.alpha,
        beta=args.beta,
        q_table=q_table,
        scale=scale,
        y_min=y_min,
        coeffs=coeffs,
    )


if __name__ == "__main__":
    main()
