#!/usr/bin/env python3
"""
MCU Cycle Count Simulator for Jacobi Float vs LUT.

Simulates approximate cycle counts for ARM Cortex-M0/M3 WITHOUT FPU.
This gives theoretical speedup estimation for embedded targets.

Typical cycle counts (Cortex-M0 without FPU):
- Integer add/sub: 1 cycle
- Integer mul (32-bit): 1 cycle  
- Integer shift: 1 cycle
- Memory load (cached): 2 cycles
- Float add (soft-float): 50-100 cycles
- Float mul (soft-float): 70-150 cycles
- Float div (soft-float): 150-300 cycles
- tanh() soft-float: 500-1500 cycles (Taylor series)

Usage:
    python scripts/sim_mcu_cycles.py
    python scripts/sim_mcu_cycles.py --degrees 3 5 10 20
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class MCUProfile:
    """Cycle counts for different MCU types."""
    name: str
    # Integer ops
    int_add: int = 1
    int_mul: int = 1
    int_shift: int = 1
    int_load: int = 2
    int_store: int = 2
    # Float ops (soft-float library)
    float_add: int = 70
    float_mul: int = 100
    float_div: int = 200
    float_cmp: int = 30
    # Special functions (soft-float)
    tanh: int = 800      # Taylor series or CORDIC
    exp: int = 600
    sqrt: int = 300


# Predefined profiles
MCU_PROFILES = {
    "cortex_m0": MCUProfile(
        name="ARM Cortex-M0 (no FPU)",
        float_add=70, float_mul=100, float_div=200,
        tanh=1000, exp=700,
    ),
    "cortex_m3": MCUProfile(
        name="ARM Cortex-M3 (no FPU)", 
        float_add=50, float_mul=80, float_div=150,
        tanh=800, exp=500,
    ),
    "cortex_m4f": MCUProfile(
        name="ARM Cortex-M4F (with FPU)",
        float_add=1, float_mul=1, float_div=14,
        tanh=200, exp=150,  # Still soft-float for transcendentals
    ),
    "esp32_no_fpu": MCUProfile(
        name="ESP32 (FPU disabled)",
        float_add=60, float_mul=90, float_div=180,
        tanh=900, exp=600,
    ),
    "avr": MCUProfile(
        name="AVR (8-bit, no FPU)",
        int_mul=2,
        float_add=150, float_mul=200, float_div=400,
        tanh=2000, exp=1500,
    ),
}


def estimate_jacobi_float_cycles(
    degree: int,
    in_dim: int,
    out_dim: int,
    N: int,
    profile: MCUProfile,
) -> dict:
    """
    Estimate cycles for Jacobi float forward pass.
    
    For each sample and each input dimension:
    1. tanh(x) - 1 call
    2. P[0] = 1 - 1 store
    3. P[1] = ((a-b) + (a+b+2)*x) / 2 - 3 mul, 3 add, 1 div
    4. For i in 2..degree:
       P[i] = (A*x + B) * P[i-1] + C * P[i-2]
       - 4 mul, 2 add per iteration
    5. For each output: sum(P * coeffs) - (degree+1) mul, degree add
    
    Total per (sample, in_dim):
    - 1 tanh
    - 3 + 4*(degree-1) = 4*degree - 1 mul
    - 3 + 2*(degree-1) + degree = 3*degree + 1 add  
    - 1 div
    - For each out_dim: (degree+1) mul, degree add
    """
    
    # Per input coordinate
    cycles_per_coord = (
        profile.tanh +  # tanh normalization
        (4 * degree - 1) * profile.float_mul +  # polynomial recurrence muls
        (3 * degree + 1) * profile.float_add +  # polynomial recurrence adds
        profile.float_div  # division in P[1]
    )
    
    # Accumulation to outputs
    cycles_per_coord += out_dim * (
        (degree + 1) * profile.float_mul +  # P * coeffs
        degree * profile.float_add  # sum
    )
    
    total_cycles = N * in_dim * cycles_per_coord
    
    return {
        "method": "jacobi_float",
        "degree": degree,
        "cycles_per_sample": in_dim * cycles_per_coord,
        "total_cycles": total_cycles,
        "breakdown": {
            "tanh_cycles": N * in_dim * profile.tanh,
            "poly_cycles": N * in_dim * ((4*degree-1) * profile.float_mul + (3*degree+1) * profile.float_add),
            "accum_cycles": N * in_dim * out_dim * ((degree+1) * profile.float_mul + degree * profile.float_add),
        }
    }


def estimate_lut_cycles(
    in_dim: int,
    out_dim: int,
    N: int,
    L: int,
    profile: MCUProfile,
    interp: str = "linear",
) -> dict:
    """
    Estimate cycles for LUT forward pass.
    
    For each sample and each edge (in_dim * out_dim):
    1. Normalize x to [0, L-1]: 2 add, 1 mul, 1 shift (or div)
    2. Load q_table[edge, segment, idx]: 1 load (byte)
    3. If linear interp: load q_table[..., idx+1], interpolate
       - 1 load, 2 mul, 2 add (integer)
    4. Dequantize: y = y_min + scale * q
       - 1 load (y_min), 1 load (scale), 1 mul, 1 add (can be float16 or int)
    5. Accumulate to output: 1 add
    
    Using integer math throughout except final dequant.
    """
    
    edges = in_dim * out_dim
    
    if interp == "linear":
        # Index calculation: ~5 int ops
        # Two loads + interpolation: ~8 int ops  
        # Dequant (float): 1 mul, 1 add
        # Accumulate (float): 1 add
        cycles_per_edge = (
            5 * profile.int_add +  # index calc (approx)
            3 * profile.int_load +  # q[idx], q[idx+1], scale
            4 * profile.int_mul +  # interpolation
            4 * profile.int_add +  # interpolation
            profile.float_mul +  # scale * q
            2 * profile.float_add  # y_min + ..., accumulate
        )
    else:  # nearest
        cycles_per_edge = (
            5 * profile.int_add +
            2 * profile.int_load +  # q[idx], scale
            profile.float_mul +
            2 * profile.float_add
        )
    
    total_cycles = N * edges * cycles_per_edge
    
    return {
        "method": f"lut_{interp}",
        "cycles_per_sample": edges * cycles_per_edge,
        "total_cycles": total_cycles,
        "cycles_per_edge": cycles_per_edge,
    }


def estimate_lut_pure_int_cycles(
    in_dim: int,
    out_dim: int,
    N: int,
    L: int,
    profile: MCUProfile,
) -> dict:
    """
    Estimate cycles for pure integer LUT (fixed-point accumulation).
    
    No float ops at all - everything in fixed-point int32.
    This is the fastest possible on MCU without FPU.
    """
    
    edges = in_dim * out_dim
    
    # All integer: index, load, interpolate, accumulate
    cycles_per_edge = (
        4 * profile.int_add +   # index calc
        2 * profile.int_load +  # q values
        2 * profile.int_mul +   # interpolation
        2 * profile.int_add     # interpolation + accumulate
    )
    
    # Final dequant once per output (not per edge)
    cycles_final = out_dim * (profile.float_mul + profile.float_add)
    
    total_cycles = N * (edges * cycles_per_edge + cycles_final)
    
    return {
        "method": "lut_pure_int",
        "cycles_per_sample": edges * cycles_per_edge + cycles_final,
        "total_cycles": total_cycles,
    }


def run_comparison(
    degree: int,
    in_dim: int,
    out_dim: int,
    N: int,
    L: int,
    profile: MCUProfile,
) -> dict:
    """Run full comparison for given parameters."""
    
    float_est = estimate_jacobi_float_cycles(degree, in_dim, out_dim, N, profile)
    lut_linear = estimate_lut_cycles(in_dim, out_dim, N, L, profile, "linear")
    lut_nearest = estimate_lut_cycles(in_dim, out_dim, N, L, profile, "nearest")
    lut_int = estimate_lut_pure_int_cycles(in_dim, out_dim, N, L, profile)
    
    return {
        "params": {
            "degree": degree,
            "in_dim": in_dim, 
            "out_dim": out_dim,
            "edges": in_dim * out_dim,
            "N": N,
            "L": L,
            "mcu": profile.name,
        },
        "float": float_est,
        "lut_linear": lut_linear,
        "lut_nearest": lut_nearest,
        "lut_pure_int": lut_int,
        "speedup_linear": float_est["total_cycles"] / lut_linear["total_cycles"],
        "speedup_nearest": float_est["total_cycles"] / lut_nearest["total_cycles"],
        "speedup_pure_int": float_est["total_cycles"] / lut_int["total_cycles"],
    }


def main():
    parser = argparse.ArgumentParser(description="MCU cycle count simulator")
    parser.add_argument("--degrees", type=int, nargs="+", default=[3, 5, 8, 10, 15, 20])
    parser.add_argument("--in_dim", type=int, default=16)
    parser.add_argument("--out_dim", type=int, default=16)
    parser.add_argument("--N", type=int, default=1, help="Batch size (usually 1 on MCU)")
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--mcu", choices=list(MCU_PROFILES.keys()), default="cortex_m0")
    parser.add_argument("--compare_mcus", action="store_true", help="Compare all MCU profiles")
    
    args = parser.parse_args()
    
    if args.compare_mcus:
        # Compare across MCU profiles for degree=3
        print(f"\n{'='*90}")
        print(f"MCU Comparison for Jacobi degree=3, layer {args.in_dim}x{args.out_dim}")
        print(f"{'='*90}\n")
        
        print(f"{'MCU Profile':<30} {'Float (Kcyc)':<15} {'LUT int (Kcyc)':<15} {'Speedup':<10}")
        print("-" * 70)
        
        for mcu_name, profile in MCU_PROFILES.items():
            r = run_comparison(3, args.in_dim, args.out_dim, 1, args.L, profile)
            float_kc = r["float"]["cycles_per_sample"] / 1000
            lut_kc = r["lut_pure_int"]["cycles_per_sample"] / 1000
            print(f"{profile.name:<30} {float_kc:<15.1f} {lut_kc:<15.1f} {r['speedup_pure_int']:<10.1f}x")
        
        print()
        return
    
    profile = MCU_PROFILES[args.mcu]
    
    print(f"\n{'='*90}")
    print(f"MCU Cycle Simulation: {profile.name}")
    print(f"Layer: {args.in_dim}x{args.out_dim} = {args.in_dim * args.out_dim} edges")
    print(f"Batch: N={args.N}, LUT size: L={args.L}")
    print(f"{'='*90}\n")
    
    print(f"{'Degree':<8} {'Float(Kcyc)':<14} {'LUT lin(Kcyc)':<14} {'LUT int(Kcyc)':<14} "
          f"{'Speedup lin':<12} {'Speedup int':<12}")
    print("-" * 90)
    
    for degree in args.degrees:
        r = run_comparison(degree, args.in_dim, args.out_dim, args.N, args.L, profile)
        
        float_kc = r["float"]["cycles_per_sample"] / 1000
        lut_lin_kc = r["lut_linear"]["cycles_per_sample"] / 1000
        lut_int_kc = r["lut_pure_int"]["cycles_per_sample"] / 1000
        
        print(f"{degree:<8} {float_kc:<14.1f} {lut_lin_kc:<14.1f} {lut_int_kc:<14.1f} "
              f"{r['speedup_linear']:<12.1f}x {r['speedup_pure_int']:<12.1f}x")
    
    print(f"\n{'='*90}")
    print("Legend:")
    print("  Float     = Jacobi polynomial evaluation with soft-float library")
    print("  LUT lin   = LUT with linear interpolation, float dequant")
    print("  LUT int   = LUT with pure integer math, fixed-point accumulation")
    print(f"\nKey insight: On {profile.name}, tanh() alone costs ~{profile.tanh} cycles!")
    print(f"             LUT lookup + interp costs ~{r['lut_pure_int']['cycles_per_sample'] // (args.in_dim * args.out_dim)} cycles per edge")
    print()


if __name__ == "__main__":
    main()
