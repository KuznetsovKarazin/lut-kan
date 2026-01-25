#!/usr/bin/env python3
"""
QEMU-based MCU Benchmark for ARM Cortex-M.

This script provides cycle-accurate benchmarking on emulated ARM Cortex-M3
using QEMU with semihosting support.

Workflow:
1. Generate C code with LUT data
2. Compile with arm-none-eabi-gcc
3. Run on QEMU with cycle counting
4. Parse and collect results

Requirements:
    - arm-none-eabi-gcc (ARM GCC toolchain)
    - qemu-system-arm

Usage:
    python scripts/mcu_qemu_benchmark.py --degrees 3,5,10 --Ls 32,64 --dim 8
    python scripts/mcu_qemu_benchmark.py --full  # Full sweep
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.jacobi_adapter import JacobiKANSingleLayerAdapter
from src.quant.lut_builder import build_lut_for_edges


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MCUConfig:
    """MCU benchmark configuration."""
    in_dim: int = 8
    out_dim: int = 8
    degree: int = 3
    L: int = 64
    alpha: float = -0.5
    beta: float = -0.5
    use_tanh: bool = True
    x_min: float = -3.0
    x_max: float = 3.0
    num_knots: int = 9
    seed: int = 42
    num_samples: int = 100


@dataclass
class MCUResult:
    """MCU benchmark result."""
    config: dict
    float_cycles: int = 0
    lut_cycles: int = 0
    speedup: float = 0.0
    flash_bytes: int = 0
    ram_bytes: int = 0
    status: str = "OK"
    error: str = ""


# =============================================================================
# C Code Generation
# =============================================================================

def float32_to_float16_bits(val: float) -> int:
    """Convert float32 to IEEE 754 float16 bit representation."""
    try:
        packed = struct.pack('>e', float(val))
        return struct.unpack('>H', packed)[0]
    except (OverflowError, struct.error):
        return 0x7C00 if val > 0 else 0xFC00


def generate_benchmark_c_code(
    cfg: MCUConfig,
    q_table: np.ndarray,
    scale: np.ndarray,
    y_min: np.ndarray,
    coeffs: np.ndarray,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Generate complete C benchmark code."""
    
    edges = cfg.in_dim * cfg.out_dim
    
    # Generate layer header
    header_content = f'''/**
 * Auto-generated LUT-KAN benchmark layer
 * Config: {cfg.in_dim}x{cfg.out_dim}, degree={cfg.degree}, L={cfg.L}
 */

#ifndef BENCH_LAYER_H
#define BENCH_LAYER_H

#include <stdint.h>

#define IN_DIM   {cfg.in_dim}
#define OUT_DIM  {cfg.out_dim}
#define EDGES    {edges}
#define L        {cfg.L}
#define DEGREE   {cfg.degree}
#define NUM_KNOTS {cfg.num_knots}

// Domain bounds
#define X_MIN    {cfg.x_min}f
#define X_MAX    {cfg.x_max}f
#define USE_TANH {1 if cfg.use_tanh else 0}

// Jacobi parameters
#define ALPHA    {cfg.alpha}f
#define BETA     {cfg.beta}f

// LUT table [{edges}][{cfg.L}]
extern const uint8_t q_table[EDGES * L];

// Scale and y_min (float16 as uint16)
extern const uint16_t scale_f16[EDGES];
extern const uint16_t y_min_f16[EDGES];

// Float coefficients [{cfg.in_dim}][{cfg.out_dim}][{cfg.degree + 1}]
extern const float coeffs[EDGES * (DEGREE + 1)];

// Knots
extern const float knots[NUM_KNOTS];

#endif
'''
    
    # Generate data file
    data_content = f'''/**
 * Auto-generated LUT data
 */

#include "bench_layer.h"

const uint8_t q_table[EDGES * L] = {{
'''
    
    # Write q_table
    for e in range(edges):
        data_content += f"    // Edge {e}\n    "
        for i, val in enumerate(q_table[e]):
            data_content += f"{int(val):3d}"
            if i < cfg.L - 1:
                data_content += ","
            if (i + 1) % 16 == 0 and i < cfg.L - 1:
                data_content += "\n    "
        if e < edges - 1:
            data_content += ","
        data_content += "\n"
    
    data_content += "};\n\n"
    
    # Scale
    data_content += "const uint16_t scale_f16[EDGES] = {\n    "
    for e in range(edges):
        bits = float32_to_float16_bits(float(scale[e]))
        data_content += f"0x{bits:04X}"
        if e < edges - 1:
            data_content += ", "
        if (e + 1) % 8 == 0 and e < edges - 1:
            data_content += "\n    "
    data_content += "\n};\n\n"
    
    # Y_min
    data_content += "const uint16_t y_min_f16[EDGES] = {\n    "
    for e in range(edges):
        bits = float32_to_float16_bits(float(y_min[e]))
        data_content += f"0x{bits:04X}"
        if e < edges - 1:
            data_content += ", "
        if (e + 1) % 8 == 0 and e < edges - 1:
            data_content += "\n    "
    data_content += "\n};\n\n"
    
    # Coefficients
    data_content += "const float coeffs[EDGES * (DEGREE + 1)] = {\n"
    for i in range(cfg.in_dim):
        for j in range(cfg.out_dim):
            e = i * cfg.out_dim + j
            data_content += f"    // Edge {e} (in={i}, out={j})\n    "
            c = coeffs[i, j, :]
            for k, val in enumerate(c):
                data_content += f"{float(val):.8f}f"
                if k < len(c) - 1:
                    data_content += ", "
            if e < edges - 1:
                data_content += ","
            data_content += "\n"
    data_content += "};\n\n"
    
    # Knots
    knots = np.linspace(cfg.x_min, cfg.x_max, cfg.num_knots)
    data_content += "const float knots[NUM_KNOTS] = {\n    "
    for k, val in enumerate(knots):
        data_content += f"{float(val):.8f}f"
        if k < cfg.num_knots - 1:
            data_content += ", "
    data_content += "\n};\n"
    
    # Write files
    header_path = output_dir / "bench_layer.h"
    data_path = output_dir / "bench_layer_data.c"
    
    header_path.write_text(header_content)
    data_path.write_text(data_content)
    
    return header_path, data_path


def generate_main_c(output_dir: Path, num_samples: int, use_tanh: bool) -> Path:
    """Generate main benchmark C file with semihosting."""
    
    main_content = f'''/**
 * LUT-KAN MCU Benchmark with ARM Semihosting
 * Compares Jacobi float vs LUT inference
 */

#include <stdint.h>
#include <math.h>
#include "bench_layer.h"

// ARM semihosting interface
static inline int semihosting_call(int op, void *arg) {{
    register int r0 asm("r0") = op;
    register void *r1 asm("r1") = arg;
    asm volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}}

// Print string via semihosting
void sh_print(const char *s) {{
    uint32_t args[3];
    args[0] = 1;  // stdout
    args[1] = (uint32_t)s;
    // Find string length
    int len = 0;
    while (s[len]) len++;
    args[2] = len;
    semihosting_call(0x05, args);  // SYS_WRITE
}}

// Print integer
void sh_print_int(int val) {{
    char buf[16];
    int i = 15;
    buf[i--] = 0;
    int neg = 0;
    if (val < 0) {{ neg = 1; val = -val; }}
    if (val == 0) buf[i--] = '0';
    while (val > 0 && i >= 0) {{
        buf[i--] = '0' + (val % 10);
        val /= 10;
    }}
    if (neg && i >= 0) buf[i--] = '-';
    sh_print(&buf[i+1]);
}}

// DWT cycle counter
#define DWT_CTRL   (*(volatile uint32_t*)0xE0001000)
#define DWT_CYCCNT (*(volatile uint32_t*)0xE0001004)
#define DEMCR      (*(volatile uint32_t*)0xE000EDFC)

void dwt_init(void) {{
    DEMCR |= (1 << 24);  // TRCENA
    DWT_CYCCNT = 0;
    DWT_CTRL |= 1;       // CYCCNTENA
}}

uint32_t dwt_get_cycles(void) {{
    return DWT_CYCCNT;
}}

// Float16 to float32 conversion
float f16_to_f32(uint16_t h) {{
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    
    if (exp == 0) {{
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Subnormal
        while ((mant & 0x400) == 0) {{ mant <<= 1; exp--; }}
        exp++;
        mant &= 0x3FF;
    }} else if (exp == 31) {{
        // Inf or NaN
        uint32_t f = (sign << 31) | (0xFF << 23) | (mant << 13);
        return *(float*)&f;
    }}
    
    uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    return *(float*)&f;
}}

// Jacobi polynomial evaluation
void jacobi_eval(const float *x, float *out, const float *c, int degree, float alpha, float beta) {{
    float P[32];  // Max degree 31
    
    for (int n = 0; n < IN_DIM; n++) {{
        float xn = x[n];
        
        {'// tanh normalization' if use_tanh else '// No tanh'}
        {'''
        // tanh approximation: tanh(x) ≈ x for small x, ±1 for large
        float tx = xn;
        if (tx > 4.0f) tx = 1.0f;
        else if (tx < -4.0f) tx = -1.0f;
        else {{
            float x2 = tx * tx;
            tx = tx * (1.0f - x2 * (0.333333f - x2 * 0.133333f));
        }}
        xn = tx;
        ''' if use_tanh else ''}
        
        // P_0 = 1
        P[0] = 1.0f;
        
        // P_1 = ((alpha - beta) + (alpha + beta + 2) * x) / 2
        if (degree >= 1) {{
            P[1] = 0.5f * ((alpha - beta) + (alpha + beta + 2.0f) * xn);
        }}
        
        // Recurrence for P_i, i >= 2
        for (int i = 2; i <= degree; i++) {{
            float fi = (float)i;
            float A = (2*fi + alpha + beta - 1) * (2*fi + alpha + beta) / ((2*fi) * (fi + alpha + beta));
            float B = (2*fi + alpha + beta - 1) * (alpha*alpha - beta*beta) / 
                      ((2*fi) * (fi + alpha + beta) * (2*fi + alpha + beta - 2));
            float C = -2.0f * (fi + alpha - 1) * (fi + beta - 1) * (2*fi + alpha + beta) /
                      ((2*fi) * (fi + alpha + beta) * (2*fi + alpha + beta - 2));
            P[i] = (A * xn + B) * P[i-1] + C * P[i-2];
        }}
        
        // Accumulate to outputs
        for (int o = 0; o < OUT_DIM; o++) {{
            int edge = n * OUT_DIM + o;
            float sum = 0.0f;
            for (int d = 0; d <= degree; d++) {{
                sum += c[edge * (degree + 1) + d] * P[d];
            }}
            out[o] += sum;
        }}
    }}
}}

// LUT evaluation (linear interpolation)
void lut_eval(const float *x, float *out) {{
    for (int n = 0; n < IN_DIM; n++) {{
        float xn = x[n];
        
        // Clamp to domain
        if (xn < X_MIN) xn = X_MIN;
        if (xn > X_MAX) xn = X_MAX;
        
        // Find segment
        int seg = 0;
        for (int k = 0; k < NUM_KNOTS - 1; k++) {{
            if (xn >= knots[k] && xn < knots[k+1]) {{
                seg = k;
                break;
            }}
        }}
        if (xn >= knots[NUM_KNOTS - 1]) seg = NUM_KNOTS - 2;
        
        // Interpolate within segment
        float seg_start = knots[seg];
        float seg_end = knots[seg + 1];
        float t = (xn - seg_start) / (seg_end - seg_start);
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        
        float idx_f = t * (float)(L - 1);
        int idx_lo = (int)idx_f;
        int idx_hi = idx_lo + 1;
        if (idx_hi >= L) idx_hi = L - 1;
        float frac = idx_f - (float)idx_lo;
        
        // For each output
        for (int o = 0; o < OUT_DIM; o++) {{
            int edge = n * OUT_DIM + o;
            
            // Get quantized values
            uint8_t q_lo = q_table[edge * L + idx_lo];
            uint8_t q_hi = q_table[edge * L + idx_hi];
            
            // Linear interpolation
            float q_interp = (float)q_lo * (1.0f - frac) + (float)q_hi * frac;
            
            // Dequantize
            float scale = f16_to_f32(scale_f16[edge]);
            float ymin = f16_to_f32(y_min_f16[edge]);
            
            out[o] += ymin + scale * q_interp;
        }}
    }}
}}

// Test data (simple deterministic pattern)
float test_inputs[{num_samples}][IN_DIM];

void init_test_data(void) {{
    for (int i = 0; i < {num_samples}; i++) {{
        for (int j = 0; j < IN_DIM; j++) {{
            // Simple pseudo-random pattern
            float v = (float)((i * 17 + j * 31) % 1000) / 500.0f - 1.0f;
            v = v * 2.5f;  // Scale to [-2.5, 2.5]
            test_inputs[i][j] = v;
        }}
    }}
}}

int main(void) {{
    dwt_init();
    init_test_data();
    
    float out_float[OUT_DIM];
    float out_lut[OUT_DIM];
    
    sh_print("\\n=== LUT-KAN MCU Benchmark ===\\n");
    sh_print("Config: ");
    sh_print_int(IN_DIM);
    sh_print("x");
    sh_print_int(OUT_DIM);
    sh_print(", degree=");
    sh_print_int(DEGREE);
    sh_print(", L=");
    sh_print_int(L);
    sh_print("\\n");
    
    // Warm up
    for (int i = 0; i < 10; i++) {{
        for (int o = 0; o < OUT_DIM; o++) out_float[o] = 0.0f;
        jacobi_eval(test_inputs[0], out_float, coeffs, DEGREE, ALPHA, BETA);
        for (int o = 0; o < OUT_DIM; o++) out_lut[o] = 0.0f;
        lut_eval(test_inputs[0], out_lut);
    }}
    
    // Benchmark float
    uint32_t start = dwt_get_cycles();
    for (int i = 0; i < {num_samples}; i++) {{
        for (int o = 0; o < OUT_DIM; o++) out_float[o] = 0.0f;
        jacobi_eval(test_inputs[i], out_float, coeffs, DEGREE, ALPHA, BETA);
    }}
    uint32_t float_cycles = dwt_get_cycles() - start;
    
    // Benchmark LUT
    start = dwt_get_cycles();
    for (int i = 0; i < {num_samples}; i++) {{
        for (int o = 0; o < OUT_DIM; o++) out_lut[o] = 0.0f;
        lut_eval(test_inputs[i], out_lut);
    }}
    uint32_t lut_cycles = dwt_get_cycles() - start;
    
    // Report results
    sh_print("\\nResults:\\n");
    sh_print("FLOAT_CYCLES=");
    sh_print_int(float_cycles / {num_samples});
    sh_print("\\n");
    sh_print("LUT_CYCLES=");
    sh_print_int(lut_cycles / {num_samples});
    sh_print("\\n");
    sh_print("SPEEDUP_X10=");
    sh_print_int((float_cycles * 10) / lut_cycles);
    sh_print("\\n");
    
    sh_print("\\n=== BENCHMARK_COMPLETE ===\\n");
    
    // Exit via semihosting
    semihosting_call(0x18, (void*)0);  // SYS_EXIT
    
    while(1);
    return 0;
}}
'''
    
    main_path = output_dir / "main.c"
    main_path.write_text(main_content)
    return main_path


def generate_linker_script(output_dir: Path) -> Path:
    """Generate linker script for LM3S6965."""
    
    linker_content = '''/* Linker script for LM3S6965 (QEMU) */
MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 256K
    SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 64K
}

ENTRY(Reset_Handler)

SECTIONS
{
    .text : {
        KEEP(*(.isr_vector))
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
    } > FLASH

    .data : {
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > SRAM AT > FLASH

    .bss : {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > SRAM

    _estack = ORIGIN(SRAM) + LENGTH(SRAM);
}
'''
    
    linker_path = output_dir / "link.ld"
    linker_path.write_text(linker_content)
    return linker_path


def generate_startup(output_dir: Path) -> Path:
    """Generate startup code."""
    
    startup_content = '''/* Startup code for Cortex-M3 */
.syntax unified
.cpu cortex-m3
.thumb

.section .isr_vector, "a"
    .word _estack
    .word Reset_Handler
    .word NMI_Handler
    .word HardFault_Handler
    /* ... more handlers ... */
    .fill 240, 4, 0

.section .text
.thumb_func
.global Reset_Handler
Reset_Handler:
    /* Set stack pointer */
    ldr r0, =_estack
    mov sp, r0
    
    /* Copy .data from flash to SRAM */
    ldr r0, =_sdata
    ldr r1, =_edata
    ldr r2, =_sidata
    movs r3, #0
    b copy_data_check
copy_data_loop:
    ldr r4, [r2, r3]
    str r4, [r0, r3]
    adds r3, r3, #4
copy_data_check:
    adds r4, r0, r3
    cmp r4, r1
    bcc copy_data_loop
    
    /* Zero .bss */
    ldr r0, =_sbss
    ldr r1, =_ebss
    movs r2, #0
    b zero_bss_check
zero_bss_loop:
    str r2, [r0]
    adds r0, r0, #4
zero_bss_check:
    cmp r0, r1
    bcc zero_bss_loop
    
    /* Call main */
    bl main
    b .

.thumb_func
NMI_Handler:
HardFault_Handler:
Default_Handler:
    b .

_sidata = LOADADDR(.data);
'''
    
    startup_path = output_dir / "startup.s"
    startup_path.write_text(startup_content)
    return startup_path


# =============================================================================
# Build and Run
# =============================================================================

def check_toolchain() -> bool:
    """Check if ARM toolchain is available."""
    try:
        result = subprocess.run(
            ["arm-none-eabi-gcc", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_qemu() -> bool:
    """Check if QEMU is available."""
    try:
        result = subprocess.run(
            ["qemu-system-arm", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def build_firmware(work_dir: Path) -> Optional[Path]:
    """Build firmware with ARM GCC."""
    
    elf_path = work_dir / "firmware.elf"
    
    # Compile command
    cmd = [
        "arm-none-eabi-gcc",
        "-mcpu=cortex-m3",
        "-mthumb",
        "-O2",
        "-fno-common",
        "-ffunction-sections",
        "-fdata-sections",
        "-Wall",
        "-I", str(work_dir),
        "-T", str(work_dir / "link.ld"),
        "-nostartfiles",
        "-specs=nosys.specs",
        str(work_dir / "startup.s"),
        str(work_dir / "main.c"),
        str(work_dir / "bench_layer_data.c"),
        "-o", str(elf_path),
        "-lm",
        "-Wl,--gc-sections",
        "-Wl,-Map=" + str(work_dir / "firmware.map"),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
    
    if result.returncode != 0:
        print(f"[ERROR] Compilation failed:\n{result.stderr}")
        return None
    
    return elf_path


def run_qemu(elf_path: Path, timeout: int = 30) -> Optional[str]:
    """Run firmware on QEMU with semihosting."""
    
    cmd = [
        "qemu-system-arm",
        "-M", "lm3s6965evb",
        "-cpu", "cortex-m3",
        "-nographic",
        "-semihosting-config", "enable=on,target=native",
        "-kernel", str(elf_path),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print("[WARN] QEMU timeout")
        return None


def parse_qemu_output(output: str) -> Dict:
    """Parse benchmark results from QEMU output."""
    results = {
        "float_cycles": 0,
        "lut_cycles": 0,
        "speedup": 0.0,
    }
    
    if not output:
        return results
    
    # Parse FLOAT_CYCLES=...
    match = re.search(r"FLOAT_CYCLES=(\d+)", output)
    if match:
        results["float_cycles"] = int(match.group(1))
    
    # Parse LUT_CYCLES=...
    match = re.search(r"LUT_CYCLES=(\d+)", output)
    if match:
        results["lut_cycles"] = int(match.group(1))
    
    # Parse SPEEDUP_X10=...
    match = re.search(r"SPEEDUP_X10=(\d+)", output)
    if match:
        results["speedup"] = float(match.group(1)) / 10.0
    
    return results


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_single_benchmark(cfg: MCUConfig) -> MCUResult:
    """Run single MCU benchmark."""
    
    result = MCUResult(config=asdict(cfg))
    
    # Create adapter and build LUT
    try:
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={
                "in_dim": cfg.in_dim,
                "out_dim": cfg.out_dim,
                "degree": cfg.degree,
                "alpha": cfg.alpha,
                "beta": cfg.beta,
                "use_tanh": cfg.use_tanh,
                "x_min": cfg.x_min,
                "x_max": cfg.x_max,
                "num_knots": cfg.num_knots,
            },
            seed=cfg.seed,
        )
        
        edges = adapter.extract_edges()
        
        art = build_lut_for_edges(
            edges=edges,
            L=cfg.L,
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
        
        q_table = art.q_table[:, 0, :]  # [E, K, L] -> [E, L] (single segment for simplicity)
        scale = art.scale[:, 0]
        y_min = art.y_min[:, 0]
        coeffs = adapter.coeffs
        
    except Exception as e:
        result.status = "FAIL"
        result.error = f"Model/LUT build error: {e}"
        return result
    
    # Create temp directory for build
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        
        # Generate all C code
        generate_benchmark_c_code(cfg, q_table, scale, y_min, coeffs, work_dir)
        generate_main_c(work_dir, cfg.num_samples, cfg.use_tanh)
        generate_linker_script(work_dir)
        generate_startup(work_dir)
        
        # Build
        elf_path = build_firmware(work_dir)
        if elf_path is None:
            result.status = "FAIL"
            result.error = "Compilation failed"
            return result
        
        # Get binary size
        result.flash_bytes = elf_path.stat().st_size
        
        # Run on QEMU
        output = run_qemu(elf_path)
        if output is None:
            result.status = "FAIL"
            result.error = "QEMU timeout"
            return result
        
        # Parse results
        parsed = parse_qemu_output(output)
        result.float_cycles = parsed["float_cycles"]
        result.lut_cycles = parsed["lut_cycles"]
        result.speedup = parsed["speedup"]
        
        if result.lut_cycles == 0:
            result.status = "FAIL"
            result.error = "No benchmark results in output"
    
    return result


def main():
    parser = argparse.ArgumentParser(description="MCU QEMU Benchmark for LUT-KAN")
    
    parser.add_argument("--full", action="store_true", help="Full sweep")
    parser.add_argument("--degrees", type=str, default="3,5,10")
    parser.add_argument("--Ls", type=str, default="32,64")
    parser.add_argument("--dim", type=int, default=8, help="Layer dimension (square)")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="")
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not check_toolchain():
        print("ERROR: arm-none-eabi-gcc not found")
        print("Install with: apt install gcc-arm-none-eabi")
        sys.exit(1)
    
    if not check_qemu():
        print("ERROR: qemu-system-arm not found")
        print("Install with: apt install qemu-system-arm")
        sys.exit(1)
    
    # Parse parameters
    if args.full:
        degrees = [2, 3, 4, 5, 6, 8, 10, 15, 20]
        Ls = [16, 32, 64, 128]
        dims = [4, 8, 16]
    else:
        degrees = [int(x) for x in args.degrees.split(",")]
        Ls = [int(x) for x in args.Ls.split(",")]
        dims = [args.dim]
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"outputs/mcu_qemu_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MCU QEMU Benchmark for LUT-KAN")
    print("=" * 70)
    print(f"Degrees: {degrees}")
    print(f"LUT sizes: {Ls}")
    print(f"Dimensions: {dims}")
    print(f"Output: {out_dir}")
    print()
    
    results: List[MCUResult] = []
    total = len(degrees) * len(Ls) * len(dims)
    idx = 0
    
    for dim in dims:
        for degree in degrees:
            for L in Ls:
                idx += 1
                cfg = MCUConfig(
                    in_dim=dim,
                    out_dim=dim,
                    degree=degree,
                    L=L,
                    num_samples=args.samples,
                )
                
                print(f"[{idx}/{total}] {dim}x{dim} d={degree} L={L} ... ", end="", flush=True)
                
                result = run_single_benchmark(cfg)
                results.append(result)
                
                if result.status == "OK":
                    print(f"OK  float={result.float_cycles} lut={result.lut_cycles} speedup={result.speedup:.1f}x")
                else:
                    print(f"FAIL: {result.error}")
    
    # Save results
    json_path = out_dir / "mcu_results.json"
    with json_path.open("w") as f:
        json.dump({
            "timestamp": timestamp,
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    
    print()
    print(f"Results saved to: {json_path}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("Summary: MCU Speedup (float cycles / LUT cycles)")
    print("=" * 70)
    print(f"{'Dim':<8} {'Degree':<8} {'L':<8} {'Float':<12} {'LUT':<12} {'Speedup':<10}")
    print("-" * 70)
    
    for r in results:
        if r.status == "OK":
            c = r.config
            print(f"{c['in_dim']}x{c['out_dim']:<4} {c['degree']:<8} {c['L']:<8} {r.float_cycles:<12} {r.lut_cycles:<12} {r.speedup:.1f}x")


if __name__ == "__main__":
    main()
