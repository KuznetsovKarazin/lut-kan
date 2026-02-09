#!/usr/bin/env python3
"""Generate MCU benchmark cases for LUT-KAN (segment-wise LUT).

Supports two basis types:
  - jacobi: Jacobi polynomial recurrence (Chebyshev, Legendre, Gegenbauer, etc.)
  - bspline: Cox-de Boor B-spline (cubic by default, with SiLU base function)

Each case is a standalone PlatformIO project under mcu_auto/cases/<target>/<case_id>/
containing: platformio.ini, src/main.cpp, include/case_layer.h, diagram.json, wokwi.toml, meta.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(REPO_ROOT))

from src.models.jacobi_adapter import JacobiKANSingleLayerAdapter
from src.models.bspline_adapter import BSplineKANSingleLayerAdapter
from src.quant.lut_builder import build_lut_for_edges

# ---- Wokwi-verified targets (only boards supported by Wokwi) ----
TARGETS: Dict[str, Dict[str, str]] = {
    "uno":       {"platform": "atmelavr",     "board": "uno",                    "framework": "arduino"},
    "nano":      {"platform": "atmelavr",     "board": "nanoatmega328",          "framework": "arduino"},
    "mega":      {"platform": "atmelavr",     "board": "megaatmega2560",         "framework": "arduino"},
    "pico":      {"platform": "raspberrypi",  "board": "pico",                   "framework": "arduino"},
    "esp32":     {"platform": "espressif32",  "board": "esp32dev",               "framework": "arduino"},
    "esp32c3":   {"platform": "espressif32",  "board": "esp32-c3-devkitm-1",     "framework": "arduino"},
    "esp32s3":   {"platform": "espressif32",  "board": "esp32-s3-devkitc-1",     "framework": "arduino"},
    "stm32f103": {"platform": "ststm32",      "board": "bluepill_f103c8",        "framework": "arduino"},
}

# Targets that are NOT supported by Wokwi (documented for reference):
# - esp8266: Wokwi does not simulate ESP8266
# - stm32f401 (Nucleo): Wokwi only supports STM32F103 (Blue Pill)


@dataclass
class CaseSpec:
    """Complete specification for one benchmark case."""
    basis_type: str  # "jacobi" or "bspline"
    in_dim: int
    out_dim: int
    # Jacobi fields
    poly_family: str = "chebyshev_t"
    degree: int = 3
    alpha: float = -0.5
    beta: float = -0.5
    # B-spline fields
    bspline_degree: int = 3
    grid_points: int = 5
    base_kind: str = "silu"
    # LUT fields
    L: int = 32
    segments: int = 8
    interp: str = "linear"
    scheme: str = "uint8_asymm"
    # Domain / preprocessing
    use_tanh: bool = True
    clip_x: bool = True
    x_min: float = -3.0
    x_max: float = 3.0
    # Harness
    seed: int = 42
    iters: int = 300
    repeats: int = 7
    warmup: int = 20
    input_mode: str = "linspace"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _case_id(cs: CaseSpec) -> str:
    if cs.basis_type == "jacobi":
        return (
            f"i{cs.in_dim}o{cs.out_dim}_jacobi_{cs.poly_family}_d{cs.degree}"
            f"_L{cs.L}_K{cs.segments}_{cs.interp}_{cs.scheme}"
            f"_in{cs.input_mode}_s{cs.seed}"
        )
    else:
        return (
            f"i{cs.in_dim}o{cs.out_dim}_bspline_k{cs.bspline_degree}_g{cs.grid_points}"
            f"_L{cs.L}_K{cs.segments}_{cs.interp}_{cs.scheme}"
            f"_in{cs.input_mode}_s{cs.seed}"
        )


def _poly_params(poly_family: str) -> tuple:
    """Map a symbolic polynomial family to Jacobi (alpha, beta)."""
    pf = (poly_family or "chebyshev_t").lower()
    if pf in {"chebyshev_t", "chebyshev", "t"}:
        return -0.5, -0.5
    if pf in {"legendre", "leg"}:
        return 0.0, 0.0
    if pf.startswith("gegenbauer"):
        try:
            lam = float(pf.split("_")[-1].replace("_", "."))
        except Exception:
            lam = 1.0
        return lam - 0.5, lam - 0.5
    if pf.startswith("jacobi"):
        parts = pf.split("_")
        if len(parts) >= 3:
            return float(parts[-2]), float(parts[-1])
    return -0.5, -0.5


def _write_platformio_ini(case_dir: Path, target: str) -> None:
    t = TARGETS[target]
    content = f"""[platformio]
default_envs = mcu

[env:mcu]
platform = {t['platform']}
board = {t['board']}
framework = {t['framework']}
monitor_speed = 115200
build_flags =
  -O3
  -ffast-math
  -fno-exceptions
  -fno-rtti
"""
    (case_dir / "platformio.ini").write_text(content, encoding="utf-8")


def _float16_bits(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).view(np.uint16)


# ---------------------------------------------------------------------------
# Header export helpers
# ---------------------------------------------------------------------------

def _arr_u8(name: str, data: np.ndarray, per_line: int = 16) -> str:
    parts = []
    for i, v in enumerate(data.tolist()):
        if i % per_line == 0:
            parts.append("\n  ")
        parts.append(str(int(v)))
        if i != len(data) - 1:
            parts.append(",")
    return f"static const uint8_t {name}[{len(data)}] LUT_PROGMEM = {{{''.join(parts)}\n}};\n"


def _arr_u16(name: str, data: np.ndarray, per_line: int = 12) -> str:
    parts = []
    for i, v in enumerate(data.tolist()):
        if i % per_line == 0:
            parts.append("\n  ")
        parts.append(f"0x{int(v):04X}")
        if i != len(data) - 1:
            parts.append(",")
    return f"static const uint16_t {name}[{len(data)}] LUT_PROGMEM = {{{''.join(parts)}\n}};\n"


def _arr_f32(name: str, data: np.ndarray, per_line: int = 6) -> str:
    parts = []
    for i, v in enumerate(data.tolist()):
        if i % per_line == 0:
            parts.append("\n  ")
        parts.append(f"{float(v):.8f}f")
        if i != len(data) - 1:
            parts.append(",")
    return f"static const float {name}[{len(data)}] = {{{''.join(parts)}\n}};\n"


def _arr_f32_pgm(name: str, data: np.ndarray, per_line: int = 6) -> str:
    """Float32 array with LUT_PROGMEM (for scale/ymin in Flash on AVR)."""
    parts = []
    for i, v in enumerate(data.tolist()):
        if i % per_line == 0:
            parts.append("\n  ")
        parts.append(f"{float(v):.8f}f")
        if i != len(data) - 1:
            parts.append(",")
    return f"static const float {name}[{len(data)}] LUT_PROGMEM = {{{''.join(parts)}\n}};\n"


def _export_case_header(case_dir: Path, target: str, cs: CaseSpec,
                        art: Any, adapter: Any) -> None:
    """Write include/case_layer.h with all arrays."""

    edges = cs.in_dim * cs.out_dim
    K = cs.segments
    L = cs.L
    q_table = art.q_table

    if cs.scheme == "uint8_asymm":
        scale = art.scale.astype(np.float32)
        y_min = art.y_min.astype(np.float32)
        q_u8 = q_table.astype(np.uint8)
        scheme_asymm = 1
        q_desc = "uint8_asymm"
    elif cs.scheme == "int8_symm":
        q_i8 = q_table.astype(np.int8)
        q_u8 = (q_i8.astype(np.int16) + 128).astype(np.uint8)
        scale = art.scale.astype(np.float32)
        y_min = np.zeros_like(scale, dtype=np.float32)
        scheme_asymm = 0
        q_desc = "int8_symm"
    else:
        raise ValueError(f"Unknown scheme: {cs.scheme}")

    interp_linear = 1 if cs.interp == "linear" else 0

    q_flat = q_u8.reshape(edges * K * L)
    scale_flat = scale.reshape(edges * K)
    y_min_flat = y_min.reshape(edges * K)

    knots = np.linspace(cs.x_min, cs.x_max, cs.segments + 1).astype(np.float32)
    knots_list = ",".join(f"{float(v):.8f}f" for v in knots.tolist())

    # Basis-specific defines
    if cs.basis_type == "jacobi":
        basis_type_int = 0
        degree_for_header = cs.degree
        coeffs = adapter.coeffs.astype(np.float32)
        coeffs_flat = coeffs.reshape(edges * (cs.degree + 1))
        # B-spline placeholders (minimal to compile)
        num_coef = 1
        num_knots_aug = 2
    else:
        basis_type_int = 1
        degree_for_header = cs.bspline_degree
        num_coef = adapter.num_coef
        num_knots_aug = len(adapter.knots_aug)
        coeffs_flat = np.zeros(0, dtype=np.float32)  # not used

    out_path = case_dir / "include" / "case_layer.h"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"""#pragma once

// ---- PROGMEM support for AVR (LUT arrays go to Flash, not RAM) ----
#if defined(__AVR__)
  #include <avr/pgmspace.h>
  #define LUT_PROGMEM PROGMEM
  #define LUT_RD_U8(addr)  pgm_read_byte(addr)
  #define LUT_RD_U16(addr) pgm_read_word(addr)
  static inline float _lut_rd_f32(const float* addr) {{
    float v; memcpy_P(&v, addr, sizeof(float)); return v;
  }}
  #define LUT_RD_F32(addr) _lut_rd_f32(addr)
#else
  #define LUT_PROGMEM
  #define LUT_RD_U8(addr)  (*(addr))
  #define LUT_RD_U16(addr) (*(addr))
  #define LUT_RD_F32(addr) (*(addr))
#endif

#define CASE_TARGET \"{target}\"
#define CASE_ID \"{_case_id(cs)}\"

// Basis type: 0 = Jacobi, 1 = B-spline
#define CASE_BASIS_TYPE {basis_type_int}

// ---- Jacobi-specific ----
#define CASE_POLY_FAMILY \"{cs.poly_family}\"
#define CASE_DEGREE {cs.degree}
#define CASE_ALPHA ({cs.alpha}f)
#define CASE_BETA ({cs.beta}f)

// ---- B-spline-specific ----
#define CASE_BSPLINE_DEGREE {cs.bspline_degree}
#define CASE_NUM_COEF {num_coef}
#define CASE_NUM_KNOTS_AUG {num_knots_aug}

// ---- Shared geometry ----
#define CASE_IN_DIM {cs.in_dim}
#define CASE_OUT_DIM {cs.out_dim}
#define CASE_L {cs.L}
#define CASE_NUM_SEGMENTS {cs.segments}
#define CASE_NUM_KNOTS (CASE_NUM_SEGMENTS + 1)

#define CASE_X_MIN ({cs.x_min}f)
#define CASE_X_MAX ({cs.x_max}f)

#define CASE_USE_TANH {1 if cs.use_tanh else 0}
#define CASE_CLIP_X {1 if cs.clip_x else 0}

#define CASE_INTERP_LINEAR {interp_linear}
#define CASE_Q_SCHEME_ASYMM {scheme_asymm}
#define CASE_ITERS {cs.iters}
#define CASE_REPEATS {cs.repeats}
#define CASE_WARMUP {cs.warmup}
#define CASE_INPUT_MODE \"{cs.input_mode}\"

#define CASE_INTERP_NAME \"{cs.interp}\"
#define CASE_Q_SCHEME_NAME \"{q_desc}\"

static const float CASE_KNOTS[CASE_NUM_KNOTS] = {{ {knots_list} }};

"""

    header += _arr_u8("CASE_Q_TABLE", q_flat)
    header += "\n"
    header += _arr_f32_pgm("CASE_SCALE_F32", scale_flat)
    header += "\n"
    header += _arr_f32_pgm("CASE_YMIN_F32", y_min_flat)
    header += "\n"

    # Jacobi coefficients (always present, may be empty array for bspline)
    if cs.basis_type == "jacobi":
        header += _arr_f32("CASE_FLOAT_COEFFS", coeffs_flat)
        # Minimal bspline arrays to keep compiler happy
        header += "\nstatic const float CASE_BSPLINE_COEFFS[1] = {0};\n"
        header += "static const float CASE_KNOTS_AUG[2] = {0, 0};\n"
        header += "static const float CASE_BSPLINE_SCALES[3] = {0, 0, 0};\n"
    else:
        # Minimal jacobi array
        header += "static const float CASE_FLOAT_COEFFS[1] = {0};\n\n"

        # B-spline coefficients per edge
        bsp_coeffs = adapter.coef.astype(np.float32)  # [in_dim, out_dim, num_coef]
        # Flatten to edge order matching edge iteration (out_idx outer, in_idx inner)
        bsp_flat = np.zeros(edges * num_coef, dtype=np.float32)
        for out_j in range(cs.out_dim):
            for in_i in range(cs.in_dim):
                edge = in_i * cs.out_dim + out_j
                bsp_flat[edge * num_coef: (edge + 1) * num_coef] = bsp_coeffs[in_i, out_j, :]
        header += _arr_f32("CASE_BSPLINE_COEFFS", bsp_flat)
        header += "\n"

        # Augmented knot vector
        knots_aug = adapter.knots_aug.astype(np.float32)
        header += _arr_f32("CASE_KNOTS_AUG", knots_aug)
        header += "\n"

        # Per-edge scales: [sb, ss, m] packed as [edges * 3]
        scales_flat = np.zeros(edges * 3, dtype=np.float32)
        for out_j in range(cs.out_dim):
            for in_i in range(cs.in_dim):
                edge = in_i * cs.out_dim + out_j
                scales_flat[edge * 3 + 0] = float(adapter.sb[in_i, out_j])
                scales_flat[edge * 3 + 1] = float(adapter.ss[in_i, out_j])
                scales_flat[edge * 3 + 2] = float(adapter.m[in_i, out_j])
        header += _arr_f32("CASE_BSPLINE_SCALES", scales_flat)

    out_path.write_text(header, encoding="utf-8")


def _copy_project_template(case_dir: Path) -> None:
    tmpl = REPO_ROOT / "mcu_auto" / "templates" / "pio_project"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(tmpl, case_dir)


def _copy_wokwi_templates(case_dir: Path, target: str) -> None:
    tdir = REPO_ROOT / "mcu_auto" / "targets" / target
    shutil.copy2(tdir / "diagram.json", case_dir / "diagram.json")
    shutil.copy2(tdir / "wokwi.toml", case_dir / "wokwi.toml")


# ---------------------------------------------------------------------------
# Build a single case
# ---------------------------------------------------------------------------

# ---- Target memory limits (conservative) ----
# Flash budget = total Flash - code overhead estimate
# RAM budget = total RAM - stack/runtime overhead
TARGET_LIMITS: Dict[str, Dict[str, int]] = {
    "uno":       {"flash": 32768 - 15000,   "ram": 2048 - 800},
    "nano":      {"flash": 32768 - 15000,   "ram": 2048 - 800},
    "mega":      {"flash": 262144 - 15000,  "ram": 8192 - 2000},
    "pico":      {"flash": 2097152 - 40000, "ram": 264000 - 10000},
    "esp32":     {"flash": 4194304 - 60000, "ram": 520000 - 30000},
    "esp32c3":   {"flash": 4194304 - 60000, "ram": 400000 - 30000},
    "esp32s3":   {"flash": 8388608 - 60000, "ram": 512000 - 30000},
    "stm32f103": {"flash": 65536 - 15000,   "ram": 20480 - 4000},
}


def _estimate_case_size(cs: CaseSpec) -> dict:
    """Estimate memory needs for a case (bytes).
    With PROGMEM: Q_TABLE/SCALE/YMIN → Flash; float arrays → RAM.
    Without PROGMEM: everything → RAM (but those targets have plenty).
    """
    edges = cs.in_dim * cs.out_dim
    K = cs.segments
    q_bytes = edges * K * cs.L                     # uint8 Q table
    scale_bytes = edges * K * 4                     # float32 scale
    ymin_bytes = edges * K * 4                      # float32 y_min
    flash_for_lut = q_bytes + scale_bytes + ymin_bytes  # PROGMEM arrays

    # Float arrays stay in RAM
    if cs.basis_type == "jacobi":
        float_bytes = edges * (cs.degree + 1) * 4   # coefficients
    else:
        num_coef = cs.grid_points + cs.bspline_degree - 1
        float_bytes = edges * num_coef * 4           # bspline coeffs
        float_bytes += edges * 3 * 4                 # scales [sb,ss,m]
        num_knots_aug = cs.grid_points + 2 * cs.bspline_degree
        float_bytes += num_knots_aug * 4             # augmented knots
    knot_bytes = (K + 1) * 4
    ram_for_float = float_bytes + knot_bytes

    return {
        "flash_for_lut": flash_for_lut,
        "ram_for_float": ram_for_float,
        "q_table_bytes": q_bytes,
    }


def _build_one_case(target: str, cs: CaseSpec, out_root: Path) -> str:
    case_id = _case_id(cs)
    case_dir = out_root / target / case_id

    # Pre-flight size check
    limits = TARGET_LIMITS.get(target)
    if limits:
        sz = _estimate_case_size(cs)
        if sz["flash_for_lut"] > limits["flash"]:
            case_dir.mkdir(parents=True, exist_ok=True)
            reason = (f"LUT arrays ({sz['flash_for_lut']}B) exceed Flash budget "
                      f"({limits['flash']}B) for {target}")
            (case_dir / "SKIPPED").write_text(reason, encoding="utf-8")
            print(f"  SKIP {target}/{case_id}: {reason}")
            return case_id
        if sz["ram_for_float"] > limits["ram"]:
            case_dir.mkdir(parents=True, exist_ok=True)
            reason = (f"Float arrays ({sz['ram_for_float']}B) exceed RAM budget "
                      f"({limits['ram']}B) for {target}")
            (case_dir / "SKIPPED").write_text(reason, encoding="utf-8")
            print(f"  SKIP {target}/{case_id}: {reason}")
            return case_id

    if cs.basis_type == "jacobi":
        adapter = JacobiKANSingleLayerAdapter.from_arch(
            arch={
                "in_dim": cs.in_dim, "out_dim": cs.out_dim,
                "degree": cs.degree,
                "alpha": cs.alpha, "beta": cs.beta,
                "use_tanh": cs.use_tanh,
                "x_min": cs.x_min, "x_max": cs.x_max,
                "num_knots": cs.segments + 1,
            },
            seed=cs.seed,
        )
    else:
        adapter = BSplineKANSingleLayerAdapter.from_arch(
            arch={
                "in_dim": cs.in_dim, "out_dim": cs.out_dim,
                "degree": cs.bspline_degree,
                "grid_points": cs.grid_points,
                "x_min": cs.x_min, "x_max": cs.x_max,
                "base_kind": cs.base_kind,
            },
            seed=cs.seed,
        )

    edges = adapter.extract_edges()

    # Decouple LUT segmentation from basis-specific grid.
    # B-spline adapter uses grid_points as knots (e.g. 5 pts = 4 segs),
    # but we want cs.segments segments in the LUT for all basis types.
    from dataclasses import replace as _dc_replace
    lut_knots = np.linspace(cs.x_min, cs.x_max, cs.segments + 1).astype(np.float32)
    edges = [_dc_replace(e, knots=lut_knots) for e in edges]

    if cs.scheme == "uint8_asymm":
        dtype, scheme_name, qmin, qmax = "uint8", "asymmetric", 0, 255
    else:
        dtype, scheme_name, qmin, qmax = "int8", "symmetric", -127, 127

    art = build_lut_for_edges(
        edges=edges,
        L=cs.L,
        interp=cs.interp,
        oob_behavior="clip",
        boundary_mode="closed",
        y_range_method="minmax",
        lower_pct=0.1,
        upper_pct=99.9,
        dtype=dtype,
        scheme=scheme_name,
        qmin=qmin,
        qmax=qmax,
        meta_dtype="float16",
        value_representation="phi",
    )

    _copy_project_template(case_dir)
    _copy_wokwi_templates(case_dir, target)
    _write_platformio_ini(case_dir, target)
    _export_case_header(case_dir, target, cs, art, adapter)

    (case_dir / "meta.json").write_text(
        json.dumps(cs.__dict__, indent=2, sort_keys=True), encoding="utf-8"
    )

    return case_id


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------

def _expand_suite(grid: Dict, suite: str) -> List[CaseSpec]:
    """Expand a suite into a list of CaseSpecs, handling Jacobi/B-spline split."""
    params = grid["suites"][suite]
    base = grid.get("base", {})

    basis_types = params.get("basis_type", [base.get("basis_type", "jacobi")])
    if isinstance(basis_types, str):
        basis_types = [basis_types]

    specs: List[CaseSpec] = []

    for bt in basis_types:
        bt = str(bt).lower()

        if bt == "jacobi":
            poly_families = params.get("poly_family", [base.get("poly_family", "chebyshev_t")])
            degrees = params.get("degree", [base.get("degree", 3)])
            for in_dim, out_dim, pf, deg, L, segs, interp, scheme, im, iters, reps in itertools.product(
                params["in_dim"], params["out_dim"],
                poly_families, degrees,
                params["L"], params["segments"],
                params["interp"], params["scheme"],
                params.get("input_mode", [base.get("input_mode", "linspace")]),
                params.get("iters", [base.get("iters", 300)]),
                params.get("repeats", [base.get("repeats", 7)]),
            ):
                alpha, beta = _poly_params(str(pf))
                specs.append(CaseSpec(
                    basis_type="jacobi",
                    in_dim=int(in_dim), out_dim=int(out_dim),
                    poly_family=str(pf), degree=int(deg),
                    alpha=float(alpha), beta=float(beta),
                    L=int(L), segments=int(segs),
                    interp=str(interp), scheme=str(scheme),
                    use_tanh=bool(base.get("use_tanh", True)),
                    clip_x=bool(base.get("clip_x", True)),
                    x_min=float(base.get("x_min", -3.0)),
                    x_max=float(base.get("x_max", 3.0)),
                    seed=int(base.get("seed", 42)),
                    iters=int(iters), repeats=int(reps),
                    warmup=int(base.get("warmup", 20)),
                    input_mode=str(im),
                ))

        elif bt == "bspline":
            bsp_degrees = params.get("bspline_degree", [base.get("bspline_degree", 3)])
            grid_pts = params.get("grid_points", [base.get("grid_points", 5)])
            for in_dim, out_dim, bk, gp, L, segs, interp, scheme, im, iters, reps in itertools.product(
                params["in_dim"], params["out_dim"],
                bsp_degrees, grid_pts,
                params["L"], params["segments"],
                params["interp"], params["scheme"],
                params.get("input_mode", [base.get("input_mode", "linspace")]),
                params.get("iters", [base.get("iters", 300)]),
                params.get("repeats", [base.get("repeats", 7)]),
            ):
                specs.append(CaseSpec(
                    basis_type="bspline",
                    in_dim=int(in_dim), out_dim=int(out_dim),
                    bspline_degree=int(bk), grid_points=int(gp),
                    base_kind=str(base.get("base_kind", "silu")),
                    L=int(L), segments=int(segs),
                    interp=str(interp), scheme=str(scheme),
                    use_tanh=bool(base.get("use_tanh", True)),
                    clip_x=bool(base.get("clip_x", True)),
                    x_min=float(base.get("x_min", -3.0)),
                    x_max=float(base.get("x_max", 3.0)),
                    seed=int(base.get("seed", 42)),
                    iters=int(iters), repeats=int(reps),
                    warmup=int(base.get("warmup", 20)),
                    input_mode=str(im),
                ))

    return specs


def generate_cases(targets: List[str], grid_path: Path, out_root: Path, suite: str) -> int:
    grid = _load_yaml(grid_path)
    if suite not in grid.get("suites", {}):
        raise ValueError(f"Suite '{suite}' not found. Available: {list(grid['suites'])}")

    specs = _expand_suite(grid, suite)

    count = 0
    for target in targets:
        if target not in TARGETS:
            raise ValueError(f"Unknown target '{target}'. Known: {list(TARGETS)}")
        for cs in specs:
            _build_one_case(target, cs, out_root)
            count += 1

    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="all", help="Comma list or 'all'")
    ap.add_argument("--grid", default=str(REPO_ROOT / "mcu_auto" / "grids" / "grid.yaml"))
    ap.add_argument("--suite", default="smoke",
                    help="Suite name from grid.yaml (e.g. smoke, full, stress, extra_degrees)")
    ap.add_argument("--out", default=str(REPO_ROOT / "mcu_auto" / "cases"))
    args = ap.parse_args()

    if args.targets == "all":
        targets = list(TARGETS)
    else:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    grid_path = Path(args.grid)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    n = generate_cases(targets, grid_path, out_root, args.suite)
    print(f"Generated {n} cases under {out_root}")


if __name__ == "__main__":
    main()
