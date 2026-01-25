#!/usr/bin/env python3
"""Publication-ready MCU cycle simulation grid: Jacobi float vs LUT.

This script provides a defensible, reproducible *cost model* to explain
why LUT may be slower on desktop CPU yet faster on MCU-class targets.

It sweeps BOTH:
  - Jacobi polynomial degree (degree)
  - LUT resolution per segment (L)

Optionally sweeps multiple MCU profiles and interpolation modes.

What it outputs (under outputs/mcu_sim/<run_id>/):
  - results_long.csv
  - results_long.json
  - tables/*.md  (Markdown pivot tables)
  - tables/*.tex (LaTeX pivot tables)

NOTE
- This is a simplified model. Treat absolute cycle counts as approximate.
- The model is most valuable for *relative* comparisons and scaling trends.

Examples:
  python scripts/sim_mcu_grid.py --degrees 3,5,8,10 --Ls 16,32,64,128 --mcus cortex_m0,cortex_m3
  python scripts/sim_mcu_grid.py --interp nearest,linear --flash_load_cycles 6
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class MCUProfile:
    name: str
    # Integer ops
    int_add: int = 1
    int_mul: int = 1
    int_shift: int = 1
    int_load: int = 2
    int_store: int = 2
    # Float ops (soft-float)
    float_add: int = 70
    float_mul: int = 100
    float_div: int = 200
    float_cmp: int = 30
    # Special functions
    tanh: int = 800
    exp: int = 600
    sqrt: int = 300


MCU_PROFILES: Dict[str, MCUProfile] = {
    "cortex_m0": MCUProfile(
        name="ARM Cortex-M0 (no FPU)",
        float_add=70,
        float_mul=100,
        float_div=200,
        tanh=1000,
        exp=700,
    ),
    "cortex_m3": MCUProfile(
        name="ARM Cortex-M3 (no FPU)",
        float_add=50,
        float_mul=80,
        float_div=150,
        tanh=800,
        exp=500,
    ),
    "cortex_m4f": MCUProfile(
        name="ARM Cortex-M4F (with FPU)",
        float_add=1,
        float_mul=1,
        float_div=14,
        tanh=200,
        exp=150,
    ),
    "esp32_no_fpu": MCUProfile(
        name="ESP32 (FPU disabled)",
        float_add=60,
        float_mul=90,
        float_div=180,
        tanh=900,
        exp=600,
    ),
    "avr": MCUProfile(
        name="AVR (8-bit, no FPU)",
        int_mul=2,
        float_add=150,
        float_mul=200,
        float_div=400,
        tanh=2000,
        exp=1500,
    ),
}


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def _parse_str_list(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def _system_meta() -> Dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor() or "",
    }


def estimate_jacobi_float_cycles(
    *,
    degree: int,
    in_dim: int,
    out_dim: int,
    N: int,
    profile: MCUProfile,
    use_tanh: bool,
) -> Dict:
    """Rough cycle estimate for float Jacobi forward.

    Model assumptions (explicit so reviewers can judge):
    - input normalization: optional tanh(x)
    - recurrence to build P_0..P_degree
    - per-output linear combination of (degree+1) basis values

    This is intentionally conservative and is best used for trends.
    """

    tanh_cycles = profile.tanh if use_tanh else 0

    # Recurrence cost to build basis (rough)
    # P1 has one division; higher degrees use a handful of mul/add
    poly_mul = (4 * degree - 1)
    poly_add = (3 * degree + 1)
    poly_div = 1

    cycles_per_coord = (
        tanh_cycles
        + poly_mul * profile.float_mul
        + poly_add * profile.float_add
        + poly_div * profile.float_div
    )

    # Output accumulation per (input coord -> out_dim)
    cycles_per_coord += out_dim * (
        (degree + 1) * profile.float_mul + degree * profile.float_add
    )

    total_cycles = N * in_dim * cycles_per_coord
    return {
        "method": "jacobi_float",
        "cycles_per_sample": int(in_dim * cycles_per_coord),
        "total_cycles": int(total_cycles),
        "assumptions": {
            "use_tanh": bool(use_tanh),
            "poly_mul": int(poly_mul),
            "poly_add": int(poly_add),
            "poly_div": int(poly_div),
        },
    }


def estimate_lut_cycles(
    *,
    in_dim: int,
    out_dim: int,
    N: int,
    L: int,
    profile: MCUProfile,
    interp: str,
    flash_load_cycles: int,
    dequant_mode: str,
) -> Dict:
    """Rough cycle estimate for LUT forward.

    Parameters
    - interp: nearest|linear
    - flash_load_cycles: extra cycles per LUT load when table resides in flash
      (set to 0 to approximate cached SRAM reads)
    - dequant_mode:
        * "float_per_edge": dequant per edge with float mul/add
        * "int_accum": pure integer accumulation + final per-output dequant

    NOTE: L does not directly change cycles here, except for potential index math.
    We keep L in the record for tables; reviewers expect it.
    """

    edges = in_dim * out_dim

    load = profile.int_load + int(max(0, flash_load_cycles))

    if interp not in ("nearest", "linear"):
        raise ValueError("interp must be nearest or linear")

    if dequant_mode == "int_accum":
        # Pure integer per edge + final dequant per output.
        if interp == "nearest":
            cycles_per_edge = (
                4 * profile.int_add
                + 1 * load
                + 1 * profile.int_mul
                + 1 * profile.int_add
            )
        else:
            cycles_per_edge = (
                4 * profile.int_add
                + 2 * load
                + 2 * profile.int_mul
                + 2 * profile.int_add
            )

        cycles_final = out_dim * (profile.float_mul + profile.float_add)
        total_cycles = N * (edges * cycles_per_edge + cycles_final)
        return {
            "method": f"lut_{interp}_int_accum",
            "cycles_per_edge": int(cycles_per_edge),
            "cycles_per_sample": int(edges * cycles_per_edge + cycles_final),
            "total_cycles": int(total_cycles),
            "assumptions": {
                "flash_load_cycles": int(flash_load_cycles),
                "dequant_mode": dequant_mode,
            },
        }

    # Default: float dequant per edge
    if interp == "nearest":
        cycles_per_edge = (
            5 * profile.int_add
            + 2 * load
            + profile.float_mul
            + 2 * profile.float_add
        )
    else:
        cycles_per_edge = (
            5 * profile.int_add
            + 3 * load
            + 4 * profile.int_mul
            + 4 * profile.int_add
            + profile.float_mul
            + 2 * profile.float_add
        )

    total_cycles = N * edges * cycles_per_edge
    return {
        "method": f"lut_{interp}_float_per_edge",
        "cycles_per_edge": int(cycles_per_edge),
        "cycles_per_sample": int(edges * cycles_per_edge),
        "total_cycles": int(total_cycles),
        "assumptions": {
            "flash_load_cycles": int(flash_load_cycles),
            "dequant_mode": dequant_mode,
        },
    }


def _pivot_table(rows: List[Dict], row_key: str, col_key: str, val_key: str) -> Tuple[List[str], List[List[str]]]:
    row_vals = sorted({r[row_key] for r in rows})
    col_vals = sorted({r[col_key] for r in rows})

    header = [row_key] + [str(c) for c in col_vals]
    table: List[List[str]] = []
    for rv in row_vals:
        row: List[str] = [str(rv)]
        for cv in col_vals:
            m = [r for r in rows if r[row_key] == rv and r[col_key] == cv]
            if not m:
                row.append("-")
            else:
                v = m[0].get(val_key, None)
                if v is None:
                    row.append("-")
                elif isinstance(v, float):
                    row.append(f"{v:.3f}")
                else:
                    row.append(str(v))
        table.append(row)

    return header, table


def _write_markdown_table(path: Path, header: List[str], table: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for row in table:
            f.write("| " + " | ".join(row) + " |\n")


def _write_latex_table(path: Path, caption: str, label: str, header: List[str], table: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ncols = len(header)
    colspec = "l" + "r" * (ncols - 1)
    with path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by scripts/sim_mcu_grid.py\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n")
        f.write("\\toprule\n")
        f.write(" ".join([h.replace("_", "\\_") for h in header]) + " \\\\ \n")
        f.write("\\midrule\n")
        for row in table:
            f.write(" ".join([c.replace("_", "\\_") for c in row]) + " \\\\ \n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="MCU cycle simulation grid (degree x L)")
    ap.add_argument("--degrees", type=str, default="3,5,8,10,15,20")
    ap.add_argument("--Ls", type=str, default="16,32,64,128")
    ap.add_argument("--in_dim", type=int, default=16)
    ap.add_argument("--out_dim", type=int, default=16)
    ap.add_argument("--N", type=int, default=1)
    ap.add_argument("--mcus", type=str, default="cortex_m0")
    ap.add_argument("--interp", type=str, default="nearest,linear")
    ap.add_argument("--use_tanh", type=int, default=1, help="1 uses tanh cost in float model")
    ap.add_argument("--flash_load_cycles", type=int, default=0, help="Extra cycles per LUT read from flash")
    ap.add_argument(
        "--dequant_modes",
        type=str,
        default="float_per_edge,int_accum",
        help="Comma-separated: float_per_edge,int_accum",
    )
    ap.add_argument("--out_dir", type=str, default="")

    args = ap.parse_args()

    degrees = _parse_int_list(args.degrees)
    Ls = _parse_int_list(args.Ls)
    mcus = _parse_str_list(args.mcus)
    interps = _parse_str_list(args.interp)
    dequant_modes = _parse_str_list(args.dequant_modes)

    if not degrees or not Ls:
        raise SystemExit("--degrees and --Ls must be non-empty")

    for m in mcus:
        if m not in MCU_PROFILES:
            raise SystemExit(f"Unknown MCU profile: {m}. Choices: {', '.join(MCU_PROFILES.keys())}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.out_dir) if args.out_dir else Path("outputs/mcu_sim") / f"jacobi_grid_{run_id}"
    out_base.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "system": _system_meta(),
        "args": {
            "degrees": degrees,
            "Ls": Ls,
            "in_dim": args.in_dim,
            "out_dim": args.out_dim,
            "N": args.N,
            "mcus": mcus,
            "interp": interps,
            "use_tanh": bool(args.use_tanh),
            "flash_load_cycles": int(args.flash_load_cycles),
            "dequant_modes": dequant_modes,
        },
        "mcu_profiles": {k: asdict(v) for k, v in MCU_PROFILES.items() if k in mcus},
    }
    (out_base / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rows: List[Dict] = []

    for mcu_key in mcus:
        prof = MCU_PROFILES[mcu_key]
        for degree in degrees:
            for L in Ls:
                float_est = estimate_jacobi_float_cycles(
                    degree=degree,
                    in_dim=args.in_dim,
                    out_dim=args.out_dim,
                    N=args.N,
                    profile=prof,
                    use_tanh=bool(args.use_tanh),
                )

                for interp in interps:
                    for deq in dequant_modes:
                        lut_est = estimate_lut_cycles(
                            in_dim=args.in_dim,
                            out_dim=args.out_dim,
                            N=args.N,
                            L=L,
                            profile=prof,
                            interp=interp,
                            flash_load_cycles=int(args.flash_load_cycles),
                            dequant_mode=deq,
                        )
                        speedup = float(float_est["total_cycles"]) / float(lut_est["total_cycles"])
                        rows.append(
                            {
                                "mcu": mcu_key,
                                "mcu_name": prof.name,
                                "degree": int(degree),
                                "L": int(L),
                                "in_dim": int(args.in_dim),
                                "out_dim": int(args.out_dim),
                                "edges": int(args.in_dim * args.out_dim),
                                "N": int(args.N),
                                "use_tanh": int(bool(args.use_tanh)),
                                "interp": interp,
                                "dequant_mode": deq,
                                "float_cycles_per_sample": int(float_est["cycles_per_sample"]),
                                "lut_cycles_per_sample": int(lut_est["cycles_per_sample"]),
                                "float_total_cycles": int(float_est["total_cycles"]),
                                "lut_total_cycles": int(lut_est["total_cycles"]),
                                "lut_cycles_per_edge": int(lut_est.get("cycles_per_edge", 0)),
                                "speedup": float(speedup),
                                "flash_load_cycles": int(args.flash_load_cycles),
                            }
                        )

    # Write long-form CSV
    csv_path = out_base / "results_long.csv"
    keys = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    (out_base / "results_long.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8")

    # Generate pivot tables: one per (mcu, interp, dequant_mode)
    tables_dir = out_base / "tables"
    for mcu_key in mcus:
        for interp in interps:
            for deq in dequant_modes:
                sub = [r for r in rows if r["mcu"] == mcu_key and r["interp"] == interp and r["dequant_mode"] == deq]
                if not sub:
                    continue

                # speedup pivot degree x L
                header, table = _pivot_table(sub, "degree", "L", "speedup")
                stem = f"speedup_{mcu_key}_{interp}_{deq}".replace("-", "_")
                _write_markdown_table(tables_dir / f"{stem}.md", header, table)
                _write_latex_table(
                    tables_dir / f"{stem}.tex",
                    caption=f"Estimated speedup (float / LUT) for {MCU_PROFILES[mcu_key].name}, interp={interp}, dequant={deq}.",
                    label=f"tab:{stem}",
                    header=header,
                    table=table,
                )

                # cycles per sample pivot
                header2, table2 = _pivot_table(sub, "degree", "L", "lut_cycles_per_sample")
                stem2 = f"lut_cycles_{mcu_key}_{interp}_{deq}".replace("-", "_")
                _write_markdown_table(tables_dir / f"{stem2}.md", header2, table2)

    print(f"\nSaved MCU simulation outputs to: {out_base}")
    print(f"- {csv_path}")
    print(f"- {out_base / 'tables'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
