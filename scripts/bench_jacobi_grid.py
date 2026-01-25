#!/usr/bin/env python3
"""Publication-ready benchmark sweep: Jacobi float vs LUT.

This script is intended to produce stable, archivable benchmark tables for papers.
It sweeps BOTH:
  - Jacobi polynomial degree (degree)
  - LUT resolution per segment (L)

It measures:
  - float forward (Jacobi adapter)
  - LUT forward (dense numpy)
  - LUT forward (dense numba, if available)

And reports:
  - runtime (ms / iteration)
  - speedup (float_ms / lut_ms)
  - approximation error (RMSE, max_abs)
  - memory footprint of packed LUT (bytes)

Outputs (under outputs/benchmarks/<run_id>/):
  - results_long.csv
  - results_long.json
  - tables/*.md  (Markdown pivot tables)
  - tables/*.tex (LaTeX tabular pivot tables)

Examples:
  python scripts/bench_jacobi_grid.py --degrees 2,3,5,8,10 --Ls 16,32,64,128
  python scripts/bench_jacobi_grid.py --interp nearest,linear --N 1,2048
  python scripts/bench_jacobi_grid.py --in_dim 64 --out_dim 64 --iters 50
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Ensure project root import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.jacobi_adapter import JacobiKANSingleLayerAdapter
from src.quant.lut_builder import build_lut_for_edges
from src.kernels.lut_contract import PackedLUT, pack_dense_layer
from src.kernels.lut_backend_dense_numpy import forward_dense_numpy

try:
    from src.kernels.lut_backend_dense_numba import (
        forward_dense_numba,
        numba_available,
        warmup_numba,
    )

    HAS_NUMBA = bool(numba_available())
except Exception:
    HAS_NUMBA = False


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


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
        "numpy": getattr(np, "__version__", ""),
        "numba": "available" if HAS_NUMBA else "not_available",
    }


def _packed_mem_bytes(p: PackedLUT) -> int:
    # Keep this explicit so it is defensible in papers.
    total = 0
    total += int(p.q_flat.nbytes)
    total += int(p.scale.nbytes)
    total += int(p.y_min.nbytes)
    total += int(p.knots.nbytes)
    total += int(p.coef_base.nbytes)
    total += int(p.coef_lut.nbytes)
    total += int(p.coef_out.nbytes)
    return total


def _time_ms(fn, warmup: int, iters: int, repeats: int) -> Tuple[float, List[float]]:
    """Return (median_ms, samples_ms)."""
    # Warmup
    for _ in range(max(0, warmup)):
        fn()

    samples: List[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dt = time.perf_counter() - t0
        samples.append((dt / iters) * 1000.0)

    samples_sorted = sorted(samples)
    median = samples_sorted[len(samples_sorted) // 2]
    return float(median), samples


def _format_md_table(headers: List[str], rows: List[List[str]]) -> str:
    # Minimal markdown table generator (no external deps).
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line1, line2] + body) + "\n"


def _format_latex_table(headers: List[str], rows: List[List[str]], caption: str, label: str) -> str:
    # Conservative LaTeX tabular; user can wrap into table environment if desired.
    cols = "l" + "r" * (len(headers) - 1)
    lines = []
    lines.append("% " + caption)
    lines.append("% " + label)
    lines.append("\\begin{tabular}{" + cols + "}")
    lines.append("\\hline")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\hline")
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def _pivot(
    records: List[Dict[str, object]],
    *,
    row_key: str,
    col_key: str,
    val_key: str,
    fmt: str,
    missing: str = "",
) -> Tuple[List[str], List[List[str]]]:
    """Return (headers, rows) for a pivot table."""
    # Collect sorted unique keys.
    row_vals = sorted({int(r[row_key]) for r in records})
    col_vals = sorted({int(r[col_key]) for r in records})

    # Map (row, col) -> val
    cell: Dict[Tuple[int, int], object] = {}
    for r in records:
        cell[(int(r[row_key]), int(r[col_key]))] = r.get(val_key, None)

    headers = [row_key] + [str(c) for c in col_vals]
    rows: List[List[str]] = []

    for rv in row_vals:
        row: List[str] = [str(rv)]
        for cv in col_vals:
            v = cell.get((rv, cv), None)
            if v is None:
                row.append(missing)
            else:
                try:
                    row.append(fmt.format(float(v)))
                except Exception:
                    row.append(str(v))
        rows.append(row)

    return headers, rows


def bench_one(
    *,
    degree: int,
    L: int,
    interp: str,
    in_dim: int,
    out_dim: int,
    N: int,
    warmup: int,
    iters: int,
    repeats: int,
    seed: int,
    x_min: float,
    x_max: float,
    use_tanh: bool,
    alpha: float,
    beta: float,
    num_knots: int,
) -> Dict[str, object]:
    """Benchmark a single (degree, L, interp, N) point."""

    adapter = JacobiKANSingleLayerAdapter.from_arch(
        arch={
            "in_dim": in_dim,
            "out_dim": out_dim,
            "degree": degree,
            "alpha": alpha,
            "beta": beta,
            "use_tanh": bool(use_tanh),
            "x_min": float(x_min),
            "x_max": float(x_max),
            "num_knots": int(num_knots),
        },
        seed=int(seed),
    )

    edges = adapter.extract_edges()

    art = build_lut_for_edges(
        edges=edges,
        L=int(L),
        interp=str(interp),
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

    rng = np.random.default_rng(int(seed) + 12345)
    x = rng.normal(size=(N, in_dim)).astype(np.float32)
    x = np.clip(x, x_min, x_max)

    # Float
    y_float = adapter.forward_float(x)

    float_ms, float_samples = _time_ms(lambda: adapter.forward_float(x), warmup, iters, repeats)

    # LUT numpy
    y_lut_np = forward_dense_numpy(x, packed)
    lut_np_ms, lut_np_samples = _time_ms(lambda: forward_dense_numpy(x, packed), warmup, iters, repeats)

    # LUT numba
    lut_nb_ms: Optional[float] = None
    lut_nb_samples: Optional[List[float]] = None
    if HAS_NUMBA:
        warmup_numba(packed, in_dim=in_dim, out_dim=out_dim)
        y_lut_nb = forward_dense_numba(x, packed)
        lut_nb_ms, lut_nb_samples = _time_ms(lambda: forward_dense_numba(x, packed), warmup, iters, repeats)
        # Keep last for sanity (avoid unused warning)
        _ = y_lut_nb

    # Errors computed against numpy LUT (the artifact); numba and numpy should match.
    rmse = float(np.sqrt(np.mean((y_float - y_lut_np) ** 2)))
    max_abs = float(np.max(np.abs(y_float - y_lut_np)))

    mem_bytes = _packed_mem_bytes(packed)

    rec: Dict[str, object] = {
        "degree": int(degree),
        "L": int(L),
        "interp": str(interp),
        "in_dim": int(in_dim),
        "out_dim": int(out_dim),
        "edges": int(in_dim * out_dim),
        "N": int(N),
        "float_ms": float(float_ms),
        "lut_numpy_ms": float(lut_np_ms),
        "lut_numba_ms": float(lut_nb_ms) if lut_nb_ms is not None else None,
        "speedup_numpy": float(float_ms / lut_np_ms) if lut_np_ms > 0 else None,
        "speedup_numba": float(float_ms / lut_nb_ms) if (lut_nb_ms is not None and lut_nb_ms > 0) else None,
        "rmse": float(rmse),
        "max_abs": float(max_abs),
        "mem_bytes": int(mem_bytes),
        "mem_kb": float(mem_bytes / 1024.0),
        "mem_per_edge_bytes": float(mem_bytes / float(in_dim * out_dim)),
        "timing_samples": {
            "float_ms": float_samples,
            "lut_numpy_ms": lut_np_samples,
            "lut_numba_ms": lut_nb_samples,
        },
    }

    return rec


def write_long_csv(path: Path, records: List[Dict[str, object]]) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    # Flatten timing samples for CSV by excluding it.
    fieldnames = [f for f in fieldnames if f != "timing_samples"]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            rr = {k: v for k, v in r.items() if k != "timing_samples"}
            w.writerow(rr)


def main() -> None:
    p = argparse.ArgumentParser(description="Jacobi float vs LUT benchmark grid (degree x L)")

    p.add_argument("--degrees", type=str, default="2,3,5,8,10,15,20", help="Comma-separated degrees")
    p.add_argument("--Ls", type=str, default="16,32,64,128", help="Comma-separated LUT resolutions per segment")
    p.add_argument("--interp", type=str, default="linear", help="Comma-separated: linear,nearest")

    p.add_argument("--in_dim", type=int, default=16)
    p.add_argument("--out_dim", type=int, default=16)
    p.add_argument("--N", type=str, default="2048", help="Comma-separated batch sizes")

    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--repeats", type=int, default=5, help="Median over repeats for stability")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--x_min", type=float, default=-3.0)
    p.add_argument("--x_max", type=float, default=3.0)
    p.add_argument("--use_tanh", type=int, default=1)

    p.add_argument("--alpha", type=float, default=-0.5)
    p.add_argument("--beta", type=float, default=-0.5)
    p.add_argument("--num_knots", type=int, default=9)

    p.add_argument("--out_dir", type=str, default="", help="Override output dir")

    args = p.parse_args()

    degrees = _parse_int_list(args.degrees)
    Ls = _parse_int_list(args.Ls)
    interps = _parse_str_list(args.interp)
    Ns = _parse_int_list(args.N)

    if not degrees or not Ls or not interps or not Ns:
        raise SystemExit("degrees/Ls/interp/N must be non-empty")

    run_id = datetime.now().strftime("jacobi_grid_%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) if args.out_dir else Path("outputs/benchmarks") / run_id
    tables_dir = out_root / "tables"
    out_root.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "system": _system_meta(),
        "args": vars(args),
        "has_numba": HAS_NUMBA,
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Output dir: {out_root}")
    print(f"Grid: degrees={degrees}  Ls={Ls}  interp={interps}  N={Ns}")
    print(f"Layer: {args.in_dim}x{args.out_dim} (edges={args.in_dim * args.out_dim})")

    records: List[Dict[str, object]] = []
    total = len(degrees) * len(Ls) * len(interps) * len(Ns)
    idx = 0

    for interp in interps:
        for N in Ns:
            for degree in degrees:
                for L in Ls:
                    idx += 1
                    print(f"[{idx}/{total}] degree={degree} L={L} interp={interp} N={N}")
                    r = bench_one(
                        degree=degree,
                        L=L,
                        interp=interp,
                        in_dim=args.in_dim,
                        out_dim=args.out_dim,
                        N=N,
                        warmup=args.warmup,
                        iters=args.iters,
                        repeats=args.repeats,
                        seed=args.seed,
                        x_min=args.x_min,
                        x_max=args.x_max,
                        use_tanh=bool(args.use_tanh),
                        alpha=args.alpha,
                        beta=args.beta,
                        num_knots=args.num_knots,
                    )
                    records.append(r)

    # Save long-form artifacts
    (out_root / "results_long.json").write_text(json.dumps({"meta": meta, "records": records}, indent=2), encoding="utf-8")
    write_long_csv(out_root / "results_long.csv", records)

    # Build pivot tables per (interp, N)
    for interp in interps:
        for N in Ns:
            subset = [r for r in records if r["interp"] == interp and int(r["N"]) == int(N)]
            if not subset:
                continue

            # Choose best available runtime metric: prefer numba if present.
            val_speed = "speedup_numba" if HAS_NUMBA else "speedup_numpy"
            val_rt = "lut_numba_ms" if HAS_NUMBA else "lut_numpy_ms"

            headers, rows = _pivot(subset, row_key="degree", col_key="L", val_key=val_speed, fmt="{:.2f}")
            md = _format_md_table(headers, rows)
            (tables_dir / f"speedup_{interp}_N{N}.md").write_text(md, encoding="utf-8")
            tex = _format_latex_table(headers, rows, caption=f"Speedup (float / LUT) for interp={interp}, N={N}", label=f"tab:speedup_{interp}_N{N}")
            (tables_dir / f"speedup_{interp}_N{N}.tex").write_text(tex, encoding="utf-8")

            headers, rows = _pivot(subset, row_key="degree", col_key="L", val_key=val_rt, fmt="{:.3f}")
            md = _format_md_table(headers, rows)
            (tables_dir / f"lut_ms_{interp}_N{N}.md").write_text(md, encoding="utf-8")
            tex = _format_latex_table(headers, rows, caption=f"LUT runtime (ms/iter) for interp={interp}, N={N}", label=f"tab:lutms_{interp}_N{N}")
            (tables_dir / f"lut_ms_{interp}_N{N}.tex").write_text(tex, encoding="utf-8")

            headers, rows = _pivot(subset, row_key="degree", col_key="L", val_key="rmse", fmt="{:.6f}")
            md = _format_md_table(headers, rows)
            (tables_dir / f"rmse_{interp}_N{N}.md").write_text(md, encoding="utf-8")
            tex = _format_latex_table(headers, rows, caption=f"RMSE (float vs LUT) for interp={interp}, N={N}", label=f"tab:rmse_{interp}_N{N}")
            (tables_dir / f"rmse_{interp}_N{N}.tex").write_text(tex, encoding="utf-8")

            headers, rows = _pivot(subset, row_key="degree", col_key="L", val_key="mem_kb", fmt="{:.2f}")
            md = _format_md_table(headers, rows)
            (tables_dir / f"mem_kb_{interp}_N{N}.md").write_text(md, encoding="utf-8")
            tex = _format_latex_table(headers, rows, caption=f"Packed LUT memory (KB) for interp={interp}, N={N}", label=f"tab:memkb_{interp}_N{N}")
            (tables_dir / f"mem_kb_{interp}_N{N}.tex").write_text(tex, encoding="utf-8")

    print("Done.")
    print(f"Long results: {out_root / 'results_long.csv'}")
    print(f"Tables dir:   {tables_dir}")


if __name__ == "__main__":
    main()
