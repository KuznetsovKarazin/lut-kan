#!/usr/bin/env python3
"""
Regenerate tables and plots from saved benchmark results.
Fixes UTF-8 encoding issue on Windows.

Usage:
    python scripts/regenerate_tables.py outputs/unified_benchmark_20260119_224639
"""

import json
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import sys


@dataclass
class BenchmarkResult:
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
    alpha: float
    beta: float
    float_ms: float
    lut_numpy_ms: float
    lut_numba_ms: Optional[float]
    speedup_numpy: float
    speedup_numba: Optional[float]
    rmse: float
    mae: float
    max_abs: float
    lut_mem_bytes: int
    lut_mem_per_edge: float
    float_mem_bytes: int
    mcu_float_cycles: int
    mcu_lut_cycles: int
    mcu_speedup: float


def load_results(results_dir: Path) -> List[BenchmarkResult]:
    """Load results from JSON or CSV file."""
    json_path = results_dir / "raw_results.json"
    csv_path = results_dir / "raw_results.csv"
    
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        results = []
        for r in data["results"]:
            # Handle field name variations
            if "max_abs_err" in r and "max_abs" not in r:
                r["max_abs"] = r.pop("max_abs_err")
            if "lut_memory_bytes" in r and "lut_mem_bytes" not in r:
                r["lut_mem_bytes"] = r.pop("lut_memory_bytes")
            if "float_memory_bytes" in r and "float_mem_bytes" not in r:
                r["float_mem_bytes"] = r.pop("float_memory_bytes")
            # Add missing field if needed
            if "lut_mem_per_edge" not in r:
                r["lut_mem_per_edge"] = r["lut_mem_bytes"] / r["edges"] if r["edges"] > 0 else 0.0
            results.append(BenchmarkResult(**r))
        return results
    
    if csv_path.exists():
        results = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle field name variations
                if "max_abs_err" in row and "max_abs" not in row:
                    row["max_abs"] = row.pop("max_abs_err")
                if "lut_memory_bytes" in row and "lut_mem_bytes" not in row:
                    row["lut_mem_bytes"] = row.pop("lut_memory_bytes")
                if "float_memory_bytes" in row and "float_mem_bytes" not in row:
                    row["float_mem_bytes"] = row.pop("float_memory_bytes")
                
                for key in ["degree", "L", "in_dim", "out_dim", "edges", "batch_size", 
                           "lut_mem_bytes", "float_mem_bytes", "mcu_float_cycles", "mcu_lut_cycles"]:
                    if key in row:
                        row[key] = int(row[key])
                for key in ["alpha", "beta", "float_ms", "lut_numpy_ms", "rmse", "mae", 
                           "max_abs", "speedup_numpy", "mcu_speedup", "lut_mem_per_edge"]:
                    if key in row:
                        row[key] = float(row[key])
                for key in ["lut_numba_ms", "speedup_numba"]:
                    if key in row:
                        row[key] = float(row[key]) if row[key] and row[key] not in ("None", "") else None
                
                # Add missing field if needed
                if "lut_mem_per_edge" not in row:
                    row["lut_mem_per_edge"] = row["lut_mem_bytes"] / row["edges"] if row["edges"] > 0 else 0.0
                
                results.append(BenchmarkResult(**row))
        return results
    
    raise FileNotFoundError(f"No results found in {results_dir}")


def sanitize_text(text: str) -> str:
    """Replace problematic Unicode characters."""
    return text.replace("λ", "lambda").replace("α", "alpha").replace("β", "beta")


def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    headers = [sanitize_text(str(h)) for h in headers]
    rows = [[sanitize_text(str(cell)) for cell in row] for row in rows]
    
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    
    lines = []
    lines.append("| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |")
    lines.append("| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |")
    for row in rows:
        lines.append("| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) + " |")
    
    return "\n".join(lines)


def format_latex_table(headers: List[str], rows: List[List[str]], caption: str = "", label: str = "") -> str:
    headers = [h.replace("λ", r"$\lambda$").replace("alpha", r"$\alpha$").replace("beta", r"$\beta$") for h in headers]
    
    n_cols = len(headers)
    col_spec = "l" + "r" * (n_cols - 1)
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}" if caption else "",
        f"\\label{{{label}}}" if label else "",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(sanitize_text(str(cell)) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    
    return "\n".join(line for line in lines if line)


def pivot_results(results: List[BenchmarkResult], row_key: str, col_key: str, val_key: str,
                  filters: Optional[Dict[str, Any]] = None, fmt: str = ".2f") -> tuple:
    filtered = results
    if filters:
        for k, v in filters.items():
            filtered = [r for r in filtered if getattr(r, k) == v]
    
    if not filtered:
        return [], []
    
    rows_set = sorted(set(getattr(r, row_key) for r in filtered), 
                      key=lambda x: (0, str(x)) if isinstance(x, str) else (1, x))
    cols_set = sorted(set(getattr(r, col_key) for r in filtered))
    
    pivot = {row: {col: [] for col in cols_set} for row in rows_set}
    for r in filtered:
        val = getattr(r, val_key)
        if val is not None:
            pivot[getattr(r, row_key)][getattr(r, col_key)].append(val)
    
    headers = [row_key] + [str(c) for c in cols_set]
    table_rows = []
    for row in rows_set:
        row_data = [str(row)]
        for col in cols_set:
            vals = pivot[row][col]
            row_data.append(f"{sum(vals)/len(vals):{fmt}}" if vals else "-")
        table_rows.append(row_data)
    
    return headers, table_rows


def generate_tables(results: List[BenchmarkResult], output_dir: Path):
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    has_numba = any(r.speedup_numba is not None for r in results)
    speedup_key = "speedup_numba" if has_numba else "speedup_numpy"
    
    # Find common filter values
    all_Ls = sorted(set(r.L for r in results))
    all_dims = sorted(set(r.in_dim for r in results))
    all_batches = sorted(set(r.batch_size for r in results))
    
    L_mid = all_Ls[len(all_Ls)//2] if all_Ls else 64
    dim_mid = all_dims[len(all_dims)//2] if all_dims else 16
    batch_mid = all_batches[len(all_batches)//2] if all_batches else 256
    
    # 1. CPU Speedup
    headers, rows = pivot_results(results, "basis_type", "degree", speedup_key,
        filters={"L": L_mid, "in_dim": dim_mid, "interp": "linear", "batch_size": batch_mid}, fmt=".2f")
    if rows:
        (tables_dir / "cpu_speedup_by_basis_degree.md").write_text(format_markdown_table(headers, rows), encoding="utf-8")
        (tables_dir / "cpu_speedup_by_basis_degree.tex").write_text(
            format_latex_table(headers, rows, "CPU Speedup by Basis Type and Degree", "tab:cpu_speedup"), encoding="utf-8")
        print("  cpu_speedup_by_basis_degree.md/tex")
    
    # 2. MCU Speedup
    headers, rows = pivot_results(results, "basis_type", "degree", "mcu_speedup",
        filters={"L": L_mid, "in_dim": dim_mid, "interp": "linear", "batch_size": 1}, fmt=".1f")
    if rows:
        (tables_dir / "mcu_speedup_by_basis_degree.md").write_text(format_markdown_table(headers, rows), encoding="utf-8")
        (tables_dir / "mcu_speedup_by_basis_degree.tex").write_text(
            format_latex_table(headers, rows, "MCU Speedup by Basis Type and Degree", "tab:mcu_speedup"), encoding="utf-8")
        print("  mcu_speedup_by_basis_degree.md/tex")
    
    # 3. RMSE for first basis type
    first_basis = results[0].basis_type if results else ""
    basis_results = [r for r in results if r.basis_type == first_basis]
    headers, rows = pivot_results(basis_results, "degree", "L", "rmse",
        filters={"in_dim": dim_mid, "interp": "linear", "batch_size": 1}, fmt=".2e")
    if rows:
        safe_name = sanitize_text(first_basis).replace(" ", "_").lower()
        (tables_dir / f"rmse_{safe_name}_degree_L.md").write_text(format_markdown_table(headers, rows), encoding="utf-8")
        print(f"  rmse_{safe_name}_degree_L.md")
    
    # 4. Summary by basis
    print("\n  === SUMMARY ===")
    by_basis = {}
    for r in results:
        if r.basis_type not in by_basis:
            by_basis[r.basis_type] = {"cpu": [], "mcu": [], "rmse": []}
        by_basis[r.basis_type]["cpu"].append(r.speedup_numpy)
        by_basis[r.basis_type]["mcu"].append(r.mcu_speedup)
        by_basis[r.basis_type]["rmse"].append(r.rmse)
    
    summary_rows = []
    for basis in sorted(by_basis.keys()):
        d = by_basis[basis]
        summary_rows.append([
            sanitize_text(basis),
            f"{min(d['cpu']):.2f}x - {max(d['cpu']):.2f}x",
            f"{min(d['mcu']):.0f}x - {max(d['mcu']):.0f}x",
            f"{min(d['rmse']):.2e} - {max(d['rmse']):.2e}",
        ])
        print(f"  {sanitize_text(basis):30s}: CPU {min(d['cpu']):6.2f}-{max(d['cpu']):6.2f}x, MCU {min(d['mcu']):5.0f}-{max(d['mcu']):5.0f}x")
    
    (tables_dir / "summary_by_basis.md").write_text(
        format_markdown_table(["Basis Type", "CPU Speedup", "MCU Speedup", "RMSE Range"], summary_rows), encoding="utf-8")
    print("\n  summary_by_basis.md")


def generate_plots(results: List[BenchmarkResult], output_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed")
        return
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Find filter values
    all_Ls = sorted(set(r.L for r in results))
    all_dims = sorted(set(r.in_dim for r in results))
    L_mid = all_Ls[len(all_Ls)//2] if all_Ls else 64
    dim_mid = all_dims[len(all_dims)//2] if all_dims else 16
    
    # 1. CPU Speedup vs Degree
    fig, ax = plt.subplots(figsize=(12, 7))
    filtered = [r for r in results if r.L == L_mid and r.in_dim == dim_mid and r.interp == "linear" and r.batch_size == 256]
    
    by_basis = {}
    for r in filtered:
        basis = sanitize_text(r.basis_type)
        if basis not in by_basis:
            by_basis[basis] = {}
        if r.degree not in by_basis[basis]:
            by_basis[basis][r.degree] = []
        by_basis[basis][r.degree].append(r.speedup_numpy)
    
    for basis in sorted(by_basis.keys()):
        degrees = sorted(by_basis[basis].keys())
        speedups = [sum(by_basis[basis][d])/len(by_basis[basis][d]) for d in degrees]
        ax.plot(degrees, speedups, marker='o', label=basis, linewidth=2, markersize=6)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Break-even')
    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('CPU Speedup (LUT / Float)', fontsize=12)
    ax.set_title(f'CPU Speedup vs Polynomial Degree (L={L_mid}, {dim_mid}x{dim_mid}, linear, N=256)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / "cpu_speedup_vs_degree.png", dpi=150)
    plt.close()
    print("  cpu_speedup_vs_degree.png")
    
    # 2. MCU Speedup vs Degree
    fig, ax = plt.subplots(figsize=(12, 7))
    filtered = [r for r in results if r.L == L_mid and r.in_dim == dim_mid and r.interp == "linear" and r.batch_size == 1]
    
    by_basis = {}
    for r in filtered:
        basis = sanitize_text(r.basis_type)
        if basis not in by_basis:
            by_basis[basis] = {}
        if r.degree not in by_basis[basis]:
            by_basis[basis][r.degree] = []
        by_basis[basis][r.degree].append(r.mcu_speedup)
    
    for basis in sorted(by_basis.keys()):
        degrees = sorted(by_basis[basis].keys())
        speedups = [sum(by_basis[basis][d])/len(by_basis[basis][d]) for d in degrees]
        ax.plot(degrees, speedups, marker='s', label=basis, linewidth=2, markersize=6)
    
    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('MCU Speedup (Float cycles / LUT cycles)', fontsize=12)
    ax.set_title(f'MCU Speedup vs Polynomial Degree (Cortex-M3, L={L_mid}, {dim_mid}x{dim_mid})', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "mcu_speedup_vs_degree.png", dpi=150)
    plt.close()
    print("  mcu_speedup_vs_degree.png")
    
    # 3. RMSE vs L
    fig, ax = plt.subplots(figsize=(12, 7))
    test_degrees = [5, 10, 15] if max(r.degree for r in results) >= 15 else [3, 5, 10]
    filtered = [r for r in results if r.in_dim == dim_mid and r.interp == "linear" and r.batch_size == 1 and r.degree in test_degrees]
    
    by_config = {}
    for r in filtered:
        key = f"{sanitize_text(r.basis_type)} d={r.degree}"
        if key not in by_config:
            by_config[key] = {}
        if r.L not in by_config[key]:
            by_config[key][r.L] = []
        by_config[key][r.L].append(r.rmse)
    
    for config in sorted(by_config.keys()):
        Ls = sorted(by_config[config].keys())
        rmses = [sum(by_config[config][L])/len(by_config[config][L]) for L in Ls]
        ax.plot(Ls, rmses, marker='o', label=config, linewidth=2)
    
    ax.set_xlabel('LUT Size (L)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'Quantization Error vs LUT Size ({dim_mid}x{dim_mid}, linear)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    plt.tight_layout()
    plt.savefig(plots_dir / "rmse_vs_L.png", dpi=150)
    plt.close()
    print("  rmse_vs_L.png")
    
    print(f"\n  Plots saved to: {plots_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python regenerate_tables.py <results_directory>")
        print("Example: python regenerate_tables.py outputs/unified_benchmark_20260119_224639")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Error: {results_dir} not found")
        sys.exit(1)
    
    print(f"Loading: {results_dir}")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} results\n")
    
    print("Generating tables...")
    generate_tables(results, results_dir)
    
    print("\nGenerating plots...")
    generate_plots(results, results_dir)
    
    print("\n" + "=" * 50)
    print("DONE!")
    print("=" * 50)


if __name__ == "__main__":
    main()
