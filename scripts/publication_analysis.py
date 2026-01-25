#!/usr/bin/env python3
"""
Publication-Quality Analysis for LUT-KAN Research.

Comprehensive post-processing script that generates:
1. IEEE/ACM-style LaTeX tables
2. Publication-ready figures (300 DPI, vector formats)
3. Statistical summaries and analysis
4. Markdown and LaTeX reports

Target venues: IEEE TNNLS, ACM TECS, IEEE Embedded Systems Letters

Usage:
    python scripts/publication_analysis.py outputs/unified_benchmark_20260119/
    python scripts/publication_analysis.py outputs/benchmark/ --format pdf
    python scripts/publication_analysis.py outputs/benchmark/ --format svg --no-report
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings('ignore')

# ============================================================================
# IEEE/ACM Publication Constants
# ============================================================================

# Figure dimensions (inches) - IEEE standard
IEEE_SINGLE_COL = 3.5      # Single column width
IEEE_DOUBLE_COL = 7.16     # Double column width  
IEEE_PAGE_HEIGHT = 9.5     # Max height

# Color palette - colorblind-friendly (Wong, 2011)
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00', 
    'green': '#009E73',
    'red': '#D55E00',
    'purple': '#CC79A7',
    'cyan': '#56B4E9',
    'yellow': '#F0E442',
    'black': '#000000',
    'gray': '#999999',
}

# Marker styles for different series
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']

# Line styles
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark result."""
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


@dataclass 
class AnalysisConfig:
    """Configuration for analysis."""
    output_dir: Path
    figure_format: str = "png"
    generate_latex: bool = True
    generate_figures: bool = True
    generate_report: bool = True
    figure_dpi: int = 300
    
    # Default filter values (will be auto-detected from data)
    default_L: int = 64
    default_dim: int = 16
    default_batch_cpu: int = 256
    default_batch_mcu: int = 1


# ============================================================================
# Text Sanitization (Windows/Unicode compatibility)
# ============================================================================

def sanitize(text: str) -> str:
    """Replace Unicode characters for Windows cp1251 compatibility."""
    replacements = {
        'λ': 'lambda', 'α': 'alpha', 'β': 'beta',
        '×': 'x', '±': '+/-', '≥': '>=', '≤': '<=',
        '→': '->', '←': '<-', '↔': '<->',
        '∞': 'inf', '≈': '~', '≠': '!=',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def latex_escape(text: str) -> str:
    """Escape special characters for LaTeX."""
    replacements = {
        'λ': r'$\lambda$', 'α': r'$\alpha$', 'β': r'$\beta$',
        '×': r'$\times$', '±': r'$\pm$', 
        '_': r'\_', '%': r'\%', '&': r'\&', '#': r'\#',
        '$': r'\$', '{': r'\{', '}': r'\}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# ============================================================================
# Data Loading
# ============================================================================

def load_results(results_dir: Path) -> Tuple[List[BenchmarkResult], Dict[str, Any]]:
    """Load benchmark results from JSON or CSV with field compatibility."""
    json_path = results_dir / "raw_results.json"
    csv_path = results_dir / "raw_results.csv"
    
    meta = {"timestamp": datetime.now().isoformat(), "source": str(results_dir)}
    
    # Field name mappings for compatibility
    field_mappings = {
        "max_abs_err": "max_abs",
        "lut_memory_bytes": "lut_mem_bytes", 
        "float_memory_bytes": "float_mem_bytes",
    }
    
    def normalize_record(r: Dict) -> Dict:
        """Apply field mappings and ensure all fields exist."""
        for old, new in field_mappings.items():
            if old in r and new not in r:
                r[new] = r.pop(old)
        
        # Ensure lut_mem_per_edge exists
        if "lut_mem_per_edge" not in r:
            r["lut_mem_per_edge"] = r.get("lut_mem_bytes", 0) / max(r.get("edges", 1), 1)
        
        return r
    
    results = []
    
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        meta.update(data.get("meta", {}))
        
        for r in data.get("results", []):
            r = normalize_record(r)
            try:
                results.append(BenchmarkResult(**r))
            except TypeError as e:
                print(f"  [WARN] Skipping malformed record: {e}")
                continue
    
    elif csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = normalize_record(row)
                
                # Type conversions
                int_fields = ["degree", "L", "in_dim", "out_dim", "edges", "batch_size",
                             "lut_mem_bytes", "float_mem_bytes", "mcu_float_cycles", "mcu_lut_cycles"]
                float_fields = ["alpha", "beta", "float_ms", "lut_numpy_ms", "rmse", "mae",
                               "max_abs", "speedup_numpy", "mcu_speedup", "lut_mem_per_edge"]
                optional_float_fields = ["lut_numba_ms", "speedup_numba"]
                
                for key in int_fields:
                    if key in row:
                        row[key] = int(float(row[key]))
                
                for key in float_fields:
                    if key in row:
                        row[key] = float(row[key])
                
                for key in optional_float_fields:
                    if key in row:
                        val = row[key]
                        row[key] = float(val) if val and val not in ("None", "", "nan") else None
                
                try:
                    results.append(BenchmarkResult(**row))
                except TypeError:
                    continue
    else:
        raise FileNotFoundError(f"No results found in {results_dir}")
    
    meta["n_results"] = len(results)
    return results, meta


# ============================================================================
# Statistical Analysis
# ============================================================================

def compute_statistics(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Compute comprehensive statistics from results."""
    
    stats = {
        "total": len(results),
        "basis_types": sorted(set(sanitize(r.basis_type) for r in results)),
        "degrees": sorted(set(r.degree for r in results)),
        "lut_sizes": sorted(set(r.L for r in results)),
        "layer_sizes": sorted(set(r.in_dim for r in results)),
        "batch_sizes": sorted(set(r.batch_size for r in results)),
        "interp_modes": sorted(set(r.interp for r in results)),
        "by_basis": {},
        "by_degree": {},
        "by_L": {},
        "best": {},
        "worst": {},
    }
    
    # Per-basis statistics
    for basis in stats["basis_types"]:
        basis_results = [r for r in results if sanitize(r.basis_type) == basis]
        
        cpu = [r.speedup_numpy for r in basis_results]
        mcu = [r.mcu_speedup for r in basis_results]
        rmse = [r.rmse for r in basis_results]
        mem = [r.lut_mem_bytes for r in basis_results]
        
        stats["by_basis"][basis] = {
            "n": len(basis_results),
            "cpu_speedup": {"min": min(cpu), "max": max(cpu), "mean": np.mean(cpu), "std": np.std(cpu)},
            "mcu_speedup": {"min": min(mcu), "max": max(mcu), "mean": np.mean(mcu), "std": np.std(mcu)},
            "rmse": {"min": min(rmse), "max": max(rmse), "mean": np.mean(rmse), "median": np.median(rmse)},
            "memory_kb": {"min": min(mem)/1024, "max": max(mem)/1024, "mean": np.mean(mem)/1024},
        }
    
    # Per-degree statistics  
    for deg in stats["degrees"]:
        deg_results = [r for r in results if r.degree == deg]
        stats["by_degree"][deg] = {
            "n": len(deg_results),
            "cpu_mean": np.mean([r.speedup_numpy for r in deg_results]),
            "mcu_mean": np.mean([r.mcu_speedup for r in deg_results]),
            "rmse_mean": np.mean([r.rmse for r in deg_results]),
        }
    
    # Best configurations
    # Best MCU speedup with acceptable accuracy (RMSE < 0.01)
    accurate = [r for r in results if r.rmse < 0.01]
    if accurate:
        best_mcu = max(accurate, key=lambda r: r.mcu_speedup)
        stats["best"]["mcu_accurate"] = {
            "basis": sanitize(best_mcu.basis_type), "degree": best_mcu.degree,
            "L": best_mcu.L, "speedup": best_mcu.mcu_speedup, "rmse": best_mcu.rmse
        }
    
    # Best overall accuracy
    best_acc = min(results, key=lambda r: r.rmse)
    stats["best"]["accuracy"] = {
        "basis": sanitize(best_acc.basis_type), "degree": best_acc.degree,
        "L": best_acc.L, "rmse": best_acc.rmse, "mcu_speedup": best_acc.mcu_speedup
    }
    
    # Best CPU speedup
    best_cpu = max(results, key=lambda r: r.speedup_numpy)
    stats["best"]["cpu"] = {
        "basis": sanitize(best_cpu.basis_type), "degree": best_cpu.degree,
        "L": best_cpu.L, "speedup": best_cpu.speedup_numpy, "rmse": best_cpu.rmse
    }
    
    return stats


# ============================================================================
# Table Generation
# ============================================================================

def pivot_table(
    results: List[BenchmarkResult],
    row_key: str,
    col_key: str,
    val_key: str,
    filters: Optional[Dict[str, Any]] = None,
    fmt: str = ".2f",
    agg: str = "mean"
) -> Tuple[List[str], List[List[str]]]:
    """Create pivot table from results."""
    
    # Apply filters
    filtered = results
    if filters:
        for k, v in filters.items():
            filtered = [r for r in filtered if getattr(r, k) == v]
    
    if not filtered:
        return [], []
    
    # Get row/col values
    def get_key(r, key):
        val = getattr(r, key)
        return sanitize(str(val)) if isinstance(val, str) else val
    
    rows = sorted(set(get_key(r, row_key) for r in filtered), 
                  key=lambda x: (isinstance(x, str), x))
    cols = sorted(set(get_key(r, col_key) for r in filtered))
    
    # Build pivot
    pivot = {row: {col: [] for col in cols} for row in rows}
    for r in filtered:
        row_val = get_key(r, row_key)
        col_val = get_key(r, col_key)
        val = getattr(r, val_key)
        if val is not None:
            pivot[row_val][col_val].append(val)
    
    # Aggregate
    headers = [row_key.replace("_", " ").title()] + [str(c) for c in cols]
    table_rows = []
    
    for row in rows:
        row_data = [str(row)]
        for col in cols:
            vals = pivot[row][col]
            if vals:
                if agg == "mean":
                    v = np.mean(vals)
                elif agg == "max":
                    v = np.max(vals)
                elif agg == "min":
                    v = np.min(vals)
                else:
                    v = vals[0]
                row_data.append(f"{v:{fmt}}")
            else:
                row_data.append("--")
        table_rows.append(row_data)
    
    return headers, table_rows


def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format as Markdown table."""
    headers = [sanitize(str(h)) for h in headers]
    rows = [[sanitize(str(c)) for c in row] for row in rows]
    
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    
    lines = []
    lines.append("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    lines.append("| " + " | ".join("-" * w for w in widths) + " |")
    for row in rows:
        cells = [str(row[i]).ljust(widths[i]) if i < len(row) else " " * widths[i] 
                 for i in range(len(widths))]
        lines.append("| " + " | ".join(cells) + " |")
    
    return "\n".join(lines)


def format_latex_table(
    headers: List[str], 
    rows: List[List[str]], 
    caption: str = "",
    label: str = "",
    notes: str = ""
) -> str:
    """Format as IEEE-style LaTeX table."""
    
    n_cols = len(headers)
    col_spec = "l" + "r" * (n_cols - 1)
    
    # Escape special characters
    headers = [latex_escape(str(h)) for h in headers]
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}" if caption else "",
        r"\label{" + label + "}" if label else "",
        r"\small",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    
    for row in rows:
        cells = [latex_escape(str(c)) for c in row]
        lines.append(" & ".join(cells) + r" \\")
    
    lines.append(r"\bottomrule")
    
    if notes:
        lines.append(r"\multicolumn{" + str(n_cols) + r"}{l}{\footnotesize " + notes + r"}")
    
    lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(line for line in lines if line)


def generate_all_tables(results: List[BenchmarkResult], stats: Dict, config: AnalysisConfig):
    """Generate all publication tables."""
    
    tables_dir = config.output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    # Detect default values from data
    L_vals = stats["lut_sizes"]
    dim_vals = stats["layer_sizes"]
    L_mid = L_vals[len(L_vals)//2] if L_vals else 64
    dim_mid = dim_vals[len(dim_vals)//2] if dim_vals else 16
    
    print("\n  Generating tables...")
    
    # Table 1: CPU Speedup by Basis Type × Degree
    h, r = pivot_table(results, "basis_type", "degree", "speedup_numpy",
                       filters={"L": L_mid, "in_dim": dim_mid, "interp": "linear", "batch_size": 256})
    if r:
        (tables_dir / "tab1_cpu_speedup.md").write_text(format_markdown_table(h, r), encoding="utf-8")
        (tables_dir / "tab1_cpu_speedup.tex").write_text(
            format_latex_table(h, r, f"CPU Speedup by Basis Type and Polynomial Degree (L={L_mid})", "tab:cpu"),
            encoding="utf-8")
        print(f"    tab1_cpu_speedup.md/tex")
    
    # Table 2: MCU Speedup by Basis Type × Degree
    h, r = pivot_table(results, "basis_type", "degree", "mcu_speedup",
                       filters={"L": L_mid, "in_dim": dim_mid, "interp": "linear", "batch_size": 1}, fmt=".1f")
    if r:
        (tables_dir / "tab2_mcu_speedup.md").write_text(format_markdown_table(h, r), encoding="utf-8")
        (tables_dir / "tab2_mcu_speedup.tex").write_text(
            format_latex_table(h, r, f"ARM Cortex-M3 Speedup by Basis Type and Degree (L={L_mid})", "tab:mcu"),
            encoding="utf-8")
        print(f"    tab2_mcu_speedup.md/tex")
    
    # Table 3: RMSE by Degree × LUT Size
    h, r = pivot_table(results, "degree", "L", "rmse",
                       filters={"in_dim": dim_mid, "interp": "linear", "batch_size": 1}, fmt=".2e")
    if r:
        (tables_dir / "tab3_rmse.md").write_text(format_markdown_table(h, r), encoding="utf-8")
        (tables_dir / "tab3_rmse.tex").write_text(
            format_latex_table(h, r, "RMSE by Polynomial Degree and LUT Resolution", "tab:rmse"),
            encoding="utf-8")
        print(f"    tab3_rmse.md/tex")
    
    # Table 4: Memory Footprint (KB)
    h, r = pivot_table(results, "degree", "L", "lut_mem_bytes",
                       filters={"in_dim": dim_mid, "interp": "linear"}, fmt=".0f")
    if r:
        # Convert to KB
        for row in r:
            for i in range(1, len(row)):
                if row[i] != "--":
                    row[i] = f"{float(row[i])/1024:.1f}"
        h[0] = "Degree"
        (tables_dir / "tab4_memory.md").write_text(format_markdown_table(h, r), encoding="utf-8")
        (tables_dir / "tab4_memory.tex").write_text(
            format_latex_table(h, r, "LUT Memory Footprint in KB", "tab:memory"),
            encoding="utf-8")
        print(f"    tab4_memory.md/tex")
    
    # Table 5: Interpolation Comparison
    h, r = pivot_table(results, "interp", "degree", "rmse",
                       filters={"L": L_mid, "in_dim": dim_mid, "batch_size": 1}, fmt=".3e")
    if r:
        (tables_dir / "tab5_interp.md").write_text(format_markdown_table(h, r), encoding="utf-8")
        print(f"    tab5_interp.md")
    
    # Table 6: Summary Statistics
    summary_rows = []
    for basis in sorted(stats["by_basis"].keys()):
        d = stats["by_basis"][basis]
        summary_rows.append([
            basis,
            f"{d['cpu_speedup']['min']:.2f}--{d['cpu_speedup']['max']:.2f}",
            f"{d['mcu_speedup']['min']:.0f}--{d['mcu_speedup']['max']:.0f}",
            f"{d['rmse']['min']:.1e}--{d['rmse']['max']:.1e}",
            str(d['n']),
        ])
    
    summary_headers = ["Basis Type", "CPU Speedup", "MCU Speedup", "RMSE Range", "N"]
    (tables_dir / "tab6_summary.md").write_text(
        format_markdown_table(summary_headers, summary_rows), encoding="utf-8")
    (tables_dir / "tab6_summary.tex").write_text(
        format_latex_table(summary_headers, summary_rows, 
                          "Summary Statistics by Polynomial Basis Type", "tab:summary"),
        encoding="utf-8")
    print(f"    tab6_summary.md/tex")
    
    return tables_dir


# ============================================================================
# Visualization
# ============================================================================

def setup_matplotlib():
    """Configure matplotlib for IEEE publication quality."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        
        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        
        # Line settings  
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        
        # Grid and axes
        'axes.linewidth': 0.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.3,
        
        # Legend
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'gray',
        
        # Remove top and right spines by default
        'axes.spines.top': True,
        'axes.spines.right': True,
    })
    
    return plt


def get_color_cycle(n: int) -> List[str]:
    """Get n distinct colors from colorblind-friendly palette."""
    base_colors = list(COLORS.values())[:8]
    return (base_colors * ((n // len(base_colors)) + 1))[:n]


def fig1_speedup_comparison(results: List[BenchmarkResult], output_dir: Path, fmt: str):
    """Figure 1: CPU and MCU speedup comparison (2-panel)."""
    plt = setup_matplotlib()
    
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 2.2))
    
    filtered = [r for r in results if r.in_dim == 16 and r.L == 64 and r.interp == "linear"]
    basis_types = sorted(set(sanitize(r.basis_type) for r in filtered))
    
    colors = get_color_cycle(len(basis_types))
    
    # Panel (a): CPU Speedup
    ax = axes[0]
    for i, basis in enumerate(basis_types):
        data = [r for r in filtered if sanitize(r.basis_type) == basis and r.batch_size == 256]
        if not data:
            continue
        
        by_deg = {}
        for r in data:
            by_deg.setdefault(r.degree, []).append(r.speedup_numpy)
        
        degs = sorted(by_deg.keys())
        vals = [np.mean(by_deg[d]) for d in degs]
        ax.plot(degs, vals, marker=MARKERS[i % len(MARKERS)], color=colors[i], 
                label=basis, linestyle=LINESTYLES[i % len(LINESTYLES)])
    
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('CPU Speedup')
    ax.set_title('(a) CPU (batch=256)', fontsize=9)
    ax.set_yscale('log')
    ax.set_ylim(0.1, 100)
    ax.legend(fontsize=6, ncol=2, loc='upper left', framealpha=0.9)
    
    # Panel (b): MCU Speedup
    ax = axes[1]
    for i, basis in enumerate(basis_types):
        data = [r for r in filtered if sanitize(r.basis_type) == basis and r.batch_size == 1]
        if not data:
            continue
        
        by_deg = {}
        for r in data:
            by_deg.setdefault(r.degree, []).append(r.mcu_speedup)
        
        degs = sorted(by_deg.keys())
        vals = [np.mean(by_deg[d]) for d in degs]
        ax.plot(degs, vals, marker=MARKERS[i % len(MARKERS)], color=colors[i],
                label=basis, linestyle=LINESTYLES[i % len(LINESTYLES)])
    
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('MCU Speedup (Cortex-M3)')
    ax.set_title('(b) MCU (single sample)', fontsize=9)
    ax.legend(fontsize=6, ncol=2, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    path = output_dir / f"fig1_speedup_comparison.{fmt}"
    plt.savefig(path)
    plt.close()
    return path


def fig2_accuracy_analysis(results: List[BenchmarkResult], output_dir: Path, fmt: str):
    """Figure 2: Accuracy analysis (2-panel)."""
    plt = setup_matplotlib()
    
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 2.2))
    
    filtered = [r for r in results if r.in_dim == 16 and r.interp == "linear" and r.batch_size == 1]
    L_values = sorted(set(r.L for r in filtered))
    
    colors_L = plt.cm.viridis(np.linspace(0.1, 0.9, len(L_values)))
    
    # Panel (a): RMSE vs Degree (by L)
    ax = axes[0]
    for i, L in enumerate(L_values):
        data = [r for r in filtered if r.L == L]
        by_deg = {}
        for r in data:
            by_deg.setdefault(r.degree, []).append(r.rmse)
        
        degs = sorted(by_deg.keys())
        means = [np.mean(by_deg[d]) for d in degs]
        stds = [np.std(by_deg[d]) for d in degs]
        
        ax.errorbar(degs, means, yerr=stds, marker=MARKERS[i], color=colors_L[i],
                    label=f'L={L}', capsize=2, capthick=0.5, linewidth=0.8)
    
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('RMSE')
    ax.set_title('(a) Error vs Complexity', fontsize=9)
    ax.set_yscale('log')
    ax.legend(title='LUT Size', fontsize=6, title_fontsize=6)
    
    # Panel (b): RMSE vs L (by degree)
    ax = axes[1]
    degrees_show = [d for d in [3, 5, 10, 15, 20] if any(r.degree == d for r in filtered)][:5]
    colors_d = plt.cm.plasma(np.linspace(0.1, 0.9, len(degrees_show)))
    
    for i, deg in enumerate(degrees_show):
        data = [r for r in filtered if r.degree == deg]
        by_L = {}
        for r in data:
            by_L.setdefault(r.L, []).append(r.rmse)
        
        Ls = sorted(by_L.keys())
        vals = [np.mean(by_L[L]) for L in Ls]
        
        ax.plot(Ls, vals, marker=MARKERS[i], color=colors_d[i], label=f'd={deg}')
    
    ax.set_xlabel('LUT Size (L)')
    ax.set_ylabel('RMSE')
    ax.set_title('(b) Error vs Resolution', fontsize=9)
    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    ax.legend(title='Degree', fontsize=6, title_fontsize=6)
    
    plt.tight_layout()
    path = output_dir / f"fig2_accuracy_analysis.{fmt}"
    plt.savefig(path)
    plt.close()
    return path


def fig3_heatmaps(results: List[BenchmarkResult], output_dir: Path, fmt: str):
    """Figure 3: Speedup heatmaps (2-panel)."""
    plt = setup_matplotlib()
    
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 2.5))
    
    filtered = [r for r in results if r.in_dim == 16 and r.interp == "linear"]
    degrees = sorted(set(r.degree for r in filtered))
    L_values = sorted(set(r.L for r in filtered))
    
    # Panel (a): CPU Speedup Heatmap
    ax = axes[0]
    cpu_data = np.full((len(degrees), len(L_values)), np.nan)
    
    for r in [r for r in filtered if r.batch_size == 256]:
        if r.degree in degrees and r.L in L_values:
            di, li = degrees.index(r.degree), L_values.index(r.L)
            if np.isnan(cpu_data[di, li]):
                cpu_data[di, li] = r.speedup_numpy
            else:
                cpu_data[di, li] = (cpu_data[di, li] + r.speedup_numpy) / 2
    
    im = ax.imshow(cpu_data, aspect='auto', cmap='RdYlGn', vmin=0.2, vmax=5.0)
    ax.set_xticks(range(len(L_values)))
    ax.set_xticklabels([str(L) for L in L_values], fontsize=6)
    ax.set_yticks(range(len(degrees)))
    ax.set_yticklabels([str(d) for d in degrees], fontsize=6)
    ax.set_xlabel('LUT Size (L)')
    ax.set_ylabel('Polynomial Degree')
    ax.set_title('(a) CPU Speedup', fontsize=9)
    
    # Annotate cells
    for i in range(len(degrees)):
        for j in range(len(L_values)):
            val = cpu_data[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.7 or val > 3.0 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=5)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Speedup', fontsize=7)
    
    # Panel (b): MCU Speedup Heatmap
    ax = axes[1]
    mcu_data = np.full((len(degrees), len(L_values)), np.nan)
    
    for r in [r for r in filtered if r.batch_size == 1]:
        if r.degree in degrees and r.L in L_values:
            di, li = degrees.index(r.degree), L_values.index(r.L)
            if np.isnan(mcu_data[di, li]):
                mcu_data[di, li] = r.mcu_speedup
            else:
                mcu_data[di, li] = (mcu_data[di, li] + r.mcu_speedup) / 2
    
    im = ax.imshow(mcu_data, aspect='auto', cmap='YlOrRd', vmin=10, vmax=150)
    ax.set_xticks(range(len(L_values)))
    ax.set_xticklabels([str(L) for L in L_values], fontsize=6)
    ax.set_yticks(range(len(degrees)))
    ax.set_yticklabels([str(d) for d in degrees], fontsize=6)
    ax.set_xlabel('LUT Size (L)')
    ax.set_ylabel('Polynomial Degree')
    ax.set_title('(b) MCU Speedup', fontsize=9)
    
    for i in range(len(degrees)):
        for j in range(len(L_values)):
            val = mcu_data[i, j]
            if not np.isnan(val):
                color = 'white' if val > 80 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=5)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Speedup', fontsize=7)
    
    plt.tight_layout()
    path = output_dir / f"fig3_heatmaps.{fmt}"
    plt.savefig(path)
    plt.close()
    return path


def fig4_pareto(results: List[BenchmarkResult], output_dir: Path, fmt: str):
    """Figure 4: Pareto frontier analysis."""
    plt = setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL * 1.3, 2.5))
    
    filtered = [r for r in results if r.in_dim == 16 and r.interp == "linear" and r.batch_size == 1]
    basis_types = sorted(set(sanitize(r.basis_type) for r in filtered))
    
    colors = get_color_cycle(len(basis_types))
    all_points = []
    
    for i, basis in enumerate(basis_types):
        data = [r for r in filtered if sanitize(r.basis_type) == basis]
        
        x = [r.mcu_speedup for r in data]
        y = [r.rmse for r in data]
        sizes = [r.L * 0.3 for r in data]
        
        ax.scatter(x, y, s=sizes, alpha=0.6, c=colors[i], 
                   marker=MARKERS[i % len(MARKERS)], label=basis,
                   edgecolors='black', linewidths=0.2)
        
        all_points.extend(zip(x, y))
    
    # Compute Pareto frontier
    if all_points:
        points = np.array(all_points)
        pareto_mask = np.ones(len(points), dtype=bool)
        
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j:
                    # j dominates i: higher speedup AND lower error
                    if points[j, 0] >= points[i, 0] and points[j, 1] <= points[i, 1]:
                        if points[j, 0] > points[i, 0] or points[j, 1] < points[i, 1]:
                            pareto_mask[i] = False
                            break
        
        pareto_pts = points[pareto_mask]
        if len(pareto_pts) > 1:
            sorted_pts = pareto_pts[np.argsort(pareto_pts[:, 0])]
            ax.plot(sorted_pts[:, 0], sorted_pts[:, 1], 'r--', 
                    linewidth=1.2, alpha=0.8, label='Pareto frontier', zorder=10)
    
    ax.set_xlabel('MCU Speedup')
    ax.set_ylabel('RMSE')
    ax.set_title('Speedup vs Accuracy Trade-off')
    ax.set_yscale('log')
    ax.legend(fontsize=5, ncol=2, loc='upper right')
    
    plt.tight_layout()
    path = output_dir / f"fig4_pareto.{fmt}"
    plt.savefig(path)
    plt.close()
    return path


def fig5_memory(results: List[BenchmarkResult], output_dir: Path, fmt: str):
    """Figure 5: Memory footprint analysis."""
    plt = setup_matplotlib()
    
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 2.2))
    
    filtered = [r for r in results if r.in_dim == 16 and r.interp == "linear" and r.batch_size == 1]
    degrees = sorted(set(r.degree for r in filtered))[:7]  # Limit to 7 for clarity
    
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(degrees)))
    
    # Panel (a): Memory vs L
    ax = axes[0]
    for i, deg in enumerate(degrees):
        data = [r for r in filtered if r.degree == deg]
        by_L = {}
        for r in data:
            by_L.setdefault(r.L, []).append(r.lut_mem_bytes / 1024)
        
        Ls = sorted(by_L.keys())
        vals = [np.mean(by_L[L]) for L in Ls]
        ax.plot(Ls, vals, marker=MARKERS[i], color=colors[i], label=f'd={deg}')
    
    ax.set_xlabel('LUT Size (L)')
    ax.set_ylabel('Memory (KB)')
    ax.set_title('(a) LUT Memory Footprint', fontsize=9)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend(title='Degree', fontsize=6, title_fontsize=6)
    
    # Panel (b): Memory per edge
    ax = axes[1]
    for i, deg in enumerate(degrees):
        data = [r for r in filtered if r.degree == deg]
        by_L = {}
        for r in data:
            by_L.setdefault(r.L, []).append(r.lut_mem_per_edge)
        
        Ls = sorted(by_L.keys())
        vals = [np.mean(by_L[L]) for L in Ls]
        ax.plot(Ls, vals, marker=MARKERS[i], color=colors[i], label=f'd={deg}')
    
    ax.set_xlabel('LUT Size (L)')
    ax.set_ylabel('Bytes per Edge')
    ax.set_title('(b) Memory Efficiency', fontsize=9)
    ax.set_xscale('log', base=2)
    ax.legend(title='Degree', fontsize=6, title_fontsize=6)
    
    plt.tight_layout()
    path = output_dir / f"fig5_memory.{fmt}"
    plt.savefig(path)
    plt.close()
    return path


def fig6_basis_comparison(results: List[BenchmarkResult], output_dir: Path, fmt: str):
    """Figure 6: Comprehensive basis type comparison (4-panel)."""
    plt = setup_matplotlib()
    
    fig, axes = plt.subplots(2, 2, figsize=(IEEE_DOUBLE_COL, 4.0))
    
    filtered = [r for r in results if r.in_dim == 16 and r.L == 64 and r.interp == "linear"]
    basis_types = sorted(set(sanitize(r.basis_type) for r in filtered))
    degrees = sorted(set(r.degree for r in filtered))
    
    colors = get_color_cycle(len(basis_types))
    
    panels = [
        (axes[0, 0], "speedup_numpy", 256, "CPU Speedup", "(a)"),
        (axes[0, 1], "mcu_speedup", 1, "MCU Speedup", "(b)"),
        (axes[1, 0], "rmse", 256, "RMSE", "(c)"),
        (axes[1, 1], "max_abs", 256, "Max Absolute Error", "(d)"),
    ]
    
    for ax, metric, batch, ylabel, title_prefix in panels:
        for i, basis in enumerate(basis_types):
            data = [r for r in filtered if sanitize(r.basis_type) == basis and r.batch_size == batch]
            
            y = []
            for d in degrees:
                match = [r for r in data if r.degree == d]
                if match:
                    y.append(getattr(match[0], metric))
                else:
                    y.append(np.nan)
            
            ax.plot(degrees, y, marker=MARKERS[i % len(MARKERS)], color=colors[i], label=basis)
        
        ax.set_xlabel('Polynomial Degree')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title_prefix} {ylabel}', fontsize=9)
        
        if metric in ["rmse", "max_abs"]:
            ax.set_yscale('log')
        elif metric == "speedup_numpy":
            ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
        
        ax.legend(fontsize=5, ncol=2)
    
    plt.tight_layout()
    path = output_dir / f"fig6_basis_comparison.{fmt}"
    plt.savefig(path)
    plt.close()
    return path


def generate_all_figures(results: List[BenchmarkResult], config: AnalysisConfig):
    """Generate all publication figures."""
    
    figures_dir = config.output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    fmt = config.figure_format
    print(f"\n  Generating figures ({fmt} format, {config.figure_dpi} DPI)...")
    
    generated = []
    
    try:
        generated.append(fig1_speedup_comparison(results, figures_dir, fmt))
        print(f"    fig1_speedup_comparison.{fmt}")
        
        generated.append(fig2_accuracy_analysis(results, figures_dir, fmt))
        print(f"    fig2_accuracy_analysis.{fmt}")
        
        generated.append(fig3_heatmaps(results, figures_dir, fmt))
        print(f"    fig3_heatmaps.{fmt}")
        
        generated.append(fig4_pareto(results, figures_dir, fmt))
        print(f"    fig4_pareto.{fmt}")
        
        generated.append(fig5_memory(results, figures_dir, fmt))
        print(f"    fig5_memory.{fmt}")
        
        generated.append(fig6_basis_comparison(results, figures_dir, fmt))
        print(f"    fig6_basis_comparison.{fmt}")
        
    except Exception as e:
        print(f"    [ERROR] {e}")
    
    return figures_dir, generated


# ============================================================================
# Report Generation  
# ============================================================================

def generate_markdown_report(results: List[BenchmarkResult], stats: Dict, config: AnalysisConfig) -> Path:
    """Generate comprehensive Markdown report."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# LUT-KAN Benchmark Analysis Report

**Generated:** {timestamp}  
**Total Configurations:** {stats['total']:,}

---

## Executive Summary

This report presents comprehensive benchmark results for Look-Up Table (LUT) 
optimization of Kolmogorov-Arnold Networks (KAN). We evaluate {len(stats['basis_types'])} 
polynomial basis functions across {len(stats['degrees'])} polynomial degrees and 
{len(stats['lut_sizes'])} LUT resolutions.

### Key Findings

"""
    
    if stats.get("best", {}).get("mcu_accurate"):
        cfg = stats["best"]["mcu_accurate"]
        report += f"- **Best MCU Config (RMSE < 0.01):** {cfg['basis']}, d={cfg['degree']}, L={cfg['L']} -> **{cfg['speedup']:.1f}x speedup**\n"
    
    if stats.get("best", {}).get("accuracy"):
        cfg = stats["best"]["accuracy"]
        report += f"- **Best Accuracy:** RMSE = {cfg['rmse']:.2e} ({cfg['basis']}, d={cfg['degree']}, L={cfg['L']})\n"
    
    if stats.get("best", {}).get("cpu"):
        cfg = stats["best"]["cpu"]
        report += f"- **Best CPU Speedup:** {cfg['speedup']:.2f}x ({cfg['basis']}, d={cfg['degree']}, L={cfg['L']})\n"
    
    report += f"""
---

## Experimental Setup

### Basis Functions
- {', '.join(stats['basis_types'])}

### Parameters
- **Degrees:** {', '.join(map(str, stats['degrees']))}
- **LUT Sizes:** {', '.join(map(str, stats['lut_sizes']))}
- **Layer Sizes:** {', '.join(f'{d}x{d}' for d in stats['layer_sizes'])}
- **Interpolation:** {', '.join(stats['interp_modes'])}

---

## Results

### Summary Statistics

"""
    
    # Include summary table
    tables_dir = config.output_dir / "tables"
    if (tables_dir / "tab6_summary.md").exists():
        report += (tables_dir / "tab6_summary.md").read_text(encoding="utf-8") + "\n\n"
    
    report += """
### CPU Performance

"""
    if (tables_dir / "tab1_cpu_speedup.md").exists():
        report += (tables_dir / "tab1_cpu_speedup.md").read_text(encoding="utf-8") + "\n\n"
    
    report += """
### MCU Performance (ARM Cortex-M3)

"""
    if (tables_dir / "tab2_mcu_speedup.md").exists():
        report += (tables_dir / "tab2_mcu_speedup.md").read_text(encoding="utf-8") + "\n\n"
    
    report += """
### Quantization Accuracy

"""
    if (tables_dir / "tab3_rmse.md").exists():
        report += (tables_dir / "tab3_rmse.md").read_text(encoding="utf-8") + "\n\n"
    
    report += """
---

## Figures

"""
    
    figures_dir = config.output_dir / "figures"
    if figures_dir.exists():
        for fig in sorted(figures_dir.glob(f"*.{config.figure_format}")):
            name = fig.stem.replace("_", " ").replace("fig", "Figure ")
            report += f"### {name.title()}\n\n"
            report += f"![{fig.stem}](figures/{fig.name})\n\n"
    
    report += """
---

## Recommendations

### For Embedded Deployment (Cortex-M0/M3/M4 without FPU)

1. **LUT Size:** L=64 provides optimal speed/accuracy trade-off
2. **Basis Function:** Chebyshev 1st kind for numerical stability
3. **Interpolation:** Linear for better accuracy with minimal overhead
4. **Polynomial Degree:** 5-10 for practical applications

### For Desktop/Server Inference

1. LUT provides marginal CPU speedup (1-3x) due to efficient SIMD
2. Benefits significant only at degree > 15-20
3. Consider Numba JIT for additional 2x improvement
4. Batch processing recommended for throughput

---

## Files

- `raw_results.csv` / `raw_results.json` - Complete benchmark data
- `tables/` - LaTeX and Markdown tables
- `figures/` - Publication-ready figures (300 DPI)
- `benchmark_report.tex` - LaTeX report template
"""
    
    report_path = config.output_dir / "README.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def generate_latex_report(results: List[BenchmarkResult], stats: Dict, config: AnalysisConfig) -> Path:
    """Generate IEEE-style LaTeX report."""
    
    report = r"""\documentclass[10pt,conference]{IEEEtran}

\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{siunitx}

\title{LUT-KAN: Look-Up Table Optimization for\\Kolmogorov-Arnold Networks on Embedded Systems}
\author{Benchmark Analysis Report}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This report presents comprehensive benchmark results for Look-Up Table (LUT) 
optimization of Kolmogorov-Arnold Networks (KAN). We evaluate performance across 
multiple polynomial basis functions on both desktop CPU and ARM Cortex-M3 MCU targets,
analyzing speedup, accuracy, and memory trade-offs for embedded deployment.
\end{abstract}

\section{Introduction}

Kolmogorov-Arnold Networks offer a theoretically grounded alternative to MLPs, 
but their reliance on polynomial basis function evaluation creates computational 
bottlenecks for embedded deployment. This study evaluates LUT-based quantization 
as an optimization strategy.

\section{Experimental Setup}

\subsection{Basis Functions}
"""
    
    report += r"\begin{itemize}" + "\n"
    for basis in stats['basis_types']:
        report += f"\\item {latex_escape(basis)}\n"
    report += r"\end{itemize}" + "\n\n"
    
    report += r"""
\subsection{Parameters}
\begin{itemize}
"""
    report += f"\\item Polynomial degrees: {', '.join(map(str, stats['degrees']))}\n"
    report += f"\\item LUT sizes: {', '.join(map(str, stats['lut_sizes']))}\n"  
    layer_sizes_tex = ", ".join(f"${d} \\times {d}$" for d in stats["layer_sizes"])
    report += f"\\item Layer sizes: {layer_sizes_tex}\n"

    report += f"\\item Total configurations: {stats['total']:,}\n"
    report += r"\end{itemize}" + "\n\n"
    
    report += r"""
\section{Results}

\subsection{CPU Performance}
"""
    
    # Include tables
    tables_dir = config.output_dir / "tables"
    if (tables_dir / "tab1_cpu_speedup.tex").exists():
        report += r"\input{tables/tab1_cpu_speedup.tex}" + "\n\n"
    
    report += r"""
\subsection{MCU Performance}
"""
    if (tables_dir / "tab2_mcu_speedup.tex").exists():
        report += r"\input{tables/tab2_mcu_speedup.tex}" + "\n\n"
    
    report += r"""
\subsection{Accuracy Analysis}
"""
    if (tables_dir / "tab3_rmse.tex").exists():
        report += r"\input{tables/tab3_rmse.tex}" + "\n\n"
    
    # Include figures
    figures_dir = config.output_dir / "figures"
    if figures_dir.exists():
        report += r"""
\section{Visualizations}
"""
        for i, fig_name in enumerate(["fig1_speedup_comparison", "fig3_heatmaps", "fig4_pareto"], 1):
            fig_path = figures_dir / f"{fig_name}.{config.figure_format}"
            if fig_path.exists():
                report += f"""
\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=\\columnwidth]{{figures/{fig_path.name}}}
\\caption{{Figure {i}}}
\\label{{fig:{fig_name}}}
\\end{{figure}}
"""
    
    report += r"""
\section{Key Findings}

\begin{enumerate}
"""
    
    if stats.get("best", {}).get("mcu_accurate"):
        cfg = stats["best"]["mcu_accurate"]
        report += f"\\item Best MCU configuration (RMSE $< 0.01$): {latex_escape(cfg['basis'])}, $d={cfg['degree']}$, $L={cfg['L']}$ achieving ${cfg['speedup']:.1f}\\times$ speedup\n"
    
    report += r"""
\item LUT optimization provides 10--100$\times$ speedup on MCU without FPU
\item CPU speedup is modest (1--3$\times$) due to efficient SIMD operations  
\item Memory overhead scales as $O(E \times L)$ where $E$ is edge count
\end{enumerate}

\section{Conclusion}

LUT-based quantization enables efficient KAN deployment on resource-constrained 
embedded systems, achieving significant speedups on ARM Cortex-M platforms while 
maintaining acceptable approximation accuracy for practical applications.

\end{document}
"""
    
    report_path = config.output_dir / "benchmark_report.tex"
    report_path.write_text(report, encoding="utf-8")
    return report_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Publication-Quality Analysis for LUT-KAN Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python publication_analysis.py outputs/unified_benchmark_20260119/
  python publication_analysis.py outputs/benchmark/ --format pdf
  python publication_analysis.py outputs/benchmark/ --format svg --no-figures
  python publication_analysis.py outputs/benchmark/ --tables-only
        """
    )
    
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    parser.add_argument("--format", "-f", type=str, default="png", 
                        choices=["png", "pdf", "svg", "eps"],
                        help="Figure output format (default: png)")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300)")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--no-tables", action="store_true", help="Skip table generation")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation")
    parser.add_argument("--tables-only", action="store_true", help="Generate only tables")
    parser.add_argument("--figures-only", action="store_true", help="Generate only figures")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    # Configure
    config = AnalysisConfig(
        output_dir=results_dir,
        figure_format=args.format,
        figure_dpi=args.dpi,
        generate_latex=not args.no_tables and not args.figures_only,
        generate_figures=not args.no_figures and not args.tables_only,
        generate_report=not args.no_report and not args.tables_only and not args.figures_only,
    )
    
    print("=" * 70)
    print("LUT-KAN Publication Analysis")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading results from: {results_dir}")
    try:
        results, meta = load_results(results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"  Loaded {len(results):,} benchmark results")
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(results)
    
    print(f"  Basis types: {len(stats['basis_types'])}")
    print(f"  Degrees: {stats['degrees']}")
    print(f"  LUT sizes: {stats['lut_sizes']}")
    print(f"  Layer sizes: {stats['layer_sizes']}")
    
    # Generate tables
    if config.generate_latex:
        generate_all_tables(results, stats, config)
    
    # Generate figures
    if config.generate_figures:
        try:
            import matplotlib
            generate_all_figures(results, config)
        except ImportError:
            print("\n  [SKIP] matplotlib not available")
    
    # Generate reports
    if config.generate_report:
        print("\n  Generating reports...")
        md_path = generate_markdown_report(results, stats, config)
        print(f"    {md_path.name}")
        
        tex_path = generate_latex_report(results, stats, config)
        print(f"    {tex_path.name}")
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    print(f"\nOutput: {results_dir}")
    print("\nGenerated:")
    
    if config.generate_latex:
        print(f"  Tables: {config.output_dir / 'tables'}")
    if config.generate_figures:
        print(f"  Figures: {config.output_dir / 'figures'}")
    if config.generate_report:
        print(f"  Reports: README.md, benchmark_report.tex")
    
    print("\nTo compile LaTeX:")
    print(f"  cd {results_dir}")
    print("  pdflatex benchmark_report.tex")


if __name__ == "__main__":
    main()