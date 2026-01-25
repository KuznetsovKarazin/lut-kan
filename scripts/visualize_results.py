#!/usr/bin/env python3
"""
Publication-Quality Visualization for LUT-KAN Benchmarks.

Generates:
1. Speedup heatmaps (CPU and MCU)
2. Accuracy vs Complexity trade-off plots
3. Memory footprint analysis
4. Pareto frontier plots
5. Comparison bar charts

Usage:
    python scripts/visualize_results.py results/unified_benchmark_20241215_123456/
    python scripts/visualize_results.py results/unified_benchmark_20241215_123456/ --format pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Matplotlib configuration for publication quality
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Custom colormap for speedup heatmaps
SPEEDUP_CMAP = LinearSegmentedColormap.from_list(
    'speedup', ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
)


def load_results(results_dir: Path) -> Dict:
    """Load benchmark results from JSON file."""
    json_path = results_dir / "raw_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Results not found: {json_path}")
    
    with json_path.open() as f:
        return json.load(f)


def plot_speedup_heatmap(
    results: List[Dict],
    title: str,
    speedup_key: str,
    row_key: str,
    col_key: str,
    filter_fn,
    output_path: Path,
    vmin: float = 0.5,
    vmax: float = 5.0,
):
    """Generate speedup heatmap."""
    # Filter and organize data
    filtered = [r for r in results if filter_fn(r)]
    if not filtered:
        print(f"[WARN] No data for heatmap: {title}")
        return
    
    row_vals = sorted(set(r[row_key] for r in filtered))
    col_vals = sorted(set(r[col_key] for r in filtered))
    
    # Build matrix
    data = np.full((len(row_vals), len(col_vals)), np.nan)
    for r in filtered:
        ri = row_vals.index(r[row_key])
        ci = col_vals.index(r[col_key])
        val = r.get(speedup_key)
        if val is not None:
            data[ri, ci] = val
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(data, aspect='auto', cmap=SPEEDUP_CMAP, vmin=vmin, vmax=vmax)
    
    # Labels
    ax.set_xticks(range(len(col_vals)))
    ax.set_xticklabels([str(c) for c in col_vals])
    ax.set_yticks(range(len(row_vals)))
    ax.set_yticklabels([str(r) for r in row_vals])
    
    ax.set_xlabel(col_key.replace("_", " ").title())
    ax.set_ylabel(row_key.replace("_", " ").title())
    ax.set_title(title)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, label='Speedup (×)')
    
    # Annotate cells
    for i in range(len(row_vals)):
        for j in range(len(col_vals)):
            val = data[i, j]
            if not np.isnan(val):
                color = 'white' if val > 2.5 else 'black'
                ax.text(j, i, f'{val:.1f}×', ha='center', va='center', color=color, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_vs_complexity(
    results: List[Dict],
    output_path: Path,
    basis_filter: Optional[str] = None,
):
    """Plot RMSE vs polynomial degree for different LUT sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get unique L values and basis types
    L_values = sorted(set(r['L'] for r in results))
    basis_types = sorted(set(r['basis_type'] for r in results))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    # Filter to specific basis if requested
    if basis_filter:
        results = [r for r in results if r['basis_type'] == basis_filter]
        basis_types = [basis_filter]
    
    # Left plot: RMSE vs Degree (different L values)
    ax = axes[0]
    for i, L in enumerate(L_values):
        filtered = [r for r in results if r['L'] == L and r['in_dim'] == 16 and r['interp'] == 'linear' and r['batch_size'] == 256]
        if not filtered:
            continue
        
        degrees = sorted(set(r['degree'] for r in filtered))
        rmse_by_deg = {}
        for r in filtered:
            d = r['degree']
            if d not in rmse_by_deg:
                rmse_by_deg[d] = []
            rmse_by_deg[d].append(r['rmse'])
        
        x = sorted(rmse_by_deg.keys())
        y = [np.mean(rmse_by_deg[d]) for d in x]
        yerr = [np.std(rmse_by_deg[d]) for d in x]
        
        ax.errorbar(x, y, yerr=yerr, label=f'L={L}', color=colors[i], marker=markers[i % len(markers)], capsize=3)
    
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('RMSE')
    ax.set_title('Approximation Error vs Complexity')
    ax.legend(title='LUT Size')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Right plot: RMSE vs L (different degrees)
    ax = axes[1]
    degrees_to_show = [3, 5, 10, 15, 20]
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(degrees_to_show)))
    
    for i, deg in enumerate(degrees_to_show):
        filtered = [r for r in results if r['degree'] == deg and r['in_dim'] == 16 and r['interp'] == 'linear' and r['batch_size'] == 256]
        if not filtered:
            continue
        
        L_vals = sorted(set(r['L'] for r in filtered))
        rmse_by_L = {}
        for r in filtered:
            L = r['L']
            if L not in rmse_by_L:
                rmse_by_L[L] = []
            rmse_by_L[L].append(r['rmse'])
        
        x = sorted(rmse_by_L.keys())
        y = [np.mean(rmse_by_L[L]) for L in x]
        
        ax.plot(x, y, label=f'd={deg}', color=colors2[i], marker=markers[i % len(markers)])
    
    ax.set_xlabel('LUT Size (L)')
    ax.set_ylabel('RMSE')
    ax.set_title('Approximation Error vs LUT Resolution')
    ax.legend(title='Degree')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_mcu_comparison(results: List[Dict], output_path: Path):
    """Plot MCU speedup comparison across basis types."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Filter for MCU-relevant configs (batch_size=1)
    filtered = [r for r in results if r['batch_size'] == 1 and r['in_dim'] == 16 and r['interp'] == 'linear']
    
    basis_types = sorted(set(r['basis_type'] for r in filtered))
    degrees = sorted(set(r['degree'] for r in filtered))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(basis_types)))
    
    # Left: MCU speedup by degree
    ax = axes[0]
    width = 0.8 / len(basis_types)
    x = np.arange(len(degrees))
    
    for i, basis in enumerate(basis_types):
        basis_data = [r for r in filtered if r['basis_type'] == basis and r['L'] == 64]
        speedups = []
        for d in degrees:
            matching = [r for r in basis_data if r['degree'] == d]
            if matching:
                speedups.append(matching[0]['mcu_speedup'])
            else:
                speedups.append(0)
        
        ax.bar(x + i * width - 0.4 + width/2, speedups, width, label=basis, color=colors[i])
    
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('MCU Speedup (×)')
    ax.set_title('MCU Speedup by Basis Type (Cortex-M3, L=64)')
    ax.set_xticks(x)
    ax.set_xticklabels(degrees)
    ax.legend(title='Basis Type', loc='upper left')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right: Cycle breakdown
    ax = axes[1]
    sample_degrees = [3, 10, 20]
    
    for j, deg in enumerate(sample_degrees):
        # Get sample result
        sample = [r for r in filtered if r['degree'] == deg and r['basis_type'] == 'Chebyshev 1st' and r['L'] == 64]
        if not sample:
            continue
        s = sample[0]
        
        # Bar chart for float vs LUT cycles
        x_pos = j
        ax.bar(x_pos - 0.2, s['mcu_float_cycles'] / 1000, 0.35, label='Float' if j == 0 else '', color='coral')
        ax.bar(x_pos + 0.2, s['mcu_lut_cycles'] / 1000, 0.35, label='LUT' if j == 0 else '', color='steelblue')
    
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Cycles per Sample (×1000)')
    ax.set_title('MCU Cycle Count: Float vs LUT (Chebyshev 1st)')
    ax.set_xticks(range(len(sample_degrees)))
    ax.set_xticklabels([f'd={d}' for d in sample_degrees])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_memory_analysis(results: List[Dict], output_path: Path):
    """Plot memory usage analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Filter
    filtered = [r for r in results if r['in_dim'] == 16 and r['interp'] == 'linear' and r['batch_size'] == 256]
    
    # Left: Memory by L
    ax = axes[0]
    L_values = sorted(set(r['L'] for r in filtered))
    degrees = [3, 5, 10, 15, 20]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(degrees)))
    
    for i, deg in enumerate(degrees):
        mem_by_L = []
        for L in L_values:
            matching = [r for r in filtered if r['degree'] == deg and r['L'] == L]
            if matching:
                mem_by_L.append(matching[0]['lut_mem_bytes'] / 1024)  # KB
            else:
                mem_by_L.append(0)
        
        ax.plot(L_values, mem_by_L, marker='o', label=f'd={deg}', color=colors[i])
    
    ax.set_xlabel('LUT Size (L)')
    ax.set_ylabel('LUT Memory (KB)')
    ax.set_title('Memory Footprint vs LUT Resolution')
    ax.legend(title='Degree')
    ax.grid(True, alpha=0.3)
    
    # Right: Memory per edge
    ax = axes[1]
    for i, deg in enumerate(degrees):
        mem_per_edge = []
        for L in L_values:
            matching = [r for r in filtered if r['degree'] == deg and r['L'] == L]
            if matching:
                mem_per_edge.append(matching[0]['lut_mem_per_edge'])
            else:
                mem_per_edge.append(0)
        
        ax.plot(L_values, mem_per_edge, marker='s', label=f'd={deg}', color=colors[i])
    
    ax.set_xlabel('LUT Size (L)')
    ax.set_ylabel('Bytes per Edge')
    ax.set_title('Memory Efficiency (Bytes per Edge)')
    ax.legend(title='Degree')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_pareto_frontier(results: List[Dict], output_path: Path):
    """Plot Pareto frontier: Speedup vs Accuracy trade-off."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Filter
    filtered = [r for r in results if r['in_dim'] == 16 and r['interp'] == 'linear' and r['batch_size'] == 256]
    
    # Group by basis type
    basis_types = sorted(set(r['basis_type'] for r in filtered))
    colors = plt.cm.tab10(np.linspace(0, 1, len(basis_types)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    
    all_points = []
    
    for i, basis in enumerate(basis_types):
        basis_data = [r for r in filtered if r['basis_type'] == basis]
        
        x = [r['mcu_speedup'] for r in basis_data]
        y = [r['rmse'] for r in basis_data]
        sizes = [r['L'] * 2 for r in basis_data]  # Size based on L
        
        ax.scatter(x, y, s=sizes, alpha=0.6, c=[colors[i]], 
                   marker=markers[i % len(markers)], label=basis, edgecolors='black', linewidths=0.5)
        
        all_points.extend(zip(x, y))
    
    # Find and plot Pareto frontier
    points = np.array(all_points)
    if len(points) > 0:
        # Pareto optimal: max speedup, min error
        pareto_mask = np.ones(len(points), dtype=bool)
        for i, (x1, y1) in enumerate(points):
            for j, (x2, y2) in enumerate(points):
                if i != j:
                    # Point j dominates point i if x2 >= x1 and y2 <= y1 (and strictly better in one)
                    if x2 >= x1 and y2 <= y1 and (x2 > x1 or y2 < y1):
                        pareto_mask[i] = False
                        break
        
        pareto_points = points[pareto_mask]
        if len(pareto_points) > 1:
            # Sort by speedup for line
            pareto_sorted = pareto_points[np.argsort(pareto_points[:, 0])]
            ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 'r--', linewidth=2, alpha=0.7, label='Pareto frontier')
    
    ax.set_xlabel('MCU Speedup (×)')
    ax.set_ylabel('RMSE')
    ax.set_title('Pareto Frontier: Speedup vs Accuracy Trade-off')
    ax.set_yscale('log')
    ax.legend(title='Basis Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_basis_comparison(results: List[Dict], output_path: Path):
    """Comprehensive basis type comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Filter
    filtered = [r for r in results if r['in_dim'] == 16 and r['L'] == 64 and r['interp'] == 'linear']
    
    basis_types = sorted(set(r['basis_type'] for r in filtered))
    degrees = sorted(set(r['degree'] for r in filtered))
    colors = plt.cm.Set2(np.linspace(0, 1, len(basis_types)))
    
    # 1. CPU Speedup comparison
    ax = axes[0, 0]
    for i, basis in enumerate(basis_types):
        basis_data = [r for r in filtered if r['basis_type'] == basis and r['batch_size'] == 256]
        speedup_key = 'speedup_numba' if any(r.get('speedup_numba') for r in basis_data) else 'speedup_numpy'
        
        y = []
        for d in degrees:
            matching = [r for r in basis_data if r['degree'] == d]
            if matching and matching[0].get(speedup_key):
                y.append(matching[0][speedup_key])
            else:
                y.append(np.nan)
        
        ax.plot(degrees, y, marker='o', label=basis, color=colors[i])
    
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('CPU Speedup (×)')
    ax.set_title('CPU Speedup by Basis Type (L=64, N=256)')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. MCU Speedup comparison
    ax = axes[0, 1]
    for i, basis in enumerate(basis_types):
        basis_data = [r for r in filtered if r['basis_type'] == basis and r['batch_size'] == 1]
        
        y = []
        for d in degrees:
            matching = [r for r in basis_data if r['degree'] == d]
            if matching:
                y.append(matching[0]['mcu_speedup'])
            else:
                y.append(np.nan)
        
        ax.plot(degrees, y, marker='s', label=basis, color=colors[i])
    
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('MCU Speedup (×)')
    ax.set_title('MCU Speedup by Basis Type (L=64, N=1)')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. RMSE comparison
    ax = axes[1, 0]
    for i, basis in enumerate(basis_types):
        basis_data = [r for r in filtered if r['basis_type'] == basis and r['batch_size'] == 256]
        
        y = []
        for d in degrees:
            matching = [r for r in basis_data if r['degree'] == d]
            if matching:
                y.append(matching[0]['rmse'])
            else:
                y.append(np.nan)
        
        ax.plot(degrees, y, marker='^', label=basis, color=colors[i])
    
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('RMSE')
    ax.set_title('Approximation Error by Basis Type (L=64)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Max absolute error
    ax = axes[1, 1]
    for i, basis in enumerate(basis_types):
        basis_data = [r for r in filtered if r['basis_type'] == basis and r['batch_size'] == 256]
        
        y = []
        for d in degrees:
            matching = [r for r in basis_data if r['degree'] == d]
            if matching:
                y.append(matching[0]['max_abs'])
            else:
                y.append(np.nan)
        
        ax.plot(degrees, y, marker='D', label=basis, color=colors[i])
    
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Max Absolute Error')
    ax.set_title('Max Error by Basis Type (L=64)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize LUT-KAN benchmark results")
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"],
                        help="Output format")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Load results
    data = load_results(results_dir)
    results = data['results']
    
    if not results:
        print("Error: No results found")
        sys.exit(1)
    
    print(f"Loaded {len(results)} benchmark results")
    
    # Create plots directory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    ext = args.format
    
    # Generate all plots
    print("\nGenerating visualizations...")
    
    # 1. CPU Speedup heatmap (Degree x L)
    plot_speedup_heatmap(
        results,
        title="CPU Speedup: Degree × LUT Size (Chebyshev 1st, 16×16, linear)",
        speedup_key="speedup_numpy",
        row_key="degree",
        col_key="L",
        filter_fn=lambda r: r['basis_type'] == 'Chebyshev 1st' and r['in_dim'] == 16 and r['interp'] == 'linear' and r['batch_size'] == 256,
        output_path=plots_dir / f"cpu_speedup_heatmap.{ext}",
    )
    
    # 2. MCU Speedup heatmap
    plot_speedup_heatmap(
        results,
        title="MCU Speedup: Degree × LUT Size (Chebyshev 1st, 16×16)",
        speedup_key="mcu_speedup",
        row_key="degree",
        col_key="L",
        filter_fn=lambda r: r['basis_type'] == 'Chebyshev 1st' and r['in_dim'] == 16 and r['interp'] == 'linear' and r['batch_size'] == 1,
        output_path=plots_dir / f"mcu_speedup_heatmap.{ext}",
        vmin=1.0,
        vmax=50.0,
    )
    
    # 3. Accuracy vs Complexity
    plot_accuracy_vs_complexity(results, plots_dir / f"accuracy_vs_complexity.{ext}")
    
    # 4. MCU Comparison
    plot_mcu_comparison(results, plots_dir / f"mcu_comparison.{ext}")
    
    # 5. Memory Analysis
    plot_memory_analysis(results, plots_dir / f"memory_analysis.{ext}")
    
    # 6. Pareto Frontier
    plot_pareto_frontier(results, plots_dir / f"pareto_frontier.{ext}")
    
    # 7. Basis Comparison
    plot_basis_comparison(results, plots_dir / f"basis_comparison.{ext}")
    
    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
