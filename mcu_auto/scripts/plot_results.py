#!/usr/bin/env python3
"""Generate publication-quality figures from MCU benchmark results.

Reads summary.csv from mcu_auto/reports/ and produces:
  1. Speedup vs Degree (Jacobi) — line plot, per target
  2. Speedup by Basis x Target — grouped bar chart
  3. Accuracy vs Speedup Pareto front
  4. B-spline speedup by grid points
  5. LaTeX tables for paper

Usage:
    python mcu_auto/scripts/plot_results.py [--csv path/to/summary.csv]
"""

from __future__ import annotations
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    "mega":      "#E69F00",   # orange
    "pico":      "#56B4E9",   # sky blue
    "stm32f103": "#009E73",   # green
    "esp32c3":   "#CC79A7",   # pink
    "uno":       "#D55E00",   # vermilion
    "nano":      "#F0E442",   # yellow
}
MARKERS = {"mega": "s", "pico": "o", "stm32f103": "^", "esp32c3": "D", "uno": "v", "nano": "P"}
TARGET_LABELS = {
    "mega": "ATmega2560\n(AVR 8-bit, 16 MHz)",
    "pico": "RP2040\n(Cortex-M0+, 133 MHz)",
    "stm32f103": "STM32F103\n(Cortex-M3, 72 MHz)",
    "esp32c3": "ESP32-C3\n(RISC-V, 160 MHz)",
}


def load_results(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("status") != "OK":
                continue
            try:
                r["speedup"] = float(r["speedup"])
                r["max_abs_err"] = float(r["max_abs_err"])
                r["t_float_us"] = int(r["t_float_us"])
                r["t_lut_us"] = int(r["t_lut_us"])
                r["degree"] = int(r.get("degree", 0))
                r["grid_points"] = int(r.get("grid_points", 0))
                r["in_dim"] = int(r.get("in_dim", 0))
                r["out_dim"] = int(r.get("out_dim", 0))
                rows.append(r)
            except (ValueError, KeyError):
                pass
    return rows


def _median(vals):
    s = sorted(vals)
    return s[len(s) // 2]


def fig1_speedup_vs_degree(rows, out_dir):
    """Line plot: speedup vs polynomial degree (Jacobi), one line per target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jacobi = [r for r in rows if r["basis_type"] == "jacobi"]
    if not jacobi:
        print("  [skip] No Jacobi results for Fig 1")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4))

    targets = sorted(set(r["target"] for r in jacobi))
    for tgt in targets:
        subset = [r for r in jacobi if r["target"] == tgt]
        by_deg = defaultdict(list)
        for r in subset:
            by_deg[r["degree"]].append(r["speedup"])

        degs = sorted(by_deg.keys())
        medians = [_median(by_deg[d]) for d in degs]
        mins = [min(by_deg[d]) for d in degs]
        maxs = [max(by_deg[d]) for d in degs]

        color = COLORS.get(tgt, "#999999")
        marker = MARKERS.get(tgt, "o")
        label = TARGET_LABELS.get(tgt, tgt).replace("\n", " ")
        ax.plot(degs, medians, marker=marker, color=color, label=label,
                linewidth=2, markersize=7, zorder=3)
        ax.fill_between(degs, mins, maxs, alpha=0.15, color=color)

    ax.set_xlabel("Polynomial Degree", fontsize=11)
    ax.set_ylabel("Speedup (LUT / Float)", fontsize=11)
    ax.set_title("Jacobi LUT-KAN Speedup vs. Degree", fontsize=12, fontweight="bold")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = out_dir / "fig1_speedup_vs_degree.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig1_speedup_vs_degree.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def fig2_speedup_bars(rows, out_dir):
    """Grouped bar chart: median speedup by basis type x target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    targets = sorted(set(r["target"] for r in rows))
    bases = sorted(set(r["basis_type"] for r in rows))

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(targets))
    width = 0.35
    offsets = np.linspace(-width * (len(bases) - 1) / 2,
                          width * (len(bases) - 1) / 2, len(bases))

    basis_colors = {"jacobi": "#0072B2", "bspline": "#E69F00"}
    basis_labels = {"jacobi": "Jacobi (all degrees)", "bspline": "B-spline (cubic)"}

    for i, basis in enumerate(bases):
        medians, errs_lo, errs_hi = [], [], []
        for tgt in targets:
            subset = [r["speedup"] for r in rows
                      if r["target"] == tgt and r["basis_type"] == basis]
            if subset:
                s = sorted(subset)
                med = s[len(s) // 2]
                medians.append(med)
                errs_lo.append(med - s[0])
                errs_hi.append(s[-1] - med)
            else:
                medians.append(0)
                errs_lo.append(0)
                errs_hi.append(0)

        color = basis_colors.get(basis, "#999999")
        ax.bar(x + offsets[i], medians, width,
               label=basis_labels.get(basis, basis),
               color=color, alpha=0.85, zorder=3,
               yerr=[errs_lo, errs_hi], capsize=4,
               error_kw={"linewidth": 1})

    ax.set_xlabel("Target MCU", fontsize=11)
    ax.set_ylabel("Median Speedup", fontsize=11)
    ax.set_title("LUT-KAN Speedup by Basis and Platform", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    xlabels = [TARGET_LABELS.get(t, t) for t in targets]
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "fig2_speedup_by_basis.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig2_speedup_by_basis.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def fig3_pareto(rows, out_dir):
    """Scatter: accuracy vs speedup, colored by basis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4))
    basis_colors = {"jacobi": "#0072B2", "bspline": "#E69F00"}

    for basis in sorted(set(r["basis_type"] for r in rows)):
        subset = [r for r in rows if r["basis_type"] == basis]
        sp = [r["speedup"] for r in subset]
        err = [r["max_abs_err"] for r in subset]
        color = basis_colors.get(basis, "#999999")
        ax.scatter(sp, err, c=color, label=basis, alpha=0.6, s=40,
                   edgecolors="none", zorder=3)

    ax.set_xlabel("Speedup (x)", fontsize=11)
    ax.set_ylabel("Max Absolute Error", fontsize=11)
    ax.set_title("Accuracy-Speedup Trade-off", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    path = out_dir / "fig3_accuracy_vs_speedup.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig3_accuracy_vs_speedup.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def fig4_bspline_speedup(rows, out_dir):
    """B-spline speedup by grid_points x target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bsp = [r for r in rows if r["basis_type"] == "bspline"]
    if not bsp:
        print("  [skip] No B-spline results for Fig 4")
        return

    targets = sorted(set(r["target"] for r in bsp))
    fig, ax = plt.subplots(figsize=(5.5, 4))

    for tgt in targets:
        subset = [r for r in bsp if r["target"] == tgt]
        by_gp = defaultdict(list)
        for r in subset:
            by_gp[r["grid_points"]].append(r["speedup"])

        gps = sorted(by_gp.keys())
        medians = [_median(by_gp[g]) for g in gps]
        color = COLORS.get(tgt, "#999999")
        marker = MARKERS.get(tgt, "o")
        label = TARGET_LABELS.get(tgt, tgt).replace("\n", " ")
        ax.plot(gps, medians, marker=marker, color=color, label=label,
                linewidth=2, markersize=7, zorder=3)

    ax.set_xlabel("B-spline Grid Points", fontsize=11)
    ax.set_ylabel("Speedup (x)", fontsize=11)
    ax.set_title("B-spline LUT Speedup vs. Grid Complexity", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "fig4_bspline_speedup.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig4_bspline_speedup.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def table_latex(rows, out_dir):
    """Generate LaTeX tables for paper."""
    targets = sorted(set(r["target"] for r in rows))
    bases = sorted(set(r["basis_type"] for r in rows))

    # Table 1: Summary
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{LUT-KAN inference speedup across MCU platforms.}")
    lines.append(r"\label{tab:mcu_speedup}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Target & Basis & $N$ & Med.\ $\times$ & Max $\times$ & Med.\ Err \\")
    lines.append(r"\midrule")

    for tgt in targets:
        for basis in bases:
            subset = [r for r in rows if r["target"] == tgt and r["basis_type"] == basis]
            if not subset:
                continue
            sp = sorted([r["speedup"] for r in subset])
            errs = sorted([r["max_abs_err"] for r in subset])
            n = len(sp)
            lines.append(f"{tgt} & {basis} & {n} & "
                         f"{sp[n//2]:.1f}$\\times$ & {sp[-1]:.1f}$\\times$ & "
                         f"{errs[n//2]:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path = out_dir / "table_speedup.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {path}")

    # Table 2: Degree sweep
    jacobi = [r for r in rows if r["basis_type"] == "jacobi"]
    if jacobi:
        lines2 = []
        lines2.append(r"\begin{table}[t]")
        lines2.append(r"\centering")
        lines2.append(r"\caption{Jacobi LUT speedup scaling with polynomial degree.}")
        lines2.append(r"\label{tab:degree_sweep}")
        ncols = len(targets)
        lines2.append(r"\begin{tabular}{l" + "r" * ncols + "}")
        lines2.append(r"\toprule")
        lines2.append("Degree & " + " & ".join(targets) + r" \\")
        lines2.append(r"\midrule")

        all_degs = sorted(set(r["degree"] for r in jacobi))
        for d in all_degs:
            vals = []
            for tgt in targets:
                subset = [r["speedup"] for r in jacobi
                          if r["target"] == tgt and r["degree"] == d]
                if subset:
                    vals.append(f"{_median(subset):.1f}$\\times$")
                else:
                    vals.append("--")
            lines2.append(f"{d} & " + " & ".join(vals) + r" \\")

        lines2.append(r"\bottomrule")
        lines2.append(r"\end{tabular}")
        lines2.append(r"\end{table}")

        path2 = out_dir / "table_degree_sweep.tex"
        path2.write_text("\n".join(lines2), encoding="utf-8")
        print(f"  -> {path2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    csv_path = args.csv or str(repo / "mcu_auto" / "reports" / "summary.csv")
    out_dir = Path(args.out) if args.out else repo / "mcu_auto" / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {csv_path}")
    rows = load_results(csv_path)
    if not rows:
        sys.exit(f"No OK results in {csv_path}")

    print(f"Loaded {len(rows)} results")
    print(f"Targets: {sorted(set(r['target'] for r in rows))}")
    print(f"Basis types: {sorted(set(r['basis_type'] for r in rows))}")

    print("\nGenerating figures...")
    fig1_speedup_vs_degree(rows, out_dir)
    fig2_speedup_bars(rows, out_dir)
    fig3_pareto(rows, out_dir)
    fig4_bspline_speedup(rows, out_dir)

    print("\nGenerating LaTeX tables...")
    table_latex(rows, out_dir)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
