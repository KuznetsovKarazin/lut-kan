from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def safe_get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def as_float(x, default=float("nan")) -> float:
    try:
        return default if x is None else float(x)
    except Exception:
        return default


def find_results(root: Path) -> List[Path]:
    return sorted(root.rglob("results.json"))


def infer_run_params(j: Dict[str, Any]) -> Dict[str, Any]:
    rp = safe_get(j, "run_params", {}) or {}
    if not isinstance(rp, dict):
        rp = {}
    out = dict(rp)

    # Backward-compatible fallbacks
    out.setdefault("L", safe_get(j, "converter.L", None))
    out.setdefault("dtype", safe_get(j, "converter.quant.dtype", None))
    out.setdefault("scheme", safe_get(j, "converter.quant.scheme", None))
    out.setdefault("interp", safe_get(j, "converter.interp.mode", None))
    out.setdefault("oob_policy_mode", safe_get(j, "run_semantics.oob_policy_mode", None))
    out.setdefault("boundary_mode", safe_get(j, "run_semantics.boundary_mode", None))
    out.setdefault("value_representation", safe_get(j, "run_semantics.value_representation", None))
    out.setdefault(
        "seed",
        out.get(
            "runtime_seed",
            safe_get(j, "evaluation_inputs.seed", safe_get(j, "calibration.seed", None)),
        ),
    )

    return out


def extract_metrics(j: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def pull(prefix: str, key: str):
        out[f"{prefix}_mae"] = safe_get(j, f"{key}.mae", None)
        out[f"{prefix}_rmse"] = safe_get(j, f"{key}.rmse", None)
        out[f"{prefix}_max_abs"] = safe_get(j, f"{key}.max_abs", None)

    pull("out", "output_sanity")
    pull("in", "output_sanity_in_range")
    pull("oob", "output_sanity_oob_only")

    out["oob_any_frac"] = safe_get(j, "input_sanity.OOB_any_frac", None)

    # Baseline correctness
    out["bspline_numpy_vs_float_max_abs"] = safe_get(j, "bspline_numpy_vs_float.max_abs", None)
    out["bspline_numba_vs_float_max_abs"] = safe_get(j, "bspline_numba_vs_float.max_abs", None)

    # Speed (ms/iter)
    out["ms_float"] = safe_get(j, "speed_float.per_iter_ms", None)
    out["ms_ref"] = safe_get(j, "speed_ref.per_iter_ms", None)
    out["ms_bspline_numpy"] = safe_get(j, "speed_bspline_numpy.per_iter_ms", None)
    out["ms_bspline_numba"] = safe_get(j, "speed_bspline_numba.per_iter_ms", None)
    out["ms_lut_numpy"] = safe_get(j, "speed_dense_numpy.per_iter_ms", None)
    out["ms_lut_numba"] = safe_get(j, "speed_dense_numba.per_iter_ms", None)

    # Speedups (may be absent)
    sp_np = safe_get(j, "speedup_lut_numpy_vs_bspline_numpy", None)
    sp_nb = safe_get(j, "speedup_lut_numba_vs_bspline_numba", None)

    if sp_np is None:
        b = safe_get(j, "speed_bspline_numpy.per_iter_ms", None)
        l = safe_get(j, "speed_dense_numpy.per_iter_ms", None)
        if b and l:
            sp_np = float(b) / float(l)
    if sp_nb is None:
        b = safe_get(j, "speed_bspline_numba.per_iter_ms", None)
        l = safe_get(j, "speed_dense_numba.per_iter_ms", None)
        if b and l:
            sp_nb = float(b) / float(l)

    out["speedup_lut_numpy_vs_bspline_numpy"] = sp_np
    out["speedup_lut_numba_vs_bspline_numba"] = sp_nb

    # Memory
    out["model_bytes"] = safe_get(j, "memory.model.total_bytes", None)
    out["lut_bytes"] = safe_get(j, "memory.lut.lut_total_bytes", None)
    out["lut_over_model"] = safe_get(j, "memory.ratios.lut_over_model", None)
    if out["lut_over_model"] is None and out["model_bytes"] and out["lut_bytes"]:
        out["lut_over_model"] = float(out["lut_bytes"]) / float(out["model_bytes"])

    return out


def ci95(std: float, n: int) -> float:
    if n <= 1 or not math.isfinite(std):
        return float("nan")
    return 1.96 * std / math.sqrt(n)


def agg_mean_std_ci(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    # Results from specialized sweeps may contain metric columns that are
    # structurally present but entirely empty (for example, a speed field that
    # is not measured in an OOB-only run).  Coerce candidate metrics to numeric
    # and aggregate only columns with at least one finite/non-NaN value.
    work = df.copy()
    valid_metrics: List[str] = []
    for c in metric_cols:
        if c not in work.columns:
            continue
        work[c] = pd.to_numeric(work[c], errors="coerce")
        if work[c].notna().any():
            valid_metrics.append(c)

    g = work.groupby(group_cols, dropna=False)
    counts = g.size().rename("n__count")

    if not valid_metrics:
        return counts.reset_index()

    means = g[valid_metrics].mean().add_suffix("__mean")
    stds = g[valid_metrics].std(ddof=1).add_suffix("__std")
    out = pd.concat([counts, means, stds], axis=1).reset_index()

    for c in valid_metrics:
        std_col = f"{c}__std"
        ci_col = f"{c}__ci95"
        out[ci_col] = [
            ci95(as_float(v), int(n))
            for v, n in zip(out[std_col], out["n__count"])
        ]

    return out


def export_latex(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tex = df.to_latex(index=index, float_format=lambda x: f"{x:.6g}")
    path.write_text(tex, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs", help="Root directory to search for results.json")
    ap.add_argument("--outdir", type=str, default="outputs/summary", help="Output directory for CSV tables")
    ap.add_argument("--latex_dir", type=str, default="", help="If set, also export LaTeX tables into this directory")
    ap.add_argument(
        "--group_keys",
        type=str,
        default="oob_policy_mode,boundary_mode,value_representation,dtype,scheme,interp,L",
        help="Comma-separated grouping keys (from run_params).",
    )
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    latex_dir = Path(args.latex_dir) if args.latex_dir.strip() else None
    group_cols = [k.strip() for k in args.group_keys.split(",") if k.strip()]

    paths = find_results(root)
    if not paths:
        raise SystemExit(f"No results.json found under: {root}")

    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        rp = infer_run_params(j)
        met = extract_metrics(j)

        row: Dict[str, Any] = {
            "results_path": str(p),
            "run_dir": str(p.parent),
            "exp_name": safe_get(j, "experiment.name", None),
        }
        for k in set(group_cols + ["seed"]):
            row[k] = rp.get(k, None)
        row.update(met)
        rows.append(row)

    df = pd.DataFrame(rows)

    if "L" in df.columns:
        df["L"] = pd.to_numeric(df["L"], errors="coerce").astype("Int64")

    df.to_csv(outdir / "all_runs.csv", index=False)

    metrics_speed = [c for c in [
        "ms_bspline_numpy", "ms_lut_numpy", "speedup_lut_numpy_vs_bspline_numpy",
        "ms_bspline_numba", "ms_lut_numba", "speedup_lut_numba_vs_bspline_numba",
    ] if c in df.columns]

    metrics_acc = [c for c in [
        "in_mae", "in_max_abs",
        "oob_mae", "oob_max_abs",
        "out_mae", "out_max_abs",
        "oob_any_frac",
        "bspline_numpy_vs_float_max_abs",
        "bspline_numba_vs_float_max_abs",
    ] if c in df.columns]

    metrics_mem = [c for c in ["model_bytes", "lut_bytes", "lut_over_model"] if c in df.columns]

    t_speed = agg_mean_std_ci(df, group_cols, metrics_speed)
    t_acc = agg_mean_std_ci(df, group_cols, metrics_acc)
    t_mem = agg_mean_std_ci(df, group_cols, metrics_mem)

    t_speed.to_csv(outdir / "table_speed.csv", index=False)
    t_acc.to_csv(outdir / "table_accuracy.csv", index=False)
    t_mem.to_csv(outdir / "table_memory.csv", index=False)

    main_metrics = [c for c in [
        "in_max_abs", "in_mae",
        "oob_max_abs", "oob_any_frac",
        "speedup_lut_numba_vs_bspline_numba", "speedup_lut_numpy_vs_bspline_numpy",
        "lut_bytes", "lut_over_model",
    ] if c in df.columns]
    t_main = agg_mean_std_ci(df, group_cols, main_metrics)
    t_main.to_csv(outdir / "table_main.csv", index=False)

    if latex_dir is not None:
        export_latex(t_main, latex_dir / "table_main.tex")
        export_latex(t_speed, latex_dir / "table_speed.tex")
        export_latex(t_acc, latex_dir / "table_accuracy.tex")
        export_latex(t_mem, latex_dir / "table_memory.tex")

    print(f"[OK] all_runs.csv: {outdir/'all_runs.csv'}")
    print(f"[OK] table_main.csv: {outdir/'table_main.csv'}")
    print(f"[OK] table_speed.csv: {outdir/'table_speed.csv'}")
    print(f"[OK] table_accuracy.csv: {outdir/'table_accuracy.csv'}")
    print(f"[OK] table_memory.csv: {outdir/'table_memory.csv'}")
    if latex_dir is not None:
        print(f"[OK] LaTeX tables: {latex_dir}")


if __name__ == "__main__":
    main()
