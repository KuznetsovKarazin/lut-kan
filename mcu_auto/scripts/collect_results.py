#!/usr/bin/env python3
"""Collect Wokwi MCU benchmark results.

Parses logs produced by run_wokwi.py, extracts JSON lines prefixed with 'LUTKAN:',
and emits:
  - reports/summary.csv
  - reports/summary.md

The CSV has one row per (target, case). The MD report provides quick pivots
(speedup medians per target + basis type, and top/worst cases).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS_ROOT = REPO_ROOT / "mcu_auto" / "logs"
DEFAULT_CASES_ROOT = REPO_ROOT / "mcu_auto" / "cases"
DEFAULT_REPORTS_ROOT = REPO_ROOT / "mcu_auto" / "reports"

LINE_RE = re.compile(r"LUTKAN:(\{.*\})")


def _iter_logs(logs_root: Path) -> List[Path]:
    return sorted([p for p in logs_root.rglob("*.log") if p.is_file()])


def parse_logs(logs_root: Path, cases_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in _iter_logs(logs_root):
        text = p.read_text(encoding="utf-8", errors="replace")
        target = p.parent.name
        case_id = p.stem

        case_dir = cases_root / target / case_id
        meta_path = case_dir / "meta.json"
        bm_path = case_dir / "build_metrics.json"
        meta_obj: Dict[str, Any] = {}
        bm_obj: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta_obj = {"meta_status": "BAD_META"}
        if bm_path.exists():
            try:
                bm_obj = json.loads(bm_path.read_text(encoding="utf-8"))
            except Exception:
                bm_obj = {"build_status": "BAD_BUILD_METRICS"}

        matches = LINE_RE.findall(text)
        if not matches:
            rows.append({"target": target, "case_id": case_id, "status": "NO_JSON",
                         "log": str(p), **meta_obj, **bm_obj})
            continue

        raw = matches[-1]
        try:
            obj = json.loads(raw)
            obj.setdefault("status", "OK")
            obj.setdefault("log", str(p))
            rows.append({**meta_obj, **bm_obj, **obj})
        except json.JSONDecodeError:
            rows.append({"target": target, "case_id": case_id, "status": "BAD_JSON",
                         "log": str(p), **meta_obj, **bm_obj})
    return rows


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    cols = [
        "target", "case_id", "basis_type",
        "poly_family", "input_mode", "iters", "repeats", "warmup",
        "in_dim", "out_dim", "degree", "L", "segments",
        "interp", "scheme",
        "bspline_degree", "grid_points", "num_coef",
        "t_float_us", "t_lut_us",
        "t_float_min_us", "t_float_max_us", "t_lut_min_us", "t_lut_max_us",
        "speedup", "max_abs_err",
        "flash_bytes", "ram_bytes", "pio_version",
        "status", "log"
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def write_md(rows: List[Dict[str, Any]], out_md: Path) -> None:
    ok = [r for r in rows if r.get("status") == "OK" and isinstance(r.get("speedup"), (int, float))]

    # Median speedup per (target, basis_type)
    by_group: Dict[str, List[float]] = {}
    for r in ok:
        key = f"{r.get('target', '?')} / {r.get('basis_type', '?')}"
        by_group.setdefault(key, []).append(float(r["speedup"]))

    lines: List[str] = []
    lines.append("# MCU LUT-KAN benchmark summary\n")
    lines.append(f"Total logs: {len(rows)}; OK: {len(ok)}\n")

    if by_group:
        lines.append("## Median speedup by target / basis type\n")
        lines.append("| target / basis | n | median speedup |\n|---|---:|---:|")
        for key, vals in sorted(by_group.items()):
            lines.append(f"| {key} | {len(vals)} | {median(vals):.3f} |")
        lines.append("")

    ok_sorted = sorted(ok, key=lambda r: float(r["speedup"]))
    if ok_sorted:
        lines.append("## Worst 10 speedups\n")
        lines.append("| target | basis | case_id | speedup | max_abs_err |\n|---|---|---|---:|---:|")
        for r in ok_sorted[:10]:
            lines.append(
                f"| {r.get('target','')} | {r.get('basis_type','')} | {r['case_id']} "
                f"| {float(r['speedup']):.3f} | {float(r.get('max_abs_err',0.0)):.6f} |"
            )
        lines.append("")

        lines.append("## Best 10 speedups\n")
        lines.append("| target | basis | case_id | speedup | max_abs_err |\n|---|---|---|---:|---:|")
        for r in ok_sorted[-10:][::-1]:
            lines.append(
                f"| {r.get('target','')} | {r.get('basis_type','')} | {r['case_id']} "
                f"| {float(r['speedup']):.3f} | {float(r.get('max_abs_err',0.0)):.6f} |"
            )
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(DEFAULT_LOGS_ROOT))
    ap.add_argument("--cases", default=str(DEFAULT_CASES_ROOT))
    ap.add_argument("--reports", default=str(DEFAULT_REPORTS_ROOT))
    args = ap.parse_args()

    logs_root = Path(args.logs)
    cases_root = Path(args.cases)
    reports_root = Path(args.reports)

    rows = parse_logs(logs_root, cases_root)
    write_csv(rows, reports_root / "summary.csv")
    write_md(rows, reports_root / "summary.md")
    print(f"Wrote {reports_root / 'summary.csv'} and {reports_root / 'summary.md'}")
    print(f"  Total rows: {len(rows)}, OK: {sum(1 for r in rows if r.get('status')=='OK')}")


if __name__ == "__main__":
    main()
