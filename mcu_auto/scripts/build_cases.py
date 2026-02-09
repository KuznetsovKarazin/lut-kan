#!/usr/bin/env python3
"""Build generated MCU benchmark cases using PlatformIO.

This script walks mcu_auto/cases/<target>/<case_id> and runs:
  pio run -e mcu

It is safe to run multiple times; it will rebuild when required.
Failures are logged but do NOT stop the entire batch.

Prerequisites:
  - PlatformIO Core installed (pio command available)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_ROOT = REPO_ROOT / "mcu_auto" / "cases"


def _iter_case_dirs(cases_root: Path, targets: List[str]) -> List[Path]:
    dirs: List[Path] = []
    for t in targets:
        tdir = cases_root / t
        if not tdir.exists():
            continue
        for d in sorted(tdir.iterdir()):
            if d.is_dir() and (d / "platformio.ini").exists():
                dirs.append(d)
    return dirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(DEFAULT_CASES_ROOT))
    ap.add_argument("--targets", default="all", help="Comma list or 'all'")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel compile jobs per PlatformIO")
    args = ap.parse_args()

    cases_root = Path(args.cases)
    if args.targets == "all":
        targets = [p.name for p in sorted(cases_root.iterdir()) if p.is_dir()]
    else:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    case_dirs = _iter_case_dirs(cases_root, targets)
    if not case_dirs:
        raise SystemExit(f"No cases found under {cases_root}. Run gen_cases.py first.")

    def _parse_size(stdout: str) -> dict:
        out = {}
        m = re.search(r"(?:RAM|Data):\s*(\d+)\s*bytes", stdout)
        if m:
            out["ram_bytes"] = int(m.group(1))
        m = re.search(r"(?:Flash|Program):\s*(\d+)\s*bytes", stdout)
        if m:
            out["flash_bytes"] = int(m.group(1))
        m = re.search(r"Program:\s*(\d+)\s*bytes", stdout)
        if m:
            out["flash_bytes"] = int(m.group(1))
        m = re.search(r"Data:\s*(\d+)\s*bytes", stdout)
        if m:
            out["ram_bytes"] = int(m.group(1))
        return out

    pio_version = subprocess.run(["pio", "--version"], capture_output=True, text=True).stdout.strip()

    ok, fail, skip = 0, 0, 0
    failed_list: List[str] = []

    for i, case_dir in enumerate(case_dirs, 1):
        print(f"[{i}/{len(case_dirs)}] Building {case_dir}")

        # Skip cases already marked as skipped (RAM overflow) or previously failed
        skip_marker = case_dir / "SKIPPED"
        if skip_marker.exists():
            print(f"  -> SKIPPED (pre-marked: {skip_marker.read_text(encoding='utf-8').strip()[:80]})")
            skip += 1
            continue

        cmd = ["pio", "run", "-e", "mcu", "-j", str(args.jobs)]
        result = subprocess.run(cmd, cwd=str(case_dir), capture_output=True, text=True)

        if result.returncode != 0:
            combined = result.stdout + "\n" + result.stderr
            if "size of array" in combined or "too large" in combined or "overflowed" in combined:
                reason = "RAM/Flash overflow"
            else:
                reason = "build error"

            print(f"  -> FAILED ({reason})")
            fail += 1
            failed_list.append(f"{case_dir.parent.name}/{case_dir.name}: {reason}")

            # Write failure marker so run_wokwi.py can skip it
            (case_dir / "BUILD_FAILED").write_text(
                f"{reason}\n{combined[-500:]}", encoding="utf-8"
            )
            continue

        ok += 1

        # Record size metrics
        size = subprocess.run(
            ["pio", "run", "-e", "mcu", "-t", "size"],
            cwd=str(case_dir),
            capture_output=True,
            text=True,
        )
        metrics = {
            "pio_version": pio_version,
            **_parse_size(size.stdout + "\n" + size.stderr),
        }
        (case_dir / "build_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nBuild summary: {ok} OK, {fail} FAILED, {skip} SKIPPED (total {len(case_dirs)})")
    if failed_list:
        print("Failed cases:")
        for f in failed_list:
            print(f"  {f}")


if __name__ == "__main__":
    main()
