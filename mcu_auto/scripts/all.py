#!/usr/bin/env python3
"""End-to-end MCU benchmark pipeline.

Steps:
  1) generate cases (gen_cases.py)
  2) build firmware (build_cases.py)
  3) simulate on Wokwi headless (run_wokwi.py)
  4) collect JSON into reports (collect_results.py)

Examples:
  python mcu_auto/scripts/all.py --suite smoke --targets all
  python mcu_auto/scripts/all.py --suite full --targets esp32,esp32c3 --jobs 4
  python mcu_auto/scripts/all.py --suite smoke --targets esp32 --skip-build  # re-run sims only
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run(cmd, cwd=None, check=True):
    print("\n$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if r.returncode != 0 and check:
        raise SystemExit(f"Command failed (rc={r.returncode}): {' '.join(cmd)}")
    return r.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="smoke",
                    help="Suite name from grid.yaml (e.g. smoke, full, stress, extra_degrees, stm32_light)")
    ap.add_argument("--targets", default="all")
    ap.add_argument("--grid", default=str(REPO_ROOT / "mcu_auto" / "grids" / "grid.yaml"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timeout-ms", type=int, default=120000)
    ap.add_argument("--skip-gen", action="store_true", help="Skip case generation (reuse existing)")
    ap.add_argument("--skip-build", action="store_true", help="Skip PlatformIO build (reuse firmware)")
    ap.add_argument("--skip-existing", action="store_true", help="Skip Wokwi sim for cases with existing results")
    args = ap.parse_args()

    scripts = REPO_ROOT / "mcu_auto" / "scripts"

    if not args.skip_gen:
        run(["python", str(scripts / "gen_cases.py"),
             "--suite", args.suite, "--targets", args.targets, "--grid", args.grid])

    if not args.skip_build:
        run(["python", str(scripts / "build_cases.py"),
             "--targets", args.targets, "--jobs", str(args.jobs)],
            check=False)  # build_cases.py handles failures internally

    wokwi_cmd = ["python", str(scripts / "run_wokwi.py"),
         "--targets", args.targets, "--timeout-ms", str(args.timeout_ms),
         "--jobs", str(max(1, args.jobs))]
    if args.skip_existing:
        wokwi_cmd.append("--skip-existing")
    run(wokwi_cmd)

    run(["python", str(scripts / "collect_results.py")])


if __name__ == "__main__":
    main()
