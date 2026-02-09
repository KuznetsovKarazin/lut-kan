#!/usr/bin/env python3
"""Run headless Wokwi simulations for generated cases.

For each case directory with wokwi.toml, this script runs:
  wokwi-cli --timeout <ms> --serial-log-file <log> --expect-text "LUTKAN:"

and captures the serial log into mcu_auto/logs/<target>/<case_id>.log

Prerequisites:
  - Wokwi CLI installed (v0.22+)
  - environment variable WOKWI_CLI_TOKEN set (required for CI/headless). See docs.

Notes:
  - The firmware prints exactly one JSON line prefixed with 'LUTKAN:'
  - --expect-text makes wokwi-cli exit as soon as the result line appears
  - --serial-log-file captures serial output (stdout only shows CLI messages)
"""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_ROOT = REPO_ROOT / "mcu_auto" / "cases"
DEFAULT_LOGS_ROOT = REPO_ROOT / "mcu_auto" / "logs"


def _iter_case_dirs(cases_root: Path, targets: List[str]) -> List[Path]:
    dirs: List[Path] = []
    for t in targets:
        tdir = cases_root / t
        if not tdir.exists():
            continue
        for d in sorted(tdir.iterdir()):
            if d.is_dir() and (d / "wokwi.toml").exists():
                dirs.append(d)
    return dirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(DEFAULT_CASES_ROOT))
    ap.add_argument("--logs", default=str(DEFAULT_LOGS_ROOT))
    ap.add_argument("--targets", default="all")
    ap.add_argument("--timeout-ms", type=int, default=120000)
    ap.add_argument("--jobs", type=int, default=4, help="Parallel simulations")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip cases where log already contains LUTKAN: line")
    ap.add_argument("--wokwi-bin", default=os.environ.get("WOKWI_CLI_BIN", "wokwi-cli"))
    args = ap.parse_args()

    cases_root = Path(args.cases)
    logs_root = Path(args.logs)
    logs_root.mkdir(parents=True, exist_ok=True)

    if args.targets == "all":
        targets = [p.name for p in sorted(cases_root.iterdir()) if p.is_dir()]
    else:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    case_dirs = _iter_case_dirs(cases_root, targets)
    if not case_dirs:
        raise SystemExit(f"No cases found under {cases_root}. Run gen_cases.py first.")

    if not os.environ.get("WOKWI_CLI_TOKEN"):
        print("[WARN] WOKWI_CLI_TOKEN is not set. Headless runs may fail. See Wokwi CI docs.")

    def _run_one(case_dir: Path) -> tuple[str, str, int, Path]:
        target = case_dir.parent.name
        case_id = case_dir.name
        out_dir = logs_root / target
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"{case_id}.log"

        # Skip cases that failed to build
        if (case_dir / "BUILD_FAILED").exists() or (case_dir / "SKIPPED").exists():
            log_path.write_text("SKIPPED: build failed or case skipped\n", encoding="utf-8")
            return target, case_id, -1, log_path

        # Skip cases that already have results
        if args.skip_existing and log_path.exists():
            try:
                content = log_path.read_text(encoding="utf-8", errors="replace")
                if "LUTKAN:" in content:
                    return target, case_id, -2, log_path  # -2 = cached
            except OSError:
                pass

        cmd = [
            args.wokwi_bin,
            "--timeout", str(args.timeout_ms),
            "--serial-log-file", str(log_path),
            "--expect-text", "LUTKAN:",
            "--timeout-exit-code", "42",
        ]

        rc = 1
        for attempt in range(1, args.retries + 2):
            p = subprocess.run(cmd, cwd=str(case_dir),
                               capture_output=True, text=True)
            rc = p.returncode
            if rc == 0:
                break
            # rc=42 means timeout — might need more time, retry
            if rc != 42:
                break

        # If timed out, write note to log
        if rc == 42:
            with log_path.open("a", encoding="utf-8") as f:
                f.write("\n# TIMEOUT: simulation did not produce LUTKAN: line\n")

        return target, case_id, rc, log_path

    ok, fail, skip, timeout, cached = 0, 0, 0, 0, 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = [ex.submit(_run_one, d) for d in case_dirs]
        done = 0
        for fut in as_completed(futs):
            target, case_id, rc, log_path = fut.result()
            done += 1
            if rc == -2:
                cached += 1
                print(f"[{done}/{len(case_dirs)}] [CACHED] {target}/{case_id}")
            elif rc == -1:
                skip += 1
                print(f"[{done}/{len(case_dirs)}] [SKIP] {target}/{case_id}")
            elif rc == 42:
                timeout += 1
                print(f"[{done}/{len(case_dirs)}] [TIMEOUT] {target}/{case_id} -> {log_path}")
            elif rc != 0:
                fail += 1
                print(f"[{done}/{len(case_dirs)}] [FAIL] {target}/{case_id} (rc={rc}) -> {log_path}")
            else:
                ok += 1
                print(f"[{done}/{len(case_dirs)}] [OK] {target}/{case_id} -> {log_path}")

    print(f"\nWokwi summary: {ok} OK, {fail} FAIL, {timeout} TIMEOUT, {skip} SKIP, {cached} CACHED (total {len(case_dirs)})")


if __name__ == "__main__":
    main()
