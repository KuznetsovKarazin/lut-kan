#!/usr/bin/env python3
"""Run benchmarks on real hardware via PlatformIO upload + serial capture.

This is the "serious" complement to Wokwi:
  - builds a case (if needed)
  - uploads via PlatformIO
  - listens on a serial port until it sees a 'LUTKAN:{...}' line
  - stores logs into mcu_auto/hw_logs/<target>/<case_id>.log

Usage examples:
  python mcu_auto/scripts/run_hardware.py --targets esp32 --suite smoke --port /dev/ttyUSB0
  python mcu_auto/scripts/run_hardware.py --case mcu_auto/cases/esp32/<case_id> --port COM5

Notes:
  - You must have PlatformIO installed.
  - You must install pyserial:  pip install pyserial
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_ROOT = REPO_ROOT / "mcu_auto" / "cases"
LOGS_ROOT = REPO_ROOT / "mcu_auto" / "hw_logs"


def _iter_case_dirs(cases_root: Path, targets: list[str]) -> list[Path]:
    out: list[Path] = []
    for t in targets:
        tdir = cases_root / t
        if not tdir.exists():
            continue
        for d in sorted(tdir.iterdir()):
            if d.is_dir() and (d / "platformio.ini").exists():
                out.append(d)
    return out


def _capture_serial(port: str, baud: int, timeout_s: int) -> str:
    # Import lazily so the repository remains usable without pyserial.
    import serial  # type: ignore

    t_end = time.time() + timeout_s
    lines: list[str] = []
    with serial.Serial(port, baudrate=baud, timeout=0.2) as ser:
        # Give the board time to reboot after upload
        time.sleep(0.8)
        while time.time() < t_end:
            b = ser.readline()
            if not b:
                continue
            try:
                s = b.decode("utf-8", errors="replace").rstrip("\r\n")
            except Exception:
                s = repr(b)
            lines.append(s)
            if "LUTKAN:{" in s:
                break
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="all", help="Comma list or 'all'")
    ap.add_argument("--cases", default=str(CASES_ROOT))
    ap.add_argument("--case", default="", help="Run a single case directory")
    ap.add_argument("--port", required=True, help="Serial port (e.g., /dev/ttyUSB0, COM5)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout-s", type=int, default=30)
    ap.add_argument("--upload", action="store_true", help="Upload firmware before capturing")
    args = ap.parse_args()

    cases_root = Path(args.cases)
    if args.case:
        case_dirs = [Path(args.case)]
    else:
        if args.targets == "all":
            targets = [p.name for p in sorted(cases_root.iterdir()) if p.is_dir()]
        else:
            targets = [t.strip() for t in args.targets.split(",") if t.strip()]
        case_dirs = _iter_case_dirs(cases_root, targets)

    if not case_dirs:
        raise SystemExit("No cases found. Run gen_cases.py first.")

    LOGS_ROOT.mkdir(parents=True, exist_ok=True)

    for i, case_dir in enumerate(case_dirs, 1):
        target = case_dir.parent.name
        case_id = case_dir.name
        out_dir = LOGS_ROOT / target
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"{case_id}.log"

        print(f"[{i}/{len(case_dirs)}] {target}/{case_id}")

        if args.upload:
            subprocess.run(["pio", "run", "-e", "mcu", "-t", "upload", "--upload-port", args.port], cwd=str(case_dir), check=True)

        txt = _capture_serial(args.port, args.baud, args.timeout_s)
        log_path.write_text(txt, encoding="utf-8")
        print(f"  -> {log_path}")


if __name__ == "__main__":
    main()
