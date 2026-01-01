# scripts/run_experiment.py
from __future__ import annotations

import sys
from pathlib import Path

from src.experiments.runner import run


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_experiment.py <path/to/config.yaml>")
        raise SystemExit(2)
    out_dir = run(Path(sys.argv[1]))
    print("DONE. Output:", out_dir)


if __name__ == "__main__":
    main()
