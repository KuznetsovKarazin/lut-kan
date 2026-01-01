# scripts/smoke_config.py
from __future__ import annotations

import sys
from pathlib import Path

from src.utils.parse_config import load_and_validate


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/spec.yaml")
    cfg, _ = load_and_validate(path)
    print("OK:", cfg.experiment.name)
    print("converter.enabled:", cfg.converter.enabled)
    print("quant:", cfg.converter.quant.dtype, cfg.converter.quant.scheme)


if __name__ == "__main__":
    main()
