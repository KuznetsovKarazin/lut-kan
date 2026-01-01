# scripts/generate_experiment_grid.py
from __future__ import annotations

import csv
from itertools import product
from pathlib import Path
from typing import Any, Dict

import yaml


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    out_dir = Path("configs/generated")
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_L = [16, 32, 64, 128]
    grid_interp = ["nearest", "linear"]
    grid_dtype = ["uint8", "int8"]

    manifest_path = out_dir / "manifest.csv"
    rows = []

    for L, interp, dtype in product(grid_L, grid_interp, grid_dtype):
        # Enforce simple validity here; strict validation will also catch issues.
        if dtype == "uint8":
            scheme = "asymmetric"
            quant_block = {"dtype": "uint8", "scheme": "asymmetric", "qmin": 0, "qmax": 255, "zero_point": "auto"}
        else:
            # Default portable symmetric int8
            scheme = "symmetric"
            quant_block = {"dtype": "int8", "scheme": "symmetric", "qmin": -127, "qmax": 127, "zero_point": 0}

        exp_name = f"L{L}_{dtype}_{scheme}_{interp}_v1"
        cfg = {
            "_base_": "spec.yaml",
            "experiment": {
                "name": exp_name,
                "group": "sensitivity_grid",
                "description": f"L={L}, dtype={dtype}, scheme={scheme}, interp={interp}",
            },
            "converter": {
                "build_lut": {"L": L},
                "interp": {"mode": interp},
                "quant": quant_block,
            },
            "logging": {"out_dir": f"outputs/exp_runs/{exp_name}"},
        }

        dump_yaml(out_dir / f"{exp_name}.yaml", cfg)
        rows.append(
            {
                "config_file": str((out_dir / f"{exp_name}.yaml").as_posix()),
                "name": exp_name,
                "L": L,
                "dtype": dtype,
                "scheme": scheme,
                "interp": interp,
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config_file", "name", "L", "dtype", "scheme", "interp"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
