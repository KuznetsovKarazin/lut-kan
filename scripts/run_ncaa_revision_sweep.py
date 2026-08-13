"""Run the exact controlled matrix needed for the NCA R1 revision.

The matrix reproduces the manuscript's controlled case with PyKAN width [10,8],
grid=8, k=3, 1024 benchmark inputs, five seeds, linear interpolation, and
backend-matched B-spline/LUT timing.  It uses the corrected endpoint-inclusive
LUT builder and true min/max quantization from LUT values (no external
quantization calibration dataset).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "configs" / "ncaa_revision_base.yaml"
OUT_ROOT = REPO_ROOT / "outputs" / "ncaa_r1_controlled"
CFG_ROOT = OUT_ROOT / "resolved_run_configs"


def set_path(d: dict, path: str, value) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def load_base() -> dict:
    with BASE.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Generated configs live under outputs/, so make inheritance unambiguous.
    cfg["_base_"] = str((REPO_ROOT / "configs" / "spec.yaml").resolve())
    return cfg


def quant_block(dtype: str) -> dict:
    if dtype == "int8":
        return {
            "dtype": "int8",
            "scheme": "symmetric",
            "qmin": -127,
            "qmax": 127,
            "zero_point": 0,
            "per_segment": True,
            "meta_dtype": "float16",
        }
    if dtype == "uint8":
        return {
            "dtype": "uint8",
            "scheme": "asymmetric",
            "qmin": 0,
            "qmax": 255,
            "zero_point": "auto",
            "per_segment": True,
            "meta_dtype": "float16",
        }
    raise ValueError(dtype)


def unique_cases():
    cases = []
    seen = set()

    def add(seed: int, L: int, dtype: str, boundary: str, oob: str, family: str):
        key = (seed, L, dtype, boundary, oob)
        if key in seen:
            return
        seen.add(key)
        cases.append(
            {
                "seed": seed,
                "L": L,
                "dtype": dtype,
                "boundary": boundary,
                "oob": oob,
                "family": family,
            }
        )

    # Main accuracy/speed/memory tables: 4 L x 2 schemes x 5 seeds.
    for seed in range(5):
        for L in (16, 32, 64, 128):
            for dtype in ("int8", "uint8"):
                add(seed, L, dtype, "closed", "clip_x", "main")

    # OOB 2x2 matrix at L=64, symmetric int8.
    for seed in range(5):
        for boundary in ("closed", "half_open"):
            for oob in ("clip_x", "zero_spline"):
                add(seed, 64, "int8", boundary, oob, "oob")

    # Symmetric/asymmetric comparison under half_open + clip_x.
    for seed in range(5):
        for dtype in ("int8", "uint8"):
            add(seed, 64, dtype, "half_open", "clip_x", "scheme_oob")

    return cases


def materialize(case: dict) -> tuple[Path, dict]:
    cfg = copy.deepcopy(load_base())
    seed = int(case["seed"])
    L = int(case["L"])
    dtype = str(case["dtype"])
    boundary = str(case["boundary"])
    oob = str(case["oob"])
    scheme = "symmetric" if dtype == "int8" else "asymmetric"

    tag = f"s{seed}_L{L}_{dtype}_{scheme}_{boundary}_{oob}"
    set_path(cfg, "experiment.name", f"ncaa_r1_{tag}")
    set_path(cfg, "experiment.group", f"ncaa_r1_{case['family']}")
    set_path(cfg, "runtime.seed", seed)
    set_path(cfg, "float_model.arch.seed", seed)
    set_path(cfg, "evaluation_inputs.seed", seed)
    set_path(cfg, "converter.build_lut.L", L)
    set_path(cfg, "converter.oob_policy.boundary", boundary)
    set_path(cfg, "converter.oob_policy.mode", oob)
    cfg["converter"]["quant"] = quant_block(dtype)
    set_path(cfg, "logging.out_dir", str((OUT_ROOT / "runs" / tag).resolve()))
    set_path(cfg, "logging.versioning", "increment")

    CFG_ROOT.mkdir(parents=True, exist_ok=True)
    path = CFG_ROOT / f"{tag}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Generate configs without running experiments")
    ap.add_argument(
        "--family",
        choices=["all", "main", "oob", "scheme_oob"],
        default="all",
        help="Restrict to one logical family; duplicate physical cases are still run only once.",
    )
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    cases = unique_cases()
    if args.family != "all":
        cases = [c for c in cases if c["family"] == args.family]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, case in enumerate(cases, start=1):
        cfg_path, _ = materialize(case)
        manifest.append({**case, "config": str(cfg_path)})
        print(f"[{idx:02d}/{len(cases):02d}] {cfg_path.name}")
        if not args.dry_run:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["NUMBA_NUM_THREADS"] = "1"
            env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            subprocess.check_call(
                [args.python, "scripts/run_experiment.py", str(cfg_path)],
                cwd=REPO_ROOT,
                env=env,
            )

    (OUT_ROOT / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Manifest: {OUT_ROOT / 'sweep_manifest.json'}")
    if args.dry_run:
        print("Dry run only; experiments were not executed.")
    else:
        print("Controlled sweep complete. Aggregate with:")
        print(
            "  python scripts/collect_results.py "
            "--root outputs/ncaa_r1_controlled/runs "
            "--outdir outputs/ncaa_r1_controlled/summary "
            "--latex_dir outputs/ncaa_r1_controlled/summary/latex"
        )


if __name__ == "__main__":
    main()
