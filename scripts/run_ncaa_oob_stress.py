"""Run the NCA R1 genuine out-of-domain stress test.

This supplemental sweep keeps the controlled PyKAN case fixed and changes only
boundary/OOB semantics.  Evaluation inputs are sampled from an unclipped
N(0,1), so rows outside the augmented spline domain [-1.75, 1.75] are genuine
out-of-domain cases rather than values clipped exactly onto the upper boundary.

Matrix: 5 seeds x 2 boundary modes x 2 OOB policies = 20 runs.
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
OUT_ROOT = REPO_ROOT / "outputs" / "ncaa_r1_oob_stress"
CFG_ROOT = OUT_ROOT / "resolved_run_configs"


def set_path(d: dict, path: str, value) -> None:
    keys = path.split(".")
    cur = d
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def load_base() -> dict:
    with BASE.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_base_"] = str((REPO_ROOT / "configs" / "spec.yaml").resolve())
    return cfg


def materialize(seed: int, boundary: str, oob: str) -> Path:
    cfg = copy.deepcopy(load_base())
    tag = f"s{seed}_L64_int8_symmetric_{boundary}_{oob}_unclipped"

    set_path(cfg, "experiment.name", f"ncaa_r1_oob_stress_{tag}")
    set_path(cfg, "experiment.group", "ncaa_r1_oob_stress")
    set_path(
        cfg,
        "experiment.description",
        "NCA R1 genuine OOB stress test with unclipped normal inputs",
    )
    set_path(cfg, "runtime.seed", seed)
    set_path(cfg, "float_model.arch.seed", seed)
    set_path(cfg, "evaluation_inputs.seed", seed)

    # Genuine OOB inputs: do not project samples back onto the LUT domain.
    set_path(cfg, "evaluation_inputs.inputs.distribution", "normal")
    set_path(cfg, "evaluation_inputs.inputs.mean", 0.0)
    set_path(cfg, "evaluation_inputs.inputs.std", 1.0)
    set_path(cfg, "evaluation_inputs.inputs.clip_min", None)
    set_path(cfg, "evaluation_inputs.inputs.clip_max", None)

    set_path(cfg, "converter.build_lut.L", 64)
    set_path(cfg, "converter.oob_policy.boundary", boundary)
    set_path(cfg, "converter.oob_policy.mode", oob)
    cfg["converter"]["quant"] = {
        "dtype": "int8",
        "scheme": "symmetric",
        "qmin": -127,
        "qmax": 127,
        "zero_point": 0,
        "per_segment": True,
        "meta_dtype": "float16",
    }

    # OOB behavior is the target here; timing was already characterized by the
    # main 60-run sweep.  Disabling repeated timing keeps this supplement small.
    set_path(cfg, "evaluation.extra.speed.enable", False)
    set_path(cfg, "logging.out_dir", str((OUT_ROOT / "runs" / tag).resolve()))
    set_path(cfg, "logging.versioning", "increment")

    CFG_ROOT.mkdir(parents=True, exist_ok=True)
    path = CFG_ROOT / f"{tag}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path


def cases() -> list[dict]:
    return [
        {"seed": seed, "boundary": boundary, "oob": oob}
        for seed in range(5)
        for boundary in ("closed", "half_open")
        for oob in ("clip_x", "zero_spline")
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    run_cases = cases()
    manifest = []
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for idx, case in enumerate(run_cases, start=1):
        cfg_path = materialize(**case)
        manifest.append({**case, "L": 64, "dtype": "int8", "config": str(cfg_path)})
        print(f"[{idx:02d}/{len(run_cases):02d}] {cfg_path.name}")
        if args.dry_run:
            continue

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["NUMBA_NUM_THREADS"] = "1"
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.check_call(
            [args.python, "-m", "scripts.run_experiment", str(cfg_path)],
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
        print("OOB stress sweep complete. Aggregate with:")
        print(
            "  python scripts/collect_results.py "
            "--root outputs/ncaa_r1_oob_stress/runs "
            "--outdir outputs/ncaa_r1_oob_stress/summary "
            "--latex_dir outputs/ncaa_r1_oob_stress/summary/latex"
        )


if __name__ == "__main__":
    main()
