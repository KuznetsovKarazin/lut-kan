from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]  # E:\lut-kan
CONFIGS_DIR = REPO_ROOT / "configs"
BASE_CFG = CONFIGS_DIR / "exp_pykan_lut.yaml"
PY = os.environ.get("PYTHON", "python")


def _env_with_pythonpath() -> dict:
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    parts = [p for p in pp.split(os.pathsep) if p]
    if str(REPO_ROOT) not in parts:
        parts.insert(0, str(REPO_ROOT))
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _absolutize_base_paths(cfg: dict) -> dict:
    """
    Your loader resolves 'base:' relative to the config file location.
    When we emit tmp YAML elsewhere, relative base breaks.

    Fix: if cfg['base'] is relative, convert it to an absolute path relative to configs/.
    """
    if isinstance(cfg, dict) and "base" in cfg and isinstance(cfg["base"], str):
        base_path = Path(cfg["base"])
        if not base_path.is_absolute():
            # interpret relative 'base' exactly as when exp_pykan_lut.yaml is in configs/
            abs_base = (CONFIGS_DIR / base_path).resolve()
            cfg["base"] = str(abs_base)
    return cfg


def run_one(cfg: dict, tag: str) -> None:
    tmp_dir = REPO_ROOT / "outputs" / "sweeps" / "tmp_cfgs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cfg = _absolutize_base_paths(cfg)

    tmp_path = tmp_dir / f"tmp_{tag}.yaml"
    with tmp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    subprocess.check_call(
        [PY, "scripts/run_experiment.py", str(tmp_path)],
        cwd=str(REPO_ROOT),
        env=_env_with_pythonpath(),
    )


def load_base() -> dict:
    with BASE_CFG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_path(d: dict, path: str, value) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or cur[k] is None:
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def main() -> None:
    base = load_base()
    base = _absolutize_base_paths(base)

    # 1) L sweep
    for L in (16, 32, 64, 128):
        cfg = copy.deepcopy(base)
        set_path(cfg, "converter.build_lut.L", int(L))
        set_path(cfg, "experiment.name", f"pykan_lut_L{L}")
        run_one(cfg, f"L{L}")

    # 2) Quant scheme for L=32 and 64
    for L in (32, 64):
        for dtype, scheme, tag in [
            ("int8", "symmetric", "int8sym"),
            ("uint8", "asymmetric", "uint8asym"),
        ]:
            cfg = copy.deepcopy(base)
            set_path(cfg, "converter.build_lut.L", int(L))
            set_path(cfg, "converter.quant.dtype", dtype)
            set_path(cfg, "converter.quant.scheme", scheme)
            set_path(cfg, "experiment.name", f"pykan_lut_L{L}_{tag}")
            run_one(cfg, f"L{L}_{tag}")

    # 3) OOB policy for L=64 int8 symmetric
    for mode in ("clip_x", "zero_spline"):
        cfg = copy.deepcopy(base)
        set_path(cfg, "converter.build_lut.L", 64)
        set_path(cfg, "converter.quant.dtype", "int8")
        set_path(cfg, "converter.quant.scheme", "symmetric")
        set_path(cfg, "converter.oob_policy.mode", mode)
        set_path(cfg, "experiment.name", f"pykan_lut_L64_{mode}")
        run_one(cfg, f"L64_{mode}")


if __name__ == "__main__":
    main()
