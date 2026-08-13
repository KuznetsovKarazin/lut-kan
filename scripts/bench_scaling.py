from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.utils.parse_config import load_and_validate
from src.experiments.runner import run as run_experiment


def _ensure_dict(root: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = root.get(key, None)
    if not isinstance(v, dict):
        root[key] = {}
    return root[key]


def _ensure_path(root: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    cur: Dict[str, Any] = root
    for k in keys:
        cur = _ensure_dict(cur, k)
    return cur


def _deep_get(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_sizes(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("Empty --sizes")
    return out


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_diff(rep: Dict[str, Any], key: str) -> Dict[str, float]:
    """
    Supports either:
      rep[key] = {"mae":..,"rmse":..,"max_abs":..}
    or flat:
      rep[f"{key}_rmse"] / rep[f"{key}_max_abs"]
    """
    blk = rep.get(key, None)
    if isinstance(blk, dict):
        return {
            "rmse": float(blk.get("rmse", float("nan"))),
            "max_abs": float(blk.get("max_abs", float("nan"))),
        }
    # fallback to flat
    return {
        "rmse": float(rep.get(f"{key}_rmse", float("nan"))),
        "max_abs": float(rep.get(f"{key}_max_abs", float("nan"))),
    }


def _extract_speed_ms(rep: Dict[str, Any], key: str) -> float:
    blk = rep.get(key, None)
    if isinstance(blk, dict) and "per_iter_ms" in blk:
        v = blk.get("per_iter_ms", None)
        return float("nan") if v is None else float(v)
    return float("nan")


def _override_for_size(raw_merged: Dict[str, Any], size: int, out_root: Path, warmup: int, iters: int) -> Dict[str, Any]:
    cfg = deepcopy(raw_merged)
    cfg.pop("_base_", None)

    # experiment
    exp = _ensure_path(cfg, ["experiment"])
    base_name = exp.get("name", "bench_scaling")
    exp["name"] = f"{base_name}_scale_{size}x{size}"
    exp.setdefault("group", "dummy_scaling")
    exp.setdefault("description", f"Scaling benchmark size={size}")

    # logging
    logging = _ensure_path(cfg, ["logging"])
    logging["out_dir"] = str((out_root / "exp_runs" / exp["name"]).as_posix())
    logging["versioning"] = "increment"

    # runtime/debug keep as-is, but ensure runtime exists if you use it in runner
    _ensure_path(cfg, ["runtime"])

    # evaluation inputs used for approximation/OOB/latency measurements
    eval_inputs = _ensure_path(cfg, ["evaluation_inputs"])
    eval_inputs["num_samples"] = int(eval_inputs.get("num_samples", 4096))

    # float_model: update arch according to the selected backend
    fm = _ensure_path(cfg, ["float_model"])
    backend = str(fm.get("backend", "dummy"))

    arch = _ensure_path(cfg, ["float_model", "arch"])
    if backend == "pykan":
        # Single-layer KAN benchmark: width=[in_dim,out_dim]
        arch["width"] = [int(size), int(size)]
        arch["in_dim"] = int(size)
        arch["out_dim"] = int(size)
        arch.setdefault("grid", 5)
        arch.setdefault("k", 3)
        arch.setdefault("grid_eps", 1.0)
        arch.setdefault("enforce_shared_grid", True)
        arch.setdefault("device", "cpu")
    else:
        # Default to dummy backend
        fm["backend"] = "dummy"
        arch["in_dim"] = int(size)
        arch["out_dim"] = int(size)
        arch.setdefault("num_knots", 9)
        arch.setdefault("x_min", -3.0)
        arch.setdefault("x_max", 3.0)

        # Keep legacy keys for older configs
        dummy = _ensure_path(cfg, ["float_model", "dummy"])
        dummy["in_dim"] = int(size)
        dummy["out_dim"] = int(size)
        dummy.setdefault("num_knots", 9)
        dummy.setdefault("x_min", -3.0)
        dummy.setdefault("x_max", 3.0)

    # converter
    conv = _ensure_path(cfg, ["converter"])
    conv["enabled"] = True
    _ensure_path(cfg, ["converter", "build_lut"])
    _ensure_path(cfg, ["converter", "quant"])
    _ensure_path(cfg, ["converter", "interp"])
    _ensure_path(cfg, ["converter", "oob_policy"])
    _ensure_path(cfg, ["converter", "y_range"])

    # evaluation
    eval_ = _ensure_path(cfg, ["evaluation", "extra"])
    speed = _ensure_path(cfg, ["evaluation", "extra", "speed"])
    speed["enable"] = True
    speed["warmup_iters"] = int(warmup)
    speed["measure_iters"] = int(iters)

    mem = _ensure_path(cfg, ["evaluation", "extra", "memory"])
    mem.setdefault("enable", True)
    mem.setdefault("breakdown", True)

    phi = _ensure_path(cfg, ["evaluation", "extra", "phi_error"])
    phi.setdefault("enable", True)

    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True)
    ap.add_argument("--sizes", type=str, required=True)
    ap.add_argument("--out", type=str, default="outputs/bench_scaling")
    ap.add_argument("--speed-warmup", type=int, default=200)
    ap.add_argument("--speed-iters", type=int, default=2000)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg_dir = out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: use your loader so _base_ is resolved to a merged dict
    _, raw_merged = load_and_validate(args.base)

    sizes = _parse_sizes(args.sizes)

    rows: List[Dict[str, Any]] = []
    for size in sizes:
        row: Dict[str, Any] = {
            "size": int(size),
            "status": "OK",
            "error": "",
            "run_dir": "",
        }
        try:
            cfg_eff = _override_for_size(raw_merged, size, out_root, args.speed_warmup, args.speed_iters)
            cfg_path = cfg_dir / f"scale_{size}x{size}.yaml"
            _dump_yaml(cfg_path, cfg_eff)

            run_dir = Path(run_experiment(str(cfg_path)))
            row["run_dir"] = str(run_dir.as_posix())

            rep = _read_json(run_dir / "results.json")

            # sanity: show key run params
            rp = rep.get("run_params", {}) or {}
            row["in_dim"] = rp.get("in_dim", "")
            row["out_dim"] = rp.get("out_dim", "")
            row["edges"] = rp.get("edges", "")
            row["L"] = rp.get("L", "")
            row["dtype"] = rp.get("dtype", "")
            row["interp"] = rp.get("interp", "")
            row["oob_mode"] = rp.get("oob_mode", "")

            # kernel-to-kernel diffs
            d_fast = _extract_diff(rep, "lut_fast_vs_lut")
            d_v2 = _extract_diff(rep, "lut_v2_vs_lut")
            d_nb = _extract_diff(rep, "lut_numba_vs_lut")
            row["fast_max_abs"] = d_fast["max_abs"]
            row["fast_rmse"] = d_fast["rmse"]
            row["v2_max_abs"] = d_v2["max_abs"]
            row["v2_rmse"] = d_v2["rmse"]
            row["numba_max_abs"] = d_nb["max_abs"]
            row["numba_rmse"] = d_nb["rmse"]

            # speeds
            row["speed_float_ms"] = _extract_speed_ms(rep, "speed_float")
            row["speed_lut_ms"] = _extract_speed_ms(rep, "speed_lut")
            row["speed_lut_fast_ms"] = _extract_speed_ms(rep, "speed_lut_fast")
            row["speed_lut_v2_ms"] = _extract_speed_ms(rep, "speed_lut_v2")
            row["speed_lut_numba_ms"] = _extract_speed_ms(rep, "speed_lut_numba")

            # numba meta
            numba_blk = rep.get("numba", {}) or {}
            row["numba_available"] = numba_blk.get("available", "")
            row["numba_version"] = numba_blk.get("version", "")
            row["numba_threads"] = numba_blk.get("num_threads", "")
            row["numba_compile_ms"] = numba_blk.get("compile_ms", "")
            row["numba_error"] = numba_blk.get("error", "")

            # memory
            mem = rep.get("memory", {}) or {}
            row["mem_total_bytes"] = mem.get("lut_total_bytes", mem.get("total_bytes", ""))
            br = mem.get("breakdown", {}) or {}
            row["mem_q_table_bytes"] = br.get("q_table_bytes", "")
            row["mem_knots_bytes"] = br.get("knots_bytes", "")
            row["mem_scale_bytes"] = br.get("scale_bytes", "")
            row["mem_y_min_bytes"] = br.get("y_min_bytes", "")

        except Exception as e:
            row["status"] = "FAIL"
            row["error"] = f"{type(e).__name__}: {e}"

        rows.append(row)

    # write csv
    # stable header (union of keys)
    header: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in header:
                header.append(k)

    csv_path = out_root / "bench_scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"DONE. CSV: {csv_path.as_posix()}")


if __name__ == "__main__":
    main()
