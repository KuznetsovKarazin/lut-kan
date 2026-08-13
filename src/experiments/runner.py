# src/experiments/runner.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Dict
import inspect
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.kernels.lut_backend_reference import forward_reference
from src.kernels.lut_backend_dense_numpy import forward_dense_numpy
from src.kernels.lut_backend_dense_numba import (
    forward_dense_numba,
    numba_available,
    warmup_numba,
)
from src.kernels.lut_contract import pack_dense_layer

from src.metrics.memory import lut_memory_report
from src.metrics.model_memory import torch_model_memory_bytes
from src.metrics.perf import measure_latency
from src.metrics.phi_error import evaluate_phi_error_on_grid

from src.models.dummy_adapter import DummyKANAdapter
from src.models.jacobi_adapter import JacobiKANSingleLayerAdapter
from src.quant.lut_builder import LUTArtifact
from src.quant.lut_io import save_lut_npz

from src.utils.git_info import get_git_hash
from src.utils.io import dump_json, dump_yaml, ensure_dir, get_env_snapshot
from src.utils.parse_config import load_and_validate
from src.utils.pip_info import pip_freeze
from src.utils.pykan_info import get_pykan_info


# --- B-spline baseline (PyKAN only) ---
# Fair float baselines (NumPy/Numba) for B-spline evaluation
_BSPLINE_IMPORT_ERROR: str | None = None
try:
    from src.kernels.bspline_contract import pack_bspline_dense_layer_from_pykankan_adapter
    from src.kernels.bspline_backend_dense_numpy import forward_bspline_dense_numpy
    from src.kernels.bspline_backend_dense_numba import forward_bspline_dense_numba
    try:
        from src.kernels.bspline_backend_dense_numba import warmup_bspline_numba
    except Exception:
        warmup_bspline_numba = None

except Exception as e:
    _BSPLINE_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    pack_bspline_dense_layer_from_pykankan_adapter = None
    forward_bspline_dense_numpy = None
    forward_bspline_dense_numba = None
    warmup_bspline_numba = None


def _evaluation_inputs(cfg, raw: dict | None) -> SimpleNamespace:
    """
    Return evaluation input-generation settings as an object with attributes.

    These inputs are used for approximation/OOB/latency evaluation; they are
    not used to determine LUT quantization ranges.

    Priority:
      1) validated cfg.evaluation_inputs.inputs
      2) raw['evaluation_inputs']['inputs']
      3) legacy raw['calibration']['inputs']
      4) empty object (defaults via getattr)
    """
    inp = getattr(getattr(cfg, "evaluation_inputs", None), "inputs", None)
    if inp is not None:
        return inp

    if isinstance(raw, dict):
        for key in ("evaluation_inputs", "calibration"):
            d = (raw.get(key) or {}).get("inputs", None)
            if isinstance(d, dict):
                return SimpleNamespace(**d)

    return SimpleNamespace()


def _resolve_out_dir(base_out_dir: str, versioning: str, repo_root: Path) -> Path:
    base = Path(base_out_dir)
    if versioning == "timestamp":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return base / stamp
    if versioning == "git_hash":
        h = get_git_hash(repo_root) or "nogit"
        return base / h
    if versioning == "increment":
        base.mkdir(parents=True, exist_ok=True)
        i = 0
        while (base / f"run_{i:03d}").exists():
            i += 1
        return base / f"run_{i:03d}"
    return base


def _call_build_lut_for_edges_compat(edges, **kwargs):
    """
    Calls build_lut_for_edges() passing only supported keyword args.
    Keeps runner stable across builder refactors.
    """
    from src.quant.lut_builder import build_lut_for_edges

    sig = inspect.signature(build_lut_for_edges)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return build_lut_for_edges(edges=edges, **filtered)


def _resolve_quant(cfg) -> Dict[str, Any]:
    """
    Normalize quant config into a minimal dict.
    Compatible with existing schema (may include zero_point).
    """
    q = cfg.converter.quant
    dtype = str(q.dtype).lower().strip()
    scheme = str(q.scheme).lower().strip()

    if dtype == "uint8":
        scheme = "asymmetric"
        qmin, qmax = 0, 255
        zero_point = None
    else:
        # int8
        qmin, qmax = -127, 127
        if scheme == "symmetric":
            zero_point = 0
        else:
            zero_point = None  # auto

    # Optional overrides
    if isinstance(getattr(q, "qmin", None), int):
        qmin = int(q.qmin)
    if isinstance(getattr(q, "qmax", None), int):
        qmax = int(q.qmax)
    if isinstance(getattr(q, "zero_point", None), int):
        zero_point = int(q.zero_point)

    return {
        "dtype": dtype,
        "scheme": scheme,
        "qmin": int(qmin),
        "qmax": int(qmax),
        "zero_point": zero_point,
        "meta_dtype": str(getattr(q, "meta_dtype", "float16")).lower().strip(),
    }


def _diff_report(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    d = a - b
    return {
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "max_abs": float(np.max(np.abs(d))),
    }


def _worst_case(y_hat: np.ndarray, y_ref: np.ndarray, x: np.ndarray) -> Dict[str, Any]:
    y_hat = np.asarray(y_hat, dtype=np.float32)
    y_ref = np.asarray(y_ref, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)

    d = np.abs(y_hat - y_ref)
    idx = int(np.argmax(d))
    n, j = np.unravel_index(idx, d.shape)
    return {
        "n": int(n),
        "j": int(j),
        "abs_err": float(d[n, j]),
        "y_hat": float(y_hat[n, j]),
        "y_ref": float(y_ref[n, j]),
        "x": x[n].astype(float).tolist(),
    }


def _oob_stats(x: np.ndarray, x_min: float, x_max: float, boundary_mode: str) -> Dict[str, float]:
    """
    OOB statistics for x with shape [N, in_dim].

    boundary_mode:
      - "half_open": in-range is [x_min, x_max)
      - "closed": in-range is [x_min, x_max]
    """
    x = np.asarray(x, dtype=np.float32)
    boundary_mode = boundary_mode if boundary_mode in ("half_open", "closed") else "half_open"

    if boundary_mode == "half_open":
        in_mask = (x >= x_min) & (x < x_max)
    else:
        in_mask = (x >= x_min) & (x <= x_max)

    all_in = np.all(in_mask, axis=1)
    any_oob = ~all_in
    per_coord_oob = 1.0 - np.mean(in_mask, axis=0)

    return {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "OOB_any_frac": float(np.mean(any_oob)),
        "OOB_all_in_frac": float(np.mean(all_in)),
        "OOB_per_coord_mean": float(np.mean(per_coord_oob)),
        "OOB_per_coord_max": float(np.max(per_coord_oob)),
    }


def _normalize_converter_semantics(cfg) -> Dict[str, Any]:
    """
    Convert config fields into builder/runtime semantics.

    Notes:
      - Config may use legacy oob_policy.mode: clip_x | saturate_y | zero_spline
      - Some configs may set converter.oob_behavior directly; accept aliases:
          clip, clip_x, zero, zero_spline, zero_phi
      - Builder/runtime understand: oob_behavior in {"clip","zero"}
    """
    conv = cfg.converter

    # value representation
    vr_cfg = str(getattr(conv, "value_representation", "") or "").lower().strip()
    vk_cfg = str(getattr(conv, "value_kind", "") or "").lower().strip()

    if vr_cfg in ("phi", "spline_component"):
        value_representation = vr_cfg
    elif vk_cfg == "spline":
        value_representation = "spline_component"
    else:
        value_representation = "phi"

    # OOB raw (new field or legacy policy)
    oob_cfg = str(getattr(conv, "oob_behavior", "") or "").lower().strip()
    boundary_mode = "half_open"
    oob_policy_mode = ""

    if hasattr(conv, "oob_policy") and conv.oob_policy is not None:
        oob_policy_mode = str(getattr(conv.oob_policy, "mode", "") or "").lower().strip()
        boundary_mode = str(getattr(conv.oob_policy, "boundary", boundary_mode) or boundary_mode).lower().strip()

    if not oob_cfg:
        oob_cfg = oob_policy_mode

    # Map to effective runtime behavior
    if oob_cfg in ("clip", "clip_x", "saturate_y"):
        oob_behavior = "clip"
    elif oob_cfg in ("zero", "zero_spline", "zero_phi"):
        oob_behavior = "zero"
    else:
        oob_behavior = "clip"

    if boundary_mode not in ("half_open", "closed"):
        boundary_mode = "half_open"

    return {
        "value_representation": value_representation,
        "boundary_mode": boundary_mode,
        "oob_mode_cfg": oob_cfg,
        "oob_policy_mode": oob_policy_mode,
        "oob_behavior": oob_behavior,
    }


def _speed_cfg(cfg) -> SimpleNamespace:
    extra = getattr(getattr(cfg, "evaluation", None), "extra", None)
    sp = getattr(extra, "speed", None) if extra is not None else None
    if sp is None:
        return SimpleNamespace(enable=False, warmup_iters=5, measure_iters=20)
    return sp


def _print_results_summary(results: Dict[str, Any]) -> None:
    """
    Console-friendly summary (does not affect artifacts).
    Prints only if meaningful blocks are present.
    """
    def _p(title: str, obj: Any) -> None:
        print(f"\n[RESULTS] {title}")
        print(json.dumps(obj, indent=2, ensure_ascii=False))

    # Always useful
    if "experiment" in results:
        _p("experiment", results["experiment"])
    if "float_backend" in results:
        _p("float_backend", results["float_backend"])
    if "converter_enabled" in results:
        _p("converter_enabled", results["converter_enabled"])

    # Single-layer pipeline blocks (present for dummy/pykan/jacobi when converter is enabled)
    for k in ("run_params", "input_sanity", "output_sanity", "phi_error_summary"):
        if k in results:
            _p(k, results[k])

    # Memory block
    if "memory" in results and results["memory"]:
        _p("memory", results["memory"])

    # Speed block
    speed_keys = [k for k in ("speed_float", "speed_ref", "speed_dense_numpy", "speed_dense_numba",
                              "speed_bspline_numpy", "speed_bspline_numba") if k in results]
    if speed_keys:
        _p("speed", {k: results.get(k) for k in speed_keys})

    # Optional convenience diffs
    for k in ("dense_numpy_vs_ref", "numba_vs_ref", "bspline_numpy_vs_float", "bspline_numba_vs_float"):
        if k in results:
            _p(k, results[k])



def _run_single_layer_pipeline(adapter, edges, x, cfg, out_dir: Path, results: Dict[str, Any], label: str) -> None:
    results.setdefault("run_params", {})
    results.setdefault("debug", {})
    results.setdefault("numba", {})
    results.setdefault("memory", {})

    sp = _speed_cfg(cfg)

    # -------------------------
    # Float baseline
    # -------------------------
    y_float = adapter.forward_float(x)

    # -------------------------
    # B-spline baseline (NumPy/Numba) for PyKAN only (fair baseline)
    # -------------------------
    sem = _normalize_converter_semantics(cfg)

    if getattr(cfg.float_model, "backend", "") == "pykan":
        # Report availability explicitly (helps debugging)
        results.setdefault("bspline_baseline", {})
        results["bspline_baseline"]["available"] = bool(
            pack_bspline_dense_layer_from_pykankan_adapter is not None
            and forward_bspline_dense_numpy is not None
        )
        results["bspline_baseline"]["import_error"] = _BSPLINE_IMPORT_ERROR

        if (
            pack_bspline_dense_layer_from_pykankan_adapter is not None
            and forward_bspline_dense_numpy is not None
        ):
            try:
                packed_bs = pack_bspline_dense_layer_from_pykankan_adapter(
                    adapter,
                    boundary_mode=str(sem["boundary_mode"]),
                )

                # NumPy
                y_bs_np = forward_bspline_dense_numpy(x, packed_bs)
                results["bspline_numpy_vs_float"] = _diff_report(y_bs_np, y_float)
                results["debug"]["worst_bspline_numpy_vs_float"] = _worst_case(
                    y_hat=y_bs_np, y_ref=y_float, x=x
                )

                if bool(getattr(sp, "enable", False)):
                    results["speed_bspline_numpy"] = measure_latency(
                        lambda: forward_bspline_dense_numpy(x, packed_bs),
                        int(getattr(sp, "warmup_iters", 5)),
                        int(getattr(sp, "measure_iters", 20)),
                    )

                # Numba (optional)
                if (
                    numba_available()
                    and forward_bspline_dense_numba is not None
                    and warmup_bspline_numba is not None
                ):
                    if warmup_bspline_numba is not None:
                        warmup_bspline_numba(packed_bs)
                    results["speed_bspline_numba"] = measure_latency(
                        lambda: forward_bspline_dense_numba(x, packed_bs),
                        int(getattr(sp, "warmup_iters", 5)),
                        int(getattr(sp, "measure_iters", 20)),
                    )

                    y_bs_nb = forward_bspline_dense_numba(x, packed_bs)

                    results["bspline_numba_vs_float"] = _diff_report(y_bs_nb, y_float)

                    if bool(getattr(sp, "enable", False)):
                        results["speed_bspline_numba"] = measure_latency(
                            lambda: forward_bspline_dense_numba(x, packed_bs),
                            int(getattr(sp, "warmup_iters", 5)),
                            int(getattr(sp, "measure_iters", 20)),
                        )

            except Exception as e:
                results["bspline_baseline"]["error"] = f"{type(e).__name__}: {e}"
        else:
            # If kernels are missing, do NOT attempt to call them (avoid NoneType callable)
            results["bspline_baseline"]["error"] = (
                "B-spline baseline kernels are not available (import failed)."
            )

    # -------------------------
    # Original float model memory (params + buffers)
    # -------------------------
    try:
        model_obj = getattr(adapter, "model", None)
        if model_obj is not None:
            results["memory"]["model"] = torch_model_memory_bytes(model_obj)
        else:
            results["memory"]["model"] = {"error": "adapter has no .model attribute"}
    except Exception as e:
        results["memory"]["model"] = {"error": f"{type(e).__name__}: {e}"}

    # -------------------------
    # Float-only run
    # -------------------------
    if not bool(cfg.converter.enabled):
        if bool(getattr(sp, "enable", False)):
            results["speed_float"] = measure_latency(
                lambda: adapter.forward_float(x),
                int(getattr(sp, "warmup_iters", 5)),
                int(getattr(sp, "measure_iters", 20)),
            )
        dump_json(out_dir / "results.json", results)
        _print_results_summary(results)
        return

    # -------------------------
    # LUT build/pack/run
    # -------------------------
    qspec = _resolve_quant(cfg)

    t0 = perf_counter()
    art: LUTArtifact = _call_build_lut_for_edges_compat(
        edges,
        L=int(cfg.converter.build_lut.L),
        interp=str(cfg.converter.interp.mode),
        y_range_method=str(cfg.converter.y_range.method),
        lower_pct=float(getattr(cfg.converter.y_range, "lower_percentile", 0.1)),
        upper_pct=float(getattr(cfg.converter.y_range, "upper_percentile", 99.9)),
        dtype=qspec["dtype"],
        scheme=qspec["scheme"],
        qmin=qspec["qmin"],
        qmax=qspec["qmax"],
        zero_point=qspec["zero_point"],
        meta_dtype=qspec["meta_dtype"],
        value_representation=str(sem["value_representation"]),
        oob_behavior=str(sem["oob_behavior"]),
        boundary_mode=str(sem["boundary_mode"]),
    )
    build_ms = float((perf_counter() - t0) * 1000.0)

    save_lut_npz(out_dir / "lut_artifact.npz", art)

    packed = pack_dense_layer(
        art,
        edges=edges,
        in_dim=int(adapter.in_dim),
        out_dim=int(adapter.out_dim),
        boundary_mode=str(sem["boundary_mode"]),
    )

    # Run LUT backends
    t0 = perf_counter()
    y_ref = forward_reference(x, packed)
    ref_ms = float((perf_counter() - t0) * 1000.0)

    t0 = perf_counter()
    y_dense = forward_dense_numpy(x, packed)
    dense_ms = float((perf_counter() - t0) * 1000.0)

    # Sanity comparisons
    results["output_sanity"] = _diff_report(y_ref, y_float)
    results["dense_numpy_vs_ref"] = _diff_report(y_dense, y_ref)
    results["output_sanity_dense_numpy"] = _diff_report(y_dense, y_float)

    results["debug"]["worst_output_sanity"] = _worst_case(y_hat=y_ref, y_ref=y_float, x=x)
    results["debug"]["worst_dense_numpy_vs_ref"] = _worst_case(y_hat=y_dense, y_ref=y_ref, x=x)

    # OOB / in-range split analysis at output level (based on LUT domain)
    x_min = float(packed.x_min)
    x_max = float(packed.x_max)
    results["input_sanity"] = _oob_stats(x, x_min=x_min, x_max=x_max, boundary_mode=sem["boundary_mode"])

    if sem["boundary_mode"] == "half_open":
        in_mask = (x >= x_min) & (x < x_max)
    else:
        in_mask = (x >= x_min) & (x <= x_max)

    in_rows = np.all(in_mask, axis=1)
    oob_rows = ~in_rows

    if np.any(in_rows):
        results["output_sanity_in_range"] = _diff_report(y_ref[in_rows], y_float[in_rows])
    else:
        results["output_sanity_in_range"] = {"mae": 0.0, "rmse": 0.0, "max_abs": 0.0}

    if np.any(oob_rows):
        results["output_sanity_oob_only"] = _diff_report(y_ref[oob_rows], y_float[oob_rows])
    else:
        results["output_sanity_oob_only"] = {"mae": 0.0, "rmse": 0.0, "max_abs": 0.0}

    # Optional phi error (edge-level)
    extra = getattr(getattr(cfg, "evaluation", None), "extra", None)
    phi_cfg = getattr(extra, "phi_error", None) if extra is not None else None
    if bool(getattr(phi_cfg, "enable", False)):
        num_points = int(getattr(phi_cfg, "num_points", 256))
        topk = int(getattr(phi_cfg, "report_topk_functions", 10))

        phi_rep = evaluate_phi_error_on_grid(
            edges=edges,
            art=art,
            num_points=num_points,
            topk=topk,
        )
        results["phi_error_summary"] = phi_rep.summary
        results["phi_error_topk"] = phi_rep.topk_by_max_abs

    # Optional LUT memory
    mem_cfg = getattr(extra, "memory", None) if extra is not None else None
    if bool(getattr(mem_cfg, "enable", False)):
        results["memory"]["lut"] = lut_memory_report(
            art,
            breakdown=bool(getattr(mem_cfg, "breakdown", False)),
        )

        # Optional ratios
        try:
            model_total = results["memory"].get("model", {}).get("total_bytes", None)
            lut_rep = results["memory"].get("lut", {}) or {}
            lut_total = (
                lut_rep.get("total_bytes", None)
                or lut_rep.get("bytes_total", None)
                or lut_rep.get("lut_total_bytes", None)
            )
            if isinstance(model_total, int) and model_total > 0 and isinstance(lut_total, (int, float)) and float(lut_total) > 0:
                results["memory"]["ratios"] = {
                    "lut_over_model": float(lut_total) / float(model_total),
                    "model_over_lut": float(model_total) / float(lut_total),
                }
        except Exception:
            pass

    # Optional speed profiling (LUT side + float side)
    if bool(getattr(sp, "enable", False)):
        results["speed_float"] = measure_latency(
            lambda: adapter.forward_float(x),
            int(getattr(sp, "warmup_iters", 5)),
            int(getattr(sp, "measure_iters", 20)),
        )
        results["speed_ref"] = measure_latency(
            lambda: forward_reference(x, packed),
            int(getattr(sp, "warmup_iters", 5)),
            int(getattr(sp, "measure_iters", 20)),
        )
        results["speed_dense_numpy"] = measure_latency(
            lambda: forward_dense_numpy(x, packed),
            int(getattr(sp, "warmup_iters", 5)),
            int(getattr(sp, "measure_iters", 20)),
        )

    # Numba LUT backend (optional)
    y_numba = None
    if numba_available():
        try:
            import numba as nb  # optional

            results["numba"]["version"] = getattr(nb, "__version__", None)
            try:
                results["numba"]["num_threads"] = int(nb.get_num_threads())
            except Exception:
                results["numba"]["num_threads"] = None

            t0 = perf_counter()
            warmup_numba(packed, in_dim=int(adapter.in_dim), out_dim=int(adapter.out_dim))
            results["numba"]["compile_ms"] = float((perf_counter() - t0) * 1000.0)

            y_numba = forward_dense_numba(x, packed)
            results["numba_vs_ref"] = _diff_report(y_numba, y_ref)
            results["output_sanity_numba"] = _diff_report(y_numba, y_float)
            results["debug"]["worst_output_sanity_numba"] = _worst_case(y_hat=y_numba, y_ref=y_float, x=x)
        except Exception as e:
            results["numba"]["error"] = f"{type(e).__name__}: {e}"
            results["numba"]["compile_ms"] = None
            results["numba_vs_ref"] = None
            results["output_sanity_numba"] = None
    else:
        results["numba"]["compile_ms"] = None
        results["numba_vs_ref"] = None
        results["output_sanity_numba"] = None

    if bool(getattr(sp, "enable", False)):
        if y_numba is not None:
            results["speed_dense_numba"] = measure_latency(
                lambda: forward_dense_numba(x, packed),
                int(getattr(sp, "warmup_iters", 5)),
                int(getattr(sp, "measure_iters", 20)),
            )
        else:
            results["speed_dense_numba"] = None

    # Flat convenience keys
    results["dense_numpy_vs_ref_max_abs"] = results["dense_numpy_vs_ref"]["max_abs"]
    results["dense_numpy_vs_ref_rmse"] = results["dense_numpy_vs_ref"]["rmse"]
    if results.get("numba_vs_ref") is not None:
        results["numba_vs_ref_max_abs"] = results["numba_vs_ref"]["max_abs"]
        results["numba_vs_ref_rmse"] = results["numba_vs_ref"]["rmse"]

    # Compute (optional) speedups vs B-spline baselines when available
    try:
        if results.get("speed_bspline_numpy") and results.get("speed_dense_numpy"):
            bs_np = float(results["speed_bspline_numpy"]["per_iter_ms"])
            lut_np = float(results["speed_dense_numpy"]["per_iter_ms"])
            results["speedup_lut_numpy_vs_bspline_numpy"] = (bs_np / lut_np) if lut_np > 0 else None
    except Exception:
        pass

    try:
        if results.get("speed_bspline_numba") and results.get("speed_dense_numba"):
            bs_nb = float(results["speed_bspline_numba"]["per_iter_ms"])
            lut_nb = float(results["speed_dense_numba"]["per_iter_ms"])
            results["speedup_lut_numba_vs_bspline_numba"] = (bs_nb / lut_nb) if lut_nb and lut_nb > 0 else None
    except Exception:
        pass

    # Run params (contract + provenance)
    inp_raw = results.get("_raw_evaluation_inputs", None)
    inp = _evaluation_inputs(cfg, {"evaluation_inputs": {"inputs": inp_raw}} if isinstance(inp_raw, dict) else None)

    dist = str(getattr(inp, "distribution", "normal") or "normal").lower().strip()
    clip_min = getattr(inp, "clip_min", None)
    clip_max = getattr(inp, "clip_max", None)

    x_gen_mean = float(getattr(inp, "mean", 0.0)) if dist == "normal" else None
    x_gen_std = float(getattr(inp, "std", 1.0)) if dist == "normal" else None
    x_gen_x_min = float(getattr(inp, "x_min", -2.2)) if dist == "uniform" else None
    x_gen_x_max = float(getattr(inp, "x_max", 2.2)) if dist == "uniform" else None

    results["run_params"].update(
        {
            "label": str(label),
            "N": int(x.shape[0]),
            "runtime_seed": int(getattr(cfg.runtime, "seed", 0)),
            "evaluation_input_seed": int(getattr(cfg.evaluation_inputs, "seed", 42)),
            "in_dim": int(adapter.in_dim),
            "out_dim": int(adapter.out_dim),
            "edges": int(len(edges)),
            "device": str(getattr(cfg.runtime, "device", "cpu")),
            "L": int(cfg.converter.build_lut.L),
            "interp": str(cfg.converter.interp.mode),
            "value_representation": sem["value_representation"],
            "boundary_mode": sem["boundary_mode"],
            "x_min": float(x_min),
            "x_max": float(x_max),
            "x_sample_min": float(np.min(x)),
            "x_sample_max": float(np.max(x)),
            "x_gen_distribution": str(dist),
            "x_gen_mean": x_gen_mean,
            "x_gen_std": x_gen_std,
            "x_gen_x_min": x_gen_x_min,
            "x_gen_x_max": x_gen_x_max,
            "x_gen_clip_min": (float(clip_min) if clip_min is not None else None),
            "x_gen_clip_max": (float(clip_max) if clip_max is not None else None),
            "dtype": str(qspec["dtype"]),
            "scheme": str(qspec["scheme"]),
            "qmin": int(qspec["qmin"]),
            "qmax": int(qspec["qmax"]),
            "build_ms": float(build_ms),
            "ref_ms": float(ref_ms),
            "dense_numpy_ms": float(dense_ms),
            "oob_behavior_effective": sem["oob_behavior"],
            "oob_mode_cfg": sem.get("oob_mode_cfg", ""),
            "oob_policy_mode": sem.get("oob_policy_mode", ""),
        }
    )

    dump_json(out_dir / "results.json", results)
    _print_results_summary(results)


def run(config_path: str | Path) -> Path:
    cfg, raw = load_and_validate(config_path)

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = _resolve_out_dir(str(cfg.logging.out_dir), str(cfg.logging.versioning), repo_root)
    ensure_dir(out_dir)

    dump_yaml(out_dir / "resolved_config.yaml", raw)
    dump_json(out_dir / "env.json", get_env_snapshot())

    meta: Dict[str, Any] = {
        "experiment": {
            "name": cfg.experiment.name,
            "group": cfg.experiment.group,
            "description": cfg.experiment.description,
        },
        "out_dir": str(out_dir.as_posix()),
        "git_hash": get_git_hash(repo_root),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "converter_enabled": bool(cfg.converter.enabled),
        "float_backend": cfg.float_model.backend,
        "pip_freeze": pip_freeze(),
        "pykan": get_pykan_info(),
    }
    dump_json(out_dir / "run_meta.json", meta)

    results: Dict[str, Any] = {
        "experiment": meta["experiment"],
        "float_backend": cfg.float_model.backend,
        "converter_enabled": bool(cfg.converter.enabled),
    }

    # Keep raw evaluation input-generation settings for provenance; accept legacy calibration.inputs.
    results["_raw_evaluation_inputs"] = (raw.get("evaluation_inputs") or raw.get("calibration") or {}).get("inputs", None) if isinstance(raw, dict) else None

    seed = int(getattr(cfg.runtime, "seed", 0))
    rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    # Dummy pipeline
    # -------------------------------------------------------------------------
    if cfg.float_model.backend == "dummy":
        arch = cfg.float_model.arch or {}
        adapter = DummyKANAdapter(
            in_dim=int(arch.get("in_dim", 2)),
            out_dim=int(arch.get("out_dim", 2)),
            num_knots=int(arch.get("num_knots", 9)),
            x_min=float(arch.get("x_min", -3.0)),
            x_max=float(arch.get("x_max", 3.0)),
            seed=seed,
        )
        edges = adapter.extract_edges()
        N = min(4096, max(256, int(cfg.evaluation_inputs.num_samples)))
        x = rng.normal(size=(N, adapter.in_dim)).astype(np.float32)
        _run_single_layer_pipeline(adapter, edges, x, cfg, out_dir, results, label="dummy")
        return out_dir

    # -------------------------------------------------------------------------
    # Jacobi-KAN pipeline (single-layer, torch-free)
    # -------------------------------------------------------------------------
    if cfg.float_model.backend in ("jacobi", "jacobikan", "jacobi_kan"):
        arch = cfg.float_model.arch or {}
        adapter = JacobiKANSingleLayerAdapter.from_arch(arch=arch, seed=seed)
        edges = adapter.extract_edges()

        N = int(cfg.evaluation_inputs.num_samples)
        inp = _evaluation_inputs(cfg, raw)
        dist = str(getattr(inp, "distribution", "normal") or "normal").lower().strip()

        if dist == "uniform":
            x_min = float(getattr(inp, "x_min", -2.2))
            x_max = float(getattr(inp, "x_max", 2.2))
            x = rng.uniform(low=x_min, high=x_max, size=(N, adapter.in_dim)).astype(np.float32)
        elif dist == "normal":
            mean = float(getattr(inp, "mean", 0.0))
            std = float(getattr(inp, "std", 1.0))
            x = rng.normal(loc=mean, scale=std, size=(N, adapter.in_dim)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported evaluation_inputs.inputs.distribution='{dist}' (use 'normal' or 'uniform').")

        clip_min = getattr(inp, "clip_min", None)
        clip_max = getattr(inp, "clip_max", None)
        if clip_min is not None and clip_max is not None:
            x = np.clip(x, float(clip_min), float(clip_max)).astype(np.float32, copy=False)

        _run_single_layer_pipeline(adapter, edges, x, cfg, out_dir, results, label="jacobi")
        return out_dir

    # -------------------------------------------------------------------------
    # PyKAN pipeline (single-layer)
    # -------------------------------------------------------------------------
    if cfg.float_model.backend == "pykan":
        arch = cfg.float_model.arch or {}
        device = str(arch.get("device", "cpu"))

        from src.models.kan_wrapper import PyKANSingleLayerAdapter

        adapter = PyKANSingleLayerAdapter.from_arch(
            arch=arch,
            checkpoint=getattr(cfg.float_model, "checkpoint", None),
            device=device,
        )
        edges = adapter.extract_edges()

        N = int(cfg.evaluation_inputs.num_samples)
        inp = _evaluation_inputs(cfg, raw)
        dist = str(getattr(inp, "distribution", "normal") or "normal").lower().strip()

        if dist == "uniform":
            x_min = float(getattr(inp, "x_min", -2.2))
            x_max = float(getattr(inp, "x_max", 2.2))
            x = rng.uniform(low=x_min, high=x_max, size=(N, adapter.in_dim)).astype(np.float32)
        elif dist == "normal":
            mean = float(getattr(inp, "mean", 0.0))
            std = float(getattr(inp, "std", 1.0))
            x = rng.normal(loc=mean, scale=std, size=(N, adapter.in_dim)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported evaluation_inputs.inputs.distribution='{dist}' (use 'normal' or 'uniform').")

        clip_min = getattr(inp, "clip_min", None)
        clip_max = getattr(inp, "clip_max", None)
        if clip_min is not None and clip_max is not None:
            x = np.clip(x, float(clip_min), float(clip_max)).astype(np.float32, copy=False)

        _run_single_layer_pipeline(adapter, edges, x, cfg, out_dir, results, label="pykan")
        return out_dir

    # -------------------------------------------------------------------------
    # Other backends: scaffold only
    # -------------------------------------------------------------------------
    dump_json(out_dir / "results.json", results)
    return out_dir
