"""Reproduce the NCA DoS case study with corrected LUT sampling.

This script is intentionally revision-specific.  It evaluates one frozen PyKAN
model under the same end-to-end data/model state using:

* PyTorch/PyKAN float inference (classification reference),
* a matched NumPy B-spline baseline,
* a matched Numba B-spline baseline,
* corrected endpoint-inclusive LUTs in NumPy and Numba.

It also sweeps LUT resolution L and reports fixed-threshold classification
metrics, matched-backend latency, and LUT artifact memory.  No external
"calibration" dataset is used to determine quantization ranges: the LUT builder
derives them from the sampled spline values in each segment.

The script supports the additive KAN architecture used in the paper
([78, 32, 16, 1]).  It deliberately refuses active symbolic or multiplicative
nodes because the LUT/B-spline runtime in this repository does not implement
those branches.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from src.kernels.bspline_backend_dense_numba import (
    forward_bspline_dense_numba,
    warmup_bspline_numba,
)
from src.kernels.bspline_backend_dense_numpy import forward_bspline_dense_numpy
from src.kernels.bspline_contract import pack_bspline_dense_layer_from_pykankan_adapter
from src.kernels.lut_backend_dense_numba import forward_dense_numba, warmup_numba
from src.kernels.lut_backend_dense_numpy import forward_dense_numpy
from src.kernels.lut_contract import pack_dense_layer
from src.models.kan_wrapper import PyKANSingleLayerAdapter
from src.quant.lut_builder import artifact_memory_bytes, build_lut_for_edges
from src.quant.lut_io import save_lut_npz


def _to_numpy(x) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _torch_load(path: Path):
    import torch

    # PyTorch >=2.6 changed the default of weights_only.  The experiment files
    # are local trusted artifacts produced by this project and contain ordinary
    # Python dictionaries/tensors.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_experiment(experiment_dir: Path):
    try:
        import torch  # noqa: F401
        from kan import KAN
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "This script requires torch and the PyKAN package providing `from kan import KAN`."
        ) from exc

    dataset_path = experiment_dir / "dataset.pt"
    model_path = experiment_dir / "trained_model.pt"
    if not dataset_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Expected {dataset_path} and {model_path}. Run the leakage-free DoS training first."
        )

    dataset = _torch_load(dataset_path)
    model_data = _torch_load(model_path)
    if not isinstance(model_data, dict) or "architecture" not in model_data or "model_state_dict" not in model_data:
        raise ValueError("trained_model.pt does not contain architecture/model_state_dict")

    model = KAN(**model_data["architecture"])
    incompatible = model.load_state_dict(model_data["model_state_dict"], strict=True)
    # Strict loading normally returns empty missing/unexpected lists.  Keep the
    # check explicit for PyTorch/PyKAN variants with a compatible return object.
    if getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None):
        raise RuntimeError(f"Checkpoint load mismatch: {incompatible}")
    model.to("cpu")
    model.eval()

    if "test_input" not in dataset or "test_label" not in dataset:
        raise ValueError("dataset.pt must contain test_input and test_label")

    x_test = _to_numpy(dataset["test_input"]).astype(np.float32, copy=False)
    y_test = _to_numpy(dataset["test_label"]).reshape(-1).astype(np.int64, copy=False)
    if x_test.ndim != 2 or x_test.shape[0] != y_test.size:
        raise ValueError(f"Unexpected test shapes: x={x_test.shape}, y={y_test.shape}")

    return model, model_data, x_test, y_test


def _validate_supported_model(model) -> None:
    depth = int(getattr(model, "depth", len(getattr(model, "act_fun", []))))
    if depth <= 0:
        raise ValueError("Cannot determine KAN depth")

    # The paper uses a purely additive KAN.  Refuse multiplication nodes rather
    # than silently benchmarking a different graph.
    width = getattr(model, "width", None)
    if width is not None:
        for level, w in enumerate(width[1:], start=1):
            if isinstance(w, (list, tuple)) and len(w) >= 2 and int(w[1]) != 0:
                raise NotImplementedError(
                    f"Multiplicative KAN nodes detected at level {level}: width={w}. "
                    "The NCA revision evaluator supports the additive architecture used in the paper."
                )

    # Symbolic branches are not part of the LUT runtime.  The standard trained
    # model has zero symbolic masks; make this an auditable precondition.
    for l, sym in enumerate(getattr(model, "symbolic_fun", [])):
        mask = getattr(sym, "mask", None)
        if mask is not None and np.any(np.abs(_to_numpy(mask)) > 0):
            raise NotImplementedError(
                f"Active symbolic edges detected in layer {l}; matched LUT/B-spline evaluation would be incomplete."
            )


def _extract_post_layer_affine(model, layer_idx: int, out_dim: int):
    """Return the affine transform applied after a numerical additive KANLayer.

    PyKAN MultKAN.forward applies subnode affine followed by node affine.  With
    no multiplication nodes (the paper's architecture), these collapse to:

        x_out = a * y_edge_sum + b

    where a = node_scale * subnode_scale and
          b = node_scale * subnode_bias + node_bias.
    """

    def vec(seq_name: str, default: float) -> np.ndarray:
        seq = getattr(model, seq_name, None)
        if seq is None:
            return np.full(out_dim, default, dtype=np.float32)
        arr = _to_numpy(seq[layer_idx]).astype(np.float32, copy=False).reshape(-1)
        if arr.size != out_dim:
            raise ValueError(
                f"{seq_name}[{layer_idx}] has {arr.size} values; expected {out_dim} for additive layer"
            )
        return arr

    sub_scale = vec("subnode_scale", 1.0)
    sub_bias = vec("subnode_bias", 0.0)
    node_scale = vec("node_scale", 1.0)
    node_bias = vec("node_bias", 0.0)

    scale = node_scale * sub_scale
    bias = node_scale * sub_bias + node_bias
    return scale.astype(np.float32), bias.astype(np.float32)


def _build_layer_contracts(model, L: int, artifact_dir: Path):
    lut_layers = []
    bspline_layers = []
    affines = []
    memory_rows = []

    for layer_idx in range(len(model.act_fun)):
        adapter = PyKANSingleLayerAdapter(model=model, layer_idx=layer_idx, device="cpu")
        edges = adapter.extract_edges()

        art = build_lut_for_edges(
            edges=edges,
            L=int(L),
            interp="linear",
            y_range_method="minmax",
            lower_pct=0.0,
            upper_pct=100.0,
            dtype="int8",
            scheme="symmetric",
            qmin=-127,
            qmax=127,
            meta_dtype="float16",
            value_representation="spline_component",
            oob_behavior="clip",
            boundary_mode="closed",
        )
        if int(art.format_version) < 2 or getattr(art, "sample_grid", "") != "endpoint_inclusive":
            raise RuntimeError("Builder did not produce the corrected endpoint-inclusive LUT artifact")

        artifact_path = artifact_dir / f"L{L}" / f"layer_{layer_idx}.npz"
        save_lut_npz(artifact_path, art)

        packed_lut = pack_dense_layer(
            art,
            edges=edges,
            in_dim=adapter.in_dim,
            out_dim=adapter.out_dim,
            boundary_mode="closed",
        )
        packed_bspline = pack_bspline_dense_layer_from_pykankan_adapter(
            adapter,
            boundary_mode="closed",
        )
        affine = _extract_post_layer_affine(model, layer_idx, adapter.out_dim)

        lut_layers.append(packed_lut)
        bspline_layers.append(packed_bspline)
        affines.append(affine)
        memory_rows.append(
            {
                "L": int(L),
                "layer": int(layer_idx),
                "in_dim": int(adapter.in_dim),
                "out_dim": int(adapter.out_dim),
                "edges": int(adapter.in_dim * adapter.out_dim),
                "segments_K": int(art.knots.size - 1),
                "samples_per_segment_L": int(L),
                "artifact_bytes": int(artifact_memory_bytes(art)),
                "npz_bytes": int(artifact_path.stat().st_size),
            }
        )

    return lut_layers, bspline_layers, affines, memory_rows


def _forward_chain(x: np.ndarray, layers: Sequence, affines: Sequence, layer_forward: Callable) -> np.ndarray:
    h = np.asarray(x, dtype=np.float32)
    for packed, (scale, bias) in zip(layers, affines):
        h = layer_forward(h, packed)
        h = h * scale[None, :] + bias[None, :]
        h = np.asarray(h, dtype=np.float32)
    return h


def _predict_chunked(
    x: np.ndarray,
    layers: Sequence,
    affines: Sequence,
    layer_forward: Callable,
    chunk_size: int,
) -> np.ndarray:
    outs = []
    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        outs.append(_forward_chain(x[start:stop], layers, affines, layer_forward))
    return np.concatenate(outs, axis=0)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    # Stable sigmoid without scipy dependency.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _classification_metrics(y_true: np.ndarray, logits: np.ndarray, threshold: float, method: str, L=None):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    prob = _sigmoid(logits)
    pred = (prob > float(threshold)).astype(np.int64)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    return {
        "method": method,
        "L": "" if L is None else int(L),
        "threshold": float(threshold),
        "n": int(y_true.size),
        "benign_n": int((y_true == 0).sum()),
        "attack_n": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _time_callable(fn: Callable[[], np.ndarray], warmup: int, iters: int):
    for _ in range(max(0, warmup)):
        _ = fn()
    samples = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        _ = fn()
        t1 = time.perf_counter_ns()
        samples[i] = (t1 - t0) / 1e6
    return {
        "mean_ms": float(samples.mean()),
        "std_ms": float(samples.std(ddof=1)) if iters > 1 else 0.0,
        "median_ms": float(np.median(samples)),
        "min_ms": float(samples.min()),
        "max_ms": float(samples.max()),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _parse_int_list(text: str) -> list[int]:
    vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not vals or any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive integers")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-dir", type=Path, default=Path("../kan-dos-detection/experiment_data_ncaa_r1"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/ncaa_r1_dos"))
    ap.add_argument("--L", type=_parse_int_list, default=[16, 32, 64, 128], help="e.g. 16,32,64,128")
    ap.add_argument("--batch-sizes", type=_parse_int_list, default=[1, 16, 256, 1024])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--metric-chunk", type=int, default=4096)
    ap.add_argument("--validation-samples", type=int, default=512)
    ap.add_argument("--bspline-validation-tol", type=float, default=5e-3)
    args = ap.parse_args()

    if not (0.0 < args.threshold < 1.0):
        ap.error("--threshold must be in (0,1)")
    if args.iters <= 0 or args.metric_chunk <= 0 or args.validation_samples <= 0:
        ap.error("iters, metric-chunk and validation-samples must be positive")

    out = args.output_dir
    artifacts_dir = out / "artifacts"
    out.mkdir(parents=True, exist_ok=True)

    model, model_data, x_test, y_test = _load_experiment(args.experiment_dir)
    _validate_supported_model(model)

    import torch

    torch.set_grad_enabled(False)
    x_ref = torch.from_numpy(x_test)
    with torch.no_grad():
        float_logits = model(x_ref).detach().cpu().numpy().astype(np.float32, copy=False)

    metrics_rows = [
        _classification_metrics(y_test, float_logits, args.threshold, method="Float PyKAN", L=None)
    ]
    latency_rows: list[dict] = []
    memory_rows: list[dict] = []
    validation_rows: list[dict] = []

    # Build the exact B-spline contracts once from L-independent model data via
    # the first requested LUT build; validate numerical chaining against PyKAN.
    first_bspline_layers = None
    first_affines = None

    for L in args.L:
        print(f"\n=== Building/evaluating L={L} ===")
        lut_layers, bspline_layers, affines, mem = _build_layer_contracts(model, L, artifacts_dir)
        memory_rows.extend(mem)
        if first_bspline_layers is None:
            first_bspline_layers = bspline_layers
            first_affines = affines

            n_val = min(args.validation_samples, x_test.shape[0])
            xv = np.ascontiguousarray(x_test[:n_val], dtype=np.float32)
            py = float_logits[:n_val]
            bs_np = _forward_chain(xv, bspline_layers, affines, forward_bspline_dense_numpy)
            bs_max = float(np.max(np.abs(bs_np - py)))
            bs_mae = float(np.mean(np.abs(bs_np - py)))
            validation_rows.append(
                {
                    "check": "PyKAN_vs_matched_Bspline_NumPy",
                    "n": n_val,
                    "mae": bs_mae,
                    "max_abs": bs_max,
                    "tolerance": float(args.bspline_validation_tol),
                    "passed": bool(bs_max <= args.bspline_validation_tol),
                }
            )
            print(f"PyKAN vs matched NumPy B-spline: MAE={bs_mae:.3e}, MaxAbs={bs_max:.3e}")
            if bs_max > args.bspline_validation_tol:
                raise RuntimeError(
                    "Matched NumPy B-spline chain does not reproduce the frozen PyKAN model closely enough. "
                    "Do not report latency until the contract mismatch is resolved."
                )

            for packed in bspline_layers:
                warmup_bspline_numba(packed)
            bs_nb = _forward_chain(xv, bspline_layers, affines, forward_bspline_dense_numba)
            nb_max = float(np.max(np.abs(bs_nb - bs_np)))
            nb_mae = float(np.mean(np.abs(bs_nb - bs_np)))
            validation_rows.append(
                {
                    "check": "Bspline_NumPy_vs_Numba",
                    "n": n_val,
                    "mae": nb_mae,
                    "max_abs": nb_max,
                    "tolerance": 1e-4,
                    "passed": bool(nb_max <= 1e-4),
                }
            )
            print(f"B-spline NumPy vs Numba: MAE={nb_mae:.3e}, MaxAbs={nb_max:.3e}")
            if nb_max > 1e-4:
                raise RuntimeError("NumPy and Numba B-spline baselines disagree")

        for packed in lut_layers:
            warmup_numba(packed, in_dim=packed.q_flat.shape[0], out_dim=packed.q_flat.shape[1])

        # Fixed-threshold downstream metrics for the corrected LUT.  Both
        # backends are evaluated independently to verify implementation parity.
        logits_lut_np = _predict_chunked(
            x_test, lut_layers, affines, forward_dense_numpy, args.metric_chunk
        )
        logits_lut_nb = _predict_chunked(
            x_test, lut_layers, affines, forward_dense_numba, args.metric_chunk
        )
        lut_backend_max = float(np.max(np.abs(logits_lut_np - logits_lut_nb)))
        lut_backend_mae = float(np.mean(np.abs(logits_lut_np - logits_lut_nb)))
        validation_rows.append(
            {
                "check": f"LUT_NumPy_vs_Numba_L{L}",
                "n": int(y_test.size),
                "mae": lut_backend_mae,
                "max_abs": lut_backend_max,
                "tolerance": 1e-4,
                "passed": bool(lut_backend_max <= 1e-4),
            }
        )
        if lut_backend_max > 1e-4:
            raise RuntimeError(f"NumPy and Numba LUT outputs disagree for L={L}")

        metrics_rows.append(
            _classification_metrics(y_test, logits_lut_np, args.threshold, method="LUT NumPy", L=L)
        )
        metrics_rows.append(
            _classification_metrics(y_test, logits_lut_nb, args.threshold, method="LUT Numba", L=L)
        )

        # Latency under matched input/model/hardware conditions.  Exact B-spline
        # baselines are re-used because they do not depend on L.
        for batch in args.batch_sizes:
            if batch > x_test.shape[0]:
                continue
            xb = np.ascontiguousarray(x_test[:batch], dtype=np.float32)

            xt_batch = torch.from_numpy(xb)

            def _pytorch_forward(xt=xt_batch):
                with torch.no_grad():
                    return model(xt)

            funcs = {
                "Float PyKAN": _pytorch_forward,
                "B-spline NumPy": lambda xb=xb: _forward_chain(
                    xb, bspline_layers, affines, forward_bspline_dense_numpy
                ),
                "LUT NumPy": lambda xb=xb: _forward_chain(
                    xb, lut_layers, affines, forward_dense_numpy
                ),
                "B-spline Numba": lambda xb=xb: _forward_chain(
                    xb, bspline_layers, affines, forward_bspline_dense_numba
                ),
                "LUT Numba": lambda xb=xb: _forward_chain(
                    xb, lut_layers, affines, forward_dense_numba
                ),
            }
            timed = {name: _time_callable(fn, args.warmup, args.iters) for name, fn in funcs.items()}
            speed_np = timed["B-spline NumPy"]["mean_ms"] / timed["LUT NumPy"]["mean_ms"]
            speed_nb = timed["B-spline Numba"]["mean_ms"] / timed["LUT Numba"]["mean_ms"]
            pytorch_ms = timed["Float PyKAN"]["mean_ms"]

            for name, t in timed.items():
                if name == "Float PyKAN":
                    backend = "PyTorch"
                    representation = "Float PyKAN"
                    matched_speedup = None
                    stack_speedup = 1.0
                else:
                    backend = "NumPy" if "NumPy" in name else "Numba"
                    representation = "B-spline" if name.startswith("B-spline") else "LUT"
                    matched_speedup = 1.0
                    if name == "LUT NumPy":
                        matched_speedup = speed_np
                    elif name == "LUT Numba":
                        matched_speedup = speed_nb
                    stack_speedup = pytorch_ms / t["mean_ms"]

                latency_rows.append(
                    {
                        "L": int(L),
                        "batch": int(batch),
                        "backend": backend,
                        "representation": representation,
                        **t,
                        "ms_per_sample": float(t["mean_ms"] / batch),
                        "matched_speedup_vs_bspline": matched_speedup,
                        "stack_speedup_vs_pytorch": float(stack_speedup),
                        "warmup": int(args.warmup),
                        "iters": int(args.iters),
                    }
                )
            print(
                f"batch={batch}: matched speedup NumPy={speed_np:.2f}x, Numba={speed_nb:.2f}x"
            )

    # The B-spline baseline is independent of L.  Repeated measurements are
    # retained in the raw CSV so every LUT resolution has a directly matched
    # row; downstream analysis can aggregate or select L=64 for the paper.

    # Summarize total LUT memory by L.
    totals = []
    for L in args.L:
        rows = [r for r in memory_rows if int(r["L"]) == int(L)]
        totals.append(
            {
                "L": int(L),
                "layer": "TOTAL",
                "in_dim": "",
                "out_dim": "",
                "edges": int(sum(int(r["edges"]) for r in rows)),
                "segments_K": "",
                "samples_per_segment_L": int(L),
                "artifact_bytes": int(sum(int(r["artifact_bytes"]) for r in rows)),
                "npz_bytes": int(sum(int(r["npz_bytes"]) for r in rows)),
            }
        )
    memory_rows.extend(totals)

    _write_csv(out / "classification_metrics.csv", metrics_rows)
    _write_csv(out / "latency_matched.csv", latency_rows)
    _write_csv(out / "memory_by_layer.csv", memory_rows)
    _write_csv(out / "validation_checks.csv", validation_rows)

    manifest = {
        "experiment_dir": str(args.experiment_dir.resolve()),
        "output_dir": str(out.resolve()),
        "architecture": str(model_data.get("architecture")),
        "test_shape": [int(v) for v in x_test.shape],
        "test_class_counts": {
            "benign": int((y_test == 0).sum()),
            "attack": int((y_test == 1).sum()),
        },
        "L": [int(v) for v in args.L],
        "batch_sizes": [int(v) for v in args.batch_sizes],
        "threshold": float(args.threshold),
        "lut_contract": {
            "sample_grid": "endpoint_inclusive",
            "format_version": 2,
            "quantization": "symmetric int8 [-127,127]",
            "quantization_range_source": "sampled LUT values per segment; no external calibration dataset",
            "interpolation": "linear",
            "boundary_mode": "closed",
            "oob_behavior": "clip",
            "value_representation": "spline_component",
            "meta_dtype": "float16",
        },
        "classification_metrics": metrics_rows,
        "validation_checks": validation_rows,
    }
    with (out / "revision_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nDone.")
    print(f"  classification: {out / 'classification_metrics.csv'}")
    print(f"  matched latency: {out / 'latency_matched.csv'}")
    print(f"  memory: {out / 'memory_by_layer.csv'}")
    print(f"  validation: {out / 'validation_checks.csv'}")
    print(f"  manifest: {out / 'revision_manifest.json'}")


if __name__ == "__main__":
    main()
