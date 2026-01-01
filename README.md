# LUT-KAN: Segment-wise LUT Quantization for Fast KAN Inference

This repository provides a reproducible benchmark suite for **segment-wise LUT (look-up table) inference** for KAN edges,
including (i) **numerically validated** B-spline baselines and (ii) **fair speed comparisons** under matched optimization
levels (NumPy vs NumPy and Numba vs Numba).

The implementation targets CPU inference and emphasizes:
- deterministic and dependency-light deployment (pure NumPy/Numba runtime for LUT inference),
- explicit trade-offs among **accuracy**, **speed**, and **memory**,
- robustness analysis for **out-of-bounds (OOB)** inputs via a policy × boundary-mode matrix.

## Repository layout

```
lut-kan/
  configs/                 experiment specs + generated sweep configs
  scripts/                 CLI entrypoints (used for reproduction)
  src/
    experiments/           experiment runner + instrumentation
    kernels/               B-spline and LUT backends (numpy/numba/reference)
    metrics/               accuracy and OOB-split metrics
    models/                PyKAN adapter (optional dependency)
    quant/                 LUT build/IO and quantization utilities
    utils/                 config parsing and helpers
  tests/                   numerical correctness tests
```

## Environment (reference)

Benchmarks were executed on a Windows workstation (CPU-only):
- CPU: AMD Ryzen 7 7840HS (3.80 GHz)
- RAM: 64 GB
- OS: Windows

## Installation

### Core (no PyKAN)
This installs LUT/B-spline backends, metrics, and aggregation utilities.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

Run unit tests (PyKAN-dependent tests are skipped if PyKAN is not installed):

```bash
pytest -q
```

### Optional: PyKAN reference (required for `exp_pykan_*` configs)
The experiment runner uses PyKAN as a reference model source (import: `from kan import KAN`).

Important: do not install the unrelated PyPI package `kan==0.0.2` (it does not provide `KAN`).
If it is installed, remove it:

```bash
pip uninstall -y kan
```

Install PyKAN:

```bash
pip install pykan
# or (recommended, most robust):
pip install git+https://github.com/KindXiaoming/pykan.git
```

Sanity-check:

```bash
python -c "from kan import KAN; print(KAN)"
```

### Optional: development tools
```bash
pip install -r requirements-dev.txt
```

## Running experiments

### Single run
```bash
python scripts/run_experiment.py configs/exp_pykan_lut_inrange_closed.yaml
```

### Sweep over LUT resolution L
```bash
python scripts/run_experiment.py configs/sweeps/inrange_closed_L16.yaml
python scripts/run_experiment.py configs/sweeps/inrange_closed_L32.yaml
python scripts/run_experiment.py configs/sweeps/inrange_closed_L64.yaml
python scripts/run_experiment.py configs/sweeps/inrange_closed_L128.yaml
```

### Aggregate results into paper-ready tables
```bash
python scripts/collect_results.py --root outputs --outdir outputs/summary --latex_dir outputs/summary_latex
```

Outputs (CSV):
- `all_runs.csv` (one row per run)
- `table_main.csv` (compact trade-off table)
- `table_speed.csv`, `table_accuracy.csv`, `table_memory.csv`
- `table_oob_robustness.csv` (OOB diagnostics)

## Quantization schemes

This codebase supports two per-segment LUT quantization schemes:

1) Symmetric quantization (int8)  
Per segment, store a scale $s>0$ and quantize to $q\in[-127,127]$.  
Dequantization: $\hat y = s\cdot q$.

2) Asymmetric (affine) quantization (uint8)  
Per segment, store $(y_\min, s)$ and quantize to $q\in[0,255]$.  
Dequantization: $\hat y = y_\min + s\cdot q$.  
This implementation uses the explicit $y_\min$ offset and does not accept a manual zero-point.

## Fair speed baseline (NumPy vs NumPy; Numba vs Numba)

We report LUT speedups against an optimized B-spline baseline under the same backend:
- NumPy B-spline evaluation vs NumPy LUT evaluation
- Numba-jitted B-spline evaluation vs Numba-jitted LUT evaluation

Representative results (mean over `n` seeds; configuration: `spline_component`, `clip_x`, `closed`, `int8 symmetric`):

| L | B-spline (NumPy) ms/iter | LUT (NumPy) ms/iter | Speedup | B-spline (Numba) ms/iter | LUT (Numba) ms/iter | Speedup | n |
|---|---|---|---|---|---|---|---|
| 16 | 29.277 | 2.1264 | 13.949 | 6.4110 | 0.589095 | 11.056 | 5 |
| 32 | 26.473 | 2.4775 | 11.491 | 5.6979 | 0.637206 | 9.5175 | 5 |
| 64 | 28.935 | 2.2417 | 13.130 | 6.0992 | 0.605397 | 10.122 | 10 |
| 128 | 27.239 | 2.3415 | 11.981 | 6.0808 | 0.595058 | 10.205 | 5 |

Interpretation:
- LUT inference is approximately 11–14× faster than NumPy B-spline evaluation.
- Under Numba JIT, LUT inference remains approximately 9.5–11× faster than the Numba B-spline baseline.
This isolates the effect of representation (LUT vs B-spline), not merely vectorization/JIT.

## Accuracy vs LUT resolution L (int8 symmetric vs uint8 asymmetric)

Representative accuracy numbers for `half_open` boundary mode (which induces OOB samples), showing both
symmetric int8 and asymmetric uint8:

| L | in MAE (int8 sym) | in max_abs | in MAE (uint8 asym) | in max_abs | OOB-only max_abs | OOB-any frac | n |
|---|---|---|---|---|---|---|---|
| 16 | 0.000639015 | 0.00322639 | 0.000641466 | 0.00324219 | 0.00266178 | 0.101367 | 5 |
| 32 | 0.000317815 | 0.00162551 | 0.000317912 | 0.00161513 | 0.00133172 | 0.101367 | 5 |
| 64 | 0.00016015 | 0.000802402 | 0.00015904 | 0.000833417 | 0.000660668 | 0.101367 | 10 |
| 128 | 8.37491e-05 | 0.000429052 | 8.01496e-05 | 0.000425662 | 0.000347595 | 0.101367 | 5 |

Key points:
- Increasing $L$ improves accuracy (lower MAE/max_abs).
- Symmetric int8 and asymmetric uint8 are close in this setting; the latter is the affine/asymmetric scheme.

## Memory footprint (LUT vs float parameters)

The LUT artifact stores quantized tables plus per-segment metadata. For this benchmark layer, the float parameter size is constant,
while the LUT size grows linearly with $L$:

| L | Float params bytes | LUT artifact bytes | LUT/Float | n |
|---|---|---|---|---|
| 16 | 4608 | 14128 | 3.0660 | 5 |
| 32 | 4608 | 25392 | 5.5104 | 5 |
| 64 | 4608 | 47920 | 10.399 | 10 |
| 128 | 4608 | 92976 | 20.177 | 5 |

This demonstrates an explicit accuracy–memory trade-off: higher $L$ improves fidelity but increases bytes.

## OOB robustness matrix

We measure OOB incidence and errors separately on `in-range` and `OOB-only` subsets for the Cartesian product:
- `oob_policy_mode` ∈ {`clip_x`, `zero_spline`}
- `boundary_mode` ∈ {`half_open`, `closed`}

Example (L=64, `spline_component`, int8 symmetric):

| oob_policy_mode | boundary_mode | OOB-any frac | OOB-only max_abs | n |
|---|---|---|---|---|
| clip_x | closed | 0 | 0 | 10 |
| clip_x | half_open | 0.101367 | 0.000660668 | 10 |
| zero_spline | closed | 0 | 0 | 10 |
| zero_spline | half_open | 0.101367 | 0.000660668 | 10 |

Notes:
- `closed` boundary mode prevents OOB samples by construction (OOB fraction ≈ 0).
- `half_open` produces a controlled OOB fraction (≈ 0.10 in this benchmark).
- OOB sensitivity can be substantially larger when benchmarking full `phi` (base + spline) rather than the contracted `spline_component`.

## Reproducibility and outputs
- Runs are seeded; aggregation reports mean/std/CI95 per configuration.
- Large artifacts under `outputs/` are intentionally excluded from version control (see `.gitignore`).

## License
MIT License (see `LICENSE`).
