# LUT-KAN: Segment-wise LUT Quantization for Fast KAN Inference

[![arXiv](https://img.shields.io/badge/arXiv-2601.03332-b31b1b.svg)](https://arxiv.org/abs/2601.03332)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides a reproducible benchmark suite for **segment-wise LUT (look-up table) inference** for Kolmogorov-Arnold Networks (KAN), supporting both **B-spline** and **Jacobi polynomial** basis functions.

**v2.0 Features:**
- 🆕 **Jacobi polynomial basis**: Chebyshev (1st/2nd kind), Legendre, Gegenbauer
- 🆕 **MCU benchmarking**: Cycle-accurate ARM Cortex-M3 via QEMU
- 🆕 **C code export**: Direct deployment to embedded systems
- 🆕 **Publication tools**: IEEE/ACM-ready tables and figures

## Key Results

LUT inference achieves significant speedups over direct polynomial evaluation:

| Basis | Platform | Speedup Range | Best Config |
|-------|----------|---------------|-------------|
| B-spline | CPU (NumPy) | 11–14× | L=64, int8 |
| B-spline | CPU (Numba) | 9.5–11× | L=64, int8 |
| Jacobi (Chebyshev) | MCU (Cortex-M3) | 2–6× | L=32–64, degree=3–5 |

## Repository Layout

```
lut-kan/
├── configs/
│   ├── jacobi_types/       # Jacobi polynomial configurations
│   └── sweeps/             # Parameter sweep configs
├── scripts/
│   ├── mcu_qemu_benchmark.py    # ARM Cortex-M3 benchmark
│   ├── export_lut_to_c.py       # LUT → C header export
│   ├── publication_analysis.py  # IEEE/ACM tables & figures
│   └── unified_benchmark_sweeper.py
├── src/
│   ├── kernels/            # B-spline and LUT backends
│   ├── models/             # Adapters (PyKAN, Jacobi, B-spline)
│   ├── quant/              # LUT builder & quantization
│   └── metrics/            # Accuracy and performance metrics
└── tests/                  # Numerical correctness tests
```

## Installation

### Core Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### Optional: PyKAN (for B-spline reference models)

```bash
# Remove conflicting package if installed
pip uninstall -y kan

# Install PyKAN
pip install git+https://github.com/KindXiaoming/pykan.git
```

### Optional: MCU Benchmarking

Requires ARM GCC toolchain and QEMU:

```bash
# Ubuntu/Debian
sudo apt install gcc-arm-none-eabi qemu-system-arm

# macOS (Homebrew)
brew install --cask gcc-arm-embedded
brew install qemu
```

## Quick Start

### 1. Run B-spline Benchmark

```bash
python scripts/run_experiment.py configs/exp_pykan_lut_inrange_closed.yaml
```

### 2. Run Jacobi Benchmark

```bash
# Single configuration
python scripts/run_experiment.py configs/exp_jacobi_lut.yaml

# Sweep over polynomial types
python scripts/bench_jacobi_grid.py --degrees 3,5,7 --Ls 32,64,128
```

### 3. MCU Benchmark (ARM Cortex-M3)

```bash
python scripts/mcu_qemu_benchmark.py --degrees 3,5 --Ls 32,64 --dim 8
```

### 4. Export LUT to C

```bash
python scripts/export_lut_to_c.py --in_dim 8 --out_dim 8 --degree 3 --L 64 --output layer.h
```

### 5. Generate Publication Tables

```bash
python scripts/publication_analysis.py outputs/benchmark_results/
```

## Supported Basis Functions

### B-splines (via PyKAN)
- Cox-de Boor recursive evaluation
- Configurable spline order (default: cubic)

### Jacobi Polynomials
| Name | Parameters (α, β) | Notes |
|------|-------------------|-------|
| Chebyshev 1st | (-0.5, -0.5) | Best speedup in benchmarks |
| Chebyshev 2nd | (0.5, 0.5) | |
| Legendre | (0, 0) | Orthogonal on [-1,1] |
| Gegenbauer | (λ-0.5, λ-0.5) | Ultraspherical |

## Quantization Schemes

1. **Symmetric (int8)**: `ŷ = scale × q`, where `q ∈ [-127, 127]`
2. **Asymmetric (uint8)**: `ŷ = y_min + scale × q`, where `q ∈ [0, 255]`

## Benchmark Results

### CPU Performance (NumPy/Numba)

| L | B-spline (NumPy) | LUT (NumPy) | Speedup | B-spline (Numba) | LUT (Numba) | Speedup |
|---|------------------|-------------|---------|------------------|-------------|---------|
| 16 | 29.28 ms | 2.13 ms | 13.9× | 6.41 ms | 0.59 ms | 11.1× |
| 32 | 26.47 ms | 2.48 ms | 11.5× | 5.70 ms | 0.64 ms | 9.5× |
| 64 | 28.94 ms | 2.24 ms | 13.1× | 6.10 ms | 0.61 ms | 10.1× |
| 128 | 27.24 ms | 2.34 ms | 12.0× | 6.08 ms | 0.60 ms | 10.2× |

### MCU Performance (ARM Cortex-M3, QEMU)

| Degree | L | Float Cycles | LUT Cycles | Speedup |
|--------|---|--------------|------------|---------|
| 3 | 32 | 45,230 | 12,450 | 3.6× |
| 5 | 64 | 89,120 | 18,320 | 4.9× |
| 7 | 64 | 156,800 | 24,100 | 6.5× |

### Accuracy vs LUT Resolution

| L | MAE (int8) | Max Error | MAE (uint8) | Max Error |
|---|------------|-----------|-------------|-----------|
| 16 | 6.39e-4 | 3.23e-3 | 6.41e-4 | 3.24e-3 |
| 32 | 3.18e-4 | 1.63e-3 | 3.18e-4 | 1.62e-3 |
| 64 | 1.60e-4 | 8.02e-4 | 1.59e-4 | 8.33e-4 |
| 128 | 8.37e-5 | 4.29e-4 | 8.01e-5 | 4.26e-4 |

## Configuration Reference

### Jacobi Configuration Example

```yaml
float_model:
  backend: jacobi
  adapter: jacobi
  arch:
    in_dim: 16
    out_dim: 16
    degree: 3
    alpha: -0.5   # Chebyshev 1st kind
    beta: -0.5
    use_tanh: true
    x_min: -3.0
    x_max: 3.0

converter:
  build_lut:
    L: 64
  interp:
    mode: linear
  quant:
    dtype: uint8
    scheme: asymmetric
```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{Kuznetsov_2026,
  title   = {LUT-KAN: Segment-wise LUT Quantization for Fast KAN Inference},
  author  = {Kuznetsov, Oleksandr},
  journal = {arXiv preprint arXiv:2601.03332},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.03332}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Author**: Oleksandr Kuznetsov
- **Email**: oleksandr.o.kuznetsov@gmail.com
- **Repository**: https://github.com/KuznetsovKarazin/lut-kan

## Acknowledgments

- [PyKAN](https://github.com/KindXiaoming/pykan) - B-spline KAN implementation
- [JacobiKAN](https://github.com/SpaceLearner/JacobiKAN) - Jacobi polynomial reference
