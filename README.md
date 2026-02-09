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

# MCU Auto-Benchmark

Automated MCU benchmark pipeline for LUT-KAN supporting both **Jacobi polynomials** and **B-splines**.

## What This Does

1. **Generates** firmware variants with different LUT configurations
2. **Builds** with PlatformIO for each target MCU
3. **Simulates** headlessly on Wokwi (or runs on real hardware)
4. **Collects** timing and accuracy metrics into publication-ready reports

## Quick Start

```bash
# 1. Install dependencies
pip install -r ../requirements-mcu.txt
npm install -g @anthropic-ai/wokwi-cli

# 2. Set Wokwi token (get from https://wokwi.com/dashboard/ci)
export WOKWI_CLI_TOKEN="your-token-here"

# 3. Run smoke test (fast, for CI)
python scripts/all.py --suite smoke --targets esp32 --jobs 4

# 4. View results
cat reports/summary.md
```

## Supported Targets

- `uno` - Arduino UNO (ATmega328P)
- `nano` - Arduino Nano (ATmega328P)
- `mega` - Arduino Mega 2560 (ATmega2560)
- `pico` - Raspberry Pi Pico (ARM Cortex-M0+)
- `esp32` - ESP32 DevKit V1 (Xtensa LX6, **with FPU**)
- `esp32c3` - ESP32-C3 (RISC-V)
- `esp32s3` - ESP32-S3 (Xtensa LX7, **with FPU**)
- `stm32f103` - Blue Pill (ARM Cortex-M3)

All targets verified against [Wokwi's supported hardware](https://docs.wokwi.com/parts/board-list).

## Directory Structure

```
mcu_auto/
├── scripts/
│   ├── all.py              # One-command pipeline
│   ├── gen_cases.py        # Generate firmware from grid
│   ├── build_cases.py      # Compile with PlatformIO
│   ├── run_wokwi.py        # Headless simulation
│   ├── run_hardware.py     # Real board upload (optional)
│   ├── collect_results.py  # Parse logs → CSV/Markdown
│   └── plot_results.py     # Visualization
├── grids/
│   └── grid.yaml           # Parameter sweeps
├── targets/
│   └── <target>/           # Wokwi diagram + config per target
├── templates/
│   └── pio_project/        # PlatformIO project template
├── docs/
│   └── README_MCU.md       # 📖 Full documentation
├── cases/                  # Generated (gitignored)
├── logs/                   # Generated (gitignored)
└── reports/                # Generated (gitignored)
```

## Supported Basis Types

| Basis | Families | Float Baseline | LUT Forward |
|-------|----------|----------------|-------------|
| `jacobi` | Chebyshev T/U, Legendre, Gegenbauer | 3-term recurrence | Segment-wise quantized LUT |
| `bspline` | Cubic B-splines (configurable degree) | Cox-de Boor + SiLU | Segment-wise quantized LUT |

## Example Output

After running `all.py`, check `reports/summary.md`:

## LUT-KAN Inference Speedup on MCU Platforms

The table below presents the inference speedup results using Look-Up Table (LUT) based Kolmogorov–Arnold Networks (KAN) across various microcontroller units (MCUs).  
The data is sourced from the original research paper and highlights the efficiency of the method for both B‑spline and Jacobi polynomial basis functions.  

The columns represent:
- **Target**: MCU platform.
- **Basis**: Type of basis function used (B‑spline or Jacobi polynomial).
- **N**: Number of basis functions.
- **Med. ×**: Median speedup factor compared to full floating‑point computation.
- **Max ×**: Maximum observed speedup factor.
- **Med. Err**: Median approximation error introduced by the LUT method.

| Target      | Basis   |  N  | Med. × | Max × | Med. Err |
|-------------|---------|-----|--------|-------|----------|
| esp32c3     | bspline | 16  | 15.6×  | 22.3× | 0.067    |
| esp32c3     | jacobi  | 48  | 4.4×   | 7.4×  | 0.103    |
| mega        | bspline | 8   | 23.8×  | 27.2× | 0.051    |
| mega        | jacobi  | 48  | 4.8×   | 9.0×  | 0.103    |
| pico        | bspline | 12  | 20.0×  | 28.6× | 0.061    |
| pico        | jacobi  | 48  | 4.1×   | 7.0×  | 0.103    |
| stm32f103   | bspline | 1   | 13.6×  | 13.6× | 0.076    |
| stm32f103   | jacobi  | 24  | 3.9×   | 6.3×  | 0.122    |

**Note:** Higher speedup factors indicate greater computational efficiency, while the error column quantifies the trade‑off in approximation accuracy. B‑spline bases generally achieve higher speedups with lower errors compared to Jacobi polynomial bases on these MCU platforms.


## Customizing Parameter Sweeps

Edit `grids/grid.yaml`:

```yaml
suites:
  my_experiment:
    in_dim: [8]
    out_dim: [8]
    basis_type: [jacobi, bspline]
    poly_family: [chebyshev_t, legendre]
    degree: [3, 5, 8]
    L: [32, 64, 128]
    segments: [8]
    interp: [linear]
    scheme: [uint8_asymm]
```

Then run:
```bash
python scripts/all.py --suite my_experiment --targets all --jobs 4
```

## Step-by-Step Usage

If you prefer manual control:

```bash
# 1. Generate firmware variants
python scripts/gen_cases.py --suite smoke --targets esp32

# 2. Build with PlatformIO
python scripts/build_cases.py --targets esp32 --jobs 4

# 3. Simulate on Wokwi
python scripts/run_wokwi.py --targets esp32 --timeout-ms 30000 --jobs 4

# 4. Collect results
python scripts/collect_results.py

# 5. Visualize (optional)
python scripts/plot_results.py
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
