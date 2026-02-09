# MCU Auto-Benchmark (Wokwi + PlatformIO)

Automated MCU benchmark pipeline for LUT-KAN supporting **both Jacobi polynomials and B-splines**:

1. **Generate** many firmware variants (different LUT parameters, basis types)
2. **Compile** per MCU target with PlatformIO
3. **Run** headless simulations on Wokwi
4. **Collect** JSON outputs into CSV/Markdown reports

## Supported MCU targets

All targets have been verified against Wokwi's supported hardware list:

| Target ID    | Board                    | Architecture     | FPU    | Wokwi Part Type                |
|--------------|--------------------------|------------------|--------|--------------------------------|
| `uno`        | Arduino UNO              | AVR ATmega328P   | None   | `wokwi-arduino-uno`           |
| `nano`       | Arduino Nano             | AVR ATmega328P   | None   | `wokwi-arduino-nano`          |
| `mega`       | Arduino Mega 2560        | AVR ATmega2560   | None   | `wokwi-arduino-mega`          |
| `pico`       | Raspberry Pi Pico        | ARM Cortex-M0+   | None   | `wokwi-pi-pico`               |
| `esp32`      | ESP32 DevKit V1          | Xtensa LX6       | Yes    | `wokwi-esp32-devkit-v1`       |
| `esp32c3`    | ESP32-C3 DevKitM-1       | RISC-V           | None   | `board-esp32-c3-devkitm-1`    |
| `esp32s3`    | ESP32-S3 DevKitC-1       | Xtensa LX7       | Yes    | `board-esp32-s3-devkitc-1`    |
| `stm32f103`  | Blue Pill (STM32F103C8)  | ARM Cortex-M3    | None   | `board-stm32-bluepill`        |

> **Note:** ESP8266 and STM32F401 (Nucleo) are **not supported** by Wokwi and have been removed.

## Supported basis types

| Basis      | Polynomial families              | Float baseline       | LUT forward |
|------------|----------------------------------|----------------------|-------------|
| `jacobi`   | Chebyshev T, Legendre, Gegenbauer, general Jacobi(α,β) | 3-term recurrence | segment-wise quantized LUT |
| `bspline`  | Cubic B-splines (configurable degree) | Cox-de Boor recursion + SiLU base | segment-wise quantized LUT |

## One-time prerequisites

```bash
# 1. Python env (your existing LUT-KAN venv)
pip install pyyaml numpy

# 2. PlatformIO Core
pip install platformio
# or: https://docs.platformio.org/en/latest/core/installation.html

# 3. Wokwi CLI
# https://docs.wokwi.com/wokwi-ci/getting-started
npm install -g @anthropic-ai/wokwi-cli  # or download from Wokwi

# 4. Wokwi token (required for headless / CI runs)
export WOKWI_CLI_TOKEN="your-token-here"
```

## Quick start

From the **repo root**:

```bash
# Smoke test: 2 basis types × 2 poly families × 2 degrees × 2 LUT sizes × ...
# On all 8 targets — fast for CI
python mcu_auto/scripts/all.py --suite smoke --targets all --jobs 4

# Single target for quick iteration
python mcu_auto/scripts/all.py --suite smoke --targets esp32 --jobs 4
```

**Artifacts:**
```
mcu_auto/cases/<target>/<case_id>/     # generated PlatformIO projects
mcu_auto/logs/<target>/<case_id>.log   # Wokwi serial output
mcu_auto/reports/summary.csv           # machine-readable results
mcu_auto/reports/summary.md            # human-readable pivot tables
```

## Step-by-step usage

### 1. Generate cases

```bash
# Generate cases for specific targets and suite
python mcu_auto/scripts/gen_cases.py --suite full --targets esp32,stm32f103

# Custom grid file
python mcu_auto/scripts/gen_cases.py --suite smoke --grid path/to/custom_grid.yaml
```

Each case becomes a standalone PlatformIO project with:
- `platformio.ini` — build configuration
- `src/main.cpp` — benchmark firmware (shared template)
- `include/case_layer.h` — generated LUT data + config defines
- `diagram.json` + `wokwi.toml` — Wokwi simulator config
- `meta.json` — full case specification (for reproducibility)

### 2. Build firmware

```bash
python mcu_auto/scripts/build_cases.py --targets esp32,stm32f103 --jobs 4
```

This runs `pio run` for each case and records Flash/RAM usage in `build_metrics.json`.

### 3. Run simulations

```bash
python mcu_auto/scripts/run_wokwi.py --targets esp32,stm32f103 --timeout-ms 30000 --jobs 4
```

### 4. Collect results

```bash
python mcu_auto/scripts/collect_results.py
```

Merges three data sources per case:
- `meta.json` — configuration parameters
- `build_metrics.json` — Flash/RAM from PlatformIO
- Wokwi logs — timing measurements and accuracy

## Custom sweeps

Edit `mcu_auto/grids/grid.yaml`:

```yaml
suites:
  my_sweep:
    in_dim: [8]
    out_dim: [8]
    basis_type: [jacobi, bspline]    # test both basis types
    # Jacobi parameters
    poly_family: [chebyshev_t, legendre]
    degree: [3, 5, 8]
    # B-spline parameters
    bspline_degree: [3]
    grid_points: [5, 8]
    # LUT parameters (shared)
    L: [32, 64, 128]
    segments: [8]
    interp: [linear]
    scheme: [uint8_asymm]
    input_mode: [linspace]
    iters: [300]
    repeats: [7]
```

Then run with `--suite my_sweep` (add the suite name to argparse choices first, or use gen_cases directly).

## Firmware protocol

The firmware prints exactly one JSON line prefixed with `LUTKAN:`:

```json
LUTKAN:{"target":"esp32","case_id":"...","basis_type":"jacobi","poly_family":"chebyshev_t",
  "degree":3,"in_dim":8,"out_dim":8,"L":64,"segments":8,"interp":"linear",
  "scheme":"uint8_asymm","t_float_us":12345,"t_lut_us":678,
  "speedup":18.2083,"max_abs_err":0.00123456}
```

The collector extracts the **last** such line per log file.

## Real hardware runs (optional)

Wokwi is great for automation, but for publication you should validate on real boards:

```bash
pip install pyserial

# Upload each case to a connected board and capture serial output
python mcu_auto/scripts/run_hardware.py --targets esp32 --port /dev/ttyUSB0 --upload

# Collect hardware results (same collector, different log directory)
python mcu_auto/scripts/collect_results.py --logs mcu_auto/hw_logs --reports mcu_auto/reports_hw
```

## Research-grade recommendations

1. Use `smoke` for CI, `full` for the paper, `stress` for nightly regression.
2. Record **(Flash, RAM)** alongside timing — `build_cases.py` captures this automatically.
3. Report **median** across `repeats` with min/max (firmware does this). Robust to simulator jitter.
4. Use **two input regimes**: `linspace` (deterministic) and `rng_uniform` (stress different LUT regions).
5. Store raw logs and the exact `grid.yaml`. Treat `mcu_auto/cases/` as the immutable artifact set.
6. For B-spline vs Jacobi comparison, keep `segments`, `L`, `interp`, `scheme` identical.

## Architecture notes

- **Jacobi float baseline**: Standard 3-term Jacobi recurrence. Cost grows linearly with degree.
- **B-spline float baseline**: Cox-de Boor recursion with SiLU base function. Cost grows with `degree × num_coef`.
- **LUT forward** (shared): Segment-wise quantized lookup. Cost is O(edges × segments) regardless of original polynomial complexity.
- The massive speedup on no-FPU platforms (AVR, Cortex-M3, RISC-V) comes from replacing expensive float ops with integer table lookups.
