# Changelog

All notable repository-level changes are recorded here.

## [2.2.0] - 2026-08-15

### Fixed

- Aligned per-segment LUT sampling with runtime linear interpolation by using an endpoint-inclusive grid with `L` stored samples and `L-1` interpolation intervals.
- Added artifact metadata for the LUT sample-grid contract and warnings for legacy non-endpoint-inclusive artifacts.
- Fixed result aggregation when a specialized experiment contains an entirely empty metric family.

### Changed

- Separated LUT sampling geometry from runtime `closed` / `half_open` boundary semantics.
- Clarified that per-segment quantization ranges are derived from sampled LUT values rather than an external calibration dataset.
- Renamed evaluation-input configuration away from the misleading calibration terminology.
- Updated the NCA revision interpretation: corrected 8-bit experiments reach a quantization floor rather than supporting the previous empirical `O(1/L)` claim.

### Added

- Controlled NCA R1 sweep over `L={16,32,64,128}`, int8/uint8 quantization, and five seeds.
- Genuine unclipped OOB stress experiment.
- CICIDS2017 DoS-specific matched NumPy/Numba B-spline and LUT benchmarks.
- Downstream accuracy-latency-memory Pareto analysis.
- Compact machine-readable R1 reproducibility bundle under `reproducibility/ncaa_r1/`.

## [2.0.1]

Pre-R1 repository snapshot containing the MCU/Jacobi/fixed-point development line.

## [2.0.0]

Added Jacobi polynomial support, MCU benchmarking, and C export.
