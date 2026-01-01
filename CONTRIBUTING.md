# Contributing Guidelines

We welcome contributions that improve numerical correctness, reproducibility, and clarity of experimental reporting.

## Scope
Relevant contributions include:
- performance optimizations that preserve numerical fidelity,
- additional downstream benchmarks (symbolic regression, Feynman-like tasks),
- improved aggregation/reporting utilities,
- documentation improvements and reproducible examples.

## Development Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
   - `pip install -r requirements-dev.txt`

## Code Quality
- Please add or update unit tests when modifying numerical code paths.
- Ensure `pytest -q` passes prior to submitting a pull request.

## Reporting Performance Changes
When submitting performance-related changes, report:
- the exact hardware used,
- measured speed in ms/iter and speedups,
- any changes in numerical error metrics (MAE, max_abs).
