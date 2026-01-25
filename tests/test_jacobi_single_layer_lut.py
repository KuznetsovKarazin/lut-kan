import json
from pathlib import Path

import pytest

from src.experiments.runner import run  


TEST_CFG = Path("configs/exp_jacobi_lut.yaml")


def _load_results(out_dir: Path) -> dict:
    p = out_dir / "results.json"
    assert p.exists(), f"results.json not found in {out_dir}"
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.smoke
def test_jacobi_experiment_runs_and_has_contract_keys():
    out_dir = run(TEST_CFG)
    r = _load_results(Path(out_dir))

    assert r["float_backend"] == "jacobi"
    assert "experiment" in r
    assert "converter_enabled" in r
    assert "run_params" in r

    assert "output_sanity" in r
    assert "phi_error_summary" in r
    assert "memory" in r


@pytest.mark.numeric
def test_jacobi_lut_error_is_bounded():
    out_dir = run(TEST_CFG)
    r = _load_results(Path(out_dir))

    out = r["output_sanity"]

    assert out["rmse"] <= 1e-2
    assert out["max_abs"] <= 5e-2


@pytest.mark.regression
def test_jacobi_memory_budget_is_stable():
    out_dir = run(TEST_CFG)
    r = _load_results(Path(out_dir))

    lut_bytes = r["memory"]["lut"]["lut_total_bytes"]
    assert lut_bytes > 0
    assert lut_bytes < 10_000_000  
