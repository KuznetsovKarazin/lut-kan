from __future__ import annotations

from src.schemas.config import RootConfig


def _minimal_raw() -> dict:
    return {
        "experiment": {"name": "test", "group": "test"},
        "dataset": {"name": "dummy", "path": "unused"},
        "float_model": {"backend": "dummy", "checkpoint": "unused", "adapter": "unused"},
    }


def test_legacy_calibration_key_migrates_to_evaluation_inputs() -> None:
    raw = _minimal_raw()
    raw["calibration"] = {
        "num_samples": 321,
        "batch_size": 17,
        "seed": 9,
        "inputs": {"distribution": "uniform", "x_min": -1.0, "x_max": 1.0},
    }

    cfg = RootConfig.model_validate(raw)

    assert cfg.evaluation_inputs.num_samples == 321
    assert cfg.evaluation_inputs.batch_size == 17
    assert cfg.evaluation_inputs.seed == 9
    assert cfg.evaluation_inputs.inputs.distribution == "uniform"


def test_new_evaluation_inputs_key_is_canonical() -> None:
    raw = _minimal_raw()
    raw["evaluation_inputs"] = {
        "num_samples": 123,
        "inputs": {"distribution": "normal", "mean": 0.25, "std": 0.5},
    }

    cfg = RootConfig.model_validate(raw)

    assert cfg.evaluation_inputs.num_samples == 123
    assert cfg.evaluation_inputs.inputs.distribution == "normal"
    assert cfg.evaluation_inputs.inputs.mean == 0.25
