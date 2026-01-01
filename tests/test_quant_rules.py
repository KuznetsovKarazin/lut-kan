# tests/test_quant_rules.py
from __future__ import annotations

import pytest

from src.schemas.config import QuantConfig


def test_quant_uint8_rejects_symmetric() -> None:
    with pytest.raises(Exception):
        QuantConfig(dtype="uint8", scheme="symmetric")


def test_quant_int8_symmetric_requires_zero_point_zero() -> None:
    with pytest.raises(Exception):
        QuantConfig(dtype="int8", scheme="symmetric")  # zero_point defaults to 'auto' -> must raise

    q = QuantConfig(dtype="int8", scheme="symmetric", zero_point=0, qmin=-127, qmax=127)
    assert q.zero_point == 0
    assert q.qmin == -127
    assert q.qmax == 127


def test_quant_invalid_range_raises() -> None:
    with pytest.raises(Exception):
        QuantConfig(dtype="int8", scheme="symmetric", zero_point=0, qmin=-128, qmax=127)
