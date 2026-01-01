# src/utils/parse_config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from pydantic import ValidationError

from src.schemas.config import RootConfig
from src.utils.config import ConfigError, load_config


class ConfigValidationError(RuntimeError):
    pass


def load_and_validate(config_path: str | Path) -> Tuple[RootConfig, Dict[str, Any]]:
    raw = load_config(config_path)
    try:
        cfg = RootConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigValidationError(str(e)) from e
    return cfg, raw
