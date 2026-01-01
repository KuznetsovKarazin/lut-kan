# src/utils/config.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(RuntimeError):
    pass


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"YAML root must be a mapping: {path}")
        return data
    except FileNotFoundError as e:
        raise ConfigError(f"Config not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parse error in {path}: {e}") from e


def deep_merge(base: Any, override: Any) -> Any:
    """
    Deep-merge rules:
      - dict + dict -> recursive merge
      - list in override replaces list in base
      - scalar/None in override replaces base
    """
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            if k in out:
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return override


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Loads YAML config with _base_ inheritance.
    Relative _base_ is resolved against the directory of the current YAML file.
    """
    config_path = Path(config_path).expanduser()
    config_path = Path(os.path.expandvars(str(config_path))).resolve()

    cfg = load_yaml(config_path)

    if "_base_" in cfg:
        base_rel = cfg["_base_"]
        if not isinstance(base_rel, str) or not base_rel.strip():
            raise ConfigError(f"_base_ must be a non-empty string: {config_path}")

        base_path = (config_path.parent / base_rel).expanduser()
        base_path = Path(os.path.expandvars(str(base_path))).resolve()

        base_cfg = load_config(base_path)
        cfg = deep_merge(base_cfg, cfg)
        cfg.pop("_base_", None)

    return cfg
