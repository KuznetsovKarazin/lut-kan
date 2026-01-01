from pathlib import Path
import yaml

from src.utils.config import load_config


def test_base_inheritance_resolves_relative(tmp_path: Path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"

    base.write_text("a:\n  b: 1\n  c: [1,2]\n", encoding="utf-8")
    child.write_text("_base_: base.yaml\na:\n  b: 2\n", encoding="utf-8")

    cfg = load_config(child)
    assert cfg["a"]["b"] == 2
    assert cfg["a"]["c"] == [1, 2]
    assert "_base_" not in cfg
