# src/utils/pykan_info.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import importlib
import importlib.metadata
import subprocess


@dataclass(frozen=True)
class PyKANInfo:
    present: bool
    module_name: str
    version: Optional[str]
    file: Optional[str]
    git_hash: Optional[str]


def _git_hash_from_path(p: Path) -> Optional[str]:
    try:
        # Walk up a few levels to find a git repo
        cur = p
        for _ in range(6):
            if (cur / ".git").exists():
                out = subprocess.check_output(["git", "-C", str(cur), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                return out.decode("utf-8", errors="ignore").strip() or None
            cur = cur.parent
    except Exception:
        return None
    return None


def get_pykan_info() -> Dict[str, Any]:
    """
    Best-effort metadata about the installed pykan (module name 'kan').
    Returns a dict suitable for JSON serialization.
    """
    module_name = "kan"
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return asdict(PyKANInfo(present=False, module_name=module_name, version=None, file=None, git_hash=None))

    file = getattr(mod, "__file__", None)
    version = None
    for dist_name in ("pykan", "kan"):
        try:
            version = importlib.metadata.version(dist_name)
            if version:
                break
        except Exception:
            continue
    if version is None:
        version = getattr(mod, "__version__", None)

    git_hash = None
    if file:
        git_hash = _git_hash_from_path(Path(file).resolve())

    return asdict(PyKANInfo(present=True, module_name=module_name, version=version, file=file, git_hash=git_hash))
