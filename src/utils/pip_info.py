# src/utils/pip_info.py
from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Optional


def pip_freeze(max_lines: int = 5000) -> Dict[str, Any]:
    """
    Captures `python -m pip freeze` output for reproducibility.
    This may fail in restricted environments; failures are reported in the returned dict.
    """
    try:
        out = subprocess.check_output(
            ["python", "-m", "pip", "freeze"],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", errors="ignore")
        lines: List[str] = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... truncated ({len(lines)-max_lines} more lines)"]
        return {"ok": True, "lines": lines}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "lines": []}
