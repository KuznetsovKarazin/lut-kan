# src/metrics/model_memory.py
from __future__ import annotations

from typing import Any, Dict


def torch_model_memory_bytes(model: Any) -> Dict[str, int]:
    """
    Returns parameter/buffer memory in bytes.
    Works for torch.nn.Module-like objects.

    Keys:
      - params_bytes
      - buffers_bytes
      - total_bytes
      - params_count
      - buffers_count
    """
    try:
        import torch  # noqa: F401
    except Exception:
        raise ImportError("torch is required to measure torch model memory")

    params_bytes = 0
    buffers_bytes = 0
    params_count = 0
    buffers_count = 0

    for p in model.parameters():
        params_count += p.numel()
        params_bytes += p.numel() * p.element_size()

    for b in model.buffers():
        buffers_count += b.numel()
        buffers_bytes += b.numel() * b.element_size()

    return {
        "params_bytes": int(params_bytes),
        "buffers_bytes": int(buffers_bytes),
        "total_bytes": int(params_bytes + buffers_bytes),
        "params_count": int(params_count),
        "buffers_count": int(buffers_count),
    }
