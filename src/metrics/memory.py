# src/metrics/memory.py
from __future__ import annotations

from typing import Dict

from src.quant.lut_builder import LUTArtifact


def lut_memory_report(art: LUTArtifact, breakdown: bool = True) -> Dict[str, object]:
    knots_b = int(art.knots.nbytes)
    q_b = int(art.q_table.nbytes)
    s_b = int(art.scale.nbytes)
    y_b = int(art.y_min.nbytes)
    total = knots_b + q_b + s_b + y_b

    if not breakdown:
        return {"lut_total_bytes": total}

    return {
        "lut_total_bytes": total,
        "breakdown": {
            "knots_bytes": knots_b,
            "q_table_bytes": q_b,
            "scale_bytes": s_b,
            "y_min_bytes": y_b,
        },
    }
