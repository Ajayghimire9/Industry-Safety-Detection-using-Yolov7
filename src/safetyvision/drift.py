from __future__ import annotations

import numpy as np


def psi(reference, current, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    a, _ = np.histogram(reference, bins=edges)
    b, _ = np.histogram(current, bins=edges)
    a = np.clip(a / max(len(reference), 1), 1e-6, None)
    b = np.clip(b / max(len(current), 1), 1e-6, None)
    return float(np.sum((b - a) * np.log(b / a)))


def status(value: float) -> str:
    return "critical" if value >= 0.25 else "warning" if value >= 0.10 else "stable"
