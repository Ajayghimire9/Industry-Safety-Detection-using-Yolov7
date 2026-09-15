from __future__ import annotations

import numpy as np
from prometheus_client import Counter, Histogram

REQUESTS = Counter("safetyvision_inference_requests_total", "Inference requests")
ERRORS = Counter("safetyvision_inference_errors_total", "Inference errors")
LATENCY = Histogram("safetyvision_inference_latency_seconds", "Inference latency")
DETECTIONS = Counter("safetyvision_detections_total", "Detections by class", ["label"])


def detection_rate(reference, current) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    return float(np.mean(cur) - np.mean(ref))
