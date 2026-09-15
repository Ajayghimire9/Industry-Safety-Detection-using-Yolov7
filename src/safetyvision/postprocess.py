"""Class-aware NMS for exported models returning [x1,y1,x2,y2,score,class]."""

import numpy as np


def filter_detections(rows, confidence=0.35, iou_threshold=0.45):
    a = np.asarray(rows, dtype=float)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.size == 0:
        return []
    if a.ndim != 2 or a.shape[1] != 6 or not np.isfinite(a).all():
        raise ValueError("Export must return finite Nx6 post-decoded detections")
    if ((a[:, 4] < 0) | (a[:, 4] > 1) | (a[:, 5] < 0) | (a[:, 5] != np.floor(a[:, 5]))).any():
        raise ValueError("Invalid confidence or class identifier")
    a = a[(a[:, 4] >= confidence) & (a[:, 2] > a[:, 0]) & (a[:, 3] > a[:, 1])]
    a = a[np.argsort(-a[:, 4], kind="stable")]
    keep = []
    while len(a):
        best, rest = a[0], a[1:]
        keep.append(best.tolist())
        left = np.maximum(best[:2], rest[:, :2])
        right = np.minimum(best[2:4], rest[:, 2:4])
        intersection = np.prod(np.maximum(right - left, 0), axis=1)
        area = (best[2] - best[0]) * (best[3] - best[1])
        other = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        iou = intersection / np.maximum(area + other - intersection, 1e-12)
        a = rest[(rest[:, 5] != best[5]) | (iou <= iou_threshold)]
    return keep
