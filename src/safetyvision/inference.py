from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .config import Settings
from .contracts import Detection


class Detector:
    """TorchScript adapter. The training/export stack can remain independent of the API."""

    def __init__(self, weights: str | Path, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.model = torch.jit.load(str(weights), map_location="cpu").eval()

    @torch.inference_mode()
    def predict(self, payload: bytes) -> list[Detection]:
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image payload")
        tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        output = self.model(tensor)
        rows = output[0] if isinstance(output, (tuple, list)) else output
        detections: list[Detection] = []
        for row in rows.detach().cpu().numpy().tolist():
            if len(row) < 6 or row[4] < self.settings.confidence_threshold:
                continue
            detections.append(Detection(label=str(int(row[5])), confidence=float(row[4]), x1=row[0], y1=row[1], x2=row[2], y2=row[3]))
        return detections
