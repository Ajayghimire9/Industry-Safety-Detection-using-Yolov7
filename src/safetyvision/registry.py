from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(weights: str | Path, metrics: dict, output: str | Path) -> None:
    payload = {
        "model_version": "yolov7-safety-2.0.0",
        "artifact": str(weights),
        "sha256": sha256(weights),
        "metrics": metrics,
        "lifecycle": "candidate",
    }
    Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
