from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(weights: str, manifest: str) -> None:
    # The real detector evaluation is delegated to the YOLOv7 validation command.
    # This gate consumes its exported metrics file and records deployment eligibility.
    metrics_path = Path("artifacts/metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {"mAP50": 0.0, "mAP50_95": 0.0}
    payload = {
        "weights": weights,
        "metrics": metrics,
        "deployment_eligible": float(metrics.get("mAP50", 0)) >= 0.50,
    }
    Path(manifest).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    evaluate(args.weights, args.manifest)
