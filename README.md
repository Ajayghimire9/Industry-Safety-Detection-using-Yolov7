# SafetyVision

Object-detection serving and review routing.

SafetyVision separates an exported object detector from the HTTP service around it. The service validates uploads, applies a documented detection-output contract and returns findings for review.

## Run locally

Use Python 3.11 or newer in a virtual environment.

```bash
pip install -e ".[dev]"
# Set SAFETYVISION_WEIGHTS to a compatible TorchScript export.
uvicorn safetyvision.api:app --host 127.0.0.1 --port 8000
```

## Design decisions

The export must return decoded Nx6 rows: x1, y1, x2, y2, confidence and class ID. A single batch dimension is accepted.

Postprocessing validates finite coordinates and class IDs, applies a confidence threshold and runs class-aware non-maximum suppression.

No detections means unknown, not proof that a scene is safe. Detected objects produce review status; class-specific safety policy is intentionally separate.

Input reads are bounded, readiness requires loaded weights, and Prometheus exposes serving telemetry.

## Technology

Python, PyTorch/TorchScript, OpenCV, FastAPI, Prometheus, Docker, Kubernetes configuration.

## Validation

Run `python -m pytest tests -q` from the repository root. CI runs the maintained test suite and lint checks. Tests use local fixtures or mocks and do not deploy cloud resources.

## Scope and limitations

Raw YOLO output is not automatically compatible with this adapter: resize, letterboxing and decoding must agree with the exported model. The vendored yolov7 directory retains its upstream license and code. The service is not a certified workplace-safety system. Legacy AWS deployment requires manual dispatch.
