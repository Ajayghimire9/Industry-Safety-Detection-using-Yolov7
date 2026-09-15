# SafetyVision — Industrial Computer Vision MLOps Platform

SafetyVision upgrades the original YOLOv7 safety-detection project into an end-to-end ML engineering platform for industrial image inference.

## Architecture

```text
Images / annotated dataset
          |
          v
Data validation + dataset versioning
          |
          v
YOLOv7 training pipeline
          |
          +--> MLflow experiments
          |       |
          |       v
          |   evaluation gate
          |       |
          |       v
          |   model manifest + SHA-256
          |
          v
   TorchScript export
          |
          v
 FastAPI inference gateway
      |         |
      v         v
Prometheus   request audit
      |
      v
drift / quality monitoring
      |
      v
scheduled retraining

GitHub Actions -> quality -> container build
Docker -> Kubernetes -> HPA
```

## What is implemented

### Computer vision
- YOLOv7-compatible training workflow
- TorchScript inference adapter
- Confidence and IoU configuration
- Structured bounding-box responses
- Clear/warning inference status

### MLOps
- MLflow experiment tracking
- Reproducible training command orchestration
- Evaluation/deployment gate
- Model manifest with SHA-256 integrity hash
- Candidate lifecycle metadata
- Scheduled Airflow retraining DAG
- Image-distribution drift detection using PSI

### Serving
- FastAPI `/health`, `/ready`, `/v1/predict`
- Strict upload validation
- Request IDs
- Prometheus request, error, latency and detection metrics
- Lazy model readiness handling

### DevOps
- Installable `src/` Python package
- Ruff + Pytest GitHub Actions
- Container build workflow
- Kubernetes Deployment + Service
- Readiness/liveness probes
- CPU/memory resource boundaries
- Horizontal Pod Autoscaler
- Non-privileged container design target

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
ruff check src tests
pytest -q
```

Run the API when a TorchScript artifact exists at `artifacts/model.ts`:

```bash
uvicorn safetyvision.api:app --host 0.0.0.0 --port 8000
```

Metrics:

```text
http://localhost:8000/metrics
```

## Training

The repository keeps model training decoupled from serving. A YOLOv7 training command can be executed under MLflow:

```bash
python -m safetyvision.train \
  --command 'python yolov7/train.py --data data/safety.yaml --weights weights/yolov7.pt --epochs 50'
```

After validation, the exported model can be represented by a manifest containing the artifact checksum and evaluation metrics.

## Production deployment

Kubernetes manifests live under `k8s/`. Prometheus configuration is under `monitoring/`. The CI pipeline intentionally builds and tests the application without requiring cloud credentials in source control.

## Repository structure

```text
src/safetyvision/
  api.py
  config.py
  contracts.py
  drift.py
  inference.py
  monitoring.py
  registry.py
  train.py
  validation.py

airflow/dags/retrain.py
scripts/evaluate_model.py
k8s/deployment.yaml
k8s/hpa.yaml
monitoring/prometheus.yml
tests/
.github/workflows/
```

## Honest boundary

This repository demonstrates the engineering lifecycle around an object-detection model. It does not claim that the model is production-validated for workplace compliance, and no performance number is fabricated. Real deployment requires a verified labelled dataset, class definitions, validation metrics, calibration, security controls, and operational approval.
