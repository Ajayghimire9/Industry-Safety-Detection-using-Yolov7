from __future__ import annotations

import os
import time
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import make_asgi_app

from .config import Settings
from .contracts import HealthResponse, PredictionResponse
from .inference import Detector
from .monitoring import ERRORS, LATENCY, REQUESTS, DETECTIONS
from .validation import InputValidationError, validate_upload

settings = Settings()
weights = os.getenv("SAFETYVISION_WEIGHTS", "artifacts/model.ts")
detector = None
try:
    detector = Detector(weights, settings)
except (FileNotFoundError, RuntimeError):
    detector = None

app = FastAPI(title="SafetyVision Inference Gateway", version="2.0.0")
app.mount("/metrics", make_asgi_app())


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok" if detector else "degraded", model_version=settings.model_version, model_loaded=detector is not None)


@app.get("/ready")
def ready():
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"ready": True}


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    REQUESTS.inc()
    payload = await file.read()
    try:
        validate_upload(file.filename or "image.jpg", payload, settings)
        if detector is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        detections = detector.predict(payload)
    except InputValidationError as exc:
        ERRORS.inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        ERRORS.inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    for item in detections:
        DETECTIONS.labels(item.label).inc()
    elapsed = time.perf_counter() - started
    LATENCY.observe(elapsed)
    return PredictionResponse(request_id=request_id, model_version=settings.model_version, detections=detections, safety_status="warning" if detections else "clear", latency_ms=elapsed * 1000)
