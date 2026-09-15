from typing import Literal

from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    x1: float
    y1: float
    x2: float
    y2: float


class PredictionResponse(BaseModel):
    request_id: str
    model_version: str
    detections: list[Detection]
    safety_status: Literal["unknown", "review"]
    latency_ms: float = Field(ge=0)


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
