from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_version: str = "yolov7-safety-2.0.0"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    max_upload_bytes: int = 15_000_000
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
