from __future__ import annotations

from pathlib import Path

from .config import Settings


class InputValidationError(ValueError):
    pass


def validate_upload(filename: str, payload: bytes, settings: Settings | None = None) -> None:
    settings = settings or Settings()
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise InputValidationError("Unsupported image format")
    if not payload:
        raise InputValidationError("Empty image payload")
    if len(payload) > settings.max_upload_bytes:
        raise InputValidationError("Image exceeds configured upload limit")
