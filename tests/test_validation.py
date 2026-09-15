import pytest

from safetyvision.validation import InputValidationError, validate_upload


def test_rejects_unknown_extension():
    with pytest.raises(InputValidationError):
        validate_upload("payload.exe", b"abc")


def test_rejects_empty_payload():
    with pytest.raises(InputValidationError):
        validate_upload("frame.jpg", b"")
