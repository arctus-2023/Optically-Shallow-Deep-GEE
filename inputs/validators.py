from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .readers.base_reader import RasterInput


@dataclass
class ValidationResult:
    ok: bool
    messages: List[str]


def validate_required_bands(r: RasterInput, required: Sequence[str]) -> ValidationResult:
    """Validates the raster contains required band names."""
    missing = [b for b in required if b not in r.band_names]
    if missing:
        return ValidationResult(
            ok=False,
            messages=[
                f"Missing required band(s): {missing}",
                f"Available bands: {r.band_names}",
            ],
        )
    return ValidationResult(ok=True, messages=[])
