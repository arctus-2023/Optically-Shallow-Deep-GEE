from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Any

import numpy as np


@dataclass
class RasterInput:
    """Normalized raster input used by the pipeline.

    image: HxWxC numeric array (typically int16/uint16 or float32)
    profile: rasterio profile-like dict (crs/transform/width/height/etc.)
    band_names: list of band identifiers aligned with image channels
    nodata: nodata value if known
    metadata: any extra info (source-specific)
    """

    image: np.ndarray
    profile: Dict[str, Any]
    band_names: List[str]
    nodata: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseReader(Protocol):
    """Reader interface. Implementations convert a file into RasterInput."""

    def read(self, path: str) -> RasterInput:
        ...
