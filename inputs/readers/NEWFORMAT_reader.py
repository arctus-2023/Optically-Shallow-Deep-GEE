from __future__ import annotations

from typing import List

import numpy as np
import rasterio

from .base_reader import RasterInput


class GeeToaGeotiffReader:
    """Reads a multiband GeoTIFF downloaded from Google Earth Engine (TOA reflectance).
    """

    def __init__(self, required_band_names: List[str] | None = None):
        self.required_band_names = required_band_names

    def read(self, path: str) -> RasterInput:
        with rasterio.open(path) as src:
            profile = src.profile.copy()
            band_names = list(src.descriptions) if src.descriptions else []

            if not band_names or all(b is None or str(b).strip() == '' for b in band_names):
                raise ValueError(
                    "Input GeoTIFF is missing band descriptions. "
                )

            # rasterio uses 1-based band indices
            data = src.read()  # (C, H, W)
            nodata = src.nodata

        # Convert to (H, W, C)
        image = np.transpose(data, (1, 2, 0))

        if self.required_band_names:
            missing = [b for b in self.required_band_names if b not in band_names]
            if missing:
                raise ValueError(f"Missing required bands: {missing}. Found: {band_names}")

        return RasterInput(image=image, profile=profile, band_names=[str(b) for b in band_names], nodata=nodata, metadata={"source": "gee_toa_geotiff"})
