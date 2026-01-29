import numpy as np
from .readers.base_reader import RasterInput

def to_int16(r: RasterInput) -> RasterInput:
    """Legacy code expects int16/uint16 style arrays."""
    if r.image.dtype == np.int16:
        return r
    img = r.image
    if np.issubdtype(img.dtype, np.floating):
        # If float reflectance in [0,1], scale to legacy 1e4 integers.
        if np.nanmax(img) <= 1.0:
            img = (img * 1e4).round()
        img = img.astype(np.int16)
    else:
        img = img.astype(np.int16, copy=False)
    return RasterInput(
        image=img,
        profile=r.profile,
        band_names=list(r.band_names),
        nodata=r.nodata,
        metadata=r.metadata,
    )
