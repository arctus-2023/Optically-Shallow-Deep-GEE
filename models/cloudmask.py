import numpy as np
import rasterio
from scipy import ndimage
from omnicloudmask import predict_from_array

from models.utility import dialate_mask


def omni_cloud_mask(file_l1: str, buffer_size: int = 8) -> np.ndarray:
    """Cloud mask for GEE-downloaded TOA GeoTIFF using omnicloudmask.
    Returns:
      2D boolean mask (True = cloud or cloud-buffer)
    """

    R, G, NIR = "B4", "B3", "B8A"

    with rasterio.open(file_l1) as src:
        bandnames = list(src.descriptions) if src.descriptions else []
        if not bandnames:
            raise ValueError(
                "GeoTIFF band descriptions are missing; expected names like 'B4','B3','B8A'."
            )

        try:
            red_index = bandnames.index(R)
            green_index = bandnames.index(G)
            nir_index = bandnames.index(NIR)
        except ValueError as e:
            raise ValueError(
                f"Required bands not found in descriptions. Need {R},{G},{NIR}. Found: {bandnames}"
            ) from e

        data = src.read([red_index + 1, green_index + 1, nir_index + 1]).astype(np.float32) * 1e-4

        # Nodata handling (GEE exports often set nodata to 0)
        nodata_val = src.nodata
        if nodata_val is None:
            nodata_mask = (data[0] <= 0)
        else:
            nodata_mask = (data[0] == nodata_val) | (data[0] <= 0)

        data[:, nodata_mask] = np.nan

        # Fill NaNs with nearest values (predict_from_array is sensitive to NaNs)
        if np.any(np.isnan(data)):
            mask = np.isnan(data)
            nearest_idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
            data = data[tuple(nearest_idx)]

        cld_msk = predict_from_array(data)[0]

    cld_msk = (cld_msk != 0)
    return dialate_mask(cld_msk, int(buffer_size))