import numpy as np
import rasterio
from scipy import ndimage

def cloud_mask(file_L1C: str, buffer_size: int = 8):
    """Cloud mask for a *multi-band GeoTIFF* exported from Google Earth Engine (TOA reflectance).

    Expects band descriptions to include at least: B4 (red), B3 (green), B8A (NIR).
    The input is typically uint16 reflectance scaled by 1e4 (so we multiply by 1e-4).
    """
    from omnicloudmask import predict_from_array

    print('Making cloud mask (omnicloudmask)...')

    R, G, NIR = 'B4', 'B3', 'B8A'

    with rasterio.open(file_L1C) as src:
        bandnames = list(src.descriptions) if src.descriptions is not None else []
        if (not bandnames) or any(b is None for b in bandnames):
            raise ValueError(
                "Input GeoTIFF has missing band descriptions. "
                "Export from GEE with band names preserved (descriptions like 'B4', 'B3', 'B8A')."
            )

        try:
            red_index = bandnames.index(R)
            green_index = bandnames.index(G)
            nir_index = bandnames.index(NIR)
        except ValueError as e:
            raise ValueError(
                f"Required bands not found in descriptions. Need {R}, {G}, {NIR}. Found: {bandnames}"
            ) from e

        data = src.read([red_index + 1, green_index + 1, nir_index + 1]).astype('float32') * 1e-4

        nodata_val = src.nodata
        if nodata_val is None:
            nodata_mask = (data[0] <= 0)
        else:
            nodata_mask = (data[0] == nodata_val) | (data[0] <= 0)

        data[:, nodata_mask] = np.nan

        # Fill NaNs with nearest values (predict_from_array can misbehave with NaNs).
        if np.any(np.isnan(data)):
            mask = np.isnan(data)
            nearest_idx = ndimage.distance_transform_edt(
                mask, return_distances=False, return_indices=True
            )
            data = data[tuple(nearest_idx)]

        cld_msk = predict_from_array(data)[0]

    cld_msk = (cld_msk != 0).astype('uint8')

    struct1 = ndimage.generate_binary_structure(2, 1)
    mask_cloud_buffered = ndimage.binary_dilation(
        cld_msk, structure=struct1, iterations=int(buffer_size)
    ).astype('uint8')

    print('Done')
    return mask_cloud_buffered
