import os
import numpy as np
import rasterio as rio

import matplotlib.pyplot as plt
from skimage import exposure
from skimage.filters import threshold_otsu
from scipy.signal import find_peaks

from models.utility import dialate_mask


def display_rgb(r, g, b):
    rgb = np.moveaxis(np.asarray([r, g, b]), 0, 2)
    # print(, )
    rgb = np.astype((rgb - np.nanmin(rgb)) / (np.nanmax(rgb) - np.nanmin(rgb)) * 255, np.uint8)
    p2, p98 = np.percentile(rgb, (2, 98))  # ignore outliers
    rgb_stretched = exposure.rescale_intensity(rgb, in_range=(p2, p98))
    # print(rgb)
    return rgb_stretched


def find_ndwi_threshold(
        ndwi_valid, bin_centers, counts,
        prominence_scale=0.01, distance=10,
        max_recursion=5
):
    """
    Automatically find a threshold between two major peaks in an NDWI histogram.
    Returns:
        threshold (float): min point between iz and water peaks
        otsu (float): Otsu threshold within [low, high]
        sorted_peaks (np.ndarray): detected peak indices
    """

    bin_width = bin_centers[1] - bin_centers[0]
    prominence = prominence_scale * np.max(counts)

    # --- Find peaks ---
    peaks, props = find_peaks(counts, prominence=prominence, distance=distance, width=1)
    if len(peaks) < 2:
        # Recursively relax prominence if too few peaks
        if max_recursion <= 0:
            # print(peaks[0])
            if bin_centers[peaks[0]] > 0.2:
                otsu = otsu_2d(ndwi_valid)
                if otsu > 0.2:
                    return 0, 0, peaks
                return otsu, otsu, peaks
            elif bin_centers[peaks[0]] < -0.15:
                otsu = otsu_2d(ndwi_valid)
                if otsu < -0.15:
                    return 0, 0, peaks
                return otsu, otsu, peaks
            else:
                return -1, -1, peaks

        return find_ndwi_threshold(
            ndwi_valid, bin_centers, counts,
            prominence_scale=prominence_scale * 0.5,
            distance=distance,
            max_recursion=max_recursion - 1
        )

    peaks = np.asarray(sorted(peaks, key=lambda p: bin_centers[p]))
    widths = props["widths"]

    # print(peaks, [bin_centers[p] for p in peaks])

    # --- Estimate range [low, high] ---
    low = bin_centers[0]
    high = bin_centers[-1]

    # --- Identify land (<-0.15), interface (~[-0.15, 0]), and water (>0) peaks ---
    iz_p, water_p, land_p = None, None, None
    iz_width = water_width = land_width = None

    for p, w in zip(peaks, widths):
        val = bin_centers[p]

        if val < -0.15:
            land_p, land_width = p, w
        elif -0.15 <= val <= 0.05:
            # print(val, high)
            # iz_p, iz_width = p, w
            if iz_p is None:
                iz_p, iz_width = p, w
            # elif val<=0:
            elif val < bin_centers[peaks[-1]]:
                iz_p, iz_width = p, w
            else:
                water_p, water_width = p, w

        elif val > 0.05 and water_p is None:
            water_p, water_width = p, w
            break  # we found water, stop

    # print(land_p, iz_p, water_p)

    if land_p is not None:
        low = bin_centers[land_p] - 0.5 * bin_width * land_width
    if iz_p is not None:
        low = bin_centers[iz_p] - 0.5 * bin_width * iz_width

    if water_p is not None:
        high = bin_centers[water_p] + 0.5 * bin_width * water_width
    elif iz_p is not None:
        if high > 0.2:
            high = 0.2  # fallback if only iz found

    # --- Handle missing peaks gracefully ---
    if iz_p is None:
        if (land_p is not None):
            if water_p is not None:
                otsu = otsu_2d(ndwi_valid)
                if otsu < 0.2 and otsu > -0.15:
                    return otsu, otsu, peaks
                iz_p = land_p
            else:
                iz_p = land_p
        else:
            iz_p = np.abs(bin_centers + 0.15).argmin()

    if water_p is None:
        # fallback thresholding if no clear water peak
        mask = (ndwi_valid > low) & (ndwi_valid < high)
        otsu = otsu_2d(ndwi_valid[mask])
        return otsu, otsu, peaks

    # print(iz_p, water_p)

    # --- Find minimum between iz and water peaks ---
    start, end = sorted([iz_p, water_p])
    idx_min = np.argmin(counts[start:end + 1]) + start
    threshold = bin_centers[idx_min]

    # --- Otsu threshold within [low, high] range ---
    mask = (ndwi_valid > low) & (ndwi_valid < high)
    otsu = otsu_2d(ndwi_valid[mask])

    return threshold, otsu, peaks


def otsu_2d(arr):
    """Compute Otsu threshold for one 2D NDWI slice."""
    arr = np.ravel(arr[~np.isnan(arr)])
    if arr.size == 0:
        return np.nan
    return threshold_otsu(arr)

def enhanced_otsu_watermask(file_l1: str, cloud_mask: np.ndarray,cldmask_buffer_size:int,save_dir:str) -> np.ndarray:
    '''
    water masking using enhanced otsu method, developed by Yan for the IZMAPPING project.
    Args:
        file_l1: l1c geotiff image
        cloud_mask:
        cldmask_buffer_size: buffer size used in cloud mask generation
    Returns:
    '''
    basename = os.path.basename(file_l1).replace('.tif', '')
    _clear_mask = cloud_mask == 0
    with rio.open(file_l1) as src:
        profile = src.profile
        bandnames = src.descriptions
        count = src.count
        blue_index = bandnames.index('B2')
        green_index = bandnames.index('B3')
        red_index = bandnames.index('B4')
        nir_index = bandnames.index('B8')
        swir_index = bandnames.index('B11')

        data = src.read([green_index + 1, nir_index + 1, swir_index + 1]) * 1e-4
        blue = src.read([blue_index + 1]) * 1e-4
        red = src.read([red_index + 1]) * 1e-4

        nodata_mask = (data[0] == src.nodata) | (data[0] <= 0)

    clear_mask = _clear_mask & (~nodata_mask)
    ndwi = np.full_like(nodata_mask, np.nan, dtype=np.float32)
    ndwi[clear_mask] = (data[0][clear_mask] - data[1][clear_mask]) / (data[0][clear_mask] + data[1][clear_mask])

    ndti = np.full_like(nodata_mask, np.nan, dtype=np.float32)
    ndti[clear_mask] = (data[0][clear_mask] - red[0][clear_mask]) / (data[0][clear_mask] + red[0][clear_mask])

    br_ratio = np.full_like(nodata_mask, np.nan, dtype=np.float32)
    br_ratio[clear_mask] = blue[0][clear_mask] / red[0][clear_mask]

    # ndsi = np.full_like(nodata_mask, np.nan, dtype=np.float32)
    # ndsi[clear_mask] = (data[0][clear_mask]- data[2][clear_mask]) / (data[0][clear_mask]+ data[2][clear_mask])

    ndwi_valid = ndwi[~np.isnan(ndwi)]
    br_ratio_valid = br_ratio[(~np.isnan(br_ratio)) & (ndwi > -0.15)]
    ndti_valid = ndti[(~np.isnan(ndti)) & (ndwi > -0.15)]

    counts_ndwi, bin_edges_ndwi = np.histogram(ndwi_valid, bins=100)
    bin_centers_ndwi = (bin_edges_ndwi[:-1] + bin_edges_ndwi[1:]) / 2
    counts_ndwi = (counts_ndwi - counts_ndwi.min()) / (counts_ndwi.max() - counts_ndwi.min())

    ### ndvi
    # counts_ndvi, bin_edges_ndvi = np.histogram(ndvi_valid, bins=100)
    # bin_centers_ndvi = (bin_edges_ndvi[:-1] + bin_edges_ndvi[1:]) / 2
    # counts_ndvi = (counts_ndvi - counts_ndvi.min()) / (counts_ndvi.max() - counts_ndvi.min())

    ###
    counts_br, bin_edges_br = np.histogram(br_ratio_valid, bins=100)
    bin_centers_br = (bin_edges_br[:-1] + bin_edges_br[1:]) / 2
    counts_br = (counts_br - counts_br.min()) / (counts_br.max() - counts_br.min())

    counts_ndti, bin_edges_ndti = np.histogram(ndti_valid, bins=100)
    bin_centers_ndti = (bin_edges_ndti[:-1] + bin_edges_ndti[1:]) / 2
    counts_ndti = (counts_ndti - counts_ndti.min()) / (counts_ndti.max() - counts_ndti.min())

    rgb = display_rgb(r=data[1], g=data[0], b=blue[0])

    otsu = None
    threashold = None

    otsu_overall_ndwi = otsu_2d(ndwi_valid)

    try:
        threashold, otsu, peaks = find_ndwi_threshold(ndwi_valid, bin_centers_ndwi, counts_ndwi, prominence_scale=0.005,
                                                      distance=5, max_recursion=3)

        # threashold_ndvi, otsu_ndvi, peaks_ndvi = find_ndwi_threshold(ndvi_valid, bin_centers_ndvi, counts_ndvi, prominence_scale=0.01, distance=5, max_recursion=5)

        pstr = ','.join([str(round(bin_centers_ndwi[i], 3)) for i in peaks])
    except Exception as e:
        return None

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
    fig.suptitle(f'{basename}')
    axes = axes.flatten()

    # threashold = 0.123
    water_mask = (ndwi > threashold).astype(np.uint8) + 1
    # water_mask[(br_ratio<0.92) & (ndwi>-0.05) & (data[2]<0.05)] = 2
    water_mask[(br_ratio < 0.92) & (ndti < 0) & (ndwi > -0.05) & (data[2] < 0.05)] = 2
    water_mask[~clear_mask] = 3  ## cloud
    water_mask[nodata_mask] = 0  ## nodata

    water_mask_overall = (ndwi > otsu_overall_ndwi).astype(np.uint8) + 1
    water_mask_overall[~clear_mask] = 3
    water_mask_overall[nodata_mask] = 0

    # water_mask_ndvi = (ndvi > threashold)

    axes[0].plot(bin_centers_ndwi, counts_ndwi, label='NDWI')
    # axes[0].plot(bin_centers_ndvi,counts_ndvi, label='NDVI')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Pixel Counts')
    axes[0].set_title('NDWI Histogram')

    for i, p in enumerate(peaks):
        x = bin_centers_ndwi[p]
        y = counts_ndwi[p]
        axes[0].scatter([x], [y], marker='*', s=100, label=f'peak {i}:{x:.3f}')
    axes[0].axvline(x=otsu, color='r', linestyle='--', linewidth=2, label=f'otsu = {otsu:.3f}')
    if otsu != otsu_overall_ndwi:
        axes[0].axvline(x=otsu_overall_ndwi, color='b', linestyle='--', linewidth=2,
                        label=f'otsu_overall = {otsu_overall_ndwi:.3f}')
    if threashold is not None:
        axes[0].axvline(x=threashold, color='g', linestyle='--', linewidth=2, label=f'min = {threashold:.3f}')

    axes[0].legend()
    axes[1].imshow(rgb)
    axes[1].set_title('NIRGB')
    axes[2].imshow(ndwi)
    axes[2].set_title('NDWI')
    axes[3].imshow(br_ratio)
    axes[3].set_title('Blue/Red')
    # axes[2].plot(bin_centers_ndvi,counts_ndvi, label='Histogram')
    # axes[2].set_xlabel('NDVI')
    # axes[2].set_ylabel('Pixel Counts')
    axes[4].imshow(water_mask_overall)
    # axes[4].plot(bin_centers_br,counts_br, label='BR Ratio')
    axes[4].set_title('otsu overall')
    axes[5].imshow(water_mask)
    axes[5].set_title('new')

    axes[6].plot(bin_centers_br, counts_br, label='B/R')
    axes[6].set_title('B/R Histogram')
    axes[6].set_xlabel('Value')
    axes[6].set_ylabel('Pixel Counts')

    axes[7].imshow(ndti)
    axes[7].set_title('NDTI')
    axes[8].plot(bin_centers_ndti, counts_ndti, label='ndti')
    axes[8].set_title('NDTI Histogram')
    axes[8].set_xlabel('Value')
    axes[8].set_ylabel('Pixel Counts')

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(str(save_dir / basename.replace('.tif', '_watermsk_hist.png')))
    # plt.close()

    profile_c = profile.copy()
    profile_c.update({'dtype': rio.uint8, 'nodata': 0, 'count': 1})

    o_f = str(save_dir / basename.replace('.tif', '_watermsk.tif'))
    with rio.open(o_f, 'w', **profile_c) as dst:
        descriptions = f'otsu:{threashold:.3f},cldmask_buffer_size:{cldmask_buffer_size}'
        dst.update_tags(info=descriptions)
        dst.write(water_mask, 1)

    plt.close()
    return water_mask

def update_watermask_with_cloudbuffer(file_watermsk: str, buffer_size: int) -> np.ndarray:
    '''
    dilate cloud mask in watermask by buffer_size
    Args:
        file_watermsk: watermask file path
        buffer_size: dilation size
    Returns:
        updated watermask
    '''
    with rio.open(file_watermsk) as src:
        watermask = src.read(1)  # 1=land, 2=water, 3=cloud
        tags = src.tags()
        nodata = src.nodata

    cld_buffer_size = 0
    if tags['info'].find('cldmask_buffer_size') > -1:
        cld_buffer_size = int(tags['info'].split('cldmask_buffer_size:')[1].split(',')[0])
    buffer_size = max(0, buffer_size - cld_buffer_size)
    cldmask = (watermask == 3)
    buffered = dialate_mask(cldmask, int(buffer_size))
    watermask[buffered & (watermask != nodata)] = 3  ## update cloud mask in watermask
    return buffered, watermask

def waterglint_mask(file_l1: str, water_mask: np.ndarray):
    '''
    add glint mask to water mask
    glint = 4
    Args:
        file_l1: l1c geotif
        water_mask: water_mask (1=land, 2=water, 3=cloud)
    Returns:
    '''
    with rio.open(file_l1) as src:
        bandnames = src.descriptions
        swir_index = bandnames.index('B11')
        data = src.read([swir_index + 1])
    water_mask[water_mask == 2 & (data[0]>1500)] = 4
    return water_mask