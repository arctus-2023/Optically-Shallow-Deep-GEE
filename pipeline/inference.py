from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from inputs.normalizer import to_int16
# from pipeline.legacy.opticallyshallowdeep.parse_string import parse_string
from pipeline.strip_processing import run_legacy_strip_pipeline


import re
def parse_string(s, bandnames):
    if s.lower() in ["lat", "long", "lat_abs"]:
        return [s.lower()]#parses the column names for our model and makes into things that our scripts can interpret
    match = re.match(r'(B\d+)([a-zA-Z]?)-w_(\d+)(_?(\w+))?', s) #used to parse model_column names
    if match:
        groups = match.groups()
        first_group = f'{groups[0]}{groups[1]}' if groups[1] else groups[0]
        # first_group = first_group.replace('8a', '9')# Replace '8a' with '9'
        try:
            first_group_int = bandnames.index(first_group)
        except ValueError:
            first_group_int = first_group
        return [first_group_int, int(groups[2]), groups[4]] if groups[4] else [first_group_int, int(groups[2]), None]
    else:
        return None


@dataclass
class InferenceResult:
    rgb_img: np.ndarray
    output_path: str


def write_osw_geotiff(reference_profile, reference_transform, reference_crs, out_path: str, rgb_img: np.ndarray):
    """Write final OSW/ODW layer as single-band GeoTIFF, matching legacy output behavior."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    height, width, _ = rgb_img.shape
    out = rgb_img[:, :, 1].copy()
    mask = rgb_img[:, :, 2]
    out[mask == 0] = 255

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": out.dtype,
        "crs": reference_crs,
        "transform": reference_transform,
        "nodata": 255,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)



def run_toa_inference(
    raster_input,
    src_path: str,
    folder_out: str,
    water_mask: np.ndarray
) -> InferenceResult:
    """Run TOA inference on a GEE-exported multiband GeoTIFF."""
    # Normalize to legacy expectations
    # r = reorder_to_legacy_toa(to_int16(raster_input))
    r  = to_int16(raster_input)

    # Legacy model columns (kept identical to original project)
    model_columns = [
        "long", "lat_abs", "B2-w_15_sd", "B3-w_3_sd", "B3-w_7_avg", "B3-w_9_avg", "B3-w_11_sd",
        "B3-w_15_sd", "B4-w_5_avg", "B4-w_11_sd", "B4-w_13_avg", "B4-w_13_sd", "B4-w_15_sd",
        "B5-w_13_sd", "B5-w_15_sd", "B8-w_9_sd", "B8-w_13_sd", "B8-w_15_sd", "B11-w_9_avg",
        "B11-w_15_sd",
    ]
    selected_columns = [parse_string(s,r.band_names) for s in model_columns]

    model_path = str(Path(__file__).resolve().parent.parent / "resources" / "TOA.h5")

    # Run legacy strip pipeline. We set if_sr=False and file_in=src_path; legacy code
    # has been patched to handle GeoTIFF inputs without SAFE metadata.
    rgb = run_legacy_strip_pipeline(
        full_img=r.image,
        image_path=src_path,
        if_sr=False,
        model_path=model_path,
        selected_columns=selected_columns,
        model_columns=model_columns,
        file_in=src_path,
        water_mask=water_mask,
    )

    out_name = Path(src_path).stem + "_OSW_ODW.tif"
    out_path = str(Path(folder_out) / out_name)

    # Write georeferenced output using the source GeoTIFF georeferencing
    ref_crs = r.profile.get("crs")
    ref_transform = r.profile.get("transform")
    write_osw_geotiff(r.profile, ref_transform, ref_crs, out_path, rgb)

    return InferenceResult(rgb_img=rgb, output_path=out_path)
