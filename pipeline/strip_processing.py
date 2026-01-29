from __future__ import annotations

import numpy as np

from pipeline.legacy.opticallyshallowdeep.make_vertical_strips import make_vertical_strips
from pipeline.legacy.opticallyshallowdeep.process_as_strips import process_as_strips


def split_cloud_mask_into_strips(cloud_mask: np.ndarray):
    """Return list of 5 vertical strips of the cloud mask (bool arrays)."""
    return make_vertical_strips(cloud_mask.astype(bool, copy=False))


def run_legacy_strip_pipeline(
    full_img: np.ndarray,
    image_path: str,
    if_sr: bool,
    model_path: str,
    selected_columns,
    model_columns,
    file_in: str,
    water_mask: np.ndarray,
):
    """Run the original strip-based inference using a provided cloud mask."""
    water_list = split_cloud_mask_into_strips(water_mask == 2)
    return process_as_strips(
        full_img,
        image_path,
        if_sr,
        model_path,
        selected_columns,
        model_columns,
        file_in,
        water_list,
    )
