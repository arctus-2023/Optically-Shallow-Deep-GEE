from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import rasterio
from typing import Optional

import numpy as np
from models.cloudmask import omni_cloud_mask
from models.watermask import enhanced_otsu_watermask, waterglint_mask, update_watermask_with_cloudbuffer


@dataclass
class PreprocessOutputs:
    cloud_mask: np.ndarray  # 2D boolean array, True = cloud
    water_mask: np.ndarray # 2D uint array, 1=land, 2=water, 3=cloud



def preprocess(file_l1: str, file_watermsk:str = None, output_dir:str = None,buffer_size: int = 8) -> PreprocessOutputs:
    '''
    cloud & water masking
    Args:
        file_l1: input L1 TOA geotiff file path
        buffer_size:this will be applied to cloud mask. Default is 8, which means dilate cloud mask by 8 pixels.
        file_watermsk: if provided, use this watermask instead of generating one.
                       and buffer_size should be ignored if the cloud mask in watermask has alredy been dialated before,
                       otherwise, the cloud mask in watermask will be further dialated by buffer_size, and the
                       watermask will be updated.
    Returns:
    '''

    if file_watermsk is None:
        cldmask = omni_cloud_mask(file_l1, buffer_size=buffer_size)
        watermask = enhanced_otsu_watermask(Path(file_l1), cloud_mask=cldmask,cldmask_buffer_size=buffer_size, save_dir=output_dir)
    else:
        cldmask, watermask = update_watermask_with_cloudbuffer(Path(file_watermsk),
                                          buffer_size=buffer_size
                                          )

    watermask = waterglint_mask(Path(file_l1), water_mask=watermask)
    return PreprocessOutputs(cloud_mask=cldmask, water_mask=watermask)
