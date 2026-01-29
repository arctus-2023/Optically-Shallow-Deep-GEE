import numpy as np
from scipy import ndimage

def dialate_mask(mask: np.ndarray, buffer_size: int) -> np.ndarray:
    struct1 = ndimage.generate_binary_structure(2, 1)
    buffered = ndimage.binary_dilation(mask, structure=struct1, iterations=int(buffer_size))
    return buffered.astype(bool)