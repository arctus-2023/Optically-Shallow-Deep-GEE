import os
import glob
import gc
import numpy as np
import rasterio

def make_multiband_image(file_in: str, folder_out: str):
    """Create (or reuse) a multi-band GeoTIFF for the downstream pipeline.

    Supports:
    - Sentinel-2 L1C SAFE folder (original): stacks JP2 bands to GeoTIFF.
    - Multi-band GeoTIFF (e.g., GEE export): returns the input path directly.
    """

    if os.path.isfile(file_in) and os.path.splitext(file_in)[1].lower() in ('.tif', '.tiff'):
        print('Input is already a GeoTIFF; using it directly: ' + str(file_in))
        return file_in

    basename = os.path.basename(file_in).rstrip('.SAFE').rstrip('.safe')
    imageFile = os.path.join(folder_out, f"{basename}.tif")

    if os.path.exists(imageFile):
        print('Multi-band geotiff exists: ' + str(imageFile))
        return imageFile

    print('Making multi-band geotiff: ' + str(imageFile))

    band_numbers = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    S2Files = [glob.glob(f'{file_in}/**/IMG_DATA/**/*{band}.jp2', recursive=True)[0] for band in band_numbers]

    b2File = S2Files[1]
    band2 = rasterio.open(b2File)
    res = int(band2.transform[0])

    arrayList = []
    for bandFile in S2Files:
        band = rasterio.open(bandFile)
        ar = band.read(1)
        bandRes = int(band.transform[0])

        if bandRes == res:
            ar = ar.astype('int16')
        elif bandRes > res:
            finerRatio = int(bandRes / res)
            ar = np.kron(ar, np.ones((finerRatio, finerRatio), dtype='int16')).astype('int16')

        arrayList.append(ar)
        del ar
        band.close()

    stack = np.dstack(arrayList)
    stackTransposed = stack.transpose(2, 0, 1)

    with rasterio.Env():
        profile = band2.profile
        profile.update(driver='GTiff', count=len(S2Files), compress='lzw')
        with rasterio.open(imageFile, 'w', **profile) as dst:
            dst.write(stackTransposed)

    band2.close()
    del stack, stackTransposed, S2Files, band2
    gc.collect()
    print('Done')
    return imageFile
