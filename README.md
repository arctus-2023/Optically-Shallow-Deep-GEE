# optical-shallow-deep-new

Refactored version of the **Optically-Shallow-Deep** (https://github.com/yulunwu8/Optically-Shallow-Deep) pipeline 

This project currently supports **Sentinel-2 TOA multiband GeoTIFFs downloaed from Google Earth Engine** (GEE).

## Input expectations (GEE GeoTIFF)

- A multiband GeoTIFF with **band descriptions** set (rasterio `src.descriptions`) like: `B2, B3, B4, B5, B8, B8A, B11` (others can be present too).
- Pixel values are typically integer reflectance scaled by `1e4`.

## Cloud and water masking

- omnicloudmask is employed for Cloud masking
- an enhanced OTSU-based water masking approach is integrated into the piepline.

## Install dependencies

In a clean env, install dependencies:

```bash
pip install rasterio scikit-image scipy numpy tifffile tensorflow joblib omnicloudmask
```

## Run

```bash
##run for help
python -m cli.main
##run without a watermask file
python -m cli.main --input /path/to/S2_TOA_GEE.tif --out /path/to/output --cloud-buffer 8
##run with a watermask file
python -m cli.main --input /path/to/S2_TOA_GEE.tif --watermask /path/to/S2_TOA_GEE_watermask.tif --out /path/to/output --cloud-buffer 8
```

The output will be written to:

`<out>/<input_stem>_OSW_ODW.tif`
