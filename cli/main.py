from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Optical Shallow/Deep (refactored) runner")
    p.add_argument("--input", required=True, help="Path to input raster (GeoTIFF TOA)")
    p.add_argument("--watermask",default='None', help="Path to watermask raster (GeoTIFF)")
    p.add_argument("--out", required=True, help="Output folder")
    p.add_argument("--cloud-buffer", type=int, default=8, help="Cloud buffer dilation iterations")
    return p

def gen_osw(l1toa_path:str,watermask_path:str = None, output_dir:str = None):
    from inputs.readers.NEWFORMAT_reader import GeeToaGeotiffReader
    from inputs.validators import validate_required_bands
    from pipeline.preprocessing import preprocess
    from pipeline.inference import run_toa_inference

    in_path = Path(l1toa_path)
    watermask_path = Path(watermask_path) if watermask_path is not None else None
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose reader
    if in_path.suffix.lower() in (".tif", ".tiff"):
        reader = GeeToaGeotiffReader()
    else:
        raise SystemExit(f"Unsupported input type: {in_path.suffix}")

    raster = reader.read(str(in_path))

    # Validate
    required = ["B3", "B4", "B8A", "B2", "B5", "B8", "B11"]
    # Note: validator expects original bandnames; for GEE inputs these are typically B2,B3,...
    v = validate_required_bands(raster, required)
    if not v.ok:
        for m in v.messages:
            print("ERROR:", m)
        raise SystemExit(2)

    # cloud, water, and glint masking
    pp = preprocess(str(in_path),file_watermsk=watermask_path,output_dir=out_dir, buffer_size=8)
    result = run_toa_inference(raster,
                               src_path=str(in_path),
                               folder_out=str(out_dir),
                               water_mask=pp.water_mask)

    print(f"\nDone. Output written to: {result.output_path}")

def main(argv=None):
    args = build_parser().parse_args(argv)
    print(f"watermask: {args.watermask}")
    in_path = args.input
    watermask_path = str(args.watermask) if args.watermask.lower() != 'none' else None
    out_dir = Path(args.out)
    gen_osw(l1toa_path=str(in_path),watermask_path=watermask_path, output_dir=str(out_dir))

if __name__ == "__main__":
    main()
