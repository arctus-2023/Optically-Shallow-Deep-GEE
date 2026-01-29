from pathlib import Path
from cli.main import gen_osw

def test_gen_osw():
    in_path = Path("/home/yan/WorkSpace/pycharm_proj/optical-shallow-deep-new/examples/S2_L1TOA_20200825T1628_330094_10m.tif")
    # in_path = Path(
    #     "/home/yan/WorkSpace/pycharm_proj/optical-shallow-deep-new/examples/lakes/L1/s2_msi/test/2023/S2_L1TOA_20230813T1638_test_10m.tif")

    watermask_path = Path("/home/yan/WorkSpace/pycharm_proj/optical-shallow-deep-new/examples/S2_L1TOA_20200825T1628_330094_10m_watermsk.tif")
    # watermask_path = None
    out_dir = Path("/home/yan/WorkSpace/pycharm_proj/optical-shallow-deep-new/examples")
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_osw(l1toa_path=str(in_path), watermask_path=str(watermask_path), output_dir=str(out_dir))


if __name__ == "__main__":
    test_gen_osw()