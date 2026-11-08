# SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Write the World Ocean Atlas basin mask as mask.bin, one byte per cell.

The mask is xarray's tutorial dataset `basin_mask`: basin codes on a 1-degree
grid at 33 depths, NaN on land. mask.bin holds (Z, Y, X) = (33, 180, 360) bytes,
1 for ocean and 0 for land, with the depths in metres in depths.txt.

    python export_mask.py [path/to/basin_mask.nc]
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

here = Path(__file__).parent
if len(sys.argv) > 1:
    ds = xr.open_dataset(sys.argv[1])
else:
    ds = xr.tutorial.open_dataset("basin_mask")
basin = ds["basin"].transpose("Z", "Y", "X")
ocean = (~np.isnan(basin.values)).astype(np.uint8)
assert ocean.shape == (33, 180, 360), ocean.shape
(here / "mask.bin").write_bytes(ocean.tobytes())
(here / "depths.txt").write_text("\n".join(f"{z:g}" for z in basin["Z"].values) + "\n")
print(f"wrote mask.bin: {ocean.shape}, ocean fraction {ocean.mean():.3f}")
