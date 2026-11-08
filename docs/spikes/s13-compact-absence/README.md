<!--
SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike 13: where do compact arrays stop paying for their absent cells?

**Question** (design §12.3, spike 13,
[#34](https://github.com/xqlsystems/datafusion-accelerated/issues/34)). A
table in long format has a row only for each cell that exists, so land in an
ocean model costs nothing. A *compact array* (design §2.1) stores every cell
of its domain, with a mask, so land comes back as masked values that are
stored and computed. At what density does that stop paying? And does a
*blocked* layout, dense blocks of a tile with the all-land blocks left out,
keep most land unstored?

**Answer: compact arrays pay in time down to a few percent ocean, but not
in memory below about 40%, and 8 by 8 blocks beat both on a real
coastline.** On every one of the 33 depth levels of a real 1° ocean mask, from 64%
ocean at the surface to 8% at 5,500 m, a compact array ran the convolution
faster than rows in DataFusion: 15.7 times at the surface and 2.2 times at
the deepest level. But it stored more bytes than rows below about 40% ocean,
because a row holds a value with its coordinates (20 bytes here) while an
array holds a value for every cell, ocean or land (8 bytes). Blocks of 8 by
8 cells were the fastest layout at every level and stored 1.28 to 2.78 cells
per ocean cell, against 1.56 to 12.04 for the compact array. They help only
because land is clustered: on a mask with the same ocean cells scattered at
random, blocks stored as much as the compact array and ran slightly slower.

## What was run

The mask is the World Ocean Atlas basin mask from xarray's tutorial data
(`basin_mask`): 180 by 360 cells at 33 depths from 0 to 5,500 m, 54% ocean
overall, with land growing with depth. It stands in for Samudra's grid,
since land is what matters here and it is the same planet at the same
resolution. [Samudra](https://github.com/m2lines/Samudra) is the ocean
emulator xarray-sql's demo 4 trains, a U-Net, which is a stack of
convolutions with pooling between them.

On each level, one 3 by 3 convolution with C input and C output channels,
the core of a U-Net layer, runs in three layouts:

- **rows**: tables `x(y, x, c, v)` with one row per ocean cell and channel,
  `w(dy, dx, ci, co, v)` and `cells(y, x)`, in DataFusion 55.2 on one
  partition, as a join and a grouped sum in SQL:
  ```sql
  WITH p AS (
    SELECT x.y - w.dy AS y, x.x - w.dx AS x, w.co AS c, SUM(x.v * w.v) AS v
    FROM x JOIN w ON x.c = w.ci
    GROUP BY x.y - w.dy, x.x - w.dx, w.co)
  SELECT p.y, p.x, p.c, p.v FROM p JOIN cells ON p.y = cells.y AND p.x = cells.x
  ```
- **compact**: one dense `Float64` array per level, land stored as zero,
  and a mask of one bit per cell. Every cell is computed, land included, as
  a dense kernel on a GPU would, and the mask is applied to the output.
- **blocked**: dense blocks of B by B cells, B = 8, 16 or 32, with only the
  blocks that hold an ocean cell stored. Each block gathers a tile of
  (B + 2)² cells, its halo from neighbouring blocks, as an exchange would
  move them (design §10.5), then computes every cell of the block.

A **scattered** control places the same number of ocean cells at random on
each level, so the effect of the coastline's shape can be told apart from
density.

Every output was checked against a direct sum over ocean neighbours, to
within design §6.3's bound for 9C terms. **All 66 runs at C = 4, and all 6
at C = 16, matched.** Times are the best of three runs, on an Intel i7-8700,
one core for the arrays and one DataFusion partition for rows.

## Results

The tables are in [results.md](results.md).

**Time.** Rows cost in proportion to the ocean cells, 233 to 266 ms per unit
of density at C = 4. The compact array costs the same at every level, about
10 ms. So, extrapolating, compact wins wherever density is above about 4% at C = 4, and the
ratio grows with channels, since the array's loops vectorize and a row
join doesn't:

| depth | ocean | rows | compact | blocked 8 | rows / compact |
|---|---|---|---|---|---|
| 0 m, C = 4 | 64% | 155 ms | 9.9 ms | 8.9 ms | 15.7 |
| 4,000 m, C = 4 | 39% | 95 ms | 10.0 ms | 6.5 ms | 9.5 |
| 5,500 m, C = 4 | 8% | 22 ms | 10.2 ms | 2.5 ms | 2.2 |
| 0 m, C = 16 | 64% | 2,281 ms | 59 ms | 52 ms | 38 |
| 5,500 m, C = 16 | 8% | 294 ms | 59 ms | 15 ms | 5.0 |

**Memory.** A row stores 20 bytes per present value (three `Int32`
coordinates and a `Float64`), and the compact array 8 bytes per cell. So the
array holds fewer bytes only when more than 8 / 20 = 40% of cells are
ocean, whatever C is. The measured crossover is at 4,000 m, where the
ocean is 39%: 2.04 MB as rows, 2.08 MB as an array. Over the whole volume,
compact stores 69 MB against rows' 92 MB, because most levels are above
40%. 8 by 8 blocks store 49 MB.

**Blocks.** On the real mask, 8 by 8 blocks stored 1.28 to 1.31 cells per
ocean cell down to 2,000 m, and 2.78 at 5,500 m, where the compact array
stores 12.04. They were the fastest layout on every level, and stored the
fewest bytes on every level but the deepest (0.49 MB against rows' 0.43).
Larger blocks keep more land: 16 by 16 stored 1.52 to 4.95 cells per ocean
cell, and 32 by 32 1.75 to 7.61. On the scattered mask nearly every block
holds an ocean cell, so blocks stored slightly more than the compact array
(from padding the grid to whole blocks) and ran 3 to 14% slower, from the
halo gathers.

| over all 33 levels, C = 4 | rows | compact | blocked 8 | blocked 32 |
|---|---|---|---|---|
| time, real mask | 4,214 ms | 329 ms | 248 ms | 352 ms |
| time, scattered | 4,673 ms | 336 ms | 359 ms | 370 ms |
| bytes, real mask | 92.4 MB | 68.7 MB | 49.3 MB | 73.9 MB |
| bytes, scattered | 92.4 MB | 68.7 MB | 70.3 MB | 78.2 MB |

## What this changes in the design

1. **Density alone is the wrong test.** The time and memory crossovers are
   an order of magnitude apart, about 4% and 40% here. A threshold on
   density can't serve both, so the layout rule in design §5.2 should
   choose by what each layout would store and compute, not by density.
2. **Choose by stored cells per present cell.** That ratio, per layout and
   block size, is cheap to compute from the domain's mask when a table is
   registered, and it captures clustering, which density doesn't: the real
   and scattered masks have the same density on every level, and very
   different ratios once blocked. It belongs with einfold's facts, as a
   measured fact about the domain.
3. **Blocks are the compact layout, and a dense array is one block.** On a
   real coastline, small blocks won on time and memory together. Where
   absence isn't clustered they cost a few percent, which a block as large
   as the whole array avoids. The block size is a schedule entry (§5.6),
   with a default from the ratio above.
4. **Rows stay for sparse data and tight memory.** Below about 40% present,
   rows hold fewer bytes than a dense array. Under a memory budget the
   schedule should prefer blocks, then rows, before spilling.

## Limits

- **CPU only, and simple loops.** A GPU changes the constants: dense
  kernels gain more from it than a join does, which would push the time
  crossover lower still. The memory arithmetic doesn't change.
- **One convolution, not a U-Net.** A U-Net pools between levels, and a
  coarser grid has fewer land cells, so its deeper layers are denser than
  the grid it starts from. Every layer of a real network would need its own
  layout.
- **Rows' bytes count the input table only**, not DataFusion's hash tables
  for the join and the grouped sum, which make rows costlier still.
- **The grid doesn't wrap in longitude**, so cells at 0° and 360° aren't
  neighbours. That changes a few edge cells, the same way in every layout.
- **The mask is the World Ocean Atlas's**, not Samudra's own grid, which
  differs in detail.

## Running it

```sh
python export_mask.py                  # writes mask.bin and depths.txt, from xarray's tutorial data
cargo run --release                    # all 33 levels, C = 4; the first build takes about 6 minutes
CHANNELS=16 LEVELS=0,28,32 cargo run --release
```

`export_mask.py` needs xarray and netCDF4. It accepts a path to
`basin_mask.nc` when xarray's tutorial data is already downloaded.
