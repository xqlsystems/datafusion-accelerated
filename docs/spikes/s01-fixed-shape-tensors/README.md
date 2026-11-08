<!--
SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike 1: fixed-shape tensors through DataFusion's operators

**Question** (design §12.3, spike 1). Arrow's canonical *fixed-shape
tensor* extension type is a `FixedSizeList` column whose field carries the
extension's name and shape, so that each row's value is a dense array.
Design C (supplement S2.3) would have DataFusion carry tiles of arrays this
way, keeping SQL's semantics in the engine. How far does such a column pass
through DataFusion's joins, aggregates and other operators, keeping its
type, and without being copied or expanded to rows?

**Answer: DataFusion carries the type faithfully, copies tiles wherever it
selects or reorders rows, and can't compute on them.**
- **The type survives.** Every operator that kept the column kept its
  extension metadata: projection, filter, limit, sort, a hash join's
  payload, `UNION ALL`, a window, and `first_value` in a `GROUP BY`.
- **Values are zero-copy only where an operator slices.** Projection,
  `LIMIT` and `UNION ALL` pass the input's own buffer. Filter, sort, a hash
  join's payload, a window and an aggregate copy the tiles. A copy is a
  contiguous block per row, far cheaper than expanding a tile to rows, but
  it is a copy.
- **No arithmetic on tensors.** `t * 2.0` and `SUM(t)` fail to plan, and
  `array_agg(t)` gives a list of tensors, which loses the tensor type. Every
  array operation would have to be a user-defined function, each needing a
  derivative rule in ddx, as S2.3 expected.
- **No device buffers.** arrow-rs, DataFusion's Arrow library, keeps every
  buffer in host memory, so the spike's "on device buffers" half can't run
  in DataFusion today.

So design C stays closed for the milestones: it would need tensor-valued
functions in DataFusion and device memory in arrow-rs, both upstream, and
it would still copy tiles at every filter, sort and join. Design B is the
path. Tiles along a trailing dimension, kept as an option in S2.3, remain a
possible layout for later.

**And one correction to the design.** The design says `DISTRIBUTE BY`,
"which DataFusion already parses", can override an intermediate's
partitioning (§10.5). DataFusion 55.2 parses it, but its physical planner
refuses it: "This feature is not implemented: Physical plan does not
support DistributeBy partitioning". §10.5 now says so, and the override
goes through the schedule instead.

Status: done. Date: 2026-11-07. DataFusion 55.2.0, Rust 1.99, on an x86-64
Linux machine.

## How

[`src/main.rs`](src/main.rs) builds a table `x` of 1,000 rows, in four
batches, with a key `k` and a tensor column `t` of 8 × 8 float64 values per
row, marked `arrow.fixed_shape_tensor` with shape `[8, 8]`, and a second
table `y` of keys and weights. It runs 15 queries at 1 and 4 target
partitions. For each output tensor column it checks the field's extension
metadata, and whether the values' buffer lies inside one of the input's
buffers (zero-copy) or not (a copy). Run it with `cargo run --release` in
this directory. [`results.md`](results.md) has every run.

## Results

The same at 1 and 4 target partitions:

| Operator | Runs | Keeps the tensor type | Values |
|---|---|---|---|
| projection | yes | yes | zero-copy |
| `LIMIT` | yes | yes | zero-copy |
| `UNION ALL` | yes | yes | zero-copy |
| filter | yes | yes | copied |
| sort | yes | yes | copied |
| hash join, tensor as payload | yes | yes | copied |
| window (`ROW_NUMBER() OVER (ORDER BY k)`) | yes | yes | copied |
| `GROUP BY` with `first_value(t)` | yes | yes | copied |
| `GROUP BY` with `array_agg(t)` | yes | no: a list of tensors | |
| element access, `t[1]` | yes | a scalar out | |
| join on tensor equality | yes | the key out | |
| `CAST(t AS DOUBLE[])` | yes | a plain list out | |
| `t * 2.0` | no: can't coerce arithmetic on a `FixedSizeList` | | |
| `SUM(t)` | no: no signature of `sum` matches | | |
| `DISTRIBUTE BY k` | no: the physical planner doesn't support `DistributeBy` | | |

## What it doesn't show

- **The cost of the copies.** The spike tells a copy from a zero-copy, but
  doesn't time either.
- **User-defined functions on tensors.** A function taking and returning
  `FixedSizeList` would get the same buffers as any column; the spike didn't
  write one.
- **DuckDB**, which torch-sql's spike S4 found turns the type into a plain
  fixed-size list.
