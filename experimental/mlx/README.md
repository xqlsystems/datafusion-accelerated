<!--
SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors

SPDX-License-Identifier: Apache-2.0
-->

# datafusion-accelerated-mlx (experimental)

**This package is experimental and temporary.** It holds the MLX path that
Ray Cromwell ([@cromwellian](https://github.com/cromwellian)) built, an XQL
author, until DataFusion Accelerated's milestones replace it, part by
part. Nothing outside this repository's `experimental/` directory and
XQL's demos may import it, so that retiring it touches only this
repository and the demos.

## Background

[DataFusion](https://datafusion.apache.org/) is a query engine. It plans
SQL into logical operators (scans, joins, aggregates and so on), then into
physical operators, and runs those on Arrow record batches, one row per
array cell.

The MLX path runs some of that SQL on *compact arrays* instead: dense
NumPy or [MLX](https://github.com/ml-explore/mlx) arrays indexed by the
SQL's coordinate columns. MLX is Apple's array library for its GPUs.
With it, Ray trained a 123.6M-parameter GPT-2 reproduction from SQL on an
Apple M5 Ultra, at 238.95 ms per warm step, against 185.40 ms for PyTorch
on the same GPU (DataFusion Accelerated's supplement, S1.1). The plan for
moving that work here is
[issue #41](https://github.com/xqlsystems/datafusion-accelerated/issues/41).

## What's here, and what replaces it

| Part | From | Retires at |
|---|---|---|
| `DenseProgram`, which interprets DataFusion's logical operators on compact arrays, and its proofs | einfold#157 and Ray's fork of einfold, to `3108f7a` | M6 (B-2) |
| The native module, `_native`: `shared_ctes`, which splits a `SELECT` into its common table expressions, and `_SortedArrowTable`, a table that declares its sort order | xarray-sql's `src/shared.rs`, at `a1d237a` on Ray's branch `codex/numerical-value-identities` | with `SharedExecutor`, once DataFusion computes a shared CTE once (#39, xarray-sql#290) |
| `SharedExecutor` (`shared.py`), which computes each shared CTE once per run | xarray-sql#302, to `a1d237a` | once DataFusion computes a shared CTE once (#39, xarray-sql#290) |
| `DenseExecutor` (`dense.py`, `_dense_*.py`), which runs SQL on compact arrays, the MLX operators (`mlx.py`) and `register_erf` (`math.py`) | xarray-sql#303, to `a1d237a` | M2 on Apple's GPUs, then M5 |
| `TrainingSession` (`training.py`), which keeps a whole training step on the GPU | xarray-sql#305, to `a1d237a` | wrapped behind `Program` at M1; retires at M6 |

The milestones are in the design, [§12.2](../../docs/design.md).

## What it depends on

It uses only released, public APIs: xarray-sql 0.8.0's `Context` records,
ddxdb 0.4.12's `to_sql` and `ad.dataframe`, and einfold 0.3.0rc1's dense
backend. Ray's code reached private names and forks for these; issue #41
lists what replaced each.

## Using it

The package has a native module in Rust (`rust/`), which maturin builds
when uv installs the package, so installing it from source needs a Rust
toolchain, 1.94 or later.

```sh
uv run --project experimental/mlx pytest experimental/mlx/tests
```

The tests of the float32 path run only with `MLX_ENABLE_TF32=0` set before
Python starts, as MLX reads it on first use. MLX has no GPU on Linux, so with
MLX 0.32.4 on an x86-64 Linux machine (issue #41),
26 of them fail, on Ray's code as on this package: 16 need a GPU
stream, 7 differ in float32 by more than their tolerance, and 3 count GPU
work that ran on the CPU.
