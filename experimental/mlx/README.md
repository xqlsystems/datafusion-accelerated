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

The milestones are in the design, [§12.2](../../docs/design.md).

## Using it

The package has a native module in Rust (`rust/`), which maturin builds
when uv installs the package, so installing it from source needs a Rust
toolchain, 1.94 or later.

```sh
uv run --project experimental/mlx pytest experimental/mlx/tests
```
