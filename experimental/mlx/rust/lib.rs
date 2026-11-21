// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The experimental MLX package's native module,
//! `datafusion_accelerated_mlx._native` (issue #41).
//!
//! `SharedExecutor` uses both of its parts, which it can't get from Python's
//! public APIs: [`shared_ctes`] splits a `SELECT` into the common table
//! expressions (CTEs, the named subqueries of its `WITH`) it can compute
//! once, and [`SortedArrowTable`] is a table of Arrow batches that tells
//! DataFusion its sort order. Ported from xarray-sql's `src/shared.rs` at
//! `a1d237a`, by Ray Cromwell.

use pyo3::prelude::*;

mod shared_ctes;
mod sorted_arrow_table;

pub use shared_ctes::{Part, shared_ctes};
pub use sorted_arrow_table::SortedArrowTable;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(shared_ctes::py_shared_ctes, module)?)?;
    module.add_class::<SortedArrowTable>()?;
    Ok(())
}
