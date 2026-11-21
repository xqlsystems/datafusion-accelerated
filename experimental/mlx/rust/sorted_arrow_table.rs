// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A table of one Arrow batch that tells DataFusion its sort order.

use std::sync::Arc;

use arrow::pyarrow::FromPyArrow;
use arrow::record_batch::RecordBatch;
use datafusion::catalog::TableProvider;
use datafusion::common::{Column, Result};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::Expr;
use datafusion_ffi::table_provider::FFI_TableProvider;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

/// One Arrow batch, as a table whose rows are sorted by `keys`, ascending
/// with NULLs last, so that DataFusion can skip sorting it again.
///
/// The table doesn't check the order: `SharedExecutor` sorts the batch, or
/// checks it, before it builds one, and doesn't change the batch while a
/// plan reads it. A batch out of order gives wrong answers.
#[pyclass(
    module = "datafusion_accelerated_mlx._native",
    name = "_SortedArrowTable"
)]
pub struct SortedArrowTable {
    table: Arc<dyn TableProvider>,
}

impl SortedArrowTable {
    /// The table of `batch`, sorted by `keys`.
    pub fn try_new(batch: RecordBatch, keys: Vec<String>) -> Result<Self> {
        let ordering = keys
            .into_iter()
            .map(|key| Expr::Column(Column::from_name(key)).sort(true, false))
            .collect();
        let table =
            MemTable::try_new(batch.schema(), vec![vec![batch]])?.with_sort_order(vec![ordering]);
        Ok(Self {
            table: Arc::new(table),
        })
    }

    /// The table, as a DataFusion provider.
    #[must_use]
    pub fn provider(&self) -> Arc<dyn TableProvider> {
        Arc::clone(&self.table)
    }
}

#[pymethods]
impl SortedArrowTable {
    #[new]
    fn py_new(batch: &Bound<'_, PyAny>, keys: Vec<String>) -> PyResult<Self> {
        let batch = RecordBatch::from_pyarrow_bound(batch)?;
        Self::try_new(batch, keys).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// The table for datafusion-python's `register_table`, through
    /// datafusion-ffi.
    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
        session: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let codec = datafusion_python_util::ffi_logical_codec_from_pycapsule(session, None)?;
        let provider = FFI_TableProvider::new_with_ffi_codec(self.provider(), true, None, codec);
        PyCapsule::new_with_value(py, provider, c"datafusion_table_provider")
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::Int64Array;
    use datafusion::physical_plan::ExecutionPlanProperties;
    use datafusion::prelude::SessionContext;

    use super::*;

    #[tokio::test]
    async fn the_table_declares_its_order_to_the_plan() -> Result<()> {
        let batch =
            RecordBatch::try_from_iter([("k", Arc::new(Int64Array::from(vec![1, 2, 3])) as _)])?;
        let ctx = SessionContext::new();
        let table = SortedArrowTable::try_new(batch, vec!["k".to_string()])?;
        ctx.register_table("t", table.provider())?;
        let plan = ctx.table("t").await?.create_physical_plan().await?;
        let ordering = plan.output_ordering().expect("an ordering");
        assert_eq!(ordering.to_string(), "k@0 ASC NULLS LAST");
        let sorted = ctx.sql("SELECT k FROM t ORDER BY k").await?;
        let plan = sorted.create_physical_plan().await?;
        let shown = datafusion::physical_plan::displayable(plan.as_ref())
            .indent(true)
            .to_string();
        assert!(!shown.contains("SortExec"), "{shown}");
        Ok(())
    }
}
