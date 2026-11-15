// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A table handle as a DataFusion table.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::array::Array;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::stats::Precision;
use datafusion::common::{Result, Statistics};
use datafusion::datasource::TableType;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;

use crate::handle_scan_exec::HandleScanExec;
use crate::table_handle::TableHandle;

/// A table handle as a DataFusion table provider.
#[derive(Debug)]
pub(crate) struct HandleTable {
    handle: TableHandle,
}

impl HandleTable {
    pub(crate) fn new(handle: TableHandle) -> Self {
        Self { handle }
    }

    pub(crate) fn handle(&self) -> &TableHandle {
        &self.handle
    }
}

#[async_trait]
impl TableProvider for HandleTable {
    fn schema(&self) -> SchemaRef {
        self.handle.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn statistics(&self) -> Option<Statistics> {
        Some(statistics(&self.handle.read(), &self.handle.schema()))
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let scan = HandleScanExec::try_new(self.handle.clone(), projection.cloned())?;
        Ok(Arc::new(scan))
    }
}

/// The statistics of a table's rows, which are always inexact (design §5.2).
///
/// Without row counts, DataFusion plans every join partitioned. But a plan
/// outlives the rows it was planned on, and DataFusion answers some
/// queries, such as `COUNT(*)`, from exact statistics without reading a row
/// (spike 4a).
pub(crate) fn statistics(rows: &[RecordBatch], schema: &SchemaRef) -> Statistics {
    let mut statistics = Statistics::new_unknown(schema);
    let count: usize = rows.iter().map(RecordBatch::num_rows).sum();
    let bytes: usize = rows.iter().map(sliced_size).sum();
    statistics.num_rows = Precision::Inexact(count);
    statistics.total_byte_size = Precision::Inexact(bytes);
    statistics
}

/// The bytes of a batch's slices of its buffers. A batch's
/// `get_array_memory_size` counts each buffer whole, so slices of one buffer
/// would count it once each.
fn sliced_size(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|column| {
            column
                .to_data()
                .get_slice_memory_size()
                .unwrap_or_else(|_| column.get_array_memory_size())
        })
        .sum()
}
