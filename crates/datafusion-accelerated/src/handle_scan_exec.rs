// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The scan of a table handle.

use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Result, Statistics, exec_err};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::memory::MemoryStream;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};

use crate::handle_table;
use crate::table_handle::TableHandle;

/// The scan of a table handle. Planning records the handle, not its rows:
/// each run reads the rows of its snapshot when it executes.
///
/// It has one partition, since a plan's partitions are fixed when it's
/// planned and a table's batches aren't. DataFusion repartitions above it
/// as for any input.
#[derive(Debug)]
pub(crate) struct HandleScanExec {
    handle: TableHandle,
    projection: Option<Vec<usize>>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl HandleScanExec {
    pub(crate) fn try_new(handle: TableHandle, projection: Option<Vec<usize>>) -> Result<Self> {
        let schema = match &projection {
            Some(columns) => Arc::new(handle.schema().project(columns)?),
            None => handle.schema(),
        };
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            handle,
            projection,
            schema,
            properties,
        })
    }
}

impl DisplayAs for HandleScanExec {
    fn fmt_as(&self, _format: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HandleScanExec: table={}", self.handle.name())?;
        if let Some(columns) = &self.projection {
            write!(f, ", projection={columns:?}")?;
        }
        Ok(())
    }
}

impl ExecutionPlan for HandleScanExec {
    fn name(&self) -> &str {
        "HandleScanExec"
    }

    fn apply_expressions(
        &self,
        _visit: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Arc<Statistics>> {
        let rows = self.handle.read()?;
        let rows = self.project(&rows)?;
        Ok(Arc::new(handle_table::statistics(&rows, &self.schema)))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return exec_err!("HandleScanExec has one partition, not {partition}");
        }
        let rows = self.handle.read_for_run(&context)?;
        let batches = self.project(&rows)?;
        let stream = MemoryStream::try_new(batches, Arc::clone(&self.schema), None)?;
        Ok(Box::pin(stream))
    }
}

impl HandleScanExec {
    fn project(&self, rows: &[RecordBatch]) -> Result<Vec<RecordBatch>> {
        let Some(columns) = &self.projection else {
            return Ok(rows.to_vec());
        };
        let mut projected = Vec::with_capacity(rows.len());
        for batch in rows {
            projected.push(batch.project(columns)?);
        }
        Ok(projected)
    }
}
