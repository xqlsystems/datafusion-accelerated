// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A table the runtime owns (design §5.2).

use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{DataFusionError, Result};
use datafusion::execution::TaskContext;

use crate::instance::{Instance, Layout, Owner, Tier};
use crate::table_store::{HandleId, TableStore};

/// A table the runtime owns, as a buffer handle refers to an allocation.
///
/// Plans read it by its name. A plan's scan of it reads its rows when the
/// plan runs, not when it's planned, so replacing them never forces a plan
/// to be planned again (D16). Each run reads one snapshot of the runtime's
/// tables. Cloning a handle shares it.
#[derive(Debug, Clone)]
pub struct TableHandle {
    id: HandleId,
    name: String,
    schema: SchemaRef,
    // Shared with the session's runtime and with every scan of the table.
    store: Arc<TableStore>,
}

impl TableHandle {
    pub(crate) fn new(
        store: Arc<TableStore>,
        name: String,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Result<Self> {
        check_schema(&name, &schema, &batches)?;
        let id = store.insert(batches);
        Ok(Self {
            id,
            name,
            schema,
            store,
        })
    }

    /// The table's name in its session, by which plans read it.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The table's schema.
    #[must_use]
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// The table's copies. In M0, one in host memory, as Arrow rows, owned
    /// by the runtime.
    #[must_use]
    pub fn instances(&self) -> Vec<Instance> {
        vec![Instance::new(Tier::Host, Layout::Rows, Owner::Runtime)]
    }

    /// Replaces the table's rows. Runs that already read the table keep the
    /// rows they read.
    ///
    /// Fails if a batch's schema isn't the table's, or the table was
    /// released.
    pub fn replace(&self, batches: Vec<RecordBatch>) -> Result<()> {
        check_schema(&self.name, &self.schema, &batches)?;
        if self.store.replace(self.id, batches) {
            Ok(())
        } else {
            Err(self.released())
        }
    }

    /// The table's current rows. This is the only way rows leave the
    /// runtime, and it's explicit.
    ///
    /// Fails if the table was released.
    pub fn read(&self) -> Result<Arc<[RecordBatch]>> {
        self.store.current(self.id).ok_or_else(|| self.released())
    }

    /// The table's rows as the run of `task` sees them.
    pub(crate) fn read_for_run(&self, task: &Arc<TaskContext>) -> Result<Arc<[RecordBatch]>> {
        self.store
            .for_run(task, self.id)
            .ok_or_else(|| self.released())
    }

    pub(crate) fn release(&self) {
        self.store.remove(self.id);
    }

    fn released(&self) -> DataFusionError {
        DataFusionError::Execution(format!("the table {} was released", self.name))
    }
}

fn check_schema(name: &str, schema: &SchemaRef, batches: &[RecordBatch]) -> Result<()> {
    for batch in batches {
        if !schema.contains(&batch.schema()) {
            return Err(DataFusionError::Plan(format!(
                "a batch for the table {name} has the schema {}, not the table's {schema}",
                batch.schema()
            )));
        }
    }
    Ok(())
}
