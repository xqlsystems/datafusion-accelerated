// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A table the runtime owns (design §5.2).

use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Result;

use crate::Error;
use crate::instance::{Instance, Layout, Owner, Tier};
use crate::table_store::{HandleId, TableStore};

/// A table the runtime owns, as a buffer handle refers to an allocation.
///
/// Plans read it by its name. A plan's scan of it reads its rows when the
/// plan runs, not when it's planned, so replacing them never forces a plan
/// to be planned again (D16).
///
/// Cloning a handle shares it. The table's rows live while anything holds
/// it: the session's catalog, a plan that scans it, or a handle. So
/// `DROP TABLE` or `deregister_table` frees them once no plan or handle
/// holds the table.
#[derive(Clone)]
pub struct TableHandle {
    inner: Arc<HandleInner>,
}

struct HandleInner {
    id: HandleId,
    name: String,
    schema: SchemaRef,
    // Shared with the session's runtime and with every handle it owns.
    store: Arc<TableStore>,
}

impl Drop for HandleInner {
    fn drop(&mut self) {
        self.store.remove(self.id);
    }
}

impl TableHandle {
    pub(crate) fn new(
        store: Arc<TableStore>,
        name: String,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Result<Self> {
        let batches = stamped(&name, &schema, batches)?;
        let id = store.insert(batches);
        Ok(Self {
            inner: Arc::new(HandleInner {
                id,
                name,
                schema,
                store,
            }),
        })
    }

    /// The table's name in its session, as SQL writes it.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// The table's schema.
    #[must_use]
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.inner.schema)
    }

    /// The table's copies. In M0, one in host memory, as Arrow rows, owned
    /// by the runtime.
    #[must_use]
    pub fn instances(&self) -> Vec<Instance> {
        vec![Instance::new(Tier::Cpu, Layout::Rows, Owner::Runtime)]
    }

    /// Replaces the table's rows. Runs that already began keep the rows
    /// they pinned (D20).
    ///
    /// Fails if a batch's schema isn't contained in the table's. The batches
    /// are stored with the table's schema, so reads give it back.
    pub fn replace(&self, batches: Vec<RecordBatch>) -> Result<()> {
        let batches = stamped(&self.inner.name, &self.inner.schema, batches)?;
        self.inner.store.replace(self.inner.id, batches);
        Ok(())
    }

    /// The table's current rows, each batch with the table's schema. This
    /// is the only way rows leave the runtime, and it's explicit.
    #[must_use]
    pub fn read(&self) -> Arc<[RecordBatch]> {
        self.inner.store.current(self.inner.id)
    }

    /// The table's rows as `run` sees them, or its current rows outside a
    /// run.
    pub(crate) fn read_in(&self, run: Option<u64>) -> Result<Arc<[RecordBatch]>> {
        match run {
            Some(run) => Ok(self.inner.store.for_run(run, self.inner.id)?),
            None => Ok(self.read()),
        }
    }
}

// By hand, so that printing a plan doesn't print the table's rows.
impl fmt::Debug for TableHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableHandle")
            .field("name", &self.inner.name)
            .field("id", &self.inner.id)
            .field("schema", &self.inner.schema)
            .finish()
    }
}

/// `batches`, each with `schema`, or an error if one's schema isn't
/// contained in it.
fn stamped(name: &str, schema: &SchemaRef, batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
    batches
        .into_iter()
        .map(|batch| {
            let found = batch.schema();
            batch.with_schema(Arc::clone(schema)).map_err(|_| {
                Error::SchemaMismatch {
                    table: name.to_string(),
                    expected: Arc::clone(schema),
                    found,
                }
                .into()
            })
        })
        .collect()
}
