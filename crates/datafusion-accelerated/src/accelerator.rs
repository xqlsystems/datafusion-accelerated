// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The runtime's state for one session.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::Session;
use datafusion::common::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::prelude::{SessionConfig, SessionContext};

use crate::handle_table::HandleTable;
use crate::table_store::TableStore;
use crate::{AcceleratedConfig, AcceleratorOptions, TableHandle};

/// The runtime's state for one session.
///
/// It is kept in the session's configuration, as a typed extension
/// (`SessionConfig::with_extension`), so the runtime's nodes find it
/// through the `TaskContext` that DataFusion gives each of them. That's why
/// it's shared through an `Arc`: DataFusion's extension map holds it so.
#[derive(Debug)]
pub struct Accelerator {
    options: AcceleratorOptions,
    // Shared with every handle and every scan of one.
    store: Arc<TableStore>,
    tables: Mutex<HashMap<String, TableHandle>>,
}

impl Accelerator {
    pub(crate) fn new(options: AcceleratorOptions) -> Self {
        Self {
            options,
            store: Arc::default(),
            tables: Mutex::default(),
        }
    }

    /// The options the session was built with.
    #[must_use]
    pub fn options(&self) -> &AcceleratorOptions {
        &self.options
    }

    /// The runtime of a session.
    ///
    /// Fails if the session was built without
    /// [`with_accelerator`](crate::SessionStateBuilderExt::with_accelerator).
    pub fn of(session: &dyn Session) -> Result<Arc<Accelerator>> {
        Self::of_config(session.config())
    }

    /// The runtime of the session a running node belongs to.
    pub fn of_task(task: &TaskContext) -> Result<Arc<Accelerator>> {
        Self::of_config(task.session_config())
    }

    /// The session's current `accelerated` settings, including any `SET`.
    pub fn settings(session: &dyn Session) -> Result<AcceleratedConfig> {
        Self::of(session)?;
        session
            .config()
            .options()
            .extensions
            .get::<AcceleratedConfig>()
            .cloned()
            .ok_or_else(not_attached)
    }

    /// Creates a table the runtime owns, with `batches` as its rows, and
    /// registers it with `ctx` under `name`.
    ///
    /// Fails if the session has a table by that name, or a batch's schema
    /// isn't `schema`.
    ///
    /// # Panics
    ///
    /// If another thread panicked while it held the runtime's tables.
    pub fn create_table(
        &self,
        ctx: &SessionContext,
        name: &str,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Result<TableHandle> {
        let mut tables = self.tables();
        if tables.contains_key(name) || ctx.table_exist(name)? {
            return Err(DataFusionError::Plan(format!(
                "the session already has a table named {name}"
            )));
        }
        let handle = TableHandle::new(Arc::clone(&self.store), name.to_string(), schema, batches)?;
        ctx.register_table(name, Arc::new(HandleTable::new(handle.clone())))?;
        tables.insert(name.to_string(), handle.clone());
        Ok(handle)
    }

    /// The table the runtime owns under `name`, if there is one.
    ///
    /// # Panics
    ///
    /// If another thread panicked while it held the runtime's tables.
    #[must_use]
    pub fn table(&self, name: &str) -> Option<TableHandle> {
        self.tables().get(name).cloned()
    }

    /// Deregisters a table from `ctx` and drops its rows. Runs that already
    /// read it keep the rows they read until they end.
    ///
    /// Fails if the runtime owns no table by that name.
    ///
    /// # Panics
    ///
    /// If another thread panicked while it held the runtime's tables.
    pub fn release(&self, ctx: &SessionContext, name: &str) -> Result<()> {
        let Some(handle) = self.tables().remove(name) else {
            return Err(DataFusionError::Plan(format!(
                "the runtime owns no table named {name}"
            )));
        };
        ctx.deregister_table(name)?;
        handle.release();
        Ok(())
    }

    fn tables(&self) -> MutexGuard<'_, HashMap<String, TableHandle>> {
        self.tables
            .lock()
            .expect("a thread panicked while it held the runtime's tables")
    }

    fn of_config(config: &SessionConfig) -> Result<Arc<Accelerator>> {
        config
            .get_extension::<Accelerator>()
            .ok_or_else(not_attached)
    }
}

fn not_attached() -> DataFusionError {
    DataFusionError::Configuration(
        "this session has no DataFusion Accelerated runtime: build it with \
         SessionStateBuilder::with_accelerator(), and call with_config() \
         before with_accelerator(), since with_config() replaces the \
         configuration"
            .to_string(),
    )
}
