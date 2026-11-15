// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The runtime's state for one session.

use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::config::ConfigExtension;
use datafusion::common::{Result, TableReference};
use datafusion::execution::TaskContext;
use datafusion::prelude::{SessionConfig, SessionContext};

use crate::handle_table::HandleTable;
use crate::table_store::TableStore;
use crate::{AcceleratedConfig, Error, Run, TableHandle};

/// The runtime's state for one session.
///
/// It is kept in the session's configuration, as a typed extension
/// (`SessionConfig::with_extension`), so the runtime's nodes find it
/// through the `TaskContext` that DataFusion gives each of them. That's why
/// it's shared through an `Arc`: DataFusion's extension map holds it so.
///
/// The session's catalog, not the runtime, knows the runtime's tables by
/// name: each is a table provider in it. So SQL, `DROP TABLE` and
/// `SHOW TABLES` treat them as any other table.
#[derive(Debug)]
pub struct Accelerator {
    // Shared with every handle and every run.
    store: Arc<TableStore>,
}

impl Accelerator {
    pub(crate) fn new() -> Self {
        Self {
            store: Arc::default(),
        }
    }

    /// The runtime of a session.
    ///
    /// Fails if the session was built without
    /// [`with_accelerator`](crate::AcceleratorExt::with_accelerator).
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
        Ok(settings_of(session.config())?.clone())
    }

    /// Creates a table the runtime owns, with `batches` as its rows, and
    /// registers it in `ctx` under `name`, which DataFusion normalizes as
    /// SQL does: `W` is `w`, and `"W"` is `W`.
    ///
    /// Fails if `ctx` wasn't built with the runtime, the session has a table
    /// by that name, or a batch's schema isn't contained in `schema`.
    pub fn create_table(
        ctx: &SessionContext,
        name: &str,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Result<TableHandle> {
        let store = Arc::clone(&Self::of_config(ctx.state_ref().read().config())?.store);
        let reference = TableReference::from(name);
        let name = reference.to_quoted_string();
        if ctx.table_exist(reference.clone())? {
            return Err(Error::NameTaken { name }.into());
        }
        let handle = TableHandle::new(store, name, schema, batches)?;
        ctx.register_table(reference, Arc::new(HandleTable::new(handle.clone())))?;
        Ok(handle)
    }

    /// The table the runtime owns under `name` in `ctx`, if there is one.
    pub async fn table(ctx: &SessionContext, name: &str) -> Result<Option<TableHandle>> {
        let reference = TableReference::from(name);
        if !ctx.table_exist(reference.clone())? {
            return Ok(None);
        }
        let provider: Arc<dyn TableProvider> = ctx.table_provider(reference).await?;
        Ok(provider
            .downcast_ref::<HandleTable>()
            .map(|table| table.handle().clone()))
    }

    /// Begins a run of `ctx`'s plans, which read one snapshot of the
    /// runtime's tables until the run is dropped (D20).
    pub fn begin_run(ctx: &SessionContext) -> Result<Run> {
        let mut state = ctx.state();
        let store = Arc::clone(&Self::of(&state)?.store);
        let id = store.begin_run();
        let config = state.config_mut();
        config
            .options_mut()
            .extensions
            .get_mut::<AcceleratedConfig>()
            .ok_or(Error::NotAttached)?
            .run = Some(id);
        let task = Arc::new(TaskContext::from(&state));
        Ok(Run::new(id, task, store))
    }

    fn of_config(config: &SessionConfig) -> Result<Arc<Accelerator>> {
        Ok(config
            .get_extension::<Accelerator>()
            .ok_or(Error::NotAttached)?)
    }
}

/// The `accelerated` settings of a session's configuration.
pub(crate) fn settings_of(config: &SessionConfig) -> Result<&AcceleratedConfig, Error> {
    config
        .options()
        .extensions
        .get::<AcceleratedConfig>()
        .ok_or(Error::NotAttached)
}

/// The run a plan executing with `config` belongs to, if any (D20).
///
/// A session built in Rust holds the settings as an `AcceleratedConfig`.
/// A plan that datafusion-python executes gets them across datafusion-ffi
/// as an `FFI_ExtensionOptions` instead, whose entries carry their
/// namespace, so the run is read from the `accelerated.run` entry. A value
/// there that isn't a run id is an error, never "no run", so a form of the
/// settings this doesn't know fails closed.
pub(crate) fn run_of(config: &SessionConfig) -> Result<Option<u64>, Error> {
    if let Ok(settings) = settings_of(config) {
        return Ok(settings.run);
    }
    let key = format!("{}.run", AcceleratedConfig::PREFIX);
    let Some(entry) = config
        .options()
        .entries()
        .into_iter()
        .find(|entry| entry.key == key)
    else {
        return Ok(None);
    };
    match entry.value.as_deref() {
        None | Some("") => Ok(None),
        Some(value) => value
            .parse()
            .map(Some)
            .map_err(|_| Error::UnreadableSetting {
                key,
                value: value.to_string(),
            }),
    }
}
