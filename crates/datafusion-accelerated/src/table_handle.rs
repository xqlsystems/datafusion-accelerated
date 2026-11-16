// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A table the runtime owns (design §5.2).

use std::fmt;
use std::sync::{Arc, OnceLock, Weak};

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::CatalogProviderList;
use datafusion::common::{Result, TableReference, plan_err};
use datafusion::execution::session_state::SessionState;
use futures::FutureExt;

use uuid::Uuid;

use crate::handle_table::HandleTable;
use crate::instance::{Instance, Layout, Owner, Tier};
use crate::table_store::{HandleId, TableInfo, TableStore};
use crate::{Error, TableVersion};

/// A table the runtime owns, as a buffer handle refers to an allocation.
///
/// Plans read it by its name. A plan's scan of it reads its rows when the
/// plan runs, not when it's planned, so replacing them never forces a plan
/// to be planned again (D16).
///
/// Cloning a handle shares it. The table's rows live while anything holds
/// it: the session's catalog, a plan that scans it, or a handle. So
/// `DROP TABLE` or `deregister_table` frees them once no plan or handle
/// holds the table. A handle can't be replaced after that
/// (`Error::Detached`).
#[derive(Clone)]
pub struct TableHandle {
    inner: Arc<HandleInner>,
}

struct HandleInner {
    id: HandleId,
    name: String,
    schema: SchemaRef,
    private: bool,
    // Where the session's catalog has the table, once it's registered.
    home: OnceLock<Home>,
    // Shared with the session's runtime and with every handle it owns.
    store: Arc<TableStore>,
}

/// Where a session's catalog has a table. The catalog is held weakly, so a
/// handle doesn't keep its session's tables alive.
struct Home {
    catalogs: Weak<dyn CatalogProviderList>,
    catalog: String,
    schema: String,
    table: String,
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
        Self::with_privacy(store, name, schema, batches, false)
    }

    /// A table of a program's own, such as an input, which snapshots don't
    /// list.
    pub(crate) fn new_private(
        store: Arc<TableStore>,
        name: String,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Result<Self> {
        Self::with_privacy(store, name, schema, batches, true)
    }

    fn with_privacy(
        store: Arc<TableStore>,
        name: String,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
        private: bool,
    ) -> Result<Self> {
        let batches = stamped(&name, &schema, batches)?;
        let info = TableInfo {
            name: name.clone(),
            table: Uuid::new_v4(),
            private,
        };
        let id = store.insert(info, batches);
        Ok(Self {
            inner: Arc::new(HandleInner {
                id,
                name,
                schema,
                private,
                home: OnceLock::new(),
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

    /// The table's current version.
    #[must_use]
    pub fn version(&self) -> TableVersion {
        self.inner.store.version(self.inner.id)
    }

    /// The table's copies. In M0, one in host memory, as Arrow rows, owned
    /// by the runtime.
    #[must_use]
    pub fn instances(&self) -> Vec<Instance> {
        vec![Instance::new(Tier::Cpu, Layout::Rows, Owner::Runtime)]
    }

    /// Replaces the table's rows, as its next version. Runs that already
    /// began keep the rows they pinned (D20).
    ///
    /// Fails if a batch's schema isn't contained in the table's, or with
    /// `Error::Detached` if the session no longer has this table under its
    /// name: after `DROP TABLE`, `CREATE OR REPLACE TABLE`, or when the
    /// session is gone. The batches are stored with the table's schema, so
    /// reads give it back.
    pub fn replace(&self, batches: Vec<RecordBatch>) -> Result<()> {
        self.check_attached()?;
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

    pub(crate) fn id(&self) -> HandleId {
        self.inner.id
    }

    /// Records where `state`'s catalog has the table, once it's registered
    /// under `reference`.
    pub(crate) fn set_home(&self, state: &SessionState, reference: TableReference) {
        let options = &state.config().options().catalog;
        let resolved = reference.resolve(&options.default_catalog, &options.default_schema);
        let _ = self.inner.home.set(Home {
            catalogs: Arc::downgrade(state.catalog_list()),
            catalog: resolved.catalog.to_string(),
            schema: resolved.schema.to_string(),
            table: resolved.table.to_string(),
        });
    }

    /// Fails with `Error::Detached` if the table was registered and the
    /// session no longer has it under its name. DataFusion tells a table
    /// nothing when it's dropped, so this asks the catalog.
    fn check_attached(&self) -> Result<()> {
        let Some(home) = self.inner.home.get() else {
            return Ok(());
        };
        let schema = home
            .catalogs
            .upgrade()
            .and_then(|catalogs| catalogs.catalog(&home.catalog))
            .and_then(|catalog| catalog.schema(&home.schema));
        let attached = match schema {
            None => false,
            // A catalog in memory, as a session's is by default, answers at
            // once. One that would wait can't be asked here, so this fails
            // closed.
            Some(schema) => match schema.table(&home.table).now_or_never() {
                Some(provider) => provider?.is_some_and(|provider| {
                    provider
                        .downcast_ref::<HandleTable>()
                        .is_some_and(|table| table.handle().id() == self.inner.id)
                }),
                None => {
                    return plan_err!(
                        "the session's catalog didn't answer at once, so replace can't check \
                         that {} is still the session's table",
                        self.inner.name
                    );
                }
            },
        };
        if attached {
            Ok(())
        } else {
            Err(Error::Detached {
                table: self.inner.name.clone(),
            }
            .into())
        }
    }

    /// Whether the table is a program's own, such as an input.
    pub(crate) fn is_private(&self) -> bool {
        self.inner.private
    }

    /// The table's rows as `run` sees them, or its current rows outside a
    /// run.
    pub(crate) fn read_in(&self, run: Option<u64>) -> Result<Arc<[RecordBatch]>> {
        match run {
            Some(run) => match self.inner.store.for_run(run, self.inner.id)? {
                Some(rows) => Ok(rows),
                None => Err(Error::CreatedAfterRun {
                    table: self.inner.name.clone(),
                    run,
                }
                .into()),
            },
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
pub(crate) fn stamped(
    name: &str,
    schema: &SchemaRef,
    batches: Vec<RecordBatch>,
) -> Result<Vec<RecordBatch>> {
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
