// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A program's own tables, in front of its session's.
//!
//! DataFusion's SQL planner finds tables through the session's catalog. A
//! program's inputs, state tables, parameters and step results must be
//! found by name too, without being registered in the session, where they'd
//! collide with users' tables. So a program plans its steps in a copy of the
//! session's state whose default schema puts the program's tables in front
//! of the session's. `lower` refuses a name that is both, so neither hides
//! the other.

use std::collections::HashMap;
use std::sync::{Arc, PoisonError, RwLock};

use async_trait::async_trait;
use datafusion::catalog::{CatalogProvider, CatalogProviderList, SchemaProvider, TableProvider};
use datafusion::common::{DataFusionError, Result, plan_err};
use datafusion::execution::session_state::{SessionState, SessionStateBuilder};

/// A program's own tables, by name.
#[derive(Debug, Default)]
pub(crate) struct ProgramTables {
    tables: RwLock<HashMap<String, Arc<dyn TableProvider>>>,
}

impl ProgramTables {
    pub(crate) fn insert(&self, name: String, table: Arc<dyn TableProvider>) {
        self.tables
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(name, table);
    }

    fn get(&self, name: &str) -> Option<Arc<dyn TableProvider>> {
        self.tables
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(name)
            .cloned()
    }

    fn names(&self) -> Vec<String> {
        self.tables
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .keys()
            .cloned()
            .collect()
    }
}

/// A copy of `state` in which the default schema finds `tables` first.
pub(crate) fn layered(state: &SessionState, tables: Arc<ProgramTables>) -> Result<SessionState> {
    let options = &state.config().options().catalog;
    let (catalog_name, schema_name) = (
        options.default_catalog.clone(),
        options.default_schema.clone(),
    );
    let base_list = Arc::clone(state.catalog_list());
    let Some(base_catalog) = base_list.catalog(&catalog_name) else {
        return plan_err!("the session has no default catalog {catalog_name}");
    };
    let Some(base_schema) = base_catalog.schema(&schema_name) else {
        return plan_err!("the session has no default schema {schema_name}");
    };
    let schema = Arc::new(LayeredSchema {
        program: tables,
        base: base_schema,
    });
    let catalog = Arc::new(LayeredCatalog {
        base: base_catalog,
        schema_name,
        schema,
    });
    let list = Arc::new(LayeredCatalogList {
        base: base_list,
        catalog_name,
        catalog,
    });
    Ok(SessionStateBuilder::new_from_existing(state.clone())
        .with_catalog_list(list)
        .build())
}

#[derive(Debug)]
struct LayeredCatalogList {
    base: Arc<dyn CatalogProviderList>,
    catalog_name: String,
    catalog: Arc<LayeredCatalog>,
}

impl CatalogProviderList for LayeredCatalogList {
    fn register_catalog(
        &self,
        name: String,
        catalog: Arc<dyn CatalogProvider>,
    ) -> Option<Arc<dyn CatalogProvider>> {
        self.base.register_catalog(name, catalog)
    }

    fn catalog_names(&self) -> Vec<String> {
        self.base.catalog_names()
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn CatalogProvider>> {
        if name == self.catalog_name {
            Some(Arc::clone(&self.catalog) as Arc<dyn CatalogProvider>)
        } else {
            self.base.catalog(name)
        }
    }
}

#[derive(Debug)]
struct LayeredCatalog {
    base: Arc<dyn CatalogProvider>,
    schema_name: String,
    schema: Arc<LayeredSchema>,
}

impl CatalogProvider for LayeredCatalog {
    fn schema_names(&self) -> Vec<String> {
        self.base.schema_names()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        if name == self.schema_name {
            Some(Arc::clone(&self.schema) as Arc<dyn SchemaProvider>)
        } else {
            self.base.schema(name)
        }
    }
}

#[derive(Debug)]
struct LayeredSchema {
    program: Arc<ProgramTables>,
    base: Arc<dyn SchemaProvider>,
}

#[async_trait]
impl SchemaProvider for LayeredSchema {
    fn table_names(&self) -> Vec<String> {
        let mut names = self.program.names();
        names.extend(self.base.table_names());
        names
    }

    async fn table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>, DataFusionError> {
        match self.program.get(name) {
            Some(table) => Ok(Some(table)),
            None => self.base.table(name).await,
        }
    }

    fn table_exist(&self, name: &str) -> bool {
        self.program.get(name).is_some() || self.base.table_exist(name)
    }
}
