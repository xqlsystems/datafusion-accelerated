// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A program whose steps are planned.

use std::fmt::Write;
use std::sync::Arc;

use datafusion::catalog::TableProvider;
use datafusion::common::{Result, ScalarValue, TableReference};
use datafusion::execution::session_state::SessionState;
use datafusion::physical_plan::{ExecutionPlan, displayable};

use crate::handle_table::HandleTable;
use crate::table_store::{HandleId, TableStore};
use crate::{CompiledProgram, Error, TableHandle};

/// A program whose steps are planned, in the order their reads give
/// (design §5.4). [`explain`](LoweredProgram::explain) shows it, and
/// [`compile`](LoweredProgram::compile) fixes it to run.
#[derive(Debug)]
pub struct LoweredProgram {
    pub(crate) steps: Vec<LoweredStep>,
    pub(crate) states: Vec<(String, TableHandle)>,
    pub(crate) inputs: Vec<(String, TableHandle)>,
    pub(crate) params: Option<Params>,
    pub(crate) outputs: Vec<String>,
    /// The session's tables the steps read, as `lower` found them.
    pub(crate) session_reads: Vec<SessionRead>,
    // The session's state, with the program's tables in front.
    pub(crate) state: SessionState,
    // The session's own state, whose catalog is the session's.
    pub(crate) session: SessionState,
    pub(crate) store: Arc<TableStore>,
}

/// A table of the session that a step reads, and the provider `lower`
/// planned it with.
#[derive(Debug)]
pub(crate) struct SessionRead {
    /// The name, as SQL writes it.
    pub(crate) name: String,
    pub(crate) reference: TableReference,
    pub(crate) provider: Arc<dyn TableProvider>,
}

/// One planned step.
#[derive(Debug)]
pub(crate) struct LoweredStep {
    pub(crate) name: String,
    pub(crate) sql: String,
    /// The other steps it reads.
    pub(crate) reads: Vec<String>,
    pub(crate) plan: Arc<dyn ExecutionPlan>,
    pub(crate) target: Target,
}

/// Where a step's result goes.
#[derive(Debug)]
pub(crate) enum Target {
    /// A table of the run, which later steps read.
    Step(TableHandle),
    /// The next value of the state table at this index.
    State(usize),
}

/// A program's parameters: their table, and their initial values.
#[derive(Debug)]
pub(crate) struct Params {
    pub(crate) handle: TableHandle,
    pub(crate) values: Vec<(String, ScalarValue)>,
}

impl LoweredProgram {
    /// Fails with `Error::Stale` if the session no longer gives a name the
    /// program reads, or one of its state tables, the table `lower` found.
    /// A plan holds the table it was planned with, so without this a run
    /// would read a dropped table, or publish into one, with no error.
    pub(crate) async fn check_fresh(&self) -> Result<()> {
        let states = self.states.iter().map(|(_, handle)| {
            let provider: Arc<dyn TableProvider> = Arc::new(HandleTable::new(handle.clone()));
            (handle.name(), TableReference::from(handle.name()), provider)
        });
        let reads = self.session_reads.iter().map(|read| {
            (
                read.name.as_str(),
                read.reference.clone(),
                Arc::clone(&read.provider),
            )
        });
        for (name, reference, planned) in states.chain(reads) {
            let now = match self.session.schema_for_ref(reference.clone()) {
                Ok(schema) => schema.table(reference.table()).await?,
                Err(_) => None,
            };
            if !now.is_some_and(|now| same_table(&planned, &now)) {
                return Err(Error::Stale {
                    table: name.to_string(),
                }
                .into());
            }
        }
        Ok(())
    }

    /// The runtime tables a run's snapshot lists: the state tables and the
    /// runtime tables the steps read.
    pub(crate) fn snapshot_ids(&self) -> Vec<HandleId> {
        let states = self.states.iter().map(|(_, handle)| handle.id());
        let reads = self.session_reads.iter().filter_map(|read| {
            read.provider
                .downcast_ref::<HandleTable>()
                .map(|table| table.handle().id())
        });
        states.chain(reads).collect()
    }

    /// The program as text: its tables, how it publishes, and its steps in
    /// order, each with the steps it reads and its physical plan.
    #[must_use]
    pub fn explain(&self) -> String {
        let mut text = String::new();
        let names = |tables: &[(String, TableHandle)]| {
            tables
                .iter()
                .map(|(_, handle)| handle.name().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let _ = writeln!(text, "state: {}", names(&self.states));
        let _ = writeln!(text, "inputs: {}", names(&self.inputs));
        if let Some(params) = &self.params {
            let values: Vec<String> = params
                .values
                .iter()
                .map(|(name, value)| format!("${name} = {value} ({})", value.data_type()))
                .collect();
            let _ = writeln!(text, "parameters: {}", values.join(", "));
        }
        let _ = writeln!(text, "outputs: {}", self.outputs.join(", "));
        let _ = writeln!(
            text,
            "publish: every state table at once, after the last step; a state table replaced \
             during a run is a conflict, which fails the run unless it is run with \
             OnConflict::Overwrite"
        );
        for (i, step) in self.steps.iter().enumerate() {
            let what = match step.target {
                Target::Step(_) => String::new(),
                Target::State(index) => {
                    format!(" (the next value of {})", self.states[index].1.name())
                }
            };
            let reads = if step.reads.is_empty() {
                String::from("no other step")
            } else {
                step.reads.join(", ")
            };
            let _ = writeln!(text, "step {}: {}{what}, reads {reads}", i + 1, step.name);
            let _ = writeln!(text, "  sql: {}", step.sql.trim());
            for line in displayable(step.plan.as_ref())
                .indent(true)
                .to_string()
                .lines()
            {
                let _ = writeln!(text, "  {line}");
            }
        }
        text
    }

    /// Fixes the program's plans to run. In M0 there's nothing between
    /// DataFusion's plans and the ones that run, so this only keeps them.
    pub fn compile(self) -> Result<CompiledProgram> {
        Ok(CompiledProgram::new(self))
    }
}

/// Whether two providers are one table: the same handle, for the runtime's
/// tables, else the same provider.
fn same_table(a: &Arc<dyn TableProvider>, b: &Arc<dyn TableProvider>) -> bool {
    match (
        a.downcast_ref::<HandleTable>(),
        b.downcast_ref::<HandleTable>(),
    ) {
        (Some(a), Some(b)) => a.handle().id() == b.handle().id(),
        (None, None) => std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(b)),
        _ => false,
    }
}
