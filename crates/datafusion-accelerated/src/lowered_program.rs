// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A program whose steps are planned.

use std::fmt::Write;
use std::sync::Arc;

use datafusion::common::{Result, ScalarValue};
use datafusion::execution::session_state::SessionState;
use datafusion::physical_plan::{ExecutionPlan, displayable};

use crate::table_store::TableStore;
use crate::{CompiledProgram, TableHandle};

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
    // The session's state, with the program's tables in front.
    pub(crate) state: SessionState,
    pub(crate) store: Arc<TableStore>,
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
    pub(crate) fn new(
        steps: Vec<LoweredStep>,
        states: Vec<(String, TableHandle)>,
        inputs: Vec<(String, TableHandle)>,
        params: Option<Params>,
        outputs: Vec<String>,
        state: SessionState,
        store: Arc<TableStore>,
    ) -> Self {
        Self {
            steps,
            states,
            inputs,
            params,
            outputs,
            state,
            store,
        }
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
