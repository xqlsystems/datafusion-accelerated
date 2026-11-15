// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A program fixed to run.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::ArrayRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{Result, ScalarValue, internal_err};
use datafusion::physical_plan::collect;
use datafusion::physical_plan::execution_plan::reset_plan_states;
use futures::lock::Mutex;

use crate::lowered_program::Target;
use crate::table_handle::stamped;
use crate::{Error, LoweredProgram, Run, RunArgs, RunOutputs};

/// A program fixed to run: its steps' physical plans, which each run
/// resets and executes with no planning (spike 4a).
///
/// Runs of one compiled program execute in order: a second `run` waits for
/// the first to finish.
///
/// A compiled program doesn't notice when the session re-registers a table
/// it reads. The ddx integration, which brings ddx's session key, removes
/// that limit.
#[derive(Debug)]
pub struct CompiledProgram {
    program: LoweredProgram,
    runs: Mutex<()>,
}

impl CompiledProgram {
    pub(crate) fn new(program: LoweredProgram) -> Self {
        Self {
            program,
            runs: Mutex::new(()),
        }
    }

    /// The program as text, as [`LoweredProgram::explain`] gives it.
    #[must_use]
    pub fn explain(&self) -> String {
        self.program.explain()
    }

    /// Runs the program once: one snapshot of the session's tables, every
    /// step in order, then the next value of every state table published at
    /// once (design §5.4, D20).
    ///
    /// Fails, publishing nothing, if an input is missing, a name in `args`
    /// isn't the program's, a step fails, or a state table was replaced
    /// during the run and `args` doesn't say `OnConflict::Overwrite`.
    pub async fn run(&self, args: RunArgs) -> Result<RunOutputs> {
        let _turn = self.runs.lock().await;
        let program = &self.program;
        let inputs = self.inputs(args.inputs)?;
        let params = self.params(&args.params)?;

        let run = Run::begin(program.state.clone(), Arc::clone(&program.store))?;
        let store = run.store();
        for (handle, batches) in inputs {
            store.write_local(run.id(), handle.id(), batches)?;
        }
        if let (Some(declared), Some(row)) = (&program.params, params) {
            store.write_local(run.id(), declared.handle.id(), vec![row])?;
        }

        let mut results: HashMap<String, Arc<[RecordBatch]>> = HashMap::new();
        let mut writes = Vec::new();
        for step in &program.steps {
            let plan = reset_plan_states(Arc::clone(&step.plan))?;
            let batches = collect(plan, run.task_ctx()).await?;
            match &step.target {
                Target::Step(handle) => {
                    let batches = stamped(handle.name(), &handle.schema(), batches)?;
                    store.write_local(run.id(), handle.id(), batches.clone())?;
                    results.insert(step.name.clone(), batches.into());
                }
                Target::State(index) => {
                    let handle = &program.states[*index].1;
                    let batches = stamped(handle.name(), &handle.schema(), batches)?;
                    results.insert(step.name.clone(), batches.clone().into());
                    writes.push((handle.id(), batches));
                }
            }
        }

        let snapshot = run.snapshot();
        let publication = store.publish(run.id(), writes, args.on_conflict)?;
        let mut tables = HashMap::with_capacity(program.outputs.len());
        for name in &program.outputs {
            let rows = match results.get(name) {
                Some(rows) => Arc::clone(rows),
                // A state table no step computes keeps the value it had.
                None => {
                    let Some((_, handle)) = program.states.iter().find(|(state, _)| state == name)
                    else {
                        return internal_err!(
                            "the output {name} is neither a step nor a state table"
                        );
                    };
                    handle.read_in(Some(run.id()))?
                }
            };
            tables.insert(name.clone(), rows);
        }
        Ok(RunOutputs {
            tables,
            snapshot,
            published: publication.published,
            overwritten: publication.overwritten,
        })
    }

    /// Each input's handle with its rows for the run.
    fn inputs(
        &self,
        given: Vec<(String, Vec<RecordBatch>)>,
    ) -> Result<Vec<(crate::TableHandle, Vec<RecordBatch>)>> {
        let declared = &self.program.inputs;
        let mut rows: HashMap<String, Vec<RecordBatch>> = HashMap::new();
        for (name, batches) in given {
            let name = resolved(&name);
            if !declared.iter().any(|(input, _)| *input == name) {
                return Err(Error::UnknownName {
                    name,
                    kind: "input",
                }
                .into());
            }
            rows.insert(name, batches);
        }
        declared
            .iter()
            .map(|(name, handle)| {
                let batches = rows
                    .remove(name)
                    .ok_or_else(|| Error::MissingInput { name: name.clone() })?;
                Ok((
                    handle.clone(),
                    stamped(handle.name(), &handle.schema(), batches)?,
                ))
            })
            .collect()
    }

    /// The parameters' row for the run: each given value, cast to its
    /// parameter's type, and the initial value of the others.
    fn params(&self, given: &[(String, ScalarValue)]) -> Result<Option<RecordBatch>> {
        let Some(declared) = &self.program.params else {
            if let Some((name, _)) = given.first() {
                return Err(Error::UnknownName {
                    name: name.clone(),
                    kind: "parameter",
                }
                .into());
            }
            return Ok(None);
        };
        let mut values = declared.values.clone();
        for (name, value) in given {
            let name = name.trim_start_matches('$');
            let Some((_, slot)) = values.iter_mut().find(|(param, _)| param == name) else {
                return Err(Error::UnknownName {
                    name: name.to_string(),
                    kind: "parameter",
                }
                .into());
            };
            *slot = value.cast_to(&slot.data_type())?;
        }
        let columns = values
            .iter()
            .map(|(_, value)| value.to_array())
            .collect::<Result<Vec<ArrayRef>>>()?;
        Ok(Some(RecordBatch::try_new(
            declared.handle.schema(),
            columns,
        )?))
    }
}

fn resolved(name: &str) -> String {
    match datafusion::common::TableReference::from(name) {
        datafusion::common::TableReference::Bare { table } => table.to_string(),
        other => other.to_string(),
    }
}
