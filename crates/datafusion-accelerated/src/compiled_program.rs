// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A program fixed to run.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef};
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{Result, ScalarValue, internal_err};
use datafusion::physical_plan::collect;
use datafusion::physical_plan::execution_plan::reset_plan_states;
use futures::lock::Mutex;

use crate::lowered_program::Target;
use crate::table_handle::stamped;
use crate::{Error, LoweredProgram, NameKind, Run, RunArgs, RunOutputs, TableHandle};

/// A program fixed to run: its steps' physical plans, which each run
/// resets and executes with no planning (spike 4a).
///
/// Runs of one compiled program execute in order: a second `run` waits for
/// the first to finish.
///
/// A plan holds the tables it was planned with. So each run first checks
/// that the session still gives every name the program reads, and each of
/// its state tables, the table `lower` found, and fails with
/// `Error::Stale` if not: after `DROP TABLE`, or a table registered again,
/// lower the program again.
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
    /// isn't the program's or is given twice, a parameter's value doesn't
    /// fit its type, a table the program reads is stale, a step fails, a
    /// state table's next value has a null its schema forbids, or a state
    /// table was replaced during the run and `args` doesn't say
    /// `OnConflict::Overwrite`.
    ///
    /// The run starts when the future is first polled, as for any `async
    /// fn` in Rust. To run it while doing other work, spawn it.
    pub async fn run(&self, args: RunArgs) -> Result<RunOutputs> {
        let _turn = self.runs.lock().await;
        let program = &self.program;
        let inputs = self.inputs(args.inputs)?;
        let params = self.params(&args.params)?;
        program.check_fresh().await?;

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
                    let batches = as_state(handle, batches)?;
                    results.insert(step.name.clone(), batches.clone().into());
                    writes.push((handle.id(), batches));
                }
            }
        }

        // A state table dropped during the run would take rows no name
        // reads. The catalog isn't under the store's lock, so a drop between
        // this check and the publish still goes unseen (design §5.4).
        for (_, handle) in &program.states {
            let written = writes.iter().any(|(id, _)| *id == handle.id());
            if written && !handle.is_attached()? {
                return Err(Error::Stale {
                    table: handle.name().to_string(),
                }
                .into());
            }
        }
        let snapshot = store.snapshot_of(run.id(), &program.snapshot_ids())?;
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
    ) -> Result<Vec<(TableHandle, Vec<RecordBatch>)>> {
        let declared = &self.program.inputs;
        let mut rows: HashMap<String, Vec<RecordBatch>> = HashMap::new();
        for (name, batches) in given {
            let name = resolved(&name);
            if !declared.iter().any(|(input, _)| *input == name) {
                return Err(Error::UnknownName {
                    name,
                    kind: NameKind::Input,
                }
                .into());
            }
            if rows.contains_key(&name) {
                return Err(Error::GivenTwice {
                    name,
                    kind: NameKind::Input,
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

    /// The parameters' row for the run: each given value, of its
    /// parameter's type, and the initial value of the others.
    fn params(&self, given: &[(String, ScalarValue)]) -> Result<Option<RecordBatch>> {
        let Some(declared) = &self.program.params else {
            if let Some((name, _)) = given.first() {
                return Err(Error::UnknownName {
                    name: name.clone(),
                    kind: NameKind::Parameter,
                }
                .into());
            }
            return Ok(None);
        };
        let mut values = declared.values.clone();
        let mut seen = Vec::with_capacity(given.len());
        for (name, value) in given {
            let name = name.trim_start_matches('$');
            let Some((_, slot)) = values.iter_mut().find(|(param, _)| param == name) else {
                return Err(Error::UnknownName {
                    name: name.to_string(),
                    kind: NameKind::Parameter,
                }
                .into());
            };
            if seen.contains(&name) {
                return Err(Error::GivenTwice {
                    name: name.to_string(),
                    kind: NameKind::Parameter,
                }
                .into());
            }
            seen.push(name);
            *slot = lossless(name, value, &slot.data_type())?;
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

/// `value` as `to`: unchanged if it's of that type, or a number cast to
/// another number type that casts back to the same value. Text is never
/// read as a number, nor a number as text (R10).
fn lossless(name: &str, value: &ScalarValue, to: &DataType) -> Result<ScalarValue> {
    let from = value.data_type();
    if from == *to {
        return Ok(value.clone());
    }
    let refused = || Error::ParameterType {
        name: name.to_string(),
        expected: to.clone(),
        found: from.clone(),
    };
    if !from.is_numeric() || !to.is_numeric() {
        return Err(refused().into());
    }
    let cast = value.cast_to(to).map_err(|_| refused())?;
    match cast.cast_to(&from) {
        Ok(back) if back == *value => Ok(cast),
        _ => Err(refused().into()),
    }
}

/// A step's result as the next value of the state table `handle`, with the
/// table's schema. A field the table says isn't nullable must have no
/// nulls, though the step's may be nullable (`fits` in `program.rs`).
fn as_state(handle: &TableHandle, batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
    let schema = handle.schema();
    batches
        .into_iter()
        .map(|batch| {
            for (field, column) in schema.fields().iter().zip(batch.columns()) {
                if !field.is_nullable() && column.null_count() > 0 {
                    return Err(Error::NullInState {
                        table: handle.name().to_string(),
                        field: field.name().clone(),
                    }
                    .into());
                }
            }
            Ok(RecordBatch::try_new(
                Arc::clone(&schema),
                batch.columns().to_vec(),
            )?)
        })
        .collect()
}

fn resolved(name: &str) -> String {
    match datafusion::common::TableReference::from(name) {
        datafusion::common::TableReference::Bare { table } => table.to_string(),
        other => other.to_string(),
    }
}
