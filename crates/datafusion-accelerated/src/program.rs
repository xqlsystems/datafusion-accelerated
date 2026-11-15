// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A program of steps, before it is lowered (design §5.4).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::ControlFlow;
use std::sync::Arc;

use datafusion::arrow::array::ArrayRef;
use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{DataFusionError, Result, ScalarValue, TableReference, plan_err};
use datafusion::prelude::SessionContext;
use datafusion::sql::parser::Statement;
use datafusion::sql::sqlparser::ast::{self, visit_expressions_mut};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::parser::Parser;

use crate::handle_table::HandleTable;
use crate::lowered_program::{LoweredStep, Params, Target};
use crate::program_tables::{ProgramTables, layered};
use crate::table_store::TableStore;
use crate::{Accelerator, Error, LoweredProgram, TableHandle};

/// A program: steps, each a named `SELECT`, with the tables and parameters
/// they read (design §5.4).
///
/// It names:
/// - its **state**: tables kept across runs. Wherever a step reads a state
///   table, it reads the table's value when the run began, and the step
///   named after the table computes its next value, as a loop's carry does
///   in JAX. So no step can read the step named after a state table: a
///   value computed from the next state is the next run's to compute.
/// - its **inputs**: tables supplied on every run;
/// - its **parameters**: scalars such as a learning rate, written `$lr` in
///   the steps' SQL, which change between runs without planning again;
/// - its **outputs**: the steps and state tables the caller reads after a
///   run.
///
/// A step reads other steps by their names, and the order of the steps
/// comes from those reads. No step's result is seen by any step that
/// doesn't read it, and none is written to the session.
#[derive(Clone)]
#[must_use]
pub struct Program {
    ctx: SessionContext,
    // `None` for the one step of `Program::new`, named at `lower` after the
    // program's one state table.
    steps: Vec<(Option<String>, String)>,
    state: Vec<(String, Vec<RecordBatch>)>,
    inputs: Vec<(String, SchemaRef)>,
    params: Vec<(String, ScalarValue)>,
    outputs: Vec<String>,
}

// By hand, since `SessionContext` has no `Debug`.
impl fmt::Debug for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Program")
            .field("steps", &self.steps)
            .field(
                "state",
                &self.state.iter().map(|(name, _)| name).collect::<Vec<_>>(),
            )
            .field("inputs", &self.inputs)
            .field("params", &self.params)
            .field("outputs", &self.outputs)
            .finish_non_exhaustive()
    }
}

impl Program {
    /// A program of one step, `sql`, whose result is the next value of the
    /// program's one state table.
    pub fn new(ctx: &SessionContext, sql: impl Into<String>) -> Self {
        Self::with_steps(ctx, vec![(None, sql.into())])
    }

    /// A program of named steps.
    pub fn from_steps<N, S>(ctx: &SessionContext, steps: impl IntoIterator<Item = (N, S)>) -> Self
    where
        N: Into<String>,
        S: Into<String>,
    {
        let steps = steps
            .into_iter()
            .map(|(name, sql)| (Some(name.into()), sql.into()))
            .collect();
        Self::with_steps(ctx, steps)
    }

    fn with_steps(ctx: &SessionContext, steps: Vec<(Option<String>, String)>) -> Self {
        Self {
            ctx: ctx.clone(),
            steps,
            state: Vec::new(),
            inputs: Vec::new(),
            params: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// A state table, with its initial rows. Its schema is the first
    /// batch's, so `rows` needs one batch, if only an empty one. `lower`
    /// registers it in the session, so SQL can read it between runs.
    pub fn state(mut self, name: impl Into<String>, rows: Vec<RecordBatch>) -> Self {
        self.state.push((name.into(), rows));
        self
    }

    /// An input, whose rows each run is given.
    pub fn input(mut self, name: impl Into<String>, schema: SchemaRef) -> Self {
        self.inputs.push((name.into(), schema));
        self
    }

    /// A parameter, `$name` in the steps' SQL, with its initial value. Its
    /// type is the value's.
    pub fn param(mut self, name: impl Into<String>, value: ScalarValue) -> Self {
        self.params.push((name.into(), value));
        self
    }

    /// An output: a step or a state table the caller reads after each run.
    pub fn output(mut self, name: impl Into<String>) -> Self {
        self.outputs.push(name.into());
        self
    }

    /// Plans every step, in the order their reads give.
    ///
    /// Fails if the session wasn't built with the runtime, two of the
    /// program's names are the same, a step or an input has the name of a
    /// session table, the steps read each other in a cycle, a step isn't a
    /// `SELECT`, or a state step's result doesn't fit its table's schema.
    /// A step that reads a name nothing defines fails with DataFusion's own
    /// error. On failure, nothing is registered in the session.
    pub async fn lower(self) -> Result<LoweredProgram> {
        let base = self.ctx.state();
        let store = Arc::clone(Accelerator::of(&base)?.store());

        let states = self
            .state
            .into_iter()
            .map(|(name, rows)| Ok((bare(&name)?, rows)))
            .collect::<Result<Vec<_>>>()?;
        let inputs = self
            .inputs
            .into_iter()
            .map(|(name, schema)| Ok((bare(&name)?, schema)))
            .collect::<Result<Vec<_>>>()?;
        let params: Vec<(String, ScalarValue)> = self
            .params
            .into_iter()
            .map(|(name, value)| (name.trim_start_matches('$').to_string(), value))
            .collect();
        let steps = name_steps(self.steps, &states)?;
        let outputs = self
            .outputs
            .iter()
            .map(|name| bare(name))
            .collect::<Result<Vec<_>>>()?;

        let state_names: HashSet<String> = states.iter().map(|(name, _)| name.clone()).collect();
        check_names(&self.ctx, &states, &inputs, &params, &steps)?;
        let step_names: HashSet<String> = steps.iter().map(|(name, _)| name.clone()).collect();
        for output in &outputs {
            if !step_names.contains(output.as_str()) && !state_names.contains(output.as_str()) {
                return Err(Error::UnknownName {
                    name: output.clone(),
                    kind: "step or state table",
                }
                .into());
            }
        }

        // The program's own tables, in front of the session's.
        let tables = Arc::new(ProgramTables::default());
        let mut state_handles = Vec::with_capacity(states.len());
        for (name, rows) in states {
            let Some(first) = rows.first() else {
                return plan_err!(
                    "the state table {name} needs one batch, if only an empty one, for its schema"
                );
            };
            let schema = first.schema();
            let handle = TableHandle::new(Arc::clone(&store), quoted(&name), schema, rows)?;
            tables.insert(name.clone(), Arc::new(HandleTable::new(handle.clone())));
            state_handles.push((name, handle));
        }
        let mut input_handles = Vec::with_capacity(inputs.len());
        for (name, schema) in inputs {
            let handle =
                TableHandle::new_private(Arc::clone(&store), quoted(&name), schema, vec![])?;
            tables.insert(name.clone(), Arc::new(HandleTable::new(handle.clone())));
            input_handles.push((name, handle));
        }
        let params = if params.is_empty() {
            None
        } else {
            let handle = params_table(&store, &params)?;
            tables.insert(
                handle.name().to_string(),
                Arc::new(HandleTable::new(handle.clone())),
            );
            Some(Params {
                handle,
                values: params,
            })
        };
        let state = layered(&base, Arc::clone(&tables))?;

        // Each step, parsed, with its parameters rewritten and the steps it
        // reads.
        let dialect = state.config().options().sql_parser.dialect;
        let mut parsed = Vec::with_capacity(steps.len());
        for (name, sql) in &steps {
            let mut statement = state.sql_to_statement(sql, &dialect)?;
            rewrite_params(&mut statement, name, params.as_ref())?;
            let reads: Vec<String> = state
                .resolve_table_references(&statement)?
                .into_iter()
                .filter(|reference| reference.schema().is_none())
                .map(|reference| reference.table().to_string())
                .filter(|read| {
                    step_names.contains(read.as_str()) && !state_names.contains(read.as_str())
                })
                .collect();
            parsed.push(ParsedStep {
                name: name.clone(),
                sql: sql.clone(),
                statement,
                reads,
            });
        }

        let mut lowered = Vec::with_capacity(parsed.len());
        for ParsedStep {
            name,
            sql,
            statement,
            reads,
        } in order(parsed)?
        {
            let logical = state.statement_to_plan(statement).await?;
            let plan = state.create_physical_plan(&logical).await?;
            let schema = plan.schema();
            let target = match state_handles.iter().position(|(state, _)| *state == name) {
                Some(index) => {
                    let expected = state_handles[index].1.schema();
                    if !expected.contains(&schema) {
                        return Err(Error::SchemaMismatch {
                            table: quoted(&name),
                            expected,
                            found: schema,
                        }
                        .into());
                    }
                    Target::State(index)
                }
                None => {
                    let handle = TableHandle::new_private(
                        Arc::clone(&store),
                        quoted(&name),
                        schema,
                        vec![],
                    )?;
                    tables.insert(name.clone(), Arc::new(HandleTable::new(handle.clone())));
                    Target::Step(handle)
                }
            };
            lowered.push(LoweredStep {
                name,
                sql,
                reads,
                plan,
                target,
            });
        }

        // Only now, so a failure above registers nothing.
        for (name, handle) in &state_handles {
            self.ctx.register_table(
                TableReference::bare(name.clone()),
                Arc::new(HandleTable::new(handle.clone())),
            )?;
        }
        Ok(LoweredProgram::new(
            lowered,
            state_handles,
            input_handles,
            params,
            outputs,
            state,
            store,
        ))
    }
}

/// A name as SQL resolves it, which must be a bare table name.
fn bare(name: &str) -> Result<String> {
    match TableReference::from(name) {
        TableReference::Bare { table } => Ok(table.to_string()),
        _ => plan_err!("a program's names are bare table names, not {name}"),
    }
}

/// A resolved name as SQL writes it.
fn quoted(name: &str) -> String {
    TableReference::bare(name).to_quoted_string()
}

/// The steps, with `Program::new`'s step named after the one state table.
fn name_steps(
    steps: Vec<(Option<String>, String)>,
    states: &[(String, Vec<RecordBatch>)],
) -> Result<Vec<(String, String)>> {
    steps
        .into_iter()
        .map(|(name, sql)| match name {
            Some(name) => Ok((bare(&name)?, sql)),
            None if states.len() == 1 => Ok((states[0].0.clone(), sql)),
            None => plan_err!(
                "Program::new's one step is the next value of the program's one state table, \
                 and the program has {} state tables",
                states.len()
            ),
        })
        .collect()
}

/// Fails on two of the program's names that are the same, and on a step or
/// an input named like a session table. A step named after a state table
/// is that table's next value, so isn't a clash.
fn check_names(
    ctx: &SessionContext,
    states: &[(String, Vec<RecordBatch>)],
    inputs: &[(String, SchemaRef)],
    params: &[(String, ScalarValue)],
    steps: &[(String, String)],
) -> Result<()> {
    let mut seen: HashMap<&str, &'static str> = HashMap::new();
    let declared = states
        .iter()
        .map(|(name, _)| (name.as_str(), "state table"))
        .chain(inputs.iter().map(|(name, _)| (name.as_str(), "input")))
        .chain(params.iter().map(|(name, _)| (name.as_str(), "parameter")));
    for (name, kind) in declared {
        if let Some(first) = seen.insert(name, kind) {
            return Err(clash(name, first, kind));
        }
    }
    for (name, _) in steps {
        match seen.get(name.as_str()) {
            // The step that computes a state table's next value.
            Some(&"state table") => continue,
            Some(first) => return Err(clash(name, first, "step")),
            None => {}
        }
        seen.insert(name, "step");
    }
    for (name, kind) in seen {
        if kind == "parameter" || !ctx.table_exist(TableReference::bare(name))? {
            continue;
        }
        if kind == "state table" {
            return Err(Error::NameTaken { name: quoted(name) }.into());
        }
        return Err(clash(name, kind, "session table"));
    }
    Ok(())
}

fn clash(name: &str, first: &'static str, second: &'static str) -> DataFusionError {
    Error::NameClash {
        name: quoted(name),
        first,
        second,
    }
    .into()
}

/// A step, parsed, with the other steps it reads.
struct ParsedStep {
    name: String,
    sql: String,
    statement: Statement,
    reads: Vec<String>,
}

/// The steps in an order in which each comes after the steps it reads,
/// keeping the listed order where the reads allow. Fails on a cycle.
fn order(steps: Vec<ParsedStep>) -> Result<Vec<ParsedStep>> {
    let mut remaining = steps;
    let mut done: HashSet<String> = HashSet::new();
    let mut ordered = Vec::with_capacity(remaining.len());
    while !remaining.is_empty() {
        let Some(ready) = remaining
            .iter()
            .position(|step| step.reads.iter().all(|read| done.contains(read)))
        else {
            return Err(Error::Cycle {
                steps: remaining.into_iter().map(|step| step.name).collect(),
            }
            .into());
        };
        let step = remaining.remove(ready);
        done.insert(step.name.clone());
        ordered.push(step);
    }
    Ok(ordered)
}

/// The one-row table of the program's parameters, one column each, which
/// the steps read in place of their placeholders (supplement S3.2).
fn params_table(store: &Arc<TableStore>, params: &[(String, ScalarValue)]) -> Result<TableHandle> {
    let fields: Vec<Field> = params
        .iter()
        .map(|(name, value)| Field::new(name, value.data_type(), true))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let columns = params
        .iter()
        .map(|(_, value)| value.to_array())
        .collect::<Result<Vec<ArrayRef>>>()?;
    let row = RecordBatch::try_new(Arc::clone(&schema), columns)?;
    let name = format!("accelerated_params_{}", uuid::Uuid::new_v4().simple());
    TableHandle::new_private(Arc::clone(store), name, schema, vec![row])
}

/// Rewrites each `$name` in a step into a scalar subquery of the
/// parameters' table, so that a new value is read on each run without
/// planning again.
fn rewrite_params(statement: &mut Statement, step: &str, params: Option<&Params>) -> Result<()> {
    let Statement::Statement(inner) = statement else {
        return plan_err!("a program's step is a SELECT, and {step} isn't one");
    };
    if !matches!(inner.as_ref(), ast::Statement::Query(_)) {
        return plan_err!("a program's step is a SELECT, and {step} isn't one");
    }
    let flow = visit_expressions_mut(inner.as_mut(), |expr| {
        let ast::Expr::Value(value) = expr else {
            return ControlFlow::Continue(());
        };
        let ast::Value::Placeholder(placeholder) = &value.value else {
            return ControlFlow::Continue(());
        };
        let name = placeholder.trim_start_matches('$');
        let Some(params) = params.filter(|p| p.values.iter().any(|(param, _)| param == name))
        else {
            return ControlFlow::Break(DataFusionError::from(Error::UnknownName {
                name: placeholder.clone(),
                kind: "parameter",
            }));
        };
        let sql = format!(
            "SELECT {} FROM {}",
            quote_ident(name),
            quote_ident(params.handle.name())
        );
        match Parser::new(&GenericDialect {})
            .try_with_sql(&sql)
            .and_then(|mut parser| parser.parse_query())
        {
            Ok(query) => {
                *expr = ast::Expr::Subquery(query);
                ControlFlow::Continue(())
            }
            Err(error) => ControlFlow::Break(DataFusionError::from(error)),
        }
    });
    match flow {
        ControlFlow::Break(error) => Err(error),
        ControlFlow::Continue(()) => Ok(()),
    }
}

fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}
