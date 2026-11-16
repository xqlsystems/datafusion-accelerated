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
use datafusion::sql::sqlparser::ast::{
    self, Visit, Visitor, visit_expressions, visit_expressions_mut,
};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::parser::Parser;

use crate::handle_table::HandleTable;
use crate::lowered_program::{LoweredStep, Params, SessionRead, Target};
use crate::program_tables::{ProgramTables, layered};
use crate::table_store::TableStore;
use crate::{Accelerator, Error, LoweredProgram, NameKind, TableHandle};

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
///
/// A parameter may go wherever a value is read per row or per query: in
/// `SELECT`, `WHERE`, `JOIN ... ON`, `GROUP BY`, `HAVING`, `ORDER BY`,
/// `CASE`, `IN` and an aggregate's `FILTER`. It may not go where planning
/// needs its value: in `VALUES`, `LIMIT`, `OFFSET`, `FETCH` or a table
/// function's arguments (`Error::ParameterPosition`).
#[derive(Clone)]
#[must_use]
pub struct Program {
    ctx: SessionContext,
    // `None` for the one step of `Program::new`, named at `lower` after the
    // program's one state table.
    steps: Vec<(Option<String>, String)>,
    state: Vec<(String, StateSource)>,
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

    /// A new state table, with its initial rows. Its schema is the first
    /// batch's, so `rows` needs one batch, if only an empty one; for no
    /// rows, use [`state_with_schema`](Program::state_with_schema). `lower`
    /// registers it in the session, so SQL can read it between runs.
    pub fn state(mut self, name: impl Into<String>, rows: Vec<RecordBatch>) -> Self {
        self.state
            .push((name.into(), StateSource::New { schema: None, rows }));
        self
    }

    /// A new state table with `schema`, and initial rows, which may be
    /// none.
    pub fn state_with_schema(
        mut self,
        name: impl Into<String>,
        schema: SchemaRef,
        rows: Vec<RecordBatch>,
    ) -> Self {
        let schema = Some(schema);
        self.state
            .push((name.into(), StateSource::New { schema, rows }));
        self
    }

    /// A table the runtime already owns, such as one an earlier program
    /// trained, as a state table, starting from its current rows. Other
    /// programs and callers may still replace it: a run that finds it
    /// replaced is a conflict (design §5.4).
    pub fn state_from(mut self, name: impl Into<String>) -> Self {
        self.state.push((name.into(), StateSource::Existing));
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
    /// program's names are the same, a step, an input or a new state table
    /// has the name of a session table, the steps read each other in a
    /// cycle, a step isn't a `SELECT`, a parameter is where planning needs
    /// its value, or a state step's result doesn't fit its table's schema.
    /// A step that reads a name nothing defines fails with DataFusion's own
    /// error. Every error from planning a step names the step. On failure,
    /// nothing is registered in the session.
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
                    kind: NameKind::Output,
                }
                .into());
            }
        }

        // The program's own tables, in front of the session's.
        let tables = Arc::new(ProgramTables::default());
        let mut state_handles = Vec::with_capacity(states.len());
        let mut new_states = Vec::new();
        for (name, source) in states {
            let handle = match source {
                StateSource::New { schema, rows } => {
                    let schema = match (schema, rows.first()) {
                        (Some(schema), _) => schema,
                        (None, Some(first)) => first.schema(),
                        (None, None) => {
                            return plan_err!(
                                "the state table {name} needs one batch, if only an empty one, \
                                 for its schema, or use Program::state_with_schema"
                            );
                        }
                    };
                    let handle = TableHandle::new(Arc::clone(&store), quoted(&name), schema, rows)?;
                    new_states.push(handle.clone());
                    handle
                }
                StateSource::Existing => {
                    Accelerator::table(&self.ctx, &name).await?.ok_or_else(|| {
                        Error::UnknownName {
                            name: quoted(&name),
                            kind: NameKind::RuntimeTable,
                        }
                    })?
                }
            };
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

        // The program's own names, which the session's tables don't define.
        let mut own: HashSet<String> = step_names.clone();
        own.extend(state_names.iter().cloned());
        own.extend(input_handles.iter().map(|(name, _)| name.clone()));
        own.extend(params.iter().map(|p| p.handle.name().to_string()));

        // Each step, parsed, with its parameters rewritten, the steps it
        // reads, and the session's tables it reads.
        let options = &state.config().options().catalog;
        let (catalog, schema) = (
            options.default_catalog.as_str(),
            options.default_schema.as_str(),
        );
        let dialect = state.config().options().sql_parser.dialect;
        let mut parsed = Vec::with_capacity(steps.len());
        let mut session_reads: Vec<SessionRead> = Vec::new();
        for (name, sql) in &steps {
            let in_step = |error: DataFusionError| error.context(format!("in step {name}"));
            let mut statement = state.sql_to_statement(sql, &dialect).map_err(in_step)?;
            rewrite_params(&mut statement, name, params.as_ref()).map_err(in_step)?;
            let mut reads = Vec::new();
            for reference in state
                .resolve_table_references(&statement)
                .map_err(in_step)?
            {
                let resolved = reference.clone().resolve(catalog, schema);
                let table = resolved.table.to_string();
                let default = *resolved.catalog == *catalog && *resolved.schema == *schema;
                if default && own.contains(&table) {
                    if step_names.contains(&table) && !state_names.contains(&table) {
                        reads.push(table);
                    }
                    continue;
                }
                let quoted = reference.to_quoted_string();
                if session_reads.iter().any(|read| read.name == quoted) {
                    continue;
                }
                // A name nothing defines fails below, with DataFusion's own
                // error.
                if let Ok(schema) = base.schema_for_ref(reference.clone())
                    && let Some(provider) = schema.table(&table).await?
                {
                    session_reads.push(SessionRead {
                        name: quoted,
                        reference,
                        provider,
                    });
                }
            }
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
            let in_step = |error: DataFusionError| error.context(format!("in step {name}"));
            let logical = state.statement_to_plan(statement).await.map_err(in_step)?;
            let plan = state
                .create_physical_plan(&logical)
                .await
                .map_err(in_step)?;
            let schema = plan.schema();
            let target = match state_handles.iter().position(|(state, _)| *state == name) {
                Some(index) => {
                    let expected = state_handles[index].1.schema();
                    if !fits(&expected, &schema) {
                        return Err(Error::StateSchema {
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
        for handle in new_states {
            let reference = TableReference::from(handle.name());
            self.ctx.register_table(
                reference.clone(),
                Arc::new(HandleTable::new(handle.clone())),
            )?;
            handle.set_home(&base, reference);
        }
        Ok(LoweredProgram {
            steps: lowered,
            states: state_handles,
            inputs: input_handles,
            params,
            outputs,
            session_reads,
            state,
            session: base,
            store,
        })
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

/// Where a state table's initial value comes from.
#[derive(Clone)]
enum StateSource {
    /// A new table, of `schema` or else its first batch's.
    New {
        schema: Option<SchemaRef>,
        rows: Vec<RecordBatch>,
    },
    /// A table the runtime owns already, of the same name.
    Existing,
}

/// Whether a step's result, of schema `found`, can be the next value of a
/// state table of schema `expected`: the same fields, by name and type. A
/// field may be nullable in `found` and not in `expected`, since a
/// parameter or an outer join makes a field nullable that never holds a
/// null; each run checks.
fn fits(expected: &SchemaRef, found: &SchemaRef) -> bool {
    expected.fields().len() == found.fields().len()
        && expected
            .fields()
            .iter()
            .zip(found.fields())
            .all(|(want, got)| want.name() == got.name() && want.data_type() == got.data_type())
}

/// The steps, with `Program::new`'s step named after the one state table.
fn name_steps(
    steps: Vec<(Option<String>, String)>,
    states: &[(String, StateSource)],
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

/// Fails on two of the program's names that are the same, and on a step,
/// an input or a new state table named like a session table. A step named
/// after a state table is that table's next value, so isn't a clash.
fn check_names(
    ctx: &SessionContext,
    states: &[(String, StateSource)],
    inputs: &[(String, SchemaRef)],
    params: &[(String, ScalarValue)],
    steps: &[(String, String)],
) -> Result<()> {
    let mut seen: HashMap<&str, NameKind> = HashMap::new();
    let declared = states
        .iter()
        .map(|(name, _)| (name.as_str(), NameKind::StateTable))
        .chain(
            inputs
                .iter()
                .map(|(name, _)| (name.as_str(), NameKind::Input)),
        )
        .chain(
            params
                .iter()
                .map(|(name, _)| (name.as_str(), NameKind::Parameter)),
        );
    for (name, kind) in declared {
        if let Some(first) = seen.insert(name, kind) {
            return Err(clash(name, first, kind));
        }
    }
    for (name, _) in steps {
        match seen.get(name.as_str()) {
            // The step that computes a state table's next value.
            Some(NameKind::StateTable) => continue,
            Some(first) => return Err(clash(name, *first, NameKind::Step)),
            None => {}
        }
        seen.insert(name, NameKind::Step);
    }
    let existing: HashSet<&str> = states
        .iter()
        .filter(|(_, source)| matches!(source, StateSource::Existing))
        .map(|(name, _)| name.as_str())
        .collect();
    for (name, kind) in seen {
        if kind == NameKind::Parameter
            || existing.contains(name)
            || !ctx.table_exist(TableReference::bare(name))?
        {
            continue;
        }
        if kind == NameKind::StateTable {
            return Err(Error::NameTaken { name: quoted(name) }.into());
        }
        return Err(clash(name, kind, NameKind::SessionTable));
    }
    Ok(())
}

fn clash(name: &str, first: NameKind, second: NameKind) -> DataFusionError {
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
    if let ControlFlow::Break(position) = inner.visit(&mut PlannedPositions) {
        return Err(Error::ParameterPosition {
            step: quoted(step),
            position,
        }
        .into());
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
                kind: NameKind::Parameter,
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

/// Finds a parameter where planning needs its value, which a scalar
/// subquery can't give: there DataFusion either fails to plan, or plans a
/// step that fails on every run.
struct PlannedPositions;

impl Visitor for PlannedPositions {
    type Break = &'static str;

    fn pre_visit_query(&mut self, query: &ast::Query) -> ControlFlow<&'static str> {
        if let Some(limit) = &query.limit_clause {
            let (limit, offset) = match limit {
                ast::LimitClause::LimitOffset { limit, offset, .. } => {
                    (limit.as_ref(), offset.as_ref().map(|offset| &offset.value))
                }
                ast::LimitClause::OffsetCommaLimit { offset, limit } => (Some(limit), Some(offset)),
            };
            if limit.is_some_and(has_placeholder) {
                return ControlFlow::Break("LIMIT");
            }
            if offset.is_some_and(has_placeholder) {
                return ControlFlow::Break("OFFSET");
            }
        }
        if query.fetch.as_ref().is_some_and(has_placeholder) {
            return ControlFlow::Break("FETCH");
        }
        if values_have_placeholder(&query.body) {
            return ControlFlow::Break("VALUES");
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_table_factor(&mut self, factor: &ast::TableFactor) -> ControlFlow<&'static str> {
        let arguments = match factor {
            ast::TableFactor::Table { args: Some(_), .. }
            | ast::TableFactor::Function { .. }
            | ast::TableFactor::TableFunction { .. } => has_placeholder(factor),
            _ => false,
        };
        if arguments {
            return ControlFlow::Break("a table function's arguments");
        }
        ControlFlow::Continue(())
    }
}

/// Whether a `VALUES` list in `body`, outside any subquery, has a
/// parameter. Subqueries are visited as queries of their own.
fn values_have_placeholder(body: &ast::SetExpr) -> bool {
    match body {
        ast::SetExpr::Values(values) => has_placeholder(values),
        ast::SetExpr::SetOperation { left, right, .. } => {
            values_have_placeholder(left) || values_have_placeholder(right)
        }
        _ => false,
    }
}

fn has_placeholder<V: Visit>(node: &V) -> bool {
    visit_expressions(node, |expr| match expr {
        ast::Expr::Value(value) if matches!(value.value, ast::Value::Placeholder(_)) => {
            ControlFlow::Break(())
        }
        _ => ControlFlow::Continue(()),
    })
    .is_break()
}
