// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Steps given as DataFusion plans (#49): from a `DataFrame`, or built
//! with `LogicalPlanBuilder` over scans that hold only a schema, mixed
//! with steps in SQL.

use std::sync::Arc;

use datafusion::arrow::array::Int64Array;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{DataFusionError, ScalarValue};
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::logical_expr::{
    LogicalPlan, LogicalPlanBuilder, LogicalTableSource, col, in_subquery, lit, placeholder,
};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_accelerated::{
    AcceleratorExt, AcceleratorOptions, Error, NameKind, Program, RunArgs, Step,
};

fn session() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(4))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    SessionContext::new_with_state(state)
}

fn kv_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, true),
    ]))
}

fn kv(rows: &[(i64, i64)]) -> RecordBatch {
    let k: Int64Array = rows.iter().map(|(k, _)| *k).collect();
    let v: Int64Array = rows.iter().map(|(_, v)| *v).collect();
    RecordBatch::try_new(kv_schema(), vec![Arc::new(k), Arc::new(v)]).expect("the columns match")
}

/// The `(k, v)` rows of `batches`, sorted.
fn pairs(batches: &[RecordBatch]) -> Vec<(i64, i64)> {
    let mut rows = Vec::new();
    for batch in batches {
        let column = |name: &str| {
            batch
                .column_by_name(name)
                .expect("the column")
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64")
                .clone()
        };
        let (k, v) = (column("k"), column("v"));
        rows.extend((0..batch.num_rows()).map(|i| (k.value(i), v.value(i))));
    }
    rows.sort_unstable();
    rows
}

/// A scan of `name` that holds only a schema, as a plan built before the
/// program exists has.
fn scan(name: &str, schema: SchemaRef) -> Result<LogicalPlanBuilder> {
    LogicalPlanBuilder::scan(name, Arc::new(LogicalTableSource::new(schema)), None)
}

/// `batches`, of `kv_schema()`, as a table.
fn rows(batches: Vec<RecordBatch>) -> Result<Arc<MemTable>> {
    Ok(Arc::new(MemTable::try_new(kv_schema(), vec![batches])?))
}

fn found(err: &DataFusionError) -> Option<&Error> {
    Error::find(err)
}

/// A program of a plan built by hand, a `DataFrame` and a SQL step gives,
/// on each run, what the same steps give as plain SQL with the parameter
/// written as a literal (R1).
#[tokio::test]
async fn plans_and_sql_in_one_program_run_as_their_sql_does() -> Result<()> {
    let ctx = session();
    let x = kv(&[(0, 10), (1, 20), (2, 30)]);
    ctx.register_batch("x", x.clone())?;
    let scaled = scan("w", kv_schema())?
        .filter(col("v").gt(placeholder("$k")))?
        .project(vec![col("k"), (col("v") * lit(2_i64)).alias("v")])?
        .build()?;
    let big = ctx
        .table("x")
        .await?
        .filter(col("v").gt(placeholder("$k") * lit(10_i64)))?;
    const UPDATE: &str = "SELECT w.k, w.v + COALESCE(s.v, 0) + x.v AS v \
                          FROM w JOIN x ON w.k = x.k LEFT JOIN scaled s ON w.k = s.k";
    let w0 = kv(&[(0, 1), (1, 2), (2, 3)]);
    let lowered = Program::from_steps(
        &ctx,
        [
            ("scaled", Step::plan(scaled)),
            ("big", Step::from(big)),
            ("w", Step::sql(UPDATE)),
        ],
    )
    .state("w", vec![w0.clone()])
    .param("k", ScalarValue::from(0_i64))
    .output("scaled")
    .output("big")
    .output("w")
    .lower()
    .await?;
    assert!(lowered.explain().contains("logical plan:"));
    let compiled = lowered.compile()?;

    let plain = SessionContext::new();
    plain.register_batch("x", x)?;
    plain.register_batch("w", w0)?;
    for k in [0_i64, 2, 15] {
        let outputs = compiled
            .run(RunArgs::new().param("k", ScalarValue::from(k)))
            .await?;
        let sql = |sql: String| {
            let plain = plain.clone();
            async move { plain.sql(&sql).await?.collect().await }
        };
        let scaled = sql(format!("SELECT k, v * 2 AS v FROM w WHERE v > {k}")).await?;
        let big = sql(format!("SELECT k, v FROM x WHERE v > {k} * 10")).await?;
        plain.register_table("scaled", rows(scaled.clone())?)?;
        let w = sql(UPDATE.to_string()).await?;
        plain.deregister_table("scaled")?;
        plain.deregister_table("w")?;
        plain.register_table("w", rows(w.clone())?)?;

        assert_eq!(pairs(&outputs.table("scaled")?), pairs(&scaled), "k = {k}");
        assert_eq!(pairs(&outputs.table("big")?), pairs(&big), "k = {k}");
        assert_eq!(pairs(&outputs.table("w")?), pairs(&w), "k = {k}");
    }
    Ok(())
}

/// A scan that holds only a schema, of a name that isn't the program's,
/// reads the session's table, and is checked on each run as SQL's reads
/// are.
#[tokio::test]
async fn a_scan_with_only_a_schema_reads_the_sessions_table() -> Result<()> {
    let ctx = session();
    ctx.register_batch("x", kv(&[(0, 10)]))?;
    let plan = scan("x", kv_schema())?.build()?;
    let compiled = Program::from_steps(&ctx, [("out", plan)])
        .output("out")
        .lower()
        .await?
        .compile()?;
    let outputs = compiled.run(RunArgs::new()).await?;
    assert_eq!(pairs(&outputs.table("out")?), vec![(0, 10)]);

    ctx.deregister_table("x")?;
    ctx.register_batch("x", kv(&[(0, 20)]))?;
    let err = compiled.run(RunArgs::new()).await.unwrap_err();
    assert!(
        matches!(found(&err), Some(Error::Stale { table }) if table == "x"),
        "{err}"
    );
    Ok(())
}

/// A `DataFrame` built before its table was registered again holds the old
/// table, which SQL no longer reads, so `lower` refuses it.
#[tokio::test]
async fn a_plan_built_against_a_replaced_table_is_stale() -> Result<()> {
    let ctx = session();
    ctx.register_batch("x", kv(&[(0, 10)]))?;
    let frame = ctx.table("x").await?;
    ctx.deregister_table("x")?;
    ctx.register_batch("x", kv(&[(0, 20)]))?;
    let err = Program::from_steps(&ctx, [("out", frame)])
        .output("out")
        .lower()
        .await
        .unwrap_err();
    assert!(
        matches!(found(&err), Some(Error::Stale { table }) if table == "x"),
        "{err}"
    );
    Ok(())
}

/// A plan's scan of the program's table must have that table's fields.
#[tokio::test]
async fn a_scan_of_a_program_table_with_other_fields_fails() -> Result<()> {
    let ctx = session();
    let wrong = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Utf8, true),
    ]));
    let plan = scan("w", wrong)?.build()?;
    let err = Program::from_steps(&ctx, [("out", plan)])
        .state("w", vec![kv(&[(0, 1)])])
        .output("out")
        .lower()
        .await
        .unwrap_err();
    assert!(
        matches!(
            found(&err),
            Some(Error::ScanSchema { step, table, .. }) if step == "out" && table == "w"
        ),
        "{err}"
    );
    assert!(
        err.to_string().contains("the field v of type Utf8"),
        "{err}"
    );
    Ok(())
}

/// A parameter in a plan's subquery is read on each run, as one in SQL is.
#[tokio::test]
async fn a_parameter_in_a_plans_subquery_changes_per_run() -> Result<()> {
    let ctx = session();
    let inner = scan("w", kv_schema())?
        .alias("inner")?
        .filter(col("inner.v").gt(placeholder("$k")))?
        .project(vec![col("inner.k")])?
        .build()?;
    let plan = scan("w", kv_schema())?
        .filter(in_subquery(col("w.k"), Arc::new(inner)))?
        .build()?;
    let compiled = Program::from_steps(&ctx, [("out", plan)])
        .state("w", vec![kv(&[(0, 1), (1, 2), (2, 3)])])
        .param("k", ScalarValue::from(0_i64))
        .output("out")
        .lower()
        .await?
        .compile()?;
    for (k, want) in [(1, vec![(1, 2), (2, 3)]), (2, vec![(2, 3)])] {
        let outputs = compiled
            .run(RunArgs::new().param("k", ScalarValue::from(k)))
            .await?;
        assert_eq!(pairs(&outputs.table("out")?), want, "k = {k}");
    }
    Ok(())
}

/// A plan's parameter where planning needs its value, or one the program
/// doesn't have, fails at `lower`, as in SQL.
#[tokio::test]
async fn a_plans_parameter_where_planning_needs_it_or_unknown_fails() -> Result<()> {
    let ctx = session();
    let lower = |plan: LogicalPlan| {
        let ctx = ctx.clone();
        async move {
            Program::from_steps(&ctx, [("out", plan)])
                .state("w", vec![kv(&[(0, 1)])])
                .param("k", ScalarValue::from(1_i64))
                .output("out")
                .lower()
                .await
                .unwrap_err()
        }
    };
    let values = LogicalPlanBuilder::values(vec![vec![lit(0_i64), placeholder("$k")]])?.build()?;
    let limit = scan("w", kv_schema())?
        .limit_by_expr(None, Some(placeholder("$k")))?
        .build()?;
    for (plan, position) in [(values, "VALUES"), (limit, "LIMIT")] {
        let err = lower(plan).await;
        assert!(
            matches!(
                found(&err),
                Some(Error::ParameterPosition { step, position: p }) if step == "out" && *p == position
            ),
            "{position}: {err}"
        );
    }

    let unknown = scan("w", kv_schema())?
        .filter(col("v").gt(placeholder("$nope")))?
        .build()?;
    let err = lower(unknown).await;
    assert!(
        matches!(
            found(&err),
            Some(Error::UnknownName { name, kind: NameKind::Parameter }) if name == "$nope"
        ),
        "{err}"
    );
    Ok(())
}

/// A plan that writes, as an `INSERT` plans to, isn't a step.
#[tokio::test]
async fn a_plan_that_isnt_a_query_fails() -> Result<()> {
    let ctx = session();
    ctx.register_batch("x", kv(&[(0, 10)]))?;
    let insert = ctx
        .state()
        .create_logical_plan("INSERT INTO x VALUES (1, 20)")
        .await?;
    let err = Program::from_steps(&ctx, [("out", insert)])
        .output("out")
        .lower()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("isn't one"), "{err}");
    Ok(())
}
