// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Programs: steps as named `SELECT`s, runs, versions and publication.

use std::sync::Arc;

use datafusion::arrow::array::{Float64Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::logical_expr::{ColumnarValue, Volatility, create_udf};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_accelerated::{
    Accelerator, AcceleratorExt, AcceleratorOptions, Error, NameKind, OnConflict, Program, RunArgs,
};

fn session() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(4))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    SessionContext::new_with_state(state)
}

fn weights_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, false),
        Field::new("v", DataType::Float64, true),
    ]))
}

fn data_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, false),
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
    ]))
}

fn weights(values: &[f64]) -> RecordBatch {
    let i: Int64Array = (0..values.len() as i64).collect();
    RecordBatch::try_new(
        weights_schema(),
        vec![Arc::new(i), Arc::new(Float64Array::from(values.to_vec()))],
    )
    .expect("the columns match")
}

fn data(step: usize) -> RecordBatch {
    let n = 8;
    let i: Int64Array = (0..n).map(|k| k % 2).collect();
    let x: Float64Array = (0..n).map(|k| (k + step as i64) as f64 / 4.0).collect();
    let y: Float64Array = (0..n).map(|k| ((k * 3 + step as i64) % 5) as f64).collect();
    RecordBatch::try_new(data_schema(), vec![Arc::new(i), Arc::new(x), Arc::new(y)])
        .expect("the columns match")
}

/// The values of column `v`, ordered by `i`.
fn by_index(batches: &[RecordBatch]) -> Vec<f64> {
    let mut rows = Vec::new();
    for batch in batches {
        let i = batch
            .column_by_name("i")
            .expect("an i column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64");
        let v = batch
            .column_by_name("v")
            .expect("a v column")
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Float64");
        for row in 0..batch.num_rows() {
            rows.push((i.value(row), v.value(row)));
        }
    }
    rows.sort_by_key(|(i, _)| *i);
    rows.into_iter().map(|(_, v)| v).collect()
}

fn scalar(batches: &[RecordBatch]) -> f64 {
    let column = batches
        .iter()
        .find(|batch| batch.num_rows() > 0)
        .expect("a row")
        .column(0);
    column
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64")
        .value(0)
}

/// Prediction, loss, gradient and update, for a linear model per index.
const PRED: &str = "SELECT d.i, d.x * w.v AS p, d.y FROM d JOIN w ON d.i = w.i";
const LOSS: &str = "SELECT AVG((p - y) * (p - y)) AS loss FROM pred";
const UPDATE: &str = "SELECT w.i, w.v - $lr * grad.g AS v FROM w JOIN grad ON w.i = grad.i";

/// Each index's mean of 2 (x v - y) x, the gradient of the squared error.
const GRAD_SIMPLE: &str = "SELECT d.i, AVG(2 * (d.x * w.v - d.y) * d.x) AS g \
                           FROM d JOIN w ON d.i = w.i GROUP BY d.i";

fn gradient_program(ctx: &SessionContext) -> Program {
    Program::from_steps(
        ctx,
        [
            ("w", UPDATE),
            ("pred", PRED),
            ("loss", LOSS),
            ("grad", GRAD_SIMPLE),
        ],
    )
    .state("w", vec![weights(&[0.5, -0.25])])
    .input("d", data_schema())
    .param("lr", ScalarValue::from(0.05))
    .output("loss")
    .output("w")
}

/// The same loop as plain SQL, with plain tables, on a session without the
/// runtime: the reference (R1).
async fn reference_step(
    w: &[RecordBatch],
    d: RecordBatch,
    lr: f64,
) -> Result<(f64, Vec<RecordBatch>)> {
    let ctx = SessionContext::new();
    ctx.register_table(
        "w",
        Arc::new(MemTable::try_new(weights_schema(), vec![w.to_vec()])?),
    )?;
    ctx.register_table(
        "d",
        Arc::new(MemTable::try_new(data_schema(), vec![vec![d]])?),
    )?;
    let pred = ctx.sql(PRED).await?.collect().await?;
    let schema = pred[0].schema();
    ctx.register_table("pred", Arc::new(MemTable::try_new(schema, vec![pred])?))?;
    let loss = scalar(&ctx.sql(LOSS).await?.collect().await?);
    let grad = ctx.sql(GRAD_SIMPLE).await?.collect().await?;
    let schema = grad[0].schema();
    ctx.register_table("grad", Arc::new(MemTable::try_new(schema, vec![grad])?))?;
    let next = ctx
        .sql(&UPDATE.replace("$lr", &format!("CAST({lr} AS DOUBLE)")))
        .await?
        .collect()
        .await?;
    Ok((loss, next))
}

#[tokio::test]
async fn fifty_runs_match_the_loop_as_plain_sql() -> Result<()> {
    let ctx = session();
    let program = gradient_program(&ctx).lower().await?.compile()?;
    let mut w = vec![weights(&[0.5, -0.25])];
    for step in 0..50 {
        let lr = if step < 25 { 0.05 } else { 0.02 };
        let outputs = program
            .run(
                RunArgs::new()
                    .input("d", vec![data(step)])
                    .param("lr", ScalarValue::from(lr)),
            )
            .await?;
        let (loss, next) = reference_step(&w, data(step), lr).await?;
        let got = scalar(&outputs.table("loss")?);
        // Relative, since the loss grows large and the plans may sum in
        // different orders.
        assert!(
            (got - loss).abs() <= 1e-12 * loss.abs().max(1.0),
            "run {step}: loss {got}, plain SQL {loss}"
        );
        let got = by_index(&outputs.table("w")?);
        let expected = by_index(&next);
        assert_eq!(
            got.len(),
            expected.len(),
            "run {step}: one weight per index"
        );
        for (a, b) in got.iter().zip(&expected) {
            assert!(
                (a - b).abs() < 1e-9,
                "run {step}: weights {got:?}, plain SQL {expected:?}"
            );
        }
        w = next;
    }
    let session_w = ctx.sql("SELECT i, v FROM w").await?.collect().await?;
    assert_eq!(
        by_index(&session_w),
        by_index(&w),
        "the session reads the published state"
    );
    Ok(())
}

#[tokio::test]
async fn steps_are_ordered_by_their_reads() -> Result<()> {
    let ctx = session();
    let lowered = gradient_program(&ctx).lower().await?;
    let text = lowered.explain();
    let position = |step: &str| {
        text.lines()
            .position(|line| {
                line.starts_with("step ")
                    && line.split_once(": ").is_some_and(|(_, rest)| {
                        rest.starts_with(&format!("{step},"))
                            || rest.starts_with(&format!("{step} ("))
                    })
            })
            .unwrap_or_else(|| panic!("explain lists {step}:\n{text}"))
    };
    assert!(
        position("pred") < position("loss") && position("grad") < position("w"),
        "each step comes after the steps it reads, whatever the listed order:\n{text}"
    );
    assert!(
        text.contains("$lr = 0.05"),
        "explain shows the parameters:\n{text}"
    );
    assert!(
        text.contains("HandleScanExec"),
        "explain shows the plans:\n{text}"
    );
    Ok(())
}

#[tokio::test]
async fn a_state_table_means_its_value_when_the_run_began() -> Result<()> {
    let ctx = session();
    let program = Program::from_steps(
        &ctx,
        [
            ("w", "SELECT i, v + 1 AS v FROM w"),
            ("seen", "SELECT i, v FROM w"),
        ],
    )
    .state("w", vec![weights(&[1.0])])
    .output("seen")
    .output("w")
    .lower()
    .await?
    .compile()?;
    let outputs = program.run(RunArgs::new()).await?;
    assert_eq!(
        by_index(&outputs.table("seen")?),
        vec![1.0],
        "seen reads w as the run began, not the step named w"
    );
    assert_eq!(by_index(&outputs.table("w")?), vec![2.0], "w's next value");
    Ok(())
}

#[tokio::test]
async fn the_one_statement_form_updates_the_one_state_table() -> Result<()> {
    let ctx = session();
    let program = Program::new(&ctx, "SELECT i, v * $k AS v FROM w")
        .state("w", vec![weights(&[1.0, 2.0])])
        .param("k", ScalarValue::from(3.0))
        .output("w")
        .lower()
        .await?
        .compile()?;
    program.run(RunArgs::new()).await?;
    let outputs = program
        .run(RunArgs::new().param("k", ScalarValue::from(0.5)))
        .await?;
    assert_eq!(
        by_index(&outputs.table("w")?),
        vec![1.5, 3.0],
        "times 3, then a new parameter without planning again: times 0.5"
    );
    Ok(())
}

#[tokio::test]
async fn versions_count_replaces_and_publishes_and_not_failures() -> Result<()> {
    let ctx = session();
    let program = Program::new(&ctx, "SELECT i, v / CAST($d AS DOUBLE) AS v FROM w")
        .state("w", vec![weights(&[1.0])])
        .param("d", ScalarValue::from("1"))
        .lower()
        .await?
        .compile()?;
    let w = Accelerator::table(&ctx, "w")
        .await?
        .expect("lower registers w");
    assert_eq!(w.version().number(), 1, "a new table is at version 1");

    let outputs = program.run(RunArgs::new()).await?;
    assert_eq!(outputs.snapshot().to_string(), "w@1", "the run read w@1");
    assert_eq!(outputs.published().to_string(), "w@2", "and published w@2");
    assert_eq!(w.version().number(), 2, "a publish adds one");

    w.replace(vec![weights(&[4.0])])?;
    assert_eq!(w.version().number(), 3, "a replace adds one");

    let err = program
        .run(RunArgs::new().param("d", ScalarValue::from("x")))
        .await;
    assert!(err.is_err(), "casting x to a number fails the step");
    assert_eq!(w.version().number(), 3, "a failed run publishes nothing");
    assert_eq!(
        by_index(&w.read()),
        vec![4.0],
        "and leaves the rows as they were"
    );
    Ok(())
}

#[tokio::test]
async fn a_failing_step_leaves_every_state_table_unchanged() -> Result<()> {
    let ctx = session();
    let program = Program::from_steps(
        &ctx,
        [
            ("w", "SELECT i, v + 1 AS v FROM w"),
            ("b", "SELECT i, v + 1 AS v FROM b"),
            // Casting the text "x" to a number fails, when the run comes.
            ("check", "SELECT CAST($text AS DOUBLE) AS c"),
        ],
    )
    .state("w", vec![weights(&[1.0])])
    .state("b", vec![weights(&[10.0])])
    .param("text", ScalarValue::from("0"))
    .lower()
    .await?
    .compile()?;
    program.run(RunArgs::new()).await?;
    let err = program
        .run(RunArgs::new().param("text", ScalarValue::from("x")))
        .await;
    assert!(err.is_err(), "the check step fails when told to");
    let w = Accelerator::table(&ctx, "w").await?.expect("w");
    let b = Accelerator::table(&ctx, "b").await?.expect("b");
    assert_eq!(
        (by_index(&w.read()), by_index(&b.read())),
        (vec![2.0], vec![11.0]),
        "after a failed run, both state tables hold the first run's values"
    );
    assert_eq!(
        (w.version().number(), b.version().number()),
        (2, 2),
        "and neither version moved"
    );
    Ok(())
}

#[tokio::test]
async fn a_replace_during_a_run_is_a_conflict_unless_overridden() -> Result<()> {
    let ctx = session();
    // A function whose call, inside the step, replaces w: a replace during
    // the run, at a known moment.
    let program =
        Program::new(&ctx, "SELECT i, interfere(v) AS v FROM w").state("w", vec![weights(&[1.0])]);
    let lowered = {
        let slot: Arc<std::sync::Mutex<Option<datafusion_accelerated::TableHandle>>> =
            Arc::default();
        let inner = Arc::clone(&slot);
        let interfere = create_udf(
            "interfere",
            vec![DataType::Float64],
            DataType::Float64,
            Volatility::Volatile,
            Arc::new(move |args: &[ColumnarValue]| {
                if let Some(w) = inner.lock().expect("not poisoned").as_ref() {
                    w.replace(vec![weights(&[100.0])])?;
                }
                Ok(args[0].clone())
            }),
        );
        ctx.register_udf(interfere);
        let lowered = program.lower().await?;
        *slot.lock().expect("not poisoned") = Accelerator::table(&ctx, "w").await?;
        lowered
    };
    let program = lowered.compile()?;
    let w = Accelerator::table(&ctx, "w").await?.expect("w");

    let err = program.run(RunArgs::new()).await.unwrap_err();
    assert!(
        matches!(
            Error::find(&err),
            Some(Error::Conflict { table, expected: 1, found: 2 }) if table == "w"
        ),
        "the run read w@1 and found w@2: {err}"
    );
    assert_eq!(
        by_index(&w.read()),
        vec![100.0],
        "the other caller's replace stands"
    );

    let outputs = program
        .run(RunArgs::new().on_conflict(OnConflict::Overwrite))
        .await?;
    assert_eq!(outputs.overwritten(), ["w"], "the override is recorded");
    assert_eq!(
        w.version().number(),
        4,
        "the step's replace made 3, and the publish 4"
    );
    Ok(())
}

#[tokio::test]
async fn names_clash_at_lower_with_both_named() -> Result<()> {
    let ctx = session();
    ctx.register_batch("taken", weights(&[0.0]))?;
    let cases: Vec<(Program, &str, NameKind, NameKind)> = vec![
        (
            Program::from_steps(&ctx, [("a", "SELECT 1 AS v"), ("a", "SELECT 2 AS v")]),
            "a",
            NameKind::Step,
            NameKind::Step,
        ),
        (
            Program::from_steps(&ctx, [("d", "SELECT 1 AS v")]).input("d", data_schema()),
            "d",
            NameKind::Input,
            NameKind::Step,
        ),
        (
            Program::from_steps(&ctx, [("lr", "SELECT 1 AS v")])
                .param("lr", ScalarValue::from(1.0)),
            "lr",
            NameKind::Parameter,
            NameKind::Step,
        ),
        (
            Program::from_steps(&ctx, [("taken", "SELECT 1 AS v")]),
            "taken",
            NameKind::Step,
            NameKind::SessionTable,
        ),
        (
            Program::from_steps(&ctx, [("a", "SELECT 1 AS v")]).input("taken", data_schema()),
            "taken",
            NameKind::Input,
            NameKind::SessionTable,
        ),
    ];
    for (program, name, first, second) in cases {
        let err = program.lower().await.unwrap_err();
        assert!(
            matches!(
                Error::find(&err),
                Some(Error::NameClash { name: n, first: f, second: s })
                    if n == name && *f == first && *s == second
            ),
            "{name}: a {first} and a {second}: {err}"
        );
    }
    let err = Program::new(&ctx, "SELECT i, v FROM taken")
        .state("taken", vec![weights(&[0.0])])
        .lower()
        .await
        .unwrap_err();
    assert!(
        matches!(Error::find(&err), Some(Error::NameTaken { .. })),
        "a state table can't take a session table's name: {err}"
    );
    assert!(
        !ctx.table_exist("pred")?,
        "a failed lower registers nothing"
    );
    Ok(())
}

#[tokio::test]
async fn a_cycle_and_an_unknown_name_fail_with_reasons() -> Result<()> {
    let ctx = session();
    let err = Program::from_steps(&ctx, [("a", "SELECT v FROM b"), ("b", "SELECT v FROM a")])
        .lower()
        .await
        .unwrap_err();
    assert!(
        matches!(Error::find(&err), Some(Error::Cycle { steps }) if steps.len() == 2),
        "a and b read each other: {err}"
    );
    let err = Program::from_steps(&ctx, [("a", "SELECT v FROM nowhere")])
        .lower()
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("nowhere"),
        "DataFusion's own error names the table: {err}"
    );
    let err = Program::from_steps(&ctx, [("a", "SELECT $nope AS v")])
        .lower()
        .await
        .unwrap_err();
    assert!(
        matches!(
            Error::find(&err),
            Some(Error::UnknownName {
                kind: NameKind::Parameter,
                ..
            })
        ),
        "an undeclared parameter: {err}"
    );
    Ok(())
}

#[tokio::test]
async fn a_run_needs_its_inputs() -> Result<()> {
    let ctx = session();
    let program = gradient_program(&ctx).lower().await?.compile()?;
    let err = program.run(RunArgs::new()).await.unwrap_err();
    assert!(
        matches!(Error::find(&err), Some(Error::MissingInput { name }) if name == "d"),
        "{err}"
    );
    let err = program
        .run(RunArgs::new().input("d", vec![data(0)]).input("e", vec![]))
        .await
        .unwrap_err();
    assert!(
        matches!(
            Error::find(&err),
            Some(Error::UnknownName {
                kind: NameKind::Input,
                ..
            })
        ),
        "{err}"
    );
    Ok(())
}

#[tokio::test]
async fn overlapping_runs_finish_in_order() -> Result<()> {
    let ctx = session();
    let program = Program::new(&ctx, "SELECT i, v + 1 AS v FROM w")
        .state("w", vec![weights(&[0.0])])
        .output("w")
        .lower()
        .await?
        .compile()?;
    let (first, second) = tokio::join!(program.run(RunArgs::new()), program.run(RunArgs::new()));
    let mut values = vec![
        by_index(&first?.table("w")?)[0],
        by_index(&second?.table("w")?)[0],
    ];
    values.sort_by(f64::total_cmp);
    assert_eq!(
        values,
        vec![1.0, 2.0],
        "the second run read the first's result"
    );
    Ok(())
}
