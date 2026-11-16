// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Findings from the adversarial reviews of `Program` and of a user's first
//! day with M0, as tests that fail until each is fixed.
//!
//! Each test failed when its finding was posted, and passes since
//! `e7186a2` fixed it, so it stays as a regression test. A new finding goes
//! here ignored, with a link to it, until its fix lands. Each test's **Assumes** line states what it takes the
//! docs or the design to promise, so that a disagreement shows as a
//! different assumption, not a different reading of the result.

use std::sync::Arc;

use datafusion::arrow::array::{Array, Float64Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_accelerated::{
    Accelerator, AcceleratorExt, AcceleratorOptions, CompiledProgram, Error, Program, RunArgs,
};

const PROGRAM: &str =
    "https://github.com/xqlsystems/datafusion-accelerated/pull/40#issuecomment-6903589670";
const USER: &str =
    "https://github.com/xqlsystems/datafusion-accelerated/pull/40#issuecomment-6903743806";

fn session() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(2))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    SessionContext::new_with_state(state)
}

/// `(i, v)`, both Int64, `v` nullable.
fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, false),
        Field::new("v", DataType::Int64, true),
    ]))
}

fn rows(values: &[(i64, i64)]) -> Vec<RecordBatch> {
    let i: Int64Array = values.iter().map(|r| r.0).collect();
    let v: Int64Array = values.iter().map(|r| Some(r.1)).collect();
    vec![RecordBatch::try_new(schema(), vec![Arc::new(i), Arc::new(v)]).expect("the columns match")]
}

fn mem(values: &[(i64, i64)]) -> Arc<MemTable> {
    Arc::new(MemTable::try_new(schema(), vec![rows(values)]).expect("a table"))
}

/// The `v` column of a result, sorted.
fn values(batches: &[RecordBatch]) -> Vec<i64> {
    let mut out: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            let v = b
                .column_by_name("v")
                .expect("a v column")
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64")
                .clone();
            (0..v.len()).map(move |k| v.value(k))
        })
        .collect();
    out.sort_unstable();
    out
}

async fn one_step(
    ctx: &SessionContext,
    sql: &str,
    params: &[(&str, ScalarValue)],
) -> Result<CompiledProgram> {
    let mut program = Program::from_steps(ctx, [("out", sql)])
        .state("w", rows(&[(0, 1), (1, 2), (2, 3)]))
        .output("out");
    for (name, value) in params {
        program = program.param(*name, value.clone());
    }
    program.lower().await?.compile()
}

// ---------------------------------------------------------------------------
// The review of `Program`.
// ---------------------------------------------------------------------------

/// Finding 1. **Assumes:** R1, that a run never reads rows the session no
/// longer has under that name; failing is fine, reading old rows isn't.
#[tokio::test]
async fn a_reregistered_session_table_is_never_read_stale() -> Result<()> {
    let _ = PROGRAM;
    let ctx = session();
    ctx.register_table("lookup", mem(&[(0, 1)]))?;
    let program = one_step(&ctx, "SELECT i, v FROM lookup", &[]).await?;
    ctx.deregister_table("lookup")?;
    ctx.register_table("lookup", mem(&[(0, 2)]))?;
    if let Ok(outputs) = program.run(RunArgs::new()).await {
        assert_eq!(
            values(&outputs.table("out")?),
            vec![2],
            "the run read the old lookup"
        );
    }
    Ok(())
}

/// Finding 1. **Assumes:** R1; after `DROP TABLE`, the session has no rows
/// by that name, so a run that reads it fails.
#[tokio::test]
async fn a_dropped_session_table_fails_the_run() -> Result<()> {
    let ctx = session();
    ctx.register_table("lookup", mem(&[(0, 1)]))?;
    let program = one_step(&ctx, "SELECT i, v FROM lookup", &[]).await?;
    ctx.sql("DROP TABLE lookup").await?.collect().await?;
    assert!(
        program.run(RunArgs::new()).await.is_err(),
        "the run read a dropped table"
    );
    Ok(())
}

/// Finding 2. **Assumes:** design §5.4, that a snapshot is a map from tables
/// to versions, so a name appears once, and never names a dropped table.
#[tokio::test]
async fn a_snapshot_names_each_table_once_and_the_live_one() -> Result<()> {
    let ctx = session();
    let old = Accelerator::create_table(&ctx, "t", schema(), rows(&[(0, 0)]))?;
    let program = one_step(&ctx, "SELECT i, v FROM t", &[]).await?;
    let plan = ctx
        .sql("SELECT * FROM t")
        .await?
        .create_physical_plan()
        .await?;
    ctx.sql("DROP TABLE t").await?.collect().await?;
    drop(old);
    let new = Accelerator::create_table(&ctx, "t", schema(), rows(&[(0, 9)]))?;
    // Failing as stale is fine: the program read the dropped t.
    let outputs = match program.run(RunArgs::new()).await {
        Ok(outputs) => outputs,
        Err(e) if matches!(Error::find(&e), Some(Error::Stale { .. })) => return Ok(()),
        Err(e) => return Err(e),
    };
    let names: Vec<&str> = outputs.snapshot().tables().map(|(name, _)| name).collect();
    let mut unique = names.clone();
    unique.dedup();
    assert_eq!(
        names,
        unique,
        "a name appears twice: {}",
        outputs.snapshot()
    );
    if let Some(version) = outputs.snapshot().version("t") {
        assert_eq!(
            version.table(),
            new.version().table(),
            "t names the dropped table"
        );
    }
    drop(plan);
    Ok(())
}

/// Design point C. **Assumes:** design §5.4, that a snapshot names the
/// state a run read, so it's a key for checkpoints and caches; a table the
/// program doesn't read isn't part of that state.
#[tokio::test]
async fn a_snapshot_lists_only_what_the_program_reads() -> Result<()> {
    let ctx = session();
    let unrelated = Accelerator::create_table(&ctx, "unrelated", schema(), rows(&[(0, 0)]))?;
    let program = one_step(&ctx, "SELECT i, v FROM w", &[]).await?;
    let first = program.run(RunArgs::new()).await?;
    unrelated.replace(rows(&[(0, 1)]))?;
    let second = program.run(RunArgs::new()).await?;
    assert_eq!(
        first.snapshot().version("unrelated"),
        None,
        "the snapshot lists a table no step reads: {}",
        first.snapshot()
    );
    assert_eq!(
        first.snapshot(),
        second.snapshot(),
        "nothing the program reads changed"
    );
    Ok(())
}

/// Finding 3. **Assumes:** design §5.4, that a parameter is written where a
/// literal is; where it can't go, the runtime says so in its own error,
/// naming the step and the position, not as DataFusion's internal error.
#[tokio::test]
async fn a_parameter_in_values_works_or_fails_with_the_runtimes_error() -> Result<()> {
    let ctx = session();
    let sql = "SELECT column1 AS i, column2 AS v FROM (VALUES (0, $k))";
    match one_step(&ctx, sql, &[("k", ScalarValue::from(1_i64))]).await {
        Err(e) => {
            let internal = matches!(e.find_root(), DataFusionError::Internal(_));
            let text = e.to_string();
            assert!(
                !internal && text.contains("out") && text.contains("VALUES"),
                "{text}"
            );
            Ok(())
        }
        Ok(program) => {
            let outputs = program.run(RunArgs::new()).await?;
            assert_eq!(values(&outputs.table("out")?), vec![1]);
            Ok(())
        }
    }
}

/// Finding 3. **Assumes:** that an error at `lower` names the step and the
/// position, so a user can find what to change.
#[tokio::test]
async fn a_parameter_where_none_may_go_names_the_step_and_position() {
    for (sql, position) in [
        ("SELECT i, v FROM w ORDER BY i LIMIT $k", "LIMIT"),
        ("SELECT i, v FROM w ORDER BY i OFFSET $k", "OFFSET"),
        (
            "SELECT value AS i, value AS v FROM generate_series(0, $k)",
            "table function",
        ),
    ] {
        let ctx = session();
        let err = one_step(&ctx, sql, &[("k", ScalarValue::from(1_i64))])
            .await
            .err()
            .map(|e| e.to_string());
        if let Some(err) = err {
            assert!(
                err.contains("out") && err.contains(position),
                "for {sql}: {err}"
            );
        }
    }
}

/// Finding 4. **Assumes:** R10, that the runtime makes no decision no one
/// made; a value given twice is ambiguous.
#[tokio::test]
async fn a_run_argument_given_twice_is_an_error() -> Result<()> {
    let ctx = session();
    let program = Program::from_steps(&ctx, [("out", "SELECT i, v + $k AS v FROM d")])
        .state("w", rows(&[(0, 0)]))
        .input("d", schema())
        .param("k", ScalarValue::from(0_i64))
        .output("out")
        .lower()
        .await?
        .compile()?;
    let input_twice = RunArgs::new()
        .input("d", rows(&[(0, 1)]))
        .input("d", rows(&[(0, 2)]));
    assert!(
        program.run(input_twice).await.is_err(),
        "an input given twice"
    );
    let param_twice = RunArgs::new()
        .input("d", rows(&[(0, 1)]))
        .param("k", ScalarValue::from(1_i64))
        .param("k", ScalarValue::from(2_i64));
    assert!(
        program.run(param_twice).await.is_err(),
        "a parameter given twice"
    );
    Ok(())
}

/// Finding 4. **Assumes:** R1 and R10, that a parameter's value is its
/// declared type, or converts to it without loss.
#[tokio::test]
async fn a_parameter_value_that_loses_information_is_an_error() -> Result<()> {
    let ctx = session();
    let program = one_step(
        &ctx,
        "SELECT i, v + $k AS v FROM w",
        &[("k", ScalarValue::from(0_i64))],
    )
    .await?;
    let lossy = program
        .run(RunArgs::new().param("k", ScalarValue::from(2.9_f64)))
        .await;
    assert!(lossy.is_err(), "2.9 for an Int64 parameter");
    let text = program
        .run(RunArgs::new().param("k", ScalarValue::from("1")))
        .await;
    assert!(text.is_err(), "text for an Int64 parameter");
    Ok(())
}

/// Design point B. **Assumes:** that an input's size isn't known at `lower`,
/// so the plan mustn't fix it as the side every run collects whole.
#[tokio::test]
async fn an_input_isnt_planned_as_if_empty() -> Result<()> {
    let ctx = session();
    let big: Vec<(i64, i64)> = (0..10_000).map(|i| (i, i)).collect();
    let lowered = Program::from_steps(
        &ctx,
        [(
            "out",
            "SELECT d.i, d.v + w.v AS v FROM d JOIN w ON d.i = w.i",
        )],
    )
    .state("w", rows(&big))
    .input("d", schema())
    .output("out")
    .lower()
    .await?;
    let plan = lowered.explain();
    // The build side is the join's first input: the first scan under it.
    let collect_left_on_d = plan
        .lines()
        .skip_while(|l| !l.contains("mode=CollectLeft"))
        .find(|l| l.contains("HandleScanExec"))
        .is_some_and(|build| build.contains("table=d"));
    assert!(!collect_left_on_d, "{plan}");
    Ok(())
}

/// Design point E. **Assumes:** design §5.4, that order comes from reads,
/// not from the list, so a qualified read of a step is a read.
#[tokio::test]
async fn a_qualified_read_of_a_step_lowers_in_either_order() -> Result<()> {
    for steps in [
        [
            ("s0", "SELECT i, v FROM w"),
            ("s1", "SELECT i, v FROM public.s0"),
        ],
        [
            ("s1", "SELECT i, v FROM public.s0"),
            ("s0", "SELECT i, v FROM w"),
        ],
    ] {
        let ctx = session();
        Program::from_steps(&ctx, steps)
            .state("w", rows(&[(0, 0)]))
            .output("s1")
            .lower()
            .await?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// A user's first day.
// ---------------------------------------------------------------------------

/// Item 2. **Assumes:** that a state written the plain SQL way, whose
/// literal columns are `NOT NULL`, can be updated with a parameter, as it
/// can with a literal.
#[tokio::test]
async fn a_not_null_state_can_be_updated_with_a_parameter() -> Result<()> {
    let _ = USER;
    let ctx = session();
    let w0 = ctx.sql("SELECT 0 AS k, 0.0 AS v").await?.collect().await?;
    let program = Program::from_steps(&ctx, [("w", "SELECT k, v - $lr AS v FROM w")])
        .state("w", w0)
        .param("lr", ScalarValue::from(0.5))
        .lower()
        .await?
        .compile()?;
    program.run(RunArgs::new()).await?;
    let w = Accelerator::table(&ctx, "w").await?.expect("w");
    let v = w.read()[0]
        .column_by_name("v")
        .expect("v")
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64")
        .value(0);
    assert_eq!(v, -0.5);
    Ok(())
}

/// Item 4. **Assumes:** that an error planning a step names the step, since
/// a program has many and DataFusion's message names none.
#[tokio::test]
async fn a_planning_error_names_its_step() {
    for (step, sql) in [
        ("grad_w", "SELECT nope FROM w"),
        ("grad_w", "SELECT i FROM missing_table"),
        ("grad_w", "SELECT v FROM w JOIN w AS u ON w.i = u.i"),
    ] {
        let ctx = session();
        let err = Program::from_steps(&ctx, [(step, sql)])
            .state("w", rows(&[(0, 0)]))
            .lower()
            .await
            .expect_err("the step can't be planned")
            .to_string();
        assert!(err.contains(step), "for {sql}: {err}");
    }
}

/// Item 6. **Assumes:** that an error for SQL a runtime table can't take
/// says the table is the runtime's, since DataFusion's says "Base table".
#[tokio::test]
async fn dml_on_a_runtime_table_says_whose_it_is() -> Result<()> {
    let ctx = session();
    Accelerator::create_table(&ctx, "lookup", schema(), rows(&[(0, 0)]))?;
    for sql in ["INSERT INTO lookup VALUES (1, 1)", "TRUNCATE TABLE lookup"] {
        let err = match ctx.sql(sql).await {
            Ok(df) => df.collect().await.err(),
            Err(e) => Some(e),
        };
        if let Some(err) = err.map(|e: DataFusionError| e.to_string()) {
            assert!(err.contains("runtime"), "for {sql}: {err}");
        }
    }
    Ok(())
}

/// Item 6. **Assumes:** that a handle whose table the session no longer
/// holds says so, instead of taking rows nobody reads.
#[tokio::test]
async fn a_detached_handle_refuses_a_replace() -> Result<()> {
    let ctx = session();
    let lookup = Accelerator::create_table(&ctx, "lookup", schema(), rows(&[(0, 0)]))?;
    ctx.sql("CREATE OR REPLACE TABLE lookup AS SELECT 1 AS i, 1 AS v")
        .await?
        .collect()
        .await?;
    assert!(
        lookup.replace(rows(&[(0, 2)])).is_err(),
        "SQL can't read these rows"
    );
    Ok(())
}
