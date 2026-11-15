// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Bugs found by review and by `simulation.rs`, each reduced to a small case.
//!
//! Each test asserts what the crate's docs or the design promise. A test
//! here failed until its bug was fixed, and stays as a regression test.

use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::execution_plan::reset_plan_states;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_accelerated::{Accelerator, AcceleratorExt, AcceleratorOptions};

fn session() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(4))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    SessionContext::new_with_state(state)
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]))
}

fn column(values: &[i64]) -> ArrayRef {
    Arc::new(Int64Array::from(values.to_vec()))
}

fn batch(values: &[i64]) -> RecordBatch {
    RecordBatch::try_new(schema(), vec![column(values)]).expect("the column matches")
}

async fn plan(ctx: &SessionContext, sql: &str) -> Result<Arc<dyn ExecutionPlan>> {
    ctx.sql(sql).await?.create_physical_plan().await
}

async fn sum(plan: &Arc<dyn ExecutionPlan>, task: Arc<TaskContext>) -> Result<i64> {
    let batches = collect(reset_plan_states(Arc::clone(plan))?, task).await?;
    let column = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("an Int64 column");
    Ok(column.value(0))
}

/// A caller that keeps one `TaskContext` and runs a plan with it on every
/// training step read the first step's rows forever, when a run was
/// identified by its context's address. Nothing failed: the replace
/// succeeded and the answer was silently stale. Spike 4a's Python path
/// stores one task context and runs every compiled run with it. A run is
/// now explicit (`Accelerator::begin_run`), and a context outside one reads
/// the current rows.
#[tokio::test]
async fn a_reused_task_context_still_reads_each_replace() -> Result<()> {
    let ctx = session();
    let w = Accelerator::create_table(&ctx, "w", schema(), vec![batch(&[1])])?;
    let total = plan(&ctx, "SELECT SUM(v) FROM w").await?;
    let task = ctx.task_ctx();
    assert_eq!(sum(&total, Arc::clone(&task)).await?, 1);
    w.replace(vec![batch(&[2])])?;
    assert_eq!(
        sum(&total, Arc::clone(&task)).await?,
        2,
        "a run is one collect(); the second one should see the replace"
    );
    Ok(())
}

/// datafusion-ffi rebuilds the `TaskContext` on every `execute` call that
/// crosses the boundary (`From<FFI_TaskContext> for Arc<TaskContext>`,
/// datafusion-ffi 55.2.0 src/execution/task_ctx.rs:184, which ends in
/// `Arc::new(TaskContext::new(..))`), and drops it when `execute` returns.
/// This reproduces that: each partition of one logical run executes with
/// its own context, as a plan over a handle table does when datafusion-python
/// runs it. Each scan pinned its own snapshot, so D20 didn't hold. The run's
/// id now travels in the `accelerated.run` setting, which the FFI carries
/// as it carries every setting.
#[tokio::test]
async fn one_run_across_the_ffi_reads_one_snapshot() -> Result<()> {
    let ctx = session();
    let t = Accelerator::create_table(&ctx, "t", schema(), vec![batch(&[1])])?;
    let both = plan(&ctx, "SELECT v FROM t UNION ALL SELECT v FROM t").await?;
    let run = Accelerator::begin_run(&ctx)?;
    let execute = |partition: usize| {
        // What the foreign side of the FFI builds for each call: a new
        // context, from the session's settings as strings.
        let mut state = ctx.state();
        state
            .config_mut()
            .options_mut()
            .set("accelerated.run", &run.id().to_string())?;
        let task = Arc::new(TaskContext::from(&state));
        both.execute(partition, task)
    };
    let first = datafusion::physical_plan::common::collect(execute(0)?).await?;
    t.replace(vec![batch(&[2])])?;
    let second = datafusion::physical_plan::common::collect(execute(1)?).await?;
    assert_eq!(
        first, second,
        "both scans of t belong to one run, through the FFI as in Rust"
    );
    Ok(())
}

/// Session names are normalized (unquoted identifiers are lowercased), but
/// the runtime keyed its tables by the raw string. A table created as "W"
/// was `w` to SQL and to `table_exist`, and `W` to the runtime. The
/// session's catalog is now the only map from names to tables.
#[tokio::test]
async fn names_are_normalized_as_the_session_normalizes_them() -> Result<()> {
    let ctx = session();
    Accelerator::create_table(&ctx, "W", schema(), vec![batch(&[1])])?;
    assert!(
        ctx.table_exist("w")?,
        "SQL reads it as w, so the session has w"
    );
    assert!(
        Accelerator::table(&ctx, "w").await?.is_some(),
        "the runtime should find the table SQL reads as w"
    );
    ctx.deregister_table("w")?;
    assert!(
        Accelerator::table(&ctx, "W").await?.is_none(),
        "and lose it when the session does"
    );
    Ok(())
}

/// The runtime and the session each kept a map from names to tables, and
/// nothing kept them in step. `DROP TABLE` (or `deregister_table`)
/// removed the provider, the runtime kept the rows and the name, and the
/// name couldn't be created again although the session had no such table.
#[tokio::test]
async fn a_dropped_handle_table_frees_its_name() -> Result<()> {
    let ctx = session();
    Accelerator::create_table(&ctx, "t", schema(), vec![batch(&[1])])?;
    ctx.sql("DROP TABLE t").await?.collect().await?;
    assert!(!ctx.table_exist("t")?, "DROP TABLE removed it");
    Accelerator::create_table(&ctx, "t", schema(), vec![batch(&[2])])?;
    Ok(())
}

/// `release` said it "drops its rows". A finished run's pin was pruned only
/// by the next scan of any table, so the last run's snapshot, which holds
/// every table of the session, outlived the run and the release. A run's
/// pin is now dropped with the run, and a table's rows with its last
/// holder.
#[tokio::test]
async fn dropping_after_the_last_run_drops_the_rows() -> Result<()> {
    let ctx = session();
    let values = column(&[1, 2, 3]);
    let rows = RecordBatch::try_new(schema(), vec![Arc::clone(&values)])?;
    Accelerator::create_table(&ctx, "t", schema(), vec![rows])?;
    let total = plan(&ctx, "SELECT SUM(v) FROM t").await?;
    let run = Accelerator::begin_run(&ctx)?;
    sum(&total, run.task_ctx()).await?;
    drop(run);
    drop(total);
    ctx.sql("DROP TABLE t").await?.collect().await?;
    assert_eq!(
        Arc::strong_count(&values),
        1,
        "no run is live and the table is dropped, so nothing should hold its rows"
    );
    Ok(())
}
